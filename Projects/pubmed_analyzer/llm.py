"""
pubmed_llamaindex_pipeline.py
Single-file, modular architecture:
- Config dataclasses
- Service layer (PubMedClient, LLMService, CSVExporter)
- Pure transforms
- Pipeline orchestrator
"""

from __future__ import annotations
import os
import re
import csv
import json
import argparse
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any

from dotenv import load_dotenv
from Bio import Entrez

# Optional imports guarded inside services to allow running without OpenAI key.
# from llama_index.core import VectorStoreIndex, Document, Settings
# from llama_index.llms.openai import OpenAI
# from llama_index.embeddings.openai import OpenAIEmbedding

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("pubmed-pipeline")

# ----------------------------
# Config
# ----------------------------
@dataclass
class SearchConfig:
    filter_journals: bool = False
    query_index: int = 0
    top_journals: List[str] = field(default_factory=lambda: [
        '"J Am Med Inform Assoc"[Journal]',
        '"Nat Med"[Journal]',
        '"Lancet Digit Health"[Journal]'
    ])
    queries: List[str] = field(default_factory=lambda: ["machine learning AND mental health"])
    target_regions: List[str] = field(default_factory=lambda: ["none"])
    max_results: int = 100
    model_option: str = "gpt-4o-mini"


@dataclass
class ExportConfig:
    filename: str = "papers.csv"
    auto_export: bool = False


@dataclass
class AnalysisConfig:
    batch_query: int = 5
    default_queries: List[str] = field(default_factory=lambda: [
        "What are the main topics covered in these papers?",
        "What methodologies are commonly used?"
    ])


@dataclass
class AppConfig:
    search: SearchConfig = field(default_factory=SearchConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    @staticmethod
    def load(path: str) -> "AppConfig":
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return AppConfig(
                search=SearchConfig(**data.get("search_config", {})),
                export=ExportConfig(**data.get("export_config", {})),
                analysis=AnalysisConfig(**data.get("analysis_config", {})),
            )
        except FileNotFoundError:
            log.warning("Config file %s not found. Using defaults.", path)
            return AppConfig()
        except json.JSONDecodeError as e:
            log.warning("Error parsing config %s: %s. Using defaults.", path, e)
            return AppConfig()

# ----------------------------
# Domain model
# ----------------------------
@dataclass
class PaperDoc:
    title: str
    pmid: str
    abstract: str
    corresponding_author: Optional[str] = None
    affiliation: Optional[str] = None
    # You can add journal, year, etc., if you fetch them.

# ----------------------------
# Services
# ----------------------------
class PubMedClient:
    """Thin wrapper around Entrez for search+fetch."""
    def __init__(self, email: str, tool: str = "pubmed_llamaindex_demo"):
        self.email = email
        self.tool = tool
        Entrez.email = email
        Entrez.tool = tool

    def search_ids(self, query: str, max_results: int = 100) -> List[str]:
        log.info("Searching PubMed: %s", query)
        with Entrez.esearch(db="pubmed", term=query, retmax=max_results) as handle:
            res = Entrez.read(handle)
        ids = res.get("IdList", [])
        log.info("Found %d ids", len(ids))
        return ids

    def fetch_abstracts(self, ids: List[str]) -> List[PaperDoc]:
        if not ids:
            return []
        # Use efetch to get XML records for better author parsing
        with Entrez.efetch(db="pubmed", id=",".join(ids), rettype="xml", retmode="xml") as handle:
            papers_xml = Entrez.read(handle)

        docs: List[PaperDoc] = []
        for article in papers_xml['PubmedArticle']:
            try:
                medline_citation = article['MedlineCitation']
                pmid = str(medline_citation['PMID'])
                title = medline_citation['Article']['ArticleTitle']
                
                # Extract abstract
                abstract = ""
                if 'Abstract' in medline_citation['Article']:
                    abstract_texts = medline_citation['Article']['Abstract']['AbstractText']
                    if isinstance(abstract_texts, list):
                        abstract = " ".join([str(text) for text in abstract_texts])
                    else:
                        abstract = str(abstract_texts)
                
                # Extract corresponding author and affiliation
                corresponding_author = None
                affiliation = None
                
                if 'AuthorList' in medline_citation['Article']:
                    authors = medline_citation['Article']['AuthorList']
                    
                    for author in authors:
                        if 'LastName' in author and 'ForeName' in author:
                            author_name = f"{author['ForeName']} {author['LastName']}"
                            
                            if 'AffiliationInfo' in author:
                                for affiliation_info in author['AffiliationInfo']:
                                    if 'Affiliation' in affiliation_info:
                                        affil_text = affiliation_info['Affiliation']
                                        # Use the first author with affiliation as corresponding author
                                        if not corresponding_author:
                                            corresponding_author = author_name
                                            affiliation = affil_text
                                        break
                                if corresponding_author:
                                    break
                
                docs.append(PaperDoc(
                    title=title,
                    pmid=pmid,
                    abstract=abstract or "N/A",
                    corresponding_author=corresponding_author,
                    affiliation=affiliation
                ))
                
            except Exception as e:
                log.warning("Error parsing paper: %s", e)
                continue
                
        log.info("Parsed %d records", len(docs))
        return docs


class LLMService:
    """Encapsulates LlamaIndex/OpenAI usage. Safe to disable if no key."""
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.enabled = bool(self.api_key)
        self._llm = None

    def _ensure_llm(self):
        if not self.enabled:
            return
        if self._llm is None:
            # Lazy import to keep startup light without OpenAI
            from llama_index.core import Settings
            from llama_index.llms.openai import OpenAI
            from llama_index.embeddings.openai import OpenAIEmbedding
            Settings.llm = OpenAI(model="gpt-4o-mini", api_key=self.api_key)
            Settings.embed_model = OpenAIEmbedding(api_key=self.api_key)
            self._llm = Settings.llm

    def region_filter(self, docs: List[PaperDoc], target_regions: List[str], batch_size: int = 5) -> List[PaperDoc]:
        if not target_regions or (len(target_regions) == 1 and target_regions[0].lower() == "none"):
            return docs
        if not self.enabled:
            log.warning("No OPENAI_API_KEY. Skipping region filtering.")
            return docs

        self._ensure_llm()
        regions_str = ", ".join(target_regions)
        log.info("Filtering by regions via LLM: %s", regions_str)

        # Build batches of (idx, affiliation text)
        batch_items: List[Tuple[int, str]] = []
        for i, d in enumerate(docs):
            if d.affiliation:
                batch_items.append((i, d.affiliation))

        keep_mask = [False] * len(docs)
        for start in range(0, len(batch_items), batch_size):
            batch = batch_items[start:start + batch_size]
            inst_text = "\n".join(f"{j+1}. {aff}" for j, (_, aff) in enumerate(batch))
            prompt = f"""
Analyze these institution names and determine which ones are in ANY of: {regions_str}.

Institutions:
{inst_text}

Respond with only the numbers (1,2,...) of institutions that are clearly in those regions, comma-separated.
If none, respond "NONE".
"""
            try:
                result = self._llm.complete(prompt).text.strip()
            except Exception as e:
                log.warning("LLM error on region batch: %s", e)
                # On failure, keep all in batch
                for j, (idx, _) in enumerate(batch):
                    keep_mask[idx] = True
                continue

            chosen: List[int] = []
            if result.upper() != "NONE":
                for tok in result.split(","):
                    tok = tok.strip()
                    if tok.isdigit():
                        j = int(tok) - 1
                        if 0 <= j < len(batch):
                            chosen.append(j)

            for j, (idx, _) in enumerate(batch):
                keep_mask[idx] = (j in chosen)

        kept = [d for i, d in enumerate(docs) if keep_mask[i]]
        log.info("Region filter kept %d/%d", len(kept), len(docs))
        return kept if kept else docs  # fall back to original if over-filtered

    def analyze_papers_three_words(
        self,
        papers: List[Tuple[str, str]],
        queries: List[str],
        batch_size: int = 10
    ) -> List[List[str]]:
        """Returns per-paper list of answers (one three-word answer per query)."""
        if not queries:
            return [["N/A"] * 0 for _ in papers]
        if not self.enabled:
            log.warning("No OPENAI_API_KEY. Skipping LLM analysis.")
            return [["N/A"] * len(queries) for _ in papers]

        self._ensure_llm()
        all_results: List[List[str]] = []
        for start in range(0, len(papers), batch_size):
            batch = papers[start:start + batch_size]
            batch_answers_per_query: List[List[str]] = []

            for q in queries:
                # Build prompt
                parts = []
                for i, (title, abstract) in enumerate(batch):
                    abs_txt = abstract if abstract and abstract != "N/A" else "No abstract available"
                    parts.append(f"Paper {i+1}: {title}\nAbstract: {abs_txt}\n")
                prompt = (
                    "Analyze these research papers and answer the question for each paper with ONLY THREE WORDS per paper.\n\n"
                    + "\n".join(parts) +
                    f"\nQuestion: {q}\n\n"
                    "Instructions:\n"
                    "- Respond with exactly THREE words per paper\n"
                    '- Format: "1: word1 word2 word3, 2: word4 word5 word6, ..."\n'
                    '- If unclear, respond with "not sure"\n\n'
                    "Three word answers:"
                )
                try:
                    txt = self._llm.complete(prompt).text.strip()
                except Exception as e:
                    log.warning("LLM error during analysis: %s", e)
                    batch_answers_per_query.append(["Error"] * len(batch))
                    continue

                answers = self._parse_three_word_batch_reply(txt, len(batch))
                batch_answers_per_query.append(answers)

            # Transpose per-query answers to per-paper list
            for i in range(len(batch)):
                per_paper = [batch_answers_per_query[q_i][i] for q_i in range(len(queries))]
                all_results.append(per_paper)

        return all_results

    @staticmethod
    def _parse_three_word_batch_reply(reply: str, batch_len: int) -> List[str]:
        """Parse '1: a b c, 2: d e f, ...' reliably."""
        answers = ["not sure"] * batch_len
        # Accept flexible separators; pick first three tokens after the label.
        for i in range(1, batch_len + 1):
            m = re.search(rf"\b{i}:\s*([^\n,]+)", reply)
            if not m:
                continue
            words = m.group(1).strip().split()
            answers[i - 1] = " ".join(words[:3]) if words else "not sure"
        return answers


class CSVExporter:
    def save(
        self,
        filename: str,
        papers: List[PaperDoc],
        analysis_answers: Optional[List[List[str]]] = None,
        queries: Optional[List[str]] = None
    ) -> str:
        fieldnames = ["name", "PMID", "corresponding_author", "affiliation", "abstract"]
        queries = queries or []
        for i in range(len(queries)):
            fieldnames.append(f"Q{i+1}")

        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, p in enumerate(papers):
                row = {
                    "name": p.title,
                    "PMID": p.pmid,
                    "corresponding_author": p.corresponding_author or "N/A",
                    "affiliation": p.affiliation or "N/A",
                    "abstract": p.abstract or "N/A",
                }
                if analysis_answers and i < len(analysis_answers):
                    for j, ans in enumerate(analysis_answers[i]):
                        row[f"Q{j+1}"] = ans
                writer.writerow(row)
        log.info("Exported CSV: %s", filename)
        return filename

# ----------------------------
# Transforms (pure-ish)
# ----------------------------
def to_llm_pairs(docs: List[PaperDoc]) -> List[Tuple[str, str]]:
    return [(d.title or "N/A", d.abstract or "N/A") for d in docs]

# ----------------------------
# Pipeline
# ----------------------------
class Pipeline:
    def __init__(self, cfg: AppConfig, pubmed: PubMedClient, llm: LLMService, exporter: CSVExporter):
        self.cfg = cfg
        self.pubmed = pubmed
        self.llm = llm
        self.exporter = exporter

    def run(self) -> None:
        s = self.cfg.search
        e = self.cfg.export
        a = self.cfg.analysis

        base_query = self._pick_query(s.queries, s.query_index)
        full_query = self._maybe_add_journal_filter(base_query, s)

        ids = self.pubmed.search_ids(full_query, max_results=s.max_results)
        docs = self.pubmed.fetch_abstracts(ids)

        if not docs:
            log.warning("No papers with abstracts found. Try a broader search.")
            return

        # Optional LLM-based region filter
        if s.target_regions and not (len(s.target_regions) == 1 and s.target_regions[0].lower() == "none"):
            docs = self.llm.region_filter(docs, s.target_regions, batch_size=a.batch_query)
            if not docs:
                log.warning("No papers found after region filter: %s", ", ".join(s.target_regions))
                return

        # Optional LLM analysis
        analysis_answers: List[List[str]] = []
        if a.default_queries:
            analysis_answers = self.llm.analyze_papers_three_words(
                to_llm_pairs(docs),
                a.default_queries,
                batch_size=a.batch_query
            )

        # Optional export
        if e.auto_export:
            self.exporter.save(e.filename, docs, analysis_answers, a.default_queries)

    @staticmethod
    def _pick_query(queries: List[str], idx: int) -> str:
        if not queries:
            raise ValueError("No queries configured.")
        if idx < 0 or idx >= len(queries):
            log.warning("Query index %d out of range. Using index 0.", idx)
            idx = 0
        q = queries[idx]
        log.info("Using query: %s", q)
        return q

    @staticmethod
    def _maybe_add_journal_filter(base_query: str, cfg: SearchConfig) -> str:
        if not cfg.filter_journals or not cfg.top_journals:
            return base_query
        jf = " OR ".join(cfg.top_journals)
        q = f"({base_query}) AND ({jf})"
        log.info("Journal-filtered query: %s", q)
        return q

# ----------------------------
# Defaults / bootstrap
# ----------------------------
DEFAULT_EMAIL = os.getenv("NCBI_EMAIL", "your.email@example.com")

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="PubMed + LlamaIndex (single-file modular)")
    parser.add_argument("--config", default="config.json", help="Path to JSON configuration")
    args = parser.parse_args()

    cfg = AppConfig.load(args.config)
    log.info("Config: %s", json.dumps({
        "search_config": asdict(cfg.search),
        "export_config": asdict(cfg.export),
        "analysis_config": asdict(cfg.analysis)
    }, ensure_ascii=False))

    pubmed = PubMedClient(email=DEFAULT_EMAIL)
    llm = LLMService()
    exporter = CSVExporter()

    Pipeline(cfg, pubmed, llm, exporter).run()


if __name__ == "__main__":
    main()
