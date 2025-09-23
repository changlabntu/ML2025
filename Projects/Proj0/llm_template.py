# ----------------------------
# Import necessary modules
# ----------------------------

# ----------------------------
# Import necessary modules
# ----------------------------

# ----------------------------
# Config
# ----------------------------
@dataclass
class SearchConfig:
    # a dataclass that configure search parameters,
    # including if use journal to filter, what are the top journals, 
    # query for pubmed,
    # target regions,
    # max number of results
    # model option
    ###


@dataclass
class ExportConfig:
    # a datacclass that control export parameters,
    # including filename


@dataclass
class AnalysisConfig:
    # a dataclass that control llm analysis parameters,
    # including batch size for llm query,
    # default queries
    

@dataclass
class AppConfig:
    # a dataclass to combine SearchConfig, ExportConfig, and AnalysisConfig
    # a load method to load config from json file, given path to json file
    def load:
    

# ----------------------------
# Domain model
# ----------------------------
@dataclass
class PaperDoc:
    # a dataclass that represent a paper,
    # including title, pmid, abstract, corresponding author, affiliation


# ----------------------------
# Services
# ----------------------------
class PubMedClient:
    # a class that handle pubmed search and fetch

    def __init__():
        # initialize with email and tool

    def search_ids(query):
        # search_ids method to search pubmed and return ids
        return ids

    
    def fetch_abstracts(ids):
        # fetch_abstracts method to fetch abstracts from pubmed
        # each paper in docs should have these information: title, pmid, abstract, corresponding author, affiliation
        return docs
        
class LLMService:
    # a class that handle llm analysis
    
    def __init__():
        # initialize with api key, model name,
        # we use llamaindex here, llama_index.llms.openai import OpenAI
        # if model option = "gpt-4o-mini", set llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
        # if model option = "deepseek", use huggingface_hub InferenceClient to call deepseek-ai/DeepSeek-V3-0324, api_key=HF_TOKEN)

    def region_filter(docs, regions):
        # region_filter method to filter papers by region, llm process one batch at a time
        # using the following prompt
        prompt = f"""
        Analyze these institution names and determine which ones are in ANY of: {regions_str}.

        Institutions:
        {inst_text}

        Respond with only the numbers (1,2,...) of institutions that are clearly in those regions, comma-separated.
        If none, respond "NONE".
        """
        return filtered_docs

    def analyze_papers_with_research_questions(papers, queries):
        # use llm to analyze papers with research questions using the following prompt
        prompt = f"""
        Analyze these research papers and answer the question for each paper with ONLY THREE WORDS per paper.

        Papers:
        {papers}

        Question:
        {queries}

        """
        return answers

    def add_answers_to_papers(papers, answers):
        # add answers to papers
        return papers
 

class CSVExporter:
    # a class to export papers to csv to given filename
    # csv should have these columns: title, pmid, abstract, corresponding author, affiliation, answer to research question

# ----------------------------
# Pipeline
# ----------------------------
# a class to run the total analysis pipeline
class Pipeline:
    def __init__():
        # initialize with config, pubmed client, llm service, csv exporter


    def run(self) -> None:
        # run the pipeline
        # 1. search pubmed
        # 2. fetch abstracts
        # 3. region filter
        # 4. llm analysis
        # 5. export



# ----------------------------
# Main
# ----------------------------
def main():
    # main function to run the pipeline
    # load .env file for the api_key
    # load config
    # initialize services
    # run pipeline
    # export results