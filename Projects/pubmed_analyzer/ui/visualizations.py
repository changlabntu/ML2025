import streamlit as st
import plotly.express as px

def create_university_chart(df):
    """Create university distribution chart"""
    if 'university' not in df.columns:
        return
    
    # Get top 15 universities
    uni_counts = df['university'].value_counts().head(15)
    
    if len(uni_counts) > 0:
        fig = px.bar(
            x=uni_counts.values,
            y=uni_counts.index,
            orientation='h',
            title="🏫 Top Universities by Paper Count",
            labels={'x': 'Papers', 'y': 'University'}
        )
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, width='stretch')

def create_department_chart(df):
    """Create department distribution chart"""
    if 'department' not in df.columns:
        return
    
    # Get top 10 departments
    dept_counts = df['department'].value_counts().head(10)
    
    if len(dept_counts) > 0:
        fig = px.pie(
            values=dept_counts.values,
            names=dept_counts.index,
            title="🏛️ Research Departments"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
