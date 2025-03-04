import os
import streamlit as st

# Langchain and AI libraries
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langsmith import traceable
from database import supabase

# Import your functions
from utils import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt, clean_text
from prompt import PROJECT_SUMMARY_PROMPT, CONSULTANT_MATCH_PROMPT

DEFAULT_VALUE = "N/A"

# Initialize embeddings
@st.cache_resource
@traceable(
    metadata={"embedding_model": "google/models/text-embedding-004"},
)
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Load consultant data
def process_uploaded_file(uploaded_file):
    """Process the uploaded file based on its type."""
    file_text = ""
    if uploaded_file.type == "application/pdf":
        file_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        file_text = extract_text_from_docx(uploaded_file)
    elif uploaded_file.type == "text/plain":
        file_text = extract_text_from_txt(uploaded_file)
    else:
        st.error("❌ Unsupported file type")
    return file_text

# Project summary function using AI
@traceable()
def generate_project_summary(text, prompt=PROJECT_SUMMARY_PROMPT):
    """Generate structured project summary using AI"""
    text = clean_text(text)
    max_length = 10000
    if len(text) > max_length:
        text = text[:max_length]
    llm = ChatGoogleGenerativeAI (
        model="gemini-2.0-flash", 
        temperature=0,
        api_key=os.environ["GOOGLE_API_KEY"] 
    )
    try:
        response = llm.invoke(prompt.format(text=text))
        return response.content
    except Exception as e:
        st.error(f"❌ Error generating project summary: {e}")
        return "Unable to generate summary."

# Analyze consultant match with AI
@traceable()
def analyze_consultant_match(project_summary, consultant_details, prompt=CONSULTANT_MATCH_PROMPT):
    """Generate detailed analysis of consultant match"""
    llm = ChatGoogleGenerativeAI (
        model="gemini-2.0-flash", 
        temperature=0,
        api_key=os.environ["GOOGLE_API_KEY"] 
    )
    try:
        response = llm.invoke(prompt.format(project_summary=project_summary, consultant_details=consultant_details))
        return response.content
    except Exception as e:
        st.error(f"❌ Error analyzing consultant match: {e}")
        return "Unable to generate detailed match analysis."

# Find best consultant matches
@traceable(
        run_type="retriever"
)
def find_best_consultant_matches(embeddings, project_summary, top_k=3):
    """Find the best consultant matches based on project summary using Supabase."""
    llm = ChatGoogleGenerativeAI (
        model="gemini-2.0-flash", 
        temperature=0,
        api_key=os.environ["GOOGLE_API_KEY"] 
    )

    try:
        query_embedding = embeddings.embed_query(project_summary)
        result = supabase.rpc("search_consultants_vector", {
            "query_embedding": query_embedding,
            "limit_num": top_k
        }).execute()
        matches = []
        for row in result.data:
            consultant_details = "\n".join([
                f"{key}: {value}" for key, value in row.items() 
                if key not in ["embedding", "distance", "id"]
            ])
            match_analysis = analyze_consultant_match(project_summary, consultant_details)
            matches.append({
                "Name": row.get("Name", DEFAULT_VALUE),
                "Age": row.get("Age", DEFAULT_VALUE),
                "Finance Expertise": row.get("Finance Expertise", DEFAULT_VALUE),
                "Strategy Expertise": row.get("Strategy Expertise", DEFAULT_VALUE),
                "Operations Expertise": row.get("Operations Expertise", DEFAULT_VALUE),
                "Marketing Expertise": row.get("Marketing Expertise", DEFAULT_VALUE),
                "Entrepreneurship Expertise": row.get("Entrepreneurship Expertise", DEFAULT_VALUE),
                "Education": row.get("Education", DEFAULT_VALUE),
                "Industry Expertise": row.get("Industry Expertise", DEFAULT_VALUE),
                "Bio": row.get("Bio", DEFAULT_VALUE),
                "Anticipated Availability Date": row.get("Anticipated Availability Date", DEFAULT_VALUE),
                "Availability": row.get("Availability", DEFAULT_VALUE),
                "Match Analysis": match_analysis
            })
        return matches
    except Exception as e:
        st.error(f"❌ Error finding consultant matches: {e}")
        return []
