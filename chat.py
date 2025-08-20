from langgraph.graph import END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.docstore.document import Document
from langsmith import traceable
from typing import List, Optional
from pydantic import BaseModel, Field
import streamlit as st
import os

from database import supabase
from prompt import QUERY_ANALYSIS_PROMPT, GENERATE_CHAT_RESPONSE_PROMPT

# Constants
GEMINI_FLASH = "gemini-2.5-flash"

# -------------------
# States Definition
# -------------------
class Criterion(BaseModel):
    """A single search criterion for consultant search."""
    field: str = Field(description="The field to search on (e.g., 'finance', 'marketing', 'strategy', 'operations', 'entrepreneurship', 'industry', 'availability')")
    value: str = Field(description="The required value for this field (e.g., 'expertise', 'healthcare', 'available', etc)")

class QueryAnalysis(BaseModel):
    """Analysis of a consultant database query to determine if it's criteria-based."""
    is_criteria_search: bool = Field(
        description="Whether the query is asking for consultants matching multiple criteria"
    )
    criteria: List[Criterion] = Field(
        default_factory=list,
        description="List of criteria extracted from the query"
    )

class ConsultantQueryState(BaseModel):
    """State for the consultant query processing flow."""
    query: str
    analysis: Optional[QueryAnalysis] = None
    filtered_results: List = Field(default_factory=list)
    context: str = ""
    response: str = ""
    session_messages: str = ""

   
# -------------------
# LLM Initialization
# -------------------
def get_llm(model=GEMINI_FLASH, temperature=0):
    """Initialize and return the LLM based on the specified model."""    
    return ChatGoogleGenerativeAI(
        model=GEMINI_FLASH,
        temperature=temperature,
        api_key=os.environ["GOOGLE_API_KEY"]
    )


# -------------------
# Graph Nodes
# -------------------
def analyze_query(state: ConsultantQueryState) -> ConsultantQueryState:
    """Analyze the query to detect if it's a criteria-based search."""
    try:
        llm = get_llm(temperature=0)
        structured_llm = llm.with_structured_output(QueryAnalysis)    

        analysis = structured_llm.invoke(QUERY_ANALYSIS_PROMPT.format(query=state.query))

        state.analysis = analysis    
    except Exception as e:
        st.error(f"analyze_query: Error analyzing query: {e}")
        state.analysis = QueryAnalysis(is_criteria_search=False, criteria=[])
    return state

def search_consultants(state: ConsultantQueryState, embeddings) -> ConsultantQueryState:
    """Search for consultants using structured filters then vector search with Supabase."""
    query_embedding = embeddings.embed_query(state.query)

    # Get total count of consultants
    try:
        count_result = supabase.table('consultants').select('count', count='exact').execute()
        total_consultants = count_result.count if hasattr(count_result, 'count') else "unknown"
    except Exception as e:
        st.error(f"❌ Error counting consultants: {e}")
        total_consultants = "unknown"

    # Handle special queries
    if handle_special_queries(state, total_consultants):
        return state
        
    # Get a sample of consultants to provide in the overview
    try:
        sample_result = supabase.table('consultants').select('*').limit(5).execute()
    except Exception as e:
        st.error(f"❌ Error getting sample consultants: {e}")
        sample_result = type('obj', (object,), {'data': []})()

    if handle_overview_requests(state, total_consultants, sample_result):
        return state

    # Perform criteria-based search
    params = extract_search_params(state)
    results = perform_search(params, query_embedding, supabase)

    if not results:
        # Fallback to vector search
        fallback_params = {
            "query_embedding": query_embedding,
            "finance_filter": None,
            "marketing_filter": None,
            "operations_filter": None,
            "strategy_filter": None,
            "entrepreneurship_filter": None,
            "industry_filter": None,
            "availability_filter": None,
            "limit_num": 10
        }
        results = perform_search(fallback_params, query_embedding, supabase)
        documents, context = format_search_results(results, total_consultants, is_fallback=True)
    else:
        documents, context = format_search_results(results, total_consultants)

    state.filtered_results = documents
    state.context = context
    return state

def generate_chat_response(state: ConsultantQueryState) -> ConsultantQueryState:
    """Generate a response using the context and conversation history."""
    llm = get_llm(temperature=0)
    response = llm.invoke(GENERATE_CHAT_RESPONSE_PROMPT.format(
        context=state.context,
        session_messages=state.session_messages,
        query=state.query,
    ))
    state.response = response.content
    return state

# -------------------
# Main Chat Function
# -------------------

def chat_with_consultant_database(embeddings):
    """Chat with the consultant database using a direct LLM approach instead of LangGraph."""
    
    # Define the LangGraph workflow
    workflow = StateGraph(ConsultantQueryState)
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("search_consultants", lambda state: search_consultants(state, embeddings))
    workflow.add_node("generate_response", generate_chat_response)

    workflow.add_edge("analyze_query", "search_consultants")
    workflow.add_edge("search_consultants", "generate_response")
    workflow.add_edge("generate_response", END)
    workflow.set_entry_point("analyze_query")
    
    # Compile the graph
    chatapp = workflow.compile() 
    return chatapp

# -------------------
# Helper Functions
# -------------------
def build_text_data(row: dict) -> str:
    """Format consultant data into a readable string using the column mapping."""
    fields = [
        ('Name', 'name'),
        ('Age', 'age'),
        ('Finance Expertise', 'finance_expertise'),
        ('Strategy Expertise', 'strategy_expertise'),
        ('Operations Expertise', 'operations_expertise'),
        ('Marketing Expertise', 'marketing_expertise'),
        ('Entrepreneurship Expertise', 'entrepreneurship_expertise'),
        ('Education', 'education'),
        ('Industry Expertise', 'industry_expertise'),
        ('Bio', 'bio'),
        ('Anticipated Availability Date', 'anticipated_availability_date'),
        ('Availability', 'availability')
    ]
    return "; ".join(f"{display}: {row.get(db_field, 'Unknown')}" for display, db_field in fields) + ";"

def build_text_data_normal(row: dict) -> str:
    """Format consultant data into a readable string."""
    return (
        f"Name: {row.get('Name', 'Unknown')}; "
        f"Age: {row.get('Age', 'Unknown')}; "
        f"Finance Expertise: {row.get('Finance Expertise', 'Unknown')}; "
        f"Strategy Expertise: {row.get('Strategy Expertise', 'Unknown')}; "
        f"Operations Expertise: {row.get('Operations Expertise', 'Unknown')}; "
        f"Marketing Expertise: {row.get('Marketing Expertise', 'Unknown')}; "
        f"Entrepreneurship Expertise: {row.get('Entrepreneurship Expertise', 'Unknown')}; "
        f"Education: {row.get('Education', 'Unknown')}; "
        f"Industry Expertise: {row.get('Industry Expertise', 'Unknown')}; "
        f"Bio: {row.get('Bio', 'Unknown')}; "
        f"Anticipated Availability Date: {row.get('Anticipated Availability Date', 'Unknown')}; "
        f"Availability: {row.get('Availability', 'Unknown')};"
    )

def handle_special_queries(state: ConsultantQueryState, total_consultants: int) -> bool:
    """Handle queries asking about the number of consultants."""
    query_lower = state.query.lower()
    if ("how many" in query_lower and "consultant" in query_lower) or ("number of" in query_lower and "consultant" in query_lower):
        state.context = f"There are a total of {total_consultants} consultants in the database."
        state.filtered_results = []
        return True
    return False

def handle_overview_requests(state: ConsultantQueryState, total_consultants: int, sample_result) -> bool:
    """Handle requests for a database overview."""
    query_lower = state.query.lower()
    if any(phrase in query_lower for phrase in ["overview", "summarize", "summary", "tell me about"]) and \
       any(term in query_lower for term in ["database", "consultant", "profile"]):
        state.context = f"DATABASE OVERVIEW: The consultant database contains a total of {total_consultants} consultants with various expertise and backgrounds."
        if sample_result.data:
            # print("Actual keys in the data:", sample_result.data[0].keys())
            formatted_samples = [build_text_data(row) for row in sample_result.data]
            # print(formatted_samples)
            state.filtered_results = [
                Document(page_content=text, metadata=row) for text, row in zip(formatted_samples, sample_result.data)
            ]
            state.context += (
                f"\n\nHere are a few examples of consultants in the database (showing 5 of {total_consultants}):\n\n" +
                "\n\n---\n\n".join(formatted_samples)
            )
            expertise_counts = {
                "Finance": sum(1 for row in sample_result.data if row.get("Finance Expertise")),
                "Strategy": sum(1 for row in sample_result.data if row.get("Strategy Expertise")),
                "Operations": sum(1 for row in sample_result.data if row.get("Operations Expertise")),
                "Marketing": sum(1 for row in sample_result.data if row.get("Marketing Expertise")),
                "Entrepreneurship": sum(1 for row in sample_result.data if row.get("Entrepreneurship Expertise"))
            }
            state.context += (
                f"\n\nExpertise distribution in sample: Finance ({expertise_counts['Finance']}/5), "
                f"Strategy ({expertise_counts['Strategy']}/5), Operations ({expertise_counts['Operations']}/5), "
                f"Marketing ({expertise_counts['Marketing']}/5), Entrepreneurship ({expertise_counts['Entrepreneurship']}/5)"
            )
        return True
    return False

@traceable
def extract_search_params(state: ConsultantQueryState) -> dict:
    """Extract search parameters from the query analysis."""
    params = {
        "query_embedding": None,
        "finance_filter": None,
        "marketing_filter": None,
        "operations_filter": None,
        "strategy_filter": None,
        "entrepreneurship_filter": None,
        "industry_filter": None,
        "availability_filter": None,
        "limit_num": 20
    }
    # Check if state has analysis attribute and it's not None
    if hasattr(state, 'analysis') and state.analysis:
        if state.analysis.is_criteria_search and state.analysis.criteria:
            for criterion in state.analysis.criteria:
                field = criterion.field.lower()
                value = criterion.value.lower()
                if field in ["finance", "marketing", "operations", "strategy", "entrepreneurship"]:
                    if value == "expertise":
                        params[f"{field}_filter"] = True
                elif field == "industry":
                    params["industry_filter"] = value
                elif field == "consultant availability status":
                    params["availability_filter"] = value
    return params

@traceable
def perform_search(params: dict, query_embedding, supabase) -> list:
    """Perform the search using Supabase RPC."""
    params["query_embedding"] = query_embedding
    try:
        result = supabase.rpc("search_consultants", params).execute()
        return result.data if result.data else []
    except Exception as e:
        st.error(f"❌ Error searching consultants: {e}")
        return []
    

def format_search_results(results: list, total_consultants: int, is_fallback: bool = False) -> tuple:
    """Format search results into documents and context."""
    if not results:
        context = f"There are a total of {total_consultants} consultants in the database, but no matches were found."
        return [], context
    
    documents = [
        Document(
            page_content=build_text_data_normal(row),
            metadata={k: v for k, v in row.items() if k not in ["embedding", "distance"]}
        ) for row in results
    ]
    if is_fallback:
        context = (
            f"There are a total of {total_consultants} consultants in the database.\n\n"
            f"No consultants match ALL specified criteria exactly. "
            f"Here are the closest matches ({len(results)} shown out of {total_consultants} total):\n\n" +
            "\n\n---\n\n".join([doc.page_content for doc in documents])
        )
    else:
        context = (
            f"There are a total of {total_consultants} consultants in the database.\n\n"
            f"Consultants matching ALL specified criteria ({len(results)} shown out of {total_consultants} total):\n\n" +
            "\n\n---\n\n".join([doc.page_content for doc in documents])
        )
    return documents, context
