from typing import List, Optional
import os
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
import streamlit as st

from database import supabase
from langchain.docstore.document import Document

# Constants
GEMINI_2_0_FLASH = "gemini-2.0-flash"

# Pydantic model for query analysis
class Criterion(BaseModel):
    """A single search criterion for consultant search."""
    field: str = Field(description="The field to search on (e.g., 'finance', 'marketing', 'strategy', 'operations', 'entrepreneurship', 'industry')")
    value: str = Field(description="The required value for this field (e.g., 'expertise', 'healthcare')")

class QueryAnalysis(BaseModel):
    """Analysis of a consultant database query to determine if it's criteria-based."""
    is_criteria_search: bool = Field(
        description="Whether the query is asking for consultants matching multiple criteria"
    )
    criteria: List[Criterion] = Field(
        default_factory=list,
        description="List of criteria extracted from the query"
    )

# State model for the LangGraph
class ConsultantQueryState(BaseModel):
    """State for the consultant query processing flow."""
    query: str
    analysis: Optional[QueryAnalysis] = None
    filtered_results: List = Field(default_factory=list)
    context: str = ""
    response: str = ""
    session_messages: str = ""

# Function to initialize the LLM
def get_llm(model=GEMINI_2_0_FLASH, temperature=0):
    """Initialize and return the LLM based on the specified model."""    
    return ChatGoogleGenerativeAI(
        model=GEMINI_2_0_FLASH,
        temperature=temperature,
        api_key=os.environ["GOOGLE_API_KEY"]
    )

# Graph nodes
def analyze_query(state: ConsultantQueryState) -> ConsultantQueryState:
    """Analyze the query to detect if it's a criteria-based search."""
    llm = get_llm(temperature=0)
    
    # Create a structured output model for the LLM
    structured_llm = llm.with_structured_output(QueryAnalysis)
    
    query_analysis_prompt = f"""
    Analyze this query about consultants and determine if it's asking for consultants matching multiple criteria.
    Query: "{state.query}"
    
    Focus only on expertise fields (e.g., finance, marketing), industry expertise, and availability.
    For queries about 'expertise in [area]', treat [area] as the field (e.g., "finance", "marketing") and "expertise" as the value.
    
    Examples:
    1. "Find consultants with finance expertise who know healthcare industry" should extract:
       - field: "finance", value: "expertise"
       - field: "industry", value: "healthcare"
    
    2. Please consider typos that might be in the query. for example: "Find consultants who have expertise in both stretagy and operation" should extract:
       - field: "operations", value: "expertise"
       - field: "strategy", value: "expertise"

    3. "Looking for consultants skilled in operations and with experience in tech companies" should extract:
       - field: "operations", value: "expertise"
       - field: "industry", value: "tech"

    4. "Who is available next month?" should extract:
       - field: "Consultant Availability Status", value: "available"

    5. "Tell me about the consultant database" should NOT be a criteria search.

    6. "I need a consultant who is an expert in entrepreneurship and available immediately" should extract:
       - field: "entrepreneurship", value: "expertise"
       - field: "Consultant Availability Status", value: "available"
    """
    
    # Get structured output from the LLM
    analysis = structured_llm.invoke(query_analysis_prompt)
    state.analysis = analysis
    
    return state

def build_text_data(row):
    """Build text data from a row of data."""
    return (f"Name: {row.get('Name', 'Unknown')}; "
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
            f"Availability: {row.get('Availability', 'Unknown')};")

def search_consultants(state: ConsultantQueryState, embeddings) -> ConsultantQueryState:
    """Search for consultants using structured filters then vector search with Supabase."""
    query_embedding = embeddings.embed_query(state.query)

    # First, get the total count of consultants in the database
    try:
        count_result = supabase.table('consultants').select('count', count='exact').execute()
        total_consultants = count_result.count if hasattr(count_result, 'count') else "unknown"
    except Exception as e:
        st.error(f"❌ Error counting consultants: {e}")
        total_consultants = "unknown"

    # Add a specific check for queries about the number of consultants
    if ("how many" in state.query.lower() and "consultant" in state.query.lower()) or \
       ("number of" in state.query.lower() and "consultant" in state.query.lower()):
        state.context = f"There are a total of {total_consultants} consultants in the database."
        state.filtered_results = []
        return state

     # Check for database overview requests
    if any(phrase in state.query.lower() for phrase in ["overview", "summarize", "summary", "tell me about"]) and \
       any(term in state.query.lower() for term in ["database", "consultant", "profile"]):
        # For overview requests, include the total count prominently
        state.context = f"DATABASE OVERVIEW: The consultant database contains a total of {total_consultants} consultants with various expertise and backgrounds."
        
    # Get a sample of consultants to provide in the overview
    try:
        sample_result = supabase.table('consultants').select('*').limit(5).execute()
        if sample_result.data:
            # Convert database column names to display format
            formatted_samples = []
            for row in sample_result.data:
                formatted_row = {
                    "Name": row.get("name", "Unknown"),
                    "Age": row.get("age", "Unknown"),
                    "Finance Expertise": row.get("finance_expertise", False),
                    "Strategy Expertise": row.get("strategy_expertise", False),
                    "Operations Expertise": row.get("operations_expertise", False),
                    "Marketing Expertise": row.get("marketing_expertise", False),
                    "Entrepreneurship Expertise": row.get("entrepreneurship_expertise", False),
                    "Education": row.get("education", "Unknown"),
                    "Industry Expertise": row.get("industry_expertise", "Unknown"),
                    "Bio": row.get("bio", "Unknown"),
                    "Anticipated Availability Date": str(row.get("anticipated_availability_date", "Unknown")),
                    "Availability": row.get("availability", False)
                }
                formatted_samples.append(formatted_row)
            
            state.filtered_results = [
                Document(
                    page_content=build_text_data(row),
                    metadata=row
                )
                for row in formatted_samples
            ]
            state.context += "\n\nHere are a few examples of consultants in the database (showing 5 of " + str(total_consultants) + "):\n\n" + "\n\n---\n\n".join(
                [doc.page_content for doc in state.filtered_results]
            )
            
            # Add a summary of expertise distribution
            expertise_counts = {
                "Finance": sum(1 for row in formatted_samples if row.get("Finance Expertise")),
                "Strategy": sum(1 for row in formatted_samples if row.get("Strategy Expertise")),
                "Operations": sum(1 for row in formatted_samples if row.get("Operations Expertise")),
                "Marketing": sum(1 for row in formatted_samples if row.get("Marketing Expertise")),
                "Entrepreneurship": sum(1 for row in formatted_samples if row.get("Entrepreneurship Expertise"))
            }
            
            state.context += f"\n\nExpertise distribution in sample: Finance ({expertise_counts['Finance']}/5), Strategy ({expertise_counts['Strategy']}/5), Operations ({expertise_counts['Operations']}/5), Marketing ({expertise_counts['Marketing']}/5), Entrepreneurship ({expertise_counts['Entrepreneurship']}/5)"
            
    except Exception as e:
        st.error(f"❌ Error getting sample consultants: {e}")
        state.context += f"\n\nError retrieving sample consultants: {str(e)}"
    
    params = {
        "query_embedding": query_embedding,
        "finance_filter": None,
        "marketing_filter": None,
        "operations_filter": None,
        "strategy_filter": None,
        "entrepreneurship_filter": None,
        "industry_filter": None,
        "availability_filter": None,
        "limit_num": 20
    }
    
    if state.analysis and state.analysis.is_criteria_search and state.analysis.criteria:
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
    
    try:

        result = supabase.rpc("search_consultants", params).execute()
        
        if result.data:
            state.filtered_results = [
                Document(
                    page_content=build_text_data(row),
                    metadata={k: v for k, v in row.items() if k not in ["embedding", "distance"]}
                )
                for row in result.data
            ]
            # Include total count in the context
            state.context = f"There are a total of {total_consultants} consultants in the database.\n\nConsultants matching ALL specified criteria ({len(result.data)} shown out of {total_consultants} total):\n\n" + "\n\n---\n\n".join(
                [doc.page_content for doc in state.filtered_results]
            )
        else:
            fallback_result = supabase.rpc("search_consultants", {
                "query_embedding": query_embedding,
                "finance_filter": None,
                "marketing_filter": None,
                "operations_filter": None,
                "strategy_filter": None,
                "entrepreneurship_filter": None,
                "industry_filter": None,
                "availability_filter": None,
                "limit_num": 10
            }).execute()
            state.filtered_results = [
                Document(
                    page_content=build_text_data(row),
                    metadata={k: v for k, v in row.items() if k not in ["embedding", "distance"]}
                )
                for row in fallback_result.data
            ]
            # Include total count in the context
            state.context = f"There are a total of {total_consultants} consultants in the database.\n\nNo consultants match ALL specified criteria exactly. Here are the closest matches ({len(fallback_result.data)} shown out of {total_consultants} total):\n\n" + "\n\n---\n\n".join(
                [doc.page_content for doc in state.filtered_results]
            )
    except Exception as e:
        st.error(f"❌ Error searching consultants: {e}")
        state.filtered_results = []
        state.context = f"There are a total of {total_consultants} consultants in the database, but an error occurred while searching: {str(e)}"
    
    return state

def generate_response(state: ConsultantQueryState) -> ConsultantQueryState:
    """Generate a response using the context and conversation history."""
    llm = get_llm(temperature=0)
    
    # Use the AI chat prompt template
    ai_prompt = f"""
    You are an AI assistant helping users find consultants in a database. 
    Answer the following query based on the context provided below.
    
    Context:
    {state.context}
    
    Recent conversation:
    {state.session_messages}
    
    User Query: {state.query}
    
    IMPORTANT INSTRUCTIONS:
    1. Mention the TOTAL number of consultants in the database (not just the ones shown in results) only when:
       - Providing overviews or summaries of the database
       - Answering questions about the database size
       - Describing the consultant pool in general terms
    
    2. Be explicit about how many results are being shown vs. the total in the database.
    
    3. Provide a clear, concise answer that directly addresses the user's question.
    
    4. If multiple consultants match the criteria, summarize their key qualifications.
    
    5. If no consultants match exactly, suggest the closest matches and explain why.
    """
    
    response = llm.invoke(ai_prompt)
    state.response = response.content
    
    return state

def chat_with_consultant_database(prompt, embeddings, df, model=GEMINI_2_0_FLASH):
    """
    Chat with the consultant database with improved handling of column-based queries.
    This enhanced version uses LangGraph and structured outputs.
    """
    try:
        # Get session messages if in Streamlit
        session_messages = ""
        if hasattr(st, 'session_state') and 'messages' in st.session_state:
            session_messages = ' '.join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]])
        
        # Initialize the state
        initial_state = ConsultantQueryState(
            query=prompt,
            session_messages=session_messages
        )
        
        # Define the LangGraph
        workflow = StateGraph(ConsultantQueryState)
        
        # Add nodes
        workflow.add_node("analyze_query", analyze_query)
        workflow.add_node("search_consultants", lambda state: search_consultants(state, embeddings))
        workflow.add_node("generate_response", generate_response)
        
        # Add edges
        workflow.add_edge("analyze_query", "search_consultants")
        workflow.add_edge("search_consultants", "generate_response")
        workflow.add_edge("generate_response", END)
        
        # Set the entry point
        workflow.set_entry_point("analyze_query")
        
        # Compile the graph
        app = workflow.compile()
        
        # Run the graph
        result = app.invoke(initial_state)
        
        # In LangGraph, the result is a dictionary of the final state
        # We need to access the response attribute from the state
        if hasattr(result, 'response'):
            return result.response
        elif isinstance(result, dict) and 'state' in result and hasattr(result['state'], 'response'):
            return result['state'].response
        elif isinstance(result, dict) and 'response' in result:
            return result['response']
        else:
            # Try to access the last node's output directly
            nodes = list(result.keys())
            if nodes and 'generate_response' in nodes:
                return result['generate_response'].response
            
            # Fallback to returning a diagnostic message
            return f"Query processed, but couldn't extract response. Result structure: {type(result)}"
        
    except Exception as e:
        return f"Sorry, I encountered an unexpected error: {str(e)}"

