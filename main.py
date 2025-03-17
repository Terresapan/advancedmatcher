import os
import streamlit as st
from typing import List, Optional, Union, Dict
from pydantic import BaseModel, Field

# Langchain and AI libraries
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from langsmith import traceable
from langgraph.graph import END, StateGraph
from database import supabase

# Import your functions
from utils import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt, clean_text
from prompt import PROJECT_SUMMARY_PROMPT, CONSULTANT_MATCH_PROMPT

DEFAULT_VALUE = "N/A"
GEMINI_2_0_FLASH = "gemini-2.0-flash"

# Pydantic models for project analysis
class Requirement(BaseModel):
    """A single project requirement."""
    skill: str = Field(description="The required skill (e.g., 'finance', 'marketing', etc.)")
    importance: Optional[str] = Field(None, description="The importance level of this skill")

class ProjectAnalysis(BaseModel):
    """Analysis of a project description to extract requirements."""
    project_name: str = Field(description="The name of the project")
    project_scope: str = Field(description="Brief description of the project scope")
    client_expectations: str = Field(description="What the client expects from this project")
    requirements: List[Requirement] = Field(
        default_factory=list,
        description="List of skills and expertise required for the project"
    )
    industry: Optional[str] = Field(None, description="The industry the project is in, if specified")

class Criterion(BaseModel):
    """A single search criterion for consultant search."""
    field: str = Field(description="The field to search on (e.g., 'finance', 'marketing', 'strategy', 'operations', 'entrepreneurship', 'industry', 'availability')")
    value: str = Field(description="The required value for this field (e.g., 'expertise', 'healthcare', 'available', etc)")

class ProjectRequirement(BaseModel):
    """Analysis of a consultant database query to determine if it's criteria-based."""
    is_criteria_search: bool = Field(
        description="Whether the summary of the project is matching multiple criteria"
    )
    criteria: List[Criterion] = Field(
        default_factory=list,
        description="List of criteria extracted from the summary of the project"
    )
# State model for the LangGraph
class ProjectConsultantState(BaseModel):
    """State for the project-consultant matching flow."""
    query: str = ""
    project_text: str
    analysis: Optional[ProjectRequirement] = None
    project_summary: str = ""
    consultant_matches: List = Field(default_factory=list)
    context: str = ""
    response: str = ""
    session_messages: str = ""

# Initialize embeddings
@st.cache_resource
@traceable(
    metadata={"embedding_model": "google/models/text-embedding-004"},
)
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Function to initialize the LLM
def get_llm(model=GEMINI_2_0_FLASH, temperature=0):
    """Initialize and return the LLM based on the specified model."""    
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        api_key=os.environ["GOOGLE_API_KEY"]
    )

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

# Graph nodes
def analyze_project(state: ProjectConsultantState) -> ProjectConsultantState:
    """Analyze the project text to extract structured information."""
    text = clean_text(state.project_text)
    max_length = 10000
    if len(text) > max_length:
        text = text[:max_length]
    
    llm = get_llm(temperature=0)
    
    # Generate project summary
    try:
        response = llm.invoke(PROJECT_SUMMARY_PROMPT.format(text=text))
        state.project_summary = response.content
        
        # Create a structured output model for the LLM
        structured_llm = llm.with_structured_output(ProjectAnalysis)
        
        # Extract structured information from the summary
        project_analysis_prompt = f"""
        Based on this project summary, extract structured information about the project requirements.
        IMPORTANT: Keep the extraction concise, maximum 100 words.

        Project Summary:
        {state.project_summary}

        Extract the following information:
        1. Project name
        2. Project scope
        3. Client expectations
        4. Required skills - Please provide each skill as a structured requirement with a skill name
        5. Industry (if mentioned)

        For example, a requirement should look like: {{skill: "finance", importance: "high"}}
        """
        
        project_analysis = structured_llm.invoke(project_analysis_prompt)
        
        # Now extract project requirements from the analysis
        project_requirement = extract_project_requirements(project_analysis)
        state.analysis = project_requirement
        
    except Exception as e:
        st.error(f"❌ Error analyzing project: {e}")
    
    return state

def extract_project_requirements(project_analysis: ProjectAnalysis, project_summary: str) -> ProjectRequirement:
    """Extract project requirements from the project analysis."""
    # Initialize the ProjectRequirement object
    project_requirement = ProjectRequirement(
        is_criteria_search=True,  # Always set to True to enforce criteria-based search
        criteria=[]
    )
    
    # Define expertise categories and their related keywords
    expertise_categories = {
        "finance": ["finance", "financial", "accounting", "investment", "budget", "funding", "monetary"],
        "marketing": ["marketing", "branding", "advertising", "sales", "market research", "digital marketing", "promotion"],
        "operations": ["operations", "process", "supply chain", "logistics", "project management", "efficiency"],
        "strategy": ["strategy", "strategic", "planning", "business development", "growth", "roadmap"],
        "entrepreneurship": ["entrepreneurship", "startup", "innovation", "business model", "venture", "founding"]
    }
    
    # Define common industry terms that would match with the Industry Expertise field in the database
    industry_terms = [
        "healthcare", "medical", "pharma", "biotech", 
        "technology", "tech", "software", "IT", "digital",
        "retail", "e-commerce", "consumer goods",
        "manufacturing", "industrial",
        "banking", "insurance", "financial services", "fintech",
        "education", "academic", "training",
        "energy", "oil", "gas", "renewable", "utilities",
        "real estate", "property", "construction",
        "transportation", "logistics", "automotive",
        "media", "entertainment", "gaming",
        "hospitality", "tourism", "restaurant",
        "agriculture", "farming", "food",
        "consulting", "professional services",
        "telecommunications", "telecom",
        "aerospace", "defense",
        "non-profit", "government"
    ]
    
    # Track which categories we've already added
    added_categories = set()
    
    # First check requirements from the project analysis
    if project_analysis.requirements:
        for req in project_analysis.requirements:
            skill_text = req.skill.lower()
            
            # Check each expertise category
            for category, keywords in expertise_categories.items():
                if category in added_categories:
                    continue
                    
                # If the skill matches any keyword in this category
                if any(keyword in skill_text for keyword in keywords):
                    project_requirement.criteria.append(
                        Criterion(field=category, value="expertise")
                    )
                    added_categories.add(category)
    
    # Also check project scope and expectations for expertise keywords
    combined_text = f"{project_analysis.project_name} {project_analysis.project_scope} {project_analysis.client_expectations}".lower()
    
    for category, keywords in expertise_categories.items():
        if category not in added_categories:
            if any(keyword in combined_text for keyword in keywords):
                project_requirement.criteria.append(
                    Criterion(field=category, value="expertise")
                )
                added_categories.add(category)
    
    # Also check the summary text itself for expertise keywords
    for category, keywords in expertise_categories.items():
        if category not in added_categories:
            if any(keyword in project_summary.lower() for keyword in keywords):
                project_requirement.criteria.append(
                    Criterion(field=category, value="expertise")
                )
                added_categories.add(category)
    
    # Add industry as a criterion if specified in the project analysis
    industry_found = False
    if project_analysis.industry:
        # Clean up the industry value - take just the first word or most relevant term
        industry_value = project_analysis.industry.lower().strip()
        # If it contains multiple words, try to extract the main industry term
        if " " in industry_value:
            # Check if any specific industry term exists in the industry value
            for term in industry_terms:
                if term in industry_value:
                    industry_value = term
                    break
            else:
                # If no match, just take the first word which is often the industry name
                industry_value = industry_value.split()[0]
                
        project_requirement.criteria.append(
            Criterion(field="industry", value=industry_value)
        )
        industry_found = True
    
    # If industry not found in project_analysis, try to extract from text
    if not industry_found:
        # Check combined text and project summary for industry terms
        all_text = f"{combined_text} {project_summary}".lower()
        
        for term in industry_terms:
            if term in all_text:
                project_requirement.criteria.append(
                    Criterion(field="industry", value=term)
                )
                industry_found = True
                break
    
    # IMPORTANT: If no criteria found, add at least one default criterion
    # to ensure hybrid search is performed
    if not project_requirement.criteria:
        # Default to strategy as it's broadly applicable
        project_requirement.criteria.append(
            Criterion(field="strategy", value="expertise")
        )
    
    return project_requirement

def build_consultant_text(row):
    """Build text data from a row of consultant data."""
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

def find_matching_consultants(state: ProjectConsultantState, embeddings) -> ProjectConsultantState:
    """Find the best consultant matches based on project summary and criteria."""
    try:
        # Create embedding for the project summary
        query_embedding = embeddings.embed_query(state.project_summary)
        
        # Get the total count of consultants
        count_result = supabase.table('consultants').select('count', count='exact').execute()
        total_consultants = count_result.count if hasattr(count_result, 'count') else 0
        
        # Initialize search parameters with defaults
        search_params = {
            "query_embedding": query_embedding,
            "finance_filter": False,
            "marketing_filter": False,
            "operations_filter": False,
            "strategy_filter": False,
            "entrepreneurship_filter": False,
            "industry_filter": None,
            "availability_filter": False,
            "limit_num": 5
        }
        
        # Update search parameters based on criteria
        # This will always run since we ensure state.analysis has criteria
        for criterion in state.analysis.criteria:
            field = criterion.field.lower()
            value = criterion.value.lower()
            
            if field in ["finance", "marketing", "operations", "strategy", "entrepreneurship"] and value == "expertise":
                search_params[f"{field}_filter"] = True
            elif field == "industry":
                search_params["industry_filter"] = value
            elif field == "availability" and value == "available":
                search_params["availability_filter"] = True
        
        # Execute hybrid search
        result = supabase.rpc("search_consultants", search_params).execute()
        
        # Process results
        if result.data and len(result.data) > 0:
            matches = process_matches(result.data, state.project_summary)
            state.consultant_matches = matches
            state.context = f"Found {len(matches)} consultants matching the project requirements out of {total_consultants} total consultants.\n\n"
        else:
            # Fallback to pure vector search if hybrid returns no results
            fallback_result = supabase.rpc(
                "search_consultants_vector", 
                {
                    "query_embedding": query_embedding,
                    "limit_num": 5
                }
            ).execute()
            
            if fallback_result.data and len(fallback_result.data) > 0:
                matches = process_matches(fallback_result.data, state.project_summary)
                state.consultant_matches = matches
                state.context = f"No consultants matched all specified criteria. Here are the closest matches by semantic similarity ({len(matches)} shown out of {total_consultants} total).\n\n"
            else:
                state.consultant_matches = []
                state.context = "No matching consultants found in the database."
        
        # Add project summary and matches to context
        if state.consultant_matches:
            state.context += f"Project Summary:\n{state.project_summary}\n\n"
            state.context += "Consultant Matches:\n" + "\n\n---\n\n".join(
                [build_consultant_text(match) + f"\n\nMatch Analysis: {match['Match Analysis']}" for match in state.consultant_matches]
            )
            
    except Exception as e:
        st.error(f"❌ Error finding consultant matches: {e}")
        state.context = f"Error finding consultant matches: {str(e)}"
    
    return state

def process_matches(data, project_summary):
    """Process consultant data and add match analysis."""
    matches = []
    for row in data:
        formatted_row = {
            "Name": row.get("Name", "Unknown"),
            "Age": row.get("Age", "Unknown"),
            "Finance Expertise": row.get("Finance Expertise", False),
            "Strategy Expertise": row.get("Strategy Expertise", False),
            "Operations Expertise": row.get("Operations Expertise", False),
            "Marketing Expertise": row.get("Marketing Expertise", False),
            "Entrepreneurship Expertise": row.get("Entrepreneurship Expertise", False),
            "Education": row.get("Education", "Unknown"),
            "Industry Expertise": row.get("Industry Expertise", "Unknown"),
            "Bio": row.get("Bio", "Unknown"),
            "Anticipated Availability Date": str(row.get("Anticipated Availability Date", "Unknown")),
            "Availability": row.get("Availability", False)
        }
        
        consultant_details = build_consultant_text(formatted_row)
        match_analysis = analyze_consultant_match(project_summary, consultant_details)
        formatted_row["Match Analysis"] = match_analysis
        matches.append(formatted_row)
    
    return matches

# Analyze consultant match with AI
@traceable()
def analyze_consultant_match(project_summary, consultant_details, prompt=CONSULTANT_MATCH_PROMPT):
    """Generate detailed analysis of consultant match"""
    llm = get_llm(temperature=0)
    try:
        response = llm.invoke(prompt.format(project_summary=project_summary, consultant_details=consultant_details))
        return response.content
    except Exception as e:
        st.error(f"❌ Error analyzing consultant match: {e}")
        return "Unable to generate detailed match analysis."

def generate_response(state: ProjectConsultantState) -> ProjectConsultantState:
    """Generate a response summarizing the consultant matches."""
    llm = get_llm(temperature=0)
    
    # Use the AI chat prompt template
    ai_prompt = f"""
    You are an AI assistant helping users find the best consultants for their projects. 
    Provide a summary of the consultant matches for the project described below.
    
    Context:
    {state.context}
    
    Recent conversation:
    {state.session_messages}
    
    IMPORTANT INSTRUCTIONS:
    1. Summarize the project requirements first.
    2. Provide a clear, concise summary of each consultant match, highlighting their strengths and potential limitations for this project.
    3. Rank the consultants in order of suitability for the project.
    4. If no consultants match exactly, explain why and suggest how the search could be improved.
    5. Be professional and objective in your assessment.
    """
    
    response = llm.invoke(ai_prompt)
    state.response = response.content
    
    return state

def chat_with_project_consultant_matching(project_text, embeddings, session_messages=""):
    """Chat interface for project-consultant matching using LangGraph."""
    try:
        # Get session messages if in Streamlit
        if hasattr(st, 'session_state') and 'messages' in st.session_state:
            session_messages = ' '.join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]])
        
        # Initialize the state
        initial_state = ProjectConsultantState(
            project_text=project_text,
            session_messages=session_messages
        )
        
        # Define the LangGraph
        workflow = StateGraph(ProjectConsultantState)
        
        # Add nodes
        workflow.add_node("analyze_project", analyze_project)
        workflow.add_node("find_matching_consultants", lambda state: find_matching_consultants(state, embeddings))
        workflow.add_node("generate_response", generate_response)
        
        # Add edges
        workflow.add_edge("analyze_project", "find_matching_consultants")
        workflow.add_edge("find_matching_consultants", "generate_response")
        workflow.add_edge("generate_response", END)
        
        # Set the entry point
        workflow.set_entry_point("analyze_project")
        
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
            return f"Project processed, but couldn't extract response. Result structure: {type(result)}"
        
    except Exception as e:
        return f"Sorry, I encountered an unexpected error: {str(e)}"

def generate_project_summary(project_text):
    """Generate a summary of the project text using LLM.
    
    This function takes the raw project text, cleans it, and generates a concise summary
    using the LLM with the PROJECT_SUMMARY_PROMPT.
    
    Args:
        project_text: The raw text of the project document
        
    Returns:
        A string containing the project summary
    """
    text = clean_text(project_text)
    max_length = 10000
    if len(text) > max_length:
        text = text[:max_length]
    
    llm = get_llm(temperature=0)
    
    try:
        response = llm.invoke(PROJECT_SUMMARY_PROMPT.format(text=text))
        return response.content
    except Exception as e:
        st.error(f"❌ Error generating project summary: {e}")
        return "Unable to generate project summary due to an error."
