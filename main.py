import streamlit as st
import os

# Langchain and AI libraries
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from langsmith import traceable
import asyncio

# Import database, functions, prompts and states
from database import supabase
from utils import clean_text
from prompt import PROJECT_SUMMARY_PROMPT, CONSULTANT_MATCH_PROMPT, EXTRACT_STRUCURED_ANALYSIS_PROMPT, GENERATE_RESPONSE_PROMPT
from states import ProjectConsultantState, ProjectRequirement, Criterion, ProjectAnalysis, Requirement

DEFAULT_VALUE = "N/A"
GEMINI_2_0_FLASH = "gemini-2.0-flash"

# Initialize embeddings
@st.cache_resource
@traceable(
    metadata={"embedding_model": "google/models/text-embedding-004"},
)
def get_embeddings():
    """
    Initializes and returns the Google Generative AI Embeddings model,
    ensuring an asyncio event loop is available.
    """
    try:
        # Check if an event loop is already running
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # If not, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Now, the initialization will work correctly
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Function to initialize the LLM
def get_llm(model=GEMINI_2_0_FLASH, temperature=0):
    """Initialize and return the LLM based on the specified model."""    
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        api_key=os.environ["GOOGLE_API_KEY"]
    )

# Function to analyze the project text and extract structured information
def analyze_project(state: ProjectConsultantState) -> ProjectConsultantState:
    """Analyze project text to extract structured information."""
    if state.requirements_approved:
        return state

    text = clean_text(state.project_text)[:10000]
    llm = get_llm(temperature=0)
    
    state.project_summary = generate_project_summary(llm, text)
    project_analysis = extract_structured_analysis(llm, state.project_summary)
    state.analysis = extract_project_requirements(project_analysis, state.project_summary)
    return state

# Function to handle human review of the analysis
def human_review_analysis(state: ProjectConsultantState) -> ProjectConsultantState:
    """Interrupt the graph to let the human review and modify analysis."""
    # print("Entering human_review_analysis node")
    # print(f"Project summary: {state.project_summary}")
    # print(f"Analysis: {state.analysis}")
    
    # Create the interrupt with the current state information
    interrupt({
        "project_summary": state.project_summary,
        "analysis": state.analysis,
        "message": "Please review and approve the project requirements"
    })
    
    # When the workflow resumes, it will have the updated state from the Command(resume=True, update=state_update)
    # The state will be updated by the resume command in the streamlit app
    # print("Workflow resumed after human review")
    return state

# Function to find matching consultants based on project criteria
def find_matching_consultants(state: ProjectConsultantState, embeddings) -> ProjectConsultantState:
    """Find consultant matches based on project criteria."""
    if state.analysis is None:
        st.warning("⚠️ No analysis found. Using default criteria.")
        state.analysis = ProjectRequirement(
            is_criteria_search=True, criteria=[Criterion(field="strategy", value="expertise")]
        )

    query_embedding = embeddings.embed_query(state.project_summary)
    search_params = {
        "query_embedding": query_embedding, "limit_num": 3,
        "finance_filter": False, "marketing_filter": False, "operations_filter": False,
        "strategy_filter": False, "entrepreneurship_filter": False,
        "industry_filter": None, "availability_filter": False
    }

    for criterion in state.analysis.criteria:
        field, value = criterion.field.lower(), criterion.value.lower()
        if field in ["finance", "marketing", "operations", "strategy", "entrepreneurship"] and value == "expertise":
            search_params[f"{field}_filter"] = True
        elif field == "industry":
            search_params["industry_filter"] = value
        elif field == "availability" and value == "available":
            search_params["availability_filter"] = True

    try:
        result = supabase.rpc("search_consultants", search_params).execute()
        state.consultant_matches = process_matches(result.data, state.project_summary) if result.data else []
        if not state.consultant_matches:
            fallback_result = supabase.rpc("search_consultants_vector", {"query_embedding": query_embedding, "limit_num": 3}).execute()
            state.consultant_matches = process_matches(fallback_result.data, state.project_summary) if fallback_result.data else []
            state.context = "No exact matches found; showing closest consultants." if state.consultant_matches else "No matching consultants found."
        else:
            state.context = f"Found {len(state.consultant_matches)} matching consultants."
    except Exception as e:
        st.error(f"❌ Error querying consultants: {e}")
        state.context = "Error occurred while finding consultants."

    return state

# Function to generate a response summarizing the consultant matches
def generate_response(state: ProjectConsultantState) -> ProjectConsultantState:
    """Generate a response summarizing the consultant matches."""
    # print("Entered generate_response")
    # print(f"Consultant matches: {state.consultant_matches}")
    llm = get_llm(temperature=0)
  
    response = llm.invoke(GENERATE_RESPONSE_PROMPT.format(context=state.context, consultant_matches=state.consultant_matches))
    state.response = response.content
    
    return state


# Function to build the LangGraph workflow
def build_project_consultant_workflow(embeddings):
    """Build the LangGraph workflow for project-consultant matching."""

    # Initialize the graph
    workflow = StateGraph(ProjectConsultantState)

    # Add nodes
    workflow.add_node("analyze_project", analyze_project)
    workflow.add_node("human_review", human_review_analysis)
    workflow.add_node("find_matching_consultants", lambda state: find_matching_consultants(state, embeddings))
    workflow.add_node("generate_response", generate_response)

    # Add edges
    workflow.add_edge("analyze_project", "human_review")
    workflow.add_edge("human_review", "find_matching_consultants")
    workflow.add_edge("find_matching_consultants", "generate_response")
    workflow.add_edge("generate_response", END)

    # Set the entry point
    workflow.set_entry_point("analyze_project")

    # Create a memory checkpointer for the graph
    checkpointer = MemorySaver()

    # Compile the graph with the checkpointer
    # This is required for interrupt to work
    compiled = workflow.compile(checkpointer=checkpointer)

    return compiled


##########################################################################################################################
# HELPER FUNCTIONS FOR GRAPH 
##########################################################################################################################

@traceable
# Helper function to generate a concise project summary
def generate_project_summary(llm, text: str) -> str:
    """Generate a concise project summary using the LLM."""
    llm = get_llm(temperature=0)
    try:
        response = llm.invoke(PROJECT_SUMMARY_PROMPT.format(text=text))
        if hasattr(response, 'content') and response.content and len(response.content.strip()) > 20:
            return response.content
        st.warning("⚠️ Generated summary is too short. Using default.")
        return "This project requires consultant expertise in various domains."
    except Exception as e:
        st.error(f"❌ Error generating summary: {e}")
        return "This project requires consultant expertise in various domains."

@traceable
# Helper function to extract structured analysis from the summary
def extract_structured_analysis(llm, summary: str) -> ProjectAnalysis:
    """Extract structured project analysis from the summary."""
    structured_llm = llm.with_structured_output(ProjectAnalysis)
    try:
        return structured_llm.invoke(EXTRACT_STRUCURED_ANALYSIS_PROMPT.format(summary=summary))
    except Exception as e:
        st.error(f"❌ Error extracting analysis: {e}")
        return ProjectAnalysis(
            project_name="Unknown", project_scope="Unknown", client_expectations="Unknown",
            requirements=[Requirement(skill="strategy")]
        )

@traceable
# Helper function to extract project requirements from the analysis
def extract_project_requirements(project_analysis: ProjectAnalysis, project_summary: str) -> ProjectRequirement:
    """Extract project requirements from the analysis, prioritizing structured data."""
    project_requirement = ProjectRequirement(is_criteria_search=True, criteria=[])
    added_categories = set()

    expertise_categories = {
        "finance": ["finance", "financial", "accounting", "investment"],
        "marketing": ["marketing", "branding", "advertising", "sales"],
        "operations": ["operations", "process", "logistics"],
        "strategy": ["strategy", "planning", "business development"],
        "entrepreneurship": ["entrepreneurship", "startup", "innovation"]
    }
    industry_terms = ["healthcare", "technology", "retail", "education", "finance", "digital"]

    # Create combined text from project analysis for later use
    combined_text = f"{project_analysis.project_name} {project_analysis.project_scope} {project_analysis.client_expectations}".lower()

    # Prioritize requirements from analysis
    for req in project_analysis.requirements or []:
        skill = req.skill.lower()
        for category, keywords in expertise_categories.items():
            if category not in added_categories and any(kw in skill for kw in keywords):
                project_requirement.criteria.append(Criterion(field=category, value="expertise"))
                added_categories.add(category)

    # Fallback to text only if no requirements found
    if not project_requirement.criteria:
        for category, keywords in expertise_categories.items():
            if category not in added_categories and any(kw in combined_text for kw in keywords):
                project_requirement.criteria.append(Criterion(field=category, value="expertise"))
                added_categories.add(category)

    # Handle industry
    industry = project_analysis.industry.lower().strip() if project_analysis.industry else None
    if not industry:
        for term in industry_terms:
            if term in f"{combined_text} {project_summary}".lower():
                industry = term
                break
    if industry:
        project_requirement.criteria.append(Criterion(field="industry", value=industry))

    # Ensure at least one criterion
    if not project_requirement.criteria:
        project_requirement.criteria.append(Criterion(field="strategy", value="expertise"))

    return project_requirement

@traceable
# Helper function to extract consultant data from the database and add match analysis
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

@traceable
# Helper function to build consultant text
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

@traceable
# Helper function to analyze consultant match with AI
def analyze_consultant_match(project_summary, consultant_details, prompt=CONSULTANT_MATCH_PROMPT):
    """Generate detailed analysis of consultant match"""
    llm = get_llm(temperature=0)
    try:
        response = llm.invoke(prompt.format(project_summary=project_summary, consultant_details=consultant_details))
        return response.content
    except Exception as e:
        st.error(f"❌ Error analyzing consultant match: {e}")
        return "Unable to generate detailed match analysis."
    

##########################################################################################################################
# HELPER FUNCTIONS FOR FRONTEND DISAPLAY 
##########################################################################################################################

# Helper function to format criteria for display
def format_criteria_for_display(criteria):
    """Format criteria for display in the UI."""
    expertise_fields = []
    industry = None
    availability = False
    
    # Handle case where criteria might be None
    if criteria is None:
        return {
            "expertise": [],
            "industry": None,
            "availability": False
        }
    
    # Print debug information
    # print(f"Formatting criteria for display: {criteria}")
    
    for criterion in criteria:
        # print(f"Processing criterion: {criterion.field} = {criterion.value}")
        if criterion.field.lower() in ["finance", "marketing", "operations", "strategy", "entrepreneurship"] and criterion.value.lower() == "expertise":
            # Capitalize the field name to match the UI options
            expertise_fields.append(criterion.field.capitalize())
        elif criterion.field.lower() == "industry":
            industry = criterion.value
        elif criterion.field.lower() == "availability" and criterion.value.lower() == "available":
            availability = True
    
    result = {
        "expertise": expertise_fields,
        "industry": industry,
        "availability": availability
    }
    # print(f"Formatted criteria result: {result}")
    return result


# Helper function to update criteria from user input
def update_criteria_from_user_input(selected_expertise, industry_value, availability_value, additional_text):
    """Update project requirements based on user input."""
    criteria = []
    
    # Add expertise criteria
    for expertise in selected_expertise:
        criteria.append(Criterion(field=expertise.lower(), value="expertise"))
    
    # Add industry criterion if provided
    if industry_value and industry_value.strip():
        # Clean up the industry value - remove any special characters and extra spaces
        cleaned_industry = industry_value.lower().strip()
        # If the industry contains multiple words, keep them but ensure it's clean
        cleaned_industry = " ".join([word.strip() for word in cleaned_industry.split() if word.strip()])
        criteria.append(Criterion(field="industry", value=cleaned_industry))
    
    # Add availability criterion if selected
    if availability_value:
        criteria.append(Criterion(field="availability", value="available"))
    
    # Ensure we have at least one criterion
    if not criteria:
        # Default to strategy as a fallback criterion
        criteria.append(Criterion(field="strategy", value="expertise"))
    
    # Create the project requirement object
    project_requirement = ProjectRequirement(
        is_criteria_search=True,
        criteria=criteria
    )
    
    # print(f"Created project requirement from user input: {project_requirement}")
    # print(f"Criteria: {[f'{c.field}: {c.value}' for c in project_requirement.criteria]}")
    
    return project_requirement, additional_text
