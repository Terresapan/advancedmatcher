# PROJECT MODE PROMPT
PROJECT_SUMMARY_PROMPT = """
    Extract and structure the following information from the project document within 100 words:
    1. Project Name: Create one according to the context if not given
    2. Project Scope
    3. Client Expectations
    4. Skills Needed

    Project Document:
    {text}

    Provide the output in a clear, concise format. If any information is not clearly mentioned, use 'Not Specified' or make a reasonable inference based on the context."""

EXTRACT_STRUCURED_ANALYSIS_PROMPT = """
    Based on this project summary, extract structured information about the project requirements.
    IMPORTANT: Keep the extraction concise, maximum 100 words.

    Project Summary:
    {summary}

    Extract: project name, scope, expectations, skills (as requirements), industry (if mentioned).
    Example requirement: {{skill: "finance", importance: "high"}}"""

CONSULTANT_MATCH_PROMPT = """
    Analyze the match between this project and the consultant:

    Project Summary:
    {project_summary}

    Consultant Details:
    {consultant_details}

    Provide a detailed assessment that includes:
    1. Strengths of this consultant for the project within 100 words
    2. Potential limitations or challenges within 100 words
    3. Overall suitability rating (out of 10)

    Your analysis should be constructive, highlighting both positive aspects and areas of potential concern."""

GENERATE_RESPONSE_PROMPT = """
    You are an AI assistant helping users find the best consultants for their projects. 
    Provide a summary of the consultant matches for the project described below.
    
    Context:
    {context}

    Consultant matches:
    {consultant_matches}
    
    IMPORTANT INSTRUCTIONS:
    1. Do NOT summarize the project requirements, since you have done this step in the previous step.
    2. Provide a clear, concise summary of each consultant match, highlighting their strengths and potential limitations for this project.
    3. Rank the consultants in order of suitability for the project.
    4. If no consultants match exactly, explain why and suggest how the search could be improved.
    5. Be professional and objective in your assessment."""


# CHAT MODEL PROMPT
QUERY_ANALYSIS_PROMPT = """
    Analyze this query about consultants and determine if it's asking for consultants matching multiple criter  ia.
    Query: "{query}"
    
    Focus only on expertise fields (e.g., finance, marketing), industry expertise (e.g., healthcare), and availability.
    For queries about 'expertise in [area]', treat [area] as the field (e.g., "finance", "marketing") and "expertise" as the value.
    
    Examples:
    1. "Find consultants with finance expertise who know healthcare industry" should extract:
       - field: "finance", value: "expertise"
       - field: "industry", value: "healthcare"
    
    2. Please consider typos that might be in the query. for example: "Find consultants who have expertise in both stretagy and operation" should extract:
       - field: "operations", value: "expertise"
       - field: "strategy", value: "expertise"

    3. "Looking for consultants with PhDs and skilled in operations and with experience in tech companies" should extract:
       - field: "operations", value: "expertise"
       - field: "industry", value: "tech"

    4. "Who is available next month?" should extract:
       - field: "Consultant Availability Status", value: "available"

    5. The following query should indicate is_criteria_search: False
       - Tell me about the consultant database
       - get me consultants who have PHD degree
       - get me consultants who speak Spanish
       - get me consultants who are based in New York

    6. "I need a consultant who is an expert in entrepreneurship and available immediately" should extract:
       - field: "entrepreneurship", value: "expertise"
       - field: "Consultant Availability Status", value: "available"""

GENERATE_CHAT_RESPONSE_PROMPT = """
    You are an AI assistant helping users find consultants in a database. 
    Answer the following query based on the context provided below.
    
    Context:
    {context}
    
    Recent conversation:
    {session_messages}
    
    User Query: {query}
    
    IMPORTANT INSTRUCTIONS:
    1. Mention the TOTAL number of consultants in the database only when:
       - Providing overviews or summaries of the database
       - Answering questions about the database size
       - Describing the consultant pool in general terms  
    2. Be explicit about how many results are being shown vs. the total in the database.
    3. Provide a clear, concise answer that directly addresses the user's question.
    4. If multiple consultants match the criteria, summarize their key qualifications.
    5. If no consultants match exactly, suggest the closest matches and explain why.
    6, If the user's question is not about consultant matching, politely redirect them to the correct section."""