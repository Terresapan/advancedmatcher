PROJECT_SUMMARY_PROMPT = """Extract and structure the following information from the project document:
    1. Project Name: Create one according to the context if not given
    2. Project Scope
    3. Client Expectations
    4. Skills Needed

    Project Document:
    {text}

    Provide the output in a clear, concise format. If any information is not clearly mentioned, use 'Not Specified' or make a reasonable inference based on the context."""

CONSULTANT_MATCH_PROMPT = """Analyze the match between this project and the consultant:

Project Summary:
{project_summary}

Consultant Details:
{consultant_details}

Provide a detailed assessment that includes:
1. Strengths of this consultant for the project within 100 words
2. Potential limitations or challenges within 100 words
3. Overall suitability rating (out of 10)

Your analysis should be constructive, highlighting both positive aspects and areas of potential concern."""

AI_CHAT_PROMPT = """You are a helpful project-consultant matching assistant. Use the following context to provide a detailed, specific answer:

                    Relevant Context:
                    {context}

                    User's Question:
                    {prompt}

                    Previous Messages:
                    {session_messages}

                    Instructions:
                    1. Answer the specific question asked
                    2. Reference relevant consultant details when appropriate
                    3. Keep the response focused and concise
                    4. Use the context to provide accurate information
                    5. If you don't have enough information or there is no match, just say so and do not make up information.
                    6, If the user's question is not about consultant matching, politely redirect them to the correct section."""