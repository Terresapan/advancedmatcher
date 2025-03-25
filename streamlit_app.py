import streamlit as st
import os
from main import (
    get_embeddings,
    build_project_consultant_workflow,
    format_criteria_for_display,
    update_criteria_from_user_input,
    ProjectConsultantState,
)
from chat import chat_with_consultant_database, ConsultantQueryState
from database import sync_consultant_data_to_supabase
from utils import check_password, save_feedback, load_consultant_data, process_uploaded_file
from langgraph.types import Command
import uuid


# Set API keys from Streamlit secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["llmapikey"]["GOOGLE_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["tracing"]["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Project-Consultant-Matcher - Advanced Interaction"


# Streamlit UI setup
st.set_page_config(page_title="Interactive SmartMatch Staffing Platform", layout="wide", page_icon="👐")

# Setup sidebar with instructions and feedback form
def setup_sidebar():
    """Setup the sidebar with instructions and feedback form."""
    st.sidebar.header("🤝 Interactive SmartMatch Staffing Platform")
    st.sidebar.markdown(
        "This app helps you find suitable consultants for your project based on "
        "project description and consultant expertise."
    )

    st.sidebar.write("### Instructions")
    st.sidebar.write(
        "1. :key: Enter password to access the app\n"
        "2. :pencil: Upload project description or use Text Query\n"
        "3. :mag: Provide feedback or approve AI-generated criteria to receive matching consultant candidates.\n"
        "4. :speech_balloon: Chat with our consultant database"
    )

    st.sidebar.write("### 🎧 Listen to our Podcast for more insights")
    st.sidebar.markdown(
        "[Interactive SmartMatch Staffing Platform Podcast Link](https://open.spotify.com/episode/1HA0LDPBgbQVzCkJilvKVe)"
    )

    st.sidebar.write("### 🌎 Visit my AI Agent Projects Website")
    st.sidebar.markdown(
        "[Terresa Pan's Agent Garden Link](https://ai-agents-garden.lovable.app/)"
    )
    
    # Feedback section
    if 'feedback' not in st.session_state:
        st.session_state.feedback = ""

    st.sidebar.markdown("---")
    st.sidebar.subheader("💭 Feedback")
    feedback = st.sidebar.text_area(
        "Share your thoughts",
        value=st.session_state.feedback,
        placeholder="Your feedback helps us improve..."
    )

    if st.sidebar.button("📤 Submit Feedback"):
        if feedback:
            try:
                save_feedback(feedback)
                st.session_state.feedback = ""
                st.sidebar.success("✨ Thank you for your feedback!")
            except Exception as e:
                st.sidebar.error(f"❌ Error saving feedback: {str(e)}")
        else:
            st.sidebar.warning("⚠️ Please enter feedback before submitting")

    st.sidebar.image("assets/bot01.jpg", use_container_width=True)


def main():
    """Main application function."""
    setup_sidebar()
    
    if not check_password():
        st.stop()

    st.title("🤝 Project-Consultant Matcher")

    # Initialize session state variables if not yet initialized
    if 'processed' not in st.session_state:
        st.session_state.processed = False
        st.session_state.selected_expertise = []
        st.session_state.industry_value = ''
        st.session_state.availability_value = False
        st.session_state.additional_text = ''
        st.session_state.file_processed = False
        st.session_state.requirements_editing = False
        st.session_state.thread_id = None
        st.session_state.workflow_app = None

    # Create two tabs using radio buttons
    input_method = st.radio("Choose Input Method", ["📂 File Upload", "✍️ Text Query"], horizontal=True)

    if input_method == "📂 File Upload":
        # File upload and processing section
        uploaded_file = st.file_uploader("Upload Project Document", type=["pdf", "docx", "txt"])
        
        if uploaded_file is not None:
            # Process button - only process the file when explicitly requested
            if not st.session_state.file_processed and st.button("✨ Generate Summary and Approve the Criteria"):
                with st.spinner('⚙️ Processing document...'):
                    file_text = process_uploaded_file(uploaded_file)
                    if file_text:
                        # Store the original text for later use
                        st.session_state['project_text'] = file_text
                        
                        # Initialize workflow
                        embeddings = get_embeddings()
                        workflow_app = build_project_consultant_workflow(embeddings=embeddings)
                        initial_state = ProjectConsultantState(project_text=file_text, requirements_approved=False)
                        
                        # Set thread ID for state persistence
                        if not st.session_state.thread_id:
                            st.session_state.thread_id = str(uuid.uuid4())
                        thread_config = {"configurable": {"thread_id": st.session_state.thread_id}}
                        
                        # Invoke workflow until interrupt
                        workflow_app.invoke(initial_state, config=thread_config)
                        state = workflow_app.get_state(thread_config)
                        # print(f"Current Workflow state: {state}")

                        # Check for interrupt from human_review
                        if state and hasattr(state, 'tasks') and state.tasks:
                            for task in state.tasks:
                                if hasattr(task, 'interrupts') and task.interrupts:
                                    interrupt_info = task.interrupts[0].value
                                    st.session_state['project_summary'] = interrupt_info.get('project_summary', "")
                                    st.session_state['analysis'] = interrupt_info.get('analysis', None)
                                    
                                    # Format the criteria for display
                                    if st.session_state['analysis']:
                                        formatted_criteria = format_criteria_for_display(st.session_state['analysis'].criteria)
                                        st.session_state.selected_expertise = formatted_criteria['expertise']
                                        st.session_state.industry_value = formatted_criteria['industry'] or ""
                                        st.session_state.availability_value = formatted_criteria['availability']
                                    
                                    st.session_state.file_processed = True
                                    st.session_state.requirements_editing = True
                                    st.session_state.workflow_app = workflow_app
                                    
                                    # Force a rerun to display the requirements editing UI
                                    st.rerun()
                                    break
                            else:
                                st.error("❌ Failed to process the project analysis.")
                        else:
                            st.error("❌ Failed to process the project analysis.")
            
            # Display the requirements editing UI if file has been processed
            if st.session_state.file_processed and st.session_state.requirements_editing:
                st.markdown("---")
                st.subheader("Project Summary")
                st.write(st.session_state['project_summary'])
                
                if st.session_state['analysis'] is None:
                    st.error("❌ Error: Project analysis failed. Please try again.")
                else:
                    st.write("### Extracted Expertise Requirements")
                    all_expertise_fields = ["Finance", "Marketing", "Operations", "Strategy", "Entrepreneurship"]
                    
                    st.session_state.selected_expertise = st.multiselect(
                        "Select required expertise areas:",
                        all_expertise_fields,
                        default=st.session_state.selected_expertise,
                        key="expertise_multiselect" 
                    )
                    
                    st.write("### Industry")
                    st.session_state.industry_value = st.text_input(
                        "Industry (leave blank if not applicable):",
                        value=st.session_state.industry_value,
                        key="industry_input" 
                    )
                    
                    st.session_state.availability_value = st.checkbox(
                        "Require consultant to be currently available",
                        value=st.session_state.availability_value,
                        key="availability_checkbox" 
                    )
                    
                    st.write("### Additional Context")
                    st.session_state.additional_text = st.text_area(
                        "Add any additional context for the search (optional):", 
                        value=st.session_state.additional_text, 
                        key="additional_text_area"
                    )

                    # Only find consultants when the button is clicked
                    if st.button("✅ Approve and Find Consultants"):
                        # Show a spinner to indicate processing is happening
                        with st.spinner('🔍Finding the best consultant matches...'):
                            # Step 1: Update project requirements based on user input from session state
                            updated_requirements, additional_text = update_criteria_from_user_input(
                                st.session_state.selected_expertise,  # User's selected expertise
                                st.session_state.industry_value,      # Selected industry
                                st.session_state.availability_value,  # Consultant availability preference
                                st.session_state.additional_text      # Any additional context provided
                            )

                            # Step 2: Prepare the state update dictionary for the workflow
                            state_update = {
                                    "analysis": updated_requirements,     # Updated project requirements
                                    "requirements_approved": True,        # Mark requirements as approved
                                    "project_summary": st.session_state['project_summary'],  # Original project summary
                                    "project_text": st.session_state['project_text']        # Original project text
                            }
                            # Append additional context to summary and text if provided
                            if additional_text.strip():
                                state_update["project_summary"] += f"\n\nAdditional Context: {additional_text}"
                                state_update["project_text"] += f"\n\nAdditional Context: {additional_text}"

                            # Step 3: Resume the workflow with the updated requirements and context
                            try:
                                # Retrieve the workflow app and thread configuration from session state
                                workflow_app = st.session_state['workflow_app']
                                thread_config = {"configurable": {"thread_id": st.session_state['thread_id']}}

                                # Resume the workflow with the updated state
                                final_result = workflow_app.invoke(Command(resume=True, update=state_update), config=thread_config)
                                
                                # Step 4: Process the workflow result
                                final_state = ProjectConsultantState(**final_result)  # Convert result to state object
                                st.session_state['project_state'] = final_state       # Store final state
                                st.session_state.requirements_approved = True         # Update approval status
                                
                                # Step 5: Display consultant matches
                                st.write("### Consultant Matches")
                                matches = final_state.consultant_matches  # Get list of matched consultants
                                st.session_state.current_matches = matches  # Store matches in session state
                                
                                if matches:
                                    st.write("🎯 **Best Matching Consultants**")
                                    for i, consultant in enumerate(matches, 1):
                                        with st.expander(f"👨‍💼 Consultant {i}: {consultant['Name']}"):
                                            cols = st.columns(2)
                                            with cols[0]:
                                                st.markdown(f"**👤 Age:** {consultant['Age']}")
                                                st.markdown(f"**🎓 Education:** {consultant['Education']}")
                                                st.markdown(f"**💼 Industry Expertise:** {consultant['Industry Expertise']}")
                                            with cols[1]:
                                                st.markdown(f"**📅 Availability:** {'Yes' if consultant['Availability'] else 'No'}")
                                                st.markdown(f"**📝 Bio:** {consultant['Bio']}")
                                            
                                            st.markdown("---")
                                            st.markdown("**🔍 Match Analysis:**")
                                            st.markdown(consultant['Match Analysis'])
                                else:
                                    st.error("😔 No matching consultants found.")
                                    
                                # Also display the AI-generated response if available
                                response = final_state.response
                                if response:
                                    with st.expander("💡 AI Analysis of Matches"):
                                        st.write(response)

                            except Exception as e:
                                st.error(f"Error resuming workflow: {e}")
                                print(f"Error: {e}")
                                st.write("An error occurred while processing your request.")

                    
    else:  # Text Query tab
        # Initialize chat messages if not already done
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("💭 Ask about consultant matching..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    # Get embeddings and vector store for context
                    embeddings = get_embeddings()
                    consultant_df = load_consultant_data()
                    sync_success = sync_consultant_data_to_supabase(consultant_df, embeddings)
                    if sync_success:
                        # Get the last 5 messages for context
                        last_5_messages = st.session_state.messages[-5:]
                        session_messages = " ".join(
                            [
                                f"{msg['role']}: {msg['content']}"
                                for msg in last_5_messages
                            ]
                        )
                        try:
                            # Use the LangGraph-based chat implementation
                            chatapp = chat_with_consultant_database(embeddings)
                            initial_state = ConsultantQueryState(
                                query=prompt,
                                session_messages=session_messages                       
                            )
                            final_state = chatapp.invoke(initial_state)
                            # print(type(final_state)) ---- <class 'langgraph.pregel.io.AddableValuesDict'> it's a dict-like object but not a standard dict

                            if isinstance(final_state, dict) and "response" in final_state:
                                # Extract the response from the final state
                                response = final_state["response"]
                                st.markdown(response)  

                            # We can also use the following line to extract the response
                            # response = final_state.get("response", "I couldn't generate a response. Please try again.")
                            # response = dict(final_state)["response"]
                            
                            # however response = final_state.response does not work
                            # st.markdown(response)                                                       
                            st.session_state.messages.append({"role": "assistant", "content": response})

                        except Exception as e:
                            error_msg = f"An error occurred while processing your request: {str(e)}"
                            st.error(error_msg)
                            st.markdown("I'm having trouble processing your request. Could you try rephrasing your question or providing more specific details?")
                            st.session_state.messages.append({"role": "assistant", "content": "I'm having trouble processing your request. Could you try rephrasing your question or providing more specific details?"})
                    else:
                        st.error("Could not create consultant vector store")

if __name__ == "__main__":
    main()

