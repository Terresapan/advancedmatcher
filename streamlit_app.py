import streamlit as st
from copy import deepcopy
import os
from main import (
    get_embeddings,
    process_uploaded_file,
    build_project_consultant_workflow,
    ProjectConsultantState,
    format_criteria_for_display,
    update_criteria_from_user_input,
)
from chat import chat_with_consultant_database
from database import sync_consultant_data_to_supabase
from utils import check_password, save_feedback, load_consultant_data
from langgraph.graph import END
from langgraph.types import Command


# Set API keys from Streamlit secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["llmapikey"]["GOOGLE_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["tracing"]["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Project-Consultant-Matcher-test"


# Streamlit UI setup
st.set_page_config(page_title="Interactive SmartMatch Staffing Platform", layout="wide", page_icon="🤝")

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

    # st.sidebar.write("### 🎧 Listen to our Podcast for more insights")

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


def find_consultants_with_human_approval(file_text):
    """Process project document with human approval of requirements."""
    from main import ProjectRequirement, Criterion
    embeddings = get_embeddings()
    workflow_app = build_project_consultant_workflow(embeddings=embeddings)
    initial_state = ProjectConsultantState(project_text=file_text, requirements_approved=False)

    with st.spinner("⏳ Analyzing project document..."):
        # Create a unique thread ID for this session
        if 'thread_id' not in st.session_state:
            import uuid
            st.session_state['thread_id'] = str(uuid.uuid4())

        thread_config = {"configurable": {"thread_id": st.session_state['thread_id']}}

        # Invoke the workflow - it will run until the interrupt in analyze_project
        result = workflow_app.invoke(initial_state, config=thread_config)

        # Get the current state to access the interrupt information
        state = workflow_app.get_state(thread_config)

        if hasattr(state, 'tasks') and state.tasks and hasattr(state.tasks[0], 'interrupts'):
            # We have an interrupt, extract the information
            interrupt_info = state.tasks[0].interrupts[0].value if state.tasks[0].interrupts else None

            if interrupt_info:
                st.session_state['project_summary'] = interrupt_info.get('project_summary', "")
                st.session_state['analysis'] = interrupt_info.get('analysis', None)
        else:
            # No interrupt, use the result directly
            state = ProjectConsultantState(**result)
            st.session_state['project_summary'] = state.project_summary
            st.session_state['analysis'] = state.analysis

    if not st.session_state.get('project_summary') or not st.session_state.get('analysis'):
        st.warning("⚠️ Analysis failed to generate summary or requirements.")
        st.session_state['project_summary'] = "This project requires consultant expertise in various domains."
        st.session_state['analysis'] = ProjectRequirement(
            is_criteria_search=True,
            criteria=[Criterion(field="strategy", value="expertise")]
        )

    st.session_state['workflow_app'] = workflow_app
    st.session_state.file_processed = True
    st.session_state.requirements_editing = True
    return st.session_state['project_summary'], st.session_state['analysis'], workflow_app


def main():
    """Main application function."""
    setup_sidebar()
    
    if not check_password():
        st.stop()

    st.title("🤝 Project-Consultant Matcher")

    # Initialize session state variables if not yet initialized
    if 'processed' not in st.session_state:
        st.session_state.processed = False
        st.session_state.show_approval = False
        st.session_state.selected_expertise = []
        st.session_state.industry_value = ''
        st.session_state.availability_value = False
        st.session_state.additional_text = ''
        st.session_state.file_processed = False
        st.session_state.requirements_editing = False

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
                        
                        # Get the project summary, analysis, and workflow app
                        project_summary, analysis, workflow_app = find_consultants_with_human_approval(file_text)
                        
                        # Store the project summary and analysis in session state
                        st.session_state['project_state'] = ProjectConsultantState(
                            project_text=file_text,
                            project_summary=project_summary,
                            analysis=analysis,
                            requirements_approved=False
                        )
                        
                        # Initialize default values from extracted criteria
                        if analysis is not None:
                            formatted_criteria = format_criteria_for_display(analysis.criteria)
                            all_expertise_fields = ["Finance", "Marketing", "Operations", "Strategy", "Entrepreneurship"]
                            st.session_state.selected_expertise = [expertise.capitalize() for expertise in formatted_criteria["expertise"] 
                                                                if expertise.capitalize() in all_expertise_fields]
                            st.session_state.industry_value = formatted_criteria["industry"] if formatted_criteria["industry"] else ""
                            st.session_state.availability_value = formatted_criteria["availability"]
                    else:
                        st.error("❌ Failed to process the uploaded file. Please check the file format and try again.")
            
            # Display the requirements editing UI if file has been processed
            if st.session_state.file_processed and st.session_state.requirements_editing:
                st.markdown("---")
                state = st.session_state['project_state']
                workflow_app = st.session_state['workflow_app']
                
                st.subheader("Project Summary")
                st.write(state.project_summary)
                
                if state.analysis is None:
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
                        with st.spinner('🔍Finding the best consultant matches...'):
                            # Get user inputs
                            updated_requirements, additional_text = update_criteria_from_user_input(
                                st.session_state.selected_expertise,
                                st.session_state.industry_value,
                                st.session_state.availability_value,
                                st.session_state.additional_text
                            )

                            # Get the workflow app from session state
                            workflow_app = st.session_state['workflow_app']

                            # Get the thread ID from session state
                            thread_config = {"configurable": {"thread_id": st.session_state['thread_id']}}

                            try:
                                # Get the current state from the workflow
                                current_state = workflow_app.get_state(thread_config)

                                # Create a new state with the updated requirements
                                # We need to preserve the original state structure but update specific fields
                                # Always include project_summary and project_text to ensure they're not lost
                                original_summary = st.session_state.get('project_summary', "")
                                original_text = st.session_state.get('project_text', "")
                                
                                state_update = {
                                    "analysis": updated_requirements,
                                    "requirements_approved": True,
                                    "project_summary": original_summary,
                                    "project_text": original_text
                                }

                                # Add additional context if provided
                                if additional_text.strip():
                                    # Update both project_summary and project_text to include additional context
                                    state_update["project_summary"] = state_update["project_summary"] + f"\n\nAdditional Context: {additional_text}"
                                    state_update["project_text"] = state_update["project_text"] + f"\n\nAdditional Context: {additional_text}"
                                    
                                    print(f"Added additional context: {additional_text}")

                                # Debug print
                                print("State update:", state_update)

                                # Resume the workflow with the updated state
                                # Use Command to resume with the updated requirements
                                final_result = workflow_app.invoke(
                                    Command(resume=True, update=state_update),
                                    config=thread_config
                                )

                                print("Workflow completed successfully")

                                # Convert final_result to ProjectConsultantState if it's a dict
                                if isinstance(final_result, dict):
                                    final_state = ProjectConsultantState(**final_result)
                                else:
                                    final_state = final_result

                                st.session_state['project_state'] = final_state
                                st.session_state.requirements_approved = True

                                st.write("### Consultant Matches")
                                
                                # Get the consultant matches from the final state
                                matches = getattr(final_state, 'consultant_matches', [])
                                st.session_state.current_matches = matches
                                
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
                                # response = getattr(final_state, 'response', "")
                                # if response:
                                #     with st.expander("💡 AI Analysis of Matches"):
                                #         st.write(response)

                            except Exception as e:
                                st.error(f"Error resuming workflow: {e}")
                                print(f"Error: {e}")
                                st.write("An error occurred while processing your request.")
            
            # Display results if requirements have been approved
            # elif st.session_state.file_processed and not st.session_state.requirements_editing and st.session_state.get('requirements_approved', False):
            #     st.markdown("---")
            #     state = st.session_state['project_state']
                
            #     st.subheader("Project Summary")
            #     st.write(state.project_summary)
            #     st.write("### Consultant Matches")
                
            #     # Get the consultant matches from the state
            #     matches = getattr(state, 'consultant_matches', [])
            #     st.session_state.current_matches = matches
                
            #     if matches:
            #         st.write("🎯 **Best Matching Consultants**")
            #         for i, consultant in enumerate(matches, 1):
            #             with st.expander(f"👨‍💼 Consultant {i}: {consultant['Name']}"):
            #                 cols = st.columns(2)
            #                 with cols[0]:
            #                     st.markdown(f"**👤 Age:** {consultant['Age']}")
            #                     st.markdown(f"**🎓 Education:** {consultant['Education']}")
            #                     st.markdown(f"**💼 Industry Expertise:** {consultant['Industry Expertise']}")
            #                 with cols[1]:
            #                     st.markdown(f"**📅 Availability:** {'Yes' if consultant['Availability'] else 'No'}")
            #                     st.markdown(f"**📝 Bio:** {consultant['Bio']}")
                            
            #                 st.markdown("---")
            #                 st.markdown("**🔍 Match Analysis:**")
            #                 st.markdown(consultant['Match Analysis'])
            #     else:
            #         st.error("😔 No matching consultants found.")
                    
                # Also display the AI-generated response if available
                # response = getattr(state, 'response', "")
                # if response:
                #     with st.expander("💡 AI Analysis of Matches"):
                #         st.write(response)

                    
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
                            response = chat_with_consultant_database(
                                prompt, embeddings, consultant_df, session_messages
                            )
                            
                            # Check if the response is empty or None
                            if not response or response == "No response generated.":
                                response = "I couldn't generate a specific response based on your query. Could you provide more details about what you're looking for in a consultant?"
                                
                            st.markdown(response)
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

