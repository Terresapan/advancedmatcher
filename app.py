import streamlit as st
import os
from main import (
    generate_project_summary,
    get_embeddings,
    find_best_consultant_matches,
    process_uploaded_file,
)
from chat import chat_with_consultant_database
from database import sync_consultant_data_to_supabase
from utils import check_password, save_feedback, load_consultant_data


# Set API keys from Streamlit secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["llmapikey"]["GOOGLE_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["tracing"]["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Project-Consultant-Matcher-test"


# Streamlit UI setup
st.set_page_config(page_title="SmartMatch Staffing Platform", layout="wide", page_icon="🤝")

# Setup sidebar with instructions and feedback form
def setup_sidebar():
    """Setup the sidebar with instructions and feedback form."""
    st.sidebar.header("🤝 SmartMatch Staffing Platform")
    st.sidebar.markdown(
        "This app helps you find suitable consultants for your project based on "
        "project description and consultant expertise."
    )
    
    st.sidebar.write("### Instructions")
    st.sidebar.write(
        "1. :key: Enter password to access the app\n"
        "2. :pencil: Upload project description or use Text Query\n"
        "3. :mag: Review matched consultants and analyses\n"
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


# Main Streamlit app
def main():
    """Main application function."""
    setup_sidebar()
    
    if not check_password():
        st.stop()

    st.title("🤝 Project-Consultant Matcher")

    # Create two tabs using radio buttons
    input_method = st.radio("Choose Input Method", ["📂 File Upload", "✍️ Text Query"], horizontal=True)

    if input_method == "📂 File Upload":
        # File upload and processing section
        uploaded_file = st.file_uploader("Upload Project Document", type=["pdf", "docx", "txt"])
        
        # Process new file upload if provided
        if uploaded_file is not None:
            file_text = process_uploaded_file(uploaded_file)
            
            if file_text:
                # Add Find Best Consultants button immediately after file upload
                st.markdown("---")
                if st.button("✨ Find Best Consultants", key="find_consultants"):
                    with st.spinner('⚙️ Processing project document...'):
                        project_summary = generate_project_summary(file_text)
                        st.session_state.project_summary = project_summary
                        st.write("**📋 Project Summary:**")
                        st.write(project_summary)
                        
                        embeddings = get_embeddings()
                        consultant_df = load_consultant_data()
                        if consultant_df is not None:
                            sync_success = sync_consultant_data_to_supabase(consultant_df, embeddings)
                            if sync_success:
                                with st.spinner('🔍 Finding best consultant matches...'):
                                    matches = find_best_consultant_matches(embeddings, project_summary)
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
                                                    st.markdown(f"**📅 Availability:** {consultant['Availability']}")
                                                    st.markdown(f"**📝 Bio:** {consultant['Bio']}")
                                                
                                                st.markdown("---")
                                                st.markdown("**🔍 Match Analysis:**")
                                                st.markdown(consultant['Match Analysis'])
                                    else:
                                        st.error("😔 No matching consultants found.")
                            else:
                                st.error("❌ Could not sync consultant data to Supabase")
                        else:
                            st.error("❌ Could not load consultant data")

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
                        last_5_messages = st.session_state.messages[-5:]
                        session_messages = " ".join(
                            [
                                f"{msg['role']}: {msg['content']}"
                                for msg in last_5_messages
                            ]
                        )
                        response = chat_with_consultant_database(
                            prompt, embeddings, consultant_df, session_messages
                        )
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        st.error("Could not create consultant vector store")

if __name__ == "__main__":
    main()
