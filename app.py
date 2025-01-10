import streamlit as st
from main import (
    generate_project_summary,
    get_embeddings,
    create_consultant_vector_store,
    find_best_consultant_matches,
    process_uploaded_file,
    chat_with_consultant_database
)
from utils import check_password, save_feedback, load_consultant_data

# Streamlit UI setup
st.set_page_config(page_title="Project-Consultant Matcher", layout="wide", page_icon="🤝")

# Setup sidebar with instructions and feedback form
def setup_sidebar():
    """Setup the sidebar with instructions and feedback form."""
    st.sidebar.header("🤝 Project-Consultant Matcher")
    st.sidebar.markdown(
        "This app helps you find the most suitable consultants for your project based on your project description and the consultant's expertise. "
        "To use this app, you'll need to first enter password to access the app."
    )
    st.sidebar.write("### Instructions")
    st.sidebar.write(":pencil: Upload your project description and Find Best Consultants.")
    st.sidebar.write(":point_right: Feel free to chat with our consultant database in the Text Query session.")
    st.sidebar.write(":heart_decoration: Let me know your thoughts and feedback about the app.")

    # Initialize feedback session state
    if 'feedback' not in st.session_state:
        st.session_state.feedback = ""

    # Feedback form
    st.sidebar.subheader("Feedback Form")
    feedback = st.sidebar.text_area(
        "Your thoughts and feedback", 
        value=st.session_state.feedback, 
        placeholder="Share your feedback here..."
    )

    if st.sidebar.button("Submit Feedback"):
        if feedback:
            try:
                save_feedback(feedback)
                st.session_state.feedback = ""  # Clear feedback after submission
                st.sidebar.success("Thank you for your feedback! 😊")
            except Exception as e:
                st.sidebar.error(f"Error saving feedback: {str(e)}")
        else:
            st.sidebar.error("Please enter your feedback before submitting.")

    st.sidebar.image("assets/TAP01.jpg", use_container_width=True)

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
        
        if uploaded_file is not None:
            file_text = process_uploaded_file(uploaded_file)
            
            if file_text:
                # Add Find Best Consultants button immediately after file upload
                st.markdown("---")
                if st.button("✨ Find Best Consultants", key="find_consultants"):
                    with st.spinner('⚙️ Processing project document...'):
                        project_summary = generate_project_summary(file_text)
                        st.session_state.project_summary = project_summary
                        st.write("**Project Summary:**")
                        st.write(project_summary)
                        
                        embeddings = get_embeddings()
                        consultant_df = load_consultant_data()
                        if consultant_df is not None:
                            vector_store = create_consultant_vector_store(embeddings, consultant_df)
                            if vector_store:
                                with st.spinner('🔍 Finding best consultant matches...'):
                                    matches = find_best_consultant_matches(vector_store, project_summary)
                                    st.session_state.current_matches = matches
                                    if matches:
                                        st.write("🎯 **Best Matching Consultants**")
                                        for i, consultant in enumerate(matches, 1):
                                            with st.expander(f"👨‍💼 Consultant {i}: {consultant['Name']}"):
                                                cols = st.columns(2)
                                                with cols[0]:
                                                    st.markdown(f"**Age:** {consultant['Age']}")
                                                    st.markdown(f"**Education:** {consultant['Education']}")
                                                    st.markdown(f"**Domain:** {consultant['Domain']}")
                                                with cols[1]:
                                                    st.markdown(f"**Availability:** {consultant['Availability']}")
                                                    st.markdown(f"**Bio:** {consultant['Bio']}")
                                                
                                                st.markdown("---")
                                                st.markdown("**Match Analysis:**")
                                                st.markdown(consultant['Match Analysis'])
                                    else:
                                        st.error("😔 No matching consultants found.")
                            else:
                                st.error("❌ Could not create consultant vector store")
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
        if prompt := st.chat_input("Ask about consultant matching..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Get embeddings and vector store for context
                    embeddings = get_embeddings()
                    consultant_df = load_consultant_data()
                    vector_store = create_consultant_vector_store(embeddings, consultant_df)
                    if vector_store:
                        response = chat_with_consultant_database(prompt, vector_store)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        st.error("Could not create consultant vector store")

if __name__ == "__main__":
    main()