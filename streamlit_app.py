import streamlit as st
import os
import re
import hmac

# Langchain and AI libraries
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langsmith import traceable
from utils import save_feedback

# Streamlit UI setup
st.set_page_config(page_title="Project-Consultant Matcher", layout="wide", page_icon="🤝")

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

def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False

# File processing libraries
import PyPDF2

# Attempt to import python-docx, but provide fallback
try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    st.warning("⚠️ python-docx library not installed. Word document support will be limited.")

# Set API keys from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["general"]["OPENAI_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]["API_KEY"]
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Project-Consultant-Matcher-TAP"
# --- Project Data ---

# File processing functions
def extract_text_from_pdf(file):
    """Extract text from a PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
        return text
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")
        return ""

def extract_text_from_docx(file):
    """Extract text from a Word document"""
    if not HAS_DOCX:
        st.warning("⚠️ Cannot process Word documents. Please install python-docx.")
        return ""
    try:
        doc = docx.Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"❌ Error reading DOCX: {e}")
        return ""

def extract_text_from_txt(file):
    """Extract text from a text file"""
    try:
        return file.read().decode('utf-8')
    except Exception as e:
        st.error(f"❌ Error reading text file: {e}")
        return ""

# Initialize embeddings
@st.cache_resource
@traceable(
    metadata={"embedding_model": "openai/text-embedding-3-small"},
)
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")

# Load data from Google Sheets
@st.cache_data(ttl=600)
def load_consultant_data():
    try:
        from streamlit_gsheets import GSheetsConnection
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(worksheet="Database", ttl="10m", usecols=[0, 1, 2, 3, 4, 5], nrows=1000)
        return df
    except Exception as e:
        st.error(f"❌ Error loading consultant data: {e}")
        return None

# Project summary function using AI
def generate_project_summary(text):
    """Generate structured project summary using AI"""
    text = re.sub(r'\s+', ' ', text).strip()
    max_length = 10000
    if len(text) > max_length:
        text = text[:max_length]
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.2,
        api_key=os.environ["OPENAI_API_KEY"]
    )
    prompt = f"""Extract and structure the following information from the project document:
    1. Project Name: Create one according to the context if not given
    2. Project Scope
    3. Client Expectations
    4. Skills Needed

    Project Document:
    {text}

    Provide the output in a clear, concise format. If any information is not clearly mentioned, use 'Not Specified' or make a reasonable inference based on the context."""
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        st.error(f"❌ Error generating project summary: {e}")
        return "Unable to generate summary."

# Create vector store for consultants
@st.cache_resource
@traceable(
    metadata={"vectordb": "FAISS"}
)
def create_consultant_vector_store(_embeddings, df):
    if df is None or df.empty:
        st.error("❌ Consultant DataFrame is None or empty")
        return None
    try:
        df.columns = ['Name', 'Age', 'Education', 'Domain', 'Bio', 'Availability']
        text_data = [
            f"Name: {name}; Age: {age}; Education: {education}; Domain: {domain}; Bio: {bio}; Availability: {availability}"
            for name, age, education, domain, bio, availability in zip(
                df['Name'], df['Age'], df['Education'], df['Domain'], df['Bio'], df['Availability']
            )
        ]
        metadatas = df.to_dict('records')
        vector_store = FAISS.from_texts(
            texts=text_data, 
            embedding=_embeddings,
            metadatas=metadatas
        )
        return vector_store
    except Exception as e:
        st.error(f"❌ Error creating consultant vector store: {e}")
        return None

# Analyze consultant match with AI
def analyze_consultant_match(project_summary, consultant_details):
    """Generate detailed analysis of consultant match"""
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.2,
        api_key=os.environ["OPENAI_API_KEY"]
    )
    prompt = f"""Analyze the match between this project and the consultant:

Project Summary:
{project_summary}

Consultant Details:
{consultant_details}

Provide a detailed assessment that includes:
1. Strengths of this consultant for the project within 100 words
2. Potential limitations or challenges within 100 words
3. Overall suitability rating (out of 10)

Your analysis should be constructive, highlighting both positive aspects and areas of potential concern."""
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        st.error(f"❌ Error analyzing consultant match: {e}")
        return "Unable to generate detailed match analysis."

# Find best consultant matches
@traceable(
        run_type="retriever"
)
def find_best_consultant_matches(vector_store, project_summary, top_k=5):
    """Find the best consultant matches based on project summary"""
    if not vector_store:
        return []
    try:
        results = vector_store.similarity_search(project_summary, k=top_k)
        matches = []
        for result in results:
            consultant_details = "\n".join([
                f"{key}: {value}" for key, value in result.metadata.items()
            ])
            match_analysis = analyze_consultant_match(project_summary, consultant_details)
            matches.append({
                "Name": result.metadata.get('Name', 'N/A'),
                "Age": result.metadata.get('Age', 'N/A'),
                "Education": result.metadata.get('Education', 'N/A'),
                "Domain": result.metadata.get('Domain', 'N/A'),
                "Bio": result.metadata.get('Bio', 'N/A'),
                "Availability": result.metadata.get('Availability', 'N/A'),
                "Match Analysis": match_analysis
            })
        return matches
    except Exception as e:
        st.error(f"❌ Error finding consultant matches: {e}")
        return []

# Main Streamlit app
@traceable()
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
            file_text = ""
            if uploaded_file.type == "application/pdf":
                file_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                if HAS_DOCX:
                    file_text = extract_text_from_docx(uploaded_file)
                else:
                    st.error("❌ Cannot process Word documents. Please install python-docx.")
                    return
            elif uploaded_file.type == "text/plain":
                file_text = extract_text_from_txt(uploaded_file)
            else:
                st.error("❌ Unsupported file type")
                return
            
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

### Rewritten Code:

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
                    # Use AI to generate response
                    llm = ChatOpenAI(
                        model="gpt-4o-mini", 
                        temperature=0.2,
                        api_key=os.environ["OPENAI_API_KEY"]
                    )
                    
                    # Get embeddings and vector store for context
                    embeddings = get_embeddings()
                    consultant_df = load_consultant_data()
                    vector_store = create_consultant_vector_store(embeddings, consultant_df)
                    vector_results = vector_store.similarity_search(prompt, k=3)
                    
                    # Create context from vector search results
                    context = "\n".join([doc.page_content for doc in vector_results])
                    
                    # Create prompt for the AI
                    ai_prompt = f"""You are a helpful project-consultant matching assistant. Use the following context to provide a detailed, specific answer:

                    Relevant Context:
                    {context}

                    User's Question:
                    {prompt}

                    Previous Messages:
                    {' '.join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]])}

                    Instructions:
                    1. Answer the specific question asked
                    2. Reference relevant consultant details when appropriate
                    3. Keep the response focused and concise
                    4. Use the context to provide accurate information
                    5. If you don't have enough information or there is no match, just say so and do not make up information.
                    6, If the user's question is not about consultant matching, politely redirect them to the correct section."""
                    
                    try:
                        response = llm.invoke(ai_prompt)
                        st.markdown(response.content)
                        st.session_state.messages.append({"role": "assistant", "content": response.content})
                    except Exception as e:
                        error_message = "Sorry, I encountered an error processing your question."
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()
