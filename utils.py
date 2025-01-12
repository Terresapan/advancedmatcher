import gspread
import json
from oauth2client.service_account import ServiceAccountCredentials
import streamlit as st
import hmac
import re
import PyPDF2
try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    st.warning("⚠️ python-docx library not installed. Word document support will be limited.")

# Function to save feedback to a file
def save_feedback(feedback_text):
    # Load the credentials from the secrets
    credentials_data = st.secrets["gcp"]["service_account_json"]
    # print(credentials_data)
    creds = json.loads(credentials_data, strict=False)

    # Set up the Google Sheets API credentials
    scope = ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/spreadsheets"]
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(creds, scope)
    client = gspread.authorize(credentials)

    # Open the Google Sheet
    sheet_id = '1qnFzZZ7YI-9pXj3iAXafjRmC_EIQyK9gA98AjMv29DM'
    sheet = client.open_by_key(sheet_id).worksheet("feedback")
    sheet.append_row([feedback_text])
       
    
# Password checking function
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

# Text processing functions
def clean_text(text):
    """Clean text by removing extra whitespace."""
    return re.sub(r'\s+', ' ', text).strip()

# Load consultant data from Google Sheets
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