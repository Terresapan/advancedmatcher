# Interactive SmartMatch Staffing Platform

An AI-powered consultant matching application that intelligently connects projects with the right consultants based on expertise, industry knowledge, and availability.

## 🚀 Features

### 🎯 Project-Consultant Matching

- **Document Upload**: Upload project documents (PDF, DOCX, TXT) for automated analysis
- **AI Analysis**: Extract project requirements and consultant criteria using Google Gemini AI
- **Smart Matching**: Find consultants based on expertise areas, industry, and availability
- **Human-in-the-Loop**: Review and approve AI-generated criteria before matching

### 💬 Interactive Chat Interface

- **Database Queries**: Chat with the consultant database using natural language
- **Semantic Search**: Find consultants using vector embeddings and similarity matching
- **Contextual Responses**: Get detailed information about consultant profiles and capabilities

### 📊 Consultant Database

- **Expertise Areas**: Finance, Marketing, Operations, Strategy, Entrepreneurship
- **Rich Profiles**: Education, industry expertise, bio, and availability information
- **Real-time Sync**: Integration with Google Sheets for data management

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI/ML**: LangChain, LangGraph, Google Gemini 2.5 Flash
- **Database**: Supabase with pgvector for embeddings
- **Data Integration**: Google Sheets API
- **Monitoring**: LangSmith for AI tracing
- **Authentication**: Streamlit secrets management

## 📋 Prerequisites

- Python 3.8+
- Google API key for Gemini AI
- Supabase account and database
- Google Sheets API credentials
- LangSmith account (optional, for tracing)

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd smartmatch-staffing-platform
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
   Create a `.streamlit/secrets.toml` file with the following structure:

```toml
[llmapikey]
GOOGLE_API_KEY = "your_google_api_key"

[database]
SUPABASE_URL = "your_supabase_url"
SUPABASE_KEY = "your_supabase_key"

[tracing]
LANGCHAIN_API_KEY = "your_langsmith_api_key"

[gcp]
service_account_json = '''
{
  "type": "service_account",
  "project_id": "your_project_id",
  // ... your Google service account JSON
}
'''

password = "your_app_password"
```

4. **Set up Supabase Database**
   Create a `consultants` table with the following schema:

```sql
CREATE TABLE consultants (
  id SERIAL PRIMARY KEY,
  name TEXT UNIQUE,
  age INTEGER,
  finance_expertise BOOLEAN,
  strategy_expertise BOOLEAN,
  operations_expertise BOOLEAN,
  marketing_expertise BOOLEAN,
  entrepreneurship_expertise BOOLEAN,
  education TEXT,
  industry_expertise TEXT,
  bio TEXT,
  anticipated_availability_date DATE,
  availability BOOLEAN,
  embedding VECTOR(768),
  updated_at TIMESTAMP DEFAULT NOW()
);
```

5. **Configure Google Sheets Connection**
   Set up the Google Sheets connection in `.streamlit/secrets.toml`:

```toml
[connections.gsheets]
spreadsheet = "your_google_sheet_url"
type = "service_account"
project_id = "your_project_id"
private_key_id = "your_private_key_id"
private_key = "your_private_key"
client_email = "your_client_email"
client_id = "your_client_id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
```

## 🚀 Usage

1. **Start the application**

```bash
streamlit run streamlit_app.py
```

2. **Access the platform**

- Open your browser to `http://localhost:8501`
- Enter the application password
- Choose your input method:
  - **File Upload**: Upload project documents for analysis
  - **Text Query**: Chat with the consultant database

### File Upload Workflow

1. Upload a project document (PDF, DOCX, or TXT)
2. Click "Generate Summary and Approve the Criteria"
3. Review and modify the extracted project requirements
4. Click "Approve and Find Consultants" to see matches
5. Review consultant profiles and match analysis

### Chat Interface

1. Select "Text Query" tab
2. Ask questions about consultants using natural language
3. Examples:
   - "Find consultants with finance expertise in healthcare"
   - "Who has strategy and operations experience?"
   - "Show me available consultants with marketing skills"

## 📁 Project Structure

```
├── streamlit_app.py      # Main Streamlit application
├── main.py              # Core workflow and LangGraph implementation
├── chat.py              # Chat interface and query processing
├── database.py          # Supabase integration and data sync
├── states.py            # Pydantic models for state management
├── prompt.py            # AI prompts and templates
├── utils.py             # Utility functions and file processing
├── requirements.txt     # Python dependencies
└── assets/             # Static assets (images, etc.)
```

## 🔧 Configuration

### Supabase Functions

The application uses custom Supabase RPC functions for advanced search:

```sql
-- Function for criteria-based search
CREATE OR REPLACE FUNCTION search_consultants(...)
RETURNS TABLE(...) AS $$
-- Implementation for filtering by expertise, industry, availability
$$ LANGUAGE plpgsql;

-- Function for vector similarity search
CREATE OR REPLACE FUNCTION search_consultants_vector(...)
RETURNS TABLE(...) AS $$
-- Implementation for semantic search using embeddings
$$ LANGUAGE plpgsql;
```

### Google Sheets Integration

- Consultant data is loaded from Google Sheets
- Automatic column mapping and data type conversion
- Cached for 10 minutes to optimize performance

## 🎨 Features in Detail

### AI-Powered Analysis

- **Project Summary Generation**: Extracts key information from documents
- **Structured Requirement Extraction**: Identifies needed expertise areas
- **Match Analysis**: Provides detailed consultant-project fit assessment
- **Response Generation**: Creates comprehensive matching summaries

### Search Capabilities

- **Multi-criteria Filtering**: Expertise, industry, availability
- **Vector Similarity Search**: Semantic matching using embeddings
- **Fallback Mechanisms**: Graceful degradation when exact matches aren't found
- **Contextual Results**: Explains search results and match quality

## 🔒 Security

- Password-protected access
- Secure API key management via Streamlit secrets
- Input validation and sanitization
- Error handling for external API calls

## 📊 Monitoring

- LangSmith integration for AI operation tracing
- Performance monitoring for database queries
- Error logging and user feedback collection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

- Check the [Issues](../../issues) page
- Review the documentation
- Contact the development team

## 🎧 Additional Resources

- [Podcast Episode](https://open.spotify.com/episode/1HA0LDPBgbQVzCkJilvKVe) - Learn more about the platform
- [AI Agent Projects](https://ai-agents-garden.lovable.app/) - Explore related projects

---

Built with ❤️ using Streamlit, LangChain, and Google AI
