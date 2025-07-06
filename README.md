# RAG LLM Assistant

A Retrieval-Augmented Generation (RAG) system built with Streamlit, LangChain, and Google Gemini AI. This application allows you to scrape websites, upload PDFs, and chat with your data using advanced AI capabilities.

## Features

- 🌐 **Web Scraping**: Extract content from websites
- 📄 **PDF Processing**: Upload and process PDF documents
- 💬 **Interactive Chat**: Ask questions about your data
- 🧠 **Vector Database**: ChromaDB for efficient document retrieval
- 🔍 **Smart Search**: Find relevant information from your documents

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd RAG
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
# or
source .venv/bin/activate  # On Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up Environment Variables
1. Copy the example environment file:
   ```bash
   copy .env.example .env  # On Windows
   # or
   cp .env.example .env    # On Mac/Linux
   ```

2. Edit the `.env` file and add your Google Gemini API key:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

### 5. Get Your Gemini API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key and paste it into your `.env` file

### 6. Run the Application
```bash
# Run the modular version (recommended)
streamlit run app_modular.py

# Or run the original version
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure

The project is now organized in a modular structure for better maintainability:

```
RAG/
├── app_modular.py              # Main modular application (recommended)
├── app.py                      # Original monolithic version
├── requirements.txt            # Dependencies
├── .env                       # Environment variables
├── PROJECT_STRUCTURE.md       # Detailed structure documentation
└── src/                       # Source code
    ├── config.py              # Configuration management
    ├── core/                  # Core business logic
    │   ├── rag_system.py     # Main RAG system
    │   ├── llm.py            # LLM wrapper
    │   └── document_processor.py # Document processing
    ├── components/            # UI components
    │   ├── sidebar.py        # Sidebar interface
    │   ├── web_scraping_tab.py
    │   ├── pdf_upload_tab.py
    │   ├── chat_tab.py
    │   └── help_tab.py
    └── utils/                 # Utility functions
        └── ui_helpers.py     # UI helpers
```

For detailed information about the project structure, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

## Usage

### Web Scraping
1. Go to the "🌐 Web Scraping" tab
2. Enter a website URL
3. Set the maximum number of pages to scrape
4. Click "🔍 Scrape Website"

### PDF Upload
1. Go to the "📄 PDF Upload" tab
2. Upload one or more PDF files
3. Click "📚 Process PDFs"

### Chat with Your Data
1. Go to the "💬 Chat" tab
2. Ask questions about your uploaded content
3. View sources for each answer

## Project Structure

```
RAG/
├── app_modular.py              # Main modular application (recommended)
├── app.py                      # Original monolithic version  
├── requirements.txt            # Python dependencies
├── .env                       # Environment variables (not in git)
├── .env.example              # Template for environment variables
├── .gitignore                # Files to ignore in git
├── README.md                 # This file
├── PROJECT_STRUCTURE.md      # Detailed structure documentation
├── chroma_db/                # Vector database (created automatically)
└── src/                      # Modular source code
    ├── config.py             # Configuration management
    ├── core/                 # Core business logic
    ├── components/           # UI components  
    └── utils/                # Utility functions
```

## Security Notes

- Never commit your `.env` file to version control
- Keep your API key secure and don't share it
- The `.env` file is already included in `.gitignore`

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your `GEMINI_API_KEY` is correctly set in the `.env` file
2. **Import Errors**: Run `pip install -r requirements.txt` to install all dependencies
3. **Port Already in Use**: Streamlit will automatically use the next available port

### Dependencies

- streamlit: Web application framework
- langchain: LLM application framework
- langchain-google-genai: Google Gemini integration
- langchain-huggingface: HuggingFace embeddings integration
- langchain-chroma: ChromaDB vector database integration
- chromadb: Vector database
- sentence-transformers: Text embeddings
- beautifulsoup4: Web scraping
- PyPDF2: PDF processing
- python-dotenv: Environment variable management

## License

This project is open source and available under the MIT License.
