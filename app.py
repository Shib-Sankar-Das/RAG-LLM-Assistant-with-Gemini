import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.schema import Document
import requests
from bs4 import BeautifulSoup
import PyPDF2
import io
import os
import tempfile
import shutil
from typing import List, Optional, Any
import hashlib
from urllib.parse import urljoin, urlparse
import re
from datetime import datetime
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Custom Gemini LLM wrapper for LangChain
class GeminiLLM(LLM):
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash", **kwargs):
        super().__init__(**kwargs)
        # Store private attributes to avoid Pydantic field validation
        self._api_key = api_key
        self._model_name = model_name
        self._llm = GoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
        )

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs
    ) -> str:
        try:
            # Use invoke method instead of deprecated __call__
            response = self._llm.invoke(prompt, stop=stop, **kwargs)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self._model_name}

    @property
    def _llm_type(self) -> str:
        return "gemini"

class RAGSystem:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.llm = GeminiLLM(api_key)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.vectorstore = None
        self.qa_chain = None
        self.sources = []
        
    def setup_vectorstore(self, persist_directory: Optional[str] = "./chroma_db"):
        """Initialize or load existing ChromaDB"""
        try:
            if persist_directory is None:
                # Temporary in-memory database
                self.vectorstore = Chroma(
                    embedding_function=self.embeddings
                )
                st.success("🚀 Created temporary in-memory database")
            elif os.path.exists(persist_directory):
                # Load existing persistent database
                self.vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
                count = self.vectorstore._collection.count()
                st.success(f"📂 Loaded existing database with {count} documents")
            else:
                # Create new persistent database
                self.vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
                st.success("🆕 Created new persistent database")
                
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_kwargs={"k": 5}
                ),
                return_source_documents=True
            )
            return True
        except Exception as e:
            st.error(f"Error setting up vectorstore: {str(e)}")
            return False
    
    def scrape_website(self, url: str, max_pages: int = 5) -> List[Document]:
        """Scrape website content with advanced features"""
        documents = []
        visited_urls = set()
        urls_to_visit = [url]
        
        with st.spinner(f"Scraping website: {url}"):
            progress_bar = st.progress(0)
            
            for i, current_url in enumerate(urls_to_visit[:max_pages]):
                if current_url in visited_urls:
                    continue
                    
                progress_bar.progress((i + 1) / min(max_pages, len(urls_to_visit)))
                
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    response = requests.get(current_url, headers=headers, timeout=10)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style", "nav", "footer", "header"]):
                        script.decompose()
                    
                    # Extract title
                    title = soup.find('title')
                    title_text = title.get_text().strip() if title else "No Title"
                    
                    # Extract main content
                    content_selectors = [
                        'article', 'main', '.content', '#content', 
                        '.post-content', '.entry-content'
                    ]
                    
                    content = ""
                    for selector in content_selectors:
                        elements = soup.select(selector)
                        if elements:
                            content = ' '.join([elem.get_text() for elem in elements])
                            break
                    
                    if not content:
                        content = soup.get_text()
                    
                    # Clean content
                    content = re.sub(r'\s+', ' ', content).strip()
                    
                    if len(content) > 100:  # Only add if substantial content
                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": current_url,
                                "title": title_text,
                                "type": "website",
                                "scraped_at": datetime.now().isoformat()
                            }
                        )
                        documents.append(doc)
                        visited_urls.add(current_url)
                        
                        # Find more links (for deeper scraping)
                        if i < max_pages - 1:
                            links = soup.find_all('a', href=True)
                            from bs4.element import Tag
                            for link in links[:5]:  # Limit links per page
                                if isinstance(link, Tag):
                                    href = link.get("href")
                                    if isinstance(href, str):
                                        full_url = urljoin(current_url, href)
                                        if (urlparse(full_url).netloc == urlparse(url).netloc and 
                                            full_url not in visited_urls and 
                                            full_url not in urls_to_visit):
                                            urls_to_visit.append(full_url)
                    
                except Exception as e:
                    st.warning(f"Error scraping {current_url}: {str(e)}")
                    continue
        
        return documents
    
    def process_pdf(self, pdf_file) -> List[Document]:
        """Process PDF file and extract text"""
        documents = []
        
        try:
            with st.spinner("Processing PDF..."):
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                full_text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        full_text += f"\n\nPage {page_num + 1}:\n{text}"
                
                if full_text.strip():
                    doc = Document(
                        page_content=full_text,
                        metadata={
                            "source": pdf_file.name,
                            "type": "pdf",
                            "pages": len(pdf_reader.pages),
                            "processed_at": datetime.now().isoformat()
                        }
                    )
                    documents.append(doc)
                    
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            
        return documents
    
    def add_documents_to_vectorstore(self, documents: List[Document]):
        """Add documents to ChromaDB with text splitting"""
        if not documents:
            st.warning("No documents to add")
            return
        
        try:
            if self.vectorstore is None:
                st.error("Vectorstore is not initialized. Please set up the vectorstore before adding documents.")
                return
            with st.spinner("Adding documents to vector database..."):
                # Split documents into chunks
                split_docs = self.text_splitter.split_documents(documents)
                
                # Add to vectorstore
                self.vectorstore.add_documents(split_docs)
                
                # Note: persist() is no longer needed as docs are automatically persisted in Chroma 0.4.x+
                
                # Update sources
                for doc in documents:
                    source_info = {
                        "source": doc.metadata.get("source", "Unknown"),
                        "type": doc.metadata.get("type", "Unknown"),
                        "added_at": datetime.now().isoformat()
                    }
                    if source_info not in self.sources:
                        self.sources.append(source_info)
                
                st.success(f"Added {len(split_docs)} document chunks to the database")
                
        except Exception as e:
            st.error(f"Error adding documents: {str(e)}")
    
    def query(self, question: str) -> dict:
        """Query the RAG system"""
        if not self.qa_chain:
            return {"error": "System not initialized"}
        
        try:
            with st.spinner("Searching for relevant information..."):
                result = self.qa_chain.invoke({"query": question})
                
                return {
                    "answer": result["result"],
                    "source_documents": result["source_documents"]
                }
                
        except Exception as e:
            return {"error": f"Query error: {str(e)}"}
    
    def clear_database(self):
        """Clear the vector database"""
        try:
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")
                self.vectorstore = None
                self.qa_chain = None
                self.sources = []
                st.success("Database cleared successfully")
            else:
                st.info("Database is already empty")
        except Exception as e:
            st.error(f"Error clearing database: {str(e)}")

def main():
    st.set_page_config(
        page_title="RAG LLM Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .status-card {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #4caf50;
    }
    .warning-card {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffc107;
    }
    .error-card {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dc3545;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0px 0px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 RAG LLM Assistant</h1>
        <p>Intelligent Document Processing & Chat with Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # API Key section
        with st.expander("🔑 API Configuration", expanded=True):
            api_key = os.getenv("GEMINI_API_KEY")
            
            if api_key:
                st.success("✅ API key loaded from .env file")
                # Show masked version
                masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
                st.info(f"Using API key: `{masked_key}`")
            else:
                st.error("❌ GEMINI_API_KEY not found in .env file")
                st.info("Please add your API key to the .env file:")
                st.code("GEMINI_API_KEY=your_api_key_here", language="bash")
                st.stop()
        
        # Database Management section
        st.markdown("---")
        st.markdown("### 🗄️ Database Management")
        
        # Database type selection
        db_type = st.radio(
            "Database Type",
            ["Temporary (Session Only)", "Persistent (Saved to Disk)"],
            help="Choose between temporary in-memory storage or persistent disk storage"
        )
        
        # Set persistence directory based on selection
        if db_type == "Temporary (Session Only)":
            persist_dir = None
            st.info("💡 Database will be cleared when session ends")
        else:
            persist_dir = "./chroma_db"
            st.info("💾 Database will be saved to disk")
        
        # Initialize RAG system
        if 'rag_system' not in st.session_state or st.session_state.get('db_type') != db_type:
            st.session_state.rag_system = RAGSystem(api_key)
            st.session_state.db_type = db_type
            
            with st.spinner("Initializing database..."):
                if not st.session_state.rag_system.setup_vectorstore(persist_dir):
                    st.stop()
        
        # Database stats
        st.markdown("---")
        st.markdown("### 📊 Database Statistics")
        
        if st.session_state.rag_system.vectorstore:
            try:
                count = st.session_state.rag_system.vectorstore._collection.count()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Documents", count, delta=None)
                with col2:
                    st.metric("Sources", len(st.session_state.rag_system.sources))
                
                # Database status
                if count > 0:
                    st.markdown("""
                    <div class="status-card">
                        <strong>🟢 Database Ready</strong><br>
                        Ready to answer questions!
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="warning-card">
                        <strong>🟡 Database Empty</strong><br>
                        Add documents to start chatting!
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error reading database: {str(e)}")
        
        # Sources list
        if st.session_state.rag_system.sources:
            st.markdown("---")
            st.markdown("### 📚 Recent Sources")
            for i, source in enumerate(st.session_state.rag_system.sources[-3:]):  # Show last 3
                with st.expander(f"{source['type'].title()} Source {i+1}"):
                    st.write(f"**Source:** {source['source']}")
                    st.write(f"**Added:** {source['added_at'][:19].replace('T', ' ')}")
        
        # Database actions
        st.markdown("---")
        st.markdown("### 🔧 Database Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Database", type="secondary", use_container_width=True):
                st.session_state.rag_system.clear_database()
                st.rerun()
        
        with col2:
            if st.button("🔄 Refresh Stats", type="secondary", use_container_width=True):
                st.rerun()
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🌐 Web Scraping", "📄 PDF Upload", "💬 Chat", "📋 Help"])
    
    with tab1:
        st.markdown("### 🌐 Web Content Scraping")
        st.markdown("""
        <div class="feature-card">
            <strong>Extract content from websites</strong><br>
            Enter a URL and let the system scrape and process the content for you.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            url = st.text_input(
                "Website URL",
                placeholder="https://example.com",
                help="Enter the URL of the website you want to scrape"
            )
        
        with col2:
            max_pages = st.number_input(
                "Max Pages",
                min_value=1,
                max_value=10,
                value=3,
                help="Maximum number of pages to scrape"
            )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                follow_links = st.checkbox("Follow internal links", value=True)
            with col2:
                max_depth = st.number_input("Max depth", min_value=1, max_value=3, value=2)
        
        if st.button("🔍 Scrape Website", type="primary", use_container_width=True):
            if url:
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                
                with st.status("Scraping website...", expanded=True) as status:
                    st.write(f"🔍 Starting scrape of: {url}")
                    documents = st.session_state.rag_system.scrape_website(url, max_pages)
                    
                    if documents:
                        st.write(f"✅ Successfully scraped {len(documents)} pages")
                        status.update(label="Scraping completed!", state="complete")
                        
                        # Preview scraped content
                        with st.expander("📖 Preview Scraped Content", expanded=True):
                            for i, doc in enumerate(documents):
                                st.markdown(f"**Page {i+1}: {doc.metadata['title']}**")
                                st.markdown(f"*URL: {doc.metadata['source']}*")
                                
                                # Show content preview
                                preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                                st.text_area(f"Content Preview {i+1}", preview, height=100, disabled=True)
                                
                                if i < len(documents) - 1:
                                    st.divider()
                        
                        # Add to database
                        st.session_state.rag_system.add_documents_to_vectorstore(documents)
                        st.success("🎉 Documents added to database successfully!")
                        st.rerun()
                    else:
                        st.error("❌ No content could be scraped from the website")
                        status.update(label="Scraping failed!", state="error")
            else:
                st.error("⚠️ Please enter a valid URL")
    
    with tab2:
        st.markdown("### 📄 PDF Document Processing")
        st.markdown("""
        <div class="feature-card">
            <strong>Upload and process PDF documents</strong><br>
            Extract text from your PDF files and make them searchable.
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to add to the knowledge base"
        )
        
        if uploaded_files:
            st.markdown("### 📋 Selected Files")
            for i, file in enumerate(uploaded_files):
                st.markdown(f"**{i+1}.** {file.name} ({file.size:,} bytes)")
            
            col1, col2 = st.columns(2)
            with col1:
                process_all = st.button("📚 Process All PDFs", type="primary", use_container_width=True)
            with col2:
                clear_selection = st.button("🗑️ Clear Selection", type="secondary", use_container_width=True)
            
            if clear_selection:
                st.rerun()
            
            if process_all:
                with st.status("Processing PDFs...", expanded=True) as status:
                    all_documents = []
                    
                    for i, pdf_file in enumerate(uploaded_files):
                        st.write(f"📄 Processing: {pdf_file.name}")
                        documents = st.session_state.rag_system.process_pdf(pdf_file)
                        all_documents.extend(documents)
                        
                        # Update progress
                        progress = (i + 1) / len(uploaded_files)
                        st.progress(progress, f"Processed {i+1}/{len(uploaded_files)} files")
                    
                    if all_documents:
                        st.write(f"✅ Successfully processed {len(all_documents)} PDF files")
                        status.update(label="Processing completed!", state="complete")
                        
                        # Preview PDF content
                        with st.expander("📖 Preview PDF Content", expanded=True):
                            for i, doc in enumerate(all_documents):
                                st.markdown(f"**PDF: {doc.metadata['source']}**")
                                st.markdown(f"*Pages: {doc.metadata['pages']} | Type: {doc.metadata['type']}*")
                                
                                # Show content preview
                                preview = doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content
                                st.text_area(f"Content Preview {i+1}", preview, height=120, disabled=True)
                                
                                if i < len(all_documents) - 1:
                                    st.divider()
                        
                        # Add to database
                        st.session_state.rag_system.add_documents_to_vectorstore(all_documents)
                        st.success("🎉 Documents added to database successfully!")
                        st.rerun()
                    else:
                        st.error("❌ No content could be extracted from the PDF files")
                        status.update(label="Processing failed!", state="error")
        else:
            st.info("👆 Upload PDF files to get started")
    
    with tab3:
        st.markdown("### 💬 Intelligent Chat")
        st.markdown("""
        <div class="feature-card">
            <strong>Chat with your documents</strong><br>
            Ask questions about your uploaded content and get AI-powered answers with source citations.
        </div>
        """, unsafe_allow_html=True)
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Quick actions
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("💡 Example Questions", use_container_width=True):
                st.session_state.show_examples = not st.session_state.get('show_examples', False)
        with col2:
            if st.button("� Chat Stats", use_container_width=True):
                st.session_state.show_stats = not st.session_state.get('show_stats', False)
        with col3:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        
        # Show example questions
        if st.session_state.get('show_examples', False):
            with st.expander("💡 Example Questions", expanded=True):
                st.markdown("""
                **Try asking questions like:**
                - "What is the main topic of the documents?"
                - "Summarize the key points from the content"
                - "What are the most important findings?"
                - "Can you explain [specific concept] from the documents?"
                """)
        
        # Show chat stats
        if st.session_state.get('show_stats', False):
            with st.expander("📊 Chat Statistics", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Messages", len(st.session_state.messages))
                with col2:
                    user_msgs = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
                    st.metric("User Messages", user_msgs)
                with col3:
                    ai_msgs = len([msg for msg in st.session_state.messages if msg["role"] == "assistant"])
                    st.metric("AI Responses", ai_msgs)
        
        # Chat container
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for i, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    if message["role"] == "assistant" and "sources" in message:
                        if message["sources"]:
                            with st.expander("📚 Sources & References"):
                                for j, source in enumerate(message["sources"]):
                                    st.markdown(f"**{j+1}.** {source}")
                        
                        # Feedback buttons
                        col1, col2, col3 = st.columns([1, 1, 4])
                        with col1:
                            if st.button("👍", key=f"like_{i}", help="Helpful response"):
                                st.success("Thanks for your feedback!")
                        with col2:
                            if st.button("👎", key=f"dislike_{i}", help="Not helpful"):
                                st.info("Thanks for your feedback!")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Check if database has content
            if st.session_state.rag_system.vectorstore is None:
                st.error("⚠️ Please add some data first by scraping a website or uploading PDFs")
            else:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Get response
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Thinking..."):
                        result = st.session_state.rag_system.query(prompt)
                        
                        if "error" in result:
                            st.error(f"❌ {result['error']}")
                        else:
                            answer = result["answer"]
                            sources = []
                            
                            if "source_documents" in result:
                                sources = list(set([doc.metadata.get("source", "Unknown") 
                                                  for doc in result["source_documents"]]))
                            
                            st.write(answer)
                            
                            if sources:
                                with st.expander("📚 Sources & References"):
                                    for j, source in enumerate(sources):
                                        st.markdown(f"**{j+1}.** {source}")
                            
                            # Add to chat history
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "sources": sources
                            })
                            
                            st.rerun()
    
    with tab4:
        st.markdown("### 📋 Help & Documentation")
        
        # Feature overview
        st.markdown("""
        <div class="feature-card">
            <h4>🚀 Welcome to RAG LLM Assistant</h4>
            <p>This intelligent document processing system helps you extract, organize, and chat with your data using advanced AI capabilities.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # How it works
        with st.expander("🔍 How It Works", expanded=True):
            st.markdown("""
            **1. Data Ingestion**
            - Upload PDF documents or scrape websites
            - Content is automatically extracted and processed
            
            **2. Intelligent Indexing**
            - Documents are split into chunks for better retrieval
            - Semantic embeddings are created using HuggingFace transformers
            - Content is stored in a ChromaDB vector database
            
            **3. Smart Retrieval**
            - Your questions are converted to semantic queries
            - Relevant document chunks are retrieved
            - Context is provided to the AI for accurate answers
            
            **4. AI-Powered Responses**
            - Google Gemini AI generates responses based on your documents
            - Sources are cited for transparency
            - Answers are grounded in your specific content
            """)
        
        # Database options
        with st.expander("💾 Database Options"):
            st.markdown("""
            **Temporary Database (Session Only)**
            - Data is stored in memory during your session
            - Automatically cleared when you close the browser
            - Best for: Testing, sensitive data, one-time analysis
            
            **Persistent Database (Saved to Disk)**
            - Data is saved to your local disk
            - Persists between sessions
            - Best for: Long-term projects, repeated access
            """)
        
        # Tips and best practices
        with st.expander("💡 Tips & Best Practices"):
            st.markdown("""
            **Web Scraping Tips:**
            - Start with fewer pages to test the content quality
            - Some websites may block scraping attempts
            - Clean, well-structured sites work best
            
            **PDF Processing Tips:**
            - Text-based PDFs work better than scanned images
            - Remove password protection before uploading
            - Smaller files process faster
            
            **Chat Tips:**
            - Be specific in your questions
            - Ask about concepts, summaries, or specific facts
            - Use follow-up questions to dig deeper
            - Check the sources for context
            """)
        
        # Technical details
        with st.expander("🔧 Technical Details"):
            st.markdown("""
            **Technologies Used:**
            - **Frontend:** Streamlit for the web interface
            - **AI Model:** Google Gemini for natural language processing
            - **Embeddings:** HuggingFace sentence-transformers
            - **Vector Database:** ChromaDB for efficient similarity search
            - **Text Processing:** LangChain for document handling
            
            **System Requirements:**
            - Python 3.8+
            - Internet connection for AI model access
            - ~2GB RAM for typical usage
            """)
        
        # Support
        with st.expander("❓ Support & Troubleshooting"):
            st.markdown("""
            **Common Issues:**
            
            **"API Key not found"**
            - Check your .env file contains: `GEMINI_API_KEY=your_key_here`
            - Restart the application after adding the key
            
            **"No content scraped"**
            - Try a different website
            - Check if the site allows scraping
            - Reduce the number of pages
            
            **"Slow processing"**
            - Large files take more time
            - Try processing smaller batches
            - Check your internet connection
            
            **"Empty responses"**
            - Make sure documents are added to the database
            - Try rephrasing your question
            - Check if the question relates to your documents
            """)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 20px;">
            <p>🤖 RAG LLM Assistant - Powered by Google Gemini AI</p>
            <p>Built with Streamlit, LangChain, and ChromaDB</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()