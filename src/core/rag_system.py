"""
Main RAG system implementation.
"""

import os
import shutil
from datetime import datetime
from typing import List, Optional
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document

from .llm import GeminiLLM
from .document_processor import WebScraper, PDFProcessor
from ..config import Config

class RAGSystem:
    """Main RAG system class."""
    
    def __init__(self, api_key: str):
        """Initialize the RAG system.
        
        Args:
            api_key: Google Gemini API key
        """
        self.config = Config()
        self.api_key = api_key
        self.llm = GeminiLLM(api_key, self.config.MODEL_NAME)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.EMBEDDING_MODEL,
            model_kwargs={'device': self.config.EMBEDDING_DEVICE}
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len
        )
        
        # Initialize processors
        self.web_scraper = WebScraper()
        self.pdf_processor = PDFProcessor()
        
        # Initialize storage
        self.vectorstore = None
        self.qa_chain = None
        self.sources = []
        
    def setup_vectorstore(self, persist_directory: Optional[str] = None):
        """Initialize or load existing ChromaDB.
        
        Args:
            persist_directory: Directory to persist the database, None for temporary
            
        Returns:
            True if successful, False otherwise
        """
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
                
            # Setup QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_kwargs={"k": self.config.RETRIEVAL_K}
                ),
                return_source_documents=True
            )
            return True
            
        except Exception as e:
            st.error(f"Error setting up vectorstore: {str(e)}")
            return False
    
    def scrape_website(self, url: str, max_pages: Optional[int] = None) -> List[Document]:
        """Scrape website content.
        
        Args:
            url: Website URL to scrape
            max_pages: Maximum number of pages to scrape
            
        Returns:
            List of Document objects
        """
        if max_pages is None:
            max_pages = self.config.MAX_PAGES_DEFAULT
            
        return self.web_scraper.scrape_website(url, max_pages)
    
    def process_pdf(self, pdf_file) -> List[Document]:
        """Process PDF file.
        
        Args:
            pdf_file: Streamlit uploaded file object
            
        Returns:
            List of Document objects
        """
        return self.pdf_processor.process_pdf(pdf_file)
    
    def add_documents_to_vectorstore(self, documents: List[Document]):
        """Add documents to ChromaDB with text splitting.
        
        Args:
            documents: List of Document objects to add
        """
        if not documents:
            st.warning("No documents to add")
            return
        
        try:
            if self.vectorstore is None:
                st.error("Vectorstore is not initialized. Please set up the vectorstore first.")
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
        """Query the RAG system.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and source documents
        """
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
        """Clear the vector database."""
        try:
            if os.path.exists(self.config.DEFAULT_PERSIST_DIR):
                shutil.rmtree(self.config.DEFAULT_PERSIST_DIR)
                
            self.vectorstore = None
            self.qa_chain = None
            self.sources = []
            st.success("Database cleared successfully")
            
        except Exception as e:
            st.error(f"Error clearing database: {str(e)}")
    
    def get_database_stats(self) -> dict:
        """Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        if not self.vectorstore:
            return {"document_count": 0, "source_count": 0}
        
        try:
            document_count = self.vectorstore._collection.count()
            source_count = len(self.sources)
            
            return {
                "document_count": document_count,
                "source_count": source_count,
                "sources": self.sources
            }
        except Exception:
            return {"document_count": 0, "source_count": 0}
