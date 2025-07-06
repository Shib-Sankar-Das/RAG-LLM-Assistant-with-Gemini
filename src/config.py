"""
Configuration settings for the RAG LLM Assistant.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the RAG application."""
    
    # API Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")
    
    # Database Configuration
    DEFAULT_PERSIST_DIR = "./chroma_db"
    
    # Text Processing Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Embeddings Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
    
    # Retrieval Configuration
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
    
    # Web Scraping Configuration
    MAX_PAGES_DEFAULT = int(os.getenv("MAX_PAGES_DEFAULT", "3"))
    MAX_PAGES_LIMIT = int(os.getenv("MAX_PAGES_LIMIT", "10"))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "10"))
    
    # UI Configuration
    APP_TITLE = "RAG LLM Assistant"
    APP_ICON = "🤖"
    
    @classmethod
    def validate_config(cls):
        """Validate required configuration settings."""
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required. Please set it in your .env file.")
        
        return True
