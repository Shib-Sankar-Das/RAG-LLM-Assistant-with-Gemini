"""
Help tab component.
"""

import streamlit as st
from ..utils.ui_helpers import render_feature_card

class HelpTab:
    """Help and documentation tab component."""
    
    def render(self):
        """Render the help tab."""
        st.markdown("### 📋 Help & Documentation")
        
        # Feature overview
        render_feature_card(
            "🚀 Welcome to RAG LLM Assistant",
            "This intelligent document processing system helps you extract, organize, and chat with your data using advanced AI capabilities."
        )
        
        # How it works
        self._render_how_it_works()
        
        # Database options
        self._render_database_options()
        
        # Tips and best practices
        self._render_tips_and_practices()
        
        # Technical details
        self._render_technical_details()
        
        # Support
        self._render_support()
        
        # Footer
        self._render_footer()
    
    def _render_how_it_works(self):
        """Render how it works section."""
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
    
    def _render_database_options(self):
        """Render database options section."""
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
    
    def _render_tips_and_practices(self):
        """Render tips and best practices section."""
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
    
    def _render_technical_details(self):
        """Render technical details section."""
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
    
    def _render_support(self):
        """Render support and troubleshooting section."""
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
    
    def _render_footer(self):
        """Render footer section."""
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 20px;">
            <p>🤖 RAG LLM Assistant - Powered by Google Gemini AI</p>
            <p>Built with Streamlit, LangChain, and ChromaDB</p>
        </div>
        """, unsafe_allow_html=True)
