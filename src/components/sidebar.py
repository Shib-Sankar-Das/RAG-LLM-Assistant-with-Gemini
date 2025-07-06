"""
Sidebar component for the RAG application.
"""

import streamlit as st
from ..core.rag_system import RAGSystem
from ..config import Config
from ..utils.ui_helpers import format_datetime, render_status_card

class Sidebar:
    """Sidebar component for configuration and database management."""
    
    def __init__(self):
        self.config = Config()
    
    def render(self) -> tuple:
        """Render the sidebar and return (api_key, rag_system, db_type).
        
        Returns:
            Tuple containing API key, RAG system instance, and database type
        """
        with st.sidebar:
            st.markdown("### ⚙️ Configuration")
            
            # API Key section
            api_key = self._render_api_section()
            
            # Database Management section
            db_type, persist_dir = self._render_database_section()
            
            # Initialize RAG system
            rag_system = self._initialize_rag_system(api_key, db_type, persist_dir)
            
            # Database stats
            self._render_database_stats(rag_system)
            
            # Sources list
            self._render_sources_list(rag_system)
            
            # Database actions
            self._render_database_actions(rag_system)
            
            return api_key, rag_system, db_type
    
    def _render_api_section(self) -> str:
        """Get API key without rendering UI section."""
        api_key = self.config.GEMINI_API_KEY
        
        if not api_key:
            st.error("❌ GEMINI_API_KEY not found in .env file")
            st.info("Please add your API key to the .env file:")
            st.code("GEMINI_API_KEY=your_api_key_here", language="bash")
            st.stop()
        
        return api_key
    
    def _render_database_section(self) -> tuple:
        """Render database management section."""
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
            persist_dir = self.config.DEFAULT_PERSIST_DIR
            st.info("💾 Database will be saved to disk")
        
        return db_type, persist_dir
    
    def _initialize_rag_system(self, api_key: str, db_type: str, persist_dir: str) -> RAGSystem:
        """Initialize or get RAG system instance."""
        # Initialize RAG system
        if 'rag_system' not in st.session_state or st.session_state.get('db_type') != db_type:
            st.session_state.rag_system = RAGSystem(api_key)
            st.session_state.db_type = db_type
            
            with st.spinner("Initializing database..."):
                if not st.session_state.rag_system.setup_vectorstore(persist_dir):
                    st.stop()
        
        return st.session_state.rag_system
    
    def _render_database_stats(self, rag_system: RAGSystem):
        """Render database statistics section."""
        st.markdown("---")
        st.markdown("### 📊 Database Statistics")
        
        if rag_system.vectorstore:
            try:
                stats = rag_system.get_database_stats()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Documents", stats["document_count"])
                with col2:
                    st.metric("Sources", stats["source_count"])
                
                # Database status
                if stats["document_count"] > 0:
                    render_status_card(
                        "🟢 Database Ready", 
                        "Ready to answer questions!", 
                        "status"
                    )
                else:
                    render_status_card(
                        "🟡 Database Empty", 
                        "Add documents to start chatting!", 
                        "warning"
                    )
                    
            except Exception as e:
                st.error(f"Error reading database: {str(e)}")
    
    def _render_sources_list(self, rag_system: RAGSystem):
        """Render recent sources list."""
        if rag_system.sources:
            st.markdown("---")
            st.markdown("### 📚 Recent Sources")
            
            # Show last 3 sources
            for i, source in enumerate(rag_system.sources[-3:]):
                with st.expander(f"{source['type'].title()} Source {i+1}"):
                    st.write(f"**Source:** {source['source']}")
                    st.write(f"**Added:** {format_datetime(source['added_at'])}")
    
    def _render_database_actions(self, rag_system: RAGSystem):
        """Render database action buttons."""
        st.markdown("---")
        st.markdown("### 🔧 Database Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Database", type="secondary", use_container_width=True):
                rag_system.clear_database()
                st.rerun()
        
        with col2:
            if st.button("🔄 Refresh Stats", type="secondary", use_container_width=True):
                st.rerun()
