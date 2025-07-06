"""
Main Streamlit application for the RAG LLM Assistant.

A modular, well-organized implementation of a Retrieval-Augmented Generation
system using Streamlit, LangChain, and Google Gemini AI.
"""

import streamlit as st
from src.config import Config
from src.components import Sidebar, WebScrapingTab, PDFUploadTab, ChatTab, HelpTab
from src.utils import apply_custom_css, render_main_header

def main():
    """Main application function."""
    # Configure Streamlit page
    st.set_page_config(
        page_title=Config.APP_TITLE,
        page_icon=Config.APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Validate configuration
    try:
        Config.validate_config()
    except ValueError as e:
        st.error(f"Configuration Error: {e}")
        st.stop()
    
    # Apply custom styling
    apply_custom_css()
    
    # Render main header
    render_main_header()
    
    # Initialize components
    sidebar = Sidebar()
    web_scraping_tab = WebScrapingTab()
    pdf_upload_tab = PDFUploadTab()
    chat_tab = ChatTab()
    help_tab = HelpTab()
    
    # Render sidebar and get system components
    api_key, rag_system, db_type = sidebar.render()
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🌐 Web Scraping", "📄 PDF Upload", "💬 Chat", "📋 Help"])
    
    with tab1:
        web_scraping_tab.render(rag_system)
    
    with tab2:
        pdf_upload_tab.render(rag_system)
    
    with tab3:
        chat_tab.render(rag_system)
    
    with tab4:
        help_tab.render()

if __name__ == "__main__":
    main()
