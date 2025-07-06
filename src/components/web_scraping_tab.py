"""
Web scraping tab component.
"""

import streamlit as st
from ..core.rag_system import RAGSystem
from ..utils.ui_helpers import render_feature_card, validate_url

class WebScrapingTab:
    """Web scraping tab component."""
    
    def render(self, rag_system: RAGSystem):
        """Render the web scraping tab.
        
        Args:
            rag_system: The RAG system instance
        """
        st.markdown("### 🌐 Web Content Scraping")
        render_feature_card(
            "🌐 Extract content from websites",
            "Enter a URL and let the system scrape and process the content for you."
        )
        
        # URL input and settings
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
        
        # Scrape button
        if st.button("🔍 Scrape Website", type="primary", use_container_width=True):
            self._handle_scraping(rag_system, url, max_pages)
    
    def _handle_scraping(self, rag_system: RAGSystem, url: str, max_pages: int):
        """Handle the web scraping process.
        
        Args:
            rag_system: The RAG system instance
            url: URL to scrape
            max_pages: Maximum number of pages to scrape
        """
        if not url:
            st.error("⚠️ Please enter a valid URL")
            return
        
        # Validate and format URL
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        if not validate_url(url):
            st.error("⚠️ Please enter a valid URL")
            return
        
        # Start scraping
        with st.status("Scraping website...", expanded=True) as status:
            st.write(f"🔍 Starting scrape of: {url}")
            documents = rag_system.scrape_website(url, max_pages)
            
            if documents:
                st.write(f"✅ Successfully scraped {len(documents)} pages")
                status.update(label="Scraping completed!", state="complete")
                
                # Preview scraped content
                self._show_scraped_content_preview(documents)
                
                # Add to database
                rag_system.add_documents_to_vectorstore(documents)
                st.success("🎉 Documents added to database successfully!")
                st.rerun()
            else:
                st.error("❌ No content could be scraped from the website")
                status.update(label="Scraping failed!", state="error")
    
    def _show_scraped_content_preview(self, documents):
        """Show preview of scraped content.
        
        Args:
            documents: List of scraped documents
        """
        with st.expander("📖 Preview Scraped Content", expanded=True):
            for i, doc in enumerate(documents):
                st.markdown(f"**Page {i+1}: {doc.metadata['title']}**")
                st.markdown(f"*URL: {doc.metadata['source']}*")
                
                # Show content preview
                preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                st.text_area(f"Content Preview {i+1}", preview, height=100, disabled=True)
                
                if i < len(documents) - 1:
                    st.divider()
