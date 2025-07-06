"""
Document processing utilities for web scraping and PDF processing.
"""

import re
import requests
from datetime import datetime
from typing import List, Optional
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from bs4.element import Tag
import PyPDF2
import streamlit as st
from langchain.schema import Document

from ..config import Config

class WebScraper:
    """Web scraping utility class."""
    
    def __init__(self):
        self.config = Config()
    
    def scrape_website(self, url: str, max_pages: Optional[int] = None) -> List[Document]:
        """Scrape website content with advanced features.
        
        Args:
            url: Starting URL to scrape
            max_pages: Maximum number of pages to scrape
            
        Returns:
            List of Document objects containing scraped content
        """
        if max_pages is None:
            max_pages = self.config.MAX_PAGES_DEFAULT
            
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
                    response = requests.get(
                        current_url, 
                        headers=headers, 
                        timeout=self.config.REQUEST_TIMEOUT
                    )
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style", "nav", "footer", "header"]):
                        script.decompose()
                    
                    # Extract title
                    title = soup.find('title')
                    title_text = title.get_text().strip() if title else "No Title"
                    
                    # Extract main content
                    content = self._extract_content(soup)
                    
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
                            self._find_additional_links(
                                soup, current_url, url, visited_urls, urls_to_visit
                            )
                    
                except Exception as e:
                    st.warning(f"Error scraping {current_url}: {str(e)}")
                    continue
        
        return documents
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from BeautifulSoup object."""
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
        
        return content
    
    def _find_additional_links(self, soup: BeautifulSoup, current_url: str, 
                              base_url: str, visited_urls: set, urls_to_visit: list):
        """Find additional links for deeper scraping."""
        links = soup.find_all('a', href=True)
        
        for link in links[:5]:  # Limit links per page
            if isinstance(link, Tag):
                href = link.get("href")
                if isinstance(href, str):
                    full_url = urljoin(current_url, href)
                    if (urlparse(full_url).netloc == urlparse(base_url).netloc and 
                        full_url not in visited_urls and 
                        full_url not in urls_to_visit):
                        urls_to_visit.append(full_url)


class PDFProcessor:
    """PDF processing utility class."""
    
    def process_pdf(self, pdf_file) -> List[Document]:
        """Process PDF file and extract text.
        
        Args:
            pdf_file: Streamlit uploaded file object
            
        Returns:
            List of Document objects containing PDF content
        """
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
