"""
PDF upload tab component.
"""

import streamlit as st
from ..core.rag_system import RAGSystem
from ..utils.ui_helpers import render_feature_card, format_file_size

class PDFUploadTab:
    """PDF upload tab component."""
    
    def render(self, rag_system: RAGSystem):
        """Render the PDF upload tab.
        
        Args:
            rag_system: The RAG system instance
        """
        st.markdown("### 📄 PDF Document Processing")
        render_feature_card(
            "📄 Upload and process PDF documents",
            "Extract text from your PDF files and make them searchable."
        )
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to add to the knowledge base"
        )
        
        if uploaded_files:
            self._show_file_list(uploaded_files)
            self._handle_file_processing(rag_system, uploaded_files)
        else:
            st.info("👆 Upload PDF files to get started")
    
    def _show_file_list(self, uploaded_files):
        """Show list of uploaded files.
        
        Args:
            uploaded_files: List of uploaded files
        """
        st.markdown("### 📋 Selected Files")
        for i, file in enumerate(uploaded_files):
            file_size = format_file_size(file.size)
            st.markdown(f"**{i+1}.** {file.name} ({file_size})")
    
    def _handle_file_processing(self, rag_system: RAGSystem, uploaded_files):
        """Handle PDF file processing.
        
        Args:
            rag_system: The RAG system instance
            uploaded_files: List of uploaded files
        """
        col1, col2 = st.columns(2)
        with col1:
            process_all = st.button("📚 Process All PDFs", type="primary", use_container_width=True)
        with col2:
            clear_selection = st.button("🗑️ Clear Selection", type="secondary", use_container_width=True)
        
        if clear_selection:
            st.rerun()
        
        if process_all:
            self._process_pdf_files(rag_system, uploaded_files)
    
    def _process_pdf_files(self, rag_system: RAGSystem, uploaded_files):
        """Process all uploaded PDF files.
        
        Args:
            rag_system: The RAG system instance
            uploaded_files: List of uploaded files
        """
        with st.status("Processing PDFs...", expanded=True) as status:
            all_documents = []
            
            for i, pdf_file in enumerate(uploaded_files):
                st.write(f"📄 Processing: {pdf_file.name}")
                documents = rag_system.process_pdf(pdf_file)
                all_documents.extend(documents)
                
                # Update progress
                progress = (i + 1) / len(uploaded_files)
                st.progress(progress, f"Processed {i+1}/{len(uploaded_files)} files")
            
            if all_documents:
                st.write(f"✅ Successfully processed {len(all_documents)} PDF files")
                status.update(label="Processing completed!", state="complete")
                
                # Preview PDF content
                self._show_pdf_content_preview(all_documents)
                
                # Add to database
                rag_system.add_documents_to_vectorstore(all_documents)
                st.success("🎉 Documents added to database successfully!")
                st.rerun()
            else:
                st.error("❌ No content could be extracted from the PDF files")
                status.update(label="Processing failed!", state="error")
    
    def _show_pdf_content_preview(self, documents):
        """Show preview of PDF content.
        
        Args:
            documents: List of processed documents
        """
        with st.expander("📖 Preview PDF Content", expanded=True):
            for i, doc in enumerate(documents):
                st.markdown(f"**PDF: {doc.metadata['source']}**")
                st.markdown(f"*Pages: {doc.metadata['pages']} | Type: {doc.metadata['type']}*")
                
                # Show content preview
                preview = doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content
                st.text_area(f"Content Preview {i+1}", preview, height=120, disabled=True)
                
                if i < len(documents) - 1:
                    st.divider()
