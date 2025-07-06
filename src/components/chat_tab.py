"""
Chat tab component.
"""

import streamlit as st
from ..core.rag_system import RAGSystem
from ..utils.ui_helpers import render_feature_card

class ChatTab:
    """Chat tab component."""
    
    def render(self, rag_system: RAGSystem):
        """Render the chat tab.
        
        Args:
            rag_system: The RAG system instance
        """
        st.markdown("### 💬 Intelligent Chat")
        render_feature_card(
            "💬 Chat with your documents",
            "Ask questions about your uploaded content and get AI-powered answers with source citations."
        )
        
        # Initialize chat messages
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Quick actions
        self._render_quick_actions()
        
        # Show example questions if requested
        self._render_example_questions()
        
        # Show chat stats if requested
        self._render_chat_stats()
        
        # Chat container and history
        self._render_chat_history()
        
        # Chat input
        self._handle_chat_input(rag_system)
    
    def _render_quick_actions(self):
        """Render quick action buttons."""
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("💡 Example Questions", use_container_width=True):
                st.session_state.show_examples = not st.session_state.get('show_examples', False)
        with col2:
            if st.button("📊 Chat Stats", use_container_width=True):
                st.session_state.show_stats = not st.session_state.get('show_stats', False)
        with col3:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
    
    def _render_example_questions(self):
        """Render example questions if enabled."""
        if st.session_state.get('show_examples', False):
            with st.expander("💡 Example Questions", expanded=True):
                st.markdown("""
                **Try asking questions like:**
                - "What is the main topic of the documents?"
                - "Summarize the key points from the content"
                - "What are the most important findings?"
                - "Can you explain [specific concept] from the documents?"
                """)
    
    def _render_chat_stats(self):
        """Render chat statistics if enabled."""
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
    
    def _render_chat_history(self):
        """Render chat message history."""
        chat_container = st.container()
        
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
                        self._render_feedback_buttons(i)
    
    def _render_feedback_buttons(self, message_index: int):
        """Render feedback buttons for assistant messages.
        
        Args:
            message_index: Index of the message
        """
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("👍", key=f"like_{message_index}", help="Helpful response"):
                st.success("Thanks for your feedback!")
        with col2:
            if st.button("👎", key=f"dislike_{message_index}", help="Not helpful"):
                st.info("Thanks for your feedback!")
    
    def _handle_chat_input(self, rag_system: RAGSystem):
        """Handle chat input and generate responses.
        
        Args:
            rag_system: The RAG system instance
        """
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Check if database has content
            if rag_system.vectorstore is None:
                st.error("⚠️ Please add some data first by scraping a website or uploading PDFs")
                return
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    result = rag_system.query(prompt)
                    
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
