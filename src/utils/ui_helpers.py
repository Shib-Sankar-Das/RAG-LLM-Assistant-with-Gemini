"""
Utility functions for the RAG application.
"""

import streamlit as st

def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app."""
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

def render_main_header():
    """Render the main application header."""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 RAG LLM Assistant</h1>
        <p>Intelligent Document Processing & Chat with Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)

def render_feature_card(title: str, description: str):
    """Render a feature card with title and description."""
    st.markdown(f"""
    <div class="feature-card">
        <h4>{title}</h4>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)

def render_status_card(status: str, message: str, card_type: str = "status"):
    """Render a status card with message."""
    card_class = f"{card_type}-card"
    st.markdown(f"""
    <div class="{card_class}">
        <strong>{status}</strong><br>
        {message}
    </div>
    """, unsafe_allow_html=True)

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes = int(size_bytes / 1024)
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def truncate_text(text: str, max_length: int = 50) -> str:
    """Truncate text to specified length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def validate_url(url: str) -> bool:
    """Validate if the provided URL is properly formatted."""
    if not url:
        return False
    
    if not url.startswith(('http://', 'https://')):
        return False
    
    return True

def format_datetime(datetime_str: str) -> str:
    """Format datetime string for display."""
    try:
        return datetime_str[:19].replace('T', ' ')
    except:
        return datetime_str
