"""
Session state management for the Streamlit app
"""
import streamlit as st


def initialize_session_state():
    """Initialize all session state variables"""
    
    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Conversation history for context
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    # Vector store and retriever
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    
    # Document processing state
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    
    # Uploaded files tracking
    if "uploaded_files_names" not in st.session_state:
        st.session_state.uploaded_files_names = []
    
    # Sample data usage
    if "using_sample_data" not in st.session_state:
        st.session_state.using_sample_data = False
    
    # Vector database stats
    if "vector_count" not in st.session_state:
        st.session_state.vector_count = 0
    
    # Document splits
    if "document_splits" not in st.session_state:
        st.session_state.document_splits = []

    # Suggested prompts
    if "suggested_prompts" not in st.session_state:
        st.session_state.suggested_prompts = []

    # Processing lock to prevent concurrent runs
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False


def reset_conversation():
    """Reset the conversation history"""
    st.session_state.messages = []
    st.session_state.conversation_history = []


def reset_documents():
    """Reset document-related state"""
    st.session_state.retriever = None
    st.session_state.documents_loaded = False
    st.session_state.uploaded_files_names = []
    st.session_state.using_sample_data = False
    st.session_state.vector_count = 0
    st.session_state.document_splits = []
    st.session_state.suggested_prompts = []
    st.session_state.is_processing = False
    reset_conversation()
