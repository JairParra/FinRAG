"""
FinReport RAG - Streamlit Application
Main entry point for the financial document analysis chatbot
"""
import sys
import os

# Fix Windows encoding issues with emojis - set before any other imports
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # Only reconfigure if available and not already done
    if hasattr(sys.stdout, 'reconfigure') and hasattr(sys.stdout, 'encoding'):
        try:
            if sys.stdout.encoding.lower() != 'utf-8':
                sys.stdout.reconfigure(encoding='utf-8')
            if sys.stderr.encoding.lower() != 'utf-8':
                sys.stderr.reconfigure(encoding='utf-8')
        except (AttributeError, ValueError):
            pass  # Skip if reconfigure fails

import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="FinReport RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import custom modules
from streamlit_app.config.colors import get_custom_css
from streamlit_app.utils.session_state import initialize_session_state, reset_documents, reset_conversation
from streamlit_app.utils.timing import ProcessingTimer
from streamlit_app.utils.auto_viz import generate_visualizations_from_response
from backend.app.services.processing import load_and_split_pdf, load_full_pdf_splits
from backend.app.services.vector_db import build_vector_store, extract_all_vectors
from backend.app.services.enhanced_rag import chat_with_documents, generate_suggested_prompts
from backend.app.utils.logging_config import log_info, log_error, log_debug, log_exception
from streamlit_app.utils.visualizations import (
    extract_financial_tables, 
    create_financial_chart,
    generate_financial_summary_chart,
    extract_metrics_summary
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Initialize session state
initialize_session_state()


########################################################################################
# Sidebar - Document Management

def render_sidebar():
    """Render the sidebar with document upload and management"""
    with st.sidebar:
        st.title("📊 FinReport RAG")
        st.markdown("---")
        
        # Document Source Selection
        st.header("📁 Document Source")
        
        doc_source = st.radio(
            "Choose data source:",
            ["Upload PDFs", "Use Sample Data"],
            key="doc_source"
        )
        
        if doc_source == "Upload PDFs":
            render_upload_section()
        else:
            render_sample_data_section()
        
        st.markdown("---")
        
        # Database Stats
        if st.session_state.documents_loaded:
            render_stats()
        
        st.markdown("---")
        
        # Actions
        render_actions()


def render_upload_section():
    """Render PDF upload section"""
    st.subheader("Upload Financial Reports")
    
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload financial reports in PDF format"
    )
    
    if uploaded_files:
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            process_uploaded_files(uploaded_files)


def render_sample_data_section():
    """Render sample data loading section"""
    st.subheader("Sample Financial Reports")
    
    sample_data_path = Path("sample_data")
    
    if sample_data_path.exists():
        pdf_files = list(sample_data_path.glob("*.pdf"))
        
        if pdf_files:
            st.info(f"📄 Found {len(pdf_files)} sample PDF(s)")
            
            for pdf in pdf_files:
                st.text(f"• {pdf.name}")
            
            if not st.session_state.documents_loaded:
                col1, col2 = st.columns(2)
                with col1:
                    load_clicked = st.button("🚀 Load", type="primary", use_container_width=True, help="Load sample data")
                with col2:
                    reload_clicked = st.button("🔄 Reload", use_container_width=True, help="Force reload and rebuild vector index")
                
                # Process outside columns so progress spans full width
                if load_clicked:
                    process_sample_data(str(sample_data_path), force_reload=False)
                elif reload_clicked:
                    process_sample_data(str(sample_data_path), force_reload=True)
        else:
            st.warning("No PDF files found in sample_data/")
    else:
        st.warning("sample_data/ folder not found")


def process_uploaded_files(uploaded_files):
    """Process uploaded PDF files with parallel processing"""
    try:
        log_info(f"Starting to process {len(uploaded_files)} uploaded files")
        
        # Create temp directory for uploads
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        # Create progress bar and timer
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_text = st.empty()
        timer = ProcessingTimer(2)  # 2 steps: save + process, then vector store
        
        # Step 1: Save uploaded files
        timer.start_item()
        status_text.markdown("### 📥 Saving uploaded files...")
        progress_bar.progress(0.1)
        
        # Save all files first
        for uploaded_file in uploaded_files:
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        timer.complete_item()
        progress_bar.progress(0.3)
        
        # Step 2: Process PDFs in parallel
        timer.start_item()
        status_text.markdown(f"### ⚙️ Processing {len(uploaded_files)} PDFs (Grab a cup of coffee 🍵)...")
        time_text.markdown(f"**⏱️ Elapsed:** {ProcessingTimer.format_time(timer.get_elapsed_time())} | **🔄 Processing in parallel...**")
        
        pdf_splits_dict = load_full_pdf_splits(
            str(temp_dir),
            use_parallel=True,
            max_workers=4
        )
        
        timer.complete_item()
        progress_bar.progress(0.6)
        
        # Aggregate splits
        all_splits = []
        file_names = []
        for file, splits in pdf_splits_dict.items():
            all_splits.extend(splits)
            file_names.append(file)
        
        if not all_splits:
            log_error("No documents could be processed")
            st.error("No documents could be processed")
            progress_bar.empty()
            status_text.empty()
            time_text.empty()
            return
        
        # Step 3: Build vector store
        timer.start_item()
        log_info(f"Building vector store with {len(all_splits)} chunks")
        status_text.markdown(f"### 🔍 Building vector database with {len(all_splits)} chunks (Final stretch! ☕)...")
        time_text.markdown(f"**⏱️ Elapsed:** {ProcessingTimer.format_time(timer.get_elapsed_time())} | **🔄 Generating embeddings...**")
        progress_bar.progress(0.7)
        
        st.session_state.retriever = build_vector_store(all_splits)
        timer.complete_item()
        st.session_state.documents_loaded = True
        st.session_state.uploaded_files_names = file_names
        st.session_state.using_sample_data = False
        st.session_state.document_splits = all_splits
        st.session_state.document_splits = all_splits
        st.session_state.vector_count = len(all_splits)
        
        # Generate suggested prompts
        log_info("Generating suggested prompts...")
        status_text.markdown("### 💡 Generating suggested questions...")
        
        # Create a context summary from the first few chunks
        context_summary = " ".join([split.page_content for split in all_splits[:5]])
        st.session_state.suggested_prompts = generate_suggested_prompts(context_summary)
        log_info(f"Generated {len(st.session_state.suggested_prompts)} suggested prompts")
        
        # Clean up temp files
        for file in temp_dir.glob("*"):
            file.unlink()
        temp_dir.rmdir()
        
        # Complete
        progress_bar.progress(1.0)
        total_time = ProcessingTimer.format_time(timer.get_elapsed_time())
        log_info(f"Upload processing completed in {total_time}")
        
        status_text.empty()
        time_text.empty()
        progress_bar.empty()
        
        st.success(f"✅ Processed {len(file_names)} document(s) with {len(all_splits)} chunks in {total_time}!")
        st.rerun()
        
    except Exception as e:
        log_exception("Error processing uploaded documents", e)
        st.error(f"❌ Error processing documents: {str(e)}")


def process_sample_data(data_dir: str, force_reload: bool = False):
    """Process sample data from directory with parallel processing and smart caching"""
    try:
        log_info(f"Starting to process sample data from: {data_dir} (force_reload={force_reload})")
        
        # Create progress bar and timer
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_text = st.empty()
        timer = ProcessingTimer(2)  # 2 main steps: PDF processing + vector store
        
        # Step 1: Load and split PDFs using PARALLEL processing
        timer.start_item()
        status_text.markdown("### ⚙️ Processing Documents (Grab a cup of coffee, this will take some time 🍵)...")
        time_text.markdown(f"**⏱️ Elapsed:** 0s | **🔄 Loading PDFs in parallel with 4 workers...**")
        progress_bar.progress(0.1)
        
        log_info("Loading PDFs with parallel processing (4 workers)")
        pdf_splits_dict = load_full_pdf_splits(
            data_dir,
            use_parallel=True,
            max_workers=4
        )
        
        timer.complete_item()
        progress_bar.progress(0.5)
        
        if not pdf_splits_dict:
            log_error("No PDF files could be processed")
            st.error("No PDF files found or processed")
            progress_bar.empty()
            status_text.empty()
            time_text.empty()
            return
        
        log_info(f"Successfully processed {len(pdf_splits_dict)} PDFs")
        
        # Aggregate splits
        all_splits = []
        for splits in pdf_splits_dict.values():
            all_splits.extend(splits)
        
        if not all_splits:
            log_error("No documents could be processed")
            st.error("No documents could be processed")
            progress_bar.empty()
            status_text.empty()
            time_text.empty()
            return
        
        log_info(f"Total chunks to embed: {len(all_splits)}")
        
        # Step 2: Build vector store (with smart caching)
        timer.start_item()
        status_text.markdown(f"### 🔍 Building vector database with {len(all_splits)} chunks (Almost there! ☕)...")
        time_text.markdown(f"**⏱️ Elapsed:** {ProcessingTimer.format_time(timer.get_elapsed_time())} | **🔄 Checking cache...**")
        progress_bar.progress(0.6)
        
        log_info(f"Building vector store (force_reload={force_reload})...")
        st.session_state.retriever = build_vector_store(all_splits, force_reload=force_reload)
        timer.complete_item()
        st.session_state.documents_loaded = True
        st.session_state.uploaded_files_names = list(pdf_splits_dict.keys())
        st.session_state.using_sample_data = True
        st.session_state.document_splits = all_splits
        st.session_state.document_splits = all_splits
        st.session_state.vector_count = len(all_splits)
        
        # Generate suggested prompts
        log_info("Generating suggested prompts...")
        status_text.markdown("### 💡 Generating suggested questions...")
        
        # Create a context summary from the first few chunks
        context_summary = " ".join([split.page_content for split in all_splits[:5]])
        st.session_state.suggested_prompts = generate_suggested_prompts(context_summary)
        log_info(f"Generated {len(st.session_state.suggested_prompts)} suggested prompts")
        
        # Complete
        progress_bar.progress(1.0)
        total_time = ProcessingTimer.format_time(timer.get_elapsed_time())
        log_info(f"Processing completed in {total_time}")
        
        status_text.empty()
        time_text.empty()
        progress_bar.empty()
        
        st.success(f"✅ Loaded {len(pdf_splits_dict)} sample document(s) with {len(all_splits)} chunks in {total_time}!")
        st.rerun()
        
    except Exception as e:
        log_exception("Error loading sample data", e)
        st.error(f"❌ Error loading sample data: {str(e)}")


def render_stats():
    """Render database statistics"""
    st.header("📈 Database Stats")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Documents", len(st.session_state.uploaded_files_names))
    
    with col2:
        st.metric("Chunks", st.session_state.vector_count)
    
    with st.expander("📄 Loaded Documents"):
        for filename in st.session_state.uploaded_files_names:
            st.text(f"• {filename}")


def render_actions():
    """Render action buttons"""
    st.header("⚙️ Actions")
    
    if st.button("🗑️ Clear Documents", use_container_width=True):
        reset_documents()
        st.success("Documents cleared!")
        st.rerun()
    
    if st.button("🔄 Reset Chat", use_container_width=True):
        reset_conversation()
        st.success("Chat history cleared!")
        st.rerun()


########################################################################################
# Main Content Area

def render_main_content():
    """Render the main content area"""
    
    # Header
    st.title("🤖 Financial Document Analysis Chatbot")
    
    if not st.session_state.documents_loaded:
        render_welcome_screen()
    else:
        render_chat_interface()


def render_welcome_screen():
    """Render welcome screen when no documents are loaded"""
    st.markdown("""
    ### Welcome to FinReport RAG! 👋
    
    This intelligent chatbot helps you analyze and extract insights from financial documents using advanced RAG (Retrieval-Augmented Generation) technology.
    
    **Features:**
    - 📄 Upload and analyze multiple PDF financial reports
    - 💬 Chat naturally about your documents with conversation memory
    - 📊 **Automatic visualizations** - Charts generated on-the-fly from your questions
    - 🎯 Intelligent query routing for optimal responses
    - 🔍 Advanced multi-query retrieval with RAG Fusion
    - ⏱️ Real-time progress tracking with time estimation
    
    **Get Started:**
    1. Upload your PDF documents or use sample data from the sidebar
    2. Wait for processing to complete (you'll see progress and time estimates)
    3. Start asking questions - visualizations appear automatically when relevant!
    
    ---
    """)
    
    # Example questions
    st.subheader("📝 Example Questions You Can Ask:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - What are the key financial metrics in 2023?
        - How has revenue trended over the years?
        - What risks are mentioned in the reports?
        - Show me the liquidity position
        """)
    
    with col2:
        st.markdown("""
        - Compare profit margins across years
        - What are the major expenses?
        - Extract the balance sheet data
        - Summarize the cash flow statement
        """)


def render_chat_interface():
    """Render the chat interface"""
    
    # Just render chat tab directly (visualizations are now automatic)
    render_chat_tab()


def render_chat_tab():
    """Render the chat tab"""
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display visualizations if present (only for assistant messages)
            if message["role"] == "assistant" and "visualizations" in message and message["visualizations"]:
                st.markdown("---")
                st.markdown("### 📊 Generated Visualizations")
                
                data_tables = message.get("data_tables", [])
                for idx, (fig, description) in enumerate(message["visualizations"]):
                    st.caption(description)
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{id(message)}_{idx}")
                    
                    # Show the data table used for this visualization
                    if idx < len(data_tables) and data_tables[idx] is not None:
                        with st.expander(f"📋 View Data Table {idx + 1}"):
                            st.dataframe(data_tables[idx], use_container_width=True)
            
            # Show RAG indicator with expandable sources
            if message["role"] == "assistant" and "used_rag" in message:
                if message["used_rag"] and "sources" in message and message["sources"]:
                    with st.expander("🔍 Retrieved from documents (click to view sources)"):
                        for i, source in enumerate(message["sources"], 1):
                            st.caption(f"**Source {i}:** {source['source']} (Page {source['page']}) | Relevance: {source['score']:.3f}")
                            st.text(source['excerpt'][:150] + "..." if len(source['excerpt']) > 150 else source['excerpt'])
                            if i < len(message["sources"]):
                                st.divider()
                elif message["used_rag"]:
                    st.caption("🔍 Retrieved from documents")
                else:
                    st.caption("💭 Conversational response")
    
    # Chat input
    prompt = st.chat_input("Ask a question about your financial documents...")
    
    # Suggested prompts (only if no messages yet and documents are loaded)
    if not st.session_state.messages and st.session_state.documents_loaded and st.session_state.suggested_prompts and not prompt:
        st.markdown("### 💡 Suggested Questions")
        cols = st.columns(len(st.session_state.suggested_prompts))
        for i, sp in enumerate(st.session_state.suggested_prompts):
            # Use a custom key for each button
            if cols[i].button(sp, key=f"suggest_{i}", use_container_width=True):
                prompt = sp
    
    if prompt:
        log_info(f"User question: {prompt}")
        
        # Add user message to state FIRST
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response in assistant message block
        with st.chat_message("assistant"):
            # Status placeholder for dynamic updates
            status_placeholder = st.empty()
            response_placeholder = st.empty()
            
            try:
                # Step 1: Analyzing documents
                status_placeholder.info("🤔 Analyzing documents and retrieving relevant information... Please be patient.")
                log_debug("Generating response...")
                
                response, used_rag, sources = chat_with_documents(
                    prompt,
                    st.session_state.retriever,
                    st.session_state.conversation_history
                )
                
                log_info(f"Response generated (used_rag={used_rag}, sources={len(sources)})")
                status_placeholder.empty()  # Clear status
                
                # Display the text response
                response_placeholder.markdown(response)
                
                # Step 2: Try to generate automatic visualizations
                visualizations = []
                data_tables = []
                if used_rag:  # Only for document-based responses
                    try:
                        # Show banner AFTER the text response using a NEW placeholder
                        viz_status = st.empty()
                        viz_status.info("📊 Analyzing data and creating visualizations... Please wait.")
                        log_debug("Attempting intelligent visualization generation...")
                        
                        visualizations, data_tables = generate_visualizations_from_response(
                            response, 
                            max_charts=3,
                            original_question=prompt
                        )
                        
                        viz_status.empty()  # Clear status
                        
                        if visualizations:
                            log_info(f"Generated {len(visualizations)} visualizations")
                            st.markdown("---")
                            st.markdown("### 📊 Generated Visualizations")
                            
                            for idx, (fig, description) in enumerate(visualizations):
                                st.caption(description)
                                st.plotly_chart(fig, use_container_width=True, key=f"new_chart_{idx}_{len(st.session_state.messages)}")
                                
                                # Show the data table used for this visualization
                                if idx < len(data_tables) and data_tables[idx] is not None:
                                    with st.expander(f"📋 View Data Table {idx + 1}"):
                                        st.dataframe(data_tables[idx], use_container_width=True)
                        else:
                            log_debug("No visualizations generated from response")
                    except Exception as viz_error:
                        viz_status.empty()
                        log_exception("Error generating visualizations", viz_error)
                        # Don't show error to user, just log it
                
                # Show source information with expander
                if used_rag and sources:
                    with st.expander("🔍 Retrieved from documents (click to view sources)"):
                        for i, source in enumerate(sources, 1):
                            st.caption(f"**Source {i}:** {source['source']} (Page {source['page']}) | Relevance: {source['score']:.3f}")
                            st.text(source['excerpt'][:150] + "..." if len(source['excerpt']) > 150 else source['excerpt'])
                            if i < len(sources):
                                st.divider()
                elif used_rag:
                    st.caption("🔍 Retrieved from documents")
                else:
                    st.caption("💭 Conversational response")
                
                # Update conversation history
                st.session_state.conversation_history.append({
                    "user": prompt,
                    "assistant": response
                })
                
                # Add assistant message with visualizations and tables
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "used_rag": used_rag,
                    "sources": sources,
                    "visualizations": visualizations,
                    "data_tables": data_tables
                })
                
            except Exception as e:
                status_placeholder.empty()
                log_exception("Error generating response", e)
                error_msg = f"❌ Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "used_rag": False,
                    "sources": []
                })


def render_visualizations_tab():
    """Render the visualizations tab"""
    
    st.subheader("📊 Financial Data Visualization")
    
    # Quick metrics
    st.markdown("### Key Metrics")
    
    try:
        metrics = extract_metrics_summary(st.session_state.retriever)
        
        if metrics:
            cols = st.columns(len(metrics))
            for col, (key, value) in zip(cols, metrics.items()):
                col.metric(key, value)
        else:
            st.info("No metrics extracted. Try asking for specific financial data.")
    except Exception as e:
        st.warning(f"Could not extract metrics: {str(e)}")
    
    st.markdown("---")
    
    # Custom chart generation
    st.markdown("### Generate Custom Charts")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        chart_query = st.text_input(
            "Describe the data you want to visualize:",
            placeholder="e.g., Revenue and profit by year"
        )
    
    with col2:
        chart_type = st.selectbox(
            "Chart Type",
            ["bar", "line", "area", "pie"]
        )
    
    if st.button("Generate Chart", type="primary"):
        if chart_query:
            with st.spinner("Extracting data and generating chart..."):
                try:
                    df = extract_financial_tables(st.session_state.retriever, chart_query)
                    
                    if df is not None and not df.empty:
                        st.dataframe(df, use_container_width=True)
                        
                        fig = create_financial_chart(df, chart_type, chart_query)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Could not extract tabular data for this query. Try rephrasing or asking for specific metrics.")
                        
                except Exception as e:
                    st.error(f"Error generating visualization: {str(e)}")
        else:
            st.warning("Please enter a query describing the data you want to visualize")
    
    st.markdown("---")
    
    # Auto-generate summary chart
    if st.button("📈 Generate Summary Chart"):
        with st.spinner("Generating financial summary..."):
            try:
                fig = generate_financial_summary_chart(st.session_state.retriever)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Could not generate automatic summary chart")
            except Exception as e:
                st.error(f"Error: {str(e)}")


########################################################################################
# Main App

def main():
    """Main application entry point"""
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
