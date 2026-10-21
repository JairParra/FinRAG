########################################################################################
# Part 0: Imports

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.app.config import OPENAI_API_KEY, LANGCHAIN_API_KEY


# Ensure API keys are set before using LangChain services
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

########################################################################################
# Part 1: PDF Processing Functions

def load_and_split_pdf(pdf_path: str, chunk_size: int = 550, chunk_overlap: int = 100) -> list:
    """
    Load a PDF document from the given path and split its content into smaller chunks.
    Optimized for financial reports to keep tables and narrative commentary intact.
    
    Parameters:
      - pdf_path: Path to the PDF file.
      - chunk_size: The maximum token length for each chunk (default: 550).
      - chunk_overlap: The number of overlapping tokens between chunks (default: 100).
      
    Returns:
      - List of document chunks (full splits).
    """
    # Load the PDF document using PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Initialize the text splitter with optimized settings for financial documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Split the loaded document into smaller chunks
    splits = text_splitter.split_documents(documents)
    return splits


def load_full_pdf_splits(data_dir: str, 
                         chunk_size: int = 550, 
                         chunk_overlap: int = 100, 
                         verbose: int = 0,
                         use_parallel: bool = True,
                         max_workers: int = 4,
                         progress_callback=None) -> dict:
    """
    Discover all PDF files in the given directory, load and split each PDF document,
    and return a dictionary mapping each PDF file path to its list of full document splits.
    Uses parallel processing for faster performance.
    
    Parameters:
      - data_dir: Path to the directory containing PDF files.
      - chunk_size: The maximum token length for each chunk (default: 550).
      - chunk_overlap: The number of overlapping tokens between chunks (default: 100).
      - use_parallel: Whether to use parallel processing (default: True).
      - max_workers: Maximum number of parallel workers (default: 4).
      - progress_callback: Optional callable(completed_count, total_count, file_name, chunks_count)
      
    Returns:
      - Dictionary with keys as PDF file paths and values as lists of document chunks.
    """
    # Find all PDF files
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        return {}
    
    splits_dict = {}
    total_files = len(pdf_files)
    
    if use_parallel and total_files > 1:
        # Parallel processing
        def process_single_pdf(file):
            pdf_path = os.path.join(data_dir, file)
            try:
                splits = load_and_split_pdf(pdf_path, chunk_size, chunk_overlap)
                return (file, splits, None)
            except Exception as e:
                return (file, None, str(e))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_pdf, file): file for file in pdf_files}
            
            completed_count = 0
            for future in as_completed(futures):
                file, splits, error = future.result()
                completed_count += 1
                if error:
                    print(f"Error processing {file}: {error}")
                else:
                    splits_dict[file] = splits
                if progress_callback:
                    try:
                        progress_callback(completed_count, total_files, file, len(splits) if splits else 0)
                    except Exception as cb_err:
                        print(f"Error in progress_callback: {cb_err}")
    else:
        # Sequential processing
        for i, file in enumerate(pdf_files, 1):
            pdf_path = os.path.join(data_dir, file)
            splits = None
            try:
                splits = load_and_split_pdf(pdf_path, chunk_size, chunk_overlap)
                splits_dict[file] = splits
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
            if progress_callback:
                try:
                    progress_callback(i, total_files, file, len(splits) if splits else 0)
                except Exception as cb_err:
                    print(f"Error in progress_callback: {cb_err}")
                
    return splits_dict

# # Example usage for testing purposes
# if __name__ == "__main__":
#     sample_pdf = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_report.pdf")
#     chunks = load_and_split_pdf(sample_pdf)
#     print(f"Loaded and split {len(chunks)} chunks from {sample_pdf}")
