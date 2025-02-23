########################################################################################
# Part 0: Imports
import os
from tqdm import tqdm
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

########################################################################################
# Part 1: PDF Processing Functions

def load_and_split_pdf(pdf_path: str, chunk_size: int = 300, chunk_overlap: int = 50) -> list:
    """
    Load a PDF document from the given path and split its content into smaller chunks.
    
    Parameters:
      - pdf_path: Path to the PDF file.
      - chunk_size: The maximum token length for each chunk.
      - chunk_overlap: The number of overlapping tokens between chunks.
      
    Returns:
      - List of document chunks (full splits).
    """
    # Load the PDF document using PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Split the loaded document into smaller chunks
    splits = text_splitter.split_documents(documents)
    return splits

def load_full_pdf_splits(data_dir: str, 
                         chunk_size: int = 300, 
                         chunk_overlap: int = 50, 
                         verbose: int = 0) -> dict:
    """
    Discover all PDF files in the given directory, load and split each PDF document,
    and return a dictionary mapping each PDF file path to its list of full document splits.
    
    Parameters:
      - data_dir: Path to the directory containing PDF files.
      - chunk_size: The maximum token length for each chunk (default: 300).
      - chunk_overlap: The number of overlapping tokens between chunks (default: 50).
      
    Returns:
      - Dictionary with keys as PDF file paths and values as lists of document chunks.
    """
    splits_dict = {}
    for file in os.listdir(data_dir):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file)
            try:
                splits = load_and_split_pdf(pdf_path, chunk_size, chunk_overlap)
                splits_dict[pdf_path] = splits
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
                
    return splits_dict

# # Example usage for testing purposes
# if __name__ == "__main__":
#     sample_pdf = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_report.pdf")
#     chunks = load_and_split_pdf(sample_pdf)
#     print(f"Loaded and split {len(chunks)} chunks from {sample_pdf}")
