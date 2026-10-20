########################################################################################
# Part 0: Imports
import os
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from backend.app.config import CHROMA_DB_PATH
# from langchain_community.vectorstores import Chroma # deprecated

# Default persistent directory for the vector store
DEFAULT_PERSIST_DIR = "backend/app/db/chroma_db"

########################################################################################
# Part 1: Vector Store Functions

def get_vector_store():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        chunk_size=1000  # Process 1000 texts at once for faster embedding
    )
    return Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)



def build_vector_store(documents: list, persist_dir: str = DEFAULT_PERSIST_DIR, force_reload: bool = False) -> any:
    """
    Create or update a persistent vector store.
    If the store already exists and force_reload=False, only add new documents.
    If force_reload=True, delete existing store and create fresh.
    Returns a retriever object.
    
    Parameters:
      - documents: List of document chunks to add
      - persist_dir: Directory for persistent storage
      - force_reload: If True, delete existing store and rebuild from scratch
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        chunk_size=1000  # Batch process for faster embedding
    )
    
    # Normalize path for consistent handling across platforms
    persist_dir = os.path.normpath(persist_dir)
    
    # Force reload: use ChromaDB's proper cleanup instead of file deletion
    if force_reload and os.path.exists(persist_dir) and os.listdir(persist_dir):
        import gc
        
        try:
            print(f"🗑️  Clearing existing vector store at {persist_dir}...")
            
            # Load existing store to access its collection
            temp_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            
            # Delete all documents from the collection (safer than file deletion)
            try:
                # Get all IDs
                all_data = temp_store._collection.get()
                all_ids = all_data.get("ids", [])
                
                if all_ids:
                    # Delete in batches to avoid memory issues
                    batch_size = 1000
                    for i in range(0, len(all_ids), batch_size):
                        batch_ids = all_ids[i:i + batch_size]
                        temp_store._collection.delete(ids=batch_ids)
                    print(f"🗑️  Cleared {len(all_ids)} vectors from existing store")
                else:
                    print(f"✅ Vector store was already empty")
            except Exception as e:
                print(f"⚠️  Warning: Could not clear collection: {e}")
            
            # Clean up references
            del temp_store
            gc.collect()
            
        except Exception as e:
            print(f"⚠️  Warning during force reload: {e}")
            print(f"💡 Continuing with fresh creation...")
    
    # Check if the persistent directory exists and is non-empty
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print(f"📂 Loading existing vector store from {persist_dir}...")
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        
        # Get list of already processed sources
        try:
            stored_data = vectorstore._collection.get(include=["metadatas"])
            stored_metadatas = stored_data.get("metadatas", [])
            processed_sources = {m.get("source") for m in stored_metadatas if m and "source" in m}
            print(f"✅ Found {len(processed_sources)} documents already in vector store")
        except Exception as e:
            print(f"⚠️  Warning: Unable to retrieve stored metadata: {e}")
            processed_sources = set()
        
        # Filter new documents by comparing their 'source' metadata
        new_sources = {doc.metadata.get("source") for doc in documents}
        sources_to_add = new_sources - processed_sources
        
        if sources_to_add:
            new_documents = [doc for doc in documents if doc.metadata.get("source") in sources_to_add]
            print(f"➕ Adding {len(new_documents)} new chunks from {len(sources_to_add)} documents...")
            vectorstore.add_documents(new_documents)
            print(f"✅ Vector store updated successfully")
        else:
            print(f"✅ All documents already in vector store - no new documents to add")
    else:
        # Create a new persistent vector store if none exists
        print(f"🆕 Creating new vector store with {len(documents)} chunks...")
        vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_dir)
        print(f"✅ Vector store created at {persist_dir}")
    
    return vectorstore.as_retriever()


def extract_all_vectors(persist_dir: str = DEFAULT_PERSIST_DIR, return_as_json: bool = False) -> any:
    """
    Extracts all vectors from the persistent database, converts them to numpy arrays,
    and returns a list of dictionaries with keys:
       - "vector_id": unique vector id from the database (retrieved via 'uris')
       - "vector": the extracted vector converted to list form
       - "metadata": associated metadata from the vector
       
    If return_as_json is True, returns a JSON object with each dictionary assigned an enumerated key.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    

    data = vectorstore._collection.get(include=["embeddings", "metadatas", "documents"])
    vector_ids = data.get("ids", [])
    embeddings_list = data.get("embeddings", [])
    metadatas = data.get("metadatas", [])
    documents = data.get("documents", []) # the actual chunks of text 
     
    results = []
    for vid, vec, meta, doc in zip(vector_ids, embeddings_list, metadatas, documents):
        # Ensure the vector is converted to list form
        if isinstance(vec, np.ndarray):
            vec = vec.tolist()
        results.append({
            "vector_id": vid,
            "vector": vec,
            "metadata": meta, 
            "document": doc
        })
    if return_as_json:
        json_obj = {}
        for idx, item in enumerate(results):
            json_obj[str(idx)] = item
        return json_obj
    return results
