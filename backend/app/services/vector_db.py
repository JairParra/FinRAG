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
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)



def build_vector_store(documents: list, persist_dir: str = DEFAULT_PERSIST_DIR) -> any:
    """
    Create or update a persistent vector store.
    If the store already exists, only add new documents (i.e., documents whose metadata['source'] 
    is not already in the store). Otherwise, create a new store.
    Returns a retriever object.
    """
    embeddings = OpenAIEmbeddings()
    
    # Check if the persistent directory exists and is non-empty
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        # Retrieve stored metadata to check which sources have been processed already
        try:
            stored_data = vectorstore._collection.get(include=["metadatas"])
            stored_metadatas = stored_data.get("metadatas", [])
            processed_sources = {m.get("source") for m in stored_metadatas if m and "source" in m}
        except Exception as e:
            print("Warning: Unable to retrieve stored metadata. Proceeding without filtering new documents.")
            processed_sources = set()
        # Filter new documents by comparing their 'source' metadata
        new_documents = [doc for doc in documents if doc.metadata.get("source") not in processed_sources]
        if new_documents:
            vectorstore.add_documents(new_documents)
            #vectorstore.persist()
    else:
        # Create a new persistent vector store if none exists
        vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_dir)
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
    embeddings = OpenAIEmbeddings()
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
