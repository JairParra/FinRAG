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



import re
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

class BM25SearchIndex:
    """Sparse keyword search index using BM25Okapi with financial token preservation."""
    def __init__(self, documents: list):
        self.documents = documents
        self.corpus = [self._tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(self.corpus) if self.corpus and len(self.corpus) > 0 else None

    @staticmethod
    def _tokenize(text: str) -> list:
        # Lowercase and extract alphanumeric tokens, preserving percentage signs, dollar signs, and hyphenated terms
        return [tok.strip('.,()[]{}"\'') for tok in re.findall(r'[a-zA-Z0-9$%.-]+', text.lower()) if tok.strip('.,()[]{}"\'')]

    def search(self, query: str, top_k: int = 8) -> list:
        if not self.bm25 or not self.documents:
            return []
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.documents[i] for i in top_indices if scores[i] > 0]


class HybridRetriever:
    """
    Hybrid retriever combining dense semantic search (ChromaDB)
    and sparse lexical search (BM25Okapi).
    """
    def __init__(self, dense_retriever, bm25_index):
        self.dense_retriever = dense_retriever
        self.bm25_index = bm25_index

    def invoke(self, query: str, k: int = 8) -> list:
        """Standard LangChain invoke interface with internal RRF fusion."""
        try:
            dense_docs = self.dense_retriever.invoke(query)
        except Exception:
            dense_docs = []

        try:
            bm25_docs = self.bm25_index.search(query, top_k=k)
        except Exception:
            bm25_docs = []

        # RRF Fusion
        fused = {}
        for rank, doc in enumerate(dense_docs):
            doc_key = (doc.metadata.get("source", ""), doc.metadata.get("page", 0), doc.page_content[:100])
            if doc_key not in fused:
                fused[doc_key] = {"doc": doc, "score": 0.0}
            fused[doc_key]["score"] += 1.0 / (rank + 60)

        for rank, doc in enumerate(bm25_docs):
            doc_key = (doc.metadata.get("source", ""), doc.metadata.get("page", 0), doc.page_content[:100])
            if doc_key not in fused:
                fused[doc_key] = {"doc": doc, "score": 0.0}
            fused[doc_key]["score"] += 1.0 / (rank + 60)

        sorted_docs = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs[:k]]

    def get_hybrid_results(self, query: str, k: int = 8) -> tuple:
        """Returns separate (dense_docs, bm25_docs) for multi-query RRF pipelines."""
        try:
            dense_docs = self.dense_retriever.invoke(query)
        except Exception:
            dense_docs = []

        try:
            bm25_docs = self.bm25_index.search(query, top_k=k)
        except Exception:
            bm25_docs = []

        return dense_docs, bm25_docs


def build_vector_store(documents: list, persist_dir: str = DEFAULT_PERSIST_DIR, force_reload: bool = False) -> any:
    """
    Create or update a persistent hybrid vector store (Chroma Dense + BM25 Sparse).
    Returns a HybridRetriever object.
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
            temp_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            try:
                all_data = temp_store._collection.get()
                all_ids = all_data.get("ids", [])
                if all_ids:
                    batch_size = 1000
                    for i in range(0, len(all_ids), batch_size):
                        batch_ids = all_ids[i:i + batch_size]
                        temp_store._collection.delete(ids=batch_ids)
                    print(f"🗑️  Cleared {len(all_ids)} vectors from existing store")
            except Exception as e:
                print(f"⚠️  Warning: Could not clear collection: {e}")
            del temp_store
            gc.collect()
        except Exception as e:
            print(f"⚠️  Warning during force reload: {e}")
    
    # Check if the persistent directory exists and is non-empty
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print(f"📂 Loading existing vector store from {persist_dir}...")
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        
        try:
            stored_data = vectorstore._collection.get(include=["metadatas"])
            stored_metadatas = stored_data.get("metadatas", [])
            processed_sources = {m.get("source") for m in stored_metadatas if m and "source" in m}
            print(f"✅ Found {len(processed_sources)} documents already in vector store")
        except Exception as e:
            print(f"⚠️  Warning: Unable to retrieve stored metadata: {e}")
            processed_sources = set()
        
        new_sources = {doc.metadata.get("source") for doc in documents}
        sources_to_add = new_sources - processed_sources
        
        if sources_to_add:
            new_documents = [doc for doc in documents if doc.metadata.get("source") in sources_to_add]
            print(f"➕ Adding {len(new_documents)} new chunks from {len(sources_to_add)} documents...")
            vectorstore.add_documents(new_documents)
            print(f"✅ Vector store updated successfully")
    else:
        print(f"🆕 Creating new vector store with {len(documents)} chunks...")
        vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_dir)
        print(f"✅ Vector store created at {persist_dir}")
    
    # Retrieve all documents to construct or sync the BM25 index
    try:
        all_store_data = vectorstore._collection.get(include=["documents", "metadatas"])
        bm25_docs = [
            Document(page_content=content, metadata=meta or {})
            for content, meta in zip(all_store_data.get("documents", []), all_store_data.get("metadatas", []))
        ]
        if not bm25_docs:
            bm25_docs = documents
    except Exception:
        bm25_docs = documents

    print(f"⚡ Building BM25 sparse index over {len(bm25_docs)} chunks...")
    bm25_index = BM25SearchIndex(bm25_docs)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    print(f"🚀 Hybrid Retriever (Chroma Dense + BM25 Sparse) ready.")
    return HybridRetriever(dense_retriever=dense_retriever, bm25_index=bm25_index)


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
