from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
#from sentence_transformers import SentenceTransformer
#import faiss
import numpy as np
from schemas.schemas import DocumentCreate, DocumentResponse, QueryRequest, ContextResponse, ResponseModel, RetrievedDocument
from models.models import Document
from db.database import get_db

router = APIRouter()

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS Vector Store
#index = faiss.IndexFlatL2(384)  
#vector_map = {}  # {index_pos: doc_id}

# -------------------------------
# 1. Add Document
# -------------------------------
@router.post("/documents/", response_model=DocumentResponse)
def add_document(doc: DocumentCreate, db: Session = Depends(get_db)):
    """Store a document and its vector representation."""
    embedding = embedding_model.encode(doc.content).tolist()

    new_doc = Document(
        title=doc.title,
        content=doc.content,
        metadata=doc.metadata,
        vector_embedding=embedding
    )

    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)

    # Store in FAISS
    index.add(np.array([embedding], dtype=np.float32))
    vector_map[len(vector_map)] = str(new_doc.id)

    return new_doc

# -------------------------------
# 2. Retrieve Document by ID
# -------------------------------
@router.get("/documents/{doc_id}", response_model=DocumentResponse)
def get_document(doc_id: str, db: Session = Depends(get_db)):
    """Fetch a document by its ID."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc

# -------------------------------
# 3. Query for Relevant Documents
# -------------------------------
@router.post("/query/", response_model=ContextResponse)
def search_query(query: QueryRequest, db: Session = Depends(get_db)):
    """Find relevant documents based on a query."""
    query_embedding = embedding_model.encode(query.query_text).reshape(1, -1).astype(np.float32)
    _, indices = index.search(query_embedding, query.top_k)

    retrieved_docs = []
    for i in indices[0]:
        if i in vector_map:
            doc_id = vector_map[i]
            doc = db.query(Document).filter(Document.id == doc_id).first()
            retrieved_docs.append(RetrievedDocument(id=doc.id, content=doc.content, metadata=doc.metadata, score=float(i)))

    summary = " ".join([doc.content[:100] for doc in retrieved_docs])  # Placeholder

    return ContextResponse(retrieved_documents=retrieved_docs, generated_summary=summary)

# -------------------------------
# 4. Get Final Chatbot Response
# -------------------------------
@router.get("/response/{query_id}", response_model=ResponseModel)
def get_final_response(query_id: str, db: Session = Depends(get_db)):
    """Return chatbot-generated response based on retrieved docs."""
    # Placeholder chatbot response
    response_text = "Based on the retrieved documents, here’s what I found..."
    
    return ResponseModel(
        query_id=query_id,
        final_response=response_text,
        context_used="Summarized content from relevant sources.",
        retrieved_sources=[]
    )
