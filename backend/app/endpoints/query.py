from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db.database import get_db
from schemas.query import QueryRequest, ContextResponse, RetrievedDocument
from services.vector_db import extract_all_vectors

router = APIRouter()

# ----------------------------
# 1. Query for Relevant Documents
# ----------------------------
@router.post("/query/", response_model=ContextResponse)
def search_query(query: QueryRequest, db: Session = Depends(get_db)):
    """Find relevant documents based on a query."""

    # Retrieve stored vectors from ChromaDB
    retrieved_docs = extract_all_vectors(return_as_json=False)

    if not retrieved_docs:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    # Sort results and return the top K matches
    top_k_results = sorted(retrieved_docs, key=lambda x: x['vector'], reverse=True)[:query.top_k]

    return ContextResponse(retrieved_documents=top_k_results, generated_summary="Summary of retrieved documents.")
