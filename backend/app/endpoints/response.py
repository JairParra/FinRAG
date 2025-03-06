from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db.database import get_db
from schemas.response import ResponseModel
from services.rag import final_rag_response

router = APIRouter()

# ----------------------------
# 1. Get AI-Generated Chatbot Response
# ----------------------------
@router.get("/response/{query_id}", response_model=ResponseModel)
def get_final_response(query_id: str, db: Session = Depends(get_db)):
    """Return chatbot-generated response based on retrieved documents."""
    
    response_text = final_rag_response(query_id, retriever=None)  # Pass retriever object later if needed

    return ResponseModel(
        query_id=query_id,
        final_response=response_text,
        context_used="Summarized content from relevant sources.",
        retrieved_sources=[]
    )
