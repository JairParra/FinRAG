from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
import shutil
import os
import uuid
from datetime import datetime
from db.database import get_db
from models.document import Document
from schemas.document import DocumentResponse
from services.vector_db import build_vector_store
from services.processing import load_and_split_pdf

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------------------
# 1. Upload and Store a PDF Document
# ----------------------------
@router.post("/documents/upload/", response_model=DocumentResponse)
def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a PDF file, extract content, split it, and store embeddings."""
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract and split text from PDF
    text_chunks = load_and_split_pdf(file_path)
    extracted_text = "\n".join([chunk.page_content for chunk in text_chunks])

    # Create document entry
    document = Document(
        id=file_id,
        title=file.filename,
        file_path=file_path,
        content=extracted_text,
        uploaded_at=datetime.utcnow(),
        processed=False
    )
    
    db.add(document)
    db.commit()
    db.refresh(document)

    # Store document embeddings
    build_vector_store([document])  

    return document
