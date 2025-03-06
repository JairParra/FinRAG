from sqlalchemy import Column, String, DateTime, Boolean, JSON
from datetime import datetime
from db.database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, index=True)  # UUID as string
    title = Column(String, nullable=False)
    file_path = Column(String, nullable=True)  # Path for PDF storage
    content = Column(String, nullable=True)  # Extracted text content
    metadata = Column(JSON, nullable=True)  # Store metadata as JSON
    vector_embedding = Column(JSON, nullable=True)  # Store embeddings
    processed = Column(Boolean, default=False)  # Embeddings status
    uploaded_at = Column(DateTime, default=datetime.utcnow)
