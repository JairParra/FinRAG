from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime

class DocumentCreate(BaseModel):
    title: str
    file_path: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None

class DocumentResponse(BaseModel):
    id: str
    title: str
    file_path: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None
    processed: bool
    uploaded_at: datetime
