from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uuid

class RetrievedDocument(BaseModel):
    id: str
    content: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None
    score: float

class ContextResponse(BaseModel):
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    retrieved_documents: List[RetrievedDocument]
    generated_summary: Optional[str] = None

class ResponseModel(BaseModel):
    query_id: str
    final_response: str
    context_used: Optional[str] = None
    retrieved_sources: List[str]
