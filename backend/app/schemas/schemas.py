from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uuid

# Document Schema
class DocumentCreate(BaseModel):
    title: str
    content: str
    metadata: Optional[Dict[str, str]] = None

class DocumentResponse(BaseModel):
    id: str
    title: str
    content: str
    metadata: Optional[Dict[str, str]] = None

# Query Schema
class QueryRequest(BaseModel):
    query_text: str
    metadata_filters: Optional[Dict[str, List[str]]] = None
    top_k: int = 5
    use_embedding: bool = True

# Retrieved Document Schema
class RetrievedDocument(BaseModel):
    id: str
    content: str
    metadata: Optional[Dict[str, str]] = None
    score: float

# Context Schema
class ContextResponse(BaseModel):
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    retrieved_documents: List[RetrievedDocument]
    generated_summary: Optional[str] = None

# Final Response Schema
class ResponseModel(BaseModel):
    query_id: str
    final_response: str
    context_used: Optional[str] = None
    retrieved_sources: List[str]
