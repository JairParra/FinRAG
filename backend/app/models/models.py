from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import uuid

# ----------------------------
# 1. Document Model
# ----------------------------
class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str
    metadata: Optional[Dict[str, str]] = None  # Tags, source, date info
    vector_embedding: Optional[List[float]] = None  # Precomputed embedding

# ----------------------------
# 2. Query Model
# ----------------------------
class Query(BaseModel):
    query_text: str
    metadata_filters: Optional[Dict[str, List[str]]] = None  # Filters (e.g., "tags": ["finance"])
    top_k: int = 5
    use_embedding: bool = True  # Use vector search or not

# ----------------------------
# 3. Retrieved Document Model
# ----------------------------
class RetrievedDocument(BaseModel):
    id: str
    content: str
    metadata: Optional[Dict[str, str]] = None
    score: float

# ----------------------------
# 4. Context Model (Returned Retrieved Docs)
# ----------------------------
class Context(BaseModel):
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    retrieved_documents: List[RetrievedDocument]
    generated_summary: Optional[str] = None  # AI-generated summary of retrieved documents

# ----------------------------
# 5. Final Response Model
# ----------------------------
class ResponseModel(BaseModel):
    query_id: str
    final_response: str
    context_used: Optional[str] = None
    retrieved_sources: List[str]
