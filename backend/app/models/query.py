from sqlalchemy import Column, String, Integer, Boolean, JSON
from db.database import Base

class Query(Base):
    __tablename__ = "queries"

    id = Column(String, primary_key=True, index=True)  # UUID as string
    query_text = Column(String, nullable=False)
    metadata_filters = Column(JSON, nullable=True)  # Optional filters
    top_k = Column(Integer, default=5)
    use_embedding = Column(Boolean, default=True)
