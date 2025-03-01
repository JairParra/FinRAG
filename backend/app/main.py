from fastapi import FastAPI
from endpoints import endpoints
import db.database as db

app = FastAPI(title="FnReport RAG API", version="1.0")

# Include routes from the endpoints module
app.include_router(endpoints.router)

# Initialize database tables
db.Base.metadata.create_all(bind=db.engine)

@app.get("/")
def home():
    """Home endpoint to verify the API is running."""
    return {"message": "FnReport RAG API"}
