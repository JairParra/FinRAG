from fastapi import FastAPI
from backend.app.endpoints import document
from endpoints import embeddings, queries, responses

app = FastAPI(title="FinReport RAG API", version="1.0")

# Include routes
app.include_router(document.router)
app.include_router(embeddings.router)
app.include_router(queries.router)
app.include_router(responses.router)

# Initialize DB tables
@app.on_event("startup")
def init_db():
    from db.database import Base, engine
    Base.metadata.create_all(bind=engine)

@app.get("/")
def home():
    return {"message": "FinReport RAG API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
