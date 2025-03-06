import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ----------------------------
# API Keys
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# ----------------------------
# LangChain & ChromaDB Configuration
# ----------------------------
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "backend/app/db/chroma_db")

# ----------------------------
# Database Configuration
# ----------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/finreportdb")

# ----------------------------
# Server Configuration
# ----------------------------
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

# Ensure environment variables are set before using services
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY if OPENAI_API_KEY else ""
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY if LANGCHAIN_API_KEY else ""
