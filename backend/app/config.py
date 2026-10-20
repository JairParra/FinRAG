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

# ----------------------------
# LLM Model Configuration
# ----------------------------
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")
DEFAULT_REASONING_EFFORT = os.getenv("REASONING_EFFORT", "medium")

def get_chat_llm(model: str = None, reasoning_effort: str = None, **kwargs):
    """
    Factory to return a ChatOpenAI instance configured with project defaults
    (gpt-6-luna with medium reasoning effort).
    """
    from langchain_openai import ChatOpenAI
    selected_model = model or DEFAULT_MODEL
    selected_effort = reasoning_effort if reasoning_effort is not None else DEFAULT_REASONING_EFFORT
    
    llm_kwargs = {"model": selected_model, **kwargs}
    if selected_effort:
        llm_kwargs["reasoning_effort"] = selected_effort
        # Reasoning models do not accept custom temperature
        llm_kwargs.pop("temperature", None)
    return ChatOpenAI(**llm_kwargs)

# Ensure environment variables are set before using services
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY if OPENAI_API_KEY else ""
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY if LANGCHAIN_API_KEY else ""

