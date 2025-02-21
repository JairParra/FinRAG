import os

# Define the directory structure
dirs = [
    "backend/app",
    "backend/app/routes",
    "backend/app/services",
    "backend/app/models",
    "backend/tests",
    "docker",
    "infrastructure/terraform",
    "data",
    "scripts"
]

# Define the files to create (with placeholders if needed)
files = {
    ".gitignore": "venv/\n__pycache__/\n.env\ndata/\n*.log\n",
    ".env": "DEBUG=True\nDATABASE_URL=postgres://user:password@localhost/db\n",
    "README.md": "# FnReport RAG\n\nA RAG-powered agent for financial reports.",
    "requirements.txt": "fastapi\nuvicorn\nlangchain\nopenai\nchromadb\npandas\npdfplumber\n",
    "setup.py": 'from setuptools import setup, find_packages\n\nsetup(name="FnReport-RAG", packages=find_packages())\n',
    "Makefile": "run:\n\tuvicorn backend.app.main:app --reload\n",
    "backend/app/__init__.py": "",
    "backend/app/main.py": "from fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get('/')\ndef home():\n    return {'message': 'FnReport RAG API'}\n",
    "backend/app/config.py": "import os\n\nDATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./test.db')\n",
    "backend/app/dependencies.py": "from fastapi import Depends\n\ndef get_db():\n    pass  # Placeholder for DB dependency\n",
    "backend/app/routes/__init__.py": "",
    "backend/app/routes/api.py": "from fastapi import APIRouter\n\nrouter = APIRouter()\n\n@router.get('/query')\ndef query():\n    return {'response': 'Sample response'}\n",
    "backend/app/routes/health.py": "from fastapi import APIRouter\n\nrouter = APIRouter()\n\n@router.get('/health')\ndef health_check():\n    return {'status': 'ok'}\n",
    "backend/app/services/__init__.py": "",
    "backend/app/services/rag.py": "# RAG logic implementation placeholder\n",
    "backend/app/services/processing.py": "# PDF processing placeholder\n",
    "backend/app/services/embeddings.py": "# Embeddings logic placeholder\n",
    "backend/app/models/__init__.py": "",
    "backend/app/models/document.py": "class Document:\n    pass\n",
    "backend/app/models/query.py": "class Query:\n    pass\n",
    "backend/tests/test_api.py": "def test_api():\n    pass  # Test cases\n",
    "backend/tests/test_rag.py": "def test_rag():\n    pass  # Test cases\n",
    "docker/Dockerfile": """FROM python:3.9\nWORKDIR /app\nCOPY . .\nRUN pip install -r requirements.txt\nCMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]\n""",
    "docker/docker-compose.yml": """version: '3.8'\nservices:\n  api:\n    build: .\n    ports:\n      - "8000:8000"\n""",
    "infrastructure/aws-config.yaml": "# AWS configurations placeholder\n",
    "scripts/ingest_data.py": "# Data ingestion script placeholder\n"
}

# Create directories
for d in dirs:
    os.makedirs(d, exist_ok=True)

# Create files with default content
for file, content in files.items():
    with open(file, "w") as f:
        f.write(content)

print("✅ Project structure initialized successfully!")
