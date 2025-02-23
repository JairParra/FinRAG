```
FnReport-RAG/
│── backend/                     # Core backend folder
│   ├── app/
│   │   ├── __init__.py          # Marks directory as a Python package
│   │   ├── main.py              # FastAPI application entry point
│   │   ├── config.py            # Configuration settings (e.g., env variables)
│   │   ├── dependencies.py      # Dependency injection for API components
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── api.py           # API route for queries
│   │   │   ├── health.py        # Health check endpoint
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── rag.py           # Retrieval-Augmented Generation logic
│   │   │   ├── processing.py    # PDF/Text processing utilities
│   │   │   ├── embeddings.py    # Vector embedding logic
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── document.py      # Data models for financial reports
│   │   │   ├── query.py         # Query-related data models
│   ├── tests/
│   │   ├── test_api.py          # API test cases
│   │   ├── test_rag.py          # RAG implementation tests
│── frontend/                    # Frontend repo (optional, handled separately)
│── docker/
│   ├── Dockerfile               # Docker container instructions
│   ├── docker-compose.yml       # Docker orchestration
│── infrastructure/              # Infrastructure setup (AWS Deployment)
│   ├── terraform/               # (Optional) Terraform scripts for AWS infra
│   ├── aws-config.yaml          # AWS configurations (Lambda, ECS, etc.)
│── data/                        # Folder for sample financial reports
│   ├── sample_report.pdf
│── scripts/                     # Helper scripts for local setup and ingestion
│   ├── ingest_data.py           # Script to process and store financial data
│── .env                         # Environment variables
│── .gitignore                   # Git ignore file
│── README.md                    # Documentation
│── requirements.txt             # Python dependencies
│── setup.py                     # Installable package setup
│── Makefile                     # Command shortcuts (e.g., `make run`)
```