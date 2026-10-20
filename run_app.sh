#!/bin/bash
# ========================================================
# FinReport RAG - Linux / macOS / WSL Run Script
# ========================================================

echo ""
echo "========================================"
echo "      FinReport RAG - Starting..."
echo "========================================"
echo ""

# 1. Activate virtual environment if present
if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    echo "[INFO] Activating virtual environment (venv)..."
    source venv/bin/activate
elif command -v conda &> /dev/null; then
    echo "[INFO] Conda detected. Activating finrag_env..."
    eval "$(conda shell.bash hook 2> /dev/null)"
    conda activate finrag_env 2> /dev/null || echo "[WARNING] Could not activate finrag_env conda environment."
fi

# 2. Check if .env file exists
if [ ! -f ".env" ]; then
    echo ""
    echo "[WARNING] .env file not found!"
    if [ -f ".env.example" ]; then
        echo "Creating .env from .env.example..."
        cp .env.example .env
        echo "[ACTION REQUIRED] Created .env file. Please update it with your OPENAI_API_KEY."
    fi
    exit 1
fi

# 3. Check Streamlit installation
if ! command -v streamlit &> /dev/null; then
    echo "[ERROR] Streamlit is not found in the current environment."
    echo "Please activate your environment and install dependencies:"
    echo "  conda activate finrag_env"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# 4. Run Streamlit
echo ""
echo "[INFO] Starting Streamlit app..."
echo "[INFO] Access the app at: http://localhost:8501"
echo ""
streamlit run app.py
