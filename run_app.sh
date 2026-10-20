#!/bin/bash
# FinReport RAG - Linux/Mac Run Script
# This script runs the Streamlit application

echo ""
echo "================================"
echo "  FinReport RAG - Starting..."
echo "================================"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d "finrag_env" ]; then
    echo "Activating conda environment..."
    conda activate finrag_env
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo ""
    echo "WARNING: .env file not found!"
    echo "Please create a .env file with your API keys."
    echo "See README.md for instructions."
    echo ""
    exit 1
fi

# Run Streamlit
echo ""
echo "Starting Streamlit app..."
echo "Access the app at: http://localhost:8501"
echo ""
streamlit run app.py
