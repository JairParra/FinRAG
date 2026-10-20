@echo off
REM FinReport RAG - Windows Run Script
REM This script runs the Streamlit application

echo.
echo ================================
echo   FinReport RAG - Starting...
echo ================================
echo.

REM Check if virtual environment exists
if exist venv\ (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else if exist finrag_env\ (
    echo Activating conda environment...
    call conda activate finrag_env
)

REM Check if .env file exists
if not exist .env (
    echo.
    echo WARNING: .env file not found!
    echo Please create a .env file with your API keys.
    echo See README.md for instructions.
    echo.
    pause
    exit /b 1
)

REM Run Streamlit
echo.
echo Starting Streamlit app...
echo Access the app at: http://localhost:8501
echo.
streamlit run app.py

pause
