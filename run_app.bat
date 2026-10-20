@echo off
setlocal enabledelayedexpansion
REM ========================================================
REM FinReport RAG - Windows Run Script (Python 3.14+)
REM This script activates the environment and launches the app
REM ========================================================

echo.
echo ========================================
echo       FinReport RAG - Starting...
echo ========================================
echo.

REM 1. Check for local virtual environment (venv)
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment (.\venv)...
    call "venv\Scripts\activate.bat"
    goto :check_env
)

REM 2. Check for Conda environment (finrag_env)
where conda >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [INFO] Conda detected. Checking finrag_env...
    call conda activate finrag_env 2>nul
    if %ERRORLEVEL% equ 0 goto :check_env
)

if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" (
    call "%USERPROFILE%\anaconda3\condabin\conda.bat" activate finrag_env 2>nul
    if %ERRORLEVEL% equ 0 (
        echo [INFO] Activated finrag_env via Anaconda.
        goto :check_env
    )
)

REM 3. Fallback: Add Python 3.14 standard installation directory to PATH if present
if exist "C:\Python314\python.exe" (
    echo [INFO] Using Python 3.14 at C:\Python314
    set "PATH=C:\Python314;C:\Python314\Scripts;!PATH!"
)

:check_env
REM 4. Check if .env file exists
if not exist ".env" (
    echo.
    echo [WARNING] .env file not found!
    if exist ".env.example" (
        echo Copying .env.example to .env ...
        copy ".env.example" ".env" >nul
        echo [ACTION REQUIRED] Created .env file.
        echo Please edit .env with your OPENAI_API_KEY, then run this script again.
    ) else (
        echo Please create a .env file with your API keys.
    )
    echo.
    pause
    exit /b 1
)

REM 5. Verify Python runtime
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.14 or activate your virtual environment.
    echo.
    pause
    exit /b 1
)

REM 6. Run Streamlit via Python module
echo.
echo [INFO] Starting Streamlit app with Python 3.14...
echo [INFO] Access the app at: http://localhost:8501
echo.
python -m streamlit run app.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Streamlit encountered an error.
    echo If packages are missing, install them with:
    echo   pip install -r requirements.txt
    echo.
)

pause
