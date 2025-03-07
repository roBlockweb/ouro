@echo off
REM Ouro Windows launcher script

echo Setting up Ouro - Privacy-First Local RAG System...

REM Check for Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in your PATH.
    echo Please install Python 3.9 or newer from https://www.python.org/downloads/
    pause
    exit /b
)

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install requirements if needed
if not exist .requirements_installed (
    echo Installing requirements...
    python -m pip install -r requirements.txt
    echo. > .requirements_installed
)

REM Set environment variables to prevent memory issues
set PYTORCH_ENABLE_MPS_FALLBACK=1
set OMP_NUM_THREADS=1
set TOKENIZERS_PARALLELISM=false

REM Run the application
echo Starting Ouro...
python -m src.main %*

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat

echo.
pause