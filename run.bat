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
    
    REM Install the package in development mode
    echo Installing package in development mode...
    python -m pip install -e .
    echo. > .requirements_installed
) else (
    REM Check if requirements.txt has been modified
    for /f %%i in ('dir /b /o:d requirements.txt .requirements_installed') do set NEWEST=%%i
    if "%NEWEST%"=="requirements.txt" (
        echo Requirements have been updated. Installing new dependencies...
        python -m pip install -r requirements.txt
        echo. > .requirements_installed
    )
)

REM Set environment variables for better performance
set OMP_NUM_THREADS=1
set TOKENIZERS_PARALLELISM=false

REM Create necessary directories
if not exist data\documents mkdir data\documents
if not exist data\models mkdir data\models
if not exist data\vector_store mkdir data\vector_store
if not exist data\conversations mkdir data\conversations
if not exist logs mkdir logs

REM Print help information if --help is provided
echo %* | findstr /C:"--help" >nul
if %ERRORLEVEL% EQU 0 (
    echo Ouro: Privacy-First Local RAG System
    echo.
    echo Usage:
    echo   run.bat                  - Standard interactive mode
    echo   run.bat --small          - Use Small model preset (1.1B parameters, ~2GB RAM)
    echo   run.bat --fast           - Use Fast mode for quicker responses
    echo   run.bat --no-history     - Don't use conversation history
    echo   run.bat --help           - Show this help information
    echo.
    echo See GUIDE.md for more detailed instructions.
    call venv\Scripts\deactivate.bat
    pause
    exit /b
)

REM Run the application with all arguments passed to the script
echo Starting Ouro...
python -m src.main %*

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat

echo.
pause