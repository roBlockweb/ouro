@echo off
REM Script to run Ouro RAG system on Windows
REM Usage:
REM   run.bat                 # Standard mode
REM   run.bat --model small   # Specify model size
REM   run.bat --help          # Show help information

setlocal

set SCRIPT_DIR=%~dp0
set VENV_DIR=%SCRIPT_DIR%venv
set MODEL=medium

REM Check for help flag
echo %* | findstr /C:"--help" >nul
if %ERRORLEVEL% EQU 0 (
    echo Ouro: Privacy-First Local RAG System
    echo.
    echo Usage:
    echo   run.bat                   # Standard mode with medium model
    echo   run.bat --model small     # Use small model (1-2GB RAM)
    echo   run.bat --model medium    # Use medium model (4-8GB RAM)
    echo   run.bat --model large     # Use large model (8-12GB RAM)
    echo   run.bat --model very_large # Use very large model (12-16GB+ RAM)
    echo   run.bat --help            # Show this help information
    echo.
    echo See README.md for more detailed instructions.
    goto :eof
)

REM Check for model flag
echo %* | findstr /C:"--model" >nul
if %ERRORLEVEL% EQU 0 (
    for %%a in (%*) do (
        if "%%a"=="--model" (
            set MODEL_FLAG=1
        ) else if defined MODEL_FLAG (
            if "%%a"=="small" set MODEL=small
            if "%%a"=="medium" set MODEL=medium
            if "%%a"=="large" set MODEL=large
            if "%%a"=="very_large" set MODEL=very_large
            if "%%a"=="m1_optimized" set MODEL=m1_optimized
            set MODEL_FLAG=
        )
    )
)

REM Check for Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in your PATH.
    echo Please install Python 3.9 or newer from https://www.python.org/downloads/
    pause
    exit /b
)

REM Create virtual environment if it doesn't exist
if not exist "%VENV_DIR%" (
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
)

REM Activate virtual environment
call "%VENV_DIR%\Scripts\activate.bat"

REM Install the package if needed
if not exist "%SCRIPT_DIR%.installed" (
    echo Installing package...
    pip install -e "%SCRIPT_DIR%"
    echo. > "%SCRIPT_DIR%.installed"
)

REM Check if pyproject.toml has been modified
for /f %%i in ('dir /b /o:d "%SCRIPT_DIR%pyproject.toml" "%SCRIPT_DIR%.installed"') do set NEWEST=%%i
if "%NEWEST%"=="pyproject.toml" (
    echo Package has been updated. Reinstalling...
    pip install -e "%SCRIPT_DIR%"
    echo. > "%SCRIPT_DIR%.installed"
)

REM Check if Hugging Face CLI is installed
python -c "import huggingface_hub" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Hugging Face Hub not found. Installing...
    pip install huggingface_hub
)

REM Check if user is logged in to Hugging Face (now optional)
if not exist "%USERPROFILE%\.huggingface\token" (
    echo ===============================================================
    echo NOTICE: You're not logged in to Hugging Face.
    echo While Ouro will continue to work, logging in is recommended
    echo for better access to models and to avoid download issues.
    echo.
    echo To log in, run: huggingface-cli login
    echo ===============================================================
    echo Press any key to continue anyway...
    pause > nul
    REM Continue without exiting - authentication is now optional
)

REM Create necessary directories
if not exist "%SCRIPT_DIR%ouro\data\documents" mkdir "%SCRIPT_DIR%ouro\data\documents"
if not exist "%SCRIPT_DIR%ouro\data\models" mkdir "%SCRIPT_DIR%ouro\data\models"
if not exist "%SCRIPT_DIR%ouro\data\vector_store" mkdir "%SCRIPT_DIR%ouro\data\vector_store"
if not exist "%SCRIPT_DIR%ouro\logs" mkdir "%SCRIPT_DIR%ouro\logs"
if not exist "%SCRIPT_DIR%ouro\data\conversations" mkdir "%SCRIPT_DIR%ouro\data\conversations"

REM Set environment variables for optimization
set OMP_NUM_THREADS=1
set TOKENIZERS_PARALLELISM=false

REM Run Ouro with the specified model
echo Starting Ouro with model: %MODEL%
python -m ouro --model %MODEL%

REM Deactivate virtual environment when done
call "%VENV_DIR%\Scripts\deactivate.bat"

echo.
pause
endlocal