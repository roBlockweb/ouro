@echo off
REM Cleanup script for Ouro RAG system on Windows
REM This script removes all generated files and directories, resetting the project to its initial state.

echo ============================================================
echo                 Ouro RAG System Cleanup
echo ============================================================
echo This script will remove all generated files and directories,
echo resetting the project to its initial state.
echo ============================================================
echo.

echo WARNING: This will delete all:
echo - Downloaded models
echo - Embeddings and vector stores
echo - Logs and conversation history
echo - User documents and uploads
echo - Cache files and temporary data
echo.
echo This action cannot be undone!
echo.

set /p CONFIRM="Are you sure you want to proceed? (y/N): "
if /i not "%CONFIRM%"=="y" (
    echo.
    echo Cleanup cancelled.
    goto :EOF
)

echo.
echo Starting deep cleanup...
echo.

REM Get the script directory
set SCRIPT_DIR=%~dp0
cd %SCRIPT_DIR%

REM Define function to remove directories
echo Removing data directories...
if exist "%SCRIPT_DIR%data" rmdir /s /q "%SCRIPT_DIR%data"
if exist "%SCRIPT_DIR%ouro\data" rmdir /s /q "%SCRIPT_DIR%ouro\data"

echo Removing log files...
if exist "%SCRIPT_DIR%logs" rmdir /s /q "%SCRIPT_DIR%logs"
if exist "%SCRIPT_DIR%ouro\logs" rmdir /s /q "%SCRIPT_DIR%ouro\logs"

echo Removing installation markers...
if exist "%SCRIPT_DIR%.installed" del /f /q "%SCRIPT_DIR%.installed"
if exist "%SCRIPT_DIR%ouro.egg-info" rmdir /s /q "%SCRIPT_DIR%ouro.egg-info"

echo Removing Python cache files...
for /d /r "%SCRIPT_DIR%" %%d in (__pycache__) do (
    if exist "%%d" rmdir /s /q "%%d"
)
del /s /q "%SCRIPT_DIR%\*.pyc" 2>nul
del /s /q "%SCRIPT_DIR%\*.pyo" 2>nul
del /s /q "%SCRIPT_DIR%\*.pyd" 2>nul

echo Removing temporary uploads...
if exist "%SCRIPT_DIR%ouro\uploads" rmdir /s /q "%SCRIPT_DIR%ouro\uploads"

echo.
set /p REMOVE_VENV="Do you also want to remove the virtual environment? (y/N): "
if /i "%REMOVE_VENV%"=="y" (
    echo Removing virtual environment...
    if exist "%SCRIPT_DIR%venv" rmdir /s /q "%SCRIPT_DIR%venv"
)

echo.
echo Cleanup completed successfully!
echo The project has been reset to its initial state.
echo.
echo To reinstall Ouro, run:
echo   python install.py
echo.

pause