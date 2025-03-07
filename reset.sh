#!/bin/bash
# Script to reset the Ouro environment

# Define directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Cleaning up Ouro environment..."

# Remove virtual environment
if [ -d "$SCRIPT_DIR/venv" ]; then
    echo "Removing virtual environment..."
    rm -rf "$SCRIPT_DIR/venv"
fi

# Remove vector store and models
if [ -d "$SCRIPT_DIR/data/vector_store" ]; then
    echo "Removing vector store..."
    rm -rf "$SCRIPT_DIR/data/vector_store"
fi

if [ -d "$SCRIPT_DIR/data/models" ]; then
    echo "Removing cached models..."
    rm -rf "$SCRIPT_DIR/data/models"
fi

# Remove logs
if [ -d "$SCRIPT_DIR/logs" ]; then
    echo "Removing logs..."
    rm -rf "$SCRIPT_DIR/logs"
fi

# Remove __pycache__ directories
echo "Removing Python cache files..."
find "$SCRIPT_DIR" -type d -name "__pycache__" -exec rm -rf {} +
find "$SCRIPT_DIR" -type f -name "*.pyc" -delete

# Remove requirements installed flag
if [ -f "$SCRIPT_DIR/.requirements_installed" ]; then
    echo "Removing requirements installed flag..."
    rm -f "$SCRIPT_DIR/.requirements_installed"
fi

echo "Reset complete. Run './run.sh' to start fresh."