#!/bin/bash
# Script to run Ouro RAG system

# Define directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/venv"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Check if requirements are installed
if [ ! -f "$SCRIPT_DIR/.requirements_installed" ]; then
    echo "Installing requirements..."
    pip install -r "$SCRIPT_DIR/requirements.txt"
    
    # Install the package in development mode
    echo "Installing package in development mode..."
    pip install -e "$SCRIPT_DIR"
    
    touch "$SCRIPT_DIR/.requirements_installed"
fi

# Run the application
echo "Starting Ouro..."
cd "$SCRIPT_DIR"
PYTHONPATH="$SCRIPT_DIR" python -m src.main "$@"

# Deactivate virtual environment when done
deactivate