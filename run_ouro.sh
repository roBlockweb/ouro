#!/bin/bash
# Helper script to run Ouro system with the virtual environment

# Activate the virtual environment
source venv/bin/activate

# Check if FAISS index exists, initialize if needed
if [ ! -f "faiss_store/index.faiss" ]; then
    echo "FAISS index not found, initializing..."
    python initialize_faiss.py
fi

# Run the system in autonomous mode by default
if [ -z "$1" ]; then
    echo "Running in autonomous mode (default)"
    python main.py --mode=autonomous
else
    # If argument provided, use that mode
    echo "Running in $1 mode"
    python main.py --mode=$1
fi