#!/bin/bash
# Unified script to run Ouro RAG system with various options
# Usage:
#   ./run.sh              # Standard interactive mode
#   ./run.sh --small      # Use Small model preset
#   ./run.sh --no-prompts # Non-interactive mode with Small model
#   ./run.sh --help       # Show help information

# Define directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/venv"
SMALL_MODEL_FLAG=false
NO_PROMPTS_FLAG=false
HELP_FLAG=false

# Parse command line arguments
for arg in "$@"; do
  case $arg in
    --small)
      SMALL_MODEL_FLAG=true
      shift
      ;;
    --no-prompts)
      NO_PROMPTS_FLAG=true
      SMALL_MODEL_FLAG=true # No-prompts mode always uses Small model
      shift
      ;;
    --help)
      HELP_FLAG=true
      shift
      ;;
    *)
      # Unknown option
      ;;
  esac
done

# Show help if requested
if [ "$HELP_FLAG" = true ]; then
  echo "Ouro: Privacy-First Local RAG System"
  echo ""
  echo "Usage:"
  echo "  ./run.sh              # Standard interactive mode"
  echo "  ./run.sh --small      # Use Small model preset for lower memory usage"
  echo "  ./run.sh --no-prompts # Non-interactive mode with Small model"
  echo "  ./run.sh --help       # Show this help information"
  echo ""
  echo "See GUIDE.md for more detailed instructions."
  exit 0
fi

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
else
    # Check if the requirements file has been modified since last install
    if [ "$SCRIPT_DIR/requirements.txt" -nt "$SCRIPT_DIR/.requirements_installed" ]; then
        echo "Requirements have been updated. Installing new dependencies..."
        pip install -r "$SCRIPT_DIR/requirements.txt"
        touch "$SCRIPT_DIR/.requirements_installed"
    fi
fi

# Set environment variables to prevent memory issues
# These help with Python 3.13 compatibility
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Create necessary directories
mkdir -p "$SCRIPT_DIR/data/documents" "$SCRIPT_DIR/data/models" "$SCRIPT_DIR/data/vector_store" "$SCRIPT_DIR/logs"

# Check for small model mode (more efficient than standard)
if [ "$SMALL_MODEL_FLAG" = true ] && [ "$NO_PROMPTS_FLAG" = false ]; then
    echo "Starting Ouro with Small model preset..."
    cd "$SCRIPT_DIR"
    python -c "
from src.llm import login_huggingface, load_model_with_progress
from src.embeddings import load_embedding_with_progress
from src.config import MODEL_CONFIGURATIONS, EMBEDDING_MODELS
from src.rag import OuroRAG
from src.main import main
import sys

# Set the default model to Small
if 'Small' not in MODEL_CONFIGURATIONS:
    print('Error: Small model configuration not found.')
    sys.exit(1)

# Preset Small model to be selected automatically
sys.argv = ['main.py', '--model', 'Small']

# Run the main application
main()
"
    
# Check for no-prompts mode (non-interactive)
elif [ "$NO_PROMPTS_FLAG" = true ]; then
    echo "Starting Ouro in non-interactive mode with Small model preset..."
    cd "$SCRIPT_DIR"
    python -c "
from src.llm import login_huggingface, load_model_with_progress
from src.embeddings import load_embedding_with_progress
from src.config import MODEL_CONFIGURATIONS, EMBEDDING_MODELS
from src.rag import OuroRAG
from rich.console import Console
import os

console = Console()
console.print('[bold green]Ouro[/bold green]: Initializing in non-interactive mode')

# Select Small model configuration
model_config = MODEL_CONFIGURATIONS['Small']
model_path = model_config['llm']
embedding_model = EMBEDDING_MODELS[model_config['embedding']]

console.print(f'Using model: [blue]{model_path}[/blue]')
console.print(f'Using embedding: [yellow]{embedding_model}[/yellow]')

# Authenticate without interaction
login_huggingface()

# Load components
console.print('Loading embedding model...')
embedding_manager = load_embedding_with_progress(embedding_model)

console.print('Loading language model...')
llm, _ = load_model_with_progress(model_path)

# Initialize RAG system
rag = OuroRAG(
    embedding_manager=embedding_manager,
    llm=llm,
    config=model_config
)

# Check if sample.txt exists, if not create it
sample_path = os.path.join('data/documents', 'sample.txt')
if not os.path.exists(sample_path):
    console.print('[yellow]Sample document not found. Creating a basic one...[/yellow]')
    with open(sample_path, 'w') as f:
        f.write('''# Ouro Sample Document
        
This is a sample document to test the Ouro RAG system.

## Retrieval-Augmented Generation (RAG)

RAG is a technique that combines retrieval-based and generation-based approaches 
to improve text generation. It first finds relevant information from a knowledge base,
and then uses this information to produce a response.
''')

console.print('[green]✓[/green] Ouro initialized successfully with Small model')
console.print('[green]✓[/green] Sample document is available at data/documents/sample.txt')
console.print()
console.print('To use Ouro interactively, run: ./run.sh')
console.print('For troubleshooting, see GUIDE.md')
"

# Standard mode (interactive with model selection)
else
    echo "Starting Ouro..."
    cd "$SCRIPT_DIR"
    PYTHONPATH="$SCRIPT_DIR" python -m src.main

fi

# Deactivate virtual environment when done
deactivate