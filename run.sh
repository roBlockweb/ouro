#!/bin/bash
# Unified script to run Ouro RAG system with various options
# Usage:
#   ./run.sh                # Standard interactive mode
#   ./run.sh --small        # Use Small model preset
#   ./run.sh --fast         # Use Fast mode for quicker responses
#   ./run.sh --m1           # Optimized for Apple Silicon (M1/M2)
#   ./run.sh --no-history   # Don't use conversation history
#   ./run.sh --no-prompts   # Non-interactive mode with Small model
#   ./run.sh --help         # Show help information

# Define directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/venv"
SMALL_MODEL_FLAG=false
FAST_MODE_FLAG=false
M1_OPTIMIZED_FLAG=false
NO_HISTORY_FLAG=false
NO_PROMPTS_FLAG=false
HELP_FLAG=false

# Check for Apple Silicon
if [[ "$(uname -m)" == "arm64" && "$(uname)" == "Darwin" ]]; then
    IS_APPLE_SILICON=true
else
    IS_APPLE_SILICON=false
fi

# Parse command line arguments
for arg in "$@"; do
  case $arg in
    --small)
      SMALL_MODEL_FLAG=true
      shift
      ;;
    --fast)
      FAST_MODE_FLAG=true
      shift
      ;;
    --m1)
      M1_OPTIMIZED_FLAG=true
      shift
      ;;
    --no-history)
      NO_HISTORY_FLAG=true
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
  echo "  ./run.sh                # Standard interactive mode"
  echo "  ./run.sh --small        # Use Small model preset (1.1B parameters, ~2GB RAM)"
  echo "  ./run.sh --fast         # Use Fast mode for quicker responses"
  echo "  ./run.sh --m1           # Optimized for Apple Silicon (M1/M2 Mac)"
  echo "  ./run.sh --no-history   # Don't use conversation history"
  echo "  ./run.sh --no-prompts   # Non-interactive mode with Small model"
  echo "  ./run.sh --help         # Show this help information"
  echo ""
  if [ "$IS_APPLE_SILICON" = true ]; then
    echo "Apple Silicon (M1/M2) detected! For best performance, use --m1 flag."
  fi
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

# Set environment variables to optimize performance
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Apple Silicon specific optimizations
if [ "$IS_APPLE_SILICON" = true ]; then
    echo "Applying Apple Silicon (M1/M2) optimizations..."
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    
    # Use better defaults for Apple Silicon
    if [ "$M1_OPTIMIZED_FLAG" = true ]; then
        export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
        export PYTORCH_MPS_ALLOCATOR_POLICY=4  # For better memory management
    fi
fi

# Create necessary directories
mkdir -p "$SCRIPT_DIR/data/documents" "$SCRIPT_DIR/data/models" "$SCRIPT_DIR/data/vector_store" "$SCRIPT_DIR/logs" "$SCRIPT_DIR/data/conversations"

# Create command line args based on flags
ARGS=""

if [ "$SMALL_MODEL_FLAG" = true ]; then
    ARGS="$ARGS --model Small"
fi

if [ "$M1_OPTIMIZED_FLAG" = true ] && [ "$IS_APPLE_SILICON" = true ]; then
    ARGS="$ARGS --model M1Optimized"
fi

if [ "$FAST_MODE_FLAG" = true ]; then
    ARGS="$ARGS --fast"
fi

if [ "$NO_HISTORY_FLAG" = true ]; then
    ARGS="$ARGS --no-history"
fi

# Check for no-prompts mode (non-interactive)
if [ "$NO_PROMPTS_FLAG" = true ]; then
    echo "Starting Ouro in non-interactive mode with Small model preset..."
    cd "$SCRIPT_DIR"
    python -c "
from src.llm import login_huggingface, load_model_with_progress
from src.embeddings import load_embedding_with_progress
from src.config import MODEL_CONFIGURATIONS, EMBEDDING_MODELS, DEFAULT_QUANTIZE, DEFAULT_FAST_MODE
from src.rag import OuroRAG
from rich.console import Console
import os

console = Console()
console.print('[bold green]Ouro[/bold green]: Initializing in non-interactive mode')

# Select Small model configuration
model_config = MODEL_CONFIGURATIONS['Small']
model_path = model_config['llm']
embedding_model = EMBEDDING_MODELS[model_config['embedding']]
quantize = model_config.get('quantize', DEFAULT_QUANTIZE)
fast_mode = True  # Always use fast mode in non-interactive

console.print(f'Using model: [blue]{model_path}[/blue]')
console.print(f'Using embedding: [yellow]{embedding_model}[/yellow]')
console.print(f'Optimizations: Fast Mode, Quantization: {quantize}')

# Authenticate without interaction
login_huggingface()

# Load components
console.print('Loading embedding model...')
embedding_manager = load_embedding_with_progress(embedding_model)

console.print('Loading language model...')
llm, _ = load_model_with_progress(model_path, quantize=quantize, fast_mode=fast_mode)

# Initialize RAG system
rag = OuroRAG(
    embedding_manager=embedding_manager,
    llm=llm,
    config=model_config,
    memory_turns=3,  # Use minimal memory in non-interactive mode
    save_conversations=False  # Don't save conversations in non-interactive mode
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

## Apple Silicon Optimization

When running on Apple M1/M2 chips, Ouro uses the Metal Performance Shaders (MPS) 
backend for PyTorch to accelerate matrix operations and inference.
''')

console.print('[green]✓[/green] Ouro initialized successfully with Small model')
console.print('[green]✓[/green] Sample document is available at data/documents/sample.txt')
console.print()
console.print('To use Ouro interactively, run: ./run.sh')
console.print('For better Apple Silicon performance, try: ./run.sh --m1')
console.print('For troubleshooting, see GUIDE.md')
"

# Interactive mode with command line args
else
    echo "Starting Ouro with args: $ARGS"
    cd "$SCRIPT_DIR"
    PYTHONPATH="$SCRIPT_DIR" python -m src.main $ARGS
fi

# Deactivate virtual environment when done
deactivate