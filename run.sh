#!/bin/bash
# Script to run Ouro RAG system
# Usage:
#   ./run.sh                # Standard mode
#   ./run.sh --model small  # Specify model size
#   ./run.sh --help         # Show help information

# Define directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/venv"
MODEL="medium"  # Default model
HELP_FLAG=false

# Parse command line arguments
for arg in "$@"; do
  case $arg in
    --model=*)
      MODEL="${arg#*=}"
      shift
      ;;
    --model)
      if [[ "$2" == "small" || "$2" == "medium" || "$2" == "large" || "$2" == "very_large" || "$2" == "m1_optimized" ]]; then
        MODEL="$2"
        shift 2
      else
        echo "Invalid model specified. Using default: $MODEL"
        shift
      fi
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
  echo "  ./run.sh                   # Standard mode with medium model"
  echo "  ./run.sh --model small     # Use small model (1-2GB RAM)"
  echo "  ./run.sh --model medium    # Use medium model (4-8GB RAM)"
  echo "  ./run.sh --model large     # Use large model (8-12GB RAM)"
  echo "  ./run.sh --model very_large # Use very large model (12-16GB+ RAM)"
  echo "  ./run.sh --model m1_optimized # Optimized for Apple Silicon (M1/M2 Mac)"
  echo "  ./run.sh --help            # Show this help information"
  echo ""
  echo "See README.md for more detailed instructions."
  exit 0
fi

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install the package if needed
if [ ! -f "$SCRIPT_DIR/.installed" ]; then
    echo "Installing package..."
    pip install -e "$SCRIPT_DIR"
    touch "$SCRIPT_DIR/.installed"
else
    # Check if pyproject.toml has been modified since last install
    if [ "$SCRIPT_DIR/pyproject.toml" -nt "$SCRIPT_DIR/.installed" ]; then
        echo "Package has been updated. Reinstalling..."
        pip install -e "$SCRIPT_DIR"
        touch "$SCRIPT_DIR/.installed"
    fi
fi

# Check if Hugging Face CLI is installed
if ! python -c "import huggingface_hub" &> /dev/null; then
    echo "Hugging Face Hub not found. Installing..."
    pip install huggingface_hub
fi

# Check if user is logged in to Hugging Face (now optional)
if [ ! -f ~/.huggingface/token ]; then
    echo "==============================================================="
    echo "NOTICE: You're not logged in to Hugging Face."
    echo "While Ouro will continue to work, logging in is recommended"
    echo "for better access to models and to avoid download issues."
    echo ""
    echo "To log in, run: huggingface-cli login"
    echo "==============================================================="
    # Continue without exiting - authentication is now optional
fi

# Create necessary directories
mkdir -p "$SCRIPT_DIR/ouro/data/documents" "$SCRIPT_DIR/ouro/data/models" "$SCRIPT_DIR/ouro/data/vector_store" "$SCRIPT_DIR/ouro/logs" "$SCRIPT_DIR/ouro/data/conversations"

# Set environment variables for optimization
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Apple Silicon specific optimizations
if [[ "$(uname -m)" == "arm64" && "$(uname)" == "Darwin" ]]; then
    echo "Applying Apple Silicon (M1/M2) optimizations..."
    export PYTORCH_ENABLE_MPS_FALLBACK=1
fi

# Run Ouro with the specified model
echo "Starting Ouro with model: $MODEL"
python -m ouro --model $MODEL