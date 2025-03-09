# Ouro: Privacy-First Local RAG System (v1.0.2)

Ouro is a privacy-focused Retrieval-Augmented Generation (RAG) system that runs completely offline on your local machine. It allows you to interact with your documents using AI without sending data to external services.

## Features

- **100% Private & Offline**: All processing happens on your machine
- **Dual Interface**: Choose between a terminal or web interface
- **Document Processing**: Support for TXT, PDF, Markdown, CSV, HTML, and JSON files
- **Vector Search**: Efficient similarity search to find relevant information
- **Conversation Memory**: Maintains context across multiple turns
- **Apple Silicon Optimized**: Special configurations for M1/M2 Macs
- **Minimal & User-Friendly**: Simple, intuitive operation without unnecessary complexity
- **Local API**: Optional REST endpoints for automation with tools like n8n.io
- **Easy Maintenance**: Simple cleanup and maintenance scripts included

## What's New in v1.0.2

- **Auto-Launch Browser**: Web UI now automatically opens in the default browser
- **Hallucination Prevention**: Completely redesigned prompt system to eliminate fake conversations
- **Aggressive Response Cleaning**: New filtering system to ensure high-quality, direct responses
- **Enhanced User Experience**: Much more natural conversation flow, similar to ChatGPT
- **Explicit Instruction System**: Stronger guardrails to prevent the model from generating unwanted content

## What's New in v1.0.0

- **Streamlined Authentication**: No longer requires mandatory Hugging Face login
- **Easy Reset**: Comprehensive cleanup scripts for fresh installations
- **Improved Installation**: Robust directory creation and error handling
- **Better Model Configuration**: Optimized presets for different hardware capabilities
- **Enhanced Documentation**: Clear setup and maintenance instructions

## Requirements

- **Python 3.9+** (3.10 or 3.11 recommended)
- **Hugging Face Account**: Recommended but not required
- **RAM Requirements**:
  - Small model: 2-4GB RAM
  - Medium model: 4-8GB RAM
  - Large model: 8-12GB RAM
  - Very Large model: 12-16GB+ RAM
- **Storage**: ~10GB+ for models and vector database

## Quick Start

### 1. Install Ouro

```bash
# Clone the repository
git clone https://github.com/roBlockWeb/ouro.git
cd ouro

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -e .
```

### 2. Hugging Face Authentication (Recommended)

While Ouro will work without authentication, logging in to Hugging Face is recommended for better model access and to avoid download rate limits:

```bash
huggingface-cli login
```

Follow the prompts to authenticate with your Hugging Face account. This creates a token at `~/.huggingface/token`.

**Ouro will now proceed regardless of login status, but you may encounter model download issues if not authenticated.**

### 3. Run Ouro

```bash
# Start Ouro with default settings
python -m ouro

# Or run with specific model size
python -m ouro --model medium
```

On startup, you'll be prompted to choose between:
- **Terminal Interface**: Classic command-line experience
- **Web Interface**: Browser-based UI (default after 20 seconds)

## Usage

### Web Interface

The web interface provides a user-friendly way to interact with Ouro:

1. **Chat**: Ask questions about your documents in the main chat area
2. **Settings**: Configure model, retrieval parameters, and generation settings
3. **Document Ingestion**: Upload files or paste text directly

Access the web interface at: http://localhost:7860

### Terminal Interface

The terminal interface offers a command-line experience with these commands:

- `/ingest <file_path>` - Ingest a document
- `/ingest_dir <directory_path>` - Ingest all documents in a directory
- `/ingest_text` - Ingest text directly (follow prompts)
- `/models` - List available models
- `/change_model <model_name>` - Switch models
- `/clear_memory` - Clear conversation history
- `/learn` - Learn from past conversations
- `/help` - Show help information
- `/exit` or `/quit` - Exit the application

### Local API

Ouro exposes REST endpoints that can be used by other applications:

- **POST /api/v1/query**: Query the RAG system
  ```bash
  curl -X POST http://localhost:7860/api/v1/query \
    -H "Content-Type: application/json" \
    -d '{"query": "What is RAG?", "use_history": true}'
  ```

- **POST /api/v1/ingest**: Ingest text directly
  ```bash
  curl -X POST http://localhost:7860/api/v1/ingest \
    -H "Content-Type: application/json" \
    -d '{"text": "RAG stands for Retrieval-Augmented Generation."}'
  ```

## Configuration Options

Edit `ouro/config.py` to customize settings:

- **Models**: Change default models or add new ones
- **Vector Store**: Modify embedding models, chunk size, and retrieval parameters
- **Web UI**: Change host and port for the web interface
- **API Settings**: Enable/disable API endpoints
- **System Prompt**: Customize the instruction to the LLM

## Advanced Usage

### Running on Apple Silicon (M1/M2)

Ouro includes optimized settings for Apple Silicon:

```bash
python -m ouro --model m1_optimized
```

### Enabling Quantization

On Windows/Linux, Ouro automatically uses 4-bit quantization when supported by your hardware.

### Fine-tuning on Your Data

The `/learn` command initiates a lightweight fine-tuning process using your conversation history.

### Cleaning Up Your Installation

If you need to reset Ouro to a clean state (remove all models, vector stores, logs, and user data):

```bash
# On macOS/Linux:
./final_cleanup.sh

# On Windows:
python cleanup.py
```

This is useful when:
- You want to start fresh with no previous data
- You're preparing to upgrade Ouro
- You're troubleshooting issues and want a clean state

## Troubleshooting

### Common Issues

1. **Model Download Failures**:
   - If you encounter model download issues, authenticate with `huggingface-cli login`
   - Check your internet connection
   - Verify you have enough disk space
   - Some models require explicit acceptance of terms on the Hugging Face website

2. **Out of Memory Errors**:
   - Try a smaller model: `python -m ouro --model small`
   - Reduce `max_new_tokens` in config.py

3. **Web Interface Not Loading**:
   - Ensure dependencies are installed: `pip install fastapi uvicorn jinja2 python-multipart`
   - Check if another application is using port 7860

## License

Ouro is released under the MIT License. See the LICENSE file for details.

## Acknowledgments

Ouro is built using these amazing open-source projects:
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [LangChain](https://github.com/langchain-ai/langchain)
- [FastAPI](https://github.com/tiangolo/fastapi)
- [FAISS](https://github.com/facebookresearch/faiss)
