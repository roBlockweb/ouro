# Ouro: Privacy-First Local RAG System (v3.0.0)

Ouro is a privacy-focused Retrieval-Augmented Generation (RAG) system that runs completely offline on your local machine. It allows you to interact with your documents using AI without sending data to external services.

## Features

- **100% Private & Offline**: All processing happens on your machine
- **Streamlined Terminal Interface**: Clean command-line experience with tab completion
- **Casual Chat Mode**: Dedicated chat environment for conversational interactions
- **Document Processing**: Support for TXT, PDF, Markdown, CSV, HTML, and JSON files
- **Vector Search**: Efficient similarity search to find relevant information
- **Conversation Memory**: Maintains context across multiple turns with short and long-term memory
- **Agent Capabilities**: Solve complex tasks using reasoning and specialized tools
- **Web Search Integration**: Optional web search for up-to-date information
- **Python API**: Local Python API for integration with other applications
- **Apple Silicon Optimized**: Special configurations for M1/M2 Macs
- **Ollama Integration**: Optional integration with Ollama for additional models

## What's New in v3.0.0

- **Terminal-Only Focus**: Streamlined experience without web dependencies
- **Command Autocomplete**: Tab completion for slash commands
- **Casual Chat Mode**: Dedicated environment for natural conversations
- **Command Structure**: Enhanced command handling with improved help system
- **Python API**: New programmatic interface for application integration
- **Improved Memory**: Enhanced short and long-term memory capabilities
- **Simplified Configuration**: Cleaner configuration without web UI settings
- **Enhanced Documentation**: Updated guides for all features
- **Refactored Code**: Cleaner, more maintainable codebase

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

## Terminal Interface

Ouro provides a streamlined terminal interface with these commands:

- `/chat [topic]` - Start casual chat mode (optional topic)
- `/ingest <file_path>` - Ingest a document
- `/ingest_dir <directory_path>` - Ingest all documents in a directory
- `/ingest_text` - Ingest text directly (follow prompts)
- `/models` - List available models
- `/change_model <model_name>` - Switch models
- `/clear_memory` - Clear conversation history
- `/learn` - Learn from past conversations
- `/help` - Show help information
- `/exit` or `/quit` - Exit the application

### Command Autocomplete

Ouro supports tab completion for slash commands on Unix-like systems:

1. Type `/` to begin a command
2. Press the `Tab` key to see available command options or autocomplete the current command

### Chat Mode

The `/chat` command offers a dedicated chat environment:

- Start with `/chat` for general conversation
- Use `/chat topic` to start with a specific topic
- Type `exit` or `quit` to return to command mode
- Use basic commands within chat mode

```
>> /chat
Starting casual chat mode. Type 'exit' to return to command mode.
Chat>> Tell me about yourself
I'm Ouro, a privacy-focused AI assistant that runs completely on your local machine...

Chat>> What's your favorite color?
As an AI, I don't have personal preferences in the same way humans do...

Chat>> /clear_memory
✓ Conversation memory cleared

Chat>> exit
Exiting chat mode.
```

## Document Management

Ouro can process various document types and store their content for retrieval.

### Supported File Types

- **PDF** (.pdf) - Full PDF support including text extraction
- **Plain Text** (.txt) - Standard text files
- **Markdown** (.md, .markdown) - Markdown formatting
- **CSV** (.csv) - Comma-separated values
- **HTML** (.html, .htm) - Web pages
- **JSON** (.json) - Structured data

### Python API for Document Management

You can programmatically ingest documents using the Python API:

```python
from ouro.api import create_ouro_client

client = create_ouro_client()

# Ingest a document
result = client.ingest_document(
    file_path="/path/to/document.pdf",
    metadata={"source": "research", "author": "John Doe"}
)
print(f"Ingested {result['ingested_documents']} chunks")
```

## API Integration

Ouro provides a Python API for integration with other applications:

```python
from ouro.api import create_ouro_client

# Create client
client = create_ouro_client(model_name="medium")

# Simple query
response = client.query("What is RAG?")
print(response["response"])

# Ingest text
result = client.ingest_text(
    text="RAG stands for Retrieval Augmented Generation, a technique that enhances LLM responses with retrieved knowledge.",
    metadata={"source": "definition", "category": "ai_concepts"}
)
print(f"Ingested {result['ingested_documents']} chunks")
```

### API Functionality

The API provides the following capabilities:

- **Querying**: Get responses to questions
- **Text Ingestion**: Add text to the knowledge base
- **Document Ingestion**: Process and add documents
- **Memory Management**: Clear or retrieve conversation memory
- **Model Management**: Change active models
- **Token Counting**: Count tokens in text
- **System Statistics**: Get information about the system

## Configuration Options

Edit `ouro/config.py` to customize settings:

- **Models**: Change default models or add new ones
- **Vector Store**: Modify embedding models, chunk size, and retrieval parameters
- **Memory Settings**: Configure short-term and long-term memory behavior
- **Agent Settings**: Enable/disable specific tools and reasoning capabilities
- **Web Search**: Configure web search providers and result limits
- **Logging Options**: Control structured logging and output formats
- **API Settings**: Configure authentication
- **System Prompts**: Customize instructions for different modes

## Advanced Usage

### Running on Apple Silicon (M1/M2)

Ouro includes optimized settings for Apple Silicon:

```bash
python -m ouro --model m1_optimized
```

### Using Ollama Integration

To use Ollama models for inference:

1. Start Ollama server
2. Enable Ollama integration in `config.py`:
   ```python
   OLLAMA_INTEGRATION = True
   ```
3. Start Ouro and select the "ollama" model:
   ```bash
   python -m ouro --model ollama
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
   - Close other memory-intensive applications

3. **Tab Completion Not Working**:
   - Tab completion is only available on Unix-like systems (Linux, macOS)
   - Make sure you're using the latest version of Python with readline support
   - Try using the `/help` command to see available commands

4. **Agent Tools Not Working**:
   - Verify that the specific tool is enabled in `config.py`
   - For web search, ensure internet connectivity
   - Check the log file for specific error messages

## License

Ouro is released under the MIT License. See the LICENSE file for details.

## Acknowledgments

Ouro is built using these amazing open-source projects:
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [LangChain](https://github.com/langchain-ai/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Rich](https://github.com/Textualize/rich)

## Credits

Developed by [roBlockWeb](https://github.com/roBlockweb/ouro)