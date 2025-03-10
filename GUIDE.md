# Ouro User Guide (v3.0.0)

This guide provides detailed information on how to use Ouro effectively, covering both basic and advanced features.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Terminal Interface](#terminal-interface)
3. [Document Management](#document-management)
4. [Chat Capabilities](#chat-capabilities)
5. [Agent Features](#agent-features)
6. [Memory System](#memory-system)
7. [API Integration](#api-integration)
8. [Advanced Configuration](#advanced-configuration)
9. [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

1. **Clone the repository and navigate to the directory**:
   ```bash
   git clone https://github.com/roBlockWeb/ouro.git
   cd ouro
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Ouro**:
   ```bash
   pip install -e .
   ```

4. **Authenticate with Hugging Face (Recommended)**:
   ```bash
   huggingface-cli login
   ```

### Starting Ouro

Run Ouro with default settings:
```bash
python -m ouro
```

Or specify a model size:
```bash
python -m ouro --model small
```

Available model options:
- `small` - Smallest model, best for low-resource systems
- `medium` - Default model, good balance of performance and resource usage
- `large` - More capable model, requires more RAM
- `very_large` - Most capable model, requires significant RAM
- `m1_optimized` - Optimized for Apple Silicon Macs
- `ollama` - Uses Ollama for inference (requires Ollama to be installed)

## Terminal Interface

Ouro provides a streamlined terminal interface with autocomplete for slash commands.

### Available Commands

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

### Using Tab Completion

Ouro supports tab completion for slash commands. Simply:

1. Type `/` to begin a command
2. Press the `Tab` key to see available command options or autocomplete the current command

### Chat Mode

The `/chat` command offers a dedicated chat environment:

- Start with `/chat` for general conversation
- Use `/chat topic` to start with a specific topic
- Type `exit` or `quit` to return to command mode
- Use limited slash commands (`/help`, `/clear_memory`, `/models`, `/change_model`) within chat mode

### Example Usage

```
>> /ingest documents/research-paper.pdf
✓ Ingested 15 document chunks

>> What are the main findings of the research paper?
The main findings of the research paper include...

>> /chat
Starting casual chat mode. Type 'exit' to return to command mode.
Chat>> Tell me about yourself
I'm Ouro, a privacy-focused AI assistant that runs completely on your local machine...

Chat>> What's the weather like?
I don't have real-time data like weather information, but I'd be happy to chat about other topics...

Chat>> /clear_memory
✓ Conversation memory cleared

Chat>> exit
Exiting chat mode.

>> /change_model large
✓ Changed model to large

>> /models
Available Models:
- small: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- medium: microsoft/phi-2
- large: mistralai/Mistral-7B-v0.1
- very_large: meta-llama/Llama-2-13b-chat-hf
- m1_optimized: TheBloke/Llama-2-7B-Chat-GGUF
- ollama: llama2
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

### Metadata

You can add metadata to documents for better organization and filtering:

- **source** - Where the document came from
- **date** - When the document was created
- **author** - Who created the document
- **category** - Topic or category
- **tags** - Related keywords
- **priority** - Importance (low, medium, high)

The metadata can be programmatically added via the API (see API Integration section).

## Chat Capabilities

Ouro provides three main modes for interaction:

### Standard RAG Mode

- Answers questions based on your documents
- Maintains conversation context
- Uses vector search to find relevant information
- Balances accuracy with conversational ability

### Casual Chat Mode

- Accessed via `/chat` command
- Focused on conversational interactions
- More personality and engagement
- Perfect for non-research interactions
- Provides a dedicated chat environment

### Agent Mode

- Solves complex tasks using reasoning
- Uses specialized tools when needed
- Can search the web for up-to-date information
- Explains reasoning process
- Best for complex, multi-step problems
- Access through Python API

## Agent Features

Agent capabilities provide enhanced tools for solving complex tasks.

### Available Tools

1. **Web Search**:
   - Searches the web for current information
   - Uses DuckDuckGo by default
   - Retrieves and summarizes results

2. **Math Tool**:
   - Performs mathematical calculations
   - Supports basic arithmetic operations
   - Handles complex expressions

3. **Query Generation**:
   - Creates effective search queries
   - Improves search results quality
   - Extracts key terms from questions

4. **Text Summarization**:
   - Condenses long text into key points
   - Maintains essential information
   - Adjustable output length

### Accessing Agent Features

Agent features are accessible through the Python API integration (see API Integration section).

## Memory System

Ouro's memory system helps maintain context during conversations.

### Short-Term Memory

- Stores recent conversation turns
- Helps maintain context in multi-turn dialogues
- Configurable number of turns to remember
- Cleared when you restart Ouro or use /clear_memory

### Long-Term Memory

- Stores important exchanges for future reference
- Uses embeddings for semantic retrieval
- Persists between sessions
- Can be marked as important for priority

### Memory Settings

Configure memory behavior in `config.py`:
```python
# Memory settings
MEMORY_TURNS = 10  # maximum conversation turns to keep in short-term memory
SAVE_CONVERSATIONS = True  # save conversations to disk for later retrieval
LONG_TERM_MEMORY = True  # enable long-term memory with embeddings
```

## API Integration

Ouro provides a Python API for integration with other applications.

### Authentication

By default, the API is accessible without authentication. To enable API key authentication:

1. Edit `config.py`:
   ```python
   API_KEY_REQUIRED = True
   API_KEYS = ["your-secret-key-1", "your-secret-key-2"]
   ```

2. Include the key when using the API:
   ```python
   from ouro.api import create_ouro_client
   
   client = create_ouro_client(api_key="your-secret-key-1")
   response = client.query("What is RAG?")
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

### Integration Examples

#### Python Client
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

# Get system stats
stats = client.get_stats()
print(f"Document count: {stats['document_count']}")
print(f"Current model: {stats['model']['name']}")
```

### Using External Tools with the API

You can use standard Python libraries alongside the Ouro API:

```python
import os
from ouro.api import create_ouro_client

# Create client
client = create_ouro_client()

# Process all files in a directory
dir_path = "/path/to/documents"
for filename in os.listdir(dir_path):
    if filename.endswith(".pdf") or filename.endswith(".txt"):
        file_path = os.path.join(dir_path, filename)
        result = client.ingest_document(
            file_path=file_path,
            metadata={"source": f"collection_{os.path.basename(dir_path)}"}
        )
        print(f"Ingested {result['ingested_documents']} chunks from {filename}")
```

## Advanced Configuration

For advanced users, Ouro offers extensive configuration options.

### Model Configuration

Modify or add models in `config.py`:
```python
MODELS = {
    "custom_model": {
        "name": "custom_model",
        "llm_model": "path/to/your/model",
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "memory_turns": 10,
        "quantize": True,
        "use_mps": False,
        "max_new_tokens": 512,
        "temperature": 0.2,
    },
    # ... other models
}
```

### Vector Database Options

Configure vector database settings:
```python
VECTOR_DB_TYPE = "faiss"  # Options: "faiss", "qdrant", "chroma"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ALTERNATE_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
```

### System Prompt Customization

Edit the system prompts to change the assistant's behavior:

```python
# Standard RAG mode prompt
SYSTEM_PROMPT = """
You are Ouro, a helpful AI assistant powered by a local language model.
...
"""

# Agent mode prompt
AGENT_SYSTEM_PROMPT = """
You are Ouro, an advanced AI agent capable of solving complex tasks using reasoning and tools.
...
"""

# Casual chat mode prompt
CHAT_SYSTEM_PROMPT = """
You are Ouro, a friendly and helpful AI assistant designed for casual conversation.
...
"""
```

## Troubleshooting

### Common Issues

#### Model Download Issues
- Authenticate with Hugging Face using `huggingface-cli login`
- Check internet connection
- Verify disk space
- Some models require accepting terms on the Hugging Face website

#### Out of Memory Errors
- Use a smaller model: `python -m ouro --model small`
- Reduce `max_new_tokens` in config.py
- Close other memory-intensive applications

#### Tab Completion Not Working
- Tab completion is only available on Unix-like systems (Linux, macOS)
- Make sure you're using the latest version of Python with readline support
- Try using the `/help` command to see available commands

#### Web Search Not Working
1. Check internet connection
2. Verify ENABLE_WEB_SEARCH is True in config.py
3. Ensure the web search provider is accessible

### Getting Help

If you encounter issues:
1. Check the logs in the `logs/` directory
2. Look for error messages in the terminal output
3. Submit an issue on GitHub with detailed information about the problem

## Upgrading Ouro

To upgrade to a new version:

1. Backup your data:
   ```bash
   cp -r data/ data_backup/
   ```

2. Pull the latest version:
   ```bash
   git pull origin main
   ```

3. Reinstall dependencies:
   ```bash
   pip install -e .
   ```

4. Restart Ouro:
   ```bash
   python -m ouro
   ```

## FAQ

**Q: Can Ouro work completely offline?**
A: Yes, once models are downloaded, Ouro works entirely offline, except for web search features.

**Q: How much disk space do I need?**
A: Around 10GB is recommended for models and vector database.

**Q: Can I use my own custom models?**
A: Yes, add them to the MODELS configuration in config.py.

**Q: Is my data secure and private?**
A: Yes, all data stays on your machine and is never sent to external services, except during web searches.

**Q: What's the difference between standard mode and chat mode?**
A: Standard mode is focused on answering questions using your documents, while chat mode is more conversational and personality-focused.

**Q: Can Ouro be used commercially?**
A: Yes, Ouro is released under the MIT license, but check the licenses of the underlying models you use as they may have different restrictions.