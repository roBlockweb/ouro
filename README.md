# Ouro: Privacy-First Local RAG System

Ouro is a fully offline RAG (Retrieval-Augmented Generation) system that runs completely on your local machine. Built with privacy as a priority, it leverages open-source language models and vector embeddings to provide context-aware responses based on your documents, without sending any data to external services.

<p align="center">
  <img src="docs/ouro-logo.png" alt="Ouro Logo" width="200"/>
</p>

## 🌟 Key Features

- **100% Private & Offline**: All processing happens locally—no data leaves your machine
- **Local LLM Support**: Works with Mistral, TinyLlama, Phi-2, Flan-T5, and custom models
- **Multi-Format Documents**: Process text, PDF, Markdown, and more
- **Smart Chunking**: Intelligently splits documents for better retrieval accuracy
- **Vector Search**: Fast similarity search with FAISS vector database
- **Clean Terminal UI**: User-friendly interface with rich text formatting and progress indicators
- **Simplified Configuration**: Choose from Small, Medium, Large, or custom model setups

## 📋 System Requirements

- Python 3.8 or newer
- RAM requirements depend on model choice:
  - Small: 2-4GB RAM
  - Medium: 4-6GB RAM
  - Large: 6-8GB RAM
  - Very Large: 12-16GB RAM
- Storage for downloaded models (varies by model choice)
- Hugging Face account (free) for model downloads

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/roBlock/ouro.git
cd ouro

# Set up and run (installs dependencies in virtual environment)
./run.sh
```

### Using Docker

```bash
# Build the Docker image
docker build -t ouro .

# Run the container
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  ouro
```

## 💡 Usage

After starting Ouro, the interactive interface will guide you through:

1. **Hugging Face Authentication**: Connect with your Hugging Face account
2. **Model Selection**: Choose from Small, Medium, Large, or Very Large configurations
3. **Knowledge Base Creation**: Add documents to your local database
4. **Querying**: Ask questions and receive contextual answers

### Command Reference

| Command | Description |
|---------|-------------|
| `help` | Display available commands |
| `ingest <file_path>` | Add a document to your knowledge base |
| `ingest_dir <directory>` | Add all documents in a directory |
| `ingest_text` | Add text content directly |
| `models` | Show available model configurations |
| `change_model` | Switch to a different model configuration |
| `exit` | Quit Ouro |

### Model Configurations

Ouro offers four pre-configured model setups to match your needs:

| Size | LLM Model | Embedding | Description |
|------|-----------|-----------|-------------|
| Small | TinyLlama-1.1B-Chat | MiniLM-L6-v2 | Fast, lightweight (2GB RAM) |
| Medium | Phi-2 | MPNet | Good balance of quality and speed (4GB RAM) |
| Large | Flan-T5-Large | MPNet | High quality responses (6GB RAM) |
| Very Large | Mistral-7B | BGE-Large | Best quality, requires 16GB+ RAM |

You can also specify a custom model from Hugging Face if you prefer.

## 🏗️ Architecture

Ouro follows a modular design with five core components:

1. **Document Processor**: Handles file ingestion and smart chunking
2. **Embedding Engine**: Creates vector representations of documents
3. **Vector Store**: Efficiently indexes and retrieves document chunks
4. **Local LLM**: Generates answers using retrieved context
5. **Interface**: Manages user interaction with real-time progress tracking

<p align="center">
  <img src="docs/architecture.png" alt="Ouro Architecture" width="650"/>
</p>

## 🔒 Privacy Considerations

- All data processing occurs locally on your machine
- No data is sent to remote servers for processing
- Model downloads are the only network activity
- Logs are stored locally and can be deleted any time

## 🛠️ Advanced Usage

### Environment Variables

You can configure Ouro with environment variables:

```
OURO_DEFAULT_SIZE=Medium  # Default model size (Small, Medium, Large, Very Large)
OURO_DATA_DIR=/path/to/custom/data  # Custom data directory
```

### Hugging Face Authentication

To use Ouro, you need to authenticate with Hugging Face:

1. Create a Hugging Face account at https://huggingface.co/join if you don't have one
2. Generate an access token at https://huggingface.co/settings/tokens
3. When prompted by Ouro, enter your token

### Development

```bash
# Run tests
./test.sh

# Clean environment (remove venv, cached models, etc.)
./reset.sh

# Install in development mode
pip install -e .
```

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| Memory errors | Try a smaller model configuration |
| Slow performance | Use the Small or Medium configuration |
| Model download issues | Verify Hugging Face login and internet connection |
| "Model not found" error | Ensure you're using a valid Hugging Face model path |

## 📜 License

[MIT License](LICENSE)

## 🙏 Acknowledgements

- [Hugging Face](https://huggingface.co/) for open-source models
- [LangChain](https://github.com/langchain-ai/langchain) for RAG components
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Rich](https://github.com/Textualize/rich) for terminal UI

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

Created by [roBlock](https://github.com/roBlock) • [Report Bug](https://github.com/roBlock/ouro/issues)