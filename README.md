# Ouro: Privacy-First Local RAG System

Ouro is a fully offline RAG (Retrieval-Augmented Generation) system that runs completely on your local machine. It lets you ask questions about your documents using AI, without sending any of your data to external services.

<p align="center">
  <img src="docs/ouro-logo.png" alt="Ouro Logo" width="200"/>
</p>

## ✨ What is Ouro?

Think of Ouro as a private AI assistant that helps you find information in your documents:

- **100% Private**: Everything runs on your computer - no data is sent to the cloud
- **Easy to Use**: Simple interface that guides you through each step
- **Flexible**: Works with different models based on your computer's capabilities
- **Multi-Format**: Process text files, PDFs, and Markdown documents
- **Reliable**: Built-in timeouts and robust error handling prevent hanging
- **Intuitive**: Clear UI with user/assistant distinction and helpful prompts

## 🚀 Getting Started

### Step 1: Set Up Prerequisites

Before using Ouro, you need to set up a Hugging Face account for downloading models.

📝 **Please follow the instructions in [GUIDE.md](GUIDE.md) first** to create an account and get your token.

### Step 2: Clone the Repository

```bash
# Copy and paste this in your terminal:
git clone https://github.com/roBlock/ouro.git
cd ouro
```

### Step 3: Run Ouro

```bash
# On macOS/Linux:
./run.sh

# On Windows:
./run.bat
# Or manually:
python -m pip install -r requirements.txt
python -m src.main
```

That's it! The application will guide you through the rest of the process.

> **Tips**: 
> - If you encounter memory issues, use the small model preset:
>   ```bash
>   ./run.sh --small
>   ```
> - If you have problems with interactive input, use the non-interactive mode:
>   ```bash
>   ./run.sh --no-prompts
>   ```
> - For help with additional options:
>   ```bash
>   ./run.sh --help
>   ```
> These options automatically load the smallest, most efficient model configuration.

## 🛠️ Using Ouro

Ouro's interface will walk you through these steps:

1. **Choose a Model**: Select from Small, Medium, Large, or Very Large based on your computer's capabilities
2. **Add Documents**: Use the `ingest` command to add files to your knowledge base
3. **Ask Questions**: Type your questions and get AI-generated answers based on your documents

### Commands You Can Use

| Command | What It Does |
|---------|-------------|
| `help` | Shows all available commands |
| `ingest <file_path>` | Adds a document to your knowledge base |
| `ingest_dir <directory>` | Adds all documents from a folder |
| `models` | Shows available AI models |
| `change_model` | Switches to a different model |
| `exit` | Closes Ouro |

### Model Options

Choose the right model for your computer:

| Size | Description | RAM Needed |
|------|-------------|------------|
| Small | Fast, basic responses | 2-4GB |
| Medium | Good balance of quality and speed | 4-6GB |
| Large | Higher quality responses | 6-8GB |
| Very Large | Best quality, slower | 12-16GB+ |

## 📋 Common Questions

### How do I add documents?

Type `ingest` followed by the path to your file:
```
ingest /path/to/your/document.pdf
```

Or add an entire folder:
```
ingest_dir /path/to/your/documents
```

### What file types are supported?

- Text files (.txt)
- PDF documents (.pdf)
- Markdown files (.md)
- And more!

### Where is my data stored?

All data stays on your computer in the `data` folder:
- Documents: `data/documents/`
- Models: `data/models/`
- Vector database: `data/vector_store/`

### Is an internet connection required?

Internet is only needed for the initial model download. After that, Ouro works offline.

## 🔧 Troubleshooting

### Application crashes or runs out of memory

- Choose a smaller model (use Small or Medium)
- Close other applications to free up memory
- See [GUIDE.md](GUIDE.md) for more troubleshooting tips

### Can't download models

- Check your Hugging Face login (see [GUIDE.md](GUIDE.md))
- Verify your internet connection
- Ensure you have enough disk space

## 📚 Learn More

- [How Ouro Works](docs/architecture.txt): Technical details about the system
- [Troubleshooting Guide](GUIDE.md#3-troubleshooting): Solutions to common issues
- [Project Wiki](https://github.com/roBlock/ouro/wiki): Additional documentation

## 📜 License

[MIT License](LICENSE)

---

Created with ❤️ by [roBlock](https://github.com/roBlock)