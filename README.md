# Ouro v2.1

Ouro is a privacy-first, offline AI assistant powered by Ollama. It provides a seamless, user-friendly interface for interacting with local large language models without sending your data to external services.

![Ouro Logo](assets/logo.png)

## Features

- **100% Offline Operation**: All processing happens locally on your machine
- **Privacy-Focused**: Your data never leaves your computer
- **Retrieval-Augmented Generation (RAG)**: Chat with your documents without internet
- **Minimalist UI**: Clean, monochrome interface designed for productivity
- **Dependency-Aware**: Seamlessly integrates with Ollama and Qdrant

## System Requirements

- **Operating System**: macOS, Windows, or Linux
- **RAM**: Minimum 8GB recommended (4GB for smaller models)
- **Storage**: 1GB for the application, plus space for models and documents
- **Dependencies**:
  - [Ollama](https://ollama.ai/download) - for running the language models
  - [Docker](https://www.docker.com/products/docker-desktop/) - for Qdrant (optional but recommended for document RAG)

## Installation

### From Release

1. Download the latest release for your platform from the [Releases](https://github.com/yourusername/ouro/releases) page
2. Open the DMG file (macOS) or run the installer (Windows/Linux)
3. Follow the installation wizard

### From Source

1. Clone this repository
2. Install dependencies:
   ```
   npm install
   ```
3. Build the application:
   ```
   npm run build
   ```
4. The built application will be in the `dist` directory

## Usage

1. Launch Ouro from your applications folder
2. Ouro will automatically check for Ollama and Qdrant, helping you set them up if needed
3. Select a model based on your hardware capabilities
4. Start chatting with your AI assistant
5. Use the document interface to upload and chat with your documents

## Development

### Project Structure

- `src/` - Electron application source code
  - `main/` - Main Electron process
  - `ui/` - UI for the installer
- `core/` - Core functionality
  - `web/` - Web-based chat interface
  - `config.yaml` - Configuration file
  - `start_server.py` - Python web server script

### Running in Development Mode

```
npm start
```

### Building for Distribution

```
npm run build
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Ollama](https://ollama.ai/) for making local LLMs easily accessible
- [Qdrant](https://qdrant.tech/) for the vector database used in RAG functionality
- [Electron](https://www.electronjs.org/) for the application framework