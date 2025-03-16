# Ouro v2.5 - Privacy-First Offline AI Assistant

Ouro is a privacy-focused AI assistant that runs completely on your local machine, powered by [Ollama](https://ollama.ai). This application enables you to use powerful large language models without sending any data to external servers, ensuring total privacy and control over your information.

![Ouro Logo](assets/logo.png)

## Key Features

- **100% Offline Operation**: All processing happens on your machine
- **Complete Privacy**: Your data never leaves your computer
- **Powered by Ollama**: Leverage the latest open-source language models
- **RAG Capabilities**: Chat with your own documents using vector search
- **User-Friendly Interface**: Modern, intuitive design
- **Cross-Platform**: Works on macOS (Windows and Linux support coming soon)

## System Requirements

- **Operating System**: macOS 11+ (Big Sur or newer), Windows 10+ or Linux (Ubuntu 20.04+)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 10GB for application, 5-50GB for models (depending on model size)
- **Additional Requirements**:
  - [Ollama](https://ollama.ai/download) must be installed
  - [Docker](https://www.docker.com/) (optional, for Qdrant vector database)

## Installation

### macOS

1. Download the latest release DMG file from the [Releases page](https://github.com/roBlockweb/ouro/releases)
2. Open the DMG file and drag Ouro to your Applications folder
3. Run Ouro from your Applications folder
4. Follow the setup wizard to configure your installation

### Building from Source

If you prefer to build from source:

```bash
# Clone the repository
git clone https://github.com/roBlockweb/ouro.git
cd ouro

# Install dependencies
npm install

# Start in development mode
npm start

# Build for your platform
npm run build
```

## Quick Start

1. Install and run Ouro
2. Complete the setup wizard, which will:
   - Check your system compatibility
   - Help install required dependencies (Ollama, Qdrant)
   - Choose an appropriate model for your hardware
3. Once setup is complete, you can start chatting with your AI assistant
4. Use the Documents tab to upload and chat with your own documents

## Models

Ouro supports various models through Ollama:

- **Small Models** (8-10B parameters): Great for lower-end hardware, faster responses
  - Recommended: `llama3:8b` or `phi:3-mini`
- **Medium Models** (30-70B parameters, quantized): Better quality with moderate hardware
  - Recommended: `llama3:70b-q4_K_M`
- **Large Models** (70B+ parameters): Best quality, requires high-end hardware
  - Recommended: `llama3:70b`

## RAG (Retrieval Augmented Generation)

Ouro includes RAG capabilities to chat with your documents:

1. Go to the Documents tab
2. Upload PDF, TXT, DOCX, or markdown files
3. The files will be processed and indexed in the vector database
4. Switch to Chat mode and ask questions about your documents

## Development

### Project Structure

```
ouro/
├── assets/               # Application icons and assets
├── config/               # Configuration files
├── core/                 # Python server components
│   ├── start_server.py   # Main Python server
│   ├── data/             # Document storage
│   ├── models/           # Model configurations
│   └── web/              # Web interface files
├── src/                  # Electron application source
│   ├── main/             # Main process code
│   │   ├── index.js      # Electron entry point
│   │   └── preload.js    # Preload script for IPC
│   └── ui/               # Renderer process UI code
│       ├── css/          # Stylesheets
│       ├── js/           # UI JavaScript
│       └── index.html    # Main HTML template
└── build-*.sh            # Build scripts
```

### Technology Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Node.js, Electron, Python
- **AI Components**: Ollama, Qdrant
- **Build Tools**: electron-builder

## Security

Ouro is designed with security in mind:

- **No Network Requirement**: After installation, no internet connection needed
- **Local Processing**: All data stays on your device
- **Sandboxed Execution**: Proper context isolation in Electron
- **Content Security Policy**: Restricted script execution

## Contributing

Contributions are welcome! Please see [ideas.md](ideas.md) for roadmap and development ideas.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Credits

Developed by roBlock. Special thanks to all contributors who made this project possible.

## Acknowledgments

- The [Ollama](https://ollama.ai) team for making local LLMs accessible
- The [Qdrant](https://qdrant.tech) team for their excellent vector database
- The open-source AI community

---

**Privacy Notice**: Ouro is designed to operate completely offline. No data is collected or transmitted to external servers. All processing happens locally on your device.