# 🧠 OuroGPT Autonomous LLM Chat App

Welcome to **OuroGPT**, a fully autonomous AI conversation loop powered by a local Large Language Model (LLM). This project uses **FAISS** for vector search, a local Python environment for running the Mistral 7B Instruct model, and a dynamic two-agent system (Ouro & Brain) that grows its knowledge via web scraping.

## 🚀 Features

- **Local LLM Execution**: No external API calls; runs with `llama-cpp-python` on macOS (Metal acceleration).
- **Two-Agent Conversation**: "Ouro" and "Brain" each have unique roles, exchanging messages infinitely.
- **FAISS Vector Search**: Stores and retrieves conversation embeddings for context and memory.
- **Dynamic Web Scraping**: Agents can fetch real-time data and update the FAISS store.
- **Duplicate Removal**: Automatic removal of near-duplicate vectors in FAISS to optimize storage.
- **Conversation Logging**: Saves all messages with timestamps in daily log files.
- **Topic Extraction**: Automatically identifies search topics from conversations for web research.

## 🔧 Project Structure

```
.
├── Mistral-7B-Instruct-v0.1-GGUF/    # Model Files (GGUF format)
├── faiss_store/                      # Contains FAISS index and text data
├── conversation_logs/                # Daily conversation logs
├── venv/                            # Python virtual environment
├── autonomous.py                    # Infinite conversation loop logic
├── brain.py                         # Brain agent implementation
├── ouro.py                          # Ouro agent implementation
├── topic_extractor.py               # Extracts searchable topics from text
├── llm_wrapper.py                   # Shared LLM (Mistral) wrapper
├── utils.py                         # Core utilities: FAISS, scraping, logging
├── main.py                          # Main entry point
├── config.py                        # Configuration settings
├── initialize_faiss.py              # Script to initialize/seed FAISS index
├── run_ouro.sh                      # Wrapper script to run the system
├── requirements.txt                 # Python dependencies
└── README.md                        # This documentation
```

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YourUsername/OuroGPT.git
   cd OuroGPT
   ```

2. **Download Mistral model**:
   Download the Mistral 7B Instruct model in GGUF format from [Hugging Face](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1-GGUF) and place it in the `Mistral-7B-Instruct-v0.1-GGUF` directory.

3. **Create & Activate Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Initialize FAISS Index**:
   ```bash
   python initialize_faiss.py
   ```

6. **Make the run script executable**:
   ```bash
   chmod +x run_ouro.sh
   ```

## 🏃 Running OuroGPT

### Using the Wrapper Script (Recommended)

```bash
# Run in autonomous mode (default)
./run_ouro.sh

# Run in interactive mode
./run_ouro.sh interactive
```

### Manual Execution

```bash
# First activate the virtual environment
source venv/bin/activate

# Initialize FAISS (first time only)
python initialize_faiss.py

# Run in autonomous mode
python main.py --mode=autonomous

# Or run in interactive mode
python main.py --mode=interactive
```

Press `Ctrl+C` to stop the application.

## 🧠 How It Works

1. **Autonomous Mode**: Ouro and Brain engage in continuous conversation, exploring topics and learning from the web.
   - Each agent has a unique personality and communication style
   - The conversation loop extracts topics from responses
   - Agents research these topics on the web and embed the information in FAISS
   - Future responses are enhanced using this growing knowledge base

2. **Interactive Mode**: You can directly chat with Brain, bypassing Ouro.
   - Type your messages and receive responses
   - Brain will still research topics and update the knowledge base

## 🗄️ Data & Logs

- **FAISS Index**: Vector embeddings stored in `faiss_store/index.faiss`
- **Conversation Logs**: Daily logs in `conversation_logs/` (e.g., `2025-02-27_Brain.log`)
- **System Log**: General system logs in `conversation_logs/system.log`

## 🔧 System Requirements

- **Python**: 3.9+
- **RAM**: 8GB+ (16GB recommended for optimal performance)
- **GPU**: Apple Silicon M1/M2/M3 (Metal acceleration) or NVIDIA GPU
- **Chrome/Chromium**: Required for Selenium web scraping
- **Disk Space**: ~4GB for model files and index

## 🐛 Troubleshooting

- **Import Errors**: Make sure to run the script with the virtual environment activated using `./run_ouro.sh` or by manually activating the environment.

- **LLM Not Loading**:
  - Verify the model file exists in the correct location
  - Check `LLM_MODEL_PATH` in `config.py`
  - Try a smaller quantized model if RAM is limited

- **Selenium Issues**:
  - Ensure Chrome/Chromium is installed
  - If Chrome updates cause problems, try installing a specific ChromeDriver version
  - Headless mode is enabled by default (no browser windows)

- **Memory Usage**:
  - Set `MAX_CONVERSATION_EXCHANGES` in `config.py` to restart conversations periodically
  - Use smaller quantized models (e.g., Q3_K_S instead of Q3_K_M)

## 📄 License

MIT License

Created with ❤️ by roBlock

If you find this project helpful, please star the repository and consider contributing!
