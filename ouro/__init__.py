"""
Ouro: Privacy-First Local RAG System

Ouro is a privacy-focused Retrieval-Augmented Generation (RAG) system that runs
completely offline on your local machine. It allows you to interact with your
documents using AI without sending data to external services.

Key Features:
- 100% Private & Offline: All processing happens on your machine
- Streamlined Terminal Interface: Clean command-line experience with tab completion
- Casual Chat Mode: Dedicated chat environment for conversational interactions
- Document Processing: Support for TXT, PDF, Markdown, CSV, HTML, and JSON files
- Vector Search: Efficient similarity search to find relevant information
- Conversation Memory: Maintains context across multiple turns with short and long-term memory
- Agent Capabilities: Solve complex tasks using reasoning and specialized tools
- Web Search Integration: Optional web search for up-to-date information
- Python API: Local Python API for integration with other applications
- Apple Silicon Optimized: Special configurations for M1/M2 Macs
- Ollama Integration: Optional integration with Ollama for additional models
"""

__version__ = "3.0.0"

# Import API client for ease of use
from ouro.api import create_ouro_client