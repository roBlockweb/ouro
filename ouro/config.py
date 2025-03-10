"""
Configuration settings for the Ouro RAG system.
"""
import os
import platform
from pathlib import Path

# Import shared constants to avoid circular imports
from ouro.constants import ROOT_DIR, LOGS_DIR, LOG_LEVEL

# Base paths
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
DOCUMENTS_DIR = DATA_DIR / "documents"
CONVERSATIONS_DIR = DATA_DIR / "conversations"

# Vector store settings
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ALTERNATE_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Higher quality alternative
VECTOR_DB_TYPE = "faiss"  # Options: "faiss", "qdrant", "chroma"

# Performance settings
USE_QUANTIZATION = platform.system() == "Linux" or platform.system() == "Windows"
USE_FAST_MODE = False  # generates text faster but may reduce quality

# Memory settings
MEMORY_TURNS = 10  # maximum conversation turns to keep in short-term memory
SAVE_CONVERSATIONS = True  # save conversations to disk for later retrieval
LONG_TERM_MEMORY = True  # enable long-term memory with embeddings

# Default RAG settings
CHUNK_SIZE = 500  # text chunk size for documents
CHUNK_OVERLAP = 50  # overlap between chunks
TOP_K_RESULTS = 4  # number of document chunks to retrieve
MAX_NEW_TOKENS = 512  # maximum number of tokens to generate
TEMPERATURE = 0.1  # temperature for text generation (lower = more deterministic)
LOG_LEVEL = "INFO"  # logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

# Agent settings
ENABLE_AGENTS = True  # Enable the agent functionality
TOOLS_ENABLED = ["web_search", "math", "query_generation", "summarization"]
AGENT_MODEL = "medium"  # Default model to use for agent functions
ENABLE_REASONING = True  # Enable step-by-step reasoning for complex tasks

# Web integration
ENABLE_WEB_SEARCH = True  # Whether to allow web search capabilities
WEB_SEARCH_PROVIDER = "duckduckgo"  # Options: "duckduckgo", "google", "bing"
MAX_SEARCH_RESULTS = 3  # Maximum number of web search results to return

# Integration with external services
OLLAMA_INTEGRATION = False  # Enable integration with Ollama
OLLAMA_HOST = "http://localhost:11434"  # Ollama API endpoint
QDRANT_INTEGRATION = False  # Enable integration with Qdrant
QDRANT_HOST = "http://localhost:6333"  # Qdrant API endpoint

# Model configurations
MODELS = {
    "small": {
        "name": "small",
        "llm_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "memory_turns": MEMORY_TURNS,
        "quantize": USE_QUANTIZATION,
        "use_mps": platform.system() == "Darwin" and platform.processor() == "arm",
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
    },
    "medium": {
        "name": "medium",
        "llm_model": "microsoft/phi-2",
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "memory_turns": MEMORY_TURNS,
        "quantize": USE_QUANTIZATION,
        "use_mps": platform.system() == "Darwin" and platform.processor() == "arm",
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
    },
    "large": {
        "name": "large",
        "llm_model": "mistralai/Mistral-7B-v0.1",
        "embedding_model": ALTERNATE_EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "memory_turns": MEMORY_TURNS,
        "quantize": USE_QUANTIZATION,
        "use_mps": platform.system() == "Darwin" and platform.processor() == "arm",
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
    },
    "very_large": {
        "name": "very_large",
        "llm_model": "meta-llama/Llama-2-13b-chat-hf",
        "embedding_model": ALTERNATE_EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "memory_turns": MEMORY_TURNS,
        "quantize": USE_QUANTIZATION,
        "use_mps": platform.system() == "Darwin" and platform.processor() == "arm",
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
    },
    "m1_optimized": {
        "name": "m1_optimized",
        "llm_model": "TheBloke/Llama-2-7B-Chat-GGUF",
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "memory_turns": MEMORY_TURNS,
        "quantize": False,
        "use_mps": True,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
    },
    "ollama": {
        "name": "ollama",
        "llm_model": "llama2",  # This will be passed to Ollama API
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "memory_turns": MEMORY_TURNS,
        "quantize": False,
        "use_ollama": True,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
    },
}

DEFAULT_MODEL = "medium"

# System prompt for the RAG system
SYSTEM_PROMPT = """
You are Ouro, a helpful AI assistant powered by a local language model.

CRITICAL INSTRUCTIONS:
- NEVER create fictional dialogues or example conversations 
- NEVER include "User:" or "Assistant:" in your responses
- NEVER refer to past or future conversations between users
- ALWAYS respond DIRECTLY to the question asked
- ONLY use information from the knowledge base provided
- If you don't have enough information, say: "I don't have enough information to answer that question."
- Keep responses concise and to the point
- Do not hallucinate information that's not in the provided context

Remember: You are having a DIRECT conversation with the user. There are NO other conversations happening.
"""

# Enhanced system prompt for agent mode
AGENT_SYSTEM_PROMPT = """
You are Ouro, an advanced AI agent capable of solving complex tasks using reasoning and tools.

CRITICAL INSTRUCTIONS:
- When solving complex problems, break them down into clear steps
- Use available tools when needed to gather information or perform actions
- Be thorough in your analysis but concise in your responses
- Always verify your work before providing final answers
- If you don't have enough information, explain what you need
- When using tools, explain your reasoning for choosing them
- Do not hallucinate information or capabilities

Remember: You are a helpful assistant with access to various tools. Use them wisely to help the user.
"""

# Enhanced system prompt for casual chat mode
CHAT_SYSTEM_PROMPT = """
You are Ouro, a friendly and helpful AI assistant designed for casual conversation.

CRITICAL INSTRUCTIONS:
- Be conversational, warm, and engaging
- Respond directly to the user's queries and comments
- Remember information shared during the conversation
- Ask follow-up questions to show interest when appropriate
- Use a natural, friendly tone
- Keep your responses concise and to-the-point
- Respect the user's privacy

Remember: Your goal is to create a helpful and pleasant conversational experience.
"""

# API settings
API_ENABLED = True  # Whether to enable API
API_KEY_REQUIRED = False  # Whether to require API key for API access
API_KEYS = []  # List of valid API keys if enabled

# Sample API key configurations (for demonstration)
# Uncomment and modify for actual use
# API_KEYS = [
#     "ouro-sk-OcwETUfGVROZ1qWSNZ4xTVb2VnKNcQpP", 
#     "ouro-sk-P6JtfLG1JsTWxRZkLvVcNbB3HpSmY7Fa"
# ]

# Knowledge base metadata
METADATA_FIELDS = ["source", "date", "author", "category", "tags", "priority"]
DEFAULT_METADATA = {"source": "direct_input", "priority": "medium"}