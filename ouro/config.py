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

# Web UI settings
WEB_HOST = "localhost"
WEB_PORT = 7860
WEB_TIMEOUT = 20  # seconds before defaulting to web UI on startup

# Vector store settings
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ALTERNATE_EMBEDDING_MODEL = "hkunlp/instructor-large"

# Performance settings
USE_QUANTIZATION = platform.system() == "Linux" or platform.system() == "Windows"
USE_FAST_MODE = False  # generates text faster but may reduce quality
MEMORY_TURNS = 10  # maximum conversation turns to remember
SAVE_CONVERSATIONS = True

# Default RAG settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 4
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1
LOG_LEVEL = "INFO"

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
        "llm_model": "meta-llama/Llama-2-7b-chat-hf",
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
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
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
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

# API settings
API_ENABLED = True
API_PREFIX = "/api/v1"