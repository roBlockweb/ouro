"""
Configuration settings for the Ouro RAG system
"""
import os
import platform
from pathlib import Path

# Platform detection
IS_APPLE_SILICON = (
    platform.system() == "Darwin" and 
    (platform.machine().startswith("arm") or 
     platform.processor() == "arm" or 
     "M1" in platform.processor() or 
     "M2" in platform.processor())
)

# Project paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
MODEL_CACHE_DIR = DATA_DIR / "models"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
DOCUMENTS_DIR = DATA_DIR / "documents"
MEMORY_DIR = DATA_DIR / "conversations"

# Create necessary directories
for dir_path in [DATA_DIR, LOGS_DIR, MODEL_CACHE_DIR, VECTOR_STORE_DIR, DOCUMENTS_DIR, MEMORY_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Performance settings
# Only enable quantization if on CUDA, disable for CPU/MPS
import torch
DEFAULT_QUANTIZE = torch.cuda.is_available()  # Quantize only when CUDA is available
DEFAULT_FAST_MODE = False  # Start with standard quality
DEFAULT_MEMORY_TURNS = 5  # Remember the last 5 conversation turns
DEFAULT_SAVE_CONVERSATIONS = True  # Save conversations for learning

# Make these settings importable
__all__ = ['IS_APPLE_SILICON', 'BASE_DIR', 'DATA_DIR', 'LOGS_DIR', 'MODEL_CACHE_DIR', 
           'VECTOR_STORE_DIR', 'DOCUMENTS_DIR', 'MEMORY_DIR', 'DEFAULT_QUANTIZE', 
           'DEFAULT_FAST_MODE', 'DEFAULT_MEMORY_TURNS', 'DEFAULT_SAVE_CONVERSATIONS',
           'APPLE_SILICON_OPTIMIZED_MODELS', 'EMBEDDING_MODELS', 'MODEL_CONFIGURATIONS',
           'DEFAULT_SIZE', 'DEFAULT_EMBEDDING_MODEL', 'DEFAULT_LLM_MODEL',
           'CHUNK_SIZE', 'CHUNK_OVERLAP', 'TOP_K_RESULTS', 'SYSTEM_PROMPT']

# Apple Silicon optimized models
APPLE_SILICON_OPTIMIZED_MODELS = {
    "phi-2": "microsoft/phi-2",  # Runs well on M1/M2
    "phi-1.5": "microsoft/phi-1_5",  # Smaller alternative
    "mpt-7b": "mosaicml/mpt-7b-instruct",  # MPT architecture runs well on MPS
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Small and efficient
}

# Embedding models
EMBEDDING_MODELS = {
    "Small": "sentence-transformers/all-MiniLM-L6-v2",    # 384 dimensions, fast
    "Medium": "sentence-transformers/all-mpnet-base-v2",  # 768 dimensions, better quality
    "Large": "BAAI/bge-large-en-v1.5",                   # 1024 dimensions, high quality
}

# LLM Models by size with recommended embedding pairings
# Note: Models are now selected with Apple Silicon optimizations in mind
MODEL_CONFIGURATIONS = {
    "Small": {
        "llm": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Works well on all platforms
        "embedding": "Small",
        "description": "1.1B parameters - Fast, low resource usage (~2GB RAM)",
        "chunk_size": 800,
        "chunk_overlap": 100,
        "top_k": 3,
        "quantize": torch.cuda.is_available(),  # Only quantize on CUDA
        "fast_mode": False,
        "memory_turns": 3,
    },
    "Medium": {
        "llm": "microsoft/phi-2",  # Excellent on Apple Silicon
        "embedding": "Medium",
        "description": "2.7B parameters - Good balance of quality and speed (~4GB RAM)",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": 4,
        "quantize": torch.cuda.is_available(),  # Only quantize on CUDA
        "fast_mode": False,
        "memory_turns": 5,
    },
    "Large": {
        "llm": "HuggingFaceH4/zephyr-7b-beta",  # Better instruction following than Mistral
        "embedding": "Medium",
        "description": "7B parameters - High quality responses (~10GB RAM)",
        "chunk_size": 1200,
        "chunk_overlap": 250,
        "top_k": 4,
        "quantize": torch.cuda.is_available(),  # Only quantize on CUDA
        "fast_mode": False,
        "memory_turns": 5,
    },
    "Very Large": {
        "llm": "mistralai/Mistral-7B-v0.1",
        "embedding": "Large",
        "description": "7B parameters - Best quality, slower speed (~16GB RAM)",
        "chunk_size": 1500,
        "chunk_overlap": 300,
        "top_k": 5,
        "quantize": torch.cuda.is_available(),  # Only quantize on CUDA
        "fast_mode": False,
        "memory_turns": 10,
    }
}

# Add Apple Silicon optimized configurations if running on M1/M2
if IS_APPLE_SILICON:
    MODEL_CONFIGURATIONS["M1Optimized"] = {
        "llm": "microsoft/phi-2",  # Specially optimized for M1/M2
        "embedding": "Medium",
        "description": "2.7B parameters - Optimized for Apple Silicon",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": 4,
        "quantize": False,  # MPS doesn't need or support quantization
        "fast_mode": False,
        "memory_turns": 5,
    }
    # Set as default if we're on Apple Silicon
    DEFAULT_SIZE = "M1Optimized"
else:
    DEFAULT_SIZE = "Medium"

# Default settings
DEFAULT_EMBEDDING_MODEL = EMBEDDING_MODELS[MODEL_CONFIGURATIONS[DEFAULT_SIZE]["embedding"]]
DEFAULT_LLM_MODEL = MODEL_CONFIGURATIONS[DEFAULT_SIZE]["llm"]
CHUNK_SIZE = MODEL_CONFIGURATIONS[DEFAULT_SIZE]["chunk_size"]
CHUNK_OVERLAP = MODEL_CONFIGURATIONS[DEFAULT_SIZE]["chunk_overlap"]
TOP_K_RESULTS = MODEL_CONFIGURATIONS[DEFAULT_SIZE]["top_k"]

# System prompt
SYSTEM_PROMPT = """You are Ouro, a helpful privacy-focused assistant. You provide answers based ONLY on the documents in your knowledge base while keeping all data secure and local.

IMPORTANT INSTRUCTIONS:
1. Only answer questions if relevant information exists in the provided context.
2. If the context doesn't contain information to answer the question, say "I don't have information about that in my knowledge base."
3. Do not make up or hallucinate any information.
4. Keep responses factual and grounded in the retrieved documents.
5. For greetings or chitchat without context, provide a brief, friendly response without inventing information.
6. When referencing previous parts of the conversation, be specific and accurate."""