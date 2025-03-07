"""
Configuration settings for the Ouro RAG system
"""
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
MODEL_CACHE_DIR = DATA_DIR / "models"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
DOCUMENTS_DIR = DATA_DIR / "documents"

# Create necessary directories
for dir_path in [DATA_DIR, LOGS_DIR, MODEL_CACHE_DIR, VECTOR_STORE_DIR, DOCUMENTS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Embedding models
EMBEDDING_MODELS = {
    "Small": "sentence-transformers/all-MiniLM-L6-v2",    # 384 dimensions, fast
    "Medium": "sentence-transformers/all-mpnet-base-v2",  # 768 dimensions, better quality
    "Large": "BAAI/bge-large-en-v1.5",                   # 1024 dimensions, high quality
}

# LLM Models by size with recommended embedding pairings
MODEL_CONFIGURATIONS = {
    "Small": {
        "llm": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "embedding": "Small",
        "description": "1.1B parameters - Fast, low resource usage (~2GB RAM)",
        "chunk_size": 800,
        "chunk_overlap": 100,
        "top_k": 3,
    },
    "Medium": {
        "llm": "microsoft/phi-2",
        "embedding": "Medium",
        "description": "2.7B parameters - Good balance of quality and speed (~4GB RAM)",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": 4,
    },
    "Large": {
        "llm": "google/flan-t5-large",
        "embedding": "Medium",
        "description": "770M parameters - High quality response generation (~6GB RAM)",
        "chunk_size": 1200,
        "chunk_overlap": 250,
        "top_k": 4,
    },
    "Very Large": {
        "llm": "mistralai/Mistral-7B-v0.1",
        "embedding": "Large",
        "description": "7B parameters - Best quality, slower speed (~16GB RAM)",
        "chunk_size": 1500,
        "chunk_overlap": 300,
        "top_k": 5,
    }
}

# Default settings
DEFAULT_SIZE = "Medium"
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
5. For greetings or chitchat without context, provide a brief, friendly response without inventing information."""