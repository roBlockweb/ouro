"""
Shared constants for the Ouro RAG system.
This module contains constants shared between multiple modules
to avoid circular imports.
"""
from pathlib import Path

# Base paths
ROOT_DIR = Path(__file__).parent.absolute()
LOGS_DIR = ROOT_DIR / "logs"

# Logging settings
LOG_LEVEL = "INFO"