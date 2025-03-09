"""
Logging setup for the Ouro RAG system.
"""
import logging
import os
from datetime import datetime
from pathlib import Path

from ouro.constants import LOGS_DIR, LOG_LEVEL

# Create logs directory if it doesn't exist
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure logger
logger = logging.getLogger("ouro")
logger.setLevel(logging.getLevelName(LOG_LEVEL))

# Prevent duplicate handlers when reloading modules
if not logger.handlers:
    # Create file handler
    log_file = LOGS_DIR / f"ouro_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    
    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def get_logger():
    """Return the configured logger."""
    return logger