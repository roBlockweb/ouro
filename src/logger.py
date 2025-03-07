"""
Logging configuration for the Ouro RAG system
"""
import logging
import datetime
from pathlib import Path

from src.config import LOGS_DIR

def setup_logger():
    """Configure and return a logger for the application"""
    # Create a timestamp for the log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"ouro_{timestamp}.log"
    
    # Configure the logger
    logger = logging.getLogger("ouro")
    logger.setLevel(logging.INFO)
    
    # File handler for logging to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler for displaying logs in the terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
    
    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Create the logger instance
logger = setup_logger()