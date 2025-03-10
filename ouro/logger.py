"""
Logging setup for the Ouro RAG system.
"""
import logging
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from ouro.constants import LOGS_DIR, LOG_LEVEL

# Create logs directory if it doesn't exist
os.makedirs(LOGS_DIR, exist_ok=True)

class OuroLogger(logging.Logger):
    """Extended logger class with Ouro-specific functionality."""
    
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        self.extra_data = {}
    
    def info(self, msg, *args, **kwargs):
        """Override info to handle extra event data."""
        event_type = kwargs.pop('event_type', None)
        if event_type:
            extra = kwargs.get('extra', {})
            if isinstance(extra, dict):
                extra.update({'event_type': event_type})
            else:
                extra = {'event_type': event_type}
            kwargs['extra'] = extra
            
            # Move any other non-standard args to extra
            for k, v in list(kwargs.items()):
                if k not in ['exc_info', 'stack_info', 'stacklevel', 'extra']:
                    if 'extra' not in kwargs:
                        kwargs['extra'] = {}
                    kwargs['extra'][k] = v
                    del kwargs[k]
        
        super().info(msg, *args, **kwargs)
    
    def debug(self, msg, *args, **kwargs):
        """Override debug to handle extra event data."""
        event_type = kwargs.pop('event_type', None)
        if event_type:
            extra = kwargs.get('extra', {})
            if isinstance(extra, dict):
                extra.update({'event_type': event_type})
            else:
                extra = {'event_type': event_type}
            kwargs['extra'] = extra
            
            # Move any other non-standard args to extra
            for k, v in list(kwargs.items()):
                if k not in ['exc_info', 'stack_info', 'stacklevel', 'extra']:
                    if 'extra' not in kwargs:
                        kwargs['extra'] = {}
                    kwargs['extra'][k] = v
                    del kwargs[k]
        
        super().debug(msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        """Override warning to handle extra event data."""
        event_type = kwargs.pop('event_type', None)
        if event_type:
            extra = kwargs.get('extra', {})
            if isinstance(extra, dict):
                extra.update({'event_type': event_type})
            else:
                extra = {'event_type': event_type}
            kwargs['extra'] = extra
            
            # Move any other non-standard args to extra
            for k, v in list(kwargs.items()):
                if k not in ['exc_info', 'stack_info', 'stacklevel', 'extra']:
                    if 'extra' not in kwargs:
                        kwargs['extra'] = {}
                    kwargs['extra'][k] = v
                    del kwargs[k]
        
        super().warning(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        """Override error to handle extra event data."""
        event_type = kwargs.pop('event_type', None)
        if event_type:
            extra = kwargs.get('extra', {})
            if isinstance(extra, dict):
                extra.update({'event_type': event_type})
            else:
                extra = {'event_type': event_type}
            kwargs['extra'] = extra
            
            # Move any other non-standard args to extra
            for k, v in list(kwargs.items()):
                if k not in ['exc_info', 'stack_info', 'stacklevel', 'extra']:
                    if 'extra' not in kwargs:
                        kwargs['extra'] = {}
                    kwargs['extra'][k] = v
                    del kwargs[k]
        
        super().error(msg, *args, **kwargs)
    
    def log_document_ingestion(self, source: str, num_chunks: int, metadata: Optional[Dict[str, Any]] = None):
        """Log document ingestion."""
        extra_data = {
            'source': source,
            'num_chunks': num_chunks
        }
        if metadata:
            extra_data.update(metadata)
        
        self.info(f"Ingested {num_chunks} chunks from {source}", 
                  event_type="document_ingestion", 
                  extra=extra_data)

# Register the custom logger class
logging.setLoggerClass(OuroLogger)

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