"""
Document loading and processing for the Ouro RAG system.
"""
import os
from typing import List, Dict, Any, Union
from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    UnstructuredMarkdownLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
    JSONLoader
)
from langchain_core.documents import Document

from ouro.config import DOCUMENTS_DIR
from ouro.logger import get_logger

logger = get_logger()

# File extension to loader mapping
LOADER_MAPPING = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".md": UnstructuredMarkdownLoader,
    ".csv": CSVLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
    ".json": lambda path: JSONLoader(
        file_path=path,
        jq_schema='.[]',
        text_content=False
    )
}


def load_document(file_path: Union[str, Path]) -> List[Document]:
    """
    Load a document from a file path based on its extension.
    
    Args:
        file_path: Path to the document
        
    Returns:
        List of Document objects
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"File does not exist: {file_path}")
        return []
    
    file_extension = file_path.suffix.lower()
    
    if file_extension not in LOADER_MAPPING:
        logger.warning(f"Unsupported file format: {file_extension}")
        return []
    
    try:
        logger.info(f"Loading document: {file_path}")
        loader_class = LOADER_MAPPING[file_extension]
        
        if callable(loader_class):
            loader = loader_class(str(file_path))
            documents = loader.load()
            return documents
        else:
            logger.error(f"Invalid loader configuration for {file_extension}")
            return []
            
    except Exception as e:
        logger.error(f"Error loading document {file_path}: {e}")
        return []


def load_documents_from_directory(directory: Union[str, Path]) -> List[Document]:
    """
    Load all documents from a directory.
    
    Args:
        directory: Path to the directory
        
    Returns:
        List of Document objects
    """
    directory = Path(directory)
    
    if not directory.exists() or not directory.is_dir():
        logger.error(f"Directory does not exist: {directory}")
        return []
    
    documents = []
    
    logger.info(f"Loading documents from directory: {directory}")
    
    for file_path in directory.glob("**/*"):
        if file_path.is_file() and file_path.suffix.lower() in LOADER_MAPPING:
            try:
                file_documents = load_document(file_path)
                if file_documents:
                    documents.extend(file_documents)
                    logger.info(f"Loaded: {file_path.name} ({len(file_documents)} documents)")
            except Exception as e:
                logger.error(f"Error loading document {file_path}: {e}")
    
    logger.info(f"Loaded {len(documents)} documents from {directory}")
    return documents


def load_text(text: str, metadata: Dict[str, Any] = None) -> List[Document]:
    """
    Create a document from text.
    
    Args:
        text: Text content
        metadata: Optional metadata
        
    Returns:
        List containing a single Document object
    """
    if not metadata:
        metadata = {"source": "user_input"}
    
    return [Document(page_content=text, metadata=metadata)]