"""
Document loading and processing utilities
"""
import os
from typing import List, Dict, Union, Optional, Any
from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

from src.config import DOCUMENTS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.logger import logger

# File type mappings
SUPPORTED_EXTENSIONS = {
    '.pdf': PyPDFLoader,
    '.txt': TextLoader,
    '.md': TextLoader,
    '.json': TextLoader,
    '.html': TextLoader,
    '.csv': TextLoader,
}

def load_document(file_path: Union[str, Path]) -> List[Document]:
    """
    Load a document from the specified file path
    
    Args:
        file_path: Path to the document
        
    Returns:
        List of Document objects
    """
    file_path = Path(file_path)
    logger.info(f"Loading document: {file_path}")
    
    try:
        # Ensure the file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get the file extension
        suffix = file_path.suffix.lower()
        
        # Check if file type is supported
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix}. Supported types: {', '.join(SUPPORTED_EXTENSIONS.keys())}")
        
        # Load the document using the appropriate loader
        loader_cls = SUPPORTED_EXTENSIONS[suffix]
        loader = loader_cls(str(file_path))
        
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} document segments from {file_path}")
        
        # Add source filename to metadata if not already present
        for doc in documents:
            if 'source' not in doc.metadata:
                doc.metadata['source'] = str(file_path)
        
        return documents
    
    except Exception as e:
        logger.error(f"Error loading document {file_path}: {str(e)}")
        raise

def load_documents_from_directory(directory: Union[str, Path] = DOCUMENTS_DIR) -> List[Document]:
    """
    Load all supported documents from a directory
    
    Args:
        directory: Directory containing documents
        
    Returns:
        List of Document objects
    """
    directory = Path(directory)
    logger.info(f"Loading documents from directory: {directory}")
    
    # Ensure the directory exists
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    all_documents = []
    
    try:
        # Process each supported file type
        for ext, loader_cls in SUPPORTED_EXTENSIONS.items():
            loader = DirectoryLoader(
                str(directory),
                glob=f"**/*{ext}",
                loader_cls=loader_cls
            )
            try:
                documents = loader.load()
                if documents:
                    logger.info(f"Loaded {len(documents)} {ext} documents from {directory}")
                    all_documents.extend(documents)
            except Exception as e:
                logger.warning(f"Error loading {ext} documents from {directory}: {str(e)}")
                continue
        
        return all_documents
    
    except Exception as e:
        logger.error(f"Error loading documents from {directory}: {str(e)}")
        raise

def chunk_documents(documents: List[Document], 
                   chunk_size: int = CHUNK_SIZE, 
                   chunk_overlap: int = CHUNK_OVERLAP,
                   **kwargs) -> List[Document]:
    """
    Split documents into smaller chunks for better embeddings
    
    Args:
        documents: List of Document objects
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        **kwargs: Additional arguments for the text splitter
        
    Returns:
        List of chunked Document objects
    """
    logger.info(f"Chunking {len(documents)} documents (size={chunk_size}, overlap={chunk_overlap})")
    
    if not documents:
        logger.warning("No documents to chunk")
        return []
    
    try:
        # Configure the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            **kwargs
        )
        
        # Split the documents
        chunked_documents = text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, doc in enumerate(chunked_documents):
            doc.metadata['chunk_id'] = i
            doc.metadata['total_chunks'] = len(chunked_documents)
        
        logger.info(f"Split {len(documents)} documents into {len(chunked_documents)} chunks")
        
        return chunked_documents
    
    except Exception as e:
        logger.error(f"Error chunking documents: {str(e)}")
        raise

def save_document(content: str, filename: str) -> Path:
    """
    Save a document to the documents directory
    
    Args:
        content: Text content to save
        filename: Name of the file
        
    Returns:
        Path to the saved document
    """
    # Ensure documents directory exists
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create the file path
    filepath = DOCUMENTS_DIR / filename
    logger.info(f"Saving document to {filepath}")
    
    try:
        # Write the content to the file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            
        logger.info(f"Document saved successfully to {filepath}")
        return filepath
    
    except Exception as e:
        logger.error(f"Error saving document to {filepath}: {str(e)}")
        raise