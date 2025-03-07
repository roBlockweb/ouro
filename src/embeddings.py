"""
Embeddings and vector store functionality
"""
import os
import pickle
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from pathlib import Path

import numpy as np
import faiss
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, SpinnerColumn

from src.config import (
    DEFAULT_EMBEDDING_MODEL, 
    VECTOR_STORE_DIR, 
    MODEL_CACHE_DIR,
    TOP_K_RESULTS,
    EMBEDDING_MODELS
)
from src.logger import logger

class EmbeddingManager:
    """Manager for embeddings and vector store operations"""
    
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, progress_callback: Callable = None):
        """
        Initialize the embedding manager
        
        Args:
            model_name: Name of the HuggingFace embedding model
            progress_callback: Optional callback for progress updates
        """
        self.model_name = model_name
        self.vector_store_path = VECTOR_STORE_DIR / "faiss_index"
        self.progress_callback = progress_callback
        
        logger.info(f"Initializing embedding manager with model: {model_name}")
        
        if self.progress_callback:
            self.progress_callback("Loading embedding model...", 0.1)
        
        # Initialize the embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=str(MODEL_CACHE_DIR),
            encode_kwargs={"normalize_embeddings": True}
        )
        
        if self.progress_callback:
            self.progress_callback("Loading vector store...", 0.5)
        
        # Initialize or load the vector store
        self._load_or_create_vector_store()
        
        if self.progress_callback:
            self.progress_callback("Embedding system ready", 1.0)
    
    def _load_or_create_vector_store(self):
        """Load an existing vector store or create a new one"""
        if self.vector_store_exists():
            logger.info(f"Loading existing vector store from {self.vector_store_path}")
            try:
                self.vector_store = FAISS.load_local(
                    str(self.vector_store_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Vector store loaded successfully")
            except Exception as e:
                logger.error(f"Error loading vector store: {str(e)}")
                logger.info("Creating new vector store")
                self._create_new_vector_store()
        else:
            logger.info("No existing vector store found. Creating new one.")
            self._create_new_vector_store()
    
    def _create_new_vector_store(self):
        """Create a new empty vector store"""
        # Create an empty index with a sample document
        empty_docs = [Document(page_content="Ouro initialization document", metadata={"source": "system"})]
        
        self.vector_store = FAISS.from_documents(
            empty_docs,
            self.embeddings
        )
        
        # Save the empty vector store
        self.save_vector_store()
        
        logger.info("Created and saved new empty vector store")
    
    def vector_store_exists(self) -> bool:
        """Check if a vector store exists"""
        index_path = self.vector_store_path / "index.faiss"
        docstore_path = self.vector_store_path / "index.pkl"
        return index_path.exists() and docstore_path.exists()
    
    def add_documents(self, documents: List[Document], show_progress: bool = True) -> None:
        """
        Add documents to the vector store
        
        Args:
            documents: List of Document objects to add
            show_progress: Whether to show a progress bar
        """
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        try:
            if show_progress and self.progress_callback:
                self.progress_callback(f"Creating embeddings for {len(documents)} documents...", 0.3)
                
            self.vector_store.add_documents(documents)
            
            if show_progress and self.progress_callback:
                self.progress_callback("Saving vector store...", 0.8)
                
            logger.info(f"Successfully added {len(documents)} documents to vector store")
            self.save_vector_store()
            
            if show_progress and self.progress_callback:
                self.progress_callback("Documents successfully added to knowledge base", 1.0)
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            if show_progress and self.progress_callback:
                self.progress_callback(f"Error: {str(e)}", -1.0)
            raise
    
    def similarity_search(self, query: str, k: int = TOP_K_RESULTS) -> List[Document]:
        """
        Perform a similarity search for a query
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of Document objects
        """
        logger.info(f"Performing similarity search for query: '{query}'")
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} results for query")
            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            raise
    
    def save_vector_store(self) -> None:
        """Save the current vector store to disk"""
        logger.info(f"Saving vector store to {self.vector_store_path}")
        
        try:
            # Ensure directory exists
            self.vector_store_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.vector_store.save_local(str(self.vector_store_path))
            logger.info("Vector store saved successfully")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise

def load_embedding_with_progress(model_name: str) -> Tuple[EmbeddingManager, Any]:
    """
    Load embedding model with progress display
    
    Args:
        model_name: Name of the embedding model
        
    Returns:
        Tuple of (EmbeddingManager instance, dimension info)
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[green]Loading embedding model...", total=100)
        
        def update_progress(message, percent):
            if percent < 0:  # Error
                progress.update(task, description=f"[bold red]{message}")
                return
                
            progress.update(
                task, 
                description=f"[bold green]{message}",
                completed=int(percent * 100)
            )
        
        embedding_manager = EmbeddingManager(model_name, progress_callback=update_progress)
    
    return embedding_manager

def get_available_embedding_models() -> Dict[str, str]:
    """
    Return the list of available embedding models
    
    Returns:
        Dictionary of embedding model names and their Hugging Face paths
    """
    return EMBEDDING_MODELS