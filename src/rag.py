"""
Core RAG (Retrieval-Augmented Generation) system
"""
from typing import List, Dict, Any, Optional, Union, Callable
from pathlib import Path

from src.document_loader import (
    load_document, 
    load_documents_from_directory,
    chunk_documents, 
    save_document
)
from src.embeddings import EmbeddingManager
from src.llm import LocalLLM
from src.logger import logger
from src.config import TOP_K_RESULTS

class OuroRAG:
    """Main controller for the Ouro RAG system"""
    
    def __init__(self, 
                 embedding_manager: Optional[EmbeddingManager] = None,
                 llm: Optional[LocalLLM] = None,
                 embedding_model: str = None, 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RAG system
        
        Args:
            embedding_manager: Pre-initialized embedding manager
            llm: Pre-initialized LLM
            embedding_model: Name of the embedding model (used if embedding_manager not provided)
            config: Configuration dictionary for the RAG system
        """
        logger.info("Initializing OuroRAG system")
        
        # Initialize embedding manager if not provided
        self.embedding_manager = embedding_manager or EmbeddingManager(embedding_model)
        
        # Store LLM if provided
        self.llm = llm
        
        # Store configuration
        self.config = config or {}
    
    def load_llm(self, model_name: str, progress_callback: Callable = None) -> None:
        """
        Load an LLM model
        
        Args:
            model_name: Name of the LLM model to load
            progress_callback: Optional callback for progress updates
        """
        logger.info(f"Loading LLM: {model_name}")
        if self.llm is None:
            self.llm = LocalLLM()
            self.llm.load_model(model_name, progress_callback)
        else:
            self.llm.load_model(model_name, progress_callback)
    
    def ingest_document(self, 
                        file_path: Union[str, Path], 
                        progress_callback: Callable = None) -> None:
        """
        Ingest a document into the RAG system
        
        Args:
            file_path: Path to the document
            progress_callback: Optional callback for progress updates
        """
        logger.info(f"Ingesting document: {file_path}")
        
        try:
            # Update progress
            if progress_callback:
                progress_callback(f"Loading document: {file_path}", 0.1)
            
            # Load the document
            documents = load_document(file_path)
            
            # Update progress
            if progress_callback:
                progress_callback("Chunking document...", 0.3)
            
            # Chunk the document
            chunking_config = {
                "chunk_size": self.config.get("chunk_size", 1000),
                "chunk_overlap": self.config.get("chunk_overlap", 200),
            }
            chunked_documents = chunk_documents(documents, **chunking_config)
            
            # Update progress
            if progress_callback:
                progress_callback(f"Creating embeddings for {len(chunked_documents)} chunks...", 0.5)
            
            # Add to vector store - pass the progress callback for additional updates
            self.embedding_manager.progress_callback = progress_callback
            self.embedding_manager.add_documents(chunked_documents, show_progress=True)
            
            logger.info(f"Successfully ingested document: {file_path}")
            
            # Final progress update
            if progress_callback:
                progress_callback("Document successfully ingested", 1.0)
        
        except Exception as e:
            logger.error(f"Error ingesting document: {str(e)}")
            if progress_callback:
                progress_callback(f"Error: {str(e)}", -1.0)
            raise
    
    def ingest_directory(self, 
                         directory: Union[str, Path], 
                         progress_callback: Callable = None) -> None:
        """
        Ingest all documents in a directory
        
        Args:
            directory: Directory containing documents
            progress_callback: Optional callback for progress updates
        """
        logger.info(f"Ingesting documents from directory: {directory}")
        
        try:
            # Update progress
            if progress_callback:
                progress_callback(f"Scanning directory: {directory}", 0.1)
            
            # Load all documents
            documents = load_documents_from_directory(directory)
            
            if not documents:
                logger.warning(f"No documents found in directory: {directory}")
                if progress_callback:
                    progress_callback(f"No documents found in {directory}", -1.0)
                return
            
            # Update progress
            if progress_callback:
                progress_callback(f"Chunking {len(documents)} documents...", 0.3)
            
            # Chunk the documents
            chunking_config = {
                "chunk_size": self.config.get("chunk_size", 1000),
                "chunk_overlap": self.config.get("chunk_overlap", 200),
            }
            chunked_documents = chunk_documents(documents, **chunking_config)
            
            # Update progress
            if progress_callback:
                progress_callback(f"Creating embeddings for {len(chunked_documents)} chunks...", 0.5)
            
            # Add to vector store - pass the progress callback for additional updates
            self.embedding_manager.progress_callback = progress_callback
            self.embedding_manager.add_documents(chunked_documents, show_progress=True)
            
            logger.info(f"Successfully ingested {len(documents)} documents from directory")
            
            # Final progress update
            if progress_callback:
                progress_callback("Documents successfully ingested", 1.0)
        
        except Exception as e:
            logger.error(f"Error ingesting documents from directory: {str(e)}")
            if progress_callback:
                progress_callback(f"Error: {str(e)}", -1.0)
            raise
    
    def save_and_ingest_text(self, 
                            content: str, 
                            filename: str, 
                            progress_callback: Callable = None) -> None:
        """
        Save text content to a file and ingest it
        
        Args:
            content: Text content
            filename: Name of the file to save
            progress_callback: Optional callback for progress updates
        """
        logger.info(f"Saving and ingesting text as {filename}")
        
        try:
            # Update progress
            if progress_callback:
                progress_callback(f"Saving text as {filename}", 0.1)
            
            # Save the document
            file_path = save_document(content, filename)
            
            # Update progress
            if progress_callback:
                progress_callback(f"Starting ingestion of {filename}", 0.3)
            
            # Ingest the document with progress tracking
            self.ingest_document(
                file_path, 
                progress_callback=lambda msg, pct: progress_callback(
                    msg, 
                    0.3 + (pct * 0.7)  # Scale progress to 30%-100% range
                ) if pct >= 0 else progress_callback(msg, pct)
            )
            
            logger.info(f"Successfully saved and ingested text as {filename}")
        
        except Exception as e:
            logger.error(f"Error saving and ingesting text: {str(e)}")
            if progress_callback:
                progress_callback(f"Error: {str(e)}", -1.0)
            raise
    
    def query(self, 
             query_text: str, 
             k: int = None,
             search_progress_callback: Callable = None,
             generation_progress_callback: Callable = None) -> str:
        """
        Process a query through the RAG pipeline with progress tracking
        
        Args:
            query_text: User query
            k: Number of results to retrieve (uses config value if None)
            search_progress_callback: Optional callback for search progress updates
            generation_progress_callback: Optional callback for generation progress updates
            
        Returns:
            Generated response
        """
        logger.info(f"Processing query: '{query_text}'")
        
        if not self.llm:
            raise ValueError("LLM not loaded. Please load an LLM model first.")
        
        # Use config value for k if not specified
        if k is None:
            k = self.config.get("top_k", TOP_K_RESULTS)
        
        try:
            # Start search progress
            if search_progress_callback:
                search_progress_callback("Searching knowledge base...", 0.1)
            
            # Retrieve context documents
            retrieved_docs = self.embedding_manager.similarity_search(query_text, k=k)
            
            # Extract content from retrieved documents
            context_texts = [doc.page_content for doc in retrieved_docs]
            
            # Update search progress
            if search_progress_callback:
                search_progress_callback(f"Found {len(retrieved_docs)} relevant document chunks", 1.0)
            
            # Start generation progress
            if generation_progress_callback:
                generation_progress_callback("Generating response...", 0.1)
            
            # Store original callback to restore later
            original_callback = self.llm.progress_callback
            self.llm.progress_callback = generation_progress_callback
            
            # Generate response with context
            response = self.llm.generate_with_retrieval(query_text, context_texts)
            
            # Restore original callback
            self.llm.progress_callback = original_callback
            
            # Update generation progress
            if generation_progress_callback:
                generation_progress_callback("Response generated", 1.0)
            
            logger.info("Successfully processed query")
            return response
        
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            if search_progress_callback:
                search_progress_callback(f"Error: {str(e)}", -1.0)
            if generation_progress_callback:
                generation_progress_callback(f"Error: {str(e)}", -1.0)
            raise