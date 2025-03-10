"""
Embeddings module for handling document embeddings and similarity search.
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import faiss
import numpy as np
import pickle
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

from ouro.config import (
    DEFAULT_EMBEDDING_MODEL,
    ALTERNATE_EMBEDDING_MODEL,
    VECTOR_STORE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    VECTOR_DB_TYPE,
    DEFAULT_METADATA
)
from ouro.logger import get_logger

logger = get_logger()


class EmbeddingManager:
    """Manager for document embeddings and similarity search."""
    
    def __init__(
        self, 
        embedding_model: Optional[str] = None, 
        chunk_size: int = CHUNK_SIZE, 
        chunk_overlap: int = CHUNK_OVERLAP,
        vector_db_type: str = VECTOR_DB_TYPE
    ):
        """Initialize the embedding manager.
        
        Args:
            embedding_model: Name or path to the embedding model
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between chunks
            vector_db_type: Type of vector database to use
        """
        self.model_name = embedding_model or DEFAULT_EMBEDDING_MODEL
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_db_type = vector_db_type
        
        # Initialize embeddings model
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                cache_folder=str(VECTOR_STORE_DIR)
            )
            
            # Initialize vector store directory
            os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
            self.index_dir = Path(VECTOR_STORE_DIR) / "faiss_index"
            os.makedirs(self.index_dir, exist_ok=True)
            
            # Check if index exists, otherwise create it
            if self._index_exists():
                self._load_index()
            else:
                self._create_empty_index()
                
            logger.info(f"Embedding manager initialized with model: {self.model_name}")
            logger.info(f"Vector index contains {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            raise
    
    def _index_exists(self) -> bool:
        """Check if the index files exist."""
        index_path = self.index_dir / "index.faiss"
        pkl_path = self.index_dir / "index.pkl"
        return index_path.exists() and pkl_path.exists()
    
    def _create_empty_index(self) -> None:
        """Create an empty FAISS index."""
        self.documents = []
        self.document_embeddings = None
        
        # Create an empty FAISS index
        embedding_size = len(self.embeddings.embed_query("test"))
        self.index = faiss.IndexFlatL2(embedding_size)
        
        # Save the empty index
        self._save_index()
        
        logger.info("Created empty FAISS index")
    
    def _load_index(self) -> None:
        """Load FAISS index and documents from disk."""
        try:
            # Load the index
            index_path = self.index_dir / "index.faiss"
            self.index = faiss.read_index(str(index_path))
            
            # Load the documents
            pkl_path = self.index_dir / "index.pkl"
            with open(pkl_path, "rb") as f:
                self.documents = pickle.load(f)
            
            logger.info(f"Loaded FAISS index with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            # Create a new index if loading fails
            self._create_empty_index()
    
    def _save_index(self) -> None:
        """Save FAISS index and documents to disk."""
        try:
            # Save the index
            index_path = self.index_dir / "index.faiss"
            faiss.write_index(self.index, str(index_path))
            
            # Save the documents
            pkl_path = self.index_dir / "index.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(self.documents, f)
            
            logger.info(f"Saved FAISS index with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def add_documents(
        self, 
        documents: List[Document], 
        show_progress: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            show_progress: Whether to show a progress bar
            metadata: Additional metadata to add to all documents
        """
        try:
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            chunks = []
            
            if show_progress:
                for doc in tqdm(documents, desc="Splitting documents"):
                    # Add default metadata if needed
                    if metadata:
                        # Start with defaults
                        doc_metadata = DEFAULT_METADATA.copy()
                        # Add document-specific metadata
                        if doc.metadata:
                            doc_metadata.update(doc.metadata)
                        # Add function-provided metadata
                        doc_metadata.update(metadata)
                        # Update document
                        doc.metadata = doc_metadata
                    elif not doc.metadata:
                        doc.metadata = DEFAULT_METADATA.copy()
                    
                    # Split document
                    doc_chunks = text_splitter.split_documents([doc])
                    chunks.extend(doc_chunks)
            else:
                for doc in documents:
                    # Add default metadata if needed
                    if metadata:
                        # Start with defaults
                        doc_metadata = DEFAULT_METADATA.copy()
                        # Add document-specific metadata
                        if doc.metadata:
                            doc_metadata.update(doc.metadata)
                        # Add function-provided metadata
                        doc_metadata.update(metadata)
                        # Update document
                        doc.metadata = doc_metadata
                    elif not doc.metadata:
                        doc.metadata = DEFAULT_METADATA.copy()
                    
                    # Split document
                    doc_chunks = text_splitter.split_documents([doc])
                    chunks.extend(doc_chunks)
            
            if not chunks:
                logger.warning("No chunks created from documents")
                return
            
            # Generate embeddings
            texts = [doc.page_content for doc in chunks]
            
            if show_progress:
                logger.info(f"Generating embeddings for {len(texts)} chunks")
                # No tqdm for embedding generation as it's usually fast
            
            embeddings = self.embeddings.embed_documents(texts)
            
            # Add to index
            if show_progress:
                logger.info(f"Adding {len(embeddings)} embeddings to index")
            
            if not hasattr(self, 'documents'):
                self.documents = []
            
            # Create a numpy array from the embeddings
            embeddings_array = np.array(embeddings).astype('float32')
            
            # Add to FAISS index
            self.index.add(embeddings_array)
            
            # Add documents to collection
            self.documents.extend(chunks)
            
            # Save updated index
            self._save_index()
            
            logger.info(f"Added {len(chunks)} chunks to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        filter: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0
    ) -> List[Document]:
        """Search for documents similar to the query.
        
        Args:
            query: Query string
            k: Number of documents to return
            filter: Metadata filter
            min_score: Minimum similarity score threshold
            
        Returns:
            List of similar documents
        """
        try:
            # Check if we have any documents
            if not hasattr(self, 'documents') or not self.documents:
                logger.warning("No documents in vector store")
                return []
            
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Convert to numpy array
            query_embedding_array = np.array([query_embedding]).astype('float32')
            
            # Search FAISS index
            k = min(k, len(self.documents))  # Don't try to get more docs than we have
            if k == 0:
                return []
                
            # Get distances and indices
            distances, indices = self.index.search(query_embedding_array, k)
            
            # Process results
            results = []
            for i, idx in enumerate(indices[0]):
                # Convert distance to similarity score (FAISS returns L2 distance)
                # Lower distance means higher similarity
                # We convert to a 0-1 score where 1 is most similar
                distance = distances[0][i]
                
                # Skip if index is out of bounds (happens with small document sets)
                if idx >= len(self.documents):
                    continue
                
                # Get document
                doc = self.documents[idx]
                
                # Apply metadata filtering
                if filter and not all(doc.metadata.get(key) == value for key, value in filter.items()):
                    continue
                
                # Calculate similarity score (normalized)
                # This is a rough approximation - actual normalization depends on embedding dimension
                similarity = 1.0 / (1.0 + distance)
                
                # Check minimum score
                if similarity < min_score:
                    continue
                
                # Add similarity score to metadata
                doc.metadata["similarity_score"] = similarity
                
                # Add to results
                results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def get_document_count(self) -> int:
        """Get the number of documents in the vector store."""
        if hasattr(self, 'documents'):
            return len(self.documents)
        return 0
    
    def clear(self) -> None:
        """Clear the vector store."""
        try:
            # Create a new empty index
            embedding_size = len(self.embeddings.embed_query("test"))
            self.index = faiss.IndexFlatL2(embedding_size)
            self.documents = []
            
            # Save the empty index
            self._save_index()
            
            logger.info("Cleared vector store")
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            raise
    
    def change_embedding_model(self, model_name: str) -> None:
        """Change the embedding model.
        
        Args:
            model_name: Name or path to the new embedding model
        """
        try:
            # Save current index
            self._save_index()
            
            # Initialize new embeddings model
            self.model_name = model_name
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                cache_folder=str(VECTOR_STORE_DIR)
            )
            
            logger.info(f"Changed embedding model to {self.model_name}")
            
            # We would need to re-embed all documents with the new model
            # This is a costly operation, so we warn the user
            logger.warning("Embedding model changed, but existing embeddings are not updated. Consider rebuilding the index.")
            
        except Exception as e:
            logger.error(f"Error changing embedding model: {e}")
            raise