"""
Embeddings and vector storage management for the Ouro RAG system.
"""
import os
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ouro.config import VECTOR_STORE_DIR, DEFAULT_EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from ouro.logger import get_logger

logger = get_logger()


class EmbeddingManager:
    """Manages document embeddings and vector storage."""
    
    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        vector_dir: str = VECTOR_STORE_DIR,
    ):
        """Initialize the embedding manager."""
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_dir = vector_dir
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize or load vector store
        os.makedirs(vector_dir, exist_ok=True)
        self.vector_index_path = os.path.join(vector_dir, "faiss_index.bin")
        self.document_store_path = os.path.join(vector_dir, "documents.npy")
        
        self.load_or_create_vector_store()
    
    def load_or_create_vector_store(self) -> None:
        """Load existing vector store or create a new one."""
        if os.path.exists(self.vector_index_path) and os.path.exists(self.document_store_path):
            logger.info("Loading existing vector store")
            self.index = faiss.read_index(self.vector_index_path)
            self.documents = np.load(self.document_store_path, allow_pickle=True).tolist()
            logger.info(f"Loaded {len(self.documents)} documents")
        else:
            logger.info("Creating new vector store")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.documents = []
    
    def save_vector_store(self) -> None:
        """Save vector store to disk."""
        logger.info("Saving vector store")
        faiss.write_index(self.index, self.vector_index_path)
        np.save(self.document_store_path, np.array(self.documents, dtype=object))
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        return self.text_splitter.split_documents(documents)
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts."""
        logger.info(f"Creating embeddings for {len(texts)} texts")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def add_documents(self, documents: List[Document], show_progress: bool = True) -> None:
        """Add documents to the vector store."""
        # Chunk documents
        chunked_docs = self.chunk_documents(documents)
        logger.info(f"Adding {len(chunked_docs)} document chunks to vector store")
        
        # Create embeddings
        texts = [doc.page_content for doc in chunked_docs]
        embeddings = self.create_embeddings(texts)
        
        # Add to vector store
        self.index.add(np.array(embeddings).astype('float32'))
        self.documents.extend(chunked_docs)
        
        # Save updated vector store
        self.save_vector_store()
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for documents similar to the query."""
        if not self.documents:
            logger.warning("Vector store is empty. No documents to search.")
            return []
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])[0].reshape(1, -1).astype('float32')
        
        # Search index
        distances, indices = self.index.search(query_embedding, k=min(k, len(self.documents)))
        
        # Return documents
        return [self.documents[i] for i in indices[0] if i < len(self.documents) and i >= 0]