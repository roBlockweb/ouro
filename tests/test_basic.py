"""
Basic tests for the Ouro RAG system
"""
import os
import sys
import unittest
from pathlib import Path

# Import directly from the package
from src.document_loader import save_document, load_document, chunk_documents
from src.embeddings import EmbeddingManager
from src.config import DOCUMENTS_DIR

class BasicTest(unittest.TestCase):
    """Basic tests for core functionality"""
    
    def test_document_saving_and_loading(self):
        """Test document saving and loading"""
        # Create a test document
        content = "This is a test document for Ouro RAG system."
        filename = "test_document.txt"
        file_path = save_document(content, filename)
        
        # Verify the file exists
        self.assertTrue(file_path.exists())
        
        # Load the document
        documents = load_document(file_path)
        
        # Verify the document was loaded correctly
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, content)
        
        # Clean up
        os.remove(file_path)
    
    def test_document_chunking(self):
        """Test document chunking"""
        # Create a long test document
        content = "This is a test document. " * 200  # Repeat to make it long enough to chunk
        filename = "test_chunking.txt"
        file_path = save_document(content, filename)
        
        # Load the document
        documents = load_document(file_path)
        
        # Chunk the documents
        chunked_documents = chunk_documents(documents)
        
        # Verify chunking worked
        self.assertGreater(len(chunked_documents), 1)
        
        # Clean up
        os.remove(file_path)
    
    def test_embedding_manager_initialization(self):
        """Test embedding manager initialization"""
        # Initialize embedding manager with default model
        try:
            embedding_manager = EmbeddingManager()
            self.assertTrue(embedding_manager.vector_store is not None)
        except Exception as e:
            self.fail(f"EmbeddingManager initialization failed: {str(e)}")

if __name__ == '__main__':
    unittest.main()