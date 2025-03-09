#!/usr/bin/env python
"""
Test script for Ouro RAG system.
Verifies basic functionality of all components.
"""
import os
import sys
import unittest
import tempfile
from pathlib import Path

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ouro.config import MODELS
from ouro.llm import check_hf_login, LocalLLM
from ouro.embeddings import EmbeddingManager
from ouro.document_loader import load_text
from ouro.rag import OuroRAG


class OuroTestCase(unittest.TestCase):
    """Test case for Ouro components."""
    
    def setUp(self):
        """Set up test environment."""
        print("\nRunning tests for Ouro RAG system...")
        
        # Verify Hugging Face login
        self.assertTrue(check_hf_login(), "Hugging Face login required for tests")
        
        # Use a temporary directory for test data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        
        # Create test text
        self.test_text = """
        Ouro is a privacy-first local RAG system.
        It runs completely offline on your machine.
        Your data never leaves your computer.
        """
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_llm(self):
        """Test LLM loading and generation."""
        print("Testing LLM...")
        try:
            # Use smallest model for testing
            model_config = MODELS["small"]
            llm = LocalLLM(model_config=model_config)
            self.assertIsNotNone(llm.model, "Model should be loaded")
            self.assertIsNotNone(llm.tokenizer, "Tokenizer should be loaded")
            
            # Test generation
            response_generator = llm.generate(
                system_prompt="You are a helpful AI.",
                query="Say hi!",
                stream=False
            )
            response = "".join(list(response_generator))
            self.assertIsInstance(response, str, "Response should be a string")
            self.assertGreater(len(response), 0, "Response should not be empty")
            print("✓ LLM test passed")
        except Exception as e:
            self.fail(f"LLM test failed: {e}")
    
    def test_embeddings(self):
        """Test embedding creation and similarity search."""
        print("Testing embeddings...")
        try:
            embedding_manager = EmbeddingManager()
            self.assertIsNotNone(embedding_manager.embedding_model, "Embedding model should be loaded")
            
            # Create test documents
            docs = load_text(self.test_text)
            self.assertGreater(len(docs), 0, "Should create at least one document")
            
            # Add documents to vector store
            embedding_manager.add_documents(docs)
            
            # Test similarity search
            results = embedding_manager.similarity_search("privacy")
            self.assertGreater(len(results), 0, "Should find at least one similar document")
            print("✓ Embeddings test passed")
        except Exception as e:
            self.fail(f"Embeddings test failed: {e}")
    
    def test_rag_system(self):
        """Test the full RAG system."""
        print("Testing RAG system...")
        try:
            # Initialize RAG with small model
            rag = OuroRAG(model_config=MODELS["small"])
            
            # Ingest test text
            rag.ingest_text(self.test_text)
            
            # Test query
            response_generator = rag.generate("What is Ouro?", stream=False)
            response = "".join(list(response_generator))
            self.assertIsInstance(response, str, "Response should be a string")
            self.assertGreater(len(response), 0, "Response should not be empty")
            print("✓ RAG system test passed")
        except Exception as e:
            self.fail(f"RAG system test failed: {e}")


if __name__ == "__main__":
    # Run the tests
    unittest.main()