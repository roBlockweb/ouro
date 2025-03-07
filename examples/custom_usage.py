"""
Example of how to use Ouro's components programmatically
"""
import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag import OuroRAG
from src.document_loader import save_document
from src.llm import list_available_models, login_huggingface

def main():
    """Example of programmatically using Ouro"""
    print("Ouro Programmatic Usage Example")
    print("-" * 50)
    
    # Initialize the RAG system
    print("Initializing RAG system...")
    rag = OuroRAG()
    
    # Create a test document
    print("Creating a test document...")
    content = """
    This is a test document created by the Ouro example script.
    It contains some sample text that will be embedded and retrieved.
    
    Ouro is a local RAG system that combines retrieval and generation.
    It works completely offline using models from Hugging Face.
    """
    file_path = save_document(content, "example_doc.txt")
    print(f"Saved document to {file_path}")
    
    # Ingest the document
    print("Ingesting the document...")
    rag.ingest_document(file_path)
    
    # Load an LLM model
    print("Available models:")
    for i, (name, path) in enumerate(list_available_models().items(), 1):
        print(f"  {i}. {name} ({path})")
    
    model_path = list(list_available_models().values())[0]  # Use the first model as example
    print(f"\nLoading model: {model_path}")
    rag.load_llm(model_path)
    
    # Query the system
    print("\nQuerying the system...")
    queries = [
        "What is Ouro?",
        "How does Ouro work?",
        "Where are documents stored in Ouro?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        response = rag.query(query)
        print(f"Response: {response}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()