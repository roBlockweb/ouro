import os
import faiss
from config import FAISS_INDEX_PATH, EMBEDDING_DIM

def initialize_faiss_index():
    """
    Initializes or loads a FAISS index for storing embeddings.
    Ensures the storage directory exists, then either loads an existing index
    or creates a new one using the EMBEDDING_DIM from config.py.
    """
    # Ensure the directory for the index exists
    index_dir = os.path.dirname(FAISS_INDEX_PATH)
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
        print(f"📁 Created directory for FAISS index at {index_dir}")

    if os.path.exists(FAISS_INDEX_PATH):
        print("✅ FAISS index already exists. Loading existing index...")
        index = faiss.read_index(FAISS_INDEX_PATH)
    else:
        print("🚀 Creating a new FAISS index...")
        index = faiss.IndexFlatL2(EMBEDDING_DIM)  # Use L2 distance
        faiss.write_index(index, FAISS_INDEX_PATH)
        print("✅ New FAISS index created and saved.")

    return index

if __name__ == "__main__":
    # If you run this file directly, it will create/load the index
    index = initialize_faiss_index()