import os
import logging
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import FAISS_INDEX_PATH, EMBEDDING_DIM, LOG_DIR

# Configure logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, 'initialization.log'), 'a')
    ]
)
logger = logging.getLogger(__name__)

def initialize_faiss_index():
    """
    Initializes or loads a FAISS index for storing embeddings.
    Ensures the storage directory exists, then either loads an existing index
    or creates a new one using the EMBEDDING_DIM from config.py.
    
    Returns:
        faiss.Index: The loaded or newly created FAISS index
    """
    # Ensure the directory for the index exists
    index_dir = os.path.dirname(FAISS_INDEX_PATH)
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
        print(f"📁 Created directory for FAISS index at {index_dir}")
        logger.info(f"Created directory for FAISS index at {index_dir}")

    # Check if index already exists
    if os.path.exists(FAISS_INDEX_PATH):
        print("✅ FAISS index already exists. Loading existing index...")
        logger.info("Loading existing FAISS index")
        try:
            index = faiss.read_index(FAISS_INDEX_PATH)
            print(f"👍 Loaded FAISS index with {index.ntotal} vectors.")
            logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
            return index
        except Exception as e:
            print(f"❌ Error loading existing index: {str(e)}")
            logger.error(f"Error loading existing index: {str(e)}")
            print("🔄 Creating a new index instead...")
            logger.info("Creating a new index instead")
    else:
        print("🚀 Creating a new FAISS index...")
        logger.info("Creating a new FAISS index")

    # Create a new index
    index = faiss.IndexFlatL2(EMBEDDING_DIM)  # Use L2 distance for similarity
    
    # Seed the index with some starter embeddings
    if seed_with_starter_content(index):
        print("✅ Index seeded with starter content.")
        logger.info("Index seeded with starter content")
    
    # Save the index to disk
    try:
        faiss.write_index(index, FAISS_INDEX_PATH)
        print(f"✅ New FAISS index created and saved with {index.ntotal} vectors.")
        logger.info(f"New FAISS index created and saved with {index.ntotal} vectors")
    except Exception as e:
        print(f"❌ Error saving index: {str(e)}")
        logger.error(f"Error saving index: {str(e)}")
    
    return index

def seed_with_starter_content(index):
    """
    Seeds the FAISS index with some starter content to bootstrap the system.
    
    Args:
        index (faiss.Index): The FAISS index to seed
        
    Returns:
        bool: True if seeding was successful, False otherwise
    """
    try:
        print("🌱 Seeding index with starter content...")
        logger.info("Seeding index with starter content")
        
        # Load the embedding model
        print("📦 Loading SentenceTransformer model...")
        start_time = time.time()
        model = SentenceTransformer('all-MiniLM-L6-v2')
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        logger.info(f"SentenceTransformer model loaded in {load_time:.2f} seconds")
        
        # Prepare sample content about technology, AI, and software
        starter_texts = [
            "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans.",
            "Machine learning is a subset of AI focused on giving computers the ability to learn without being explicitly programmed.",
            "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
            "Neural networks are computing systems inspired by the biological neural networks that constitute animal brains.",
            "Natural language processing (NLP) is a subfield of linguistics, computer science, and AI focused on the interactions between computers and human language.",
            "Blockchain technology is a distributed ledger system that enables secure, transparent and tamper-proof record-keeping.",
            "Cryptocurrencies are digital or virtual currencies that use cryptography for security and operate on decentralized networks.",
            "Cloud computing is the on-demand availability of computer system resources, especially data storage and computing power.",
            "Edge computing is a distributed computing paradigm that brings computation and data storage closer to the sources of data.",
            "The Internet of Things (IoT) describes physical objects with sensors, processing ability, software, and other technologies that connect and exchange data with other devices.",
            "Quantum computing is the exploitation of collective properties of quantum states to perform computation.",
            "Augmented reality (AR) is an interactive experience that combines the real world and computer-generated content.",
            "Virtual reality (VR) is a simulated experience that can be similar to or completely different from the real world.",
            "A large language model (LLM) is a type of AI model designed to understand and generate natural language text.",
            "Autonomous systems are capable of performing tasks with minimal human intervention or oversight.",
            "Robotics involves design, construction, operation, and use of robots, as well as computer systems for their control, sensory feedback, and information processing.",
            "Software development is the process of conceiving, specifying, designing, programming, documenting, testing, and bug fixing involved in creating software.",
            "Cybersecurity is the practice of protecting systems, networks, and programs from digital attacks.",
            "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms to extract knowledge from structured and unstructured data.",
            "Web3 refers to the concept of a decentralized web based on blockchain technology and token-based economics."
        ]
        
        # Embed the sample texts
        print(f"🔄 Generating embeddings for {len(starter_texts)} texts...")
        embeddings = model.encode(starter_texts, convert_to_numpy=True)
        
        # Add embeddings to the index
        index.add(embeddings)
        print(f"✅ Added {len(starter_texts)} starter embeddings to index")
        logger.info(f"Added {len(starter_texts)} starter embeddings to index")
        
        # Save texts alongside index for retrieval
        texts_path = FAISS_INDEX_PATH + ".texts"
        with open(texts_path, 'w', encoding='utf-8') as f:
            for text in starter_texts:
                f.write(text + "\n===TEXT_SEPARATOR===\n")
        
        print(f"✅ Saved starter texts to {texts_path}")
        logger.info(f"Saved starter texts to {texts_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error seeding index: {str(e)}")
        logger.error(f"Error seeding index: {str(e)}")
        return False

def test_index():
    """
    Tests the FAISS index with a sample query to verify it's working.
    """
    try:
        print("\n🔍 Testing index with a sample query...")
        logger.info("Testing index with a sample query")
        
        # Load the model and the index
        model = SentenceTransformer('all-MiniLM-L6-v2')
        index = faiss.read_index(FAISS_INDEX_PATH)
        
        # Create a sample query
        query = "How does artificial intelligence work?"
        
        # Embed the query
        query_embedding = model.encode([query], convert_to_numpy=True)
        
        # Search the index
        k = 3  # Number of results to return
        distances, indices = index.search(query_embedding, k)
        
        print(f"✅ Query: '{query}'")
        print(f"✅ Found {len(indices[0])} matches")
        
        # Display matching texts if available
        texts_path = FAISS_INDEX_PATH + ".texts"
        if os.path.exists(texts_path):
            with open(texts_path, 'r', encoding='utf-8') as f:
                content = f.read()
                texts = content.split("\n===TEXT_SEPARATOR===\n")
                if texts and texts[-1] == '':
                    texts.pop()  # Remove empty last element
                    
            print("\nMatching texts:")
            for i, idx in enumerate(indices[0]):
                if idx < len(texts):
                    print(f"{i+1}. [{distances[0][i]:.4f}] {texts[idx]}")
                    
        logger.info("Index test completed successfully")
        
    except Exception as e:
        print(f"❌ Error testing index: {str(e)}")
        logger.error(f"Error testing index: {str(e)}")

if __name__ == "__main__":
    # If you run this file directly, it will create/load the index
    print("🚀 Initializing FAISS vector store for Ouro and Brain...")
    logger.info("Starting FAISS initialization")
    
    index = initialize_faiss_index()
    test_index()
    
    print("\n✅ FAISS initialization complete. Ready for use by Ouro and Brain.")
    logger.info("FAISS initialization complete")