import os
import faiss
import numpy as np
import datetime
from sentence_transformers import SentenceTransformer
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from config import EMBEDDING_DIM, FAISS_INDEX_PATH, SELENIUM_HEADLESS, SELENIUM_DRIVER_PATH

# ===========================================
# Sentence Embedding Utility
# ===========================================
class SentenceTransformerEmbedder:
    def __init__(self):
        print("🔗 Loading SentenceTransformer Model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_text(self, texts):
        """
        Converts a list of text strings into embeddings (numpy arrays).
        """
        return self.model.encode(texts, convert_to_numpy=True)

# ===========================================
# FAISS Index Utility
# ===========================================
class FAISSStore:
    def __init__(self, index_path=FAISS_INDEX_PATH, dimension=EMBEDDING_DIM):
        self.index_path = index_path
        self.dimension = dimension

        if os.path.exists(self.index_path):
            print("📂 Loading existing FAISS index...")
            self.index = faiss.read_index(self.index_path)
        else:
            print("🚀 Creating a new FAISS index...")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.save_index()

    def save_index(self):
        print("💾 Saving FAISS index...")
        faiss.write_index(self.index, self.index_path)

    def add_embeddings(self, embeddings):
        """
        Adds embeddings to the FAISS index.
        embeddings: numpy array with shape (n, dimension).
        """
        print("➕ Adding embeddings to FAISS index...")
        self.index.add(embeddings)

    def search(self, query_embedding, top_k=5):
        """
        Searches the FAISS index for nearest embeddings.
        query_embedding: numpy array of shape (1, dimension).
        Returns (distances, indices).
        """
        print("🔍 Searching FAISS index...")
        distances, indices = self.index.search(query_embedding, top_k)
        return distances, indices

    def remove_duplicates(self, distance_threshold=0.5):
        """
        Naively removes vectors that are duplicates based on L2 distance.
        Rebuilds the entire index, skipping vectors that are 'too close.'
        This can be expensive for large indexes.
        
        :param distance_threshold: If the distance between two embeddings is below this,
                                   we consider them duplicates and remove the latter.
        """
        print(f"🧹 Removing duplicates using distance threshold {distance_threshold} ...")
        # 1) Extract all vectors
        index_size = self.index.ntotal
        if index_size < 2:
            print("No duplicates to remove, index too small.")
            return

        # We need to read out all embeddings. IndexFlat can't remove in place.
        # We'll store them in a new array.
        vectors = self._retrieve_all_embeddings()

        # 2) Rebuild in memory: keep a list of unique vectors
        unique_vectors = []
        for i, vec in enumerate(vectors):
            # check if it's near any that we kept
            # We'll do a quick L2 distance check
            # If we find any existing vector with distance < threshold, skip
            keep_it = True
            for uvec in unique_vectors:
                dist = np.sum((vec - uvec)**2)
                if dist < distance_threshold:
                    keep_it = False
                    break
            if keep_it:
                unique_vectors.append(vec)

        # 3) Rebuild the FAISS index from these unique vectors
        new_index = faiss.IndexFlatL2(self.dimension)
        if unique_vectors:
            arr = np.array(unique_vectors, dtype=np.float32)
            new_index.add(arr)
        self.index = new_index
        self.save_index()

        old_count = index_size
        new_count = len(unique_vectors)
        removed = old_count - new_count
        print(f"🧹 Duplicate removal done. Removed {removed} vectors. Final count: {new_count}")

    def _retrieve_all_embeddings(self):
        """
        Grabs all embeddings from the index. For an IndexFlatL2, we can't do partial
        ID-based lookups, so we do a brute force approach to read them.
        """
        # We can do a range search if we expand. But a simpler approach is to do
        # 'inverted files' or 'ID map'. But since it's IndexFlat, let's do a hack:
        # We'll do a "search" from each unit vector? That might be insane for large indexes...
        # Instead, let's store the vectors ourselves whenever we add them. A better approach:
        # For demonstration only, we'll do a read from memory approach with faiss.IndexFlat
        # Actually, IndexFlat doesn't store original vectors in an accessible way by default.
        # We'll rely on the 'reconstruct' function, which is available in certain faiss versions.

        ntotal = self.index.ntotal
        if ntotal == 0:
            return np.zeros((0, self.dimension), dtype=np.float32)

        # We can reconstruct each vector by ID
        # This may be slow for large indices, but demonstration only
        vectors = []
        for i in range(ntotal):
            vec = np.zeros(self.dimension, dtype=np.float32)
            self.index.reconstruct(i, vec)
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)

# ===========================================
# Conversation Logging Utility
# ===========================================
class ConversationLogger:
    """
    Logs messages from Brain and Ouro to separate daily text files with timestamps.
    """

    def __init__(self, log_dir="conversation_logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def log_message(self, speaker, message):
        """
        Logs a message to a file named after the speaker and the current date.
        Example: conversation_logs/2025-02-25_Brain.log
        """
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        timestamp_str = datetime.datetime.now().strftime("%H:%M:%S")
        filename = os.path.join(self.log_dir, f"{date_str}_{speaker}.log")

        with open(filename, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp_str}] {message}\n")

# ===========================================
# Selenium Web Scraper Utility
# ===========================================
class SeleniumScraper:
    def __init__(self):
        print("🌐 Initializing Selenium Web Scraper...")
        chrome_options = Options()
        if SELENIUM_HEADLESS:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")

        if SELENIUM_DRIVER_PATH:
            self.driver = webdriver.Chrome(options=chrome_options, executable_path=SELENIUM_DRIVER_PATH)
        else:
            self.driver = webdriver.Chrome(options=chrome_options)

        print("✅ Selenium driver initialized.")

    def scrape_text(self, url):
        """
        Scrapes and returns the text content of a given URL.
        Returns an empty string if scraping fails.
        """
        try:
            print(f"📥 Scraping URL: {url}")
            self.driver.get(url)
            body = self.driver.find_element("tag name", "body")
            return body.text
        except Exception as e:
            print(f"❌ Failed to scrape {url}: {e}")
            return ""

    def close(self):
        """
        Closes the Selenium WebDriver.
        """
        self.driver.quit()
        print("🔒 Selenium driver closed.")
