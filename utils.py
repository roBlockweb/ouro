import os
import faiss
import numpy as np
import datetime
import time
import random
import requests
import logging
import traceback
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from sentence_transformers import SentenceTransformer
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from config import (
    EMBEDDING_DIM, 
    FAISS_INDEX_PATH, 
    SELENIUM_HEADLESS, 
    SELENIUM_DRIVER_PATH,
    LOG_DIR,
    SCRAPE_TIMEOUT,
    MAX_SCRAPE_DEPTH,
    MAX_SCRAPE_PAGES
)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, 'system.log'), 'a')
    ]
)
logger = logging.getLogger(__name__)

# ===========================================
# Sentence Embedding Utility
# ===========================================
class SentenceTransformerEmbedder:
    """
    Uses the SentenceTransformer model to create embeddings for text.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        logger.info(f"Loading SentenceTransformer Model ({model_name})...")
        self.model = SentenceTransformer(model_name)
        logger.info("SentenceTransformer Model loaded successfully")

    def embed_text(self, texts):
        """
        Converts a list of text strings into embeddings (numpy arrays).
        
        Args:
            texts (list or str): Text or list of texts to embed
            
        Returns:
            numpy.ndarray: Embeddings as a numpy array
        """
        if isinstance(texts, str):
            texts = [texts]  # Ensure we have a list for single string input
            
        # Track embedding time for performance monitoring
        start_time = time.time()
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        embedding_time = time.time() - start_time
        
        logger.info(f"Embedded {len(texts)} texts in {embedding_time:.2f} seconds")
        return embeddings

# ===========================================
# FAISS Index Utility
# ===========================================
class FAISSStore:
    """
    Manages a FAISS vector index for storing and retrieving embeddings.
    """
    def __init__(self, index_path=FAISS_INDEX_PATH, dimension=EMBEDDING_DIM):
        self.index_path = index_path
        self.dimension = dimension
        self.texts = []  # Store original texts alongside embeddings
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        if os.path.exists(self.index_path):
            logger.info(f"Loading existing FAISS index from {self.index_path}")
            try:
                self.index = faiss.read_index(self.index_path)
                logger.info(f"FAISS index loaded with {self.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
            
        # Optional: Load metadata if available
        self._try_load_texts()

    def _create_new_index(self):
        """Creates a new FAISS index"""
        logger.info(f"Creating a new FAISS index with dimension {self.dimension}")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.save_index()

    def save_index(self):
        """Saves the FAISS index to disk"""
        try:
            logger.info(f"Saving FAISS index to {self.index_path}")
            faiss.write_index(self.index, self.index_path)
            # Also save texts if we have them
            self._try_save_texts()
            logger.info("FAISS index saved successfully")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            
    def _try_save_texts(self):
        """Saves the original texts to a file alongside the index"""
        if not self.texts:
            return
            
        texts_path = self.index_path + ".texts"
        try:
            with open(texts_path, 'w', encoding='utf-8') as f:
                for text in self.texts:
                    f.write(text + "\n===TEXT_SEPARATOR===\n")
        except Exception as e:
            logger.error(f"Error saving texts: {e}")
            
    def _try_load_texts(self):
        """Loads the original texts from file if available"""
        texts_path = self.index_path + ".texts"
        if os.path.exists(texts_path):
            try:
                with open(texts_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.texts = content.split("\n===TEXT_SEPARATOR===\n")
                    # Remove the last empty entry if present
                    if self.texts and not self.texts[-1]:
                        self.texts.pop()
                logger.info(f"Loaded {len(self.texts)} texts from {texts_path}")
            except Exception as e:
                logger.error(f"Error loading texts: {e}")
                self.texts = []

    def add_embeddings(self, embeddings, texts=None):
        """
        Adds embeddings to the FAISS index.
        
        Args:
            embeddings (numpy.ndarray): Array with shape (n, dimension)
            texts (list, optional): Original texts corresponding to embeddings
        """
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)
            
        # Ensure we have the right shape
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
            
        logger.info(f"Adding {embeddings.shape[0]} embeddings to FAISS index")
        
        # Add to the index
        try:
            self.index.add(embeddings)
            
            # Store original texts if provided
            if texts:
                if isinstance(texts, str):
                    texts = [texts]
                self.texts.extend(texts)
                
            logger.info(f"FAISS index now contains {self.index.ntotal} vectors")
            return True
        except Exception as e:
            logger.error(f"Error adding embeddings: {e}")
            return False

    def search(self, query_embedding, top_k=5):
        """
        Searches the FAISS index for nearest embeddings.
        
        Args:
            query_embedding (numpy.ndarray): Query vector of shape (1, dimension)
            top_k (int): Number of results to return
            
        Returns:
            tuple: (distances, indices) of nearest neighbors
        """
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype=np.float32)
            
        # Ensure we have the right shape
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        logger.info(f"Searching FAISS index for top {top_k} matches")
        
        # Perform the search
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Get original texts if available
        texts = None
        if self.texts and len(self.texts) >= self.index.ntotal:
            texts = [self.texts[i] for i in indices[0] if i < len(self.texts)]
        
        return distances, indices, texts

    def remove_duplicates(self, distance_threshold=0.5):
        """
        Removes vectors that are duplicates based on L2 distance.
        Rebuilds the entire index, skipping vectors that are 'too close.'
        
        Args:
            distance_threshold (float): Vectors below this L2 distance are considered duplicates
        """
        logger.info(f"Removing duplicates with threshold {distance_threshold}")
        
        # Check if index is large enough to process
        index_size = self.index.ntotal
        if index_size < 2:
            logger.info("No duplicates to remove, index too small")
            return
        
        # Extract all vectors
        vectors = self._retrieve_all_embeddings()
        
        # Get texts if we have them
        original_texts = self.texts.copy() if self.texts else []
        
        # Build list of unique vectors
        unique_vectors = []
        unique_indices = []
        
        for i, vec in enumerate(vectors):
            keep_it = True
            for j, uvec in enumerate(unique_vectors):
                dist = np.sum((vec - uvec)**2)
                if dist < distance_threshold:
                    keep_it = False
                    break
            if keep_it:
                unique_vectors.append(vec)
                unique_indices.append(i)
        
        # Update texts if we have them
        if original_texts and len(original_texts) == index_size:
            self.texts = [original_texts[i] for i in unique_indices]
        
        # Rebuild the FAISS index
        new_index = faiss.IndexFlatL2(self.dimension)
        if unique_vectors:
            arr = np.array(unique_vectors, dtype=np.float32)
            new_index.add(arr)
        
        self.index = new_index
        self.save_index()
        
        # Report results
        old_count = index_size
        new_count = len(unique_vectors)
        removed = old_count - new_count
        logger.info(f"Removed {removed} duplicate vectors. Final count: {new_count}")

    def _retrieve_all_embeddings(self):
        """
        Extracts all embeddings from the FAISS index.
        
        Returns:
            numpy.ndarray: All vectors in the index
        """
        ntotal = self.index.ntotal
        if ntotal == 0:
            return np.zeros((0, self.dimension), dtype=np.float32)
        
        # We can reconstruct each vector by ID
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
    Logs messages from agents to separate daily text files with timestamps.
    """
    def __init__(self, log_dir=LOG_DIR):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def log_message(self, speaker, message):
        """
        Logs a message to a file named after the speaker and the current date.
        Example: conversation_logs/2025-02-25_Brain.log
        
        Args:
            speaker (str): Name of the speaker (e.g., "Ouro", "Brain")
            message (str): The message to log
        """
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        timestamp_str = datetime.datetime.now().strftime("%H:%M:%S")
        filename = os.path.join(self.log_dir, f"{date_str}_{speaker}.log")

        with open(filename, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp_str}] {message}\n\n")
            
        logger.info(f"Logged message from {speaker} to {filename}")
        
    def get_recent_logs(self, speaker, count=5):
        """
        Retrieves the most recent log entries for a given speaker.
        
        Args:
            speaker (str): Name of the speaker
            count (int): Number of recent messages to retrieve
            
        Returns:
            list: Recent messages in chronological order
        """
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        filename = os.path.join(self.log_dir, f"{date_str}_{speaker}.log")
        
        if not os.path.exists(filename):
            return []
            
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Split by double newlines to separate messages
        messages = content.split("\n\n")
        
        # Remove empty entries
        messages = [m for m in messages if m.strip()]
        
        # Return the most recent messages
        return messages[-count:] if messages else []

# ===========================================
# Advanced Web Scraper Utility
# ===========================================
class WebScraper:
    """
    Enhanced web scraper with support for both Selenium and Requests.
    Features link extraction, throttling, and content parsing.
    """
    def __init__(self, use_selenium=True, headless=SELENIUM_HEADLESS):
        self.use_selenium = use_selenium
        self.visited_urls = set()
        
        logger.info("Initializing WebScraper")
        
        if use_selenium:
            self._init_selenium(headless)
        
        # User agent rotation for avoiding blocks
        self.user_agents = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36",
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
        ]
        
    def _init_selenium(self, headless):
        """Initializes Selenium WebDriver"""
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-extensions")
        
        if SELENIUM_DRIVER_PATH:
            service = Service(executable_path=SELENIUM_DRIVER_PATH)
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
        else:
            self.driver = webdriver.Chrome(options=chrome_options)
            
        self.driver.set_page_load_timeout(SCRAPE_TIMEOUT)
        logger.info("Selenium WebDriver initialized")
        
    def _get_random_user_agent(self):
        """Returns a random user agent from the list"""
        return random.choice(self.user_agents)
        
    def scrape_text(self, url):
        """
        Scrapes text content from a URL.
        
        Args:
            url (str): URL to scrape
            
        Returns:
            str: Extracted text content
        """
        # Check if URL has been visited
        if url in self.visited_urls:
            logger.info(f"Already visited {url}, skipping")
            return ""
            
        # Mark as visited
        self.visited_urls.add(url)
        
        try:
            if self.use_selenium:
                return self._scrape_with_selenium(url)
            else:
                return self._scrape_with_requests(url)
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            traceback.print_exc()
            return ""
            
    def _scrape_with_selenium(self, url):
        """Uses Selenium for JavaScript-heavy sites"""
        logger.info(f"Scraping with Selenium: {url}")
        
        try:
            self.driver.get(url)
            
            # Wait for the page to load
            WebDriverWait(self.driver, SCRAPE_TIMEOUT).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Add a small delay to let any JS render
            time.sleep(2)
            
            # Extract text from the page
            body = self.driver.find_element(By.TAG_NAME, "body")
            text = body.text
            
            # Get title if available
            try:
                title = self.driver.title
                if title:
                    text = f"TITLE: {title}\n\n{text}"
            except:
                pass
                
            logger.info(f"Successfully scraped {len(text)} characters from {url}")
            return text
            
        except Exception as e:
            logger.error(f"Selenium scraping error for {url}: {str(e)}")
            return ""
            
    def _scrape_with_requests(self, url):
        """Uses Requests library for simpler sites"""
        logger.info(f"Scraping with Requests: {url}")
        
        headers = {
            "User-Agent": self._get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml",
            "Accept-Language": "en-US,en;q=0.9",
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=SCRAPE_TIMEOUT)
            response.raise_for_status()
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "header", "footer", "nav"]):
                script.extract()
                
            # Get text and normalize whitespace
            text = soup.get_text(separator='\n')
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = '\n'.join(lines)
            
            # Add title if available
            title = soup.title.string if soup.title else ""
            if title:
                text = f"TITLE: {title}\n\n{text}"
                
            logger.info(f"Successfully scraped {len(text)} characters from {url}")
            return text
            
        except Exception as e:
            logger.error(f"Requests scraping error for {url}: {str(e)}")
            return ""
            
    def extract_links(self, url, max_links=10):
        """
        Extracts links from a page.
        
        Args:
            url (str): URL to extract links from
            max_links (int): Maximum number of links to extract
            
        Returns:
            list: List of extracted URLs
        """
        logger.info(f"Extracting links from {url}")
        
        try:
            if self.use_selenium:
                self.driver.get(url)
                
                # Wait for the page to load
                WebDriverWait(self.driver, SCRAPE_TIMEOUT).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                # Find all links
                link_elements = self.driver.find_elements(By.TAG_NAME, "a")
                links = []
                
                for element in link_elements:
                    try:
                        href = element.get_attribute("href")
                        if href and href.startswith(("http://", "https://")):
                            links.append(href)
                    except:
                        continue
            else:
                # Use requests + BeautifulSoup
                headers = {
                    "User-Agent": self._get_random_user_agent(),
                }
                
                response = requests.get(url, headers=headers, timeout=SCRAPE_TIMEOUT)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                links = []
                
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    # Convert relative URLs to absolute
                    if not href.startswith(("http://", "https://")):
                        href = urljoin(url, href)
                    links.append(href)
                    
            # Filter and limit links
            filtered_links = []
            base_domain = urlparse(url).netloc
            
            for link in links:
                # Skip already visited links
                if link in self.visited_urls:
                    continue
                    
                # Skip links to different domains if desired
                link_domain = urlparse(link).netloc
                if base_domain and link_domain != base_domain:
                    continue
                    
                # Skip common file types
                if any(link.endswith(ext) for ext in [".pdf", ".jpg", ".png", ".gif", ".css", ".js"]):
                    continue
                    
                filtered_links.append(link)
                
                if len(filtered_links) >= max_links:
                    break
                    
            logger.info(f"Extracted {len(filtered_links)} links from {url}")
            return filtered_links
            
        except Exception as e:
            logger.error(f"Error extracting links from {url}: {str(e)}")
            return []
            
    def close(self):
        """Closes the Selenium WebDriver if used"""
        if self.use_selenium and hasattr(self, 'driver'):
            self.driver.quit()
            logger.info("Selenium WebDriver closed")
            
    def __del__(self):
        """Cleanup on destruction"""
        self.close()

# ===========================================
# Web Search and Knowledge Extraction
# ===========================================
class KnowledgeExtractor:
    """
    Extracts knowledge from websites by traversing links,
    generating embeddings, and saving to FAISS.
    """
    def __init__(self, use_selenium=True):
        self.scraper = WebScraper(use_selenium=use_selenium)
        self.embedder = SentenceTransformerEmbedder()
        self.faiss_store = FAISSStore()
        
    def extract_knowledge_from_topic(self, topic, max_depth=MAX_SCRAPE_DEPTH, max_pages=MAX_SCRAPE_PAGES):
        """
        Extracts knowledge from websites related to a topic.
        
        Args:
            topic (str): Topic or URL to start from
            max_depth (int): Maximum depth of link traversal
            max_pages (int): Maximum number of pages to scrape
            
        Returns:
            int: Number of pages successfully processed
        """
        logger.info(f"Extracting knowledge about: {topic}")
        
        # Convert topic to search URL if not already a URL
        if not (topic.startswith("http://") or topic.startswith("https://")):
            search_url = f"https://www.google.com/search?q={topic.replace(' ', '+')}"
            logger.info(f"Topic '{topic}' converted to search URL: {search_url}")
            start_url = search_url
        else:
            start_url = topic
            
        # Track URLs to visit (URL, current depth)
        urls_to_visit = [(start_url, 0)]
        processed_count = 0
        
        while urls_to_visit and processed_count < max_pages:
            # Get next URL and its depth
            url, depth = urls_to_visit.pop(0)
            
            # Skip if too deep
            if depth > max_depth:
                continue
                
            # Process the URL
            text = self.scraper.scrape_text(url)
            if not text:
                continue
                
            # Chunk the text
            chunks = self._chunk_text(text)
            logger.info(f"Chunked text into {len(chunks)} pieces")
            
            # Embed and store the chunks
            if chunks:
                embeddings = self.embedder.embed_text(chunks)
                self.faiss_store.add_embeddings(embeddings, chunks)
                processed_count += 1
                
            # Extract links if we're not at max depth
            if depth < max_depth:
                links = self.scraper.extract_links(url)
                
                # Add links to visit queue
                for link in links:
                    urls_to_visit.append((link, depth + 1))
                    
            # Save progress periodically
            if processed_count % 5 == 0:
                self.faiss_store.save_index()
                
        # Final save
        self.faiss_store.save_index()
        logger.info(f"Processed {processed_count} pages for topic: {topic}")
        
        return processed_count
        
    def _chunk_text(self, text, chunk_size=500, overlap=50):
        """
        Splits text into chunks with overlap.
        
        Args:
            text (str): Text to chunk
            chunk_size (int): Approximate words per chunk
            overlap (int): Words to overlap between chunks
            
        Returns:
            list: List of text chunks
        """
        words = text.split()
        
        if len(words) <= chunk_size:
            return [text]
            
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
        return chunks
        
    def close(self):
        """Closes resources"""
        self.scraper.close()