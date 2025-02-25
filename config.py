"""
Configuration file for Ouro LLM Chatbot Project.
Centralizes all settings for the project.
Update these values if needed for consistency across modules.
"""

# --- General Settings ---
PROJECT_NAME = "Ouro LLM Chatbot"
VERSION = "2.0.0"  # Updated version for the new Ouro vs Brain

# --- FAISS (Vector Database) Configuration ---
FAISS_INDEX_PATH = "faiss_store/index.faiss"
EMBEDDING_DIM = 384  # Matches SentenceTransformer 'all-MiniLM-L6-v2'

# --- LLM (Large Language Model) Configuration ---
# We’re using the Mistral 7B model in GGUF format, locally hosted.
LLM_MODEL_PATH = "Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q3_K_M.gguf"
LLM_MAX_CONTEXT = 32768
LLM_BATCH_SIZE = 512
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.9

# --- Redis Cache Configuration ---
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
CACHE_TTL = 3600  # Time-to-live for cached data in seconds

# --- RabbitMQ (Message Queue) Configuration ---
RABBITMQ_HOST = "localhost"
RABBITMQ_PORT = 5672
RABBITMQ_QUEUE = "chat_queue"

# --- SQLite Database Configuration ---
DB_PATH = "faiss_index_store/chat_history.db"
MAX_CHAT_HISTORY = 10  # Number of recent chat entries to fetch if needed

# --- Logging Configuration ---
LOGGING_LEVEL = "INFO"

# --- API Rate Limits (if you build in multi-user or concurrency) ---
API_RATE_LIMIT = 1000  # Max number of calls allowed per hour in future expansions

# --- Selenium Web Scraper Configuration ---
SELENIUM_HEADLESS = True    # Run browser in headless mode
SELENIUM_DRIVER_PATH = None # Use system default driver if None

# End of configuration file.