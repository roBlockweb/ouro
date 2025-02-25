import redis
import json
from config import REDIS_HOST, REDIS_PORT, REDIS_DB, CACHE_TTL

class CacheManager:
    """
    CacheManager handles interactions with Redis to store and retrieve data quickly.
    This is helpful when you want to avoid repeated lookups or computations.
    """

    def __init__(self, host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB):
        """
        Connect to Redis using settings from config.py.
        """
        self.client = redis.Redis(host=host, port=port, db=db)
        print(f"🔗 Connected to Redis at {host}:{port}, DB: {db}")

    def set_cache(self, key, value, expiry=CACHE_TTL):
        """
        Stores a value in the Redis cache with the specified key and expiry time (in seconds).
        """
        serialized_value = json.dumps(value)
        self.client.setex(key, expiry, serialized_value)
        print(f"📝 Cached response for key '{key}' with expiry of {expiry} seconds.")

    def get_cache(self, key):
        """
        Retrieves a value from the Redis cache by its key.
        Returns the deserialized object or None if not found.
        """
        cached_value = self.client.get(key)
        if cached_value:
            print(f"📥 Retrieved cached response for key '{key}'.")
            return json.loads(cached_value)
        else:
            print(f"❌ No cache found for key '{key}'.")
            return None

    def clear_cache(self):
        """
        Clears all entries from the Redis cache (dangerous in production, but handy for dev).
        """
        self.client.flushdb()
        print("🧹 Cache cleared.")