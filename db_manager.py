import sqlite3
from config import DB_PATH, MAX_CHAT_HISTORY

class DatabaseManager:
    """
    Manages a simple SQLite database to store user and bot messages.
    Helps keep a textual log of conversations if desired.
    """

    def __init__(self, db_path=DB_PATH):
        print(f"🔗 Connecting to SQLite database at {db_path}...")
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self):
        """
        Creates the chat_history table if it doesn't exist.
        Stores user_message, bot_response, and a timestamp.
        """
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.connection.commit()
        print("📚 Chat history table initialized.")

    def save_chat(self, user_message, bot_response):
        """
        Saves a new chat entry into the chat_history table.
        Useful if you want to log Ouro–Brain messages in text form.
        """
        self.cursor.execute('''
            INSERT INTO chat_history (user_message, bot_response)
            VALUES (?, ?)
        ''', (user_message, bot_response))
        self.connection.commit()
        print(f"📝 Saved chat: '{user_message}' -> '{bot_response}'")

    def fetch_recent_chats(self, limit=MAX_CHAT_HISTORY):
        """
        Fetches the most recent chat entries from the database.
        """
        self.cursor.execute('''
            SELECT user_message, bot_response, timestamp 
            FROM chat_history
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        chats = self.cursor.fetchall()
        print(f"📥 Fetched last {limit} chat entries from the database.")
        return chats

    def close_connection(self):
        """
        Closes the SQLite database connection when done.
        """
        self.connection.close()
        print("🔒 SQLite connection closed.")