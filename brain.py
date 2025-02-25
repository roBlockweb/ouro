import sys
import traceback
from llm_wrapper import LLMWrapper
from utils import SentenceTransformerEmbedder, FAISSStore, SeleniumScraper, ConversationLogger
from config import *

class Brain:
    def __init__(self):
        """
        Brain is an AI agent that uses a single shared LLM (llm_wrapper.py) to
        generate replies. It can also scrape websites to gather data and store
        new embeddings in FAISS. We now add logging and optional duplicate removal.
        """
        print("🧠 Initializing Brain Agent...")
        # Retrieve the shared LLM instance
        self.llm = LLMWrapper.get_shared_llm()
        # We'll add scraping and embedding capabilities
        self.scraper = SeleniumScraper()
        self.embedder = SentenceTransformerEmbedder()
        self.faiss_store = FAISSStore()

        # Add a conversation logger for Brain
        self.logger = ConversationLogger()

    def generate_reply(self, user_input, context):
        """
        Generates a reply using the shared LLM.
        Combines any retrieved context with the user input to form a comprehensive prompt.
        """
        try:
            # Step 1: Construct the prompt
            full_prompt = self._construct_prompt(context, user_input)
            print("🔍 Brain is generating a reply with prompt:")
            print(full_prompt)

            # Step 2: Generate reply text from the LLM
            reply = self.llm.generate(full_prompt)
            print("✅ Brain reply generated successfully.")

            # Log the reply
            self.logger.log_message("Brain", reply)

            return reply
        except Exception as e:
            print("❌ Brain had an error generating reply:")
            traceback.print_exc(file=sys.stdout)
            return "I'm sorry, but I encountered an error."

    def _construct_prompt(self, context, user_input):
        """
        Creates a prompt by merging the background context with the user's question.
        """
        if context:
            context_section = f"Here is some background context:\n{context}\n"
        else:
            context_section = ""
        prompt = (
            "You are the 'Brain' agent in a conversation. "
            "You can provide deep, curious, thorough responses. "
            "Now, carefully read the context and answer:\n\n"
            f"{context_section}"
            "The user input (or the other agent's message) is:\n"
            f"{user_input}\n\n"
            "Answer in a detailed, thoughtful way."
        )
        return prompt

    def visit_and_update_memory(self, topic):
        """
        Similar to Ouro's method. Brain can also visit a web page or
        perform a Google search based on 'topic', scrape text, and update
        the FAISS index with newly embedded content.
        """
        try:
            if not topic.strip():
                print("⚠️ Brain has no topic to visit.")
                return False

            # If topic does not look like a URL, treat it as a search query
            if not (topic.startswith("http://") or topic.startswith("https://")):
                search_url = f"https://www.google.com/search?q={topic.replace(' ', '+')}"
                print(f"🔍 Brain sees topic '{topic}' is not a valid URL. Constructed search URL: {search_url}")
                url = search_url
            else:
                url = topic

            print(f"🌍 Brain visiting: {url} ...")

            # Scrape text from the URL
            text_data = self.scraper.scrape_text(url)
            if not text_data:
                print("⚠️ Brain found no text data to scrape.")
                return False

            # Split text into chunks for embedding
            text_chunks = self._chunk_text(text_data)
            print(f"📝 Brain dividing text into {len(text_chunks)} chunks.")

            # Embed and update FAISS
            embeddings = self.embedder.embed_text(text_chunks)
            self.faiss_store.add_embeddings(embeddings)
            self.faiss_store.save_index()

            # Optional: remove duplicates from the FAISS index
            # (distance_threshold can be adjusted, or skip entirely if you prefer)
            self.faiss_store.remove_duplicates(distance_threshold=0.5)

            print(f"✅ Brain successfully updated FAISS memory from topic: {topic}")
            return True

        except Exception as e:
            print("❌ Brain failed to visit/update memory:")
            traceback.print_exc(file=sys.stdout)
            return False

    def _chunk_text(self, text, chunk_size=500):
        """
        Splits the text into chunks of ~500 words for better embeddings.
        """
        words = text.split()
        chunks = [' '.join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks
