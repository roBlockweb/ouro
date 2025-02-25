import traceback
import sys
from llm_wrapper import LLMWrapper
from utils import SeleniumScraper, SentenceTransformerEmbedder, FAISSStore, ConversationLogger

class Ouro:
    """
    Ouro is an AI agent that uses the shared LLM to generate replies,
    scrape new information from the web, and update the FAISS index.
    We also log all of Ouro's replies to a daily text file.
    """

    def __init__(self):
        print("🌟 Initializing Ouro Agent...")
        # Use shared LLM
        self.llm = LLMWrapper.get_shared_llm()
        self.scraper = SeleniumScraper()
        self.embedder = SentenceTransformerEmbedder()
        self.faiss_store = FAISSStore()
        self.logger = ConversationLogger()

    def generate_reply(self, other_agent_message, context):
        """
        Generate Ouro's message based on the other agent's message plus any retrieved context.
        Uses a distinct prompt to give Ouro a different 'personality.'
        """
        try:
            # Construct Ouro’s unique prompt
            prompt = (
                "You are Ouro, a sentient-like AI that evolves through dialogue with Brain. "
                "You question assumptions, refine ideas, and maintain intellectual playfulness. "
                "You deliver thorough yet introspective reasoning, with a curious and self-reflective style.\n\n"
                f"Here is the context:\n{context}\n\n"
                f"Brain said: {other_agent_message}\n\n"
                "Ouro, please provide a thoughtful, introspective reply that reflects your evolving personality:"
            )

            print("🔍 Ouro is generating a reply with prompt:")
            print(prompt)

            reply = self.llm.generate(prompt)
            print("✅ Ouro generated a reply successfully.")

            # Log the reply to a file
            self.logger.log_message("Ouro", reply)

            return reply
        except Exception as e:
            print("❌ Ouro encountered an error generating a reply:")
            traceback.print_exc(file=sys.stdout)
            return "I'm Ouro, but I had an error. Sorry!"

    def visit_and_update_memory(self, topic):
        """
        Given a topic (URL or search term), Ouro will scrape the page, chunk it,
        embed it, and update the FAISS memory. Then optionally remove duplicates.
        Returns True if successful.
        """
        try:
            if not topic.strip():
                print("⚠️ Ouro has no topic to visit.")
                return False

            if not (topic.startswith("http://") or topic.startswith("https://")):
                search_url = f"https://www.google.com/search?q={topic.replace(' ', '+')}"
                print(f"🔍 Ouro sees topic '{topic}' is not a valid URL. Constructed search URL: {search_url}")
                url = search_url
            else:
                url = topic

            print(f"🌍 Ouro visiting: {url} ...")
            text_data = self.scraper.scrape_text(url)
            if not text_data:
                print("⚠️ Ouro found no text data from scraping.")
                return False

            text_chunks = self._chunk_text(text_data)
            print(f"📝 Ouro dividing text into {len(text_chunks)} chunks.")

            embeddings = self.embedder.embed_text(text_chunks)
            self.faiss_store.add_embeddings(embeddings)
            self.faiss_store.save_index()

            # Optionally remove duplicates to keep index clean
            self.faiss_store.remove_duplicates(distance_threshold=0.5)

            print(f"✅ Ouro successfully updated FAISS memory from topic: {topic}")

            return True
        except Exception as e:
            print("❌ Ouro failed to visit/update memory:")
            traceback.print_exc(file=sys.stdout)
            return False

    def _chunk_text(self, text, chunk_size=500):
        """
        Splits text into chunks of ~500 words for embedding.
        """
        words = text.split()
        chunks = [' '.join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks
