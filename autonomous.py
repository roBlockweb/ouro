import sys
import traceback
from config import *
from brain import Brain
from ouro import Ouro
from topic_extractor import TopicExtractor
from utils import FAISSStore, SentenceTransformerEmbedder, ConversationLogger

def autonomous_conversation_loop():
    """
    Runs an infinite loop where Ouro and Brain talk to each other, scrape the web
    for new information, and update the FAISS vector store until you press Ctrl+C.
    We also log conversation messages in separate daily files.
    """

    print("🤖 Starting the infinite Ouro vs. Brain conversation...")

    # Initialize components
    brain = Brain()      # Brain agent
    ouro = Ouro()        # Ouro agent
    topic_extractor = TopicExtractor()
    faiss_store = FAISSStore()
    embedder = SentenceTransformerEmbedder()

    # For logging the conversation loop itself (like a “system-level” logger, if desired)
    loop_logger = ConversationLogger()

    # We begin with a starter message from Ouro to Brain.
    ouro_message = "Hello Brain, I am Ouro. Let's begin our conversation."

    while True:
        try:
            # ================ BRAIN RESPONDS TO OURO ================
            # 1) Brain sees Ouro’s message
            brain_query_embed = embedder.embed_text([ouro_message])
            distances_b, indices_b = faiss_store.search(brain_query_embed)
            brain_context = f"Ouro said: {ouro_message}\nFAISS Distances: {distances_b}, Indices: {indices_b}"

            # 2) Brain generates a reply
            brain_message = brain.generate_reply(ouro_message, brain_context)
            print(f"\nBRAIN: {brain_message}")

            # 3) Extract a topic from Brain’s reply
            brain_topic = topic_extractor.extract_topic(brain_message)
            print(f"🔍 Brain's extracted topic: {brain_topic}")

            # 4) Brain attempts to scrape & update memory
            updated_by_brain = brain.visit_and_update_memory(brain_topic)
            if updated_by_brain:
                print(f"🌐 Brain updated knowledge with new info on: {brain_topic}")
            else:
                print(f"⚠️ Brain could not update knowledge for topic: {brain_topic}")

            # 5) Store Brain’s reply in FAISS
            brain_reply_embed = embedder.embed_text([brain_message])
            faiss_store.add_embeddings(brain_reply_embed)
            faiss_store.save_index()

            # Optional: remove duplicates at a global level if you want
            # faiss_store.remove_duplicates(distance_threshold=0.5)

            # ================ OURO RESPONDS TO BRAIN ================
            # 6) Ouro sees Brain’s new message
            ouro_query_embed = embedder.embed_text([brain_message])
            distances_o, indices_o = faiss_store.search(ouro_query_embed)
            ouro_context = f"Brain said: {brain_message}\nFAISS Distances: {distances_o}, Indices: {indices_o}"

            # 7) Ouro generates a reply
            ouro_message = ouro.generate_reply(brain_message, ouro_context)
            print(f"\nOURO: {ouro_message}")

            # 8) Extract a topic from Ouro’s reply
            ouro_topic = topic_extractor.extract_topic(ouro_message)
            print(f"🔍 Ouro's extracted topic: {ouro_topic}")

            # 9) Ouro scrapes & updates memory
            updated_by_ouro = ouro.visit_and_update_memory(ouro_topic)
            if updated_by_ouro:
                print(f"🌐 Ouro updated knowledge with new info on: {ouro_topic}")
            else:
                print(f"⚠️ Ouro could not update knowledge for topic: {ouro_topic}")

            # 10) Store Ouro’s reply in FAISS
            ouro_reply_embed = embedder.embed_text([ouro_message])
            faiss_store.add_embeddings(ouro_reply_embed)
            faiss_store.save_index()

            # Optional: remove duplicates again at this level
            # faiss_store.remove_duplicates(distance_threshold=0.5)

            # Log the “turn” at a high level (if you want to see the loop steps)
            loop_logger.log_message("Loop", f"BRAIN -> OURO iteration complete. BrainTopic: '{brain_topic}', OuroTopic: '{ouro_topic}'")

        except KeyboardInterrupt:
            print("\n🚪 Conversation interrupted by user. Exiting infinite loop...")
            break
        except Exception as e:
            print("❌ An unexpected error occurred in the conversation loop:")
            traceback.print_exc(file=sys.stdout)
            # We continue the loop rather than break, so it tries next iteration
            continue

if __name__ == "__main__":
    autonomous_conversation_loop()
