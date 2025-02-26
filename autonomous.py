import logging
import sys
import time
import traceback
import random
from config import MAX_CONVERSATION_EXCHANGES
from brain import Brain
from ouro import Ouro
from topic_extractor import TopicExtractor
from utils import (
    FAISSStore, 
    SentenceTransformerEmbedder, 
    ConversationLogger
)

# Configure logger
logger = logging.getLogger(__name__)

def autonomous_conversation_loop():
    """
    Runs an infinite loop where Ouro and Brain talk to each other, scrape the web
    for new information, and update the FAISS vector store until you press Ctrl+C.
    
    The conversation follows these steps:
    1. Ouro speaks to Brain
    2. Brain processes the message with context from FAISS
    3. Brain responds and searches for related topics
    4. Brain's reply is sent to Ouro with context from FAISS
    5. Ouro processes and responds, then searches for related topics
    6. The cycle continues indefinitely
    
    Both agents log their messages and continuously update the shared FAISS index.
    """

    print("🤖 Starting the autonomous Ouro-Brain conversation...")
    logger.info("Starting autonomous conversation loop")

    # Initialize components
    brain = Brain()
    ouro = Ouro()
    topic_extractor = TopicExtractor()
    faiss_store = FAISSStore()
    embedder = SentenceTransformerEmbedder()
    loop_logger = ConversationLogger()
    
    # Set up exchange counter
    exchanges = 0
    max_exchanges = MAX_CONVERSATION_EXCHANGES

    # We begin with a starter message from Ouro to Brain
    starter_topics = [
        "Let's discuss how AI systems like us could evolve alongside human society.",
        "I've been contemplating the intersection of technology and human creativity.",
        "What technological advancements do you think will shape the next decade?",
        "I'm interested in how emergent properties arise in complex systems like neural networks.",
        "How might decentralized systems transform our technological landscape?",
    ]
    
    ouro_message = random.choice(starter_topics)
    print(f"\nOURO (initial): {ouro_message}")
    loop_logger.log_message("System", f"Conversation started with: {ouro_message}")

    try:
        while True:
            try:
                # Update exchange counter
                exchanges += 1
                if max_exchanges > 0 and exchanges > max_exchanges:
                    print(f"\n🔄 Reached maximum exchanges ({max_exchanges}). Restarting conversation.")
                    loop_logger.log_message("System", f"Reached maximum exchanges ({max_exchanges}). Restarting conversation.")
                    ouro_message = random.choice(starter_topics)
                    exchanges = 1
                    continue
                
                # ================ BRAIN RESPONDS TO OURO ================
                # 1) Find relevant context for Brain from FAISS
                brain_query_embed = embedder.embed_text([ouro_message])
                distances, indices, texts = faiss_store.search(brain_query_embed, top_k=5)
                
                # Format context from retrieved texts
                if texts:
                    relevant_contexts = "\n---\n".join(texts[:3])  # Limit to first 3 for clarity
                    brain_context = f"Relevant information:\n{relevant_contexts}"
                else:
                    brain_context = "No specific context available from knowledge base."
                
                # 2) Brain generates a reply
                brain_message = brain.generate_reply(ouro_message, brain_context)
                print(f"\nBRAIN: {brain_message}")
                
                # 3) Extract a topic from Brain's reply for learning
                brain_topic = topic_extractor.extract_topic(brain_message)
                print(f"🔍 Brain's research topic: {brain_topic}")
                
                # 4) Brain attempts to learn more about the topic
                time.sleep(1)  # Brief pause to prevent network overload
                updated_by_brain = brain.visit_and_update_memory(brain_topic)
                if updated_by_brain:
                    print(f"🌐 Brain updated knowledge base with info on: {brain_topic}")
                
                # ================ OURO RESPONDS TO BRAIN ================
                # 5) Find relevant context for Ouro from FAISS
                ouro_query_embed = embedder.embed_text([brain_message])
                distances, indices, texts = faiss_store.search(ouro_query_embed, top_k=5)
                
                # Format context from retrieved texts
                if texts:
                    relevant_contexts = "\n---\n".join(texts[:3])  # Limit to first 3 for clarity
                    ouro_context = f"Relevant information:\n{relevant_contexts}"
                else:
                    ouro_context = "No specific context available from knowledge base."
                
                # 6) Ouro generates a reply
                ouro_message = ouro.generate_reply(brain_message, ouro_context)
                print(f"\nOURO: {ouro_message}")
                
                # 7) Extract a topic from Ouro's reply for learning
                ouro_topic = topic_extractor.extract_topic(ouro_message)
                print(f"🔍 Ouro's research topic: {ouro_topic}")
                
                # 8) Ouro attempts to learn more about the topic
                time.sleep(1)  # Brief pause to prevent network overload
                updated_by_ouro = ouro.visit_and_update_memory(ouro_topic)
                if updated_by_ouro:
                    print(f"🌐 Ouro updated knowledge base with info on: {ouro_topic}")
                
                # Optional: occasionally clean the FAISS index to remove duplicates
                if random.random() < 0.1:  # 10% chance each cycle
                    print("🧹 Performing routine maintenance on knowledge base...")
                    faiss_store.remove_duplicates(distance_threshold=0.5)
                
                print("\n" + "-" * 80 + "\n")  # Visual separator between exchanges
                
            except Exception as e:
                error_msg = f"Error in conversation cycle: {str(e)}"
                print(f"❌ {error_msg}")
                logger.error(error_msg)
                traceback.print_exc(file=sys.stdout)
                
                # Log the error and continue
                loop_logger.log_message("System", f"Error occurred: {str(e)}")
                
                # Reset the conversation with a new starter
                ouro_message = random.choice(starter_topics)
                print(f"\nResetting conversation. OURO: {ouro_message}")
                time.sleep(5)  # Pause briefly to let any rate limits reset
    
    except KeyboardInterrupt:
        print("\n👋 Conversation loop interrupted by user. Saving state and shutting down...")
        loop_logger.log_message("System", "Conversation loop interrupted by user")
        faiss_store.save_index()  # Ensure we save the latest index
        print("✅ Final state saved. Goodbye!")
        
if __name__ == "__main__":
    # This allows running the loop directly with `python autonomous.py`
    autonomous_conversation_loop()