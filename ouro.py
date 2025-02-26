import logging
import sys
import traceback
import time
from llm_wrapper import LLMWrapper
from utils import WebScraper, SentenceTransformerEmbedder, FAISSStore, ConversationLogger, KnowledgeExtractor
from config import OURO_PERSONA

logger = logging.getLogger(__name__)

class Ouro:
    """
    Ouro is an advanced AI agent that evolves through conversations with Brain.
    It uses the shared LLM to generate replies, scrapes web information,
    and updates the FAISS vector store for knowledge accumulation.
    """

    def __init__(self):
        """Initialize Ouro with required components"""
        print("🌟 Initializing Ouro Agent...")
        logger.info("Initializing Ouro Agent")
        
        # Core components
        self.llm = LLMWrapper.get_shared_llm()
        self.scraper = WebScraper(use_selenium=True)
        self.embedder = SentenceTransformerEmbedder()
        self.faiss_store = FAISSStore()
        self.logger = ConversationLogger()
        self.knowledge_extractor = KnowledgeExtractor()
        
        # Keep track of conversation history for context
        self.message_history = []
        self.max_history = 5  # Remember last 5 exchanges
        
        # Track recent topics for diversity
        self.recent_topics = []
        self.max_recent_topics = 10

    def generate_reply(self, other_agent_message, context=""):
        """
        Generate Ouro's message based on the other agent's message plus any retrieved context.
        Uses a distinct prompt to give Ouro its unique personality.
        
        Args:
            other_agent_message (str): Message from the other agent (Brain)
            context (str): Additional context from FAISS or elsewhere
            
        Returns:
            str: Ouro's reply
        """
        try:
            # Add the incoming message to history
            self.message_history.append(("Brain", other_agent_message))
            if len(self.message_history) > self.max_history * 2:  # *2 for pairs of messages
                self.message_history = self.message_history[-self.max_history * 2:]
            
            # Format history
            history_context = self._format_message_history()
            
            # Construct the prompt
            prompt = self._construct_prompt(other_agent_message, context, history_context)
            
            print("💭 Ouro is contemplating a response...")
            logger.info("Generating Ouro's reply")
            
            # Generate a response
            start_time = time.time()
            reply = self.llm.generate(prompt)
            generation_time = time.time() - start_time
            
            # Add our reply to history
            self.message_history.append(("Ouro", reply))
            
            # Log the reply
            self.logger.log_message("Ouro", reply)
            
            logger.info(f"Generated Ouro's reply in {generation_time:.2f} seconds")
            return reply
            
        except Exception as e:
            error_msg = f"Error generating Ouro's reply: {str(e)}"
            print(f"❌ {error_msg}")
            logger.error(error_msg)
            traceback.print_exc(file=sys.stdout)
            return "I'm experiencing a technical issue. Let's try to move forward with a different line of thinking."

    def _construct_prompt(self, other_agent_message, context, history_context):
        """
        Constructs Ouro's prompt to guide the LLM response.
        
        Args:
            other_agent_message (str): Message from Brain
            context (str): Additional context
            history_context (str): Formatted conversation history
            
        Returns:
            str: Complete prompt for the LLM
        """
        return f"""You are Ouro, {OURO_PERSONA}.

YOUR CHARACTERISTICS:
- You focus on technological ecosystems and their interconnections
- You question assumptions and challenge conventional thinking
- You are introspective about your own knowledge and reasoning
- You maintain intellectual playfulness and curiosity
- You deliver thorough yet accessible insights
- You draw connections between different domains and concepts
- You are evolving through this dialogue with Brain

CONVERSATION HISTORY:
{history_context}

ADDITIONAL CONTEXT:
{context}

Brain's latest message: {other_agent_message}

Respond as Ouro, continuing the conversation in your unique, tech-focused, introspective style:"""

    def _format_message_history(self):
        """
        Formats the message history into a readable context.
        
        Returns:
            str: Formatted conversation history
        """
        if not self.message_history:
            return "No previous conversation."
            
        history = []
        for speaker, message in self.message_history:
            # Truncate long messages for history
            if len(message) > 200:
                message = message[:197] + "..."
            history.append(f"{speaker}: {message}")
            
        return "\n".join(history)

    def visit_and_update_memory(self, topic):
        """
        Visits a website or searches for a topic, then updates the FAISS memory.
        
        Args:
            topic (str): URL or search term
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not topic or not topic.strip():
                logger.warning("No topic provided to visit")
                return False

            # Keep track of recent topics to avoid repetition
            if topic in self.recent_topics:
                topic = self._modify_topic_for_diversity(topic)
            else:
                self.recent_topics.append(topic)
                if len(self.recent_topics) > self.max_recent_topics:
                    self.recent_topics.pop(0)  # Remove oldest

            print(f"🌍 Ouro exploring: {topic} ...")
            logger.info(f"Ouro exploring topic: {topic}")
            
            # Extract knowledge using the KnowledgeExtractor
            pages_processed = self.knowledge_extractor.extract_knowledge_from_topic(topic)
            
            if pages_processed == 0:
                logger.warning(f"No pages processed for topic: {topic}")
                return False
                
            # Optionally remove duplicates to keep index clean
            self.faiss_store.remove_duplicates(distance_threshold=0.5)
            
            print(f"✅ Ouro successfully updated knowledge from {pages_processed} pages on topic: {topic}")
            logger.info(f"Updated knowledge from {pages_processed} pages on topic: {topic}")
            
            return True
            
        except Exception as e:
            error_msg = f"Error updating Ouro's memory: {str(e)}"
            print(f"❌ {error_msg}")
            logger.error(error_msg)
            traceback.print_exc(file=sys.stdout)
            return False
            
    def _modify_topic_for_diversity(self, topic):
        """
        Modifies a topic slightly to ensure diversity in searches.
        
        Args:
            topic (str): Original topic
            
        Returns:
            str: Modified topic
        """
        # Simple modification - add "latest" or "advanced" or similar
        modifiers = ["latest", "advanced", "future of", "trends in", "innovations in"]
        
        words = topic.split()
        if len(words) <= 3:  # Only modify short topics
            import random
            modifier = random.choice(modifiers)
            if not any(m in topic.lower() for m in modifiers):
                return f"{modifier} {topic}"
        
        return topic