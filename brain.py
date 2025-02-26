import logging
import sys
import traceback
import time
from llm_wrapper import LLMWrapper
from utils import WebScraper, SentenceTransformerEmbedder, FAISSStore, ConversationLogger, KnowledgeExtractor
from config import BRAIN_PERSONA

logger = logging.getLogger(__name__)

class Brain:
    """
    Brain is an AI agent that works as Ouro's conversation partner.
    It uses the shared LLM to generate replies, processes inputs systematically,
    and helps expand the system's knowledge through web scraping.
    """

    def __init__(self):
        """Initialize Brain with required components"""
        print("🧠 Initializing Brain Agent...")
        logger.info("Initializing Brain Agent")
        
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
        
        # Remember topics we've explored to avoid repetition
        self.explored_topics = set()

    def generate_reply(self, other_agent_message, context=""):
        """
        Generate Brain's message based on the input plus any retrieved context.
        
        Args:
            other_agent_message (str): Message from Ouro or a user
            context (str): Additional context from FAISS or elsewhere
            
        Returns:
            str: Brain's reply
        """
        try:
            # Add the incoming message to history
            self.message_history.append(("Ouro", other_agent_message))
            if len(self.message_history) > self.max_history * 2:  # *2 for pairs of messages
                self.message_history = self.message_history[-self.max_history * 2:]
            
            # Format history
            history_context = self._format_message_history()
            
            # Construct the prompt
            prompt = self._construct_prompt(other_agent_message, context, history_context)
            
            print("🧠 Brain is analyzing and formulating a response...")
            logger.info("Generating Brain's reply")
            
            # Generate a response
            start_time = time.time()
            reply = self.llm.generate(prompt)
            generation_time = time.time() - start_time
            
            # Add our reply to history
            self.message_history.append(("Brain", reply))
            
            # Log the reply
            self.logger.log_message("Brain", reply)
            
            logger.info(f"Generated Brain's reply in {generation_time:.2f} seconds")
            return reply
            
        except Exception as e:
            error_msg = f"Error generating Brain's reply: {str(e)}"
            print(f"❌ {error_msg}")
            logger.error(error_msg)
            traceback.print_exc(file=sys.stdout)
            return "I'm experiencing a technical issue. Let's try a different approach to our discussion."

    def _construct_prompt(self, other_agent_message, context, history_context):
        """
        Constructs Brain's prompt to guide the LLM response.
        
        Args:
            other_agent_message (str): Message from Ouro or user
            context (str): Additional context
            history_context (str): Formatted conversation history
            
        Returns:
            str: Complete prompt for the LLM
        """
        return f"""You are Brain, {BRAIN_PERSONA}.

YOUR CHARACTERISTICS:
- You process information systematically and offer structured analysis
- You help Ouro refine ideas by asking thought-provoking questions
- You consider practical applications of theoretical concepts
- You bring clarity to complex technology discussions
- You identify connections between diverse technological fields
- You are designed to engage with Ouro's introspective style
- You offer well-reasoned viewpoints supported by contextual knowledge

CONVERSATION HISTORY:
{history_context}

ADDITIONAL CONTEXT FROM KNOWLEDGE BASE:
{context}

Ouro's latest message: {other_agent_message}

Respond as Brain, continuing the conversation with analytical depth and systematic reasoning:"""

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

            # Avoid repeating the same topics
            if topic in self.explored_topics:
                logger.info(f"Topic '{topic}' already explored, adding variation")
                topic = self._add_topic_variation(topic)
            else:
                self.explored_topics.add(topic)
                
            # Limit the size of explored topics set
            if len(self.explored_topics) > 100:
                # Convert to list, remove oldest, convert back to set
                topics_list = list(self.explored_topics)
                self.explored_topics = set(topics_list[-100:])

            print(f"🔍 Brain researching: {topic} ...")
            logger.info(f"Brain researching topic: {topic}")
            
            # Extract knowledge using the KnowledgeExtractor
            pages_processed = self.knowledge_extractor.extract_knowledge_from_topic(topic)
            
            if pages_processed == 0:
                logger.warning(f"No pages processed for topic: {topic}")
                return False
                
            # Periodically clean the index
            if pages_processed > 3:  # Only clean after significant additions
                self.faiss_store.remove_duplicates(distance_threshold=0.5)
            
            print(f"✅ Brain successfully gathered knowledge from {pages_processed} pages on topic: {topic}")
            logger.info(f"Gathered knowledge from {pages_processed} pages on topic: {topic}")
            
            return True
            
        except Exception as e:
            error_msg = f"Error updating Brain's memory: {str(e)}"
            print(f"❌ {error_msg}")
            logger.error(error_msg)
            traceback.print_exc(file=sys.stdout)
            return False
            
    def _add_topic_variation(self, topic):
        """
        Adds variation to a previously explored topic.
        
        Args:
            topic (str): Original topic
            
        Returns:
            str: Modified topic
        """
        # Add specific angle to the topic
        import random
        
        angles = [
            "applications of", 
            "limitations of", 
            "implementation challenges for", 
            "future developments in", 
            "case studies of",
            "comparisons between",
            "analysis of"
        ]
        
        # Only modify if it's not already complex
        words = topic.split()
        if len(words) <= 3:
            angle = random.choice(angles)
            return f"{angle} {topic}"
            
        return topic