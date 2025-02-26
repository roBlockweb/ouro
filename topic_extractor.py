import logging
import re
from llm_wrapper import LLMWrapper
from config import MIN_TOPIC_LENGTH, MAX_TOPIC_LENGTH

logger = logging.getLogger(__name__)

class TopicExtractor:
    """
    Uses the shared LLM to extract searchable topics from text.
    These topics are used by agents to scrape the web for relevant information.
    """
    
    def __init__(self):
        """Initialize the topic extractor with the shared LLM instance"""
        print("🔍 Initializing Topic Extractor...")
        logger.info("Initializing Topic Extractor")
        self.llm = LLMWrapper.get_shared_llm()
    
    def extract_topic(self, text, with_url=True):
        """
        Extract a searchable topic from the given text.
        
        Args:
            text (str): The text to extract a topic from
            with_url (bool): Whether to try to extract a URL if present
            
        Returns:
            str: A searchable topic or URL
        """
        if not text.strip():
            return ""
            
        logger.info(f"Extracting topic from text ({len(text)} chars)")
        
        # First check for URLs directly in the text
        if with_url:
            url = self._extract_url(text)
            if url:
                logger.info(f"Found URL directly in text: {url}")
                return url
        
        # Use the LLM to extract a topic
        prompt = self._create_extraction_prompt(text)
        topic = self.llm.generate(prompt)
        
        # Clean up the response
        topic = self._clean_topic(topic)
        
        print(f"🔍 Extracted topic: {topic}")
        logger.info(f"Extracted topic: {topic}")
        return topic
    
    def _extract_url(self, text):
        """
        Extract a URL from the text if present.
        
        Args:
            text (str): Text to search for URLs
            
        Returns:
            str: Found URL or empty string
        """
        # Pattern to match HTTP/HTTPS URLs
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        
        # Find all URLs in the text
        urls = re.findall(url_pattern, text)
        
        if urls:
            # Return the first URL found
            return urls[0]
        else:
            return ""
    
    def _create_extraction_prompt(self, text):
        """
        Create a prompt for the LLM to extract topics.
        
        Args:
            text (str): Input text
            
        Returns:
            str: LLM prompt
        """
        return f"""As a topic extraction system, your task is to identify the most important and searchable topic from the following text. This topic will be used to search the web for more information.

GUIDELINES:
1. Extract the most specific, concrete topic that would yield useful search results
2. Focus on technology concepts, emerging trends, or specific technical areas
3. Avoid overly broad topics like "artificial intelligence" or "programming"
4. Format your response as a single topic phrase - do not include any explanations, labels, or quotation marks

TEXT:
{text}

EXTRACTED TOPIC:"""
    
    def _clean_topic(self, topic):
        """
        Clean the extracted topic.
        
        Args:
            topic (str): Raw topic from LLM
            
        Returns:
            str: Cleaned topic
        """
        # Remove any unnecessary text or formatting
        topic = topic.strip()
        
        # Remove quotation marks if present
        if topic.startswith('"') and topic.endswith('"'):
            topic = topic[1:-1]
        
        # Check length constraints
        words = topic.split()
        if len(words) < MIN_TOPIC_LENGTH and len(words) > 0:
            # Topic too short, maybe add a generic qualifier
            if len(words) == 1:
                topic = f"latest {topic} technology"
        
        if len(words) > MAX_TOPIC_LENGTH:
            # Topic too long, truncate
            topic = " ".join(words[:MAX_TOPIC_LENGTH])
        
        return topic