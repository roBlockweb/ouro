from llm_wrapper import LLMWrapper

class TopicExtractor:
    """
    Uses a shared LLM to extract a concise search query or topic from a given text.
    """

    def __init__(self):
        print("🔍 Initializing Topic Extractor...")
        # Instead of creating a new LLMWrapper, we grab the shared instance
        self.llm = LLMWrapper.get_shared_llm()

    def extract_topic(self, text):
        """
        Extracts a concise, specific search query from the provided text.
        Uses the LLM to generate a short phrase without extra words.
        """
        if not text.strip():
            return ""

        prompt = (
            "Extract a concise, specific search query from the following text. "
            "Output only a short phrase without any extra words.\n\n"
            f"Text: {text}\n\n"
            "Search Query:"
        )
        topic = self.llm.generate(prompt)
        return topic.strip()
