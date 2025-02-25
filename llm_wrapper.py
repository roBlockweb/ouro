import os
from llama_cpp import Llama
from config import (
    LLM_MODEL_PATH,
    LLM_MAX_CONTEXT,
    LLM_BATCH_SIZE,
    LLM_TEMPERATURE,
    LLM_TOP_P
)

class LLMWrapper:
    """
    A wrapper around the Mistral-7B (GGUF) model loaded via llama_cpp,
    providing a simple 'generate' method to get text completions.
    
    We now support a SHARED INSTANCE pattern so the model is only loaded once.
    Use LLMWrapper.get_shared_llm() to get the single loaded instance.
    """

    _shared_instance = None  # class-level reference to a single LLMWrapper object

    @classmethod
    def get_shared_llm(cls):
        """
        Class method to get or create the shared LLMWrapper instance.
        """
        if cls._shared_instance is None:
            cls._shared_instance = cls(_internal_init=True)
        return cls._shared_instance

    def __init__(self, _internal_init=False):
        """
        Normally, don't call LLMWrapper() directly. Instead use .get_shared_llm().
        We keep __init__ private to ensure we don't load the model multiple times.
        """
        if not _internal_init:
            raise RuntimeError(
                "Please use LLMWrapper.get_shared_llm() instead of instantiating directly."
            )

        print(f"🚀 Loading Mistral-7B LLM Model from: {LLM_MODEL_PATH}")
        # Initialize the model with parameters from config.py
        self.model = Llama(
            model_path=LLM_MODEL_PATH,
            n_ctx=LLM_MAX_CONTEXT,
            n_batch=LLM_BATCH_SIZE,
            use_mlock=True,       # Lock memory for performance
            n_threads=os.cpu_count(),
            use_metal=True        # Enable Apple Metal acceleration on M1
        )
        print("✅ Model loaded successfully and ready for inference.")

    def generate(self, prompt):
        """
        Generates text from the local Mistral LLM based on the given prompt.
        Returns the response as a string.
        """
        print("💡 LLMWrapper generating response from prompt (shared instance).")
        output = self.model(
            prompt,
            max_tokens=500,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P
        )
        response = output['choices'][0]['text'].strip()
        return response
