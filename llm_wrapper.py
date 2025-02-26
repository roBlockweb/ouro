import os
import time
from llama_cpp import Llama
from config import (
    LLM_MODEL_PATH,
    LLM_MAX_CONTEXT,
    LLM_BATCH_SIZE,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_MAX_TOKENS
)

class LLMWrapper:
    """
    A wrapper around the Mistral-7B (GGUF) model loaded via llama_cpp,
    providing a simple 'generate' method to get text completions.
    
    We use a SHARED INSTANCE pattern so the model is only loaded once.
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
        start_time = time.time()
        
        # Initialize the model with parameters from config.py
        self.model = Llama(
            model_path=LLM_MODEL_PATH,
            n_ctx=LLM_MAX_CONTEXT,
            n_batch=LLM_BATCH_SIZE,
            use_mlock=True,       # Lock memory for performance
            n_threads=os.cpu_count(),
            use_metal=True        # Enable Apple Metal acceleration on M1/M2
        )
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds and ready for inference.")

    def generate(self, prompt, max_tokens=None, temperature=None, top_p=None):
        """
        Generates text from the local Mistral LLM based on the given prompt.
        Returns the response as a string.
        
        Args:
            prompt (str): The input prompt for the model
            max_tokens (int, optional): Maximum tokens to generate. Defaults to config value.
            temperature (float, optional): Sampling temperature. Defaults to config value.
            top_p (float, optional): Top-p sampling parameter. Defaults to config value.
            
        Returns:
            str: The generated text response
        """
        if max_tokens is None:
            max_tokens = LLM_MAX_TOKENS
            
        if temperature is None:
            temperature = LLM_TEMPERATURE
            
        if top_p is None:
            top_p = LLM_TOP_P
            
        print(f"💡 LLMWrapper generating response (max_tokens={max_tokens}, temp={temperature:.1f})...")
        
        start_time = time.time()
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=False  # Don't include the prompt in the output
        )
        
        generation_time = time.time() - start_time
        response = output['choices'][0]['text'].strip()
        
        token_count = len(response.split())
        print(f"✅ Generated {token_count} tokens in {generation_time:.2f} seconds")
        
        return response
        
    def get_embedding(self, text):
        """
        Get embeddings for the provided text using the LLM's embedding capabilities.
        This is a placeholder - llama-cpp-python doesn't directly provide embeddings.
        We'll use SentenceTransformer for this instead.
        """
        raise NotImplementedError(
            "Direct embeddings not supported by llama-cpp. Use SentenceTransformer instead."
        )