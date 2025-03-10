"""
LLM loading and integration for the Ouro RAG system.
"""
import os
import platform
import time
from typing import Dict, List, Optional, Any, Generator

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    TextIteratorStreamer
)
from threading import Thread

from ouro.config import MODELS_DIR, MAX_NEW_TOKENS, TEMPERATURE, DEFAULT_MODEL, MODELS
from ouro.logger import get_logger

logger = get_logger()


def check_hf_login() -> bool:
    """Check if user is logged in to Hugging Face."""
    # First try the standard token path
    token_path = os.path.expanduser("~/.huggingface/token")
    if os.path.exists(token_path):
        logger.info("Found Hugging Face token file.")
        return True
    
    # If that fails, try running whoami
    try:
        import subprocess
        result = subprocess.run(["huggingface-cli", "whoami"], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            logger.info(f"Logged in as: {result.stdout.strip()}")
            return True
    except Exception as e:
        logger.warning(f"Error checking Hugging Face login with CLI: {e}")
    
    # For development/testing, allow bypassing
    if os.environ.get("OURO_BYPASS_HF_LOGIN") == "1":
        logger.warning("Bypassing Hugging Face login check")
        return True
    
    logger.warning("No Hugging Face login detected. Assuming user has access to models.")
    return True  # Always return True to assume the user has access


class LocalLLM:
    """Load and use local LLMs for text generation."""
    
    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
    ):
        """Initialize the LLM."""
        # Set device
        self.device = self._get_device()
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.model_config = model_config or MODELS[DEFAULT_MODEL]
        self.model_path = model_path or self.model_config["llm_model"]
        
        # Always check for login but don't fail if not found
        check_hf_login()
            
        self.tokenizer = None
        self.model = None
        self.is_seq2seq = False
        
        self.load_model()
    
    def _get_device(self) -> str:
        """Determine the best available device for inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and platform.system() == "Darwin":
            return "mps"
        else:
            return "cpu"
    
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_path}")
            start_time = time.time()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                local_files_only=False
            )
            
            # Check if model is seq2seq
            is_t5 = "t5" in self.model_path.lower() or "flan" in self.model_path.lower()
            model_class = AutoModelForSeq2SeqLM if is_t5 else AutoModelForCausalLM
            self.is_seq2seq = is_t5
            
            # Prepare model loading args
            load_args = {"local_files_only": False}
            
            # Configure quantization if enabled
            if self.model_config.get("quantize", False) and self.device == "cuda":
                logger.info("Using 4-bit quantization")
                load_args["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                load_args["device_map"] = "auto"
            else:
                # Set torch dtype based on device
                if self.device == "cuda":
                    load_args["torch_dtype"] = torch.float16
                elif self.device == "mps":
                    load_args["torch_dtype"] = torch.float32 
                else:
                    load_args["torch_dtype"] = torch.float32
            
            # Load model
            self.model = model_class.from_pretrained(
                self.model_path,
                **load_args
            )
            
            # Move model to device (not needed for quantized models which use device_map)
            if "device_map" not in load_args:
                self.model.to(self.device)
            
            # Set up generation config
            self.model.generation_config = GenerationConfig.from_pretrained(
                self.model_path,
                max_new_tokens=self.model_config.get("max_new_tokens", MAX_NEW_TOKENS),
                temperature=self.model_config.get("temperature", TEMPERATURE),
                do_sample=True,
            )
            
            logger.info(f"Model loaded in {time.time() - start_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _format_prompt(self, system_prompt: str, context: List[str], query: str, history: List[Dict[str, str]] = None) -> str:
        """Format the prompt for generation."""
        context_str = "\n\n".join(context) if context else ""
        
        # Format conversation history if available
        history_str = ""
        if history:
            for turn in history:
                history_str += f"Previous User Message: {turn['user']}\nYour Previous Response: {turn['assistant']}\n\n"
        
        # Build the full prompt with very explicit instructions
        full_prompt = f"""
{system_prompt}

IMPORTANT INSTRUCTION: Always respond directly to the user. Do NOT create fictional dialogues, sample questions, or example conversations.

Knowledge Base:
{context_str}

Previous Conversation Context:
{history_str}

Current User Message: {query}

Your Response:"""
        
        return full_prompt.strip()
    
    def generate(
        self, 
        system_prompt: str,
        query: str,
        context: List[str] = None,
        history: List[Dict[str, str]] = None,
        stream: bool = True,
        **kwargs
    ) -> Generator[str, None, None]:
        """Generate a response to the query."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model or tokenizer not loaded.")
        
        # Format the prompt
        prompt = self._format_prompt(system_prompt, context or [], query, history)
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Set up the streamer if streaming is requested
        if stream:
            streamer = TextIteratorStreamer(
                self.tokenizer, 
                timeout=10.0, 
                skip_prompt=True, 
                skip_special_tokens=True
            )
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
            )
            
            # Start generation in a separate thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Yield from the streamer
            for text in streamer:
                yield text
        else:
            # Generate without streaming
            with torch.no_grad():
                outputs = self.model.generate(**inputs)
            
            # Decode the output
            if self.is_seq2seq:
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            yield response
    
    def change_model(self, model_name_or_path: str) -> None:
        """Change the current model."""
        # If model_name is a key in MODELS, use that config
        if model_name_or_path in MODELS:
            logger.info(f"Changing to predefined model: {model_name_or_path}")
            self.model_config = MODELS[model_name_or_path]
            self.model_path = self.model_config["llm_model"]
        else:
            # Assume model_name_or_path is a direct model path
            logger.info(f"Changing to custom model: {model_name_or_path}")
            self.model_path = model_name_or_path
        
        # Unload current model and load new one
        if self.model:
            self.model = None
            self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.load_model()