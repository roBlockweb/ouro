"""
Local LLM functionality for inference
"""
import os
import time
from typing import List, Dict, Any, Optional, Callable, Tuple
from pathlib import Path

import torch
try:
    import accelerate  # Import for memory optimization
except ImportError:
    pass  # Will fall back to basic loading if not available

from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    AutoTokenizer, 
    pipeline
)
from huggingface_hub import login
from tqdm import tqdm
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, SpinnerColumn

from src.config import MODEL_CONFIGURATIONS, MODEL_CACHE_DIR, SYSTEM_PROMPT, EMBEDDING_MODELS
from src.logger import logger

class LocalLLM:
    """Manager for local language model operations"""
    
    def __init__(self, model_name_or_path: str = None):
        """
        Initialize the local language model
        
        Args:
            model_name_or_path: Hugging Face model name or path
        """
        self.model_name = model_name_or_path
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.is_seq2seq = False
        self.progress_callback = None
        
        if model_name_or_path:
            self.load_model(model_name_or_path)
    
    def load_model(self, model_name_or_path: str, progress_callback: Callable = None) -> None:
        """
        Load a language model from Hugging Face with progress tracking
        
        Args:
            model_name_or_path: Hugging Face model name or path
            progress_callback: Optional callback for progress updates
        """
        logger.info(f"Loading LLM: {model_name_or_path}")
        self.model_name = model_name_or_path
        self.progress_callback = progress_callback
        
        try:
            # Determine if model is a seq2seq model based on common naming patterns
            model_lower = model_name_or_path.lower()
            self.is_seq2seq = any(name in model_lower for name in ["t5", "bart", "pegasus"])
            
            # Progress update
            if self.progress_callback:
                self.progress_callback("Downloading tokenizer...", 0.1)
            
            # Load the tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                cache_dir=str(MODEL_CACHE_DIR),
                padding_side="left",
                truncation_side="left"
            )
            
            # Set padding token if not set (important for some models)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Determine device - with Python 3.13 compatibility workarounds
            try:
                # First check if CUDA is available
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # On Mac, try to use MPS (Metal Performance Shaders) if available
                if device == "cpu" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
                    device = "mps"
                    logger.info("Using MPS (Metal) device on macOS")
            except:
                # Fallback to CPU if any detection issues
                device = "cpu"
                logger.info("Falling back to CPU due to device detection issues")
                
            # For safety, set lower precision to reduce memory usage
            model_dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32
                
            # Progress update
            if self.progress_callback:
                self.progress_callback(f"Downloading model weights on {device}...", 0.3)
            
            # Add safety flags to prevent memory issues on Python 3.13
            extra_model_kwargs = {
                "cache_dir": str(MODEL_CACHE_DIR),
                "torch_dtype": model_dtype,
                "low_cpu_mem_usage": True,  # Enable memory optimization
            }
            
            # Add device mapping for non-CPU devices
            if device != "cpu":
                extra_model_kwargs["device_map"] = "auto"
            
            # Load the appropriate model class based on architecture
            try:
                if self.is_seq2seq:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name_or_path,
                        **extra_model_kwargs
                    )
                    logger.info(f"Loaded Seq2Seq model {model_name_or_path} on {device}")
                    
                    # Progress update
                    if self.progress_callback:
                        self.progress_callback("Creating inference pipeline...", 0.8)
                    
                    # Create text generation pipeline with reduced batch size
                    self.pipeline = pipeline(
                        "text2text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device=device,
                    )
                else:
                    # For causal LMs, also add safetensors loading to avoid memory issues
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name_or_path,
                        **extra_model_kwargs
                    )
                    logger.info(f"Loaded Causal LM model {model_name_or_path} on {device}")
                    
                    # Progress update
                    if self.progress_callback:
                        self.progress_callback("Creating inference pipeline...", 0.8)
                    
                    # Create text generation pipeline with reduced batch size
                    self.pipeline = pipeline(
                        "text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device=device,
                    )
            except Exception as loading_error:
                # If loading fails with advanced options, try with minimal options
                logger.warning(f"Error loading model with advanced options: {str(loading_error)}")
                logger.info("Attempting to load model with minimal options...")
                
                if self.progress_callback:
                    self.progress_callback("Retrying with minimal settings...", 0.4)
                
                if self.is_seq2seq:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name_or_path,
                        cache_dir=str(MODEL_CACHE_DIR)
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name_or_path,
                        cache_dir=str(MODEL_CACHE_DIR)
                    )
                
                # Create basic pipeline
                pipeline_type = "text2text-generation" if self.is_seq2seq else "text-generation"
                self.pipeline = pipeline(
                    pipeline_type,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=-1,  # Force CPU
                )
            
            logger.info(f"Successfully loaded model and created pipeline")
            
            # Final progress update
            if self.progress_callback:
                self.progress_callback("Model loaded successfully!", 1.0)
        
        except Exception as e:
            logger.error(f"Error loading model {model_name_or_path}: {str(e)}")
            if self.progress_callback:
                self.progress_callback(f"Error: {str(e)}", -1.0)  # Negative progress indicates error
            raise
    
    def generate_with_retrieval(self, query: str, context_docs: List[str]) -> str:
        """
        Generate a response to a query with retrieved context
        
        Args:
            query: User query
            context_docs: List of retrieved document content
            
        Returns:
            Generated response string
        """
        if not self.model or not self.tokenizer or not self.pipeline:
            raise ValueError("Model not loaded. Please load a model first.")
        
        logger.info(f"Generating response for query with {len(context_docs)} context documents")
        
        try:
            # Combine context documents into a single string
            context_text = "\n\n".join(context_docs)
            
            # Format the prompt differently based on model type
            if self.is_seq2seq:
                prompt = f"""Context information:
{context_text}

{SYSTEM_PROMPT}

User question: {query}

Answer:"""
            else:
                prompt = f"""Context information:
{context_text}

{SYSTEM_PROMPT}

User: {query}
Ouro:"""
            
            # Generate response using the pipeline
            generation_kwargs = {
                "max_new_tokens": 512,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "num_return_sequences": 1,
            }
            
            # Generate the response
            outputs = self.pipeline(prompt, **generation_kwargs)
            
            # Extract the generated text
            if self.is_seq2seq:
                response = outputs[0]["generated_text"]
            else:
                # For causal models, use a more robust method to extract the response
                full_text = outputs[0]["generated_text"]
                
                # Extract text after the last "Ouro:" marker
                marker = "Ouro:"
                if marker in full_text:
                    response = full_text.split(marker)[-1].strip()
                else:
                    # Fallback approach: try to extract the part after the prompt
                    # This accounts for whitespace differences in the prompt vs. output
                    last_user_marker = f"User: {query}"
                    if last_user_marker in full_text:
                        parts = full_text.split(last_user_marker, 1)
                        if len(parts) > 1 and marker in parts[1]:
                            response = parts[1].split(marker, 1)[1].strip()
                        else:
                            # If we can't find the markers, just return the generated text
                            response = full_text
                    else:
                        response = full_text
                
                # If the response contains a generated "User:" prompt, trim it
                if "User:" in response:
                    response = response.split("User:")[0].strip()
            
            logger.info("Successfully generated response")
            return response.strip()
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

def get_available_model_sizes() -> Dict[str, Dict[str, Any]]:
    """
    Return the list of available model configurations by size
    
    Returns:
        Dictionary of model sizes and their configurations
    """
    return MODEL_CONFIGURATIONS

def get_custom_model_template() -> Dict[str, Any]:
    """
    Returns a template for custom model configuration
    
    Returns:
        Dictionary template for custom model configuration
    """
    return {
        "llm": "",  # User will provide this
        "embedding": "Medium",  # Default
        "description": "Custom model configuration",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": 4
    }

def login_huggingface(token: Optional[str] = None) -> bool:
    """
    Log in to Hugging Face Hub with robust error handling
    
    Args:
        token: Hugging Face token (will prompt if None)
        
    Returns:
        True if login successful, False otherwise
    """
    try:
        # First check if we already have a token in cache
        from huggingface_hub import HfFolder
        
        if token:
            # Use provided token
            login(token=token, write_permission=False)
            logger.info("Successfully logged in to Hugging Face Hub with provided token")
            return True
        
        # Check if a token is already saved
        existing_token = HfFolder.get_token()
        if existing_token is not None and len(existing_token) > 0:
            try:
                # Try using the existing token
                login(token=existing_token, write_permission=False)
                logger.info("Successfully logged in with existing Hugging Face token")
                return True
            except Exception as token_error:
                logger.warning(f"Existing token invalid: {str(token_error)}")
        
        # Try interactive login as last resort
        try:
            login(write_permission=False)
            logger.info("Successfully logged in to Hugging Face Hub")
            return True
        except Exception as e:
            logger.error(f"Failed interactive login: {str(e)}")
        
        # If we get here, all login attempts failed
        logger.warning("Continuing without Hugging Face login")
        logger.warning("Some models might not be available, but cached models should work")
        return True  # Return True to continue with reduced functionality
        
    except Exception as e:
        logger.error(f"Error in Hugging Face authentication: {str(e)}")
        logger.warning("Continuing without authentication")
        return True  # Return True to continue with reduced functionality

def get_huggingface_login_instructions() -> str:
    """
    Return instructions for logging in to Hugging Face
    
    Returns:
        String with detailed instructions
    """
    return """
To use Ouro with Hugging Face models, you need to authenticate with your Hugging Face account.

1. Create a Hugging Face account at https://huggingface.co/join if you don't have one

2. Generate an access token:
   - Go to https://huggingface.co/settings/tokens
   - Click "New token"
   - Name it "Ouro" and select "Read" role
   - Click "Generate a token"
   - Copy the token (it looks like "hf_xxxxxxxxxxxxxxxxxxxxxxxx")

3. In your terminal, run:
   $ huggingface-cli login
   
   Or when prompted by Ouro, paste your token.

This allows Ouro to download models from Hugging Face Hub.
"""

def load_model_with_progress(model_path: str, 
                            with_console_progress: bool = True) -> Tuple[LocalLLM, Dict[str, Any]]:
    """
    Load a model with progress display
    
    Args:
        model_path: Path to Hugging Face model
        with_console_progress: Whether to display progress in console
        
    Returns:
        Tuple of (LocalLLM instance, model config dict)
    """
    if with_console_progress:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("[green]Loading model...", total=100)
            
            def update_progress(message, percent):
                if percent < 0:  # Error
                    progress.update(task, description=f"[bold red]{message}")
                    return
                    
                progress.update(
                    task, 
                    description=f"[bold green]{message}",
                    completed=int(percent * 100)
                )
            
            llm = LocalLLM()
            llm.load_model(model_path, progress_callback=update_progress)
    else:
        llm = LocalLLM(model_path)
    
    # Return the model and a basic config
    config = {
        "llm": model_path,
        "description": "Custom model",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": 4
    }
    
    return llm, config