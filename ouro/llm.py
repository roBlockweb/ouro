"""
LLM module for loading and interacting with language models.
"""
import json
import os
import torch
import requests
from pathlib import Path
from typing import Dict, Any, Generator, List, Optional, Union

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    GenerationConfig
)

from ouro.config import (
    MODELS, 
    DEFAULT_MODEL, 
    MODELS_DIR,
    SYSTEM_PROMPT,
    USE_QUANTIZATION,
    OLLAMA_INTEGRATION,
    OLLAMA_HOST
)
from ouro.logger import get_logger

logger = get_logger()


def check_hf_login() -> bool:
    """Check if user is logged in to Hugging Face.
    
    Returns:
        bool: True if logged in, False otherwise
    """
    # We're removing login requirements, so always return True
    return True


class LocalLLM:
    """Class for loading and using local language models."""
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None, model_path: Optional[str] = None):
        """Initialize the LLM.
        
        Args:
            model_config: Configuration dict for the model
            model_path: Path to a local model file
        """
        # Set up model configuration
        if model_config:
            self.model_config = model_config
        else:
            self.model_config = MODELS[DEFAULT_MODEL]
        
        # Set model path
        self.model_path = model_path or self.model_config["llm_model"]
        
        # Check if using Ollama
        self.use_ollama = self.model_config.get("use_ollama", False) and OLLAMA_INTEGRATION
        
        if not self.use_ollama:
            # Standard Hugging Face model initialization
            # Determine device
            self.device = self._determine_device()
            
            # Initialize
            self.model = None
            self.tokenizer = None
            
            # Load the model
            self._initialize_model()
        else:
            # For Ollama, we don't need to initialize the model here
            logger.info(f"Using Ollama with model {self.model_path}")
            self._check_ollama_connection()
    
    def _determine_device(self) -> str:
        """Determine the best device to use.
        
        Returns:
            str: Device string for PyTorch
        """
        if torch.cuda.is_available():
            logger.info("CUDA is available, using GPU")
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and self.model_config.get("use_mps", False):
            logger.info("MPS is available, using Apple Silicon GPU")
            return "mps"
        else:
            logger.info("No GPU is available, using CPU")
            return "cpu"
    
    def _check_ollama_connection(self) -> bool:
        """Check connection to Ollama server.
        
        Returns:
            bool: True if connected, False otherwise
        """
        try:
            response = requests.get(f"{OLLAMA_HOST}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                logger.info(f"Connected to Ollama server: {len(models)} models available")
                return True
            else:
                logger.warning(f"Ollama server returned status code {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to Ollama server: {e}")
            return False
    
    def _initialize_model(self) -> None:
        """Initialize the model and tokenizer."""
        try:
            # First, create the model directory if it doesn't exist
            os.makedirs(MODELS_DIR, exist_ok=True)
            
            # Set up quantization config if needed
            quantization_config = None
            if self.model_config.get("quantize", False) and USE_QUANTIZATION and self.device == "cuda":
                logger.info("Using 4-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=False
                )
            
            # Log model loading
            logger.info(f"Loading model {self.model_path} on {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                cache_dir=MODELS_DIR,
                token=False  # Don't use token-based auth
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=self.device,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                cache_dir=MODELS_DIR,
                quantization_config=quantization_config,
                token=False  # Don't use token-based auth
            )
            
            # Configure generation parameters
            self.model.generation_config.max_new_tokens = self.model_config.get("max_new_tokens", 512)
            self.model.generation_config.do_sample = True
            self.model.generation_config.temperature = self.model_config.get("temperature", 0.1)
            self.model.generation_config.top_p = 0.95
            
            logger.info(f"Model loaded successfully: {self.model_path}")
            logger.info(f"Model parameters: {self.model.num_parameters():,}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate(self, 
                system_prompt: str = SYSTEM_PROMPT, 
                query: str = "", 
                context: Optional[List[str]] = None,
                history: Optional[List[Dict[str, str]]] = None,
                stream: bool = True) -> Generator[str, None, None]:
        """Generate a response.
        
        Args:
            system_prompt: System prompt to guide the model
            query: User query
            context: List of relevant document contexts
            history: Conversation history
            stream: Whether to stream the response token by token
            
        Returns:
            Generator yielding response tokens
        """
        try:
            # Combine context if provided
            context_str = ""
            if context and len(context) > 0:
                context_str = "\n".join([f"[Document {i+1}]: {ctx}" for i, ctx in enumerate(context)])
            
            # Format history if provided
            history_str = ""
            if history and len(history) > 0:
                history_str = ""
                for turn in history:
                    history_str += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"
            
            # Build prompt with template appropriate for the model
            prompt = self._format_prompt(system_prompt, query, context_str, history_str)
            
            # Use Ollama API if configured
            if self.use_ollama:
                return self._generate_with_ollama(prompt, stream)
            
            # Otherwise, use Hugging Face model
            # Tokenize input
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Set up generation parameters
            gen_config = GenerationConfig(
                max_new_tokens=self.model_config.get("max_new_tokens", 512),
                do_sample=True,
                temperature=self.model_config.get("temperature", 0.1),
                top_p=0.95,
                repetition_penalty=1.1
            )
            
            # Stream response if requested
            if stream:
                generated_text = ""
                for output in self.model.generate(
                    input_ids,
                    generation_config=gen_config,
                    streamer=None,  # We'll handle streaming manually
                    return_dict_in_generate=True,
                    output_scores=False,
                ):
                    # Get new token
                    new_tokens = output.sequences[:, input_ids.shape[1]:]
                    token_str = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
                    
                    # Skip if no new content
                    if token_str == generated_text:
                        continue
                    
                    # Get just the new content
                    new_content = token_str[len(generated_text):]
                    generated_text = token_str
                    
                    yield new_content
            else:
                # Generate the complete response at once
                with torch.no_grad():
                    output = self.model.generate(
                        input_ids,
                        generation_config=gen_config,
                        return_dict_in_generate=True,
                        output_scores=False,
                    )
                
                # Decode the generated output
                generated_ids = output.sequences[0, input_ids.shape[1]:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                yield generated_text
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            yield f"Error: {str(e)}"
    
    def _generate_with_ollama(self, prompt: str, stream: bool) -> Generator[str, None, None]:
        """Generate response using Ollama API.
        
        Args:
            prompt: Formatted prompt
            stream: Whether to stream the response
            
        Returns:
            Generator yielding response tokens
        """
        try:
            # Prepare request data
            data = {
                "model": self.model_path,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": self.model_config.get("temperature", 0.1),
                    "top_p": 0.95,
                    "num_predict": self.model_config.get("max_new_tokens", 512)
                }
            }
            
            headers = {"Content-Type": "application/json"}
            
            if stream:
                # Streaming response
                with requests.post(
                    f"{OLLAMA_HOST}/api/generate",
                    json=data,
                    headers=headers,
                    stream=True
                ) as response:
                    
                    if response.status_code != 200:
                        error_msg = f"Ollama API error: {response.status_code}"
                        logger.error(error_msg)
                        yield error_msg
                        return
                    
                    # Process streaming response
                    for line in response.iter_lines():
                        if not line:
                            continue
                        
                        try:
                            chunk = json.loads(line)
                            if "response" in chunk:
                                yield chunk["response"]
                            
                            # Check if generation is done
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSON from Ollama: {line}")
            else:
                # Non-streaming response
                response = requests.post(
                    f"{OLLAMA_HOST}/api/generate",
                    json=data,
                    headers=headers
                )
                
                if response.status_code != 200:
                    error_msg = f"Ollama API error: {response.status_code}"
                    logger.error(error_msg)
                    yield error_msg
                    return
                
                result = response.json()
                yield result.get("response", "No response from Ollama")
                
        except Exception as e:
            logger.error(f"Error with Ollama generation: {e}")
            yield f"Error: {str(e)}"
    
    def _format_prompt(self, system_prompt: str, query: str, context: str, history: str) -> str:
        """Format prompt according to model type.
        
        Args:
            system_prompt: System instructions
            query: User query
            context: Document context
            history: Conversation history
            
        Returns:
            str: Formatted prompt
        """
        # Check if it's an Ollama model
        if self.use_ollama:
            return self._format_ollama_prompt(system_prompt, query, context, history)
        
        # Check if it's a Llama model
        if "llama" in self.model_path.lower():
            return self._format_llama_prompt(system_prompt, query, context, history)
        
        # Check if it's a Mistral model
        if "mistral" in self.model_path.lower():
            return self._format_mistral_prompt(system_prompt, query, context, history)
        
        # Default formatting (works well with most models)
        result = f"{system_prompt}\n\n"
        
        if context:
            result += f"Context information:\n{context}\n\n"
        
        if history:
            result += f"Previous conversation:\n{history}\n"
        
        result += f"User: {query}\n\nAssistant:"
        
        return result
    
    def _format_ollama_prompt(self, system_prompt: str, query: str, context: str, history: str) -> str:
        """Format prompt for Ollama.
        
        Args:
            system_prompt: System instructions
            query: User query
            context: Document context
            history: Conversation history
            
        Returns:
            str: Formatted prompt for Ollama
        """
        # Format prompt based on what model we're using in Ollama
        model_type = self.model_path.lower()
        
        if "llama" in model_type:
            return self._format_llama_prompt(system_prompt, query, context, history)
        elif "mistral" in model_type:
            return self._format_mistral_prompt(system_prompt, query, context, history)
        
        # Generic format for other Ollama models
        result = f"System: {system_prompt}\n\n"
        
        if context:
            result += f"Context information:\n{context}\n\n"
        
        if history:
            result += f"Previous conversation:\n{history}\n"
        
        result += f"User: {query}\n\nAssistant:"
        
        return result
    
    def _format_llama_prompt(self, system_prompt: str, query: str, context: str, history: str) -> str:
        """Format prompt for Llama models.
        
        Args:
            system_prompt: System instructions
            query: User query
            context: Document context
            history: Conversation history
            
        Returns:
            str: Formatted prompt for Llama
        """
        B_SYS, E_SYS = "<s>[INST] <<SYS>>\n", "\n<</SYS>>\n\n"
        
        # Start with system prompt
        result = f"{B_SYS}{system_prompt}{E_SYS}"
        
        # Add context if available
        if context:
            result += f"I'll provide some context information to help you:\n\n{context}\n\n"
        
        # Add conversation history if available
        if history:
            result += f"Previous conversation:\n{history}\n"
        
        # Add the query
        result += f"{query}[/INST]"
        
        return result
    
    def _format_mistral_prompt(self, system_prompt: str, query: str, context: str, history: str) -> str:
        """Format prompt for Mistral models.
        
        Args:
            system_prompt: System instructions
            query: User query
            context: Document context
            history: Conversation history
            
        Returns:
            str: Formatted prompt for Mistral
        """
        # Mistral uses a similar format to Llama 2
        result = f"<s>[INST] {system_prompt}\n\n"
        
        if context:
            result += f"Context information:\n{context}\n\n"
        
        if history:
            result += f"Previous conversation:\n{history}\n"
        
        result += f"{query} [/INST]"
        
        return result
    
    def generate_text(self, prompt: str) -> str:
        """Simple method to generate text from a raw prompt.
        
        Args:
            prompt: Raw prompt text
            
        Returns:
            str: Generated text
        """
        # Use Ollama API if configured
        if self.use_ollama:
            try:
                response = requests.post(
                    f"{OLLAMA_HOST}/api/generate",
                    json={
                        "model": self.model_path,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.model_config.get("temperature", 0.1),
                            "top_p": 0.95,
                            "num_predict": self.model_config.get("max_new_tokens", 512)
                        }
                    },
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code != 200:
                    logger.error(f"Ollama API error: {response.status_code}")
                    return f"Error: Ollama API returned status code {response.status_code}"
                
                result = response.json()
                return result.get("response", "No response from Ollama")
            
            except Exception as e:
                logger.error(f"Error with Ollama generation: {e}")
                return f"Error: {str(e)}"
        
        # Otherwise, use Hugging Face model
        try:
            # Tokenize input
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate text
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=self.model_config.get("max_new_tokens", 512),
                    do_sample=True,
                    temperature=self.model_config.get("temperature", 0.1),
                    top_p=0.95
                )
            
            # Decode and return the generated text (excluding the prompt)
            generated_ids = output[0, input_ids.shape[1]:]
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error: {str(e)}"
    
    def change_model(self, model_name_or_path: str) -> None:
        """Change the model.
        
        Args:
            model_name_or_path: Name of a preset model or path to a model
        """
        try:
            # Determine if it's a preset model or a path
            if model_name_or_path in MODELS:
                self.model_config = MODELS[model_name_or_path]
                self.model_path = self.model_config["llm_model"]
                self.use_ollama = self.model_config.get("use_ollama", False) and OLLAMA_INTEGRATION
            else:
                self.model_path = model_name_or_path
                self.use_ollama = False  # Custom path always uses HF
            
            # If using Ollama, just update the model name
            if self.use_ollama:
                logger.info(f"Switching to Ollama model: {self.model_path}")
                self._check_ollama_connection()
                return
            
            # For HF models, need to reinitialize
            # Free up memory if we have a model loaded
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                del self.tokenizer
                torch.cuda.empty_cache() if hasattr(self, 'device') and self.device == "cuda" else None
            
            # Determine device (might have changed from Ollama)
            self.device = self._determine_device()
            
            # Reinitialize the model
            self._initialize_model()
            
            logger.info(f"Model changed to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error changing model: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Text to count tokens in
            
        Returns:
            int: Number of tokens
        """
        if self.use_ollama:
            try:
                response = requests.post(
                    f"{OLLAMA_HOST}/api/tokenize",
                    json={"model": self.model_path, "prompt": text},
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code != 200:
                    logger.error(f"Ollama API error: {response.status_code}")
                    return 0
                
                result = response.json()
                return len(result.get("tokens", []))
            
            except Exception as e:
                logger.error(f"Error getting token count from Ollama: {e}")
                return 0
        else:
            # Use tokenizer to count tokens
            return len(self.tokenizer.encode(text))