"""
Core RAG system functionality for Ouro.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator, Union

from langchain.schema import Document
from tqdm import tqdm

from ouro.config import (
    SYSTEM_PROMPT, 
    TOP_K_RESULTS, 
    MEMORY_TURNS, 
    SAVE_CONVERSATIONS,
    CONVERSATIONS_DIR
)
from ouro.document_loader import load_document, load_documents_from_directory, load_text
from ouro.embeddings import EmbeddingManager
from ouro.llm import LocalLLM
from ouro.logger import get_logger

logger = get_logger()


class ConversationMemory:
    """Manages conversation history and persistence."""
    
    def __init__(self, max_turns: int = MEMORY_TURNS, save_conversations: bool = SAVE_CONVERSATIONS):
        """Initialize the conversation memory."""
        self.history = []
        self.max_turns = max_turns
        self.save_conversations = save_conversations
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation_file = None
        
        if save_conversations:
            os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
            self.conversation_file = Path(CONVERSATIONS_DIR) / f"conversation_{self.conversation_id}.jsonl"
    
    def add(self, user_message: str, assistant_message: str) -> None:
        """Add a conversation turn to memory."""
        turn = {"user": user_message, "assistant": assistant_message, "timestamp": datetime.now().isoformat()}
        
        # Add to in-memory history
        self.history.append(turn)
        
        # Limit memory size
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]
        
        # Save to disk if enabled
        if self.save_conversations and self.conversation_file:
            with open(self.conversation_file, "a") as f:
                f.write(json.dumps(turn) + "\n")
    
    def get_history(self, format_for_context: bool = False) -> Union[List[Dict[str, str]], List[Dict[str, str]]]:
        """Get the conversation history."""
        if not format_for_context:
            return self.history
        else:
            # Format for LLM context
            formatted = []
            for turn in self.history:
                formatted.append({"user": turn["user"], "assistant": turn["assistant"]})
            return formatted
    
    def clear(self) -> None:
        """Clear the conversation history."""
        self.history = []
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.save_conversations:
            self.conversation_file = Path(CONVERSATIONS_DIR) / f"conversation_{self.conversation_id}.jsonl"


class OuroRAG:
    """Main RAG system that coordinates document retrieval and generation."""
    
    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        embedding_model: Optional[str] = None,
        system_prompt: str = SYSTEM_PROMPT,
        top_k: int = TOP_K_RESULTS,
        memory_turns: int = MEMORY_TURNS,
        save_conversations: bool = SAVE_CONVERSATIONS,
    ):
        """Initialize the RAG system."""
        # Initialize conversation memory
        self.memory = ConversationMemory(
            max_turns=memory_turns,
            save_conversations=save_conversations
        )
        
        # Initialize embedding manager for document retrieval
        self.embedding_manager = EmbeddingManager(
            embedding_model=embedding_model or (model_config["embedding_model"] if model_config else None)
        )
        
        # Initialize LLM
        self.llm = LocalLLM(
            model_config=model_config,
            model_path=model_path
        )
        
        # Set configuration
        self.system_prompt = system_prompt
        self.top_k = top_k
        
        # Add thread lock for concurrent operations
        import threading
        self._lock = threading.RLock()
    
    def get_contexts(self, query: str) -> List[str]:
        """Retrieve relevant document contexts for the query."""
        similar_docs = self.embedding_manager.similarity_search(query, k=self.top_k)
        contexts = [doc.page_content for doc in similar_docs]
        return contexts
    
    def clean_response(self, response: str) -> str:
        """Clean the response of any conversation formatting artifacts."""
        # Remove "Assistant:" prefix if it appears at the beginning
        if response.lstrip().startswith("Assistant:"):
            response = response.lstrip()[len("Assistant:"):].lstrip()
        
        # Remove any fictional conversation examples
        if "User:" in response:
            # This could be a fake conversation sample - try to extract just the initial answer
            parts = response.split("User:")
            if parts and parts[0].strip():
                response = parts[0].strip()
        
        return response

    def generate(self, query: str, with_history: bool = True, stream: bool = True) -> Generator[str, None, None]:
        """Generate a response to the query."""
        # Use a lock to prevent concurrent generation requests which could cause issues
        with self._lock:
            # Retrieve relevant contexts
            logger.info(f"Searching knowledge base for: {query[:50]}...")
            contexts = self.get_contexts(query)
            
            # Get conversation history if requested
            history = self.memory.get_history(format_for_context=True) if with_history else None
            
            # Generate response
            logger.info("Generating response...")
            response_stream = self.llm.generate(
                system_prompt=self.system_prompt,
                query=query,
                context=contexts,
                history=history,
                stream=stream
            )
            
            # Collect the full response for memory while streaming
            full_response = ""
            
            for token in response_stream:
                full_response += token
                yield token
            
            # Clean response before saving to memory
            cleaned_response = self.clean_response(full_response)
            
            # Add to conversation memory
            self.memory.add(query, cleaned_response)
    
    def ingest_document(self, file_path: str, show_progress: bool = True) -> int:
        """Ingest a document into the vector store."""
        # Load the document
        documents = load_document(file_path)
        if not documents:
            logger.warning(f"No documents loaded from {file_path}")
            return 0
        
        # Add to vector store
        self.embedding_manager.add_documents(documents, show_progress=show_progress)
        
        return len(documents)
    
    def ingest_directory(self, directory_path: str, show_progress: bool = True) -> int:
        """Ingest all documents in a directory into the vector store."""
        # Load documents from directory
        documents = load_documents_from_directory(directory_path)
        if not documents:
            logger.warning(f"No documents loaded from {directory_path}")
            return 0
        
        # Add to vector store
        self.embedding_manager.add_documents(documents, show_progress=show_progress)
        
        return len(documents)
    
    def ingest_text(self, text: str, metadata: Dict[str, Any] = None, show_progress: bool = True) -> int:
        """Ingest text directly into the vector store."""
        # Create a document from text
        documents = load_text(text, metadata)
        
        # Add to vector store
        self.embedding_manager.add_documents(documents, show_progress=show_progress)
        
        return len(documents)
    
    def change_model(self, model_name_or_path: str) -> None:
        """Change the LLM model."""
        self.llm.change_model(model_name_or_path)
    
    def clear_memory(self) -> None:
        """Clear conversation memory."""
        self.memory.clear()
    
    def learn_from_conversations(self, fine_tune: bool = False) -> None:
        """Learn from past conversations."""
        # This is a placeholder for the optional fine-tuning feature
        if fine_tune:
            logger.info("Fine-tuning on past conversations not yet implemented")
            # TODO: Implement fine-tuning using LoRA or adapters