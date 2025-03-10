"""
Core RAG system functionality for Ouro.
"""
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator, Union

from langchain.schema import Document
from tqdm import tqdm

from ouro.config import (
    SYSTEM_PROMPT,
    CHAT_SYSTEM_PROMPT,
    AGENT_SYSTEM_PROMPT,
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
    """Basic conversation memory manager with short-term recall."""
    
    def __init__(self, 
                 max_turns: int = MEMORY_TURNS, 
                 save_conversations: bool = SAVE_CONVERSATIONS,
                 embedding_manager = None):
        """Initialize the conversation memory.
        
        Args:
            max_turns: Maximum number of conversation turns to keep in short-term memory
            save_conversations: Whether to save conversations to disk
            embedding_manager: Optional embedding manager (not used in this version)
        """
        # Short-term memory (recent conversation turns)
        self.history = []
        self.max_turns = max_turns
        
        # Persistence settings
        self.save_conversations = save_conversations
        self.embedding_manager = None  # Disable for now to fix issues
        
        # Conversation metadata
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation_file = None
        
        # Set up conversation file
        if save_conversations:
            os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
            self.conversation_file = Path(CONVERSATIONS_DIR) / f"conversation_{self.conversation_id}.jsonl"
            
        # Thread safety for concurrent API requests
        import threading
        self._lock = threading.RLock()
    
    def add(self, user_message: str, assistant_message: str, add_to_long_term: bool = False) -> None:
        """Add a conversation turn to memory.
        
        Args:
            user_message: The user's message
            assistant_message: The assistant's response
            add_to_long_term: Whether to add this exchange to long-term memory (not used)
        """
        with self._lock:
            try:
                turn = {
                    "user": user_message, 
                    "assistant": assistant_message, 
                    "timestamp": datetime.now().isoformat(),
                    "conversation_id": self.conversation_id
                }
                
                # Add to in-memory history (short-term)
                self.history.append(turn)
                
                # Limit short-term memory size
                if len(self.history) > self.max_turns:
                    self.history = self.history[-self.max_turns:]
                
                # Save to disk if enabled (persistence)
                if self.save_conversations and self.conversation_file:
                    try:
                        with open(self.conversation_file, "a") as f:
                            f.write(json.dumps(turn) + "\n")
                    except Exception as e:
                        logger.error(f"Error saving conversation: {e}")
            except Exception as e:
                logger.error(f"Error adding to conversation memory: {e}")
    
    def get_history(self, format_for_context: bool = False) -> Union[List[Dict[str, Any]], List[Dict[str, str]]]:
        """Get the conversation history from short-term memory.
        
        Args:
            format_for_context: Whether to format for LLM context
            
        Returns:
            List of conversation turns
        """
        with self._lock:
            try:
                if not format_for_context:
                    return self.history.copy()
                else:
                    # Format for LLM context
                    formatted = []
                    for turn in self.history:
                        formatted.append({"user": turn["user"], "assistant": turn["assistant"]})
                    return formatted
            except Exception as e:
                logger.error(f"Error getting conversation history: {e}")
                return []
    
    def retrieve_relevant_memories(self, query: str, k: int = 2) -> List[Dict[str, Any]]:
        """Placeholder for retrieving memories - not implemented in this version."""
        return []
    
    def clear(self) -> None:
        """Clear short-term conversation memory."""
        with self._lock:
            try:
                self.history = []
                self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if self.save_conversations:
                    self.conversation_file = Path(CONVERSATIONS_DIR) / f"conversation_{self.conversation_id}.jsonl"
            except Exception as e:
                logger.error(f"Error clearing conversation memory: {e}")
    
    def export_all_conversations(self) -> Dict[str, List[Dict[str, Any]]]:
        """Export all saved conversations.
        
        Returns:
            Dictionary mapping conversation IDs to lists of turns
        """
        if not self.save_conversations:
            return {}
            
        result = {}
        try:
            conversation_files = list(Path(CONVERSATIONS_DIR).glob("conversation_*.jsonl"))
            
            for file_path in conversation_files:
                conv_id = file_path.stem.replace("conversation_", "")
                result[conv_id] = []
                
                try:
                    with open(file_path, "r") as f:
                        for line in f:
                            turn = json.loads(line)
                            result[conv_id].append(turn)
                except (json.JSONDecodeError, IOError) as e:
                    logger.error(f"Error reading conversation file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error exporting conversations: {e}")
            
        return result


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
        long_term_memory: bool = False  # Disabled by default
    ):
        """Initialize the RAG system.
        
        Args:
            model_config: Configuration for the LLM
            model_path: Path to a local model file
            embedding_model: Name or path to embedding model
            system_prompt: System prompt for the LLM
            top_k: Number of documents to retrieve
            memory_turns: Number of conversation turns to remember
            save_conversations: Whether to save conversations to disk
            long_term_memory: Whether to enable long-term memory (not used currently)
        """
        try:
            # Initialize embedding manager for document retrieval
            self.embedding_manager = EmbeddingManager(
                embedding_model=embedding_model or (model_config["embedding_model"] if model_config else None)
            )
            
            # Initialize conversation memory
            self.memory = ConversationMemory(
                max_turns=memory_turns,
                save_conversations=save_conversations
            )
            
            # Initialize LLM
            self.llm = LocalLLM(
                model_config=model_config,
                model_path=model_path
            )
            
            # Set configuration
            self.system_prompt = system_prompt
            self.top_k = top_k
            
            # Thread safety for concurrent operations
            import threading
            self._lock = threading.RLock()
            
            # Get logger
            self.logger = logger
            
        except Exception as e:
            logger.error(f"Error initializing OuroRAG: {e}")
            raise
    
    def get_contexts(self, query: str, include_long_term_memory: bool = False) -> List[str]:
        """Retrieve relevant document contexts for the query.
        
        Args:
            query: The user query
            include_long_term_memory: Whether to include relevant memories from long-term storage
            
        Returns:
            List of context strings
        """
        try:
            # Get document contexts - don't use filter until we have proper metadata
            similar_docs = self.embedding_manager.similarity_search(
                query, 
                k=self.top_k
            )
            contexts = [doc.page_content for doc in similar_docs]
            
            # Long-term memory retrieval is disabled for now
            # until we implement proper conversation storage with metadata
            
            return contexts
            
        except Exception as e:
            self.logger.error(f"Error retrieving contexts: {e}")
            return []
    
    def clean_response(self, response: str) -> str:
        """Aggressively clean the response of any conversation artifacts or hallucinations."""
        # Step 1: Remove any "Assistant:" prefix
        if response.lstrip().startswith("Assistant:"):
            response = response.lstrip()[len("Assistant:"):].lstrip()
        
        # Step 2: If there's a "User:" anywhere, extract only the content before it
        if "User:" in response:
            parts = response.split("User:", 1)
            response = parts[0].strip()
        
        # Step 3: Also handle lowercase variants
        if "user:" in response:
            parts = response.split("user:", 1)
            response = parts[0].strip()
        
        # Step 4: Remove any lines that look like dialogue
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            if not any(dialogue_marker in line.lower() for dialogue_marker in ["user:", "assistant:", "human:", "ai:", "question:", "answer:"]):
                cleaned_lines.append(line)
        
        response = '\n'.join(cleaned_lines)
        
        # Step 5: If the response still has problematic patterns (e.g., Q&A format), take drastic measures
        if re.search(r'^\s*(?:Q|Question)[\s\d]*:', response, re.MULTILINE) or re.search(r'^\s*(?:A|Answer)[\s\d]*:', response, re.MULTILINE):
            # Extract the first paragraph as a fallback
            paragraphs = [p for p in response.split('\n\n') if p.strip()]
            if paragraphs:
                response = paragraphs[0].strip()
                # If the first paragraph is short, include the second one too
                if len(paragraphs) > 1 and len(response) < 100:
                    response = f"{response}\n\n{paragraphs[1].strip()}"
        
        # Final check: if the response is empty after all cleaning, provide a default
        if not response.strip():
            response = "I understand your question. Let me provide a direct answer without examples."
        
        self.logger.debug(f"Cleaned response: {response[:100]}{'...' if len(response) > 100 else ''}")
        return response

    def generate(self, 
                query: str, 
                with_history: bool = True, 
                use_long_term_memory: bool = False, 
                stream: bool = True,
                important_exchange: bool = False,
                mode: str = "standard",
                context: List[str] = None) -> Generator[str, None, None]:
        """Generate a response to the query.
        
        Args:
            query: The user query
            with_history: Whether to include short-term conversation history
            use_long_term_memory: Whether to include long-term memory
            stream: Whether to stream the response
            important_exchange: Whether this is an important exchange to remember long-term
            mode: Generation mode ("standard", "chat", or "agent")
            context: Optional pre-generated context (will skip retrieval if provided)
            
        Returns:
            Generator yielding response tokens
        """
        # Use a lock to prevent concurrent generation requests which could cause issues
        with self._lock:
            try:
                # Log user query
                self.logger.info(f"User query: {query}", 
                               event_type="user_query", 
                               query=query, 
                               with_history=with_history,
                               mode=mode)
                
                # Retrieve relevant contexts if not provided
                if context is None:
                    self.logger.info(f"Searching knowledge base for: {query[:50]}...")
                    context = self.get_contexts(query)
                
                # Get conversation history if requested
                history = self.memory.get_history(format_for_context=True) if with_history else None
                
                # Select system prompt based on mode
                if mode == "chat":
                    active_prompt = CHAT_SYSTEM_PROMPT
                elif mode == "agent":
                    active_prompt = AGENT_SYSTEM_PROMPT
                else:
                    active_prompt = self.system_prompt
                
                # Generate response
                self.logger.info(f"Generating response (mode: {mode})...")
                response_stream = self.llm.generate(
                    system_prompt=active_prompt,
                    query=query,
                    context=context,
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
                
                # Log system response
                self.logger.info(f"System response to: {query[:30]}...", 
                               event_type="system_response", 
                               query=query,
                               response_length=len(cleaned_response))
                
                # Add to conversation memory, with flag for long-term if important
                self.memory.add(query, cleaned_response, add_to_long_term=important_exchange)
                
            except Exception as e:
                self.logger.error(f"Error generating response: {e}")
                yield f"Error: {str(e)}"
    
    def ingest_document(self, file_path: str, show_progress: bool = True) -> int:
        """Ingest a document into the vector store.
        
        Args:
            file_path: Path to the document to ingest
            show_progress: Whether to show a progress bar
            
        Returns:
            Number of chunks ingested
        """
        # Lock to prevent concurrent modifications to vector store
        with self._lock:
            # Load the document
            documents = load_document(file_path)
            if not documents:
                self.logger.warning(f"No documents loaded from {file_path}")
                return 0
            
            # Add to vector store
            self.embedding_manager.add_documents(documents, show_progress=show_progress)
            
            # Log document ingestion
            self.logger.log_document_ingestion(file_path, len(documents))
            
            return len(documents)
    
    def ingest_directory(self, directory_path: str, show_progress: bool = True) -> int:
        """Ingest all documents in a directory into the vector store.
        
        Args:
            directory_path: Path to the directory containing documents
            show_progress: Whether to show a progress bar
            
        Returns:
            Number of chunks ingested
        """
        # Lock to prevent concurrent modifications to vector store
        with self._lock:
            # Load documents from directory
            documents = load_documents_from_directory(directory_path)
            if not documents:
                self.logger.warning(f"No documents loaded from {directory_path}")
                return 0
            
            # Add to vector store
            self.embedding_manager.add_documents(documents, show_progress=show_progress)
            
            # Log directory ingestion
            self.logger.log_document_ingestion(directory_path, len(documents), 
                                              {"type": "directory"})
            
            return len(documents)
    
    def ingest_text(self, text: str, metadata: Dict[str, Any] = None, show_progress: bool = True) -> int:
        """Ingest text directly into the vector store.
        
        Args:
            text: The text to ingest
            metadata: Optional metadata for the document
            show_progress: Whether to show a progress bar
            
        Returns:
            Number of chunks ingested
        """
        # Lock to prevent concurrent modifications to vector store
        with self._lock:
            # Create a document from text
            documents = load_text(text, metadata)
            
            # Add to vector store
            self.embedding_manager.add_documents(documents, show_progress=show_progress)
            
            # Log text ingestion
            self.logger.log_document_ingestion("text_input", len(documents), 
                                             {"type": "direct_text", "metadata": metadata})
            
            return len(documents)
    
    def change_model(self, model_name_or_path: str) -> None:
        """Change the LLM model.
        
        Args:
            model_name_or_path: Name or path to the new model
        """
        with self._lock:
            self.llm.change_model(model_name_or_path)
            self.logger.info(f"Changed model to {model_name_or_path}")
    
    def clear_memory(self) -> None:
        """Clear short-term conversation memory."""
        with self._lock:
            self.memory.clear()
            self.logger.info("Conversation memory cleared")
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the current conversation history.
        
        Returns:
            List of conversation turns
        """
        return self.memory.get_history()
    
    def export_conversations(self) -> Dict[str, List[Dict[str, Any]]]:
        """Export all saved conversations.
        
        Returns:
            Dictionary mapping conversation IDs to lists of turns
        """
        return self.memory.export_all_conversations()
    
    def learn_from_conversations(self, fine_tune: bool = False) -> None:
        """Learn from past conversations.
        
        Args:
            fine_tune: Whether to fine-tune the model on conversations
        """
        # First, make sure all past conversations are in long-term memory
        if self.long_term_memory:
            with self._lock:
                conversations = self.memory.export_all_conversations()
                count = 0
                
                for conv_id, turns in conversations.items():
                    for turn in turns:
                        # Create a document for this turn
                        combined_text = f"USER: {turn['user']}\nASSISTANT: {turn['assistant']}"
                        metadata = {
                            "type": "conversation",
                            "timestamp": turn.get("timestamp", ""),
                            "conversation_id": conv_id
                        }
                        
                        # Add to vector store
                        from langchain.schema import Document
                        doc = Document(page_content=combined_text, metadata=metadata)
                        self.embedding_manager.add_documents([doc], show_progress=False)
                        count += 1
                
                self.logger.info(f"Added {count} conversation turns to long-term memory")
        
        # Optional fine-tuning using LoRA/QLoRA if requested
        if fine_tune:
            self.logger.info("Preparing for fine-tuning on conversation data...")
            # Not yet implemented - would require LoRA/QLoRA setup
            self.logger.warning("Fine-tuning not yet implemented")
            # TODO: Implement fine-tuning using Peft/LoRA/QLoRA when ready