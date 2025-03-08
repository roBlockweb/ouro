"""
Core RAG (Retrieval-Augmented Generation) system with conversation memory
"""
import os
import json
import time
from typing import List, Dict, Any, Optional, Union, Callable
from pathlib import Path
from datetime import datetime

from src.document_loader import (
    load_document, 
    load_documents_from_directory,
    chunk_documents, 
    save_document
)
from src.embeddings import EmbeddingManager
from src.llm import LocalLLM
from src.logger import logger
from src.config import TOP_K_RESULTS, DATA_DIR, DOCUMENTS_DIR

class ConversationMemory:
    """Manages conversation history for short and long term memory"""
    
    def __init__(self, max_turns: int = 5, save_conversations: bool = True):
        """
        Initialize conversation memory manager
        
        Args:
            max_turns: Maximum number of conversation turns to keep in short-term memory
            save_conversations: Whether to save conversations to disk for long-term memory
        """
        self.short_term_memory = []
        self.max_turns = max_turns
        self.save_conversations = save_conversations
        self.conversation_dir = DATA_DIR / "conversations"
        self.conversation_dir.mkdir(exist_ok=True, parents=True)
        self.current_conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Initialized conversation memory with max_turns={max_turns}")
    
    def add_interaction(self, user_query: str, assistant_response: str) -> None:
        """
        Add a user-assistant interaction to memory
        
        Args:
            user_query: The user's query
            assistant_response: The assistant's response
        """
        interaction = {
            "user": user_query,
            "assistant": assistant_response,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to short-term memory
        self.short_term_memory.append(interaction)
        
        # Trim to max_turns
        if len(self.short_term_memory) > self.max_turns:
            self.short_term_memory = self.short_term_memory[-self.max_turns:]
        
        # Save to disk if enabled
        if self.save_conversations:
            self._save_to_disk(interaction)
    
    def _save_to_disk(self, interaction: Dict[str, str]) -> None:
        """
        Save an interaction to disk for long-term memory
        
        Args:
            interaction: The interaction to save
        """
        try:
            # Create a unique file path for this conversation
            file_path = self.conversation_dir / f"conversation_{self.current_conversation_id}.jsonl"
            
            # Append the interaction to the file
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(interaction) + "\n")
                
            logger.debug(f"Saved interaction to {file_path}")
        except Exception as e:
            logger.error(f"Error saving conversation: {str(e)}")
    
    def get_recent_history(self) -> List[Dict[str, str]]:
        """
        Get recent conversation history for context
        
        Returns:
            List of recent interactions
        """
        return self.short_term_memory
    
    def clear_memory(self) -> None:
        """Clear short-term memory"""
        self.short_term_memory = []
        # Start a new conversation ID for saving to disk
        self.current_conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("Cleared conversation memory")


class OuroRAG:
    """Main controller for the Ouro RAG system"""
    
    def __init__(self, 
                 embedding_manager: Optional[EmbeddingManager] = None,
                 llm: Optional[LocalLLM] = None,
                 embedding_model: str = None, 
                 config: Optional[Dict[str, Any]] = None,
                 memory_turns: int = 5,
                 save_conversations: bool = True):
        """
        Initialize the RAG system
        
        Args:
            embedding_manager: Pre-initialized embedding manager
            llm: Pre-initialized LLM
            embedding_model: Name of the embedding model (used if embedding_manager not provided)
            config: Configuration dictionary for the RAG system
            memory_turns: Number of conversation turns to keep in memory
            save_conversations: Whether to save conversations for adaptive learning
        """
        logger.info("Initializing OuroRAG system")
        
        # Initialize embedding manager if not provided
        self.embedding_manager = embedding_manager or EmbeddingManager(embedding_model)
        
        # Store LLM if provided
        self.llm = llm
        
        # Store configuration
        self.config = config or {}
        
        # Initialize conversation memory
        self.conversation_memory = ConversationMemory(
            max_turns=memory_turns,
            save_conversations=save_conversations
        )
    
    def load_llm(self, model_name: str, progress_callback: Callable = None) -> None:
        """
        Load an LLM model
        
        Args:
            model_name: Name of the LLM model to load
            progress_callback: Optional callback for progress updates
        """
        logger.info(f"Loading LLM: {model_name}")
        if self.llm is None:
            self.llm = LocalLLM()
            self.llm.load_model(model_name, progress_callback)
        else:
            self.llm.load_model(model_name, progress_callback)
    
    def ingest_document(self, 
                        file_path: Union[str, Path], 
                        progress_callback: Callable = None) -> None:
        """
        Ingest a document into the RAG system
        
        Args:
            file_path: Path to the document
            progress_callback: Optional callback for progress updates
        """
        logger.info(f"Ingesting document: {file_path}")
        
        try:
            # Update progress
            if progress_callback:
                progress_callback(f"Loading document: {file_path}", 0.1)
            
            # Load the document
            documents = load_document(file_path)
            
            # Update progress
            if progress_callback:
                progress_callback("Chunking document...", 0.3)
            
            # Chunk the document
            chunking_config = {
                "chunk_size": self.config.get("chunk_size", 1000),
                "chunk_overlap": self.config.get("chunk_overlap", 200),
            }
            chunked_documents = chunk_documents(documents, **chunking_config)
            
            # Update progress
            if progress_callback:
                progress_callback(f"Creating embeddings for {len(chunked_documents)} chunks...", 0.5)
            
            # Add to vector store - pass the progress callback for additional updates
            self.embedding_manager.progress_callback = progress_callback
            self.embedding_manager.add_documents(chunked_documents, show_progress=True)
            
            logger.info(f"Successfully ingested document: {file_path}")
            
            # Final progress update
            if progress_callback:
                progress_callback("Document successfully ingested", 1.0)
        
        except Exception as e:
            logger.error(f"Error ingesting document: {str(e)}")
            if progress_callback:
                progress_callback(f"Error: {str(e)}", -1.0)
            raise
    
    def ingest_directory(self, 
                         directory: Union[str, Path], 
                         progress_callback: Callable = None) -> None:
        """
        Ingest all documents in a directory
        
        Args:
            directory: Directory containing documents
            progress_callback: Optional callback for progress updates
        """
        logger.info(f"Ingesting documents from directory: {directory}")
        
        try:
            # Update progress
            if progress_callback:
                progress_callback(f"Scanning directory: {directory}", 0.1)
            
            # Load all documents
            documents = load_documents_from_directory(directory)
            
            if not documents:
                logger.warning(f"No documents found in directory: {directory}")
                if progress_callback:
                    progress_callback(f"No documents found in {directory}", -1.0)
                return
            
            # Update progress
            if progress_callback:
                progress_callback(f"Chunking {len(documents)} documents...", 0.3)
            
            # Chunk the documents
            chunking_config = {
                "chunk_size": self.config.get("chunk_size", 1000),
                "chunk_overlap": self.config.get("chunk_overlap", 200),
            }
            chunked_documents = chunk_documents(documents, **chunking_config)
            
            # Update progress
            if progress_callback:
                progress_callback(f"Creating embeddings for {len(chunked_documents)} chunks...", 0.5)
            
            # Add to vector store - pass the progress callback for additional updates
            self.embedding_manager.progress_callback = progress_callback
            self.embedding_manager.add_documents(chunked_documents, show_progress=True)
            
            logger.info(f"Successfully ingested {len(documents)} documents from directory")
            
            # Final progress update
            if progress_callback:
                progress_callback("Documents successfully ingested", 1.0)
        
        except Exception as e:
            logger.error(f"Error ingesting documents from directory: {str(e)}")
            if progress_callback:
                progress_callback(f"Error: {str(e)}", -1.0)
            raise
    
    def save_and_ingest_text(self, 
                            content: str, 
                            filename: str, 
                            progress_callback: Callable = None) -> None:
        """
        Save text content to a file and ingest it
        
        Args:
            content: Text content
            filename: Name of the file to save
            progress_callback: Optional callback for progress updates
        """
        logger.info(f"Saving and ingesting text as {filename}")
        
        try:
            # Update progress
            if progress_callback:
                progress_callback(f"Saving text as {filename}", 0.1)
            
            # Save the document
            file_path = save_document(content, filename)
            
            # Update progress
            if progress_callback:
                progress_callback(f"Starting ingestion of {filename}", 0.3)
            
            # Ingest the document with progress tracking
            self.ingest_document(
                file_path, 
                progress_callback=lambda msg, pct: progress_callback(
                    msg, 
                    0.3 + (pct * 0.7)  # Scale progress to 30%-100% range
                ) if pct >= 0 else progress_callback(msg, pct)
            )
            
            logger.info(f"Successfully saved and ingested text as {filename}")
        
        except Exception as e:
            logger.error(f"Error saving and ingesting text: {str(e)}")
            if progress_callback:
                progress_callback(f"Error: {str(e)}", -1.0)
            raise
    
    def query(self, 
             query_text: str, 
             k: int = None,
             search_progress_callback: Callable = None,
             generation_progress_callback: Callable = None,
             use_conversation_history: bool = True,
             add_to_memory: bool = True) -> str:
        """
        Process a query through the RAG pipeline with progress tracking and conversation memory
        
        Args:
            query_text: User query
            k: Number of results to retrieve (uses config value if None)
            search_progress_callback: Optional callback for search progress updates
            generation_progress_callback: Optional callback for generation progress updates
            use_conversation_history: Whether to include conversation history in the prompt
            add_to_memory: Whether to add this interaction to memory
            
        Returns:
            Generated response
        """
        logger.info(f"Processing query: '{query_text}'")
        
        if not self.llm:
            raise ValueError("LLM not loaded. Please load an LLM model first.")
        
        # Use config value for k if not specified
        if k is None:
            k = self.config.get("top_k", TOP_K_RESULTS)
        
        try:
            start_time = time.time()
            
            # Start search progress
            if search_progress_callback:
                search_progress_callback("Searching knowledge base...", 0.1)
            
            # Retrieve context documents
            retrieved_docs = self.embedding_manager.similarity_search(query_text, k=k)
            
            # Extract content from retrieved documents
            context_texts = [doc.page_content for doc in retrieved_docs]
            
            # If no relevant documents were found, provide an empty context with a note
            if not context_texts:
                context_texts = ["[No relevant information found in the knowledge base.]"]
            
            # Update search progress
            if search_progress_callback:
                search_progress_callback(f"Found {len(retrieved_docs)} relevant document chunks", 1.0)
            
            # Start generation progress
            if generation_progress_callback:
                generation_progress_callback("Generating response...", 0.1)
            
            # Store original callback to restore later
            original_callback = self.llm.progress_callback
            self.llm.progress_callback = generation_progress_callback
            
            # Get conversation history if enabled
            conversation_history = None
            if use_conversation_history:
                conversation_history = self.conversation_memory.get_recent_history()
            
            # Generate response with context and conversation history
            response = self.llm.generate_with_retrieval(
                query_text, 
                context_texts,
                conversation_history=conversation_history
            )
            
            # Restore original callback
            self.llm.progress_callback = original_callback
            
            # Add the interaction to memory if enabled
            if add_to_memory:
                self.conversation_memory.add_interaction(query_text, response)
            
            # Update generation progress
            if generation_progress_callback:
                generation_progress_callback("Response generated", 1.0)
            
            # Log performance metrics
            total_time = time.time() - start_time
            logger.info(f"Query processed in {total_time:.2f} seconds")
            
            return response
        
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            if search_progress_callback:
                search_progress_callback(f"Error: {str(e)}", -1.0)
            if generation_progress_callback:
                generation_progress_callback(f"Error: {str(e)}", -1.0)
            raise
            
    def clear_conversation(self) -> None:
        """
        Clear the current conversation history
        """
        self.conversation_memory.clear_memory()
        logger.info("Conversation history cleared")
    
    def learn_from_conversations(self, 
                               force_refresh: bool = False, 
                               progress_callback: Callable = None) -> None:
        """
        Learn from past conversations by adding them to the vector store
        
        Args:
            force_refresh: Whether to force reprocessing of all conversations
            progress_callback: Optional callback for progress updates
        """
        logger.info("Starting to learn from past conversations")
        
        try:
            # Create a marker file to track which conversations have been processed
            processed_marker = self.conversation_memory.conversation_dir / ".processed_conversations.json"
            processed_files = set()
            
            # Load already processed files if marker exists and we're not forcing a refresh
            if processed_marker.exists() and not force_refresh:
                try:
                    with open(processed_marker, "r", encoding="utf-8") as f:
                        processed_files = set(json.load(f))
                    logger.info(f"Loaded {len(processed_files)} previously processed conversation files")
                except Exception as e:
                    logger.warning(f"Error loading processed conversations marker: {str(e)}")
            
            # Get all conversation files
            all_files = list(self.conversation_memory.conversation_dir.glob("conversation_*.jsonl"))
            
            # Filter out already processed files if not forcing refresh
            if not force_refresh:
                files_to_process = [f for f in all_files if str(f) not in processed_files]
            else:
                files_to_process = all_files
            
            if not files_to_process:
                if progress_callback:
                    progress_callback("No new conversations to learn from", 1.0)
                logger.info("No new conversations to learn from")
                return
            
            logger.info(f"Found {len(files_to_process)} conversation files to process")
            
            # Update progress
            if progress_callback:
                progress_callback(f"Processing {len(files_to_process)} conversation files...", 0.1)
            
            # Process each conversation file
            all_processed = []
            for i, file_path in enumerate(files_to_process):
                try:
                    # Update progress
                    if progress_callback:
                        percent = 0.1 + (i / len(files_to_process)) * 0.8
                        progress_callback(f"Processing conversation {i+1}/{len(files_to_process)}", percent)
                    
                    # Extract insights from conversation
                    insights = self._extract_insights_from_conversation(file_path)
                    
                    if insights:
                        # Create a document from the insights
                        timestamp = file_path.stem.split("_")[1]  # Extract timestamp from filename
                        insight_path = self._save_insights_to_document(insights, timestamp)
                        
                        # Ingest the document
                        self.ingest_document(
                            insight_path, 
                            progress_callback=lambda msg, pct: None  # Suppress nested progress
                        )
                        
                        logger.info(f"Successfully learned from conversation {file_path.name}")
                    
                    # Mark as processed
                    all_processed.append(str(file_path))
                    
                except Exception as e:
                    logger.error(f"Error processing conversation {file_path}: {str(e)}")
            
            # Update the processed markers
            processed_files.update(all_processed)
            with open(processed_marker, "w", encoding="utf-8") as f:
                json.dump(list(processed_files), f)
            
            # Final progress update
            if progress_callback:
                progress_callback(f"Learned from {len(all_processed)} conversations", 1.0)
            
            logger.info(f"Successfully learned from {len(all_processed)} conversation files")
        
        except Exception as e:
            logger.error(f"Error learning from conversations: {str(e)}")
            if progress_callback:
                progress_callback(f"Error: {str(e)}", -1.0)
            raise
    
    def _extract_insights_from_conversation(self, conversation_file: Path) -> str:
        """
        Extract useful information from a conversation file
        
        Args:
            conversation_file: Path to the conversation file
            
        Returns:
            String containing extracted insights
        """
        try:
            # Load the conversation
            interactions = []
            with open(conversation_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        interaction = json.loads(line.strip())
                        interactions.append(interaction)
                    except:
                        continue
            
            if not interactions:
                return ""
            
            # Format the conversation into a document
            conversation_text = []
            for i, interaction in enumerate(interactions):
                user_query = interaction.get("user", "")
                assistant_response = interaction.get("assistant", "")
                timestamp = interaction.get("timestamp", "")
                
                if user_query and assistant_response:
                    conversation_text.append(f"User: {user_query}")
                    conversation_text.append(f"Assistant: {assistant_response}")
                    conversation_text.append("")  # Empty line between turns
            
            # Create the document with metadata
            insights = "\n".join(conversation_text)
            return insights
        
        except Exception as e:
            logger.error(f"Error extracting insights from {conversation_file}: {str(e)}")
            return ""
    
    def _save_insights_to_document(self, insights: str, timestamp: str) -> Path:
        """
        Save extracted insights to a document file
        
        Args:
            insights: The insights text
            timestamp: Timestamp for the filename
            
        Returns:
            Path to the saved document
        """
        # Create a directory for learned insights if it doesn't exist
        learned_dir = DOCUMENTS_DIR / "learned"
        learned_dir.mkdir(exist_ok=True, parents=True)
        
        # Create a filename
        filename = f"learned_conversation_{timestamp}.txt"
        file_path = learned_dir / filename
        
        # Write the insights to the file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# Learned Conversation ({timestamp})\n\n")
            f.write(insights)
        
        logger.info(f"Saved insights to {file_path}")
        return file_path