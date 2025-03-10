"""
Simple API for Ouro RAG system.

This module provides a minimal API for accessing Ouro functionality
from other applications. It's designed to be used locally, without
requiring a web server.
"""
import os
import json
from typing import Dict, List, Any, Optional, Union

from ouro.config import (
    DEFAULT_MODEL,
    MODELS,
    TOP_K_RESULTS,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    API_KEYS,
    API_KEY_REQUIRED
)
from ouro.rag import OuroRAG
from ouro.logger import get_logger

logger = get_logger()


class OuroAPI:
    """API wrapper for Ouro RAG system."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        """Initialize the API with specified model.
        
        Args:
            model_name: Name of the model to use
        """
        self.rag = OuroRAG(model_config=MODELS[model_name])
        self.logger = logger
        
    def authenticate(self, api_key: str) -> bool:
        """Check if an API key is valid.
        
        Args:
            api_key: API key to check
            
        Returns:
            True if valid, False otherwise
        """
        if not API_KEY_REQUIRED:
            return True
            
        return api_key in API_KEYS
    
    def query(
        self, 
        query_text: str, 
        use_history: bool = True,
        use_long_term_memory: bool = True,
        important_exchange: bool = False,
        return_contexts: bool = False,
        agent_mode: bool = False,
        use_web_search: bool = False,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Query the RAG system.
        
        Args:
            query_text: Query text
            use_history: Whether to use conversation history
            use_long_term_memory: Whether to use long-term memory
            important_exchange: Whether to mark exchange as important
            return_contexts: Whether to return contexts used
            agent_mode: Whether to use agent mode
            use_web_search: Whether to use web search
            api_key: API key for authentication
            
        Returns:
            Query response
        """
        # Authenticate if required
        if API_KEY_REQUIRED and not self.authenticate(api_key or ""):
            return {"error": "Invalid API key"}
        
        try:
            # Log query
            self.logger.info(f"API query: {query_text[:50]}...", 
                            source="api",
                            with_history=use_history)
            
            # Get contexts from RAG system
            contexts = self.rag.get_contexts(query_text)
            
            # Generate response
            response = ""
            for token in self.rag.generate(
                query=query_text,
                with_history=use_history,
                use_long_term_memory=use_long_term_memory,
                important_exchange=important_exchange,
                stream=False,
                context=contexts
            ):
                response += token
                
            # Calculate token count
            query_tokens = self.rag.llm.count_tokens(query_text)
            response_tokens = self.rag.llm.count_tokens(response)
            total_tokens = query_tokens + response_tokens
            
            # Construct result
            result = {
                "query": query_text,
                "response": response,
                "with_history": use_history,
                "tokens": total_tokens
            }
            
            # Add contexts if requested
            if return_contexts:
                result["contexts"] = contexts
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error in API query: {e}")
            return {"error": str(e)}
    
    def ingest_text(
        self, 
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Ingest text into the system.
        
        Args:
            text: Text to ingest
            metadata: Optional metadata
            api_key: API key for authentication
            
        Returns:
            Ingestion result
        """
        # Authenticate if required
        if API_KEY_REQUIRED and not self.authenticate(api_key or ""):
            return {"error": "Invalid API key"}
        
        try:
            # Prepare metadata
            meta = {"source": "api"} if metadata is None else metadata
            
            # Log ingest request
            self.logger.info(f"API ingest: {len(text)} characters", 
                           api_endpoint="ingest", 
                           metadata=meta)
            
            # Ingest the text
            num_docs = self.rag.ingest_text(
                text=text,
                metadata=meta
            )
            
            return {
                "status": "success", 
                "ingested_documents": num_docs
            }
            
        except Exception as e:
            self.logger.error(f"Error in API ingest: {e}")
            return {"error": str(e)}
    
    def ingest_document(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Ingest a document into the system.
        
        Args:
            file_path: Path to file
            metadata: Optional metadata
            api_key: API key for authentication
            
        Returns:
            Ingestion result
        """
        # Authenticate if required
        if API_KEY_REQUIRED and not self.authenticate(api_key or ""):
            return {"error": "Invalid API key"}
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}
            
            # Prepare metadata
            meta = {"source": f"api_file_{os.path.basename(file_path)}"} if metadata is None else metadata
            
            # Log ingest request
            self.logger.info(f"API ingest document: {file_path}", 
                           api_endpoint="ingest_document", 
                           metadata=meta)
            
            # Ingest the document
            num_docs = self.rag.ingest_document(
                file_path=file_path,
                metadata=meta
            )
            
            return {
                "status": "success", 
                "ingested_documents": num_docs
            }
            
        except Exception as e:
            self.logger.error(f"Error in API ingest document: {e}")
            return {"error": str(e)}
    
    def clear_memory(self, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Clear conversation memory.
        
        Args:
            api_key: API key for authentication
            
        Returns:
            Status message
        """
        # Authenticate if required
        if API_KEY_REQUIRED and not self.authenticate(api_key or ""):
            return {"error": "Invalid API key"}
        
        try:
            self.rag.clear_memory()
            return {"status": "success", "message": "Memory cleared"}
            
        except Exception as e:
            self.logger.error(f"Error clearing memory: {e}")
            return {"error": str(e)}
    
    def get_memory(
        self, 
        limit: int = 10,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get conversation memory.
        
        Args:
            limit: Maximum number of turns to return
            api_key: API key for authentication
            
        Returns:
            Memory content
        """
        # Authenticate if required
        if API_KEY_REQUIRED and not self.authenticate(api_key or ""):
            return {"error": "Invalid API key"}
        
        try:
            # Get current conversation history
            history = self.rag.memory.get_history()
            
            # Limit to requested number of entries
            if limit > 0 and limit < len(history):
                history = history[-limit:]
            
            return {
                "status": "success",
                "memory": history
            }
            
        except Exception as e:
            self.logger.error(f"Error retrieving memory: {e}")
            return {"error": str(e)}
    
    def change_model(
        self, 
        model_name: str,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Change the active model.
        
        Args:
            model_name: Name of model to use
            api_key: API key for authentication
            
        Returns:
            Status message
        """
        # Authenticate if required
        if API_KEY_REQUIRED and not self.authenticate(api_key or ""):
            return {"error": "Invalid API key"}
        
        try:
            # Verify model exists
            if model_name not in MODELS:
                return {"error": f"Unknown model: {model_name}"}
            
            # Change model
            self.rag.change_model(model_name)
            
            return {
                "status": "success",
                "message": f"Changed model to {model_name}"
            }
            
        except Exception as e:
            self.logger.error(f"Error changing model: {e}")
            return {"error": str(e)}
    
    def count_tokens(
        self, 
        text: str,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Count tokens in text.
        
        Args:
            text: Text to tokenize
            api_key: API key for authentication
            
        Returns:
            Token count
        """
        # Authenticate if required
        if API_KEY_REQUIRED and not self.authenticate(api_key or ""):
            return {"error": "Invalid API key"}
        
        try:
            # Count tokens
            token_count = self.rag.llm.count_tokens(text)
            
            return {
                "status": "success",
                "token_count": token_count,
                "text_length": len(text)
            }
            
        except Exception as e:
            self.logger.error(f"Error counting tokens: {e}")
            return {"error": str(e)}
    
    def get_stats(self, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Get system statistics.
        
        Args:
            api_key: API key for authentication
            
        Returns:
            System statistics
        """
        # Authenticate if required
        if API_KEY_REQUIRED and not self.authenticate(api_key or ""):
            return {"error": "Invalid API key"}
        
        try:
            # Get document count
            document_count = self.rag.embedding_manager.get_document_count()
            
            # Get model info
            model_info = {
                "name": self.rag.llm.model_config["name"],
                "path": self.rag.llm.model_path,
                "embedding_model": self.rag.embedding_manager.model_name
            }
            
            # Get memory info
            memory_info = {
                "turns": len(self.rag.memory.history)
            }
            
            return {
                "status": "success",
                "document_count": document_count,
                "model": model_info,
                "memory": memory_info
            }
            
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}


# Create API wrapper examples for Python clients

def create_ouro_client(model_name: str = DEFAULT_MODEL, api_key: Optional[str] = None):
    """Create an Ouro client for easy API access.
    
    Args:
        model_name: Name of model to use
        api_key: API key for authentication
        
    Returns:
        OuroAPI instance
    """
    return OuroAPI(model_name=model_name)


def query_example():
    """Example of using the API for querying."""
    api = create_ouro_client()
    response = api.query(
        query_text="What is RAG?",
        use_history=True,
        return_contexts=True
    )
    print(json.dumps(response, indent=2))


def ingest_example():
    """Example of using the API for ingestion."""
    api = create_ouro_client()
    response = api.ingest_text(
        text="RAG stands for Retrieval Augmented Generation, a technique that enhances LLM responses with retrieved knowledge.",
        metadata={"source": "example", "topic": "rag"}
    )
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    # Run examples if script is executed directly
    print("Ouro API Examples:")
    print("\nQuery Example:")
    query_example()
    print("\nIngest Example:")
    ingest_example()