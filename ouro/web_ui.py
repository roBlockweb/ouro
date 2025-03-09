"""
Web interface for the Ouro RAG system.
"""
import os
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from ouro.config import (
    WEB_HOST, 
    WEB_PORT, 
    DEFAULT_MODEL, 
    MODELS, 
    TOP_K_RESULTS,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    API_ENABLED,
    API_PREFIX
)
from ouro.rag import OuroRAG
from ouro.logger import get_logger

logger = get_logger()

# Initialize the FastAPI app
app = FastAPI(
    title="Ouro RAG",
    description="Privacy-First Local RAG System",
    version="1.0.1"
)

# Get the directory where templates and static files are located
current_dir = Path(__file__).parent
templates = Jinja2Templates(directory=current_dir / "templates")
app.mount("/static", StaticFiles(directory=current_dir / "static"), name="static")

# Initialize the RAG system
rag = OuroRAG(model_config=MODELS[DEFAULT_MODEL])

# Store uploaded files temporarily
uploaded_files = []


# API models
class QueryRequest(BaseModel):
    query: str
    use_history: bool = True


class IngestRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None


class ModelRequest(BaseModel):
    model_name: str


class SettingsRequest(BaseModel):
    model_name: str
    top_k: int = TOP_K_RESULTS
    max_new_tokens: int = MAX_NEW_TOKENS
    temperature: float = TEMPERATURE


# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request):
    """Render the main page."""
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request, 
            "title": "Ouro - Privacy-First Local RAG System",
            "models": list(MODELS.keys()),
            "current_model": DEFAULT_MODEL
        }
    )


@app.post("/query")
async def query(query_request: QueryRequest):
    """Query the RAG system."""
    # Create async generator for streaming response
    async def generate():
        for token in rag.generate(
            query=query_request.query,
            with_history=query_request.use_history,
            stream=True
        ):
            yield token + " "  # Add space for better streaming display
            await asyncio.sleep(0.01)  # Small delay for better streaming
    
    # Return streaming response
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and ingest a file."""
    try:
        # Save uploaded file temporarily
        temp_file_path = Path(current_dir) / "uploads" / file.filename
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
        
        # Write file contents
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())
        
        # Add to list of uploaded files
        uploaded_files.append(str(temp_file_path))
        
        # Ingest the document in the background
        background_tasks.add_task(rag.ingest_document, str(temp_file_path))
        
        return {"status": "success", "message": f"File {file.filename} uploaded and queued for ingestion"}
    
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/text")
async def ingest_text(ingest_request: IngestRequest):
    """Ingest text directly."""
    try:
        num_docs = rag.ingest_text(
            text=ingest_request.text,
            metadata=ingest_request.metadata
        )
        return {"status": "success", "message": f"Ingested {num_docs} documents"}
    
    except Exception as e:
        logger.error(f"Error ingesting text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/change-model")
async def change_model(model_request: ModelRequest):
    """Change the LLM model."""
    try:
        if model_request.model_name not in MODELS:
            raise ValueError(f"Unknown model: {model_request.model_name}")
        
        rag.change_model(model_request.model_name)
        return {"status": "success", "message": f"Changed model to {model_request.model_name}"}
    
    except Exception as e:
        logger.error(f"Error changing model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/settings")
async def update_settings(settings_request: SettingsRequest):
    """Update RAG settings."""
    try:
        # Update model if needed
        if settings_request.model_name != rag.llm.model_config["name"]:
            rag.change_model(settings_request.model_name)
        
        # Update other settings
        rag.top_k = settings_request.top_k
        rag.llm.model.generation_config.max_new_tokens = settings_request.max_new_tokens
        rag.llm.model.generation_config.temperature = settings_request.temperature
        
        return {
            "status": "success", 
            "message": "Settings updated",
            "settings": {
                "model": settings_request.model_name,
                "top_k": rag.top_k,
                "max_new_tokens": rag.llm.model.generation_config.max_new_tokens,
                "temperature": rag.llm.model.generation_config.temperature
            }
        }
    
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear-memory")
async def clear_memory():
    """Clear conversation memory."""
    try:
        rag.clear_memory()
        return {"status": "success", "message": "Conversation memory cleared"}
    
    except Exception as e:
        logger.error(f"Error clearing memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# API routes if enabled
if API_ENABLED:
    # API route for querying
    @app.post(f"{API_PREFIX}/query")
    async def api_query(query_request: QueryRequest):
        """API endpoint for querying the RAG system."""
        response = ""
        for token in rag.generate(
            query=query_request.query,
            with_history=query_request.use_history,
            stream=False
        ):
            response += token
        
        # Ensure response is cleaned of any conversation artifacts
        response = rag.clean_response(response)
        
        return {
            "query": query_request.query,
            "response": response,
            "with_history": query_request.use_history
        }
    
    # API route for ingestion
    @app.post(f"{API_PREFIX}/ingest")
    async def api_ingest(ingest_request: IngestRequest):
        """API endpoint for ingesting text."""
        try:
            num_docs = rag.ingest_text(
                text=ingest_request.text,
                metadata=ingest_request.metadata
            )
            return {"status": "success", "ingested_documents": num_docs}
        
        except Exception as e:
            logger.error(f"Error ingesting text via API: {e}")
            raise HTTPException(status_code=500, detail=str(e))


def is_port_in_use(port):
    """Check if a port is in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_available_port(start_port, max_attempts=10):
    """Find an available port starting from start_port."""
    for port_offset in range(max_attempts):
        port = start_port + port_offset
        if not is_port_in_use(port):
            return port
    return None

def start_web_server():
    """Start the web server with automatic port selection if default is in use."""
    port = WEB_PORT
    
    if is_port_in_use(port):
        logger.warning(f"Port {port} is already in use, trying to find an available port...")
        port = find_available_port(WEB_PORT + 1)
        if port is None:
            logger.error(f"Could not find an available port after {10} attempts")
            print(f"❌ Could not start web server: all ports from {WEB_PORT} to {WEB_PORT + 10} are in use")
            return False
        
    logger.info(f"Starting web server on http://{WEB_HOST}:{port}")
    print(f"✅ Starting web interface at http://{WEB_HOST}:{port}")
    
    try:
        uvicorn.run(app, host=WEB_HOST, port=port)
        return True
    except Exception as e:
        logger.error(f"Error starting web server: {e}")
        print(f"❌ Could not start web server: {e}")
        return False