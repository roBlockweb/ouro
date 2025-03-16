#!/usr/bin/env python3
"""
Ouro v2.5 Server
This script starts a HTTP server for the Ouro web interface and API
"""

import os
import sys
import json
import yaml
import http.server
import socketserver
import webbrowser
from pathlib import Path
import threading
import time
import argparse
import logging
import urllib.request
import urllib.error
import urllib.parse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ouro')

class OuroHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler for Ouro"""
    
    def __init__(self, *args, **kwargs):
        self.config = load_config()
        web_dir = Path(__file__).parent / 'web'
        os.chdir(web_dir)
        super().__init__(*args, **kwargs)
    
    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info("%s - - [%s] %s" % (self.address_string(),
                                        self.log_date_time_string(),
                                        format % args))
    
    def do_GET(self):
        """Handle GET requests"""
        # Handle API calls
        if self.path.startswith('/api/'):
            self.handle_api_get()
        else:
            # Serve static files
            super().do_GET()
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path.startswith('/api/'):
            self.handle_api_post()
        else:
            self.send_error(404, "Not Found")
    
    def handle_api_get(self):
        """Handle API GET requests"""
        if self.path == '/api/config':
            self.send_json_response(load_config())
        elif self.path == '/api/models':
            models = self.get_available_models()
            self.send_json_response(models)
        else:
            self.send_error(404, "API endpoint not found")
    
    def handle_api_post(self):
        """Handle API POST requests"""
        if self.path == '/api/chat':
            self.handle_chat_request()
        elif self.path == '/api/config/update':
            self.handle_config_update()
        else:
            self.send_error(404, "API endpoint not found")
    
    def handle_chat_request(self):
        """Handle chat request to Ollama API"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            model = request_data.get('model', self.config.get('ollama', {}).get('model', 'llama3:8b'))
            prompt = request_data.get('prompt', '')
            
            # Call Ollama API
            response_data = self.call_ollama_api(model, prompt)
            
            self.send_json_response(response_data)
        except Exception as e:
            logger.error(f"Error handling chat request: {e}")
            self.send_json_response({"error": str(e)}, status=500)
    
    def handle_config_update(self):
        """Handle config update request"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            update_data = json.loads(post_data.decode('utf-8'))
            
            # Update config
            current_config = load_config()
            
            # Update only the keys provided
            for key, value in update_data.items():
                if isinstance(value, dict) and key in current_config and isinstance(current_config[key], dict):
                    # Handle nested dictionaries
                    current_config[key].update(value)
                else:
                    # Handle top-level values
                    current_config[key] = value
            
            # Save updated config
            save_config(current_config)
            
            self.send_json_response({"success": True})
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            self.send_json_response({"error": str(e)}, status=500)
    
    def send_json_response(self, data, status=200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def call_ollama_api(self, model, prompt):
        """Call Ollama API"""
        try:
            url = "http://localhost:11434/api/generate"
            headers = {"Content-Type": "application/json"}
            data = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            
            req = urllib.request.Request(
                url,
                json.dumps(data).encode('utf-8'),
                headers
            )
            
            with urllib.request.urlopen(req) as response:
                response_data = json.loads(response.read().decode('utf-8'))
                return {
                    "model": model,
                    "prompt": prompt,
                    "response": response_data.get("response", "")
                }
        except urllib.error.URLError as e:
            logger.error(f"Error calling Ollama API: {e}")
            return {
                "model": model,
                "prompt": prompt,
                "response": "I'm sorry, I couldn't connect to the Ollama API. Please make sure Ollama is running.",
                "error": str(e)
            }
    
    def get_available_models(self):
        """Get available models from Ollama"""
        try:
            url = "http://localhost:11434/api/tags"
            headers = {"Content-Type": "application/json"}
            
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req) as response:
                response_data = json.loads(response.read().decode('utf-8'))
                models = []
                
                if "models" in response_data:
                    for model in response_data["models"]:
                        models.append({
                            "name": model.get("name", ""),
                            "size": model.get("size", 0),
                            "modified_at": model.get("modified_at", ""),
                            "details": model.get("details", {})
                        })
                
                return {"models": models}
        except urllib.error.URLError as e:
            logger.error(f"Error getting available models: {e}")
            return {"models": [], "error": str(e)}

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / 'config' / 'config.yaml'
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        else:
            logger.warning(f"Config file not found at {config_path}, using default config")
            return {
                'version': '2.5',
                'ollama': {'model': 'llama3:8b', 'embeddings': 'nomic-embed-text'},
                'qdrant': {'url': 'http://localhost:6333', 'collection': 'ouro_docs'},
                'ui': {'theme': 'dark', 'port': 3000}
            }
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {
            'ui': {'port': 3000}
        }

def save_config(config):
    """Save configuration to config.yaml"""
    config_path = Path(__file__).parent / 'config' / 'config.yaml'
    try:
        config_dir = config_path.parent
        if not config_dir.exists():
            config_dir.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return True
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        return False

def check_ollama():
    """Check if Ollama is running"""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags") as response:
            return response.status == 200
    except Exception:
        return False

def open_browser(port):
    """Open the browser after a short delay"""
    time.sleep(1)
    webbrowser.open(f"http://localhost:{port}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Ouro v2.5 Server')
    parser.add_argument('--port', type=int, help='Port to run the server on')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Use port from args or config
    port = args.port or config.get('ui', {}).get('port', 3000)
    
    # Check Ollama
    if not check_ollama():
        logger.warning("Ollama is not running. Some features may not work correctly.")
    
    # Start HTTP server
    handler = OuroHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            logger.info(f"Serving at http://localhost:{port}")
            
            # Open browser unless disabled
            if not args.no_browser:
                threading.Thread(target=open_browser, args=(port,), daemon=True).start()
            
            # Start server
            httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped")
    except OSError as e:
        if e.errno == 98:  # Address already in use
            logger.error(f"Port {port} is already in use. Try using a different port with --port")
        else:
            logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()