#!/bin/bash
# Ouro v2.5 Startup Script
echo "Starting Ouro v2.5..."

# Set working directory
cd "$(dirname "$0")"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
  echo "Ollama not found! Opening download page..."
  xdg-open "https://ollama.ai/download"
  exit 1
fi

# Start Qdrant if not already running
if ! nc -z localhost 6333 &>/dev/null; then
  echo "Starting Qdrant..."
  docker-compose -f config/docker-compose.yml up -d qdrant
fi

# Start Ollama if not already running
if ! pgrep -x "ollama" &>/dev/null; then
  echo "Starting Ollama..."
  ollama serve &
fi

# Wait for Ollama to start
echo "Waiting for Ollama to start..."
until curl -s http://localhost:11434/api/tags >/dev/null; do
  sleep 1
done

# Start the Python server
echo "Starting Ouro web interface..."
python3 start_server.py &

echo "Ouro v2.5 started successfully!"
