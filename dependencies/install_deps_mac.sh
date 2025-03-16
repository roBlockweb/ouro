#!/bin/bash
# Ouro v2.1 Dependency Installer for macOS
echo "Installing dependencies for Ouro v2.1..."

# Check for Docker
if ! command -v docker &> /dev/null; then
  echo "Docker not found. Opening download page..."
  open "https://www.docker.com/products/docker-desktop/"
fi

# Check for Ollama
if [ ! -d "/Applications/Ollama.app" ]; then
  echo "Ollama not found. Opening download page..."
  open "https://ollama.ai/download"
fi

# Install model
if command -v ollama &> /dev/null; then
  echo "Installing default model (llama3:8b)..."
  ollama pull llama3:8b
  echo "Installing embedding model..."
  ollama pull nomic-embed-text
fi

echo "Dependency installation completed!"
