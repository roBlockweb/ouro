#!/bin/bash

# Ouro v2.5 Standalone Build Script for all platforms
echo "Building Ouro v2.5 standalone packages..."

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
  echo "Installing dependencies..."
  npm install
fi

# Build for all platforms
echo "Building for multiple platforms..."

# Check platform
if [ "$(uname)" == "Darwin" ]; then
  # macOS
  echo "Building for macOS..."
  npm run build:mac
  
  # Optionally build for Windows/Linux if configured
  if command -v wine &> /dev/null && [ -d "$HOME/.wine" ]; then
    echo "Building for Windows..."
    npm run build:win
  fi
else
  # Linux/Windows
  echo "Building for current platform..."
  npm run build
fi

echo "Build complete! Output is in the 'dist' directory."
