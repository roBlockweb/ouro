#!/bin/bash

# Ouro v2.5 macOS Build Script
echo "Building Ouro v2.5 for macOS..."

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
  echo "Installing dependencies..."
  npm install
fi

# Run electron-builder for macOS
echo "Running electron-builder..."
npm run build:mac

echo "Build complete! Output is in the 'dist' directory."
