#!/bin/bash

# Ouro v2.5 DMG Build Script
echo "Building Ouro v2.5 DMG for macOS..."

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
  echo "Installing dependencies..."
  npm install
fi

# Run electron-builder for macOS DMG
echo "Running electron-builder..."
NODE_ENV=production npm run build

echo "Build complete! DMG is in the 'dist' directory."
