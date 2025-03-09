#!/bin/bash
# Comprehensive cleanup script for Ouro RAG system
# This script resets the project to a fresh state by removing all generated files

# ANSI color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Disable colors if not in a terminal
if [ ! -t 1 ]; then
    GREEN=''
    YELLOW=''
    RED=''
    BLUE=''
    BOLD=''
    NC=''
fi

# Display banner
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}           Ouro RAG System - Complete Cleanup               ${NC}"
echo -e "${BOLD}============================================================${NC}"
echo -e "This script will reset Ouro to a clean state by removing all"
echo -e "generated files and directories."
echo -e "${BOLD}============================================================${NC}"
echo ""

# Ask for confirmation
echo -e "${YELLOW}${BOLD}WARNING: This will delete all:${NC}"
echo -e "${YELLOW}- Downloaded models${NC}"
echo -e "${YELLOW}- Embeddings and vector stores${NC}"
echo -e "${YELLOW}- Logs and conversation history${NC}"
echo -e "${YELLOW}- User documents and uploads${NC}"
echo -e "${YELLOW}- Cache files and temporary data${NC}"
echo ""
echo -e "${RED}${BOLD}This action cannot be undone!${NC}"
echo ""

read -p "Are you sure you want to proceed? (y/N): " CONFIRM
if [[ $CONFIRM != "y" && $CONFIRM != "Y" ]]; then
    echo -e "${BLUE}Cleanup cancelled.${NC}"
    exit 0
fi

echo ""
echo -e "${BOLD}Starting deep cleanup...${NC}"
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Define function to remove directories and files
remove_item() {
    if [ -e "$1" ]; then
        if [ -d "$1" ]; then
            echo -e "${GREEN}Removing directory: $1${NC}"
            rm -rf "$1"
        else
            echo -e "${GREEN}Removing file: $1${NC}"
            rm -f "$1"
        fi
    fi
}

# Clean all generated content
echo -e "${BOLD}Removing data directories...${NC}"
remove_item "$SCRIPT_DIR/data"
remove_item "$SCRIPT_DIR/ouro/data"

echo -e "${BOLD}Removing log files...${NC}"
remove_item "$SCRIPT_DIR/logs"
remove_item "$SCRIPT_DIR/ouro/logs"

echo -e "${BOLD}Removing installation markers...${NC}"
remove_item "$SCRIPT_DIR/.installed"
remove_item "$SCRIPT_DIR/ouro.egg-info"

echo -e "${BOLD}Removing Python cache files...${NC}"
find "$SCRIPT_DIR" -type d -name "__pycache__" -exec rm -rf {} +  2>/dev/null || true
find "$SCRIPT_DIR" -name "*.pyc" -delete 2>/dev/null || true
find "$SCRIPT_DIR" -name "*.pyo" -delete 2>/dev/null || true
find "$SCRIPT_DIR" -name "*.pyd" -delete 2>/dev/null || true

echo -e "${BOLD}Removing temporary uploads...${NC}"
remove_item "$SCRIPT_DIR/ouro/uploads"

# Check if virtual environment should be removed too
echo ""
read -p "Do you also want to remove the virtual environment? (y/N): " REMOVE_VENV
if [[ $REMOVE_VENV == "y" || $REMOVE_VENV == "Y" ]]; then
    echo -e "${BOLD}Removing virtual environment...${NC}"
    remove_item "$SCRIPT_DIR/venv"
fi

echo ""
echo -e "${BOLD}${GREEN}Cleanup completed successfully!${NC}"
echo -e "The project has been reset to its initial state."
echo ""
echo -e "To reinstall Ouro, run:"
echo -e "  python install.py"
echo ""