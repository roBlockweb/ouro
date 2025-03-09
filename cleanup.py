#!/usr/bin/env python
"""
Cleanup script for Ouro RAG system.
This script removes all generated files and directories, resetting the project to its initial state.
"""
import os
import sys
import shutil
from pathlib import Path
import platform

# Terminal colors
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

    @staticmethod
    def disable_if_windows():
        if platform.system() == "Windows":
            Colors.GREEN = ''
            Colors.YELLOW = ''
            Colors.RED = ''
            Colors.BOLD = ''
            Colors.END = ''


def print_banner():
    """Print cleanup banner."""
    Colors.disable_if_windows()
    print(f"{Colors.BOLD}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}            Ouro RAG System Cleanup{Colors.END}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.END}")
    print("This script will remove all generated files and directories,")
    print("resetting the project to its initial state.")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.END}")


def confirm_cleanup():
    """Ask for confirmation before proceeding."""
    print(f"\n{Colors.YELLOW}WARNING: This will delete all generated files, including:{Colors.END}")
    print(f"{Colors.YELLOW}- All downloaded models{Colors.END}")
    print(f"{Colors.YELLOW}- All vector stores and embeddings{Colors.END}")
    print(f"{Colors.YELLOW}- All log files and conversation history{Colors.END}")
    print(f"{Colors.YELLOW}- All uploaded documents{Colors.END}")
    print(f"{Colors.YELLOW}This action cannot be undone.{Colors.END}\n")
    
    response = input(f"{Colors.BOLD}Are you sure you want to proceed? (y/N): {Colors.END}").strip().lower()
    return response == 'y'


def remove_directory(path):
    """Remove a directory if it exists."""
    try:
        if path.exists():
            if path.is_file():
                os.remove(path)
                print(f"{Colors.GREEN}Removed file: {path}{Colors.END}")
            else:
                shutil.rmtree(path)
                print(f"{Colors.GREEN}Removed directory: {path}{Colors.END}")
        return True
    except Exception as e:
        print(f"{Colors.RED}Error removing {path}: {e}{Colors.END}")
        return False


def clean_project():
    """Remove all generated files and directories."""
    ROOT_DIR = Path(__file__).parent.absolute()
    
    # List of paths to clean (files and directories)
    cleanup_paths = [
        # Data directories
        ROOT_DIR / "data",
        ROOT_DIR / "ouro" / "data",
        
        # Log directories
        ROOT_DIR / "logs",
        ROOT_DIR / "ouro" / "logs",
        
        # Installation markers
        ROOT_DIR / ".installed",
        ROOT_DIR / "ouro.egg-info",
        
        # Cache files
        ROOT_DIR / "__pycache__",
        ROOT_DIR / "ouro" / "__pycache__",
    ]
    
    success = True
    for path in cleanup_paths:
        if not remove_directory(path):
            success = False
    
    # Cleanup Python cache files recursively
    for pycache_dir in ROOT_DIR.glob("**/__pycache__"):
        if not remove_directory(pycache_dir):
            success = False
    
    return success


def main():
    """Main cleanup function."""
    print_banner()
    
    if not confirm_cleanup():
        print(f"\n{Colors.YELLOW}Cleanup cancelled.{Colors.END}")
        return True
    
    print(f"\n{Colors.BOLD}Starting cleanup...{Colors.END}")
    
    if clean_project():
        print(f"\n{Colors.GREEN}Cleanup completed successfully.{Colors.END}")
        print(f"{Colors.GREEN}The project has been reset to its initial state.{Colors.END}")
        print(f"\nYou can now run the installation script to set up Ouro:")
        print(f"  python install.py")
        return True
    else:
        print(f"\n{Colors.RED}Cleanup completed with errors.{Colors.END}")
        print(f"{Colors.RED}Some files or directories could not be removed.{Colors.END}")
        return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)