#!/usr/bin/env python
"""
Installation script for Ouro RAG system.
Sets up the virtual environment, installs dependencies, and checks for prerequisites.
"""
import os
import sys
import subprocess
import platform
from pathlib import Path

ROOT_DIR = Path(__file__).parent.absolute()
VENV_DIR = ROOT_DIR / "venv"


def print_banner():
    """Print installation banner."""
    print("=" * 60)
    print("            Ouro RAG System Installation")
    print("=" * 60)
    print("This script will set up Ouro on your system.")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print("=" * 60)


def check_python_version():
    """Check if Python version is supported."""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 9):
        print(f"Error: Python 3.9+ is required. You have Python {major}.{minor}")
        return False
    return True


def create_venv():
    """Create virtual environment if it doesn't exist."""
    if not VENV_DIR.exists():
        print(f"Creating virtual environment in {VENV_DIR}...")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
            print("Virtual environment created successfully.")
            return True
        except subprocess.CalledProcessError:
            print("Error: Failed to create virtual environment.")
            return False
    else:
        print(f"Virtual environment already exists in {VENV_DIR}")
        return True


def install_package():
    """Install the package in development mode."""
    if platform.system() == "Windows":
        pip_path = VENV_DIR / "Scripts" / "pip"
        python_path = VENV_DIR / "Scripts" / "python"
    else:
        pip_path = VENV_DIR / "bin" / "pip"
        python_path = VENV_DIR / "bin" / "python"
    
    # Make sure pip is up to date
    print("Updating pip...")
    try:
        subprocess.check_call([str(pip_path), "install", "--upgrade", "pip"])
    except subprocess.CalledProcessError:
        print("Warning: Failed to update pip.")

    # Install the package
    print("Installing Ouro in development mode...")
    try:
        subprocess.check_call([str(pip_path), "install", "-e", str(ROOT_DIR)])
        print("Ouro installed successfully.")
        return True
    except subprocess.CalledProcessError:
        print("Error: Failed to install Ouro.")
        return False


def check_huggingface():
    """Check if user is logged in to Hugging Face and provide guidance."""
    token_path = os.path.expanduser("~/.huggingface/token")
    if os.path.exists(token_path):
        print("Hugging Face authentication found. Great!")
        return True
    else:
        print("=" * 60)
        print("HUGGING FACE AUTHENTICATION RECOMMENDATION")
        print("=" * 60)
        print("We noticed you're not logged in to Hugging Face.")
        print("While Ouro will continue to work, logging in is recommended")
        print("for better access to models and to avoid download issues.")
        print("\nTo log in, run: huggingface-cli login")
        print("=" * 60)
        return True  # Return True to continue installation regardless


def create_directories():
    """Create all necessary directories for the application."""
    directories = [
        # Main data directories
        ROOT_DIR / "ouro" / "data" / "documents",
        ROOT_DIR / "ouro" / "data" / "models",
        ROOT_DIR / "ouro" / "data" / "vector_store",
        ROOT_DIR / "ouro" / "data" / "conversations",
        
        # Root data directories (for backward compatibility)
        ROOT_DIR / "data" / "documents",
        ROOT_DIR / "data" / "models",
        ROOT_DIR / "data" / "vector_store",
        ROOT_DIR / "data" / "conversations",
        
        # Log directories
        ROOT_DIR / "ouro" / "logs",
        ROOT_DIR / "logs",
        
        # Web uploads directory
        ROOT_DIR / "ouro" / "uploads",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create marker file
    (ROOT_DIR / ".installed").touch()
    
    print("All necessary directories created successfully.")
    return True


def main():
    """Main installation function."""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create virtual environment
    if not create_venv():
        return False
    
    # Install package
    if not install_package():
        return False
    
    # Create directories
    if not create_directories():
        return False
    
    # Check Hugging Face login
    check_huggingface()
    
    print("\nInstallation completed successfully.")
    print("You can now run Ouro using one of the following commands:")
    if platform.system() == "Windows":
        print("  run.bat")
    else:
        print("  ./run.sh")
    print("\nFor more information, please read the README.md file.")
    
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)