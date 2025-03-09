#!/usr/bin/env python
"""
Quick launch script for Ouro RAG system.
Handles first-time setup and redirects to the main application.
"""
import os
import sys
import platform
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).parent.absolute()
VENV_DIR = ROOT_DIR / "venv"
INSTALLED_FLAG = ROOT_DIR / ".installed"


def print_banner():
    """Print welcome banner."""
    logo = """
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃         ___  _  _ ____  ___      ┃
    ┃        / _ \\| || |  _ \\/ _ \\     ┃
    ┃       | | | | || | |_) | | | |   ┃
    ┃       | |_| |__   _  <| |_| |    ┃
    ┃        \\___/   |_|_| \\_\\\\___/    ┃
    ┃                                  ┃
    ┃   Privacy-First Local RAG System ┃
    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    """
    print(logo)


def is_first_run():
    """Check if this is the first run."""
    return not INSTALLED_FLAG.exists()


def setup_ouro():
    """Run the installation script."""
    print("First-time setup detected. Running installation...")
    print()
    
    install_script = ROOT_DIR / "install.py"
    if not install_script.exists():
        print(f"Error: Installation script {install_script} not found.")
        return False
    
    try:
        subprocess.check_call([sys.executable, str(install_script)])
        INSTALLED_FLAG.touch()
        return True
    except subprocess.CalledProcessError:
        print("Installation failed.")
        return False


def run_ouro():
    """Run the main Ouro application."""
    if platform.system() == "Windows":
        run_script = ROOT_DIR / "run.bat"
        try:
            subprocess.check_call([str(run_script)] + sys.argv[1:])
            return True
        except subprocess.CalledProcessError:
            print("Error running Ouro.")
            return False
    else:
        run_script = ROOT_DIR / "run.sh"
        if not os.access(run_script, os.X_OK):
            os.chmod(run_script, 0o755)
        
        try:
            subprocess.check_call([str(run_script)] + sys.argv[1:])
            return True
        except subprocess.CalledProcessError:
            print("Error running Ouro.")
            return False


def main():
    """Main entry point."""
    print_banner()
    
    if is_first_run():
        if not setup_ouro():
            return False
    
    return run_ouro()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)