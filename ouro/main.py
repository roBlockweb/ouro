"""
Main entry point for Ouro RAG system.
"""
import os
import sys
import time
import asyncio
import platform
from pathlib import Path
if platform.system() == "Windows":
    import msvcrt
import importlib.util

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.progress import Progress
from rich.text import Text
from rich import box

from ouro.config import (
    DEFAULT_MODEL, 
    MODELS, 
    WEB_HOST, 
    WEB_PORT, 
    WEB_TIMEOUT,
    ROOT_DIR
)
from ouro.llm import check_hf_login
from ouro.rag import OuroRAG
from ouro.logger import get_logger

logger = get_logger()
console = Console()


def check_web_dependencies() -> bool:
    """Check if web interface dependencies are installed."""
    required_modules = ["fastapi", "uvicorn", "jinja2"]
    for module in required_modules:
        if importlib.util.find_spec(module) is None:
            return False
    return True


def run_web_interface():
    """Run the web interface."""
    try:
        from ouro.web_ui import start_web_server
        console.print(f"\n[bold green]Starting web interface at http://{WEB_HOST}:{WEB_PORT}[/bold green]")
        start_web_server()
    except ImportError as e:
        console.print("[bold red]Error: Web interface dependencies not installed.[/bold red]")
        console.print("Please install required packages with: pip install fastapi uvicorn jinja2")
        logger.error(f"Web interface error: {e}")
        return
    except Exception as e:
        console.print(f"[bold red]Error starting web interface: {e}[/bold red]")
        logger.error(f"Web interface error: {e}")


def run_terminal_interface():
    """Run the terminal interface."""
    # Initialize the RAG system
    with Progress(
        transient=True,
        expand=True
    ) as progress:
        task = progress.add_task("[cyan]Loading model...", total=1)
        
        try:
            rag = OuroRAG(model_config=MODELS[DEFAULT_MODEL])
            progress.update(task, completed=1)
        except Exception as e:
            progress.stop()
            console.print(f"[bold red]Error initializing RAG system: {e}[/bold red]")
            logger.error(f"RAG initialization error: {e}")
            return
    
    # Print welcome message
    console.print(Panel(
        Markdown("# Welcome to Ouro\n\nYour privacy-focused local RAG system. Type `/help` to see available commands."),
        title="Ouro RAG",
        border_style="cyan",
        box=box.ROUNDED
    ))
    
    # Command loop
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("\n[bold cyan]>>[/bold cyan]").strip()
            
            # Process commands
            if user_input.startswith("/"):
                command = user_input[1:].split(" ")[0]
                args = user_input[1:].split(" ")[1:] if len(user_input[1:].split(" ")) > 1 else []
                
                if command == "help":
                    show_help()
                
                elif command == "ingest":
                    if not args:
                        console.print("[yellow]Usage: /ingest <file_path>[/yellow]")
                        continue
                    
                    filepath = " ".join(args)
                    if not os.path.exists(filepath):
                        console.print(f"[bold red]Error: File {filepath} does not exist[/bold red]")
                        continue
                    
                    with Progress(transient=True) as progress:
                        task = progress.add_task("[cyan]Ingesting document...", total=1)
                        try:
                            num_docs = rag.ingest_document(filepath)
                            progress.update(task, completed=1)
                            console.print(f"[green]Ingested {num_docs} document chunks[/green]")
                        except Exception as e:
                            progress.stop()
                            console.print(f"[bold red]Error ingesting document: {e}[/bold red]")
                
                elif command == "ingest_dir":
                    if not args:
                        console.print("[yellow]Usage: /ingest_dir <directory_path>[/yellow]")
                        continue
                    
                    dirpath = " ".join(args)
                    if not os.path.exists(dirpath) or not os.path.isdir(dirpath):
                        console.print(f"[bold red]Error: Directory {dirpath} does not exist[/bold red]")
                        continue
                    
                    with Progress(transient=True) as progress:
                        task = progress.add_task("[cyan]Ingesting documents...", total=1)
                        try:
                            num_docs = rag.ingest_directory(dirpath)
                            progress.update(task, completed=1)
                            console.print(f"[green]Ingested {num_docs} document chunks[/green]")
                        except Exception as e:
                            progress.stop()
                            console.print(f"[bold red]Error ingesting documents: {e}[/bold red]")
                
                elif command == "ingest_text":
                    console.print("[cyan]Enter text to ingest (type '/end' on a new line when finished):[/cyan]")
                    lines = []
                    while True:
                        line = Prompt.ask("").strip()
                        if line == "/end":
                            break
                        lines.append(line)
                    
                    text = "\n".join(lines)
                    if not text:
                        console.print("[yellow]No text provided.[/yellow]")
                        continue
                    
                    with Progress(transient=True) as progress:
                        task = progress.add_task("[cyan]Ingesting text...", total=1)
                        try:
                            num_docs = rag.ingest_text(text)
                            progress.update(task, completed=1)
                            console.print(f"[green]Ingested {num_docs} document chunks[/green]")
                        except Exception as e:
                            progress.stop()
                            console.print(f"[bold red]Error ingesting text: {e}[/bold red]")
                
                elif command == "models":
                    list_models()
                
                elif command == "change_model":
                    if not args:
                        console.print("[yellow]Usage: /change_model <model_name>[/yellow]")
                        console.print("[yellow]Available models: small, medium, large, very_large, m1_optimized[/yellow]")
                        continue
                    
                    model_name = args[0]
                    if model_name not in MODELS:
                        console.print(f"[bold red]Error: Unknown model {model_name}[/bold red]")
                        console.print("[yellow]Available models: small, medium, large, very_large, m1_optimized[/yellow]")
                        continue
                    
                    with Progress(transient=True) as progress:
                        task = progress.add_task(f"[cyan]Changing model to {model_name}...", total=1)
                        try:
                            rag.change_model(model_name)
                            progress.update(task, completed=1)
                            console.print(f"[green]Changed model to {model_name}[/green]")
                        except Exception as e:
                            progress.stop()
                            console.print(f"[bold red]Error changing model: {e}[/bold red]")
                
                elif command == "clear_memory":
                    rag.clear_memory()
                    console.print("[green]Conversation memory cleared[/green]")
                
                elif command == "exit" or command == "quit":
                    if confirm_exit():
                        break
                
                elif command == "learn":
                    with Progress(transient=True) as progress:
                        task = progress.add_task("[cyan]Learning from conversations...", total=1)
                        try:
                            rag.learn_from_conversations(fine_tune=True)
                            progress.update(task, completed=1)
                            console.print("[green]Learning completed[/green]")
                        except Exception as e:
                            progress.stop()
                            console.print(f"[bold red]Error during learning: {e}[/bold red]")
                
                else:
                    console.print(f"[bold red]Unknown command: {command}[/bold red]")
                    console.print("[yellow]Type /help to see available commands[/yellow]")
            
            # Process query
            else:
                if not user_input:
                    continue
                
                with Progress(
                    "{task.description}",
                    transient=True
                ) as progress:
                    # Step 1: Search for relevant documents
                    search_task = progress.add_task("[cyan]Searching knowledge base...", total=1)
                    contexts = rag.get_contexts(user_input)
                    progress.update(search_task, completed=1)
                    
                    # Step 2: Generate response
                    gen_task = progress.add_task("[cyan]Generating response...", total=1)
                    
                    # Get initial part of response to display progress
                    response_parts = []
                    response_generator = rag.generate(user_input)
                    for i, token in enumerate(response_generator):
                        response_parts.append(token)
                        if i == 5:  # After getting a few tokens, update progress
                            progress.update(gen_task, completed=0.3)
                        elif i == 15:
                            progress.update(gen_task, completed=0.6)
                        elif i == 30:
                            progress.update(gen_task, completed=0.9)
                    
                    progress.update(gen_task, completed=1)
                
                # Display the response
                response = "".join(response_parts)
                console.print(Panel(
                    Markdown(response),
                    border_style="green",
                    box=box.ROUNDED
                ))
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation interrupted. Type /exit to quit.[/yellow]")
            continue
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
            logger.error(f"Error in terminal interface: {e}")
            continue


def show_help():
    """Show help information."""
    help_text = """
# Ouro Commands

- `/ingest <file_path>` - Ingest a document into the knowledge base
- `/ingest_dir <directory_path>` - Ingest all documents in a directory
- `/ingest_text` - Ingest text directly
- `/models` - List available models
- `/change_model <model_name>` - Change the active model
- `/clear_memory` - Clear conversation memory
- `/learn` - Learn from past conversations
- `/help` - Show this help message
- `/exit` or `/quit` - Exit the application
    """
    console.print(Panel(
        Markdown(help_text),
        title="Help",
        border_style="cyan",
        box=box.ROUNDED
    ))


def list_models():
    """List available models."""
    console.print(Panel(
        "\n".join([f"- [cyan]{name}[/cyan]: {model['llm_model']}" for name, model in MODELS.items()]),
        title="Available Models",
        border_style="cyan",
        box=box.ROUNDED
    ))


def confirm_exit() -> bool:
    """Confirm exit."""
    response = Prompt.ask("[yellow]Are you sure you want to exit?[/yellow] (y/n)", default="n")
    return response.lower() in ["y", "yes"]


def wait_for_user_choice(timeout: int = WEB_TIMEOUT) -> str:
    """Wait for user to choose interface mode."""
    console.print(Panel(
        "Press [bold cyan]T[/bold cyan] for Terminal Interface or [bold cyan]O[/bold cyan] for Web Interface",
        title="Choose Interface",
        border_style="cyan",
        box=box.ROUNDED
    ))
    
    console.print(f"[yellow]Auto-selecting Web Interface in {timeout} seconds...[/yellow]")
    
    # Simpler implementation that works on all platforms
    try:
        # Use Python's builtin input with a timeout using select on Unix
        # or just input on Windows (will block until user input)
        if platform.system() != "Windows":
            import select
            import termios
            import tty
            import sys
            
            def getch_unix():
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(sys.stdin.fileno())
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if rlist:
                        ch = sys.stdin.read(1)
                        return ch
                    return None
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            
            start_time = time.time()
            while time.time() - start_time < timeout:
                ch = getch_unix()
                if ch == 't':
                    console.print("[bold cyan]Terminal Interface selected[/bold cyan]")
                    return "terminal"
                elif ch == 'o':
                    console.print("[bold cyan]Web Interface selected[/bold cyan]")
                    return "web"
                
                # Update remaining time every second
                elapsed = int(time.time() - start_time)
                remaining = timeout - elapsed
                if remaining != timeout - (elapsed - 1) and remaining > 0:
                    console.print(f"[yellow]Auto-selecting Web Interface in {remaining} seconds...[/yellow]", end="\r")
                
                time.sleep(0.1)
        else:
            # Windows implementation
            start_time = time.time()
            while time.time() - start_time < timeout:
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8').lower()
                    if key == 't':
                        console.print("[bold cyan]Terminal Interface selected[/bold cyan]")
                        return "terminal"
                    elif key == 'o':
                        console.print("[bold cyan]Web Interface selected[/bold cyan]")
                        return "web"
                
                # Update remaining time every second
                elapsed = int(time.time() - start_time)
                remaining = timeout - elapsed
                if remaining != timeout - (elapsed - 1) and remaining > 0:
                    console.print(f"[yellow]Auto-selecting Web Interface in {remaining} seconds...[/yellow]", end="\r")
                
                time.sleep(0.1)
    
    except Exception as e:
        # If any error occurs with the fancy input methods, fall back to basic behavior
        logger.warning(f"Error in user choice input: {e}")
        console.print("[yellow]Error detecting keypress. Defaulting to web interface.[/yellow]")
    
    console.print("[bold cyan]Web Interface auto-selected[/bold cyan]")
    return "web"


def main():
    """Main entry point for Ouro."""
    console.print("[bold cyan]Starting Ouro RAG System...[/bold cyan]")
    
    # Log warning if user is not logged in to Hugging Face
    if not check_hf_login():
        # This code won't be reached now that check_hf_login always returns True
        # But we'll keep it for future modifications if needed
        pass
        
    # Inform about Hugging Face login recommendation
    console.print(Panel(
        "For optimal experience, it's recommended to log in to Hugging Face.\n"
        "If you experience model download issues, please run [bold]huggingface-cli login[/bold] first.",
        title="Hugging Face Authentication",
        border_style="yellow",
        box=box.ROUNDED
    ))
    
    # Let user choose interface mode
    choice = wait_for_user_choice()
    
    if choice == "web":
        if check_web_dependencies():
            run_web_interface()
        else:
            console.print(Panel(
                "Web interface dependencies not installed.\n"
                "Please install required packages with:\n[bold]pip install fastapi uvicorn jinja2[/bold]\n"
                "Falling back to terminal interface.",
                title="Missing Dependencies",
                border_style="yellow",
                box=box.ROUNDED
            ))
            run_terminal_interface()
    else:
        run_terminal_interface()


if __name__ == "__main__":
    main()