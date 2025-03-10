"""
Main entry point for Ouro RAG system.
"""
import os
import sys
import platform
import readline
from pathlib import Path
import traceback

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
    ROOT_DIR
)
from ouro.llm import check_hf_login
from ouro.rag import OuroRAG
from ouro.logger import get_logger

logger = get_logger()
console = Console()

# Define all available slash commands for auto-completion
SLASH_COMMANDS = [
    "/chat", 
    "/ingest", 
    "/ingest_dir", 
    "/ingest_text", 
    "/models", 
    "/change_model", 
    "/clear_memory", 
    "/learn", 
    "/help", 
    "/exit", 
    "/quit"
]

def completer(text, state):
    """Command autocomplete function for readline."""
    # Get current input text
    buffer = readline.get_line_buffer()
    
    # Only provide suggestions for slash commands
    if buffer.startswith("/"):
        options = [cmd for cmd in SLASH_COMMANDS if cmd.startswith(buffer)]
        if state < len(options):
            return options[state]
    return None


def run_terminal_interface():
    """Run the terminal interface."""
    # Set up command auto-completion
    if platform.system() != "Windows":  # readline not available on Windows
        readline.parse_and_bind("tab: complete")
        readline.set_completer(completer)
    
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
                
                elif command == "chat":
                    # Chat mode with a topic
                    if args:
                        topic = " ".join(args)
                        console.print(f"[cyan]Starting casual chat about: {topic}[/cyan]")
                        # Process as a query using the topic with chat mode
                        process_query(rag, topic, mode="chat")
                    else:
                        console.print("[cyan]Starting casual chat mode. Type 'exit' to return to command mode.[/cyan]")
                        # Enter interactive chat mode
                        while True:
                            chat_input = Prompt.ask("[bold green]Chat>>[/bold green]").strip()
                            if chat_input.lower() in ["exit", "quit", "/exit", "/quit"]:
                                console.print("[cyan]Exiting chat mode.[/cyan]")
                                break
                            
                            # Check if user entered a slash command
                            if chat_input.startswith("/"):
                                # Extract command
                                chat_command = chat_input[1:].split(" ")[0]
                                chat_args = chat_input[1:].split(" ")[1:] if len(chat_input[1:].split(" ")) > 1 else []
                                
                                # Process command within chat
                                process_command(rag, chat_command, chat_args)
                                continue
                            
                            # Process as regular chat message with chat mode
                            process_query(rag, chat_input, mode="chat")
                
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
            
            # Process query (non-command input)
            else:
                if not user_input:
                    continue
                
                process_query(rag, user_input)
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation interrupted. Type /exit to quit.[/yellow]")
            continue
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
            logger.error(f"Error in terminal interface: {e}")
            logger.error(traceback.format_exc())
            continue


def process_command(rag, command, args):
    """Process a command within chat mode."""
    if command == "help":
        show_help()
    elif command == "clear_memory":
        rag.clear_memory()
        console.print("[green]Conversation memory cleared[/green]")
    elif command == "models":
        list_models()
    elif command == "change_model":
        if not args:
            console.print("[yellow]Usage: /change_model <model_name>[/yellow]")
            console.print("[yellow]Available models: small, medium, large, very_large, m1_optimized[/yellow]")
            return
        
        model_name = args[0]
        if model_name not in MODELS:
            console.print(f"[bold red]Error: Unknown model {model_name}[/bold red]")
            console.print("[yellow]Available models: small, medium, large, very_large, m1_optimized[/yellow]")
            return
        
        with Progress(transient=True) as progress:
            task = progress.add_task(f"[cyan]Changing model to {model_name}...", total=1)
            try:
                rag.change_model(model_name)
                progress.update(task, completed=1)
                console.print(f"[green]Changed model to {model_name}[/green]")
            except Exception as e:
                progress.stop()
                console.print(f"[bold red]Error changing model: {e}[/bold red]")
    else:
        console.print(f"[yellow]Command '/{command}' not available in chat mode. Use 'exit' to return to main mode.[/yellow]")


def process_query(rag, user_input, mode="standard"):
    """Process a user query and display the response.
    
    Args:
        rag: The RAG system instance
        user_input: The user's query
        mode: The generation mode ("standard", "chat", or "agent")
    """
    if not user_input:
        return
    
    with Progress(
        "{task.description}",
        transient=True
    ) as progress:
        # Step 1: Search for relevant documents
        search_task = progress.add_task("[cyan]Searching knowledge base...", total=1)
        contexts = rag.get_contexts(user_input)
        progress.update(search_task, completed=1)
        
        # Step 2: Generate response
        progress_message = "[cyan]Generating response..."
        if mode == "chat":
            progress_message = "[cyan]Thinking..."
        
        gen_task = progress.add_task(progress_message, total=1)
        
        # Get initial part of response to display progress
        response_parts = []
        response_generator = rag.generate(
            query=user_input,
            mode=mode,
            context=contexts
        )
        
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
    
    # Choose border color based on mode
    border_style = "green"
    if mode == "chat":
        border_style = "blue"
    elif mode == "agent":
        border_style = "purple"
    
    console.print(Panel(
        Markdown(response),
        border_style=border_style,
        box=box.ROUNDED
    ))


def show_help():
    """Show help information."""
    help_text = """
# Ouro Commands

- `/chat [topic]` - Start casual chat mode (optional topic)
- `/ingest <file_path>` - Ingest a document into the knowledge base
- `/ingest_dir <directory_path>` - Ingest all documents in a directory
- `/ingest_text` - Ingest text directly
- `/models` - List available models
- `/change_model <model_name>` - Change the active model
- `/clear_memory` - Clear conversation memory
- `/learn` - Learn from past conversations
- `/help` - Show this help message
- `/exit` or `/quit` - Exit the application

In chat mode, you can also use `/help`, `/clear_memory`, `/models`, and `/change_model` commands.
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
    
    # Run terminal interface directly
    run_terminal_interface()


if __name__ == "__main__":
    main()