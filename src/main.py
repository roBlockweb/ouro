"""
Main CLI interface for the Ouro RAG system
"""
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, SpinnerColumn
from rich.markdown import Markdown
from rich.table import Table
from rich import print as rprint

from src.rag import OuroRAG
from src.llm import (
    get_available_model_sizes, 
    get_custom_model_template,
    login_huggingface, 
    get_huggingface_login_instructions,
    load_model_with_progress
)
from src.embeddings import load_embedding_with_progress, get_available_embedding_models
from src.logger import logger
from src.config import (
    DOCUMENTS_DIR, 
    DEFAULT_EMBEDDING_MODEL,
    MODEL_CONFIGURATIONS,
    EMBEDDING_MODELS,
    DEFAULT_SIZE,
    DEFAULT_QUANTIZE,
    DEFAULT_FAST_MODE,
    DEFAULT_MEMORY_TURNS,
    DEFAULT_SAVE_CONVERSATIONS
)

# Initialize rich console
console = Console()

def welcome_message():
    """Display a welcome message"""
    welcome_text = """
    Welcome to [bold green]Ouro[/bold green] - Your Privacy-First Local RAG System
    
    Ouro is a Retrieval-Augmented Generation system that operates entirely offline
    using local LLMs and embeddings. All your data stays on your machine.
    
    Type [bold cyan]'help'[/bold cyan] at any time to see available commands.
    Type [bold cyan]'exit'[/bold cyan] to quit.
    """
    console.print(Panel(welcome_text, title="Ouro", border_style="green"))

def print_help():
    """Display help information"""
    help_text = """
    [bold]Available Commands:[/bold]
    
    [bold cyan]help[/bold cyan] - Display this help message
    
    [bold]Document Management:[/bold]
    [bold cyan]ingest <file_path>[/bold cyan] - Add a document to your knowledge base
    [bold cyan]ingest_dir <directory_path>[/bold cyan] - Add all documents from a directory
    [bold cyan]ingest_text[/bold cyan] - Enter text content directly to add to your knowledge base
    
    [bold]Performance & Configuration:[/bold]
    [bold cyan]models[/bold cyan] - Show available model configurations
    [bold cyan]change_model[/bold cyan] - Switch to a different model configuration
    [bold cyan]fast_mode[/bold cyan] - Toggle faster response generation
    [bold cyan]system_info[/bold cyan] - Display system information and current settings
    
    [bold]Memory & Learning:[/bold]
    [bold cyan]clear_memory[/bold cyan] - Clear current conversation history
    [bold cyan]learn[/bold cyan] - Process and learn from past conversations
    [bold cyan]toggle_history[/bold cyan] - Toggle conversation history usage
    
    [bold cyan]exit[/bold cyan] - Exit Ouro
    
    [bold]For questions, simply type your query and press Enter.[/bold]
    """
    console.print(Panel(help_text, title="Help", border_style="blue"))

def show_huggingface_instructions():
    """Show instructions for Hugging Face authentication"""
    instructions = get_huggingface_login_instructions()
    console.print(Panel(Markdown(instructions), title="Hugging Face Authentication", border_style="yellow"))

def check_huggingface_credentials() -> bool:
    """
    Check if Hugging Face credentials are already configured
    Returns True if credentials exist, False otherwise
    """
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        return token is not None and len(token) > 0
    except Exception:
        return False

def setup_huggingface_login() -> bool:
    """Ensure user is logged in to Hugging Face"""
    # First check if credentials already exist
    if check_huggingface_credentials():
        console.print("[bold green]✓[/bold green] Hugging Face authentication already set up.")
        return True
        
    console.print("\n[bold]Setting up Hugging Face authentication[/bold]")
    console.print("Ouro requires access to Hugging Face to download models.")
    
    # Show detailed instructions
    show_huggingface_instructions()
    
    # Check if user wants to log in
    if Confirm.ask("Do you want to log in to Hugging Face now?"):
        console.print("Please enter your Hugging Face token when prompted.")
        
        # Attempt login
        login_success = login_huggingface()
        
        if not login_success:
            console.print("\n[bold red]❌ Hugging Face authentication failed[/bold red]")
            console.print("Please check GUIDE.md for detailed setup instructions.")
            console.print("You can try again by running: huggingface-cli login")
            
            if Confirm.ask("Would you like to continue anyway?"):
                console.print("[yellow]Warning: Some models may not be accessible without authentication.[/yellow]")
                return False
            else:
                console.print("[bold yellow]Exiting Ouro.[/bold yellow]")
                console.print("Please try again after setting up Hugging Face authentication.")
                sys.exit(0)
        
        return login_success
    else:
        console.print("[bold yellow]⚠️ Hugging Face authentication is required to use Ouro.[/bold yellow]")
        console.print("Please check GUIDE.md for detailed setup instructions.")
        
        if Confirm.ask("Would you like to continue anyway?"):
            console.print("[yellow]Warning: Most features will not work without authentication.[/yellow]")
            return False
        else:
            console.print("[bold yellow]Exiting Ouro.[/bold yellow]")
            console.print("Please try again after setting up Hugging Face authentication.")
            sys.exit(0)

def display_model_options():
    """Display available model configurations in a table"""
    model_configs = get_available_model_sizes()
    
    table = Table(title="Available Model Configurations")
    table.add_column("Option", style="cyan")
    table.add_column("Size", style="green")
    table.add_column("LLM Model", style="blue")
    table.add_column("Embedding", style="yellow")
    table.add_column("Description", style="white")
    
    for i, (size, config) in enumerate(model_configs.items(), 1):
        # Format the display name for better readability
        display_name = size
        if size == "M1Optimized":
            display_name = "M1 Optimized"
            
        embedding_size = config["embedding"]
        embedding_model = EMBEDDING_MODELS[embedding_size]
        table.add_row(
            str(i),
            display_name,
            config["llm"],
            f"{embedding_size} ({embedding_model.split('/')[-1]})",
            config["description"]
        )
    
    # Add custom option
    table.add_row(
        str(len(model_configs) + 1),
        "Custom",
        "User specified",
        "Medium (default)",
        "Specify your own Hugging Face model"
    )
    
    console.print(table)

def select_model(preset_model: str = None) -> Tuple[str, Dict[str, Any]]:
    """Let the user select a model configuration
    
    Args:
        preset_model: Optional model size to use, bypassing interactive selection
    
    Returns:
        Tuple of (model_path, model_config)
    """
    model_configs = get_available_model_sizes()
    default_size = "Small"  # Default to small model for better reliability
    
    # If preset model is provided via command line
    if preset_model and preset_model in model_configs:
        size = preset_model
        config = model_configs[size]
        model_path = config["llm"]
        embedding_model = EMBEDDING_MODELS[config["embedding"]]
        
        console.print(f"\nUsing preset model configuration: [bold green]{size}[/bold green]")
        console.print(f"  - LLM: [blue]{model_path}[/blue]")
        console.print(f"  - Embedding: [yellow]{embedding_model}[/yellow]")
        console.print(f"  - Description: {config['description']}")
        
        return model_path, config
    
    # Interactive selection
    console.print("\n[bold]Select a model configuration:[/bold]")
    console.print("[italic]Choose based on your computer's capabilities and your needs:[/italic]")
    
    display_model_options()
    
    default_choice = str(list(model_configs.keys()).index(default_size) + 1)
    
    try:
        # Try to get user selection
        choice = Prompt.ask("\nSelect an option", default=default_choice)
        
        if not choice.isdigit():
            console.print(f"[yellow]Invalid choice, using default: {default_size}[/yellow]")
            choice = default_choice
            
        choice_num = int(choice)
            
    except (EOFError, KeyboardInterrupt):
        # Handle non-interactive environments
        console.print(f"[yellow]Non-interactive environment detected. Using default: {default_size}[/yellow]")
        
        # Default to Small model
        size = default_size
        config = model_configs[size]
        model_path = config["llm"]
        embedding_model = EMBEDDING_MODELS[config["embedding"]]
        
        console.print(f"Selected configuration: [bold green]{size}[/bold green]")
        console.print(f"  - LLM: [blue]{model_path}[/blue]")
        console.print(f"  - Embedding: [yellow]{embedding_model}[/yellow]")
        console.print(f"  - Description: {config['description']}")
        
        return model_path, config
    
    # Process the selection
    try:
        if 1 <= choice_num <= len(model_configs):
            # Get model config from the list
            size = list(model_configs.keys())[choice_num-1]
            config = model_configs[size]
            model_path = config["llm"]
            embedding_model = EMBEDDING_MODELS[config["embedding"]]
            
            console.print(f"Selected configuration: [bold green]{size}[/bold green]")
            console.print(f"  - LLM: [blue]{model_path}[/blue]")
            console.print(f"  - Embedding: [yellow]{embedding_model}[/yellow]")
            console.print(f"  - Description: {config['description']}")
            
            return model_path, config
        elif choice_num == len(model_configs) + 1:
            try:
                # Custom model
                model_path = Prompt.ask("Enter Hugging Face model path (e.g., 'mistralai/Mistral-7B-v0.1')")
                embedding_choice = Prompt.ask(
                    "Choose embedding size (Small, Medium, Large)", 
                    choices=["Small", "Medium", "Large"],
                    default="Medium"
                )
                
                # Create custom config
                custom_config = get_custom_model_template()
                custom_config["llm"] = model_path
                custom_config["embedding"] = embedding_choice
                
                console.print(f"Selected custom model: [bold blue]{model_path}[/bold blue]")
                console.print(f"With [yellow]{embedding_choice}[/yellow] embeddings")
                
                return model_path, custom_config
            except (EOFError, KeyboardInterrupt):
                # If any input fails, fall back to default
                console.print("[yellow]Input interrupted. Using default Small model.[/yellow]")
                size = default_size
                config = model_configs[size]
                return config["llm"], config
        else:
            # Invalid selection, use default
            console.print(f"[yellow]Invalid choice. Using default: {default_size}[/yellow]")
            size = default_size
            config = model_configs[size]
            return config["llm"], config
    except Exception as e:
        # Any other error, fall back to default model
        console.print(f"[red]Error in model selection: {str(e)}[/red]")
        console.print(f"[yellow]Using default model: {default_size}[/yellow]")
        size = default_size
        config = model_configs[size]
        return config["llm"], config

def initialize_system(preset_model: str = None, 
                   fast_mode: bool = None,
                   memory_turns: int = None) -> OuroRAG:
    """
    Initialize the RAG system with progress tracking and performance optimizations
    
    Args:
        preset_model: Optional model size to use, bypassing interactive selection
        fast_mode: Whether to use faster generation settings
        memory_turns: Number of conversation turns to keep in memory
    
    Returns:
        Initialized OuroRAG instance
    """
    # Welcome the user
    welcome_message()
    
    # Setup Hugging Face authentication
    setup_huggingface_login()
    
    # Let the user select a model configuration or use preset
    model_path, model_config = select_model(preset_model)
    
    # Get embedding model from the config
    embedding_model_name = EMBEDDING_MODELS[model_config["embedding"]]
    
    # Set optimization parameters (either from args or config)
    quantize = model_config.get("quantize", DEFAULT_QUANTIZE)
    fast_mode = fast_mode if fast_mode is not None else model_config.get("fast_mode", DEFAULT_FAST_MODE)
    memory_turns = memory_turns if memory_turns is not None else model_config.get("memory_turns", DEFAULT_MEMORY_TURNS)
    save_conversations = model_config.get("save_conversations", DEFAULT_SAVE_CONVERSATIONS)
    
    console.print(f"\n[bold]Initializing Ouro with:[/bold]")
    console.print(f"  • Language model: [blue]{model_path}[/blue]")
    console.print(f"  • Embedding model: [yellow]{embedding_model_name}[/yellow]")
    console.print(f"  • Optimizations: [green]{'Quantization ' if quantize else ''}{'Fast mode ' if fast_mode else ''}[/green]")
    console.print(f"  • Memory: [cyan]{memory_turns} conversation turns[/cyan]")
    
    # Initialize embedding system with progress tracking
    embedding_manager = load_embedding_with_progress(embedding_model_name)
    
    # Initialize LLM with progress tracking and optimizations
    llm, _ = load_model_with_progress(
        model_path, 
        quantize=quantize,
        fast_mode=fast_mode
    )
    
    # Create RAG system with the initialized components
    rag_system = OuroRAG(
        embedding_manager=embedding_manager,
        llm=llm,
        config=model_config,
        memory_turns=memory_turns,
        save_conversations=save_conversations
    )
    
    console.print("[bold green]✓[/bold green] Ouro initialized successfully! You can now ask questions or ingest documents.")
    return rag_system

def handle_ingest(rag_system: OuroRAG, args: List[str]) -> None:
    """Handle ingestion of documents with progress tracking"""
    if len(args) < 1:
        console.print("[red]Error: Please specify a file path.[/red]")
        return
    
    file_path = args[0]
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(f"[green]Ingesting {file_path}", total=100)
            
            def update_progress(message, percent):
                if percent < 0:  # Error
                    progress.update(task, description=f"[bold red]{message}")
                    return
                
                progress.update(
                    task, 
                    description=f"[bold green]{message}",
                    completed=int(percent * 100)
                )
            
            rag_system.ingest_document(file_path, progress_callback=update_progress)
        
        console.print(f"[bold green]✓[/bold green] Successfully ingested document: {file_path}")
    except Exception as e:
        console.print(f"[red]Error ingesting document: {str(e)}[/red]")

def handle_ingest_dir(rag_system: OuroRAG, args: List[str]) -> None:
    """Handle ingestion of a directory of documents with progress tracking"""
    directory = args[0] if len(args) >= 1 else str(DOCUMENTS_DIR)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(f"[green]Ingesting documents from {directory}", total=100)
            
            def update_progress(message, percent):
                if percent < 0:  # Error
                    progress.update(task, description=f"[bold red]{message}")
                    return
                
                progress.update(
                    task, 
                    description=f"[bold green]{message}",
                    completed=int(percent * 100)
                )
            
            rag_system.ingest_directory(directory, progress_callback=update_progress)
        
        console.print(f"[bold green]✓[/bold green] Successfully ingested documents from: {directory}")
    except Exception as e:
        console.print(f"[red]Error ingesting documents from directory: {str(e)}[/red]")

def handle_ingest_text(rag_system: OuroRAG) -> None:
    """Handle ingestion of text content with progress tracking"""
    console.print("[bold]Enter text content to ingest (Ctrl+D or enter an empty line to finish):[/bold]")
    
    content_lines = []
    while True:
        try:
            line = input()
            if not line and content_lines:  # Empty line and we have content
                break
            content_lines.append(line)
        except EOFError:  # Ctrl+D
            break
    
    if not content_lines:
        console.print("[yellow]No content provided.[/yellow]")
        return
    
    content = "\n".join(content_lines)
    filename = Prompt.ask("Enter a filename for this content", default="user_input.txt")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(f"[green]Ingesting text as {filename}", total=100)
            
            def update_progress(message, percent):
                if percent < 0:  # Error
                    progress.update(task, description=f"[bold red]{message}")
                    return
                
                progress.update(
                    task, 
                    description=f"[bold green]{message}",
                    completed=int(percent * 100)
                )
            
            rag_system.save_and_ingest_text(content, filename, progress_callback=update_progress)
        
        console.print(f"[bold green]✓[/bold green] Successfully ingested text as: {filename}")
    except Exception as e:
        console.print(f"[red]Error ingesting text: {str(e)}[/red]")

def handle_change_model(rag_system: OuroRAG) -> None:
    """Handle changing the model configuration"""
    model_path, model_config = select_model()
    
    try:
        embedding_model_name = EMBEDDING_MODELS[model_config["embedding"]]
        
        console.print(f"\n[bold]Changing to new configuration:[/bold]")
        console.print(f"  • Language model: [blue]{model_path}[/blue]")
        console.print(f"  • Embedding model: [yellow]{embedding_model_name}[/yellow]")
        
        # Initialize new embedding system with progress tracking if the embedding model has changed
        if embedding_model_name != rag_system.embedding_manager.model_name:
            embedding_manager = load_embedding_with_progress(embedding_model_name)
            rag_system.embedding_manager = embedding_manager
            
        # Initialize new LLM with progress tracking
        llm, _ = load_model_with_progress(model_path)
        rag_system.llm = llm
        
        # Update configuration
        rag_system.config = model_config
        
        console.print("[bold green]✓[/bold green] Successfully changed model configuration!")
    except Exception as e:
        console.print(f"[red]Error changing model configuration: {str(e)}[/red]")

def handle_models():
    """Display available model configurations"""
    display_model_options()

def handle_system_info(rag_system: OuroRAG):
    """Display system information and current settings"""
    import platform
    import torch
    
    # Get system information
    system_info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "processor": platform.processor(),
    }
    
    # Check if running on Apple Silicon
    is_apple_silicon = False
    if system_info["os"] == "Darwin" and hasattr(torch.backends, "mps"):
        is_apple_silicon = True
        system_info["mps_available"] = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    
    # Get current model configuration
    model_config = rag_system.config
    
    # Create the system info panel
    table = Table(title="System Information")
    table.add_column("Component", style="cyan")
    table.add_column("Value", style="green")
    
    # Add system rows
    table.add_row("Operating System", f"{system_info['os']} {system_info['os_version']}")
    table.add_row("Python Version", system_info['python'])
    table.add_row("PyTorch Version", system_info['torch'])
    if is_apple_silicon:
        table.add_row("Apple Silicon", "Yes")
        table.add_row("MPS Available", str(system_info.get('mps_available', False)))
    
    # Add divider
    table.add_row("", "")
    
    # Add model configuration rows
    table.add_row("Current Model", rag_system.llm.model_name)
    table.add_row("Embedding Model", rag_system.embedding_manager.model_name)
    table.add_row("Fast Mode", "Enabled" if hasattr(rag_system.llm, 'fast_mode') and rag_system.llm.fast_mode else "Disabled")
    table.add_row("Memory Turns", str(rag_system.conversation_memory.max_turns))
    table.add_row("Save Conversations", str(rag_system.conversation_memory.save_conversations))
    
    console.print(table)

def handle_fast_mode(rag_system: OuroRAG):
    """Toggle fast mode for generation"""
    if not hasattr(rag_system.llm, 'fast_mode'):
        rag_system.llm.fast_mode = False
    
    # Toggle the mode
    rag_system.llm.fast_mode = not rag_system.llm.fast_mode
    
    if rag_system.llm.fast_mode:
        console.print("[bold green]Fast mode enabled.[/bold green] Responses will be generated more quickly but may be less detailed.")
    else:
        console.print("[bold green]Fast mode disabled.[/bold green] Responses will be more detailed but may take longer to generate.")

def handle_clear_memory(rag_system: OuroRAG):
    """Clear conversation memory"""
    rag_system.clear_conversation()
    console.print("[bold green]✓[/bold green] Conversation memory cleared.")

def handle_toggle_history(rag_system: OuroRAG):
    """Toggle usage of conversation history"""
    # We'll use a config setting in the RAG system to track this
    if not hasattr(rag_system, 'use_conversation_history'):
        rag_system.use_conversation_history = True
    
    # Toggle the setting
    rag_system.use_conversation_history = not rag_system.use_conversation_history
    
    if rag_system.use_conversation_history:
        console.print("[bold green]Conversation history enabled.[/bold green] Ouro will remember previous interactions.")
    else:
        console.print("[bold green]Conversation history disabled.[/bold green] Each query will be treated independently.")

def handle_learn(rag_system: OuroRAG):
    """Process and learn from past conversations"""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("[green]Learning from past conversations...", total=100)
            
            def update_progress(message, percent):
                if percent < 0:  # Error
                    progress.update(task, description=f"[bold red]{message}")
                    return
                
                progress.update(
                    task, 
                    description=f"[bold green]{message}",
                    completed=int(percent * 100)
                )
            
            rag_system.learn_from_conversations(progress_callback=update_progress)
        
        console.print("[bold green]✓[/bold green] Learning process completed. New insights added to knowledge base.")
    except Exception as e:
        console.print(f"[red]Error during learning process: {str(e)}[/red]")

def handle_query(rag_system: OuroRAG, query: str) -> None:
    """Handle a user query with a progress indicator"""
    try:
        # Record the start time for performance measurement
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            TimeElapsedColumn(),
        ) as progress:
            # First task: Searching knowledge base
            search_task = progress.add_task("[green]Searching knowledge base...", total=100)
            
            # This will be called after search is complete to create the generation task
            generation_task = None
            
            def search_progress_callback(status, progress_percent):
                progress.update(search_task, completed=int(progress_percent * 100))
                
                if progress_percent >= 1.0:
                    progress.update(search_task, description="[bold green]✓ Knowledge retrieved")
                    nonlocal generation_task
                    generation_task = progress.add_task("[green]Generating response...", total=100)
            
            def generation_progress_callback(status, progress_percent):
                if generation_task is not None:
                    progress.update(generation_task, completed=int(progress_percent * 100))
            
            # Execute query with progress tracking, respecting conversation history setting
            use_history = True
            if hasattr(rag_system, 'use_conversation_history'):
                use_history = rag_system.use_conversation_history
                
            response = rag_system.query(
                query, 
                search_progress_callback=search_progress_callback,
                generation_progress_callback=generation_progress_callback,
                use_conversation_history=use_history
            )
        
        # Calculate total response time
        total_time = time.time() - start_time
        
        # Print the response in a panel with timing info
        console.print(Panel(
            response, 
            title="Ouro's Response", 
            subtitle=f"Generated in {total_time:.2f} seconds",
            border_style="green"
        ))
    
    except Exception as e:
        console.print(f"[red]Error processing query: {str(e)}[/red]")
        console.print("[yellow]For better performance, try using the Small model or enabling fast mode with 'fast_mode'[/yellow]")

def ensure_directories_exist():
    """Make sure all necessary directories exist"""
    try:
        # Create directories for storing data
        from src.config import DATA_DIR, DOCUMENTS_DIR, MODEL_CACHE_DIR, VECTOR_STORE_DIR, LOGS_DIR
        
        # Create each directory
        for directory in [DATA_DIR, DOCUMENTS_DIR, MODEL_CACHE_DIR, VECTOR_STORE_DIR, LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
            
        console.print(f"[dim]Data directories created at {DATA_DIR}[/dim]")
    except Exception as e:
        console.print(f"[red]Error creating directories: {str(e)}[/red]")
        console.print("[yellow]Try running with sudo or as administrator if permission issues occur[/yellow]")
        sys.exit(1)

def show_welcome_banner():
    """Show a fancy welcome banner"""
    ascii_logo = """
   ____  _    _ ____   _____
  / __ \\| |  | |  __ \\ / ____|
 | |  | | |  | | |__) | |     
 | |  | | |  | |  _  /| |     
 | |__| | |__| | | \\ \\| |____ 
  \\____/ \\____/|_|  \\_\\\\_____|
                           
    """
    # Display welcome banner
    console.print(Panel(f"[bold green]{ascii_logo}[/bold green]\n[bold]Privacy-First Local RAG System[/bold]", 
                       border_style="green", 
                       subtitle="v0.1.0"))
    console.print("Welcome to Ouro, your private AI assistant for querying documents.")
    console.print("All processing happens locally on your machine - your data stays private.\n")

def main():
    """Main entry point for the CLI"""
    # Display welcome banner
    show_welcome_banner()
    
    # Ensure all directories exist
    ensure_directories_exist()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Ouro - Privacy-First Local RAG System")
    parser.add_argument("--ingest", help="Ingest a document before starting the interactive mode")
    parser.add_argument("--ingest-dir", help="Ingest all documents in a directory before starting")
    
    # Model and performance options
    all_model_choices = list(MODEL_CONFIGURATIONS.keys())
    parser.add_argument("--model", choices=all_model_choices, 
                       help=f"Specify the model size to use: {', '.join(all_model_choices)}")
    parser.add_argument("--fast", action="store_true", help="Enable fast mode for quicker responses")
    parser.add_argument("--no-history", action="store_true", help="Disable conversation history")
    parser.add_argument("--memory-turns", type=int, help="Number of conversation turns to remember")
    
    args = parser.parse_args()
    
    # Initialize the system with parsed options
    try:
        # Convert command line args to initialization parameters
        fast_mode = args.fast if hasattr(args, 'fast') and args.fast is not None else None
        memory_turns = args.memory_turns if hasattr(args, 'memory_turns') and args.memory_turns is not None else None
        
        # Initialize the system
        rag_system = initialize_system(
            preset_model=args.model,
            fast_mode=fast_mode,
            memory_turns=memory_turns
        )
        
        # Apply any additional settings
        if hasattr(args, 'no_history') and args.no_history:
            # Initialize use_conversation_history attribute if not present
            if not hasattr(rag_system, 'use_conversation_history'):
                rag_system.use_conversation_history = True
                
            # Disable conversation history
            rag_system.use_conversation_history = False
            console.print("[bold yellow]Note:[/bold yellow] Conversation history is disabled.")
            
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Setup interrupted. Exiting Ouro.[/bold yellow]")
        console.print("Please run again when you're ready to complete the setup.")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Error during initialization: {str(e)}[/bold red]")
        console.print("Please check GUIDE.md for troubleshooting information.")
        sys.exit(1)
    
    # Handle command line options
    if args.ingest:
        handle_ingest(rag_system, [args.ingest])
    
    if args.ingest_dir:
        handle_ingest_dir(rag_system, [args.ingest_dir])
    
    # Main interaction loop
    print_help()
    console.print("\n[bold green]Ouro is ready![/bold green] Type your questions or commands.")
    console.print("[bold cyan]Quick start:[/bold cyan]")
    console.print("1. [bold]Try asking:[/bold] 'What is RAG?' to test with the sample document")
    console.print("2. [bold]Add documents:[/bold] Type 'ingest <file_path>' to add your own files")
    console.print("3. [bold]Exit:[/bold] Type 'exit' when you're done\n")
    
    while True:
        try:
            command = Prompt.ask("\n[bold blue]User[/bold blue]").strip()
            
            if not command:
                continue
            
            # Handle commands
            if command.lower() == "exit":
                console.print("[bold green]Goodbye! Thank you for using Ouro.[/bold green]")
                break
            
            elif command.lower() == "help":
                print_help()
            
            # Document Management
            elif command.lower().startswith("ingest "):
                parts = command.split(maxsplit=1)
                handle_ingest(rag_system, [parts[1]])
            
            elif command.lower().startswith("ingest_dir "):
                parts = command.split(maxsplit=1)
                handle_ingest_dir(rag_system, [parts[1]])
            
            elif command.lower() == "ingest_text":
                handle_ingest_text(rag_system)
            
            # Performance & Configuration
            elif command.lower() == "models":
                handle_models()
            
            elif command.lower() == "change_model":
                handle_change_model(rag_system)
                
            elif command.lower() == "fast_mode":
                handle_fast_mode(rag_system)
                
            elif command.lower() == "system_info":
                handle_system_info(rag_system)
            
            # Memory & Learning
            elif command.lower() == "clear_memory":
                handle_clear_memory(rag_system)
                
            elif command.lower() == "learn":
                handle_learn(rag_system)
                
            elif command.lower() == "toggle_history":
                handle_toggle_history(rag_system)
            
            else:
                # Treat as a query
                handle_query(rag_system, command)
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation interrupted.[/yellow]")
            
            try:
                # Ask if the user wants to exit
                if Confirm.ask("Do you want to exit Ouro?"):
                    console.print("[bold green]Goodbye! Thank you for using Ouro.[/bold green]")
                    break
            except (EOFError, KeyboardInterrupt):
                # If we can't get input, just exit
                console.print("[bold yellow]Input unavailable. Exiting Ouro.[/bold yellow]")
                break
                
        except EOFError:
            # EOF usually means we're in a non-interactive environment
            console.print("\n[bold yellow]EOF detected. Exiting Ouro.[/bold yellow]")
            break
            
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            console.print("[yellow]Type 'help' to see available commands or 'exit' to quit.[/yellow]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold green]Ouro shutting down. Goodbye![/bold green]")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        console.print(f"[bold red]Fatal error: {str(e)}[/bold red]")
        sys.exit(1)