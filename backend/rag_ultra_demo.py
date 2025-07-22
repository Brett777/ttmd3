#!/usr/bin/env python
"""
RAG-Ultra Demo CLI

A command-line interface for the RAG-Ultra SDK. This tool allows users to process documents (PDF, DOCX, PPTX, TXT), generate hierarchical metadata and summaries, and interactively query the processed data using advanced LLM-powered retrieval. Features include:
- Batch document processing with progress feedback
- Per-page image and metadata extraction
- Interactive Q&A with multi-document support
- Flexible model and API key selection for different LLM providers
- Robust error handling and user guidance
"""

import os
import json
import glob
import logging
import sys
from typing import List, Optional, Dict, Any
import typer
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rprint

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # If dotenv is not installed, skip loading .env

# Only log to file, not to console
logging.basicConfig(
    filename="rag_ultra_demo.log",
    encoding="utf-8",
    level=logging.DEBUG,
    filemode="w"
)
logger = logging.getLogger("rag_ultra_demo")

# Suppress httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# Suppress LiteLLM and related logs in the terminal
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("litellm.proxy").setLevel(logging.WARNING)
logging.getLogger("litellm_logging").setLevel(logging.WARNING)

# Import SDK modules
from rag_ultra import (
    convert_document_to_text,
    convert_document_pages_to_images,
    generate_batch_metadata,
    generate_document_summary,
    process_document
)
from rag_ultra.utils import (
    get_file_details,
    save_metadata_to_file,
    load_metadata_from_file,
    delete_document_from_metadata,
    is_multi_document_format,
    get_document_names,
    list_documents_in_metadata_store,
    get_document_level_metadata,
    get_page_level_metadata_range,
    get_page_metadata_field,
    get_metadata_field_for_documents
)
from rag_ultra.document_summary import (
    extract_document_details,
    assemble_document_metadata
)

# Import the new smarter retriever agent
from rag_ultra.retriever_agent import retrieve_agent

# Initialize Typer app and Rich console
app = typer.Typer(help="Hierarchical Summarization Demo - A commandline interface for the Hierarchical Summarization SDK")
console = Console()

# Suppress rag_ultra logging
logging.getLogger("rag_ultra").setLevel(logging.WARNING)

# Provider and model options
PROVIDER_MODELS = {
    "OpenAI": [
        "openai/gpt-4.1",
        "openai/gpt-4.1-mini",
        "openai/gpt-4.1-nano",
        "openai/gpt-4o",
        "openai/gpt-4o-search-preview",
        "openai/gpt-4.5-preview",
        "openai/gpt-4o-mini",
        "openai/gpt-4o-mini-search-preview",
        "openai/o1-pro",
        "openai/o1",
        "openai/o1-mini",
        "openai/o3-mini",
        "openai/o1-preview",
        "openai/chatgpt-4o-latest",
        "openai/gpt-4-turbo",
        "openai/gpt-3.5-turbo"
    ],
    "xAI": [
        "xai/grok-3-beta",
        "xai/grok-3-fast-beta",
        "xai/grok-3-fast-latest",
        "xai/grok-3-mini-beta",
        "xai/grok-3-mini-fast-beta",
        "xai/grok-3-mini-fast-latest",
        "xai/grok-beta",
        "xai/grok-2-vision-latest",
        "xai/grok-2-vision",
        "xai/grok-2",
        "xai/grok-2-latest",
        "xai/grok-vision-beta"
    ],
    "Anthropic": [
        "anthropic/claude-3-7-sonnet-latest",
        "anthropic/claude-3-7-sonnet-20250219",
        "anthropic/claude-3-haiku-20240307",
        "anthropic/claude-3-5-haiku-20241022",
        "anthropic/claude-3-5-haiku-latest",
        "anthropic/claude-3-opus-latest",
        "anthropic/claude-3-opus-20240229",
        "anthropic/claude-3-sonnet-20240229",
        "anthropic/claude-3-5-sonnet-latest",
        "anthropic/claude-3-5-sonnet-20240620",
        "anthropic/claude-3-5-sonnet-20241022",
        "anthropic/claude-instant-1",
        "anthropic/claude-instant-1.2",
        "anthropic/claude-2",
        "anthropic/claude-2.1"
    ],
    "Deepseek": [
        "deepseek/deepseek-reasoner",
        "deepseek/deepseek-chat",
        "deepseek/deepseek-coder"
    ],
    "Cohere": [
        "cohere/command",
        "cohere/command-a-03-2025",
        "cohere/command-r7b-12-2024",
        "cohere/command-r-plus",
        "cohere/command-r",
        "cohere/command-nightly",
        "cohere/command-light-nightly"
    ],
    "Gemini": [
        "gemini/gemini-2.0-pro-exp-02-05",
        "gemini/gemini-2.0-flash",
        "gemini/gemini-2.0-flash-lite",
        "gemini/gemini-2.0-flash-001",
        "gemini/gemini-2.5-pro-preview-03-25",
        "gemini/gemini-2.5-pro-exp-03-25",
        "gemini/gemini-2.0-flash-exp",
        "gemini/gemini-2.0-flash-lite-preview-02-05",
        "gemini/gemini-2.0-flash-thinking-exp",
        "gemini/gemini-2.0-flash-thinking-exp-01-21",
        "gemini/gemma-3-27b-it",
        "gemini/learnlm-1.5-pro-experimental",
        "gemini/gemini-1.5-flash-002",
        "gemini/gemini-1.5-flash-001",
        "gemini/gemini-1.5-flash",
        "gemini/gemini-1.5-flash-latest",
        "gemini/gemini-1.5-flash-8b",
        "gemini/gemini-1.5-flash-8b-exp-0924",
        "gemini/gemini-exp-1114",
        "gemini/gemini-exp-1206",
        "gemini/gemini-1.5-flash-exp-0827",
        "gemini/gemini-1.5-flash-8b-exp-0827",
        "gemini/gemini-pro",
        "gemini/gemini-1.5-pro",
        "gemini/gemini-1.5-pro-002",
        "gemini/gemini-1.5-pro-001",
        "gemini/gemini-1.5-pro-exp-0801",
        "gemini/gemini-1.5-pro-exp-0827",
        "gemini/gemini-1.5-pro-latest",
        "gemini/gemini-pro-vision",
        "gemini/gemini-gemma-2-27b-it",
        "gemini/gemini-gemma-2-9b-it"
    ]
}

PROVIDER_ENV_VARS = {
    "OpenAI": "OPENAI_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "xAI": "XAI_API_KEY",
    "Deepseek": "DEEPSEEK_API_KEY",
    "Cohere": "COHERE_API_KEY",
    "Gemini": "GEMINI_API_KEY"
}

def mask_key(key: str) -> str:
    """
    Mask an API key for display, showing only the first and last 4 characters.
    Args:
        key: The API key string.
    Returns:
        Masked key string for display.
    """
    if not key or len(key) < 8:
        return "*" * 8
    return key[:4] + "*" * (len(key) - 8) + key[-4:]

def select_model_and_key() -> (str, str):
    """
    Prompt the user to select a provider, model, and API key for LLM operations.
    Returns:
        Tuple of (model_name, api_key).
    """
    console.print("\n[bold]Select a provider:[/bold]")
    providers = list(PROVIDER_MODELS.keys())
    for idx, provider in enumerate(providers, 1):
        console.print(f"[{idx}] {provider}")
    provider_choice = Prompt.ask("Choose a provider")
    provider = providers[int(provider_choice)-1]
    
    # Model selection
    models = PROVIDER_MODELS[provider]
    console.print(f"\n[bold]Select a model from {provider}:[/bold]")
    for idx, model in enumerate(models, 1):
        console.print(f"[{idx}] {model}")
    model_choice = Prompt.ask("Choose a model")
    model = models[int(model_choice)-1]
    
    # API key selection
    env_var = PROVIDER_ENV_VARS.get(provider)
    env_key = os.environ.get(env_var) if env_var else None
    key_to_use = None
    if env_key:
        masked = mask_key(env_key)
        use_env = Confirm.ask(f"Found API key for {provider} in environment: [bold]{masked}[/bold]. Use this key?", default=True)
        if use_env:
            key_to_use = env_key
    if not key_to_use:
        key_to_use = Prompt.ask(f"Enter API key for {provider}")
    return model, key_to_use

def display_welcome():
    """
    Display a welcome message for the CLI.
    """
    console.print(Panel.fit(
        "[bold blue]Hierarchical Summarization Demo[/bold blue]\n"
        "[italic]Commandline interface for the Hierarchical Summarization SDK[/italic]",
        border_style="blue"
    ))

def process_document_wrapper(
    document_path: str,
    output_path: Optional[str] = None,
    context_length: int = 3,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a single document: extract text, images, metadata, and summaries, then assemble and optionally save the result.
    Args:
        document_path: Path to the document.
        output_path: Path to save the metadata JSON (optional).
        context_length: Number of previous pages to use as context for metadata generation.
        model: LLM model name (LiteLLM format).
        api_key: API key for the LLM provider.
        api_base: API base URL for the LLM provider.
    Returns:
        The document metadata dictionary.
    Raises:
        FileNotFoundError: If the document does not exist.
        Exception: For errors in processing steps.
    """
    logger.info(f"Processing document: {document_path}")

    # Step 1: Extract text from document
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
        refresh_per_second=10
    ) as progress:
        extract_task = progress.add_task("Extracting text from document...", total=None)
        page_texts = convert_document_to_text(document_path, max_workers=8)
        progress.update(extract_task, completed=True)

    num_pages = len(page_texts)
    console.print(f"[cyan]Found {num_pages} page(s) in the document.[/cyan]")

    # Step 2: Convert document pages to images (with per-page progress)
    page_images = {}
    with Progress(
        "[progress.percentage]{task.percentage:>3.0f}% ",
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
        refresh_per_second=10
    ) as progress:
        image_task = progress.add_task(f"Converting pages to images... (0/{num_pages})", total=num_pages)
        try:
            # Use parallel processing with 4 worker threads and lower DPI for faster image conversion
            def image_callback(_):
                progress.advance(image_task)
                completed = progress.tasks[image_task].completed
                progress.update(image_task, description=f"Converting pages to images... ({int(completed)}/{num_pages})")
            # If the SDK supports a callback, use it; otherwise, fallback to batch
            if hasattr(convert_document_pages_to_images, 'with_callback'):
                page_images = convert_document_pages_to_images(
                    document_path=document_path,
                    dpi=150,
                    max_workers=4,
                    callback=image_callback
                )
            else:
                # Fallback: process all, then update once
                page_images = convert_document_pages_to_images(
                    document_path=document_path,
                    dpi=150,
                    max_workers=4
                )
                progress.update(image_task, completed=len(page_images), description=f"Converting pages to images... ({len(page_images)}/{num_pages})")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not convert pages to images: {e}[/yellow]")
            page_images = {}
            progress.update(image_task, completed=0, description=f"Converting pages to images... (0/{num_pages})")

    # Step 3: Generate metadata for each page (with per-page progress)
    page_metadata = []
    with Progress(
        "[progress.percentage]{task.percentage:>3.0f}% ",
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
        refresh_per_second=10
    ) as progress:
        meta_task = progress.add_task(f"Generating page metadata... (0/{num_pages})", total=num_pages)
        def meta_callback(_):
            progress.advance(meta_task)
            completed = progress.tasks[meta_task].completed
            progress.update(meta_task, description=f"Generating page metadata... ({int(completed)}/{num_pages})")
        page_metadata = generate_batch_metadata(
            page_texts=page_texts,
            context_length=context_length,
            page_images=page_images,
            model=model,
            api_key=api_key,
            api_base=api_base,
            callback=meta_callback
        )

    logger.info(f"Generated metadata for {len(page_metadata)} pages")

    # Step 4: Get document details
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
        refresh_per_second=10
    ) as progress:
        details_task = progress.add_task("Extracting document details...", total=None)
        file_details = get_file_details(document_path)
        document_details = extract_document_details(page_metadata, file_details)
        progress.update(details_task, completed=True)
    logger.info(f"Extracted document details: {document_details}")

    # Step 5: Generate document summary
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
        refresh_per_second=10
    ) as progress:
        summary_task = progress.add_task("Generating document summary...", total=None)
        document_summary = generate_document_summary(
            page_metadata,
            model=model,
            api_key=api_key,
            api_base=api_base
        )
        progress.update(summary_task, completed=True)
    logger.info(f"Generated document summary: {document_summary}")

    # Step 6: Assemble complete metadata
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
        refresh_per_second=10
    ) as progress:
        assemble_task = progress.add_task("Assembling final metadata...", total=None)
        metadata = assemble_document_metadata(
            page_metadata=page_metadata,
            document_summary=document_summary,
            document_details=document_details,
            page_images=page_images
        )
        progress.update(assemble_task, completed=True)
    logger.info(f"Assembled final metadata for document: {document_path}")

    # Step 7: Save metadata if output path is provided
    if output_path:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
            refresh_per_second=10
        ) as progress:
            save_task = progress.add_task(f"Saving metadata to {output_path}...", total=None)
            save_metadata_to_file(metadata, output_path)
            logger.info(f"Saved metadata to {output_path}")
            progress.update(save_task, completed=True)

    return metadata

def process_documents_folder(
    folder_path: str,
    output_path: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process all documents in a folder and combine their metadata into a single dictionary.
    Args:
        folder_path: Path to the folder containing documents.
        output_path: Path to save the combined metadata JSON (optional).
        model: LLM model name (LiteLLM format).
        api_key: API key for the LLM provider.
        api_base: API base URL for the LLM provider.
    Returns:
        Combined metadata dictionary for all processed documents.
    Raises:
        typer.Exit: If no documents are found in the folder.
    """
    # Find all document files in the folder
    document_files = []
    for ext in ["*.pdf", "*.docx", "*.doc", "*.txt"]:
        document_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not document_files:
        console.print("[red]No document files found in the specified folder.[/red]")
        raise typer.Exit(1)
    
    console.print(f"[green]Found {len(document_files)} document(s) to process.[/green]")
    
    # Process each document
    all_metadata = {}
    for i, doc_path in enumerate(document_files, 1):
        console.print(f"\n[bold]Processing document {i}/{len(document_files)}: {os.path.basename(doc_path)}[/bold]")
        doc_metadata = process_document_wrapper(doc_path, output_path=output_path, model=model, api_key=api_key, api_base=api_base)
        
        # Add to combined metadata
        doc_name = os.path.basename(doc_path)
        all_metadata[doc_name] = doc_metadata
    
    # Save combined metadata if output path is provided
    if output_path:
        save_metadata_to_file(all_metadata, output_path)
        console.print(f"[green]Combined metadata saved to {output_path}[/green]")
    
    return all_metadata

def interactive_qa(
    metadata_dict: Dict[str, Any],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None
):
    """
    Start an interactive question-answering session using the processed metadata and selected LLM.
    Tracks conversation history and provides rich output with citations and reasoning.
    Args:
        metadata_dict: Document metadata dictionary.
        model: LLM model name (LiteLLM format).
        api_key: API key for the LLM provider.
        api_base: API base URL for the LLM provider.
    Returns:
        None
    """
    logger.info("Starting interactive Q&A session")

    # If we have multiple documents, let the user know
    multi_doc = is_multi_document_format(metadata_dict)
    if multi_doc:
        doc_names = get_document_names(metadata_dict)
        doc_titles = {doc_name: metadata_dict[doc_name].get("document_details", {}).get("title", "Untitled") for doc_name in doc_names}
        console.print(f"\n[bold green]Found {len(doc_names)} documents in the metadata:[/bold green]")
        for i, doc_name in enumerate(doc_names, 1):
            doc_title = doc_titles[doc_name]
            console.print(f"  {i}. {doc_name} - {doc_title}")
        console.print("\nYou can query all documents together, or specify a document by name in your question.")

    console.print("\n[bold green]Interactive Question-Answering Session[/bold green]")
    console.print("Type 'exit' or 'quit' to end the session.\n")

    # Conversation history: OpenAI format
    messages = []

    while True:
        question = Prompt.ask("[bold blue]Your question[/bold blue]")
        logger.info(f"User question: {question}")

        if question.lower() in ["exit", "quit"]:
            logger.info("User exited interactive Q&A session")
            break

        # Add user message to conversation history
        messages.append({"role": "user", "content": question})

        # Always pass the full metadata_dict to the agent
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,  # This makes the progress display disappear after completion
            refresh_per_second=10  # Smooth spinner animation
        ) as progress:
            task = progress.add_task("Processing...", total=None)
            retrieval_result = retrieve_agent(
                prompt=question,
                metadata_dict=metadata_dict,
                model=model,
                api_key=api_key,
                api_base=api_base,
                messages=messages  # Pass conversation history
            )
            progress.update(task, completed=True)

        # Get the answer and metadata
        answer = retrieval_result.get("answer", "No answer found")
        logger.info(f"Retrieval result: {json.dumps(retrieval_result, default=str)[:1000]}")  # Truncate for log

        # Add assistant answer to conversation history
        messages.append({"role": "assistant", "content": answer})

        # Display the answer in a nice panel
        console.print("\n[bold]Answer:[/bold]")
        console.print(Panel(answer, border_style="green"))
        
        # Display citations/source information
        relevant_pages = retrieval_result.get("relevant_pages", [])
        raw_content = retrieval_result.get("raw_content")
        if relevant_pages:
            # Build set of all referenced documents
            referenced_docs = set()
            for item in relevant_pages:
                referenced_docs.add(str(item.get("document", "?")))
            if not referenced_docs and retrieval_result.get("used_metadata"):
                referenced_docs.update(retrieval_result["used_metadata"].keys())

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Document")
            table.add_column("Page(s) Used")
            table.add_column("Page image(s) Used")
            table.add_column("Raw Text Used")
            table.add_column("Metadata Referenced")
            used_metadata_fields = retrieval_result.get("used_metadata_fields", [])

            image_content = retrieval_result.get("image_content")
            if referenced_docs:
                # Group pages by document
                doc_to_pages = {}
                for item in relevant_pages:
                    doc = str(item.get("document", "?"))
                    page = str(item.get("page", "?"))
                    doc_to_pages.setdefault(doc, []).append(page)
                # Group image pages by document
                doc_to_image_pages = {}
                if image_content:
                    for doc, pages_dict in image_content.items():
                        for page in pages_dict:
                            doc_to_image_pages.setdefault(str(doc), []).append(str(page))
                for doc in referenced_docs:
                    pages = doc_to_pages.get(doc, [])
                    pages_str = ", ".join(pages) if pages else "-"
                    image_pages = doc_to_image_pages.get(doc, [])
                    image_pages_str = ", ".join(image_pages) if image_pages else "-"
                    raw_text_used = False
                    if raw_content and doc in raw_content and pages:
                        for page in pages:
                            page_data = raw_content[doc].get(page) or raw_content[doc].get(int(page)) or raw_content[doc].get(str(page))
                            if page_data and page_data.get("raw_text"):
                                raw_text_used = True
                                break
                    metadata_fields_str = ", ".join(used_metadata_fields) if used_metadata_fields else "-"
                    table.add_row(doc, pages_str, image_pages_str, "Yes" if raw_text_used else "No", metadata_fields_str)
            else:
                table.add_row("No documents or pages cited", "-", "-", "-", "-")
            console.print(table)
        
        # Show both reasoning traces if available
        reasoning_trace = retrieval_result.get("reasoning_trace")
        if reasoning_trace:
            reasoning = reasoning_trace.get("reasoning") if isinstance(reasoning_trace, dict) else str(reasoning_trace)
            console.print("\n[bold]Agentic Reasoning Trace:[/bold]")
            console.print(Panel(reasoning, border_style="yellow"))
        # Try to show final answer reasoning if present in raw_content
        if raw_content and isinstance(raw_content, dict):
            try:
                answer_json = json.loads(answer)
                if isinstance(answer_json, dict) and "reasoning" in answer_json:
                    console.print("\n[bold]Final Answer Reasoning Trace:[/bold]")
                    console.print(Panel(answer_json["reasoning"], border_style="yellow"))
            except Exception:
                pass
        
        console.print()  # Empty line for spacing

def explore_metadata_menu():
    """
    Interactive menu for exploring metadata using the new utility functions.
    """
    while True:
        console.print("\n[bold]Explore Metadata[/bold]")
        console.print("[1] Use default metadata file (document_metadata.json)")
        console.print("[2] Specify a metadata file to explore")
        console.print("[0] Go back")
        choice = Prompt.ask("Choose an option", choices=["0", "1", "2"], default="1")
        if choice == "0":
            return
        if choice == "1":
            metadata_path = "document_metadata.json"
        else:
            metadata_path = Prompt.ask("Enter the path to your metadata file")
        try:
            metadata = load_metadata_from_file(metadata_path)
        except Exception as e:
            console.print(f"[red]Error loading metadata: {e}[/red]")
            continue
        # Sub-menu for metadata exploration
        while True:
            console.print("\n[bold]Metadata Exploration Options[/bold]")
            console.print("[1] List documents in metadata store")
            console.print("[2] Get document-level metadata")
            console.print("[3] Get page-level metadata for a page range")
            console.print("[4] Get a metadata field for a document and page")
            console.print("[5] Get a field (or fields) for all/some documents")
            console.print("[0] Go back")
            sub_choice = Prompt.ask("Choose an option", choices=["0", "1", "2", "3", "4", "5"], default="1")
            if sub_choice == "0":
                break
            elif sub_choice == "1":
                docs = list_documents_in_metadata_store(metadata)
                console.print("\n[bold]Documents in metadata store:[/bold]")
                for doc in docs:
                    console.print(f"- {doc}")
            elif sub_choice == "2":
                doc_names = list_documents_in_metadata_store(metadata)
                if len(doc_names) == 1 and doc_names[0] == 'default':
                    doc_name = None
                else:
                    console.print("\n[bold]Available documents:[/bold]")
                    for i, doc in enumerate(doc_names, 1):
                        console.print(f"[{i}] {doc}")
                    idx = Prompt.ask("Select a document", choices=[str(i) for i in range(1, len(doc_names)+1)])
                    doc_name = doc_names[int(idx)-1]
                result = get_document_level_metadata(metadata, doc_name)
                console.print(Panel(json.dumps(result, indent=2, ensure_ascii=False), title="Document-level Metadata"))
            elif sub_choice == "3":
                doc_names = list_documents_in_metadata_store(metadata)
                if len(doc_names) == 1 and doc_names[0] == 'default':
                    doc_name = None
                else:
                    console.print("\n[bold]Available documents:[/bold]")
                    for i, doc in enumerate(doc_names, 1):
                        console.print(f"[{i}] {doc}")
                    idx = Prompt.ask("Select a document", choices=[str(i) for i in range(1, len(doc_names)+1)])
                    doc_name = doc_names[int(idx)-1]
                start_page = int(Prompt.ask("Enter start page (1-based)", default="1"))
                end_page = int(Prompt.ask("Enter end page (inclusive)", default=str(start_page)))
                result = get_page_level_metadata_range(metadata, doc_name, start_page, end_page)
                console.print(Panel(json.dumps(result, indent=2, ensure_ascii=False), title=f"Page-level Metadata: {doc_name or 'default'} pages {start_page}-{end_page}"))
            elif sub_choice == "4":
                doc_names = list_documents_in_metadata_store(metadata)
                if len(doc_names) == 1 and doc_names[0] == 'default':
                    doc_name = None
                else:
                    console.print("\n[bold]Available documents:[/bold]")
                    for i, doc in enumerate(doc_names, 1):
                        console.print(f"[{i}] {doc}")
                    idx = Prompt.ask("Select a document", choices=[str(i) for i in range(1, len(doc_names)+1)])
                    doc_name = doc_names[int(idx)-1]
                page = int(Prompt.ask("Enter page number (1-based)", default="1"))
                field = Prompt.ask("Enter metadata field name (e.g., 'summary', 'keywords', etc.)")
                result = get_page_metadata_field(metadata, doc_name, page, field)
                console.print(Panel(json.dumps(result, indent=2, ensure_ascii=False), title=f"Field '{field}' for {doc_name or 'default'} page {page}"))
            elif sub_choice == "5":
                doc_names = list_documents_in_metadata_store(metadata)
                use_all = Confirm.ask("Get field(s) for all documents?", default=True)
                if use_all:
                    selected_docs = None
                else:
                    console.print("\n[bold]Available documents:[/bold]")
                    for i, doc in enumerate(doc_names, 1):
                        console.print(f"[{i}] {doc}")
                    idxs = Prompt.ask("Enter comma-separated document numbers", default="1")
                    selected_docs = [doc_names[int(i.strip())-1] for i in idxs.split(",") if i.strip().isdigit()]
                field = Prompt.ask("Enter metadata field name or comma-separated list of fields")
                if "," in field:
                    field_list = [f.strip() for f in field.split(",")]
                    result = get_metadata_field_for_documents(metadata, field_list, selected_docs)
                else:
                    result = get_metadata_field_for_documents(metadata, field, selected_docs)
                console.print(Panel(json.dumps(result, indent=2, ensure_ascii=False), title=f"Field(s) for documents"))
        # End sub-menu

@app.command()
def main(
    metadata_file: Optional[str] = typer.Option(
        None, 
        "--metadata", "-m", 
        help="Path to existing metadata file"
    ),
    documents_folder: Optional[str] = typer.Option(
        None, 
        "--folder", "-f", 
        help="Path to folder containing documents to process"
    ),
    output_file: Optional[str] = typer.Option(
        "document_metadata.json", 
        "--output", "-o", 
        help="Path to save metadata file"
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="LLM model to use (LiteLLM format, e.g., 'openai/gpt-4o', 'anthropic/claude-3-sonnet')"
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="API key for the LLM provider (overrides environment variable)"
    ),
    api_base: Optional[str] = typer.Option(
        None,
        "--api-base",
        help="API base URL for the LLM provider (if required)"
    )
):
    """
    Main entry point for the RAG-Ultra Demo CLI.
    Handles argument parsing, workflow selection, and launches the appropriate processing or interactive session.
    Args:
        metadata_file: Path to existing metadata file.
        documents_folder: Path to folder containing documents to process.
        output_file: Path to save metadata file.
        model: LLM model to use (LiteLLM format).
        api_key: API key for the LLM provider.
        api_base: API base URL for the LLM provider.
    Returns:
        None
    """
    logger.info("Starting RAG-Ultra Demo CLI")
    display_welcome()
    
    # Set API key if provided
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key  # For OpenAI compatibility
    
    # If no API key is set, prompt for it
    if not (api_key or os.environ.get("OPENAI_API_KEY")):
        api_key_input = Prompt.ask("[bold red]API key not found. Please enter your API key[/bold red]")
        os.environ["OPENAI_API_KEY"] = api_key_input
        api_key = api_key_input
    
    # Determine workflow based on provided arguments
    if metadata_file:
        # Use existing metadata file
        logger.info(f"Loading metadata from {metadata_file}...")
        try:
            metadata_dict = load_metadata_from_file(metadata_file)
            logger.info("Metadata loaded successfully!")
        except Exception as e:
            console.print(f"[red]Error loading metadata: {e}[/red]")
            logger.error(f"Error loading metadata: {e}")
            raise typer.Exit(1)
    
    elif documents_folder:
        # Process documents from folder
        # Prompt for model/key if not provided
        if not model or not api_key:
            model, api_key = select_model_and_key()
        logger.info(f"Processing documents from {documents_folder}...")
        try:
            metadata_dict = process_documents_folder(documents_folder, output_file, model=model, api_key=api_key, api_base=api_base)
            logger.info("Documents processed successfully!")
        except Exception as e:
            console.print(f"[red]Error processing documents: {e}[/red]")
            logger.error(f"Error processing documents: {e}")
            raise typer.Exit(1)
        # Prompt for retrieval/QA model/key after processing
        console.print("\n[bold]Now select a model for retrieval / question answering.[/bold]")
        qa_model, qa_api_key = select_model_and_key()
        # Start interactive Q&A session
        interactive_qa(metadata_dict, model=qa_model, api_key=qa_api_key, api_base=api_base)
        return
    
    else:
        # Interactive mode - ask user what to do
        default_metadata = "document_metadata.json"
        has_default_metadata = os.path.exists(default_metadata)

        while True:
            console.print("\n[bold]What would you like to do?[/bold]")
            menu_options = []
            if has_default_metadata:
                console.print("[0] Exit")
                menu_options.append("0")
                console.print("[1] Use default metadata file (document_metadata.json)")
                menu_options.append("1")
                console.print("[2] Use an existing metadata file (if you've previously processed documents)")
                menu_options.append("2")
                console.print("[3] Process new documents from a folder")
                menu_options.append("3")
                console.print("[4] Delete a document from the metadata store")
                menu_options.append("4")
                console.print("[5] Add a document to the metadata store")
                menu_options.append("5")
                if has_default_metadata:
                    console.print("[6] Explore metadata")
                    menu_options.append("6")
            else:
                console.print("[0] Exit")
                menu_options.append("0")
                console.print("[1] Use an existing metadata file (if you've previously processed documents)")
                menu_options.append("1")
                console.print("[2] Process new documents from a folder")
                menu_options.append("2")
                console.print("[3] Delete a document from the metadata store")
                menu_options.append("3")
                console.print("[4] Add a document to the metadata store")
                menu_options.append("4")
                console.print("[5] Explore metadata")
                menu_options.append("5")

            choice = Prompt.ask(
                "Choose an option",
                default="1",
                choices=menu_options
            )

            if choice == "0":
                console.print("[bold green]Exiting. Goodbye![/bold green]")
                raise typer.Exit(0)

            # Handle delete option
            if (has_default_metadata and choice == "4") or (not has_default_metadata and choice == "3"):
                # Ask for metadata file if not default
                if has_default_metadata:
                    metadata_path = default_metadata
                else:
                    metadata_path = Prompt.ask("Enter the path to your metadata file")
                try:
                    metadata_dict = load_metadata_from_file(metadata_path)
                except Exception as e:
                    console.print(f"[red]Error loading metadata: {e}[/red]")
                    continue
                if is_multi_document_format(metadata_dict):
                    doc_names = get_document_names(metadata_dict)
                    console.print("\n[bold]Documents in metadata:[/bold]")
                    for i, doc_name in enumerate(doc_names, 1):
                        console.print(f"  {i}. {doc_name}")
                    doc_num = Prompt.ask("Enter the number of the document to delete", choices=[str(i) for i in range(1, len(doc_names)+1)])
                    doc_to_delete = doc_names[int(doc_num)-1]
                else:
                    doc_to_delete = Prompt.ask("Enter the document name to delete (or 'default' to delete the only document)")
                updated_metadata, deleted = delete_document_from_metadata(metadata_dict, doc_to_delete)
                if deleted:
                    save_metadata_to_file(updated_metadata, metadata_path)
                    console.print(f"[green]Document '{doc_to_delete}' deleted from metadata and saved to {metadata_path}.[/green]")
                else:
                    console.print(f"[yellow]Document '{doc_to_delete}' not found in metadata. No changes made.[/yellow]")
                continue
            elif has_default_metadata and choice == "1":
                try:
                    metadata_dict = load_metadata_from_file(default_metadata)
                    logger.info("Default metadata loaded successfully!")
                except Exception as e:
                    console.print(f"[red]Error loading default metadata: {e}[/red]")
                    logger.error(f"Error loading default metadata: {e}")
                    continue
            elif (has_default_metadata and choice == "2") or (not has_default_metadata and choice == "1"):
                # Use existing metadata file
                metadata_file = Prompt.ask("Enter the path to your metadata file")
                try:
                    metadata_dict = load_metadata_from_file(metadata_file)
                    logger.info("Metadata loaded successfully!")
                except Exception as e:
                    console.print(f"[red]Error loading metadata: {e}[/red]")
                    logger.error(f"Error loading metadata: {e}")
                    continue
            elif (has_default_metadata and choice == "6") or (not has_default_metadata and choice == "5"):
                explore_metadata_menu()
                continue
            else:
                # Process documents from folder
                documents_folder = Prompt.ask("Enter the path to your documents folder")
                # Prompt for model/key for metadata generation
                model, api_key = select_model_and_key()
                try:
                    metadata_dict = process_documents_folder(documents_folder, output_file, model=model, api_key=api_key, api_base=api_base)
                    logger.info("Documents processed successfully!")
                except Exception as e:
                    console.print(f"[red]Error processing documents: {e}[/red]")
                    logger.error(f"Error processing documents: {e}")
                    continue

            # Prompt for retrieval/QA model/key after processing
            console.print("\n[bold]Now select a model for retrieval / question answering.[/bold]")
            qa_model, qa_api_key = select_model_and_key()
            # Start interactive Q&A session
            interactive_qa(metadata_dict, model=qa_model, api_key=qa_api_key, api_base=api_base)
            return
    
    # Start interactive Q&A session (if not already started above)
    interactive_qa(metadata_dict, model=model, api_key=api_key, api_base=api_base)

if __name__ == "__main__":
    app() 