"""
RAG-Ultra: Document Metadata Extraction and Retrieval SDK.

This SDK transforms various document types into hierarchical Python dictionaries with detailed metadata.
"""

__version__ = "0.1.0"

from .document_loader import convert_document_to_text
from .image_converter import convert_document_pages_to_images
from .metadata_generator import generate_page_metadata, generate_batch_metadata, generate_metadata_for_large_document
from .document_summary import generate_document_summary, extract_document_details, assemble_document_metadata
from .utils import get_file_details, save_metadata_to_file
from typing import Dict, Any, Optional, Callable

def process_document(
    document_path: str,
    output_path: Optional[str] = None,
    context_length: int = 3,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    callbacks: Optional[Dict[str, Callable]] = None,
    filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a document and extract metadata in one step.
    
    This is a SYNCHRONOUS, BLOCKING function. It should be run in a separate
    thread when used in an async environment like FastAPI.
    
    Args:
        document_path: Path to the document to process
        output_path: Optional path to save the metadata JSON
        context_length: Number of previous pages to use as context (default: 3)
        model: Optional model identifier (e.g., "openai/gpt-4o")
        api_key: Optional API key for the model provider
        api_base: Optional API base URL for the model provider
        callbacks: Dictionary of callbacks for progress updates.
                   Expected keys: 'progress'
        filename: Optional original filename to override the one from document_path
        
    Returns:
        Dictionary containing the complete document metadata
    """
    progress_callback = callbacks.get('progress') if callbacks else None

    def _update_progress(stage, current, total):
        if progress_callback:
            progress_callback(stage, current, total, f"{stage} ({current}/{total})")

    # Step 1: Extract text from document
    _update_progress("Extracting text from document...", 0, 100)
    page_texts = convert_document_to_text(document_path)
    _update_progress("Extracting text from document...", 100, 100)
    
    # Step 2: Convert document pages to images
    num_pages = len(page_texts)
    _update_progress("Converting pages to images...", 0, num_pages)
    try:
        def image_conv_callback(page_num):
            _update_progress("Converting pages to images...", page_num + 1, num_pages)
        page_images = convert_document_pages_to_images(
            document_path=document_path,
            callback=image_conv_callback
        )
    except Exception as e:
        print(f"Warning: Could not convert pages to images: {e}")
        page_images = {}
    _update_progress("Converting pages to images...", num_pages, num_pages)
    
    # Step 3: Generate metadata for each page
    _update_progress("Generating page metadata...", 0, num_pages)
    def meta_gen_callback(page_num):
        _update_progress("Generating page metadata...", page_num + 1, num_pages)
    page_metadata = generate_batch_metadata(
        page_texts=page_texts,
        context_length=context_length,
        page_images=page_images,
        model=model,
        api_key=api_key,
        api_base=api_base,
        callback=meta_gen_callback
    )
    _update_progress("Generating page metadata...", num_pages, num_pages)
    
    # Step 4: Generate document-level summary
    _update_progress("Creating document summary...", 0, 1)
    file_details = get_file_details(document_path)
    if filename:
        file_details['filename'] = filename
    document_details = extract_document_details(page_metadata, file_details)
    
    # This is the only async call, we need to handle it inside the thread
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    document_summary = loop.run_until_complete(generate_document_summary(
        page_metadata, 
        model=model,
        api_key=api_key,
        api_base=api_base
    ))
    _update_progress("Creating document summary...", 1, 1)
    
    # Step 5: Assemble complete metadata
    _update_progress("Finalizing...", 0, 1)
    metadata = assemble_document_metadata(
        page_metadata=page_metadata,
        document_summary=document_summary,
        document_details=document_details,
        page_images=page_images
    )
    _update_progress("Finalizing...", 1, 1)
    
    # Step 6: Save to file if output path provided
    if output_path:
        save_metadata_to_file(metadata, output_path)
    
    return metadata
