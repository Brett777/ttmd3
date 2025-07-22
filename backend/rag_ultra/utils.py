"""
Utility functions for the RAG-Ultra SDK.
"""

import os
import logging
import json
from typing import Dict, Any, List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag_ultra")

def count_tokens(text: str) -> int:
    """
    Count the approximate number of tokens in a text string.
    
    This is a simple approximation where we count words and multiply by 1.3
    for a conservative estimate. For more accurate counting, consider using
    a tokenizer matched to your model.
    
    Args:
        text: The text to count tokens for
        
    Returns:
        An approximate token count
    """
    # Simple approximation
    words = text.split()
    # Multiply by 1.3 for a conservative estimate
    return int(len(words) * 1.3)

def save_metadata_to_file(metadata: Dict[str, Any], output_path: str) -> None:
    """
    Save the metadata dictionary to a JSON file.
    
    Args:
        metadata: The metadata dictionary
        output_path: Path to save the JSON file
    """
    # Only try to create directory if the dirname is not empty
    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Metadata saved to {output_path}")

def load_metadata_from_file(input_path: str) -> Dict[str, Any]:
    """
    Load metadata from a JSON file.
    
    Args:
        input_path: Path to the JSON file
        
    Returns:
        The metadata dictionary
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    logger.info(f"Metadata loaded from {input_path}")
    return metadata

def get_file_details(file_path: str) -> Dict[str, Any]:
    """
    Get details about a file such as size, type, and name.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file details
    """
    file_stats = os.stat(file_path)
    file_name = os.path.basename(file_path)
    file_ext = os.path.splitext(file_name)[1].lower().lstrip('.')
    
    return {
        "filename": file_name,
        "file_type": file_ext,
        "size_kb": file_stats.st_size / 1024,
        "last_modified": file_stats.st_mtime
    }

def chunk_text(text: str, max_tokens: int = 4000) -> List[str]:
    """
    Split text into chunks based on token limits.
    
    Args:
        text: The text to chunk
        max_tokens: Maximum tokens per chunk
        
    Returns:
        List of text chunks
    """
    if count_tokens(text) <= max_tokens:
        return [text]
    
    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph_tokens = count_tokens(paragraph)
        
        # If a single paragraph is too large, split it by sentences
        if paragraph_tokens > max_tokens:
            sentences = paragraph.split('. ')
            for sentence in sentences:
                sentence_tokens = count_tokens(sentence)
                
                if count_tokens(current_chunk) + sentence_tokens > max_tokens:
                    chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
        else:
            if count_tokens(current_chunk) + paragraph_tokens > max_tokens:
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def delete_document_from_metadata(metadata: Dict[str, Any], document_name: str) -> (Dict[str, Any], bool):
    """
    Delete a document and all its metadata from the metadata dictionary.
    Supports both single-document and multi-document formats.

    Args:
        metadata: The metadata dictionary (single or multi-document)
        document_name: The name of the document to delete (filename for multi-doc, or 'default' for single-doc)

    Returns:
        (updated_metadata, deleted): Tuple of updated metadata dict and bool indicating if deletion occurred
    """
    if is_multi_document_format(metadata):
        if document_name in metadata:
            del metadata[document_name]
            return metadata, True
        else:
            return metadata, False
    else:
        # For single-document, treat any name as a delete-all
        if document_name.lower() in ("default", metadata.get("document_details", {}).get("filename", "").lower()):
            return {"deleted": True}, True
        else:
            return metadata, False

def is_multi_document_format(metadata_dict: dict) -> bool:
    """
    Determine if the metadata dictionary represents multiple documents.
    Args:
        metadata_dict: The metadata dictionary to check.
    Returns:
        True if the dictionary is in multi-document format, False otherwise.
    """
    if not isinstance(metadata_dict, dict):
        return False
    # Heuristic: multi-doc if all values are dicts with 'pages' or 'document_details' keys
    return all(
        isinstance(v, dict) and (
            'pages' in v or 'document_details' in v
        ) for v in metadata_dict.values()
    )

def get_document_names(metadata_dict: dict) -> list:
    """
    Get the list of document names from a multi-document metadata dictionary.
    Args:
        metadata_dict: The metadata dictionary.
    Returns:
        List of document names (keys).
    """
    if not is_multi_document_format(metadata_dict):
        return ['default']
    return list(metadata_dict.keys())

def list_documents_in_metadata_store(metadata: dict) -> list:
    """
    List all document names in the metadata store (supports single and multi-document formats).
    """
    return get_document_names(metadata)

def get_document_level_metadata(metadata: dict, document_name: str = None) -> dict:
    """
    Retrieve the document-level metadata for a given document.
    If single-document, document_name can be None or 'default'.
    """
    if is_multi_document_format(metadata):
        if document_name in metadata:
            doc = metadata[document_name]
            return doc.get('document_details', doc)
        else:
            raise KeyError(f"Document '{document_name}' not found in metadata store.")
    else:
        return metadata.get('document_details', metadata)

def get_page_level_metadata_range(metadata: dict, document_name: str, start_page: int, end_page: int) -> list:
    """
    Retrieve the page-level metadata for a given document and page range (inclusive, 1-based).
    Returns a list of page metadata dicts.
    """
    if is_multi_document_format(metadata):
        doc = metadata.get(document_name)
        if not doc:
            raise KeyError(f"Document '{document_name}' not found in metadata store.")
        pages = doc.get('pages', [])
    else:
        pages = metadata.get('pages', [])
    # Pages are 1-based in UI, 0-based in list
    return pages[start_page-1:end_page]

def get_page_metadata_field(metadata: dict, document_name: str, page: int, field: str):
    """
    Retrieve a specific metadata field for a given document and page (1-based).
    """
    if is_multi_document_format(metadata):
        doc = metadata.get(document_name)
        if not doc:
            raise KeyError(f"Document '{document_name}' not found in metadata store.")
        pages = doc.get('pages', [])
    else:
        pages = metadata.get('pages', [])
    if page < 1 or page > len(pages):
        raise IndexError(f"Page {page} out of range.")
    return pages[page-1].get(field)

def get_metadata_field_for_documents(metadata: dict, field: str or list, document_names: list = None):
    """
    Retrieve a metadata field (or list of fields) for all documents or a specific list of documents.
    Returns a dict mapping document name to field value(s).
    """
    if not is_multi_document_format(metadata):
        doc_name = 'default'
        docs = {doc_name: metadata}
    else:
        docs = metadata if document_names is None else {k: metadata[k] for k in document_names if k in metadata}
    result = {}
    for doc_name, doc in docs.items():
        if isinstance(field, list):
            result[doc_name] = {f: doc.get(f, doc.get('document_details', {}).get(f)) for f in field}
        else:
            result[doc_name] = doc.get(field, doc.get('document_details', {}).get(field))
    return result
