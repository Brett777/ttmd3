"""
Module for generating document-level summaries from page metadata for the RAG-Ultra SDK.
"""

import json
from typing import Dict, Any, List, Optional
import asyncio

import litellm

from .config import OPENAI_API_KEY, DEFAULT_EXTRACTION_CONFIG
from .utils import logger, count_tokens

async def generate_document_summary(
    page_metadata: Dict[int, Dict[str, Any]],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate a document-level summary from the page-level metadata.
    
    Args:
        page_metadata: Dictionary with page numbers and their metadata
        model: Optional model identifier (e.g., "openai/gpt-4o", "anthropic/claude-3-opus")
        api_key: Optional API key for the model provider
        api_base: Optional API base URL for the model provider
        config: Configuration for API call
        
    Returns:
        Dictionary with document summary information
    """
    # Use default config if none provided
    if config is None:
        config = DEFAULT_EXTRACTION_CONFIG.copy()
    
    # Override model if specified
    if model:
        config["model"] = model
    
    # Calculate total token count
    total_tokens = sum(meta.get("token_count", 0) for meta in page_metadata.values())
    
    # Determine which approach to use based on document length
    if len(page_metadata) > 1000:
        # For very long documents, use one-sentence summaries
        input_summary = "\n\n".join([
            f"Page {page_num}: {meta.get('one_sentence_summary', 'No summary available')}"
            for page_num, meta in sorted(page_metadata.items())
        ])
    else:
        # For shorter documents, use full summaries
        input_summary = "\n\n".join([
            f"Page {page_num}: {meta.get('full_summary', 'No summary available')}"
            for page_num, meta in sorted(page_metadata.items())
        ])
    
    # Create the system prompt
    system_prompt = """
You are an expert document analyzer. Your task is to generate a comprehensive document summary based on page-level summaries.
Generate the following in JSON format:

1. short_summary: A concise overview of the document (max 3 sentences)
2. detailed_summary: A comprehensive explanation of what the document is about, why it's important, and what it contains (3-5 paragraphs)
3. main_topics: A list of the main topics or themes in the document (5-10 items)
4. key_insights: A list of the most important takeaways or insights from the document (3-7 items)

Your output must be valid JSON without any prose before or after. Only include the JSON object.
"""

    # Call LiteLLM
    try:
        completion_args = {
            "model": config.get("model"),
            "temperature": config.get("temperature"),
            "max_tokens": config.get("max_tokens"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_summary}
            ],
            "response_format": {"type": "json_object"}
        }
        
        # Add API key and base URL if provided
        if api_key:
            completion_args["api_key"] = api_key
        if api_base:
            completion_args["api_base"] = api_base
            
        response = await litellm.acompletion(**completion_args)
        
        # Parse the JSON response
        result = json.loads(response.choices[0].message.content)
        
        # Add total token count
        result["total_token_count"] = total_tokens
        result["total_pages"] = len(page_metadata)
        
        logger.info("Generated document summary successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error generating document summary: {e}")
        # Return a basic document summary in case of error
        return {
            "short_summary": "Error generating document summary.",
            "detailed_summary": "An error occurred while generating the detailed document summary.",
            "main_topics": [],
            "key_insights": [],
            "total_token_count": total_tokens,
            "total_pages": len(page_metadata),
            "error": str(e)
        }

def extract_document_details(
    page_metadata: Dict[int, Dict[str, Any]],
    file_details: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extract document-level details from metadata and file information.
    
    Args:
        page_metadata: Dictionary with page numbers and their metadata
        file_details: Dictionary with file details
        
    Returns:
        Dictionary with document details
    """
    # Start with file details
    document_details = file_details.copy()
    
    # Add total number of pages
    document_details["total_pages"] = len(page_metadata)
    
    # Try to determine document title by analyzing first few pages
    title = None
    
    # Start with first 3 pages
    for page_num in sorted(page_metadata.keys())[:3]:
        page_meta = page_metadata[page_num]
        
        # Look for chapter or section name in first pages
        if page_meta.get("chapter_or_section") and "title" in page_meta.get("chapter_or_section", "").lower():
            title = page_meta.get("chapter_or_section")
            break
        
        # Look for potential title in noteworthy sentences
        for sentence in page_meta.get("noteworthy_sentences", []):
            if len(sentence.split()) <= 12 and any(word in sentence.lower() for word in ["title", "document", "report"]):
                title = sentence
                break
    
    # If we didn't find a title, use the filename without extension
    if not title and "filename" in document_details:
        title = document_details["filename"].rsplit(".", 1)[0].replace("_", " ").title()
    
    document_details["title"] = title or "Untitled Document"
    
    # Add language (assume English for now)
    document_details["language"] = "en"
    
    # Look for author information in the first few pages
    author = None
    
    for page_num in sorted(page_metadata.keys())[:5]:
        page_meta = page_metadata[page_num]
        raw_text = page_meta.get("raw_text", "")
        
        # Look for common author patterns
        author_patterns = [
            "Author:", "By:", "Written by:", "Prepared by:"
        ]
        
        for pattern in author_patterns:
            if pattern in raw_text:
                author_line = raw_text.split(pattern, 1)[1].split("\n")[0].strip()
                if author_line and len(author_line) < 100:
                    author = author_line
                    break
        
        if author:
            break
    
    document_details["author"] = author or "Unknown"
    
    # Look for a date
    date = None
    
    for page_num in sorted(page_metadata.keys())[:5]:
        page_meta = page_metadata[page_num]
        raw_text = page_meta.get("raw_text", "")
        
        # Look for common date patterns
        date_patterns = [
            "Date:", "Published:", "Released on:", "Created on:"
        ]
        
        for pattern in date_patterns:
            if pattern in raw_text:
                date_line = raw_text.split(pattern, 1)[1].split("\n")[0].strip()
                if date_line and len(date_line) < 30:
                    date = date_line
                    break
        
        if date:
            break
    
    document_details["date"] = date or "Unknown"
    
    return document_details

def assemble_document_metadata(
    page_metadata: Dict[int, Dict[str, Any]],
    document_summary: Dict[str, Any],
    document_details: Dict[str, Any],
    page_images: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """
    Assemble the complete document metadata dictionary.
    
    Args:
        page_metadata: Dictionary with page numbers and their metadata
        document_summary: Dictionary with document summary
        document_details: Dictionary with document details
        page_images: Optional dictionary with page numbers and base64 images
        
    Returns:
        Complete document metadata dictionary
    """
    # Start with document details
    result = {
        "document_details": document_details,
        "document_summary": document_summary,
        "pages": {}
    }
    
    # Add page metadata with images if available
    for page_num, metadata in page_metadata.items():
        page_data = metadata.copy()
        
        # Add base64 image if available
        if page_images and page_num in page_images:
            page_data["base64_image"] = page_images[page_num]
        
        result["pages"][page_num] = page_data
    
    return result
