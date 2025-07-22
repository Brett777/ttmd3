"""
Page-level metadata generation for RAG-Ultra using LiteLLM.
Provides functions to extract detailed metadata for each page, batch process documents, and handle large documents efficiently.
"""

import json
from typing import Dict, Any, List, Optional
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import asyncio

import litellm

from .config import OPENAI_API_KEY, DEFAULT_EXTRACTION_CONFIG
from .utils import logger, count_tokens, chunk_text

# Maximum number of parallel workers for processing
MAX_WORKERS = 8

async def generate_page_metadata(
    page_text: str,
    context_text: str = "",
    page_image: str = "",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate detailed metadata for a single page using LiteLLM.

    Args:
        page_text: Text content of the current page.
        context_text: Optional text from preceding pages for context.
        page_image: Optional base64-encoded image of the page.
        model: Model identifier (e.g., "openai/gpt-4o").
        api_key: API key for the model provider.
        api_base: Optional API base URL for the model provider.
        config: Optional configuration for the API call.
    Returns:
        Dictionary with generated metadata fields for the page.
    """
    if config is None:
        config = DEFAULT_EXTRACTION_CONFIG.copy()
    if model:
        config["model"] = model
    full_text = f"{context_text}\n\n{page_text}" if context_text else page_text
    system_prompt = """
You are an expert document analyzer. Your task is to extract detailed metadata from the document text provided.
Generate the following metadata in JSON format:

1. one_sentence_summary: A single-sentence summary of the page's content (max 25 words)
2. full_summary: A comprehensive summary capturing all key information. Use bullet points or prose-style.
3. topics: A list of topics, key concepts, tags, and categories (5-10 items)
4. keywords: A list of important keywords (5-15 words or phrases)
5. key_information: Important insight, timelines, risks, financial details, contract details, etc
6. entities: A list of named entities (people, places, dates, organizations, locations, laws, etc)
7. sentiment: The overall sentiment of the page (positive, negative, neutral)
8. acronyms: A dictionary mapping acronyms to their definitions (if any present)
9. chapter_or_section: The chapter or section name if identifiable (or null)
10. noteworthy_sentences: 1-3 sentences that contain key insights or important information
11. token_count: Approximate number of tokens in the page text
12. visual_elements: A detailed description of the visual elements in the page (images, tables, charts, infographics, complex tables, etc)

Your output must be valid JSON without any prose before or after. Only include the JSON object.
"""
    messages = [
        {"role": "system", "content": system_prompt},
    ]

    # Construct the user message carefully
    user_content = []

    # Add text content only if it's not empty
    if full_text and not full_text.isspace():
        user_content.append({"type": "text", "text": full_text})
    
    # Add image content if it exists
    if page_image:
        user_content.append({
            "type": "image_url", 
            "image_url": {"url": f"data:image/jpeg;base64,{page_image}"}
        })

    # Ensure we have something to send
    if not user_content:
        logger.warning("Both page text and image are empty. Skipping metadata generation.")
        return {
            "error": "No content to process.",
            "raw_text": page_text,
            "base64_image": page_image if page_image else None,
            "token_count": 0
        }
        
    messages.append({"role": "user", "content": user_content})

    try:
        completion_args = {
            "model": config.get("model"),
            "temperature": config.get("temperature"),
            "max_tokens": config.get("max_tokens"),
            "messages": messages,
            "response_format": {"type": "json_object"}
        }
        if api_key:
            completion_args["api_key"] = api_key
        if api_base:
            completion_args["api_base"] = api_base

        # Only call the API if there is content
        if not user_content:
             return {"error": "No content provided to model"}

        response = await litellm.acompletion(**completion_args)
        result = json.loads(response.choices[0].message.content)
        result["raw_text"] = page_text
        if page_image:
            result["base64_image"] = page_image
        if "token_count" not in result:
            result["token_count"] = count_tokens(page_text)
        logger.info("Generated metadata successfully")
        return result
    except Exception as e:
        logger.error(f"Error generating metadata: {e}")
        return {
            "one_sentence_summary": "Error generating summary",
            "full_summary": "Error occurred while generating the full summary.",
            "topics": [],
            "keywords": [],
            "acronyms": {},
            "chapter_or_section": None,
            "noteworthy_sentences": [],
            "raw_text": page_text,
            "base64_image": page_image if page_image else None,
            "token_count": count_tokens(page_text),
            "error": str(e)
        }

def generate_batch_metadata(
    page_texts: Dict[int, str],
    context_length: int = 3,
    page_images: Dict[int, str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    callback: Optional[callable] = None
) -> Dict[int, Dict[str, Any]]:
    """
    Generate metadata for multiple pages in parallel, using context from previous pages.

    Args:
        page_texts: Dict mapping page numbers to text content.
        context_length: Number of previous pages to include as context.
        page_images: Optional dict mapping page numbers to base64 images.
        model: Model identifier.
        api_key: API key for the model provider.
        api_base: Optional API base URL.
        config: Optional configuration for the API call.
        callback: Optional function called after each page is processed.
    Returns:
        Dict mapping page numbers to their generated metadata.
    """
    if config is None:
        config = DEFAULT_EXTRACTION_CONFIG.copy()
    if model:
        config["model"] = model
    if page_images is None:
        page_images = {}
    results = {}
    sorted_pages = sorted(page_texts.keys())
    def process_page(page_num: int) -> tuple[int, Dict[str, Any]]:
        current_page_text = page_texts[page_num]
        current_page_image = page_images.get(page_num, "")
        context_pages = []
        for i in range(1, context_length + 1):
            prev_page = page_num - i
            if prev_page in page_texts:
                context_pages.insert(0, page_texts[prev_page])
        context_text = "\n\n".join(context_pages)
        
        # Run the async function in a new event loop for the thread
        metadata = asyncio.run(generate_page_metadata(
            page_text=current_page_text,
            context_text=context_text,
            page_image=current_page_image,
            model=model,
            api_key=api_key,
            api_base=api_base,
            config=config
        ))
        logger.info(f"Generated metadata for page {page_num}")
        return page_num, metadata
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_page = {executor.submit(process_page, page_num): page_num for page_num in sorted_pages}
        for future in concurrent.futures.as_completed(future_to_page):
            page_num, metadata = future.result()
            results[page_num] = metadata
            if callback:
                callback(page_num)
    return results

def generate_metadata_for_large_document(
    page_texts: Dict[int, str],
    context_length: int = 3,
    page_images: Dict[int, str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    chunk_size: int = 10
) -> Dict[int, Dict[str, Any]]:
    """
    Efficiently generate metadata for a large document by processing in chunks.

    Args:
        page_texts: Dict mapping page numbers to text content.
        context_length: Number of previous pages to include as context.
        page_images: Optional dict mapping page numbers to base64 images.
        model: Model identifier.
        api_key: API key for the model provider.
        api_base: Optional API base URL.
        config: Optional configuration for the API call.
        chunk_size: Number of pages to process in each batch.
    Returns:
        Dict mapping page numbers to their generated metadata.
    """
    if config is None:
        config = DEFAULT_EXTRACTION_CONFIG.copy()
    if model:
        config["model"] = model
    if page_images is None:
        page_images = {}
    results = {}
    sorted_pages = sorted(page_texts.keys())
    for i in range(0, len(sorted_pages), chunk_size):
        chunk_pages = sorted_pages[i:i + chunk_size]
        chunk_texts = {page: page_texts[page] for page in chunk_pages}
        chunk_images = {page: page_images.get(page, "") for page in chunk_pages}
        chunk_results = generate_batch_metadata(
            page_texts=chunk_texts,
            context_length=context_length,
            page_images=chunk_images,
            model=model,
            api_key=api_key,
            api_base=api_base,
            config=config
        )
        results.update(chunk_results)
        logger.info(f"Processed pages {chunk_pages[0]} to {chunk_pages[-1]}")
    return results
