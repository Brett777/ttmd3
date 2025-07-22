"""
Retriever Agent for RAG-Ultra: Adaptive, efficient, and LLM-driven document retrieval.
- Dynamically selects metadata fields based on the user's question.
- Attempts to answer using metadata only, then falls back to raw content if needed.
- Uses structured output for all LLM calls.
"""

from typing import Dict, Any, List, Optional, Tuple
import json
import litellm
from rag_ultra.config import DEFAULT_RETRIEVER_CONFIG
from rag_ultra.utils import logger

# Model max output tokens mapping
MODEL_MAX_TOKENS = {
    # OpenAI
    "gpt-4.1": 32768,
    "gpt-4.1-mini": 32768,
    "gpt-4.1-nano": 32768,
    "gpt-4o": 4096,
    "gpt-4o-search-preview": 4096,
    "gpt-4.5-preview": 4096,
    "gpt-4o-mini": 4096,
    "gpt-4o-mini-search-preview": 4096,
    "o1-pro": 4096,
    "o1": 4096,
    "o1-mini": 4096,
    "o3-mini": 4096,
    "o1-preview": 4096,
    "chatgpt-4o-latest": 4096,
    "gpt-4-turbo": 4096,
    "gpt-3.5-turbo": 4096,
    # xAI
    "grok-3-beta": 131072,
    "grok-3-fast-beta": 131072,
    "grok-3-fast-latest": 131072,
    "grok-3-mini-beta": 131072,
    "grok-3-mini-fast-beta": 131072,
    "grok-3-mini-fast-latest": 131072,
    "grok-beta": 4096,
    "grok-2-vision-latest": 4096,
    "grok-2-vision": 4096,
    "grok-2": 4096,
    "grok-2-latest": 4096,
    "grok-vision-beta": 4096,
    # Anthropic
    "claude-3-7-sonnet-latest": 128000,
    "claude-3-7-sonnet-20250219": 128000,
    "claude-3-haiku-20240307": 128000,
    "claude-3-5-haiku-20241022": 128000,
    "claude-3-5-haiku-latest": 128000,
    "claude-3-opus-latest": 128000,
    "claude-3-opus-20240229": 128000,
    "claude-3-sonnet-20240229": 128000,
    "claude-3-5-sonnet-latest": 128000,
    "claude-3-5-sonnet-20240620": 128000,
    "claude-3-5-sonnet-20241022": 128000,
    "claude-instant-1": 128000,
    "claude-instant-1.2": 128000,
    "claude-2": 128000,
    "claude-2.1": 128000,
    # Deepseek
    "deepseek-reasoner": 8000,
    "deepseek-chat": 8000,
    "deepseek-coder": 8000,
    # Cohere
    "command": 4000,
    "command-a-03-2025": 8000,
    "command-r7b-12-2024": 4000,
    "command-r-plus": 4000,
    "command-r": 4000,
    "command-nightly": 4000,
    "command-light-nightly": 4000,
}

def get_max_tokens_for_model(model: Optional[str]) -> int:
    """
    Return the max output tokens for a given model name.
    """
    if not model:
        return 4000
    model_lower = model.lower()
    for key, val in MODEL_MAX_TOKENS.items():
        if key in model_lower:
            return val
    return 4000


def retrieve_answer_with_metadata_and_content(
    prompt: str,
    metadata_context: Dict[str, Any],
    full_metadata: Dict[str, Any],
    model: str,
    api_key: str,
    api_base: Optional[str] = None,
    max_tool_call_loops: int = 5,
    messages: Optional[list] = None
) -> Dict[str, Any]:
    """
    Run a retrieval-augmented generation (RAG) loop using metadata and, if needed, raw content.
    Attempts to answer using only metadata first, then fetches raw text or images if required.

    Args:
        prompt: The user's question.
        metadata_context: The minimal metadata fields selected for initial context.
        full_metadata: The complete metadata, used for raw content retrieval if needed.
        model: LLM model name.
        api_key: API key for the LLM provider.
        api_base: Optional API base URL.
        max_tool_call_loops: Max number of tool call loops to prevent infinite loops.
        messages: Optional conversation history (OpenAI format).
    Returns:
        A dictionary with the answer, cited pages, reasoning, and any raw/image content used.
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_raw_pages",
                "description": "Fetch raw text for specific document pages. This is useful for answering questions about the document content when the metadata alone is not sufficient.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pages": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "document": {"type": "string"},
                                    "page": {"type": "integer"}
                                },
                                "required": ["document", "page"]
                            }
                        }
                    },
                    "required": ["pages"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_raw_page_images",
                "description": "Fetch base64-encoded images for specific document pages. Use this if the visual_elements field indicates that there are several relevant visual elements worth considering. This is useful for visual questions (about charts, images, diagrams, maps, page layout, colors, etc).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pages": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "document": {"type": "string"},
                                    "page": {"type": "integer"}
                                },
                                "required": ["document", "page"]
                            }
                        }
                    },
                    "required": ["pages"]
                }
            }
        }
    ]

    system_prompt = (
        """
        ROLE
        Respond to the user prompt to the best of your ability using the available metadata and/or raw content.
        Your answers should primarily be based on the provided metadata and raw data (when provided), rather than your own knowledge.
        CONTEXT DATA
        You are given a dictionary of documents, where each key is a document name and each value is that document's metadata (including summaries and page-level metadata).
        You may also be given a dictionary of the raw page content as well, structured as {document: {page: content}}.
        For visual questions (about charts, diagrams, images, maps, page layout, colors, etc), try to answer using the 'visual_elements' field in the metadata. If the visual_elements field indicates relevant visual information, you may wish to also call the get_raw_page_images function to retrieve the base64-encoded image for that page which may provide more insight.
        TASK
        Given the user's prompt, the metadata, and the raw page content (if you decided to access it), generate a comprehensive answer.
        If the metadata alone is not sufficient to address the user's prompt, you may call the function get_raw_pages to retrieve the raw page data you need. If you need to reference a visual element, you may call get_raw_page_images to retrieve the page image.
        Within your answer, justify your explanation with references to the relevant pages inline in your response.
        Your response should be in the following JSON format:
        {
          "answer": "...",
          "cited_pages": [{"document": "...", "page": ...}],
          "reasoning": "..."
        }
        If you cannot answer, return an empty answer and explain why in the reasoning field.
        TOOL USE AND FUNCTION CALLING
        You may call both get_raw_pages and get_raw_page_images functions if needed.
        You may decide to call one or the other, or both, or none at all.
        You may call the functions multiple times if needed, to a maximum of 5 times.
        There may be a conversation history! If the user is asking for even more detail than last time, and last time only metadata was used, then you should really consider retrieving the raw content such as the raw text or the base64 images.
        """
    )

    # Build conversation history
    if messages and isinstance(messages, list) and len(messages) > 0:
        # Sanitize the history to prevent issues with complex messages from previous turns
        sanitized_history = []
        for msg in messages:
            # Only include role and content, and handle cases where content might be None
            if msg.get("content") and isinstance(msg.get("content"), str):
                sanitized_history.append({"role": msg["role"], "content": msg["content"]})

        if sanitized_history and sanitized_history[0].get("role") != "system":
            conversation = [{"role": "system", "content": system_prompt}] + sanitized_history
        else:
            conversation = sanitized_history or [{"role": "system", "content": system_prompt}]

        # Add the initial prompt if it's not already in the history
        if not any(c["role"] == "user" and c["content"] == prompt for c in conversation):
             conversation.append({"role": "user", "content": prompt})
        
        conversation.append({"role": "assistant", "content": f"Document metadata: {json.dumps(metadata_context)}"})
    else:
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"Document metadata: {json.dumps(metadata_context)}"}
        ]
    raw_content_used = False
    raw_content = None
    image_content_used = False
    image_content = None

    def get_page_images(metadata_dict, page_refs):
        """
        Retrieve base64 images for the specified [{document, page}] pairs.
        Returns a dict: {document: {page: {"base64_image": ..., ...}}}
        """
        result = {}
        for ref in page_refs:
            doc = ref["document"]
            page = ref["page"]
            if doc in metadata_dict and "pages" in metadata_dict[doc]:
                page_data = metadata_dict[doc]["pages"].get(str(page)) or metadata_dict[doc]["pages"].get(page)
                if page_data and "base64_image" in page_data:
                    if doc not in result:
                        result[doc] = {}
                    entry = {
                        "base64_image": page_data["base64_image"],
                        "page": page,
                        "chapter_or_section": page_data.get("chapter_or_section")
                    }
                    result[doc][page] = entry
        return result

    loop_count = 0
    while True:
        loop_count += 1
        if loop_count > max_tool_call_loops:
            logger.error("Exceeded maximum tool call loops. Possible infinite loop in tool calling.")
            return {
                "answer": "",
                "cited_pages": [],
                "reasoning": "Stopped due to too many tool call loops (possible infinite loop).",
                "raw_content_used": raw_content_used,
                "raw_content": raw_content,
                "image_content_used": image_content_used,
                "image_content": image_content
            }
        response = litellm.completion(
            model=model,
            messages=conversation,
            tools=tools,
            tool_choice="auto",
            api_key=api_key,
            api_base=api_base
        )
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            if hasattr(message, 'model_dump'):
                conversation.append(message.model_dump())
            else:
                conversation.append(dict(message))
            for tool_call in tool_calls:
                func_name = tool_call.function.name
                if func_name == "get_raw_pages":
                    args = json.loads(tool_call.function.arguments)
                    pages = args["pages"]
                    raw_text_content = get_raw_text_for_pages(full_metadata, pages)
                    conversation.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": func_name,
                        "content": json.dumps(raw_text_content),
                    })
                    raw_content_used = True
                    if raw_text_content:
                        raw_content = raw_text_content
                elif func_name == "get_raw_page_images":
                    args = json.loads(tool_call.function.arguments)
                    pages = args["pages"]
                    image_content = get_page_images(full_metadata, pages)
                    image_content_used = True
                    # Step 1: Append a simple text response for the tool call.
                    conversation.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": func_name,
                        "content": f"Successfully retrieved image data for {len(pages)} page(s).",
                    })
                    # Step 2: Append a new assistant message with the visual content.
                    if image_content:
                        image_message_content = [{"type": "text", "text": "Here is the image content for analysis:"}]
                        for doc_name, pages_data in image_content.items():
                            for page_num, page_content in pages_data.items():
                                if page_content.get("base64_image"):
                                    image_message_content.append({
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{page_content['base64_image']}"}
                                    })
                        conversation.append({"role": "assistant", "content": image_message_content})
                else:
                    logger.warning(f"Unsupported tool call: {func_name}")
                    # Still need to append a response, even for an unknown tool.
                    conversation.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": func_name,
                        "content": f"Tool '{func_name}' is not supported.",
                    })
            continue
        answer = message.content
        try:
            answer_json = json.loads(answer)
        except Exception:
            answer_json = {"answer": answer}
        answer_json["raw_content_used"] = raw_content_used
        if raw_content_used:
            answer_json["raw_content"] = raw_content
        answer_json["image_content_used"] = image_content_used
        if image_content_used:
            answer_json["image_content"] = image_content

        # If the model hallucinates keys, ensure they are of the correct type
        if not isinstance(answer_json.get("cited_pages"), list):
            answer_json["cited_pages"] = []
        
        # Manually construct citations from any tool calls that fetched raw content.
        # This is a robust fallback for when the model generates a correct answer from
        # the content but forgets to include the citations in its final response.
        cited_pages_from_tools = []
        if raw_content:
            for doc, pages in raw_content.items():
                for p_num in pages.keys():
                    cited_pages_from_tools.append({"document": doc, "page": int(p_num)})
        if image_content:
            for doc, pages in image_content.items():
                for p_num in pages.keys():
                    cited_pages_from_tools.append({"document": doc, "page": int(p_num)})

        # De-duplicate pages from the model's response and our manual list from tool calls
        all_cited_pages = answer_json.get("cited_pages", []) + cited_pages_from_tools
        unique_cited_pages = []
        seen_pages = set()
        for page_ref in all_cited_pages:
            # Ensure page_ref is a dict and has 'document' and 'page' keys
            if isinstance(page_ref, dict) and 'document' in page_ref and 'page' in page_ref:
                try:
                    page_tuple = (page_ref["document"], int(page_ref["page"]))
                    if page_tuple not in seen_pages:
                        unique_cited_pages.append({"document": page_tuple[0], "page": page_tuple[1]})
                        seen_pages.add(page_tuple)
                except (ValueError, TypeError):
                    logger.warning(f"Skipping invalid page reference during citation processing: {page_ref}")

        return {
            "answer": answer_json.get("answer", ""),
            "cited_pages": unique_cited_pages,
            "reasoning": answer_json.get("reasoning", ""),
            "raw_content_used": raw_content_used,
            "raw_content": raw_content,
            "image_content_used": image_content_used,
            "image_content": image_content
        }


def retrieve_agent(
    prompt: str,
    metadata_dict: Dict[str, Any],
    document_name: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
    messages: Optional[list] = None
) -> Dict[str, Any]:
    """
    Main entry point for the smarter retriever agent.
    Returns a dictionary with the answer, relevant pages, and reasoning trace.
    Always expects metadata_dict as {document_name: metadata} (even for single doc).
    If messages is provided, it is used as the conversation history (OpenAI format).
    """
    # If single doc, wrap in dict
    if not all(isinstance(v, dict) and ("pages" in v or "document_details" in v) for v in metadata_dict.values()):
        metadata_dict = {"default": metadata_dict}

    # Document selection
    if document_name:
        if document_name in metadata_dict:
            filtered_metadata = {document_name: metadata_dict[document_name]}
        else:
            filtered_metadata = metadata_dict
    else:
        selected_docs = select_relevant_documents(
            prompt=prompt,
            metadata_dict=metadata_dict,
            model=model,
            api_key=api_key,
            api_base=api_base
        )
        if not selected_docs:
            logger.warning("No documents selected for retrieval.")
            return {
                "query": prompt,
                "answer": "No relevant documents were found to answer your question.",
                "used_metadata_fields": [],
                "used_metadata": {},
                "reasoning_trace": {"reasoning": "No documents were selected for analysis based on the initial assessment."},
                "raw_content_used": False,
                "relevant_pages": [],
                "raw_content": None,
                "image_content_used": False,
                "image_content": None,
            }
        
        logger.info(f"Selected documents: {selected_docs}")

        # Step 3: Extract document-level metadata for selected documents
        doc_level_metadata = extract_document_level_metadata({k: v for k, v in metadata_dict.items() if k in selected_docs})

        # Step 4: Determine which metadata fields are relevant to the question
        available_fields = get_available_metadata_fields(doc_level_metadata)
        selected_fields = get_metadata_fields_for_question(prompt, available_fields, model, api_key, api_base)
        logger.info(f"Selected metadata fields: {selected_fields}")

        # Step 5: Extract the selected metadata fields for context
        metadata_context = extract_metadata_fields(doc_level_metadata, selected_fields)

        # Step 6: Use the RAG loop to get an answer
        retrieval_result = retrieve_answer_with_metadata_and_content(
            prompt=prompt,
            metadata_context=metadata_context,
            full_metadata=metadata_dict,
            model=model,
            api_key=api_key,
            api_base=api_base,
            messages=messages
        )
        
        # --- START OF CHANGE: Explicitly generate reasoning at the end ---
        reasoning_prompt = f"""
        Based on the following information, please provide a brief, one or two-sentence reasoning trace explaining the steps taken to answer the user's query.

        User Query: "{prompt}"
        
        Sources Used:
        - Metadata fields analyzed: {json.dumps(selected_fields)}
        - Raw text from documents was used: {retrieval_result.get('raw_content_used', False)}
        - Images from documents were used: {retrieval_result.get('image_content_used', False)}
        - Documents cited: {json.dumps([p['document'] for p in retrieval_result.get('cited_pages', [])])}
        - Pages cited: {json.dumps([p['page'] for p in retrieval_result.get('cited_pages', [])])}

        Final Answer: "{retrieval_result.get('answer', 'N/A')}"

        Example Reasoning: "To answer the user's query, I first analyzed the 'full_summary' and 'topics' metadata fields. This was not enough, so I retrieved the raw text from page 1 of 'report.pdf' to get the final answer."

        Your Reasoning:
        """
        
        try:
            reasoning_completion = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": reasoning_prompt}],
                api_key=api_key,
                api_base=api_base,
            )
            final_reasoning = reasoning_completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate final reasoning: {e}")
            final_reasoning = "Reasoning trace could not be generated."
        # --- END OF CHANGE ---

        return {
            "query": prompt,
            "answer": retrieval_result.get("answer", ""),
            "used_metadata_fields": selected_fields,
            "used_metadata": metadata_context,
            "reasoning_trace": {"reasoning": final_reasoning},
            "raw_content_used": retrieval_result.get("raw_content_used", False),
            "relevant_pages": retrieval_result.get("cited_pages", []),
            "raw_content": retrieval_result.get("raw_content"),
            "image_content_used": retrieval_result.get("image_content_used", False),
            "image_content": retrieval_result.get("image_content"),
        }


def get_metadata_fields_for_question(
    prompt: str,
    available_fields: List[str],
    model: Optional[str],
    api_key: Optional[str],
    api_base: Optional[str]
) -> List[str]:
    """
    Use an LLM to select the most relevant metadata fields for the question.
    Returns a list of field names.
    """
    system_prompt = (
        """
        You are a smart document retrieval agent.
        Given a user prompt or question and a list of available metadata fields about the available documents,
        select the minimal set of fields that would be helpful in determining where in the documents the answer would be found. 
        Return your answer in the following JSON format:
        {   
            "metadata_fields": [field_1, field_2, field_3, ...],
            "explanation": "A brief explanation of why these fields are relevant"
        }
        If the question is about general document info, you might select document level fields.
        If the question requires specific content from pages within the document, you might select page level fields.
        Never include fields that are not in the provided list below.
        Only include minimum fields needed to determine where in the documents the answer would be found.
        It would be excessive to include all fields! 
        It would be excessive to include both the short_summary and detailed_summary! Use the short_summary for general document info, or the detailed_summary for specific content.
        Your goal is to minimize the amount of metadata and tokens needed to retrieve the answer to the question.
        Therefore, at this stage we should not retieve the raw_text or the base 64 image. (But you may however retrieve the visual_elements if needed)
        In a later step, we will use these fields that you select to answer questions about the document and retrieve the pages of the document that contain the details of the answer.
        So for example, if the user is asking about a specific topic, you might consider returning the page level field, "topics", or "keywords" or "entities".

        document level fields:
        1. short_summary: A concise overview of the document (max 3 sentences)
        2. detailed_summary: A comprehensive explanation of what the document is about, why it's important, and what it contains (3-5 paragraphs)
        3. main_topics: A list of the main topics or themes in the document (5-10 items)
        4. key_insights: A list of the most important takeaways or insights from the document (3-7 items)

        page level fields:
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
        """
    )
    user_message = (
        f"User question: {prompt}\n"
        f"Available metadata fields: {json.dumps(available_fields)}"
    )
    completion_args = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0,
        "max_tokens": min(128, get_max_tokens_for_model(model)),
        "response_format": {"type": "json_object"}
    }
    if api_key:
        completion_args["api_key"] = api_key
    if api_base:
        completion_args["api_base"] = api_base
    try:
        response = litellm.completion(**completion_args)
        fields = json.loads(response.choices[0].message.content)
        if isinstance(fields, dict) and "metadata_fields" in fields:
            return fields["metadata_fields"]
        if isinstance(fields, dict) and "fields" in fields:
            return fields["fields"]
        if isinstance(fields, list):
            return fields
        return available_fields[:5]  # fallback: first 5 fields
    except Exception as e:
        logger.error(f"Error selecting metadata fields: {e}")
        return available_fields[:5]


def get_available_metadata_fields(metadata_dict: Dict[str, Any]) -> List[str]:
    """
    Return a sorted list of all available metadata fields across all documents/pages.
    """
    fields = set()
    for doc in metadata_dict.values():
        if isinstance(doc, dict):
            if "pages" in doc:
                for page in doc["pages"].values():
                    fields.update(page.keys())
            if "document_details" in doc:
                fields.update(doc["document_details"].keys())
            if "document_summary" in doc:
                fields.update(doc["document_summary"].keys())
    return sorted(fields)


def extract_metadata_fields(metadata_dict: Dict[str, Any], selected_fields: List[str]) -> Dict[str, Any]:
    """
    Extract only the selected metadata fields from each document/page.
    """
    result = {}
    for doc_name, doc in metadata_dict.items():
        doc_result = {}
        for section in ["document_details", "document_summary"]:
            if section in doc:
                # Correctly extract only the selected fields from the section
                extracted_section = {
                    field: doc[section][field] 
                    for field in selected_fields 
                    if field in doc[section]
                }
                if extracted_section:
                    doc_result[section] = extracted_section

        if "pages" in doc:
            doc_result["pages"] = {}
            for page_num, page_data in doc["pages"].items():
                page_result = {field: page_data[field] for field in selected_fields if field in page_data}
                if page_result:
                    doc_result["pages"][page_num] = page_result
        if doc_result:
            result[doc_name] = doc_result
    return result


def get_raw_text_for_pages(
    metadata_dict: Dict[str, Any],
    page_refs: list,
    need_images: bool = False
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Retrieve raw text and minimal context for the specified [{document, page}] pairs.
    Returns a dict: {document: {page: {"raw_text": ..., ...}}}
    """
    result = {}
    for ref in page_refs:
        doc = ref["document"]
        page = ref["page"]
        if doc in metadata_dict and "pages" in metadata_dict[doc]:
            page_data = metadata_dict[doc]["pages"].get(str(page)) or metadata_dict[doc]["pages"].get(page)
            if page_data and "raw_text" in page_data:
                if doc not in result:
                    result[doc] = {}
                entry = {
                    "raw_text": page_data["raw_text"],
                    "page": page,
                    "chapter_or_section": page_data.get("chapter_or_section")
                }
                result[doc][page] = entry
    return result


def extract_document_level_metadata(metadata_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only document-level metadata from each document.
    """
    result = {}
    
    # Document-level fields to extract
    doc_details_fields = [
        "filename", "file_type", "size_kb", "last_modified", 
        "total_pages", "title", "language", "author", "date"
    ]
    
    doc_summary_fields = [
        "short_summary", "detailed_summary", "main_topics", 
        "key_insights", "total_token_count", "total_pages"
    ]
    
    for doc_name, doc_metadata in metadata_dict.items():
        doc_result = {}
        
        # Extract document_details fields
        if "document_details" in doc_metadata:
            doc_result["document_details"] = {}
            for field in doc_details_fields:
                if field in doc_metadata["document_details"]:
                    doc_result["document_details"][field] = doc_metadata["document_details"][field]
        
        # Extract document_summary fields
        if "document_summary" in doc_metadata:
            doc_result["document_summary"] = {}
            for field in doc_summary_fields:
                if field in doc_metadata["document_summary"]:
                    doc_result["document_summary"][field] = doc_metadata["document_summary"][field]
        
        # Only include document if it has metadata
        if doc_result:
            result[doc_name] = doc_result
    
    return result


def select_relevant_documents(
    prompt: str,
    metadata_dict: Dict[str, Any],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None
) -> List[str]:
    """
    Use an LLM to select documents that are most relevant to the user's prompt.
    Only uses document-level metadata (not page-level) to make this determination.
    """
    # If there are fewer than 3 documents, return all of them
    if len(metadata_dict) < 3:
        return list(metadata_dict.keys())
    
    # Extract only document-level metadata
    doc_level_metadata = extract_document_level_metadata(metadata_dict)
    
    system_prompt = """
    You are a smart document retrieval agent.
    Given a user prompt or question and a set of documents with their high-level metadata, 
    select the documents that are most likely to contain information relevant to answering the question.
    
    The document metadata includes:
    - document_details: Basic information about the document (filename, title, author, date, etc.)
    - document_summary: Summary information (short_summary, detailed_summary, main_topics, key_insights)
    
    Return your answer in the following JSON format:
    {
        "relevant_documents": ["doc1", "doc2", ...],
        "explanation": "A brief explanation of why these documents were selected and others were excluded"
    }
    
    Only include documents that are likely to be relevant to the user's question.
    If all documents seem relevant, include them all.
    If none of the documents seem relevant, return an empty list.
    """
    
    user_message = (
        f"User question: {prompt}\n\n"
        f"Available documents and their metadata: {json.dumps(doc_level_metadata, indent=2)}"
    )
    
    completion_args = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"}
    }
    
    if api_key:
        completion_args["api_key"] = api_key
    if api_base:
        completion_args["api_base"] = api_base
    
    try:
        response = litellm.completion(**completion_args)
        result = json.loads(response.choices[0].message.content)
        if isinstance(result, dict) and "relevant_documents" in result:
            logger.info(f"Document selection explanation: {result.get('explanation', 'No explanation provided')}")
            return result["relevant_documents"]
        return list(metadata_dict.keys())  # Return all if response format is unexpected
    except Exception as e:
        logger.error(f"Error selecting relevant documents: {e}")
        return list(metadata_dict.keys())  # Return all documents on error 