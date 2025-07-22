# RAG-Ultra

A powerful SDK for extracting detailed metadata from various document types and building an intelligent retrieval system.

## Overview

RAG-Ultra processes documents (PDFs, Word, PowerPoint, text files) and extracts both raw text and structured metadata using LLM APIs (OpenAI, Anthropic, xAI, Cohere, Deepseek, etc). The extracted data is organized into a hierarchical Python dictionary, supporting both single and multi-document use cases. This metadata can be used for document analysis, search, and retrieval.

Key features:
- Extract text from various document formats (PDF, DOCX, PPTX, TXT)
- Convert document pages to base64 images
- Generate comprehensive metadata using LLMs
- Create document-level summaries and insights
- **Multi-document support**: process and query multiple documents at once
- **Intelligent retrieval agent**: dynamically selects minimal metadata fields for each query, reducing latency and cost
- **Citations as {document, page} pairs** for robust multi-document referencing
- **Dynamic max_tokens**: LLM calls use optimal max tokens for the selected model
- Improved error handling and output formatting

## Setup

1.  **Unzip the Project:**
    Unzip the `rag-ultra.zip` file to your desired location.

2.  **Navigate to Project Directory:**
    Open your terminal or command prompt and navigate into the unzipped `rag-ultra` directory:
    ```bash
    cd path/to/rag-ultra
    ```

3.  **Create and Activate Virtual Environment:**
    It is highly recommended to use a virtual environment. From the `rag-ultra` root directory:
    ```bash
    python -m venv venv
    ```
    Activate the virtual environment:
    *   Windows (PowerShell):
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
    *   Windows (Command Prompt):
        ```bash
        .\venv\Scripts\activate.bat
        ```
    *   macOS/Linux (bash/zsh):
        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies:**
    With the virtual environment activated, install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Install RAG-Ultra SDK:**
    Install the RAG-Ultra SDK itself in editable mode. This makes it available for import by the demo scripts and the web application:
    ```bash
    pip install -e .
    ```

## Requirements

- Python 3.7+
- API key for your LLM provider (OpenAI, Anthropic, etc)
- PDF processing tools (see below)
- Python packages: see requirements.txt

**Environment Variables and API Keys:**
For development, it's recommended to store your API keys and other sensitive configurations in a `.env` file in the project root. Create this file by copying from `.env.template` (which you should create if it doesn't exist, see the `.env.template` content provided in the project preparation discussion) or by creating it manually:

```env
OPENAI_API_KEY="your_openai_api_key_here"
ANTHROPIC_API_KEY="your_anthropic_api_key_here"
# Add other keys as needed
```
The application (especially the demo and app components) will typically load these variables using a library like `python-dotenv`. Remember to add `.env` to your `.gitignore` file.

## Quick Start

```python
import os
from rag_ultra import process_document

# Set your API key (e.g., for OpenAI)
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Process a document and extract metadata
document_path = "path/to/your/document.pdf"
metadata = process_document(
    document_path=document_path,
    output_path="document_metadata.json",  # Optional: save to file
    context_length=3,  # Optional: number of previous pages to use as context
)

# Print some basic information
print(f"Document title: {metadata['document_details']['title']}")
print(f"Total pages: {metadata['document_details']['total_pages']}")
print(f"Short summary: {metadata['document_summary']['short_summary']}")
```

## Using the Smarter Retriever Agent

```python
from rag_ultra.utils import load_metadata_from_file
from rag_ultra.retriever_agent import retrieve_agent

# Load the metadata (can be single or multi-document)
metadata_dict = load_metadata_from_file("document_metadata.json")

# Ask a question (multi-document supported)
query = "What are the key findings in the Ford and Mercedes reports?"
results = retrieve_agent(
    prompt=query,
    metadata_dict=metadata_dict,
    model="openai/gpt-4o",  # or any supported model
    api_key=os.environ["OPENAI_API_KEY"]
)

print(f"Query: {query}")
print(f"Answer: {results['answer']}")
print(f"Cited pages: {results['relevant_pages']}")  # List of {{'document': ..., 'page': ...}}
print(f"Agent reasoning: {results['reasoning_trace']}")
```

## Advanced Retrieval & Agent Reasoning

- The retriever agent uses an LLM to select the minimal set of metadata fields needed to answer each question, reducing latency and cost.
- Only the selected fields are extracted and sent to the LLM for answer generation.
- If the metadata is insufficient, the agent will fetch the raw text for the relevant pages and use that for the final answer.
- All citations are returned as a list of `{document, page}` objects, supporting robust multi-document referencing.
- The agent reasoning trace explains how the answer was derived and which pages were considered.
- LLM calls automatically use the optimal max_tokens for the selected model.

## Metadata Structure

The generated metadata includes:

### Document Details
- Title
- Author
- Date
- Total number of pages
- Language
- Filename
- File type
- File size

### Page-Level Metadata
- One-sentence summary
- Full summary
- Topics and key concepts
- Keywords
- Acronyms dictionary
- Chapter or section name
- Noteworthy sentences
- Raw text
- Base64 image of the page (if image conversion is enabled)
- Token count

### Document Summary
- Short summary (max 3 sentences)
- Detailed summary
- Main topics
- Key insights
- Total token count

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 