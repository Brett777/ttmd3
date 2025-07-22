"""
Configuration settings for the RAG-Ultra SDK.
"""

import os
from typing import Dict, Any

# Default LLM settings
DEFAULT_MODEL = "openai/gpt-4.1-mini"  # Using LiteLLM format: provider/model
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4000
DEFAULT_CONTEXT_LENGTH = 3  # Number of previous pages to include as context

# API configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
AZURE_API_KEY = os.environ.get("AZURE_API_KEY", "")
AZURE_API_BASE = os.environ.get("AZURE_API_BASE", "")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION", "")
TOGETHERAI_API_KEY = os.environ.get("TOGETHERAI_API_KEY", "")
CLARIFAI_API_KEY = os.environ.get("CLARIFAI_API_KEY", "")

# Context window sizes for different models
MODEL_CONTEXT_WINDOWS: Dict[str, int] = {
    # OpenAI models
    "openai/gpt-3.5-turbo": 16385,
    "openai/gpt-4": 8192,
    "openai/gpt-4-turbo": 128000,
    "openai/gpt-4o": 128000,
    
    # Anthropic models
    "anthropic/claude-3-opus-20240229": 200000,
    "anthropic/claude-3-sonnet-20240229": 200000,
    "anthropic/claude-3-haiku-20240307": 200000,
    
    # Google models
    "vertex_ai/gemini-1.5-pro": 1000000,
    "vertex_ai/gemini-1.5-flash": 1000000,
    
    # Together AI models
    "together_ai/togethercomputer/llama-2-70b-chat": 4096,
    "together_ai/togethercomputer/qwen-72b-chat": 8192,
    
    # Mistral models
    "mistral/mistral-large-latest": 32768,
    "mistral/mistral-medium-latest": 32768,
    "mistral/mistral-small-latest": 32768,
}

# Default retriever settings
DEFAULT_RETRIEVER_CONFIG: Dict[str, Any] = {
    "model": DEFAULT_MODEL,
    "temperature": DEFAULT_TEMPERATURE,
    "max_tokens": 1000,
}

# Metadata extraction settings
DEFAULT_EXTRACTION_CONFIG: Dict[str, Any] = {
    "model": DEFAULT_MODEL,
    "temperature": DEFAULT_TEMPERATURE,
    "max_tokens": DEFAULT_MAX_TOKENS,
}

# File type settings
SUPPORTED_FILE_TYPES = ["pdf", "docx", "pptx", "txt"]

# Model provider mappings for easy reference
MODEL_PROVIDERS = {
    "openai": {
        "env_var": "OPENAI_API_KEY",
        "models": ["gpt-4o", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
    },
    "anthropic": {
        "env_var": "ANTHROPIC_API_KEY",
        "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
    },
    "google": {
        "env_var": "GOOGLE_API_KEY",
        "models": ["gemini-1.5-pro", "gemini-1.5-flash"]
    },
    "vertex_ai": {
        "env_var": "GOOGLE_API_KEY",
        "models": ["gemini-1.5-pro", "gemini-1.5-flash"]
    },
    "azure": {
        "env_var": "AZURE_API_KEY",
        "models": ["gpt-4o", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
    },
    "together_ai": {
        "env_var": "TOGETHERAI_API_KEY",
        "models": ["togethercomputer/llama-2-70b-chat", "togethercomputer/qwen-72b-chat"]
    },
    "clarifai": {
        "env_var": "CLARIFAI_API_KEY",
        "models": ["mistralai.completion.mistral-large"]
    },
}

# Get model provider from model name (e.g., "openai/gpt-4o" -> "openai")
def get_model_provider(model: str) -> str:
    """Get the provider from a full model name."""
    if "/" in model:
        return model.split("/")[0]
    return "openai"  # Default to OpenAI if no provider specified
