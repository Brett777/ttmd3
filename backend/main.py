"""
TTMD3 Backend - AI Chat Service with Multi-Modal Capabilities

This FastAPI application serves as the backend for an AI chat service that provides:
1. Multi-model LLM chat with tool calling capabilities
2. Document analysis using RAG (Retrieval Augmented Generation)  
3. Structured data analysis with SQL generation
4. Real-time streaming responses
5. Session-based conversation management

Key Features:
- Support for multiple AI providers (OpenAI, Anthropic, Google, xAI, etc.)
- Tool calling system for external integrations (weather, stocks, web search)
- Document upload and processing with RAG-Ultra
- Data analysis with automatic SQL generation from natural language
- DataRobot Agent integration
- Real-time progress tracking for document processing
- Session-based state management

Architecture Overview:
- FastAPI backend with CORS middleware for web frontend
- Async/await patterns for concurrent processing
- In-memory session storage for scalability
- Tool calling system with parallel execution
- Streaming responses for real-time user experience
- Modular design with separate processors for different data types

Main Components:
1. Tool Functions: External API integrations (weather, stocks, search, etc.)
2. Chat Engine: LLM orchestration with tool calling
3. Document Processor: RAG-based document analysis
4. Data Analyzer: SQL generation for structured data queries
5. Session Manager: State persistence across requests
6. Progress Tracker: Real-time job status updates
"""

import os
import sys
import time
import openai

# Add the current directory to Python path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# FastAPI and web framework imports
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse

# Data validation and type hints
from pydantic import BaseModel, Field
from typing import List, Dict, Any, AsyncGenerator, Optional

# DataRobot integration for runtime parameters
from datarobot_drum import RuntimeParameters
from datarobot_drum.runtime_parameters.exceptions import InvalidJsonException

# External service clients
import httpx          # Async HTTP client for external APIs
import yfinance as yf # Yahoo Finance for stock data
import litellm       # Multi-provider LLM interface

# Utility libraries
import uuid
from dotenv import load_dotenv
import asyncio
import json
import tempfile
import aiofiles
from pathlib import Path
import logging

# Local modules for document processing and RAG
from document_processor import get_or_create_processor, ProcessingJob, DocumentProcessor
from rag_ultra import process_document
from rag_ultra.retriever_agent import retrieve_agent

# Database connectivity
import snowflake.connector
from snowflake.connector import DictCursor

# =============================================================================
# DATA MODELS AND REQUEST/RESPONSE SCHEMAS
# =============================================================================

# Models for "Talk to my data" functionality
class GenerateVizRequest(BaseModel):
    """Request model for generating Chart.js visualizations from data."""
    prompt: str
    sql: str
    data_sample: List[Dict[str, Any]]

class GenerateVizResponse(BaseModel):
    """Response model containing Chart.js configuration."""
    chart_config: Dict[str, Any]

class DbConnectionRequest(BaseModel):
    """Request model for testing database connections."""
    connection_type: str # Currently only "snowflake" is supported
    params: Dict[str, str]

class DbSchemaRequest(BaseModel):
    """Request model for fetching database schema information."""
    connection_type: str
    params: Dict[str, str]
    
class ExecuteSqlRequest(BaseModel):
    """Request model for executing SQL queries against connected databases."""
    connection_type: str
    params: Dict[str, str]
    sql: str

class RegisterSchemaRequest(BaseModel):
    """Request model for registering structured data schema for a session."""
    session_id: str
    schema_data: Dict[str, Any]

class Document(BaseModel):
    """Document metadata model."""
    id: str
    filename: str

class UnregisterSchemaRequest(BaseModel):
    session_id: str
    table_name: str

# =============================================================================
# GLOBAL SESSION STORAGE
# =============================================================================
# Note: In production, these would be replaced with Redis or another
# distributed cache for multi-instance deployments

# Document processors by session - handles RAG document analysis
SESSION_PROCESSORS: Dict[str, DocumentProcessor] = {}

# Cached document metadata by session - for quick access without reprocessing
SESSION_METADATA_CACHE: Dict[str, Dict[str, Any]] = {}

# Structured data schemas by session - for SQL generation and data analysis
SESSION_DATA_SCHEMAS: Dict[str, Dict[str, Any]] = {}

# Real-time progress tracking for long-running operations
SESSION_PROGRESS: Dict[str, Dict[str, Dict[str, Any]]] = {}
# Structure: SESSION_PROGRESS[session_id][job_id] = {status, stage, current, total, logs}

# =============================================================================
# PROGRESS TRACKING UTILITIES
# =============================================================================

def update_session_progress(session_id: str, job_id: str, status: str, 
                          stage: str = "", current: int = 0, total: int = 0, 
                          logs: list = None, error: str = None):
    """
    Update progress for a job in the session progress store.
    
    This function is used by document processors to report real-time progress
    back to the frontend for display to users.
    
    Args:
        session_id: Unique identifier for the user session
        job_id: Unique identifier for the processing job
        status: Current job status (queued, processing, completed, failed)
        stage: Human-readable description of current processing stage
        current: Current progress count
        total: Total items to process
        logs: List of log messages
        error: Error message if status is failed
    """
    if session_id not in SESSION_PROGRESS:
        SESSION_PROGRESS[session_id] = {}
    
    SESSION_PROGRESS[session_id][job_id] = {
        "status": status,
        "stage": stage,
        "current": current,
        "total": total,
        "logs": logs or [],
        "error": error,
        "timestamp": time.time()
    }
    
    # Debug logging to track progress updates
    print(f"[PROGRESS UPDATE] Session {session_id[:8]}, Job {job_id[:8]}: {status} - {stage} ({current}/{total})")
    print(f"[PROGRESS UPDATE] Total jobs in session: {len(SESSION_PROGRESS[session_id])}")


def remove_session_progress(session_id: str, job_id: str):
    """
    Remove a job from session progress tracking.
    
    Called when a job is completed, deleted, or otherwise no longer needs tracking.
    Helps prevent memory leaks from accumulating old job data.
    """
    if session_id in SESSION_PROGRESS and job_id in SESSION_PROGRESS[session_id]:
        del SESSION_PROGRESS[session_id][job_id]
        
        # Clean up empty session containers to prevent memory leaks
        if not SESSION_PROGRESS[session_id]:
            del SESSION_PROGRESS[session_id]

# =============================================================================
# ENVIRONMENT VARIABLE CONFIGURATION
# =============================================================================

# Load environment variables in local development
load_dotenv()

# If not running locally, try to load API keys from DataRobot RuntimeParameters
# This allows the application to run in DataRobot's managed environment
try:
    # List all expected API keys (should match model-metadata.yaml)
    DR_API_KEYS = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY", 
        "GEMINI_API_KEY",
        "XAI_API_KEY",
        "COHERE_API_KEY",
        "DEEPSEEK_API_KEY",
        "PERPLEXITYAI_API_KEY",
        "DATAROBOT_API_TOKEN",
    ]
    
    # For each expected key, try to load from DataRobot runtime if not already set
    for key in DR_API_KEYS:
        if not os.environ.get(key):
            val = RuntimeParameters.get(key)
            if val:
                # Handle different formats that DataRobot might return
                if isinstance(val, dict):
                    if 'credential' in val:
                        os.environ[key] = val['credential']
                    elif 'apiToken' in val:
                        os.environ[key] = val['apiToken']
                    elif 'token' in val:
                        os.environ[key] = val['token']
                    elif len(val) == 1:
                        # Fallback to taking the only value if it's a single-key dict
                        os.environ[key] = list(val.values())[0]
                    else:
                        print(f"Warning: Runtime parameter for {key} is a complex dictionary. Coercing to string: {val}")
                        os.environ[key] = str(val)
                else:
                    os.environ[key] = str(val) # Ensure it's a string
except ImportError:
    # DataRobot runtime not available - running in local development
    pass

# =============================================================================
# TOOL FUNCTIONS - External API Integrations
# =============================================================================
# These functions implement the actual logic for various tools that the AI
# can call to gather information or perform actions.


async def get_weather(location: str) -> str:
    """Return current weather for a location using Nominatim + Open-Meteo."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # 1) Geocode location → lat/lon
            geo_resp = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": location, "format": "json", "limit": 1},
                headers={"User-Agent": "agent-demo"},
            )
            geo_resp.raise_for_status()
            geo_data = geo_resp.json()
            if not geo_data:
                return f"I couldn't find the location '{location}'."
            lat = geo_data[0]["lat"]
            lon = geo_data[0]["lon"]

            # 2) Fetch weather
            weather_resp = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current_weather": True,
                    "hourly": "temperature_2m",
                    "timezone": "auto",
                },
            )
            weather_resp.raise_for_status()
            weather = weather_resp.json().get("current_weather", {})
            if not weather:
                return "Weather data currently unavailable."
            return (
                f"The current temperature in {location.title()} is {weather['temperature']}°C with "
                f"wind speed {weather['windspeed']} km/h."
            )
    except Exception as e:
        print(f"Weather tool error for location '{location}': {e}")
        return "I'm unable to fetch the weather right now."


async def get_stock_price(ticker: str) -> str:
    """Return latest stock price using yfinance."""
    try:
        # yfinance is not async, so we run it in a thread to avoid blocking
        stock = await asyncio.to_thread(yf.Ticker, ticker)
        # .info is more reliable than .fast_info
        info = await asyncio.to_thread(lambda: stock.info)
        
        # Check for valid info and a key metric
        if not info or info.get("regularMarketPrice") is None:
            return f"Data for ticker '{ticker}' could not be found. It may be an invalid symbol."

        price = info["regularMarketPrice"]
        # Use previousClose, fallback to the current price if unavailable
        pc = info.get("previousClose", price) 
        change = price - pc
        pct = (change / pc * 100) if pc else 0
        arrow = "📈" if change >= 0 else "📉"
        
        company_name = info.get('longName', ticker.upper())

        return (
            f"{arrow} {company_name} ({ticker.upper()}) is trading at ${price:.2f} "
            f"({change:+.2f}, {pct:+.2f}%)."
        )
    except Exception as e:
        # Log the specific error for debugging
        print(f"yfinance error for ticker '{ticker}': {e}")
        return f"I'm unable to retrieve stock information for '{ticker}' right now. The service may be temporarily down or the ticker is invalid."


async def search_the_web(query: str) -> str:
    """Searches the web for up-to-date information."""
    try:
        print(f"Performing web search for: {query}")
        # This is a meta-tool; it uses a specialized LLM as the implementation.
        completion = await litellm.acompletion(
            model="openai/gpt-4o-search-preview",
            messages=[{"role": "user", "content": query}],
            # We can add a system prompt to focus the model, but it's very capable as-is.
        )
        response_content = completion.choices[0].message.content
        return response_content or "No content returned from web search."
    except Exception as e:
        print(f"Web search tool error: {e}")
        return "Sorry, I'm unable to perform a web search right now."

async def search_with_perplexity(query: str) -> str:
    """Performs comprehensive research and analysis using Perplexity AI."""
    try:
        print(f"Performing Perplexity research for: {query}")
        # Use Perplexity's sonar-pro model for comprehensive research
        completion = await litellm.acompletion(
            #model="perplexity/sonar-pro",
            model="perplexity/sonar",
            messages=[{"role": "user", "content": query}],
        )
        response_content = completion.choices[0].message.content
        return response_content or "No content returned from Perplexity search."
    except Exception as e:
        print(f"Perplexity search tool error: {e}")
        return "Sorry, I'm unable to perform a Perplexity search right now."

async def current_date_and_time() -> str:
    """Returns the current date and time in New York timezone."""
    try:
        from datetime import datetime
        import pytz
        
        # Get current time in New York timezone
        ny_tz = pytz.timezone('America/New_York')
        current_time = datetime.now(ny_tz)
        
        # Format the date and time in a readable format
        formatted_time = current_time.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")
        
        print(f"Current date and time requested: {formatted_time}")
        return f"The current date and time in New York is: {formatted_time}"
    except Exception as e:
        print(f"Date/time tool error: {e}")
        return "Sorry, I'm unable to get the current date and time right now."


async def analyze_documents(query: str) -> str:
    """
    Analyzes uploaded documents to answer questions.
    This function acts as a wrapper for the RAG-Ultra retriever agent.
    It relies on context (_current_session_id, _current_model, _current_messages)
    being injected into its attributes before being called.
    """
    try:
        # These attributes are injected by the /api/chat endpoint before calling
        session_id = analyze_documents._current_session_id
        model = analyze_documents._current_model
        messages = analyze_documents._current_messages
    except AttributeError as e:
        print(f"analyze_documents is missing required context: {e}")
        return json.dumps({"answer": "Internal error: analyze_documents tool is missing context.", "citations": [], "documents_analyzed": []})

    # Get the document metadata for the current session
    document_metadata = SESSION_METADATA_CACHE.get(session_id, {})
    if not document_metadata:
        return json.dumps({
            "answer": "You haven't uploaded any documents yet. Please upload documents before asking questions about them.",
            "citations": [],
            "documents_analyzed": []
        })

    # Call the RAG-Ultra retrieve_agent
    try:
        result = await retrieve_agent(
            query=query,
            document_metadata=document_metadata,
            model=model,
            messages=messages,
            session_id=session_id
        )
        # The result from retrieve_agent should be a dictionary. We'll return it as a JSON string.
        return json.dumps(result)
    except Exception as e:
        print(f"Error calling retrieve_agent: {e}")
        logging.exception("Error details for retrieve_agent call:")
        return json.dumps({
            "answer": "Sorry, I encountered an error while analyzing your documents.",
            "citations": [],
            "documents_analyzed": []
        })


async def _generate_sql_from_prompt(
    prompt: str, 
    schema: Dict[str, Any],
    previous_sql: Optional[str] = None,
    error_message: Optional[str] = None
) -> Dict[str, Any]:
    """Helper function to generate SQL from a prompt and schema, using rich metadata."""
    logging.info(f"Generating SQL for prompt: {prompt[:100]}...")
    logging.info(f"Schema keys: {list(schema.keys())}")
    
    table_list_str = "[]" # Default value
    try:
        # Build schema/sample context with enriched statistics
        context_parts = []
        
        # This list comprehension must be inside the try block
        available_table_names = [f'"{t.get("name", table_name)}"' for table_name, t in schema.items()]
        table_list_str = f"[{', '.join(available_table_names)}]"

        for table_name, t in schema.items():
            logging.info(f"Processing table: {table_name}")
            
            # Handle different schema formats - new format vs legacy format
            if isinstance(t, list):
                # Legacy format from frontend fallback: [{name, type}, ...]
                logging.info(f"  Using legacy schema format for {table_name}")
                table_part = f"Table: \"{table_name}\"\n"
                column_parts = []
                for c in t:
                    column_name = c.get("name", "unknown")
                    column_type = c.get("type", "unknown")
                    column_parts.append(f"  - \"{column_name}\" ({column_type})")
                    logging.info(f"  Processing column: {column_name}")
            else:
                # New format from schema registration: {name, rowCount, columns, sampleRows}
                row_count = t.get("rowCount", "N/A")
                table_part = f"Table: \"{t.get('name', table_name)}\" (Total Rows: {row_count})\n"
                
                column_parts = []
                for c in t.get("columns", []):
                    logging.info(f"  Processing column: {c.get('name')}")
                    stats_str = ""
                    stats = c.get("stats")
                    if isinstance(stats, dict):
                        min_val, max_val, avg_val = stats.get("min"), stats.get("max"), stats.get("avg")
                        
                        try:
                            # Attempt to convert to float, as values might be strings
                            if all(v is not None for v in [min_val, max_val, avg_val]):
                                min_f, max_f, avg_f = float(min_val), float(max_val), float(avg_val)
                                stats_str = f" (min: {min_f:.2f}, max: {max_f:.2f}, avg: {avg_f:.2f})"
                        except (ValueError, TypeError) as e:
                            logging.warning(f"  Numeric stats conversion failed for {c.get('name')}: {e}")
        
                        # If numeric stats weren't successfully processed, try frequent_values
                        if not stats_str and "frequent_values" in stats and isinstance(stats.get("frequent_values"), list):
                            value_parts = []
                            for item in stats["frequent_values"]:
                                if isinstance(item, dict):
                                    val = item.get("value", "N/A")
                                    count = item.get("count", "N/A")
                                    value_parts.append(f"'{val}' ({count})")
                            if value_parts:
                                top_vals = ", ".join(value_parts)
                                stats_str = f" (Top values: {top_vals})"
        
                    column_name = c.get("name", "N/A")
                    column_type = c.get("type", "N/A")
                    column_parts.append(f"  - \"{column_name}\" ({column_type}){stats_str}")
            
            # Common logic for both formats
            table_part += "Columns:\n" + "\n".join(column_parts)
            
            # Only process sample rows for new format (not legacy list format)
            if not isinstance(t, list):
                sample_rows = t.get("sampleRows", [])
                if sample_rows:
                    try:
                        # Safely convert sample rows to a string format, handling lists of dicts
                        header = list(sample_rows[0].keys()) if isinstance(sample_rows[0], dict) else None
                        rows_str_parts = []
                        if header:
                            rows_str_parts.append(", ".join(map(str, header)))
                        for row in sample_rows:
                            if isinstance(row, dict):
                                rows_str_parts.append(", ".join(map(str, row.values())))
                            else:
                                rows_str_parts.append(str(row)) # Fallback for other formats
                        
                        sample_rows_str = "\n".join(rows_str_parts)
                        table_part += f"\nSample Data:\n{sample_rows_str}"
                    except Exception as e:
                        logging.error(f"Error processing sample rows for {table_name}: {e}")
                        table_part += "\nSample Data: [Error processing sample rows]"
            
            context_parts.append(table_part)
        
        context = "\n\n".join(context_parts)
        logging.info(f"Context built successfully (length: {len(context)})")
    except Exception as e:
        logging.error(f"Error building context: {e}")
        context = "[Error building schema context]"
        print(f"SQL Generation Error: {str(e)} - Prompt: {prompt[:100]}...")
        raise

    system_prompt = f"""You are a DuckDB SQL expert. Your PRIMARY GOAL is to generate SQL queries that AGGREGATE, SUMMARIZE, and ANALYZE data to produce meaningful INSIGHTS. Focus on trends, summaries, and key metrics rather than dumping raw data.

**Prompt Structure Overview**: Role/Goal → Output Format → Critical Rules → Guidelines → Rules/Constraints → Common Pitfalls → Schema.

**Output Format** (Strict)
Return ONLY a JSON object with EXACTLY these keys:
- "sql": A single, executable DuckDB SQL query string.
- "explanation": Plain English description for non-technical users, including exact table name(s) used.

NEVER add extra text, introductions (e.g., 'Here is the JSON:'), or code outside the JSON. This prevents parsing errors.

**Critical Rules** (Must Follow)
- Use ONLY exact table names from 'Available Tables' below. Do NOT invent or modify them—this causes 'table not found' errors.
- Tables are pre-loaded in DuckDB. Query DIRECTLY (e.g., FROM "sales.csv"). NEVER use read_csv_auto() or similar—this is invalid here.
- Example: Prompt: 'show me sales data'. Available: ["Superstore-Transactions.csv"]. Correct: FROM "Superstore-Transactions.csv". Wrong: FROM read_csv_auto('sales.csv').
- Available Table Names: {table_list_str}

**Query Guidelines**
Create queries that answer the prompt effectively for visualization. Keep results small and insightful: use aggregations based on metadata (e.g., stats, samples) to summarize—avoid raw data dumps.

Example (Good): For 'average sales by region', use AVG("sales") GROUP BY "region" (summarizes trends).
Example (Bad): SELECT * FROM "table" LIMIT 100 (dumps data without insight).

**DuckDB SQL Rules and Constraints** (Follow EXACTLY to Avoid Errors)

- **Environment**: In-memory database. Do NOT access filesystem (except explicit file queries below)—this fails in runtime.

- **Identifiers and Quoting**: Unquoted = case-insensitive (e.g., column). Double quotes = case-sensitive/special chars (e.g., "Column Name"). NEVER single quotes (that's for strings) or backticks (invalid). Why? Prevents syntax errors like confusing identifiers with literals.

- **String Literals**: Single quotes only: 'value'. Double quotes are for identifiers.

- **Statements and Structure**: End with ';' for multi-statements (single here). CTEs/subqueries MUST be aliased (e.g., WITH t AS (...) SELECT * FROM t)—unaliased causes errors. Use multi-line formatting for clarity.

- **Parameters**: Positional '?' only. No named params ($name)—unsupported.

- **Functions and Features**: Joins/windows/CTEs/analytics/PIVOT supported. Dates: strftime("column", '%Y-%m')—double-quote column, never single-quote or TO_CHAR (invalid). Nulls: IS NULL/NOT NULL. Sampling: SAMPLE 10%. Time Travel: AT (TIMESTAMP => '2024-01-01'). Files: FROM 'file.csv' (avoid; use pre-loaded tables).

- **UNION ALL**: Match column count/types exactly. Pad with NULL to align—this prevents mismatch errors.

- **Joins and Aliasing**: Explicit JOIN ... ON only. ALWAYS alias tables/subqueries (e.g., "table" AS t1). Qualify columns (t1."column") in multi-table queries—to avoid 'ambiguous reference' errors.

- **Conversions**: TRY_CAST for safe casting (NULL on failure). For financial data: Clean negatives/commas with REPLACE(TRIM("column"), ...) before TRY_CAST AS DOUBLE—handles formats like '(1,234.56)' or '$123.45'.

- **Best Practices**: Prefer lowercase unquoted identifiers. ALWAYS alias CTEs/subqueries. Avoid other dialects (no TOP, LIMIT ALL, backticks)—they fail in DuckDB.

**Common Pitfalls** (Avoid These)
- Pitfall: Single-quoting identifiers (e.g., FROM 'table' → treats as string, not table). Fix: Use "table".
- Pitfall: Unaliased subquery (e.g., FROM (SELECT ...)). Fix: Add AS sub.
- Pitfall: Broad queries (e.g., SELECT *). Fix: Aggregate using metadata (e.g., AVG based on min/max stats).
- Pitfall: Inventing tables (e.g., FROM "sales" when actual is "sales.csv"). Fix: Use exact names from list.

**Available Schema and Metadata**
Use this to inform queries: tables, schemas, row counts, columns, stats (min/max/avg/frequent values), and samples.
{context}"""

    # New retry augmentation
    if previous_sql and error_message:
        retry_section = f"\n\n**Retry Instructions**\nPrevious SQL Attempt: {previous_sql}\nError/Issue: {error_message}\nCorrect the SQL to fix this issue and ensure it returns meaningful results. Do not repeat the same error."
        system_prompt += retry_section
    
    try:
        completion = await litellm.acompletion(
            model="openai/gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        response_content = completion.choices[0].message.content
        logging.info(f"Generated SQL response: {response_content}")
        try:
            result = json.loads(response_content)
            sql = result.get('sql', '')
            explanation = result.get('explanation', '')
            if not sql:
                logging.warning("Generated response missing 'sql' key")
            if not explanation:
                logging.warning("Generated response missing 'explanation' key")
            logging.info(f"Generated SQL: {sql}")
            logging.info(f"Explanation: {explanation}")
            return result
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse generated response as JSON: {e}")
            raise
    except Exception as e:
        logging.error(f"Error during SQL generation: {e}")
        raise # Re-raise for callers to handle


# This will be called by the agent when it needs to generate a SQL query.
async def talk_to_data(query: str) -> Dict[str, Any]:
    """
    Generates a SQL query and a natural-language explanation based on a user's question and the provided table schema.
    This tool is called when the user asks a question about their connected data. The schema is retrieved from a session cache.
    The frontend will execute the SQL and generate visualizations.
    Args:
        query: The user's natural-language question.
    Returns:
        A dictionary containing the generated 'sql' and a natural-language 'explanation'.
    """
    try:
        # This attribute is injected by the /api/chat endpoint before calling
        session_id = talk_to_data._current_session_id
    except AttributeError:
        return {
            "sql": "SELECT 'Error: Missing session context'",
            "explanation": "I couldn't find the session context needed to retrieve your data's schema."
        }

    schema = SESSION_DATA_SCHEMAS.get(session_id)
    if not schema:
        return {
            "sql": "SELECT 'Error: Schema not found'",
            "explanation": "I could not find the schema for your data. Please ensure you have uploaded and registered your data correctly."
        }
        
    try:
        return await _generate_sql_from_prompt(prompt=query, schema=schema)
    except Exception as e:
        # For the agent tool, we return a JSON response indicating failure.
        return {
            "sql": f"SELECT 'Error generating SQL: {str(e)}'",
            "explanation": "I was unable to generate a SQL query to answer your question. Please try again."
        }


# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================

app = FastAPI(title="Agentic Chat API")

# Middleware to log request validation errors and request body for debugging
@app.middleware("http")
async def log_request_validation_errors(request: Request, call_next):
    """
    HTTP middleware that logs request validation errors and request bodies.
    
    This helps with debugging issues in production by capturing the exact
    request data that caused failures.
    """
    try:
        response = await call_next(request)
        return response
    except Exception as exc:
        # Log the request body for debugging
        body = await request.body()
        logging.error(f"Request body: {body.decode()}")
        logging.exception("Exception in request handling")
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc)},
        )

# CORS configuration - allows frontend to communicate with backend
origins = ["*"]  # Note: In production, this should be restricted to specific domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# DATA ANALYSIS ENDPOINTS ("Talk to My Data" Feature)
# =============================================================================
# These endpoints support structured data analysis with SQL generation
# and visualization capabilities for CSV files and databases.

@app.post("/api/data/generate-sql")
async def generate_sql_endpoint(request: dict):
    """
    Generate SQL from natural language prompt and schema data.
    
    This is a fallback endpoint for frontend compatibility that allows
    direct SQL generation without going through the chat interface.
    Used primarily for testing and direct data analysis workflows.
    
    Args:
        request: Dict containing 'prompt' (user question) and 'schema' (table metadata)
        
    Returns:
        Dict with 'sql' (generated query) and 'explanation' (human-readable description)
    """
    try:
        prompt = request.get("prompt", "")
        schema = request.get("schema", {})
        
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")
        if not schema:
            raise HTTPException(status_code=400, detail="schema is required")
            
        previous_sql = request.get("previous_sql")
        error_message = request.get("error_message")
        
        result = await _generate_sql_from_prompt(
            prompt=prompt, 
            schema=schema,
            previous_sql=previous_sql,
            error_message=error_message
        )
        return result
    except Exception as e:
        logging.error(f"Error in generate_sql_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate SQL: {str(e)}")

@app.post("/api/data/register-schema")
async def register_schema(request: RegisterSchemaRequest):
    """
    Register structured data schema for a session to enable SQL generation.
    
    This endpoint allows the frontend to register metadata about uploaded
    CSV files or connected databases so the AI can generate appropriate
    SQL queries. The schema includes table names, column definitions,
    statistical metadata, and sample data.
    
    Args:
        request: Contains session_id and schema_data with table metadata
        
    Returns:
        Success confirmation message
    """
    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id is required.")
    
    # Merge new schema data with existing schemas instead of replacing
    if request.session_id in SESSION_DATA_SCHEMAS:
        existing_schemas = SESSION_DATA_SCHEMAS[request.session_id]
        merged_schemas = {**existing_schemas, **request.schema_data}
        SESSION_DATA_SCHEMAS[request.session_id] = merged_schemas
        logging.info(f"Schema merged for session {request.session_id[:8]}. Previous tables: {list(existing_schemas.keys())}, New tables: {list(request.schema_data.keys())}, Total tables: {list(merged_schemas.keys())}")
    else:
        SESSION_DATA_SCHEMAS[request.session_id] = request.schema_data
        logging.info(f"Schema registered for session {request.session_id[:8]} with tables: {list(request.schema_data.keys())}")
    
    return {"status": "success", "message": "Schema registered successfully."}

@app.post('/api/data/unregister-schema')
async def unregister_schema(request: UnregisterSchemaRequest):
    if request.session_id not in SESSION_DATA_SCHEMAS:
        return {'status': 'success', 'message': 'No schema found for session, nothing to unregister.'}
    
    schemas = SESSION_DATA_SCHEMAS[request.session_id]
    if request.table_name in schemas:
        del schemas[request.table_name]
        logging.info(f'Unregistered table {request.table_name} from session {request.session_id[:8]}')
    
    if not schemas:
        del SESSION_DATA_SCHEMAS[request.session_id]
        logging.info(f'Removed empty schema entry for session {request.session_id[:8]}')
    
    return {'status': 'success', 'message': 'Schema unregistered successfully.'}

@app.post("/api/data/generate-visualizations", response_model=GenerateVizResponse)
async def generate_visualizations(request: GenerateVizRequest):
    """
    Generate Chart.js configuration for data visualization.
    
    This endpoint takes a user prompt, SQL query, and data sample, then uses
    an LLM to generate an appropriate Chart.js configuration for visualizing
    the data. The AI chooses the best chart type and styling based on the
    data characteristics and user intent.
    
    Args:
        request: Contains prompt, sql query, and data sample
        
    Returns:
        Chart.js configuration object ready for frontend rendering
    """
    system_prompt = f"""You are a data visualization expert. Your task is to generate a Chart.js configuration to visualize the given data based on a user's request.

You must follow these rules:
1.  The output must be a valid JSON object that represents a Chart.js configuration.
2.  The chart should be visually appealing and appropriate for the data and the user's prompt.
3.  Choose the best chart type (e.g., bar, line, pie, scatter) based on the data and the prompt.
4.  The configuration should include 'type', 'data' (with labels and datasets), and 'options'.
5.  Make sure colors are reasonable and labels are clear. Use the `chartjs-plugin-zoom` for panning and zooming options where appropriate (especially for time-series or scatter plots).

User's Request: {request.prompt}
SQL Query Used: {request.sql}
Data Sample:
{json.dumps(request.data_sample[:5], indent=2)}

Respond ONLY with the Chart.js JSON configuration.
"""
    try:
        completion = await litellm.acompletion(
            model="openai/gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate a chart for this request: {request.prompt}"}
            ],
            response_format={"type": "json_object"}
        )
        response_content = completion.choices[0].message.content
        chart_config = json.loads(response_content)
        return GenerateVizResponse(chart_config=chart_config)
    except Exception as e:
        logging.error(f"Error generating visualization: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate visualization: {str(e)}")


@app.post("/api/data/test-db-connection")
async def test_db_connection(request: DbConnectionRequest):
    """
    Test database connection validity.
    
    Currently supports only Snowflake connections. Validates that the
    provided connection parameters can establish a successful connection
    to the database.
    
    Args:
        request: Contains connection_type and connection parameters
        
    Returns:
        Success/failure status with message
    """
    if request.connection_type != "snowflake":
        raise HTTPException(status_code=400, detail="Only Snowflake connections are supported currently.")
    try:
        conn = snowflake.connector.connect(**request.params)
        conn.close()
        return {"status": "success", "message": "Connection successful."}
    except Exception as e:
        logging.error(f"Snowflake connection failed: {e}")
        raise HTTPException(status_code=400, detail=f"Connection failed: {str(e)}")


@app.post("/api/data/db-schema")
async def get_db_schema(request: DbSchemaRequest):
    """
    Fetch database schema information.
    
    Retrieves table and column metadata from the connected database.
    Currently supports only Snowflake. The schema information is used
    by the AI to generate appropriate SQL queries.
    
    Args:
        request: Contains connection_type and connection parameters
        
    Returns:
        Schema dictionary with table and column information
    """
    if request.connection_type != "snowflake":
        raise HTTPException(status_code=400, detail="Only Snowflake connections are supported currently.")
    try:
        conn = snowflake.connector.connect(**request.params)
        cursor = conn.cursor(DictCursor)
        # This is a simplified schema fetch. A production implementation would be more sophisticated
        cursor.execute("SHOW TERSE TABLES;")
        tables = [row['name'] for row in cursor.fetchall()]
        
        schema = {}
        for table in tables:
            cursor.execute(f'DESCRIBE TABLE "{table}";')
            columns = [{"name": col['name'], "type": col['type']} for col in cursor.fetchall()]
            schema[table] = columns

        cursor.close()
        conn.close()
        return {"schema": schema}
    except Exception as e:
        logging.error(f"Failed to fetch Snowflake schema: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch schema: {str(e)}")


@app.post("/api/data/execute-sql")
async def execute_sql(request: ExecuteSqlRequest):
    """
    Execute SQL query against connected database.
    
    Runs the provided SQL query against the database and returns results.
    Currently supports only Snowflake. Results are limited to 500 rows
    to prevent excessive memory usage and network transfer.
    
    Args:
        request: Contains connection_type, connection parameters, and SQL query
        
    Returns:
        Query results as list of dictionaries
    """
    if request.connection_type != "snowflake":
        raise HTTPException(status_code=400, detail="Only Snowflake connections are supported currently.")
    try:
        conn = snowflake.connector.connect(**request.params)
        cursor = conn.cursor(DictCursor)
        cursor.execute(request.sql)
        # Limit results to a reasonable number to avoid sending huge payloads
        results = cursor.fetchmany(500)
        cursor.close()
        conn.close()
        return {"results": results}
    except Exception as e:
        logging.error(f"Failed to execute Snowflake SQL: {e}")
        print(f"SQL Execution Error (Snowflake): {str(e)} - Query: {request.sql}")
        raise HTTPException(status_code=400, detail=f"Failed to execute SQL: {str(e)}")


# =============================================================================
# TOOL CALLING SYSTEM CONFIGURATION
# =============================================================================
# This section defines the available tools that AI models can call and their schemas

# Mapping of tool names to their implementation functions
# This dictionary connects the tool names used in the OpenAI function calling
# schema to the actual Python functions that implement the functionality
TOOLS_MAP = {
    "get_weather": get_weather,
    "get_stock_price": get_stock_price,
    "search_the_web": search_the_web,
    "search_with_perplexity": search_with_perplexity,
    "current_date_and_time": current_date_and_time,
    "analyze_documents": analyze_documents,
    "talk_to_data": talk_to_data,
}

# OpenAI function calling schema definitions for all available tools
# These schemas tell the AI models what tools are available, what parameters
# they expect, and when they should be used
TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given location.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City name, e.g., 'San Francisco'"}},
                "required": ["location"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string", "description": "Stock ticker, e.g., 'AAPL'"}},
                "required": ["ticker"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_the_web",
            "description": "Searches the web for up-to-date information on current events, people, or topics not covered by other available tools. Use this for quick, current information lookup.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "A focused, natural language query for the web search."}},
                "required": ["query"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_with_perplexity",
            "description": "Performs comprehensive research and in-depth analysis using Perplexity AI. Use this tool for complex queries requiring detailed investigation, academic research, market analysis, or when you need extensive background information with citations and sources.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "A comprehensive research query for in-depth analysis."}},
                "required": ["query"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "current_date_and_time",
            "description": "Get the current date and time in New York timezone. Use this when you need to know what time it is now, what day it is, or need current temporal context for your response. This is especially important when asked about current events, world events, or performing web or perplexity searches.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_documents",
            "description": "Analyze uploaded documents to answer questions using advanced RAG capabilities. Use this tool when the user asks questions about their uploaded documents, specific people mentioned in documents, report cards, grades, academic performance, or any content that might be found in their personal documents. This tool provides detailed answers with citations and reasoning traces.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The question or query about the documents"},
                },
                "required": ["query"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "talk_to_data",
            "description": "Analyzes data to answer questions. Use this tool whenever the user asks a question that can be answered by querying the structured data they have uploaded or connected. The tool will generate a SQL query to find the answer. It's important to use this tool anytime there is a questions relating to data analysis, even if it's a follow up question, and even if you think you konw the answer. If the answer could be found in the user's dataset, this tool will allow for a more accurate answer that it based on the most up to date user data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The user's natural-language question about the data."},
                },
                "required": ["query"],
            },
        }
    },
]

# =============================================================================
# AI MODEL DEFINITIONS
# =============================================================================
# Configuration for all supported AI models across different providers

AVAILABLE_MODELS = [
    # OpenAI Models - Industry leading LLMs with strong reasoning and tool use
    {"id": "gpt-4.1-mini", "name": "ChatGPT 4.1 mini", "provider": "OpenAI", "vision": True},
    {"id": "gpt-4.1-nano", "name": "ChatGPT 4.1 nano", "provider": "OpenAI", "vision": True},
    {"id": "gpt-4o-mini", "name": "ChatGPT 4o mini", "provider": "OpenAI", "vision": True},
    {"id": "gpt-4.1", "name": "ChatGPT 4.1", "provider": "OpenAI", "vision": True},
    {"id": "gpt-4o", "name": "ChatGPT 4o", "provider": "OpenAI", "vision": True},
    #{"id": "gpt-4o-search-preview", "name": "GPT-4o Search Preview", "provider": "OpenAI", "ui_visible": True, "vision": True},
    
    # Anthropic Claude Models - Excellent for analysis and writing
    {"id": "claude-3-5-haiku-latest", "name": "Claude 3.5 Haiku", "provider": "Anthropic", "vision": True},
    {"id": "claude-3-5-sonnet-latest", "name": "Claude 3.5 Sonnet", "provider": "Anthropic", "vision": True},
    {"id": "claude-3-7-sonnet-latest", "name": "Claude 3.7 Sonnet", "provider": "Anthropic", "vision": True},
    {"id": "claude-sonnet-4-20250514", "name": "Claude 4 Sonnet", "provider": "Anthropic", "vision": True},
    {"id": "claude-opus-4-20250514", "name": "Claude 4 Opus", "provider": "Anthropic", "vision": True},
    
    # Google Gemini Models - Strong multimodal capabilities
    {"id": "gemini-2.5-pro-preview-06-05", "name": "Gemini 2.5 Pro", "provider": "Gemini", "vision": True},
    {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro", "provider": "Gemini", "vision": True},
    {"id": "gemini-2.5-flash-preview-05-20", "name": "Gemini 2.5 Flash", "provider": "Gemini", "vision": True},
    {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash", "provider": "Gemini", "vision": True},
    {"id": "gemini-2.0-flash-lite", "name": "Gemini 2.0 Flash Lite", "provider": "Gemini", "vision": True},
    {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash", "provider": "Gemini", "vision": True},
    
    # xAI Grok Models - Fast and capable reasoning
    {"id": "grok-3", "name": "Grok 3", "provider": "xAI", "vision": True},
    
    # DataRobot Custom Agents - Specialized domain-specific agents
    # Note: These IDs are DataRobot deployment IDs for custom trained agents
    {"id": "6852c9c5fa0d6451099afd13", "name": "Brett's Custom Agent", "provider": "DataRobot Agent", "vision": False},
    {"id": "684aebd8f1302d795d443cb1", "name": "Luke's Custom Agent", "provider": "DataRobot Agent", "vision": False},
    {"id": "68432be4fb1547f3d344303e", "name": "Tim's Custom Agent", "provider": "DataRobot Agent", "vision": False},
    {"id": "685c4d781f0d599bbf45b47d", "name": "Connor's Custom Agent", "provider": "DataRobot Agent", "vision": False},
    {"id": "685c5da16e61da5f1a722c00", "name": "Gwyn's Custom Agent", "provider": "DataRobot Agent", "vision": False},
    {"id": "685c6dce1085e186e2722587", "name": "Gwyn's Custom Agent", "provider": "DataRobot Agent", "vision": False},
]



# =============================================================================
# REQUEST/RESPONSE MODELS FOR CHAT ENDPOINTS
# =============================================================================
# Pydantic models for API request/response validation and serialization

class HistoryMessage(BaseModel):
    """Individual message in conversation history."""
    role: str
    content: str

class ChatRequest(BaseModel):
    """Standard chat request for multi-provider LLM endpoints."""
    message: str
    conversation_history: List[HistoryMessage] = Field(default_factory=list)
    model: str = "gpt-3.5-turbo"
    session_id: str | None = None

class DataRobotChatRequest(BaseModel):
    """Chat request specifically for DataRobot Agent deployments."""
    message: str
    conversation_history: List[HistoryMessage] = Field(default_factory=list)
    model: str # This will be the DataRobot deployment ID
    session_id: str | None = None

class StartProcessingRequest(BaseModel):
    """Request to start processing a queued document."""
    session_id: str

class DocumentSyncRequest(BaseModel):
    """Request to sync document metadata from client to server."""
    session_id: str
    documents: List[Dict[str, Any]] = Field(default_factory=list)

class ChatResponse(BaseModel):
    """
    Legacy chat response model.
    
    Note: This model is no longer used for the main /chat endpoint response
    (which uses streaming), but the frontend might still use it for error formatting.
    """
    response: str
    model_used: str
    session_id: str

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_session(session_id: str | None) -> str:
    """
    Ensure a valid session ID exists.
    
    If no session ID is provided, generates a new UUID for the session.
    This allows stateless clients to maintain conversation context.
    
    Args:
        session_id: Optional existing session ID
        
    Returns:
        Valid session ID (either provided or newly generated)
    """
    return session_id or str(uuid.uuid4())

def sanitize_message_for_api(msg: dict) -> dict:
    """
    Sanitize a message for LiteLLM API calls to ensure proper format.
    
    Different LLM providers have varying requirements for message formatting.
    This function normalizes messages to a format that works across all providers
    while preserving important fields like tool calls and multimodal content.
    
    Args:
        msg: Raw message dictionary from conversation history
        
    Returns:
        Sanitized message dictionary safe for API calls
    """
    clean_msg = {"role": msg["role"]}
    
    # Handle content field properly for different content types
    content = msg.get("content")
    if content is not None:
        # If content is a string, use it directly
        if isinstance(content, str):
            clean_msg["content"] = content
        # If content is a list (like vision messages with images), use it directly
        elif isinstance(content, list):
            clean_msg["content"] = content
        # For any other type, convert to string
        else:
            clean_msg["content"] = str(content)
    else:
        # Only add content field if the original had it
        if "content" in msg:
            clean_msg["content"] = ""
    
    # Preserve tool call related fields for tool messages
    if msg["role"] == "tool":
        if "tool_call_id" in msg:
            clean_msg["tool_call_id"] = msg["tool_call_id"]
        if "name" in msg:
            clean_msg["name"] = msg["name"]
    
    # Preserve tool_calls for assistant messages if they exist
    if msg["role"] == "assistant" and "tool_calls" in msg:
        clean_msg["tool_calls"] = msg["tool_calls"]
    
    return clean_msg

# =============================================================================
# STREAMING RESPONSE HELPERS
# =============================================================================
# Functions for handling real-time streaming responses to the frontend

async def single_chunk_streamer(content: str, model: str, session_id: str) -> AsyncGenerator[str, None]:
    """
    Create a single-chunk stream for non-streaming responses.
    
    Used when we have a complete response that we want to send as a stream
    for consistency with the streaming interface.
    
    Args:
        content: Complete response content
        model: Model identifier
        session_id: Session identifier
        
    Yields:
        SSE-formatted data chunks
    """
    chunk = {"response": content, "model_used": model, "session_id": session_id}
    yield f"data: {json.dumps(chunk)}\n\n"
    await asyncio.sleep(0) # Yield control to the event loop

async def litellm_streamer(stream, model: str, session_id: str) -> AsyncGenerator[str, None]:
    """
    Convert LiteLLM stream to SSE format for frontend consumption.
    
    Takes the raw streaming response from LiteLLM and formats it as
    Server-Sent Events (SSE) that the frontend can consume in real-time.
    
    Args:
        stream: LiteLLM async stream
        model: Model identifier
        session_id: Session identifier
        
    Yields:
        SSE-formatted data chunks with incremental response content
    """
    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            res_chunk = {"response": content, "model_used": model, "session_id": session_id}
            yield f"data: {json.dumps(res_chunk)}\n\n"

async def enhanced_litellm_streamer(
    stream, model: str, session_id: str, 
    request_citations: list, 
    request_documents_used: list, 
    request_reasoning: str = "", 
    tools_used: list | None = None,
    request_metadata_fields: list = [],
    request_raw_content_used: bool = False,
    request_image_content_used: bool = False,
    data_analysis_result: dict | None = None
):
    """Enhanced version of litellm_streamer that can include citations and reasoning at the end."""
    async for chunk in stream:
        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content:
                chunk_data = {
                    "response": delta.content,
                    "model_used": model,
                    "session_id": session_id
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
    
    # After streaming is complete, send citations and reasoning if we have any
    if request_citations or request_documents_used or request_reasoning or (tools_used is not None) or data_analysis_result:
        final_chunk = {
            "response": "",  # Empty response, just metadata
            "model_used": model,
            "session_id": session_id,
            "citations": request_citations,
            "documentsUsed": request_documents_used,
            "usedMetadataFields": request_metadata_fields,
            "rawContentUsed": request_raw_content_used,
            "imageContentUsed": request_image_content_used
        }
        if request_reasoning:
            final_chunk["reasoning"] = request_reasoning
        if tools_used is not None:
            final_chunk["toolsUsed"] = tools_used
        if data_analysis_result is not None:
            final_chunk["data_analysis_result"] = data_analysis_result
        yield f"data: {json.dumps(final_chunk)}\n\n"

async def single_chunk_tool_response(
    content: str, model: str, session_id: str,
    citations: list, documents_used: list, reasoning: str, tools_used: list,
    used_metadata_fields: list, raw_content_used: bool, image_content_used: bool,
    data_analysis_result: dict | None = None
):
    """Creates a two-part SSE response for tool calls that have immediate content."""
    # First, yield the content from the model
    content_chunk = {
        "response": content,
        "model_used": model,
        "session_id": session_id
    }
    yield f"data: {json.dumps(content_chunk)}\n\n"

    # Then, yield the metadata in a separate chunk
    if citations or documents_used or reasoning or tools_used or data_analysis_result:
        metadata_chunk = {
            "response": "",
            "model_used": model,
            "session_id": session_id,
            "citations": citations,
            "documentsUsed": documents_used,
            "usedMetadataFields": used_metadata_fields,
            "rawContentUsed": raw_content_used,
            "imageContentUsed": image_content_used,
            "reasoning": reasoning,
            "toolsUsed": tools_used,
        }
        if data_analysis_result is not None:
            metadata_chunk["data_analysis_result"] = data_analysis_result
        yield f"data: {json.dumps(metadata_chunk)}\n\n"


# =============================================================================
# MAIN CHAT ENDPOINT
# =============================================================================
# The core endpoint that handles AI conversations with tool calling,
# multimodal input (text + images), and real-time streaming responses

@app.post("/api/chat")
async def chat(
    message: str = Form(...),
    conversation_history: str = Form("[]"),
    model: str = Form(...),
    session_id: str | None = Form(None),
    images: list[UploadFile] | None = File(None),
    has_data_analysis_history: str | None = Form(None),
):
    """
    Main chat endpoint supporting multi-modal AI conversations with tool calling.
    
    This endpoint orchestrates conversations with various AI models, providing:
    - Multi-provider LLM support (OpenAI, Anthropic, Google, xAI)
    - Tool calling for external integrations (weather, stocks, search, documents, data)
    - Multimodal input support (text + images)
    - Real-time streaming responses
    - Session-based conversation persistence
    - Automatic document and data analysis based on uploaded content
    
    The endpoint handles complex tool calling workflows where the AI can:
    1. Analyze uploaded documents using RAG
    2. Query structured data using SQL generation
    3. Search the web for current information
    4. Get weather and stock data
    5. Perform comprehensive research with citations
    
    Args:
        message: User's text message
        conversation_history: JSON string of previous messages
        model: AI model identifier
        session_id: Optional session identifier for persistence
        images: Optional uploaded images (max 3, 5MB each)
        
    Returns:
        StreamingResponse with real-time AI response chunks in SSE format
    """
    # Ensure we have a valid session ID for conversation tracking
    session_id = ensure_session(session_id)
    
    # Check if documents are available for this session (enables automatic document analysis)
    has_documents = session_id in SESSION_METADATA_CACHE and bool(SESSION_METADATA_CACHE[session_id])

    # Validate multimodal input constraints
    images = images or []
    if len(images) > 3:
        raise HTTPException(status_code=400, detail="Maximum 3 images allowed.")

    # Validate that the requested model exists and supports required features
    model_info = next((m for m in AVAILABLE_MODELS if m["id"] == model), None)
    if not model_info:
        raise HTTPException(status_code=400, detail="Invalid model selected.")

    # Ensure vision-capable model is selected when images are uploaded
    if images and not model_info.get("vision", False):
        raise HTTPException(status_code=400, detail="Selected model does not support vision input.")

    # Validate each image
    encoded_images_parts = []
    for img in images:
        if not img.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {img.filename}")
        contents = await img.read()
        if len(contents) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"{img.filename} exceeds 5 MB limit.")
        import base64, mimetypes
        mime = img.content_type or mimetypes.guess_type(img.filename)[0] or "image/png"
        data_url = f"data:{mime};base64,{base64.b64encode(contents).decode()}"
        encoded_images_parts.append({"type": "image_url", "image_url": {"url": data_url}})

    # Parse conversation history JSON string
    try:
        conv_history = json.loads(conversation_history)
    except json.JSONDecodeError:
        conv_history = []
    
    # Build system prompt with tool guidance
    system_content = "You are a helpful assistant with access to several tools. You must attempt to use one or more tools for your answer. If there is no approporiate tool, then do your best to answer based on your knowledge. Here are the tools you have access to:\n\n"
    system_content += "- For weather information: use get_weather\n"
    system_content += "- For stock prices: use get_stock_price\n"
    system_content += "- For current events or quick web search: use search_the_web\n"
    system_content += "- For comprehensive research and in-depth analysis: use search_with_perplexity\n"
    system_content += "- For current date and time: use current_date_and_time\n"
    system_content += "- For analysis of structured data, datasets, CSV files, or data analytics questions: use talk_to_data\n"
    
    if has_documents:
        print(f"Documents available for session {session_id} - enabling document analysis tool")
        system_content += "- The user might be asking about information from their documents. If that's the case: use analyze_documents\n\n"
        system_content += "IMPORTANT: The user has uploaded documents that are available for analysis. Their question likely pertains to these documents so you should probably use the analyze_documents tool to get accurate information from their documents, unless it's obvious that they're asking about something else."
    else:
        print(f"No documents available for session {session_id}")
    
    # Check if user has uploaded data
    has_data_schema = session_id in SESSION_DATA_SCHEMAS and bool(SESSION_DATA_SCHEMAS[session_id])
    if has_data_schema:
        data_tables = list(SESSION_DATA_SCHEMAS[session_id].keys())
        print(f"Data available for session {session_id} - enabling data analysis tool for tables: {data_tables}")
        print(f"Full schema data for session {session_id}: {SESSION_DATA_SCHEMAS[session_id]}")
        system_content += f"\n\nIMPORTANT: The user has uploaded structured data files that are available for analysis: {data_tables}. When they ask questions about data, statistics, counts, analysis, or anything related to these datasets, you MUST use the talk_to_data tool. The tool will generate SQL queries and return results that you should present clearly to the user.\n"
        system_content += "\nCRITICAL: For ANY question that could possibly relate to the user's data (even if it seems general or you think you know the answer from previous context), you MUST call talk_to_data. Do NOT attempt to generate SQL yourself, analyze data from memory, or skip the tool—this ensures the most accurate, up-to-date response. If in doubt, ALWAYS use the tool.\n"
        system_content += "Examples where you MUST use talk_to_data:\n"
        system_content += "- 'What are the total sales?' (direct data query)\n"
        system_content += "- 'Tell me more about the data' (general data reference)\n"
        system_content += "- Any question mentioning dataset terms like 'trends', 'averages', or table names\n"
        system_content += "Negative examples (DO NOT DO THIS):\n"
        system_content += "- Responding with made-up data or remembered info without calling the tool\n"
        system_content += "- Writing your own SQL instead of using the tool\n\n"
        
        # Add context about previous data analysis if this is a follow-up conversation
        if has_data_analysis_history == "true":
            print(f"Data analysis history detected for session {session_id} - enhancing system prompt for follow-up questions")
            system_content += f"\n\nCRITICAL CONTEXT: This conversation has previously involved data analysis using the talk_to_data tool. The user is likely asking follow-up questions about the same datasets ({data_tables}). Even if their question seems general or could be interpreted differently, if it relates to data, statistics, trends, comparisons, or analysis, you MUST use the talk_to_data tool to provide accurate, data-driven answers. ALWAYS call the tool again for follow-ups—do NOT rely on previous results or memory.\n"
            system_content += "Examples of follow-up questions that require talk_to_data:\n"
            system_content += "- 'What about the other categories?' (after showing some data)\n"
            system_content += "- 'Can you show me a different view?' (after previous analysis)\n"
            system_content += "- 'How does this compare to...?' (comparison questions)\n"
            system_content += "- 'What are the trends?' (trend analysis)\n"
            system_content += "- 'Show me more details' (after summary data)\n"
            system_content += "- Any question that could be answered with data from the available tables\n"
            system_content += "Negative examples (DO NOT DO THIS):\n"
            system_content += "- 'Based on previous data, I think...' (relying on memory instead of tool)\n"
            system_content += "- Generating your own analysis without re-calling the tool\n"
        else:
            print(f"No data analysis history detected for session {session_id}")    

    system_content += "CRITICAL: When the talk_to_data tool returns a result with 'sql' and 'explanation' fields, you MUST include both in your response. Use this exact format:\n"
    system_content += "```sql\n[THE SQL QUERY FROM THE TOOL]\n```\n\n[THE EXPLANATION FROM THE TOOL]\n\n" 
    system_content += "When the talk_to_data tool runs, you do not need to add any additional explanation to your response. Just say something like 'Here are the results' and then show the results."
    system_content += "Always use the most appropriate tool for each query. Do not make up information when tools are available. When you receive results from a tool, incorporate them fully into your response."

    system_prompt = {
        "role": "system", 
        "content": system_content
    }
    
    # Build message history for LLM
    messages = [system_prompt]
    messages.extend(conv_history)

    if encoded_images_parts:
        content_parts = [{"type": "text", "text": message}] + encoded_images_parts
        messages.append({"role": "user", "content": content_parts})
    else:
        messages.append({"role": "user", "content": message})

    # PHASE 3: Unified Tool-Calling Logic
    # Find the model details from the global list
    model_info = next((m for m in AVAILABLE_MODELS if m["id"] == model), None)

    if not model_info:
        raise HTTPException(status_code=400, detail="Invalid model selected.")

    # Construct the model name for litellm
    litellm_model_name = model
    # Prepend provider for non-OpenAI models
    if model_info["provider"].lower() != "openai":
        litellm_model_name = f"{model_info['provider'].lower()}/{model}"

    try:
        # Initialize tracking variables for iterative tool calling
        citations = []
        documents_used = []
        reasoning_trace = ""
        used_metadata_fields = []
        raw_content_used = False
        image_content_used = False
        all_tools_used = []
        data_analysis_payload = None  # To hold results from talk_to_data tool
        max_tool_rounds = 5  # Prevent infinite loops
        tool_round = 0
        
        last_assistant_content = ""  # Track the last assistant response
        
        while tool_round < max_tool_rounds:
            tool_round += 1
            print(f"Tool calling round {tool_round}")
            
            # Call LLM to see if it wants to use a tool
            # Sanitize messages for API call
            sanitized_messages = [sanitize_message_for_api(msg) for msg in messages]
            completion = await litellm.acompletion(
                model=litellm_model_name,
                messages=sanitized_messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                stream=False,
            )

            msg = completion.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))
            
            # Capture any content from the assistant
            if msg.content:
                last_assistant_content = msg.content

            # Check if the model requested tool calls
            if not (tool_calls := getattr(msg, "tool_calls", None)):
                # No more tools requested, break out of loop
                break
            
            # --- Parallel Tool Execution ---
            tool_coroutines = []
            for tool_call in tool_calls:
                fn = tool_call.function
                tool_callable = TOOLS_MAP.get(fn.name)
                if tool_callable is None:
                    continue

                # Parse arguments
                try:
                    arg_dict = json.loads(fn.arguments)
                except (json.JSONDecodeError, TypeError):
                    if isinstance(fn.arguments, dict):
                        arg_dict = fn.arguments
                    else:
                        print(f"Could not parse arguments for tool {fn.name}: {fn.arguments}")
                        arg_dict = {}

                # Inject necessary context for specific tools
                if fn.name == "analyze_documents":
                    analyze_documents._current_session_id = session_id
                    analyze_documents._current_model = model
                    
                    # Sanitize conversation history
                    sanitized_history = []
                    for msg_item in messages[:-1]:  # Exclude current assistant message with tool call
                        sanitized_history.append(sanitize_message_for_api(msg_item))
                        
                    analyze_documents._current_messages = sanitized_history
                    tool_coroutines.append(tool_callable(**arg_dict))
                elif fn.name == "talk_to_data":
                    # Inject the session_id so the tool can retrieve the schema
                    talk_to_data._current_session_id = session_id
                    tool_coroutines.append(tool_callable(**arg_dict))
                else:
                    tool_coroutines.append(tool_callable(**arg_dict))

            tool_results = await asyncio.gather(*tool_coroutines, return_exceptions=True)
            
            # Process tool results
            for i, tool_call in enumerate(tool_calls):
                result = tool_results[i]
                
                # Track tools used for frontend
                try:
                    arg_dict = json.loads(tool_call.function.arguments) if isinstance(tool_call.function.arguments, str) else tool_call.function.arguments
                except Exception:
                    arg_dict = {}
                all_tools_used.append({"name": tool_call.function.name, "arguments": arg_dict})
                
                # Special handling for analyze_documents
                if tool_call.function.name == "analyze_documents" and isinstance(result, str):
                    try:
                        doc_result = json.loads(result)
                        answer = doc_result.get("answer", str(result))
                        
                        # Accumulate metadata across all tool calls
                        citations.extend(doc_result.get("citations", []))
                        documents_used.extend(doc_result.get("documents_analyzed", []))
                        used_metadata_fields.extend(doc_result.get("used_metadata_fields", []))
                        raw_content_used |= doc_result.get("raw_content_used", False)
                        image_content_used |= doc_result.get("image_content_used", False)
                        
                        if not reasoning_trace:
                            reasoning_trace = doc_result.get("reasoning", "")
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": answer
                        })
                        
                    except json.JSONDecodeError:
                        print(f"Failed to parse analyze_documents result as JSON: {result[:200]}...")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": str(result)
                        })
                elif tool_call.function.name == "talk_to_data" and isinstance(result, dict):
                    # This is a data analysis result. Create a special payload for the frontend to execute.
                    sql_query = result.get('sql', '')
                    explanation = result.get('explanation', '')
                    
                    # Create a data analysis payload that signals the frontend to execute SQL and generate viz
                    data_analysis_payload = {
                        "sql": sql_query,
                        "explanation": explanation,
                        "requiresExecution": True,  # Flag to tell frontend to execute this
                        "data": [],  # Will be populated by frontend
                        "chartConfig": None  # Will be populated by frontend
                    }
                    
                    # Create a simplified response for the LLM
                    content_for_llm = f"I've analyzed your data request and generated a SQL query to answer your question. The query and results are displayed below."
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": content_for_llm
                    })
                    
                    # Inject a final system message to constrain the response
                    messages.append({
                        "role": "system",
                        "content": "You have received results from talk_to_data. Respond ONLY with: 'Here is my analysis:' followed by the exact SQL and explanation from the tool in the required format. Do not add any other content, analysis, or commentary."
                    })
                else:
                    # Regular tool result handling
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": str(result)
                    })

        # After all tool calling rounds, get final streaming response
        # Check if any tools were used
        if all_tools_used or citations or documents_used or reasoning_trace:
            # Tools were used, need to get final response with tool results
            
            # For Anthropic & Gemini models after tool execution, use the last assistant content
            # These providers can provide the response immediately after tool execution
            if model_info["provider"].lower() in ["anthropic", "gemini"] and last_assistant_content:
                provider_name = model_info["provider"]
                print(f"[DEBUG] Using last assistant content from tool calling loop for {provider_name}: '{last_assistant_content[:100]}...'")
                print(f"Tools were used: {all_tools_used}. Citations: {len(citations)}, Reasoning: {len(reasoning_trace)} chars")
                
                return StreamingResponse(
                    single_chunk_tool_response(
                        content=last_assistant_content, model=model, session_id=session_id,
                        citations=citations, documents_used=documents_used, reasoning=reasoning_trace,
                        tools_used=all_tools_used, used_metadata_fields=used_metadata_fields,
                        raw_content_used=raw_content_used, image_content_used=image_content_used,
                        data_analysis_result=data_analysis_payload
                    ),
                    media_type="text/event-stream"
                )
            
            # For other models, use streaming to get the final response
            # Sanitize messages for API call
            sanitized_messages = [sanitize_message_for_api(msg) for msg in messages]
            acompletion_params = {
                "model": litellm_model_name,
                "messages": sanitized_messages,
                "stream": True,
            }
            
            final_completion_stream = await litellm.acompletion(**acompletion_params)
            
            print(f"Tools were used: {all_tools_used}. Citations: {len(citations)}, Reasoning: {len(reasoning_trace)} chars")
            return StreamingResponse(
                enhanced_litellm_streamer(
                    final_completion_stream, model, session_id, 
                    citations, documents_used, reasoning_trace, all_tools_used,
                    used_metadata_fields, raw_content_used, image_content_used,
                    data_analysis_result=data_analysis_payload
                ),
                media_type="text/event-stream"
            )
        else:
            # No tools were used, use the last message from the loop
            # If we never entered the loop, we need to make a fresh call
            if tool_round == 0:
                # Never entered tool calling loop, make a regular call
                sanitized_messages = [sanitize_message_for_api(msg) for msg in messages]
                completion = await litellm.acompletion(
                    model=litellm_model_name,
                    messages=sanitized_messages,
                    tools=TOOLS_SCHEMA,
                    tool_choice="auto",
                    stream=True,
                )
                return StreamingResponse(
                    litellm_streamer(completion, model, session_id),
                    media_type="text/event-stream"
                )
            else:
                # We did enter the loop but no tools were called in the last round
                # The last message should be the assistant's final response
                last_msg = messages[-1]
                if last_msg.get("role") == "assistant" and last_msg.get("content"):
                    final_response = last_msg["content"]
                    return StreamingResponse(
                        single_chunk_streamer(final_response, model, session_id),
                        media_type="text/event-stream"
                    )
                else:
                    # Fallback: make a fresh streaming call
                    sanitized_messages = [sanitize_message_for_api(msg) for msg in messages]
                    completion = await litellm.acompletion(
                        model=litellm_model_name,
                        messages=sanitized_messages,
                        stream=True,
                    )
                    return StreamingResponse(
                        litellm_streamer(completion, model, session_id),
                        media_type="text/event-stream"
                    )

    except Exception as exc:
        # Log the full error for debugging
        print(f"Error during chat completion: {exc}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred with the AI service.")

@app.post("/api/datarobot-agent-chat")
async def datarobot_agent_chat(request: DataRobotChatRequest):
    """
    Handle chat requests for DataRobot custom agent deployments.
    
    This endpoint provides a separate pathway for interacting with custom
    DataRobot agents that have been trained for specific domains or use cases.
    It uses DataRobot's OpenAI-compatible API to communicate with deployed
    custom agents.
    
    Unlike the main chat endpoint, this endpoint:
    - Connects directly to DataRobot deployments
    - Uses deployment-specific models/agents
    - Supports domain-specific capabilities defined in the agent
    - May have different tool calling capabilities
    
    Args:
        request: DataRobotChatRequest with message, history, and deployment ID
        
    Returns:
        JSON response with agent's reply and metadata
    """
    try:
        deployment_id = request.model
        api_key = os.environ.get("DATAROBOT_API_TOKEN")

        if not api_key:
            raise HTTPException(status_code=500, detail="DATAROBOT_API_TOKEN is not configured.")
        
        # The base URL for the DataRobot OpenAI-compatible endpoint.
        # This can be made more configurable if needed (e.g., for different DR clouds).
        base_url = f"https://app.datarobot.com/api/v2/deployments/{deployment_id}"

        client = openai.OpenAI(            
            api_key=api_key,
            base_url=base_url,
            timeout=120, # Longer timeout for potentially complex agent responses
            max_retries=2,
        )

        messages = [msg.model_dump() for msg in request.conversation_history]
        messages.append({"role": "user", "content": request.message})

        # The 'model' parameter is required by the OpenAI client but may be ignored
        # by the DataRobot endpoint, which already knows the model.
        completion = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4.1",
            messages=messages,
            stream=False # Non-streaming for now
        )

        message = completion.choices[0].message
        
        response_data = {
            "response": message.content or "",
            "model_used": request.model,
            "session_id": ensure_session(request.session_id),
            "citations": [],
            "documentsUsed": [],
            "reasoning": "",
            "toolsUsed": [],
            "usedMetadataFields": [],
            "rawContentUsed": False,
            "imageContentUsed": False
        }

        # Check if the agent returned tool calls and format them for the frontend
        if message.tool_calls:
            tools_used = []
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tools_used.append({"name": tc.function.name, "arguments": args})
            response_data["toolsUsed"] = tools_used

        return JSONResponse(content=response_data)

    except openai.APIError as e:
        # Catch specific OpenAI errors to provide better feedback
        print(f"DataRobot Agent API Error: {e}")
        error_detail = f"Error from DataRobot Agent: {e.body.get('message', 'No details') if e.body else 'Unknown'}"
        raise HTTPException(status_code=e.status_code or 500, detail=error_detail)
    except Exception as e:
        print(f"Error during DataRobot agent chat: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# =============================================================================
# MODEL LISTING ENDPOINT
# =============================================================================

@app.get("/api/models")
async def get_models():
    """
    Return list of available AI models for frontend selection.
    
    Filters the available models to only include those that should be
    visible in the UI (some models may be hidden for testing purposes).
    
    Returns:
        Dictionary with list of available models and their capabilities
    """
    # Return only the models that should be visible in the UI
    ui_models = [m for m in AVAILABLE_MODELS if m.get("ui_visible", True)]
    return {"models": ui_models}

# =============================================================================
# DOCUMENT PROCESSING ENDPOINTS
# =============================================================================
# Endpoints for uploading, processing, and managing documents for RAG analysis
@app.post("/api/documents/sync")
async def sync_documents(request: DocumentSyncRequest):
    """
    Sync processed document metadata from client to server for analysis.
    
    This endpoint allows the frontend to share document metadata with the backend
    so that the AI can access and analyze previously processed documents. The
    metadata includes vectorized content, chunking information, and processing
    results that enable fast document Q&A without reprocessing.
    
    This approach allows documents to be processed client-side (in the browser)
    while making the results available for server-side AI analysis, providing
    a good balance of performance and functionality.
    
    Args:
        request: Contains session_id and list of completed document metadata
        
    Returns:
        Success confirmation with count of synced documents
    """
    session_id = request.session_id
    documents = request.documents
    
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required.")

    print(f"Syncing documents for session {session_id} with {len(documents)} documents.")
    
    loaded_metadata = {}
    
    for doc in documents:
        # We now trust the metadata coming directly from the client's IndexedDB.
        if (doc.get('status') == 'completed' and 
            doc.get('filename') and 
            doc.get('metadata')):
            
            filename = doc['filename']
            metadata = doc['metadata']
            loaded_metadata[filename] = metadata
            print(f"  - Loaded '{filename}' from client sync.")
        else:
            logging.warning(
                f"Skipping document during sync due to missing fields (id: {doc.get('id', 'N/A')}, status: {doc.get('status', 'N/A')})"
            )
            
    SESSION_METADATA_CACHE[session_id] = loaded_metadata
    print(f"Successfully synced metadata for {len(loaded_metadata)} documents into memory for session {session_id}.")
    
    return {
        "status": "success", 
        "message": f"Synced {len(loaded_metadata)} documents for analysis.",
        "synced_count": len(loaded_metadata)
    }


@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    extractionModel: str = Form(...),
    session_id: str = Form(...),
    document_id: str = Form(...), # Expect client to send a unique ID for the doc
    overwrite: bool = Form(False) # New parameter for handling duplicates
):
    """
    Upload document file and create processing job for RAG analysis.
    
    This endpoint accepts document uploads (PDF, Word, text files, etc.) and
    creates a processing job that will extract text, create embeddings, and
    prepare the document for AI analysis. The processing happens asynchronously
    and progress can be tracked via other endpoints.
    
    Features:
    - Duplicate detection and handling
    - Configurable extraction models
    - Async processing with progress tracking
    - Session-based isolation
    
    Args:
        file: Uploaded document file
        extractionModel: AI model to use for text extraction
        session_id: Session identifier for isolation
        document_id: Client-generated unique ID for the document
        overwrite: Whether to replace existing documents with same filename
        
    Returns:
        Job ID and document ID for tracking processing status
    """
    processor = get_or_create_processor(session_id, update_session_progress)
    
    # --- DUPLICATE CHECK LOGIC ---
    existing_job = None
    # Find if a job with the same filename already exists in this session
    # We don't need a lock for reading, but will use one for writing.
    for job_in_processor in processor.jobs.values():
        if job_in_processor.filename == file.filename:
            existing_job = job_in_processor
            break

    if existing_job:
        if not overwrite:
            # File exists, and we are not overwriting. Return a conflict.
            return JSONResponse(
                status_code=409,
                content={
                    "detail": "File with this name already exists in the session.",
                    "filename": file.filename,
                    "existing_document_id": existing_job.document_id
                }
            )
        else:
            # Overwriting: remove the old job before proceeding.
            print(f"Overwriting existing document: {existing_job.filename} (Job ID: {existing_job.id}) in session {session_id}")
            
            # Use a lock to safely delete the job from the processor's dictionary
            async with processor._lock:
                if existing_job.id in processor.jobs:
                    # Clean up the old temp file if it exists
                    if hasattr(existing_job, 'temp_file_path') and existing_job.temp_file_path and os.path.exists(existing_job.temp_file_path):
                        try:
                            os.remove(existing_job.temp_file_path)
                            print(f"Removed old temp file: {existing_job.temp_file_path}")
                        except OSError as e:
                            print(f"Error removing old temp file {existing_job.temp_file_path}: {e}")
                    
                    del processor.jobs[existing_job.id]
            
            # Also remove it from progress tracking
            remove_session_progress(session_id, existing_job.id)

    # Use the provided document_id as the job_id for the new job
    job = await processor.create_job(document_id=document_id, filename=file.filename)
    
    # Save the file to a robust temporary directory
    # This must be done *after* creating the job so we can attach the path to it
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, str(uuid.uuid4()) + Path(file.filename).suffix)

    try:
        # Read and write in chunks to handle large files
        chunk_size = 1024 * 1024  # 1MB chunks
        async with aiofiles.open(temp_file_path, 'wb') as f:
            while chunk := await file.read(chunk_size):
                await f.write(chunk)
        
        # Now, update the job with the path and model, and save it to cache
        job.temp_file_path = temp_file_path
        job.extraction_model = extractionModel
        job.save_to_cache(session_id)

    except Exception as e:
        # Clean up the temp file if something goes wrong during save
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    return {"jobId": job.id, "documentId": job.document_id}


@app.get("/api/documents/process/{job_id}/status")
async def get_processing_status(job_id: str, session_id: str):
    """Get the status of a processing job."""
    processor = get_or_create_processor(session_id, update_session_progress)
    job = await processor.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_dict = job.to_dict()
    
    # Debug: Log the job status being returned
    print(f"Status request for job {job_id}: {job_dict['status']} - {job_dict['progress'].get('stage', 'No stage')} ({job_dict['progress'].get('current', 0)}/{job_dict['progress'].get('total', 0)})")
    
    return job_dict


@app.get("/api/documents/progress/{session_id}")
async def get_session_progress(session_id: str):
    """Get progress for all jobs in a session - lightweight and fast."""
    session_progress = SESSION_PROGRESS.get(session_id, {})
    
    # Debug: Log what we're returning
    job_count = len(session_progress)
    active_jobs = [job_id for job_id, progress in session_progress.items() 
                   if progress.get('status') in ['queued', 'processing']]
    
    print(f"Session progress request for {session_id}: {job_count} jobs, {len(active_jobs)} active")
    
    # Debug: Log the actual data being returned
    for job_id, progress in session_progress.items():
        print(f"  Job {job_id[:8]}: status={progress.get('status')}, stage='{progress.get('stage')}', progress={progress.get('current')}/{progress.get('total')}")
    
    response_data = {
        "session_id": session_id,
        "jobs": session_progress,
        "timestamp": time.time()
    }
    
    # Debug: Log the full response
    print(f"Returning progress data: {json.dumps(response_data, indent=2)}")
    
    return response_data


@app.post("/api/documents/process/{job_id}/start")
async def start_processing(job_id: str, request: StartProcessingRequest):
    """Start processing a document (triggers actual RAG-Ultra processing)."""
    processor = get_or_create_processor(request.session_id, update_session_progress)
    job = await processor.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # If the job is already processing or completed, treat this request as idempotent and return success
    if job.status in {'processing', 'completed'}:
        return {
            "success": True,
            "message": f"Job already {job.status}"
        }
    if job.status != 'queued':
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not in queued state (current: {job.status})"
        )
    
    # Start processing in background
    asyncio.create_task(
        processor.start_job_processing(job_id=job_id)
    )
    
    return {
        "success": True,
        "message": "Processing started"
    }


@app.get("/api/documents/{session_id}")
async def list_documents(session_id: str):
    """List all documents for a session (completed jobs only)."""
    processor = get_or_create_processor(session_id, update_session_progress)
    
    documents = []
    for job in processor.jobs.values():
        if job.status == 'completed' and job.result:
            documents.append({
                "id": job.document_id,
                "filename": job.filename,
                "uploadDate": job.start_time.isoformat(),
                "processedDate": job.end_time.isoformat() if job.end_time else None,
                "status": job.status,
                "extractionModel": getattr(job, 'extraction_model', 'unknown'),
                "metadata": job.result  # Full metadata
            })
    
    return {"documents": documents}


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str, session_id: str):
    """
    Deletes a document's processing job and its cache file.
    Note: document_id is the same as job_id.
    """
    processor = get_or_create_processor(session_id, update_session_progress)
    job_id = document_id
    job_found = False

    # Delete from in-memory jobs
    async with processor._lock:
        if job_id in processor.jobs:
            del processor.jobs[job_id]
            job_found = True
            print(f"Deleted job {job_id} from memory in session {session_id}.")

    # Remove from session progress
    remove_session_progress(session_id, job_id)

    if job_found:
        return {"status": "success", "message": "Document deleted."}
    else:
        # The job might not be in memory but we can return success if it's gone
        return {"status": "not_found", "message": "Document not found."}


@app.delete("/api/documents/progress/{job_id}")
async def clear_job_progress(job_id: str, session_id: str):
    """
    Removes a completed or failed job from the session's progress tracking.
    This should be called by the client after it has acknowledged the
    final state of a job to prevent perpetual polling.
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required.")
    
    print(f"Received request to clear progress for job {job_id} in session {session_id}")
    remove_session_progress(session_id, job_id)
    
    return {"status": "success", "message": f"Progress for job {job_id} cleared."}

# =============================================================================
# STATIC FILE SERVING
# =============================================================================
# Serve the built React frontend application

# IMPORTANT: This MUST be the last route mounted to avoid interfering with API endpoints
# The React app's built files are served from the frontend/dist directory
# The html=True parameter enables SPA routing (serves index.html for non-API routes)
@app.post("/api/log-error")
async def log_error(request: dict):
    print("Received log request from frontend")
    error_message = request.get("error", "No error message provided")
    sql = request.get("sql", "No SQL provided")
    context = request.get("context", "No context provided")
    result = request.get("result", {})
    full_log = f"Frontend SQL Execution Error: {error_message} - SQL: {sql} - Context: {context} - Result: {result}"
    print(full_log)
    logging.error(full_log)
    return {"status": "logged"}

print("[STARTUP] /api/log-error endpoint registered")

if os.path.isdir("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
