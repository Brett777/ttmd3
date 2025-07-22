# Agentic Chat Application

A comprehensive AI-powered chat application with multi-modal capabilities, document analysis, and data exploration features. Built for the DataRobot Custom Applications platform with support for multiple LLM providers and advanced tool-calling capabilities.

## ✨ Features

* **Multi-Modal Chat Interface** - React 18 + Vite frontend with **TailwindCSS** & **Chakra UI** components
* **Advanced AI Backend** - Python 3.11 **FastAPI** with **LiteLLM** for multi-provider support
* **Document Intelligence** - RAG-Ultra powered document processing with real-time analysis
* **Data Exploration** - "Talk to My Data" feature with SQL generation and DuckDB-WASM
* **Built-in Tools**:
  * `get_weather(location)` – Real-time weather data
  * `get_stock_price(ticker)` – Live stock information via Yahoo Finance
  * `search_the_web(query)` – Web search capabilities
  * `current_date_and_time()` – Time and date information
  * `analyze_documents(query)` – RAG-powered document analysis
  * `talk_to_data(query)` – SQL generation for structured data
* **Multi-Provider LLM Support** - OpenAI, Anthropic, Google, xAI, Cohere, DeepSeek, and more
* **DataRobot Integration** - Direct Agent deployment connectivity
* **Real-time Streaming** - Live response streaming for enhanced UX
* **Session Management** - Persistent conversation state and document storage
* **Database Connectors** - Snowflake integration for live data analysis
* Runtime-parameter integration for provider API keys (see `metadata.yaml`)
* Ready-to-deploy in DataRobot or run locally for development

---

## 🖥️ Local Development Guide

### 1. Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.11.x |
| Node.js | ≥ 18.x |
| npm | ≥ 9.x |

Ensure `pip` and `virtualenv` (or another environment manager) are available.

### 2. Clone & Setup

```powershell
# Windows PowerShell example
cd C:\path\to\workspace
# (clone repo)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file at the project root for local runs:

```env
# Server
PORT=8080
BASE_PATH=/

# LLM Provider Keys (only add those you need)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
XAI_API_KEY=...
COHERE_API_KEY=...
DEEPSEEK_API_KEY=...
PERPLEXITYAI_API_KEY=...
GEMINI_API_KEY=...
HUGGINGFACE_API_KEY=...

# DataRobot Integration (optional)
DATAROBOT_API_TOKEN=...
DATAROBOT_AGENT_DEPLOYMENT_ID=...
```

> The backend automatically loads `.env` via **python-dotenv**.

### 4. Install Frontend Dependencies

```powershell
cd frontend
npm install
```

This installs React, TailwindCSS, Chakra UI, DuckDB-WASM, Chart.js, and other dependencies.

### 5. Running the Services

Open **two terminals** (or use background tasks).

#### Backend (FastAPI)

```powershell
# From project root (with virtualenv activated)
uvicorn backend.main:app --reload --port 8080
```

* The API is now at `http://localhost:8080/api/...`
* Docs available at `http://localhost:8080/docs` (Swagger UI).

#### Frontend (Vite Dev Server)

```powershell
cd frontend
npm run dev -- --port 5173
```

* The UI opens at `http://localhost:5173` and proxies API calls to `localhost:8080`.

### 6. Quick Test

1.  Open the browser at `http://localhost:5173`.
2.  Try different features:
    - **Chat**: Send a message like "What's the weather in Paris?"
    - **Documents**: Upload a PDF and ask questions about it
    - **Data Analysis**: Upload a CSV and ask "Show me the top 5 products by revenue"
3.  Observe tool-calling in the FastAPI console & responses in the UI.

---

## 🏗️ Production Build

DataRobot will automatically execute `build-app.sh` during container build. To test locally:

```powershell
./build-app.sh   # installs deps & builds React -> frontend/dist
./start-app.sh   # launches uvicorn with BASE_PATH & PORT detection
```

This mimics the runtime inside the DataRobot container.

---

## 📂 Project Structure

```text
├── backend/           # FastAPI app (main.py, tools, RAG integration)
├── frontend/          # React + Vite frontend
│   ├── public/
│   └── src/
│       ├── components/    # UI components (chat, documents, data analysis)
│       ├── services/      # API clients
│       ├── contexts/      # React context providers
│       └── hooks/         # Custom React hooks
├── build-app.sh       # Build script (Python + Node deps, React build)
├── start-app.sh       # Start script (uvicorn)
├── requirements.txt   # Python deps (FastAPI, LiteLLM, RAG-Ultra, etc.)
├── metadata.yaml      # DataRobot RuntimeParameters
└── plan/              # Detailed technical documentation
```

---

## 🧑‍💻 Useful Commands

| Purpose | Command |
|---------|---------|
| Run backend tests (if added) | `pytest` |
| Lint python (optional) | `ruff check backend` |
| Rebuild Tailwind styles | Handled automatically by Vite |
| Check API documentation | Visit `http://localhost:8080/docs` |

---

## 🤝 Contributing

PRs are welcome!  Please follow conventional commit messages and run linting before submitting.

---

## 📝 License

MIT © DataRobot OSS / Contributors 