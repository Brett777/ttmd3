# Agentic Chat Application

A reference template for building LLM-powered, tool-calling chat applications on the DataRobot Custom Applications platform.

## ✨ Features

* React 18 + Vite frontend styled with **Chakra UI** & **TailwindCSS**
* Python 3.11 **FastAPI** backend with **LiteLLM** function-calling agent
* Built-in tools:
  * `get_weather(location)` – Nominatim + Open-Meteo
  * `get_stock_price(ticker)` – Yahoo Finance via *yfinance*
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
```

> The backend automatically loads `.env` via **python-dotenv**.

### 4. Install Frontend Dependencies

```powershell
cd frontend
npm install
```

This installs React, Chakra UI, TailwindCSS, Vite, etc.

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
2.  Send a message like "What's the weather in Paris?".
3.  Observe tool-calling in the FastAPI console & response in the UI.

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
├── backend/           # FastAPI app (main.py, tools)
├── frontend/          # React + Vite frontend
│   ├── public/
│   └── src/
├── build-app.sh       # Build script (Python + Node deps, React build)
├── start-app.sh       # Start script (uvicorn)
├── Dockerfile         # Container definition
├── requirements.txt   # Python deps
├── metadata.yaml      # DataRobot RuntimeParameters
└── application-plan.html  # Detailed technical plan
```

---

## 🧑‍💻 Useful Commands

| Purpose | Command |
|---------|---------|
| Run backend tests (if added) | `pytest` |
| Lint python (optional) | `ruff check backend` |
| Rebuild Tailwind styles | Handled automatically by Vite |

---

## 🤝 Contributing

PRs are welcome!  Please follow conventional commit messages and run linting before submitting.

---

## 📝 License

MIT © DataRobot OSS / Contributors 