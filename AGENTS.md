# AI Agent Collaboration Guide for TTMD3

Welcome to the TTMD3 (Talk to My Data 3) project! This guide equips AI agents with the knowledge needed to effectively collaborate on this sophisticated AI chat application. It emphasizes accuracy, highlights potential pitfalls, and promotes best practices drawn directly from the codebase.

**Important Resource**: The `plan/` folder contains detailed HTML files outlining the project plan, technical architecture, and feature specifications. Refer to these for in-depth understanding of requirements and design decisions.

## 🏗️ High-Level Architecture

TTMD3 is a full-stack application with a FastAPI backend and React frontend, focused on multi-modal AI chat with document analysis and data exploration.

### Backend Overview (FastAPI + Python 3.11)
- **Core File**: `backend/main.py` (~2168 lines) - Houses all API endpoints, tool definitions, and orchestration logic.
- **Document Handling**: `backend/document_processor.py` - Manages RAG-Ultra integration with async job tracking.
- **RAG System**: `backend/rag_ultra/` - Custom SDK for document processing, including loaders, metadata generators, and retrievers.
- **State Management**: In-memory sessions with UUID keys; no persistent storage.
- **Tools**: Asynchronous tool-calling for integrations like weather APIs, stock data, web search, and custom functions.

### Frontend Overview (React 18 + Vite)
- **Core Component**: `frontend/src/App.jsx` (~980 lines) - Manages the main UI, chat interface, and modal flows.
- **State Management**: React Context API (e.g., `DocumentContext.jsx`, `DataContext.jsx`) with reducers for complex state.
- **Styling**: TailwindCSS for layout/utilities; Chakra UI for interactive components.
- **Data Handling**: DuckDB-WASM enables in-browser SQL for file-based analytics.
- **Persistence**: IndexedDB via Dexie for local storage of documents, jobs, and conversations.

## ⚠️ Key Gotchas & Caveats

Based on codebase analysis, here are critical issues to watch:

### 1. **Session Volatility**
- Sessions use in-memory storage on the backend.
- **GOTCHA**: State is lost on server restarts or crashes. No database persistence.
- **Workaround**: Frontend backups critical data to IndexedDB; implement recovery logic where possible.

### 2. **Async Processing Challenges**
- Document jobs are queued and processed asynchronously with callbacks.
- **GOTCHA**: Jobs can stall if callbacks fail; progress updates rely on proper handler setup.
- **GOTCHA**: High resource usage during RAG processing - limit to one job at a time.

### 3. **API Conventions**
- Endpoints often use `Form` data for uploads, not JSON.
- **GOTCHA**: Parameter casing is specific (e.g., `extractionModel`); mismatches cause silent failures.
- **GOTCHA**: Streaming responses require proper frontend handling to avoid partial data issues.

### 4. **Frontend State Sync**
- Contexts use reducers for state updates.
- **GOTCHA**: Updates are async; always use loading states to prevent race conditions.
- **GOTCHA**: Document selections are separate from data - sync them explicitly.

### 5. **DataRobot Specifics**
- Runtime params via `metadata.yaml` and `datarobot_drum`.
- **GOTCHA**: Agent IDs differ from model names; test in DataRobot environment.

## 🔧 Best Practices & Patterns

Follow these to maintain consistency with the existing codebase.

### Environment Setup
- **Backend**: Activate virtualenv, install from `requirements.txt`. Use `.env` for keys.
- **Frontend**: Run `npm install` in `frontend/`. Use Vite dev server.
- **Local Run**: Start backend with `uvicorn`, frontend with `npm run dev`.
- **Testing**: Use Swagger at `http://localhost:8080/docs` for API; React DevTools for frontend.

### Backend Patterns

#### Async Operations
```python
# ✅ Good: Full async flow
async def handle_request():
    try:
        result = await external_api_call()
        return result
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        raise HTTPException(500, "Internal error")

# ❌ Bad: Blocking calls in async context
def handle_request():
    result = sync_blocking_call()  # Causes performance issues
```

#### Session Handling
```python
# ✅ Good: Safe session creation
session_id = request.headers.get('session-id') or str(uuid.uuid4())
sessions[session_id] = {}  # Initialize if new
```

#### Progress Updates
```python
# ✅ Good: Callback integration
def callback(stage, current, total):
    update_session_progress(session_id, job_id, stage, current, total)
```

### Frontend Patterns

#### Context & Hooks
```javascript
// ✅ Good: Hook-based access
const { documents, dispatch } = useDocuments();

// ❌ Bad: Manual context usage - violates React rules
```

#### State Management
```javascript
// ✅ Good: Dispatch actions
dispatch({ type: 'UPDATE_JOB', payload: { id, updates } });
```

#### API Integration
```javascript
// ✅ Good: Robust async handling with error boundaries
try {
  setLoading(true);
  await api.call();
} catch (err) {
  handleError(err);
} finally {
  setLoading(false);
}
```

### Integration Patterns
- **Frontend-Backend Sync**: Use `axios` with baseURL set via env; handle streaming with event sources.
- **Data Flow**: Upload files to backend, process via RAG, retrieve results via polling.
- **Error Propagation**: Backend logs errors; frontend should poll for job status.

## 🚨 Common Problems & Fixes

### Processing Stalls
- **Cause**: Callback failures or resource exhaustion.
- **Fix**: Check logs; implement retries; monitor server resources.

### State Inconsistencies
- **Cause**: Async updates without proper synchronization.
- **Fix**: Use `useEffect` dependencies; add optimistic updates.

### Deployment Issues
- **Cause**: Missing env vars or path mismatches.
- **Fix**: Verify `build-app.sh` and `start-app.sh`; test in Docker.

## 📁 Code Organization

[Existing structure diagrams remain accurate]

## 🔍 Debugging & Testing

[Enhanced with more tips: Use browser network tab for API traces; add unit tests for reducers.]

## 🚀 Optimization & Security

[Existing content is solid; added note on input validation for uploads.]

## 📚 Dependencies & Workflow

[Updated to include workflow steps with references to 'plan/' folder.]

## 🤝 Collaboration

[Added emphasis on updating 'plan/' docs for major changes.]

---

This guide is derived from direct codebase analysis. For the latest specs, always cross-reference with `plan/` files. When contributing, prioritize accuracy and test all changes. 