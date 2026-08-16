# GraphAnchor ⚓

GraphAnchor is a lightweight, local-first Graph RAG (Retrieval-Augmented Generation) backend. It extracts structured knowledge graphs from unstructured text documents and intelligently merges entities using vector similarity.

## 🌟 What Exactly This Does

1. **Token-based Chunking**: Safely splits large incoming text documents into manageable overlapping chunks using OpenAI's `tiktoken` sliding window.
2. **Structured Entity Extraction**: Uses local LLMs (via Ollama in JSON-mode) to extract entities and their relationships from text into a strictly enforced graph schema.
3. **Smart Entity Placement**: 
   - Attempts an **exact match** against known graph entities.
   - If that fails, it falls back to **Cosine Similarity** using embedding models. If the new entity is semantically similar to an existing one (above a 0.7 threshold), they are merged.
   - Otherwise, it creates a new unconnected "island" entity in the graph.
4. **Dual Storage Architecture**: 
   - Uses **SQLite** for robust tabular storage of the graph relationships (`edges`) and document references (`chunks`).
   - Uses **ChromaDB** for vector embeddings and similarity search of entities.
5. **100% Local Processing**: No API keys, no cloud billing. Everything runs entirely on your local machine for complete data privacy.

---

## 🚀 Setup & Installation

### 1. Install Prerequisites
You will need two main tools installed on your system:
* **uv**: An extremely fast Python package and environment manager.
  * *Windows:* `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
  * *macOS/Linux:* `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **Ollama**: A local LLM runner.
  * Download and install from [ollama.com](https://ollama.com/).

### 2. Pull the AI Models
Once Ollama is installed, pull the necessary models for extraction and embedding:
```bash
# For structured JSON extraction (fast & highly capable)
ollama pull qwen2.5-coder:7b

# For vector embeddings
ollama pull snowflake-arctic-embed2:568m
```
*(Note: You can switch to `llama3.2` or other models by updating `config.yaml`)*

### 3. Setup the Project
Clone the repository and install the dependencies using `uv`:
```bash
git clone https://github.com/Lohithnath2910/GraphAnchor.git
cd GraphAnchor

# Initialize virtual environment and sync dependencies
uv sync
```

---

## 🧪 How to Test It Out

### 1. Start the Server
Activate your virtual environment and start the FastAPI server:
```bash
# Windows
.venv\Scripts\Activate

# Start the server with hot-reloading
uvicorn main:app --port 8000 --reload
```

### 2. Interactive Swagger UI
Open your browser and navigate to:
**[http://localhost:8000/docs](http://localhost:8000/docs)**

From this UI, you can manually interact with the API:
* **POST `/ingest`**: Upload a `.txt` file to have it chunked, extracted, and embedded.
* **GET `/graph/stats`**: View the current number of nodes and edges in your SQLite database.
* **GET `/query`**: Perform a semantic vector search over your ingested chunks.

### 3. Automated Demo Script
If you want to see the "Smart Entity Placement" in action automatically, we have provided a demo script. Keep the server running, open a **new terminal window**, activate the virtual environment, and run:
```bash
python demo.py
```
This will upload 3 intentionally overlapping documents to prove that the similarity threshold correctly merges related entities!

---

## Current Progress

Ingestion, storage, entity placement, and hybrid retrieval (vector search plus single-hop graph traversal on `/query`) are implemented and working end to end.

Backend hardening that has been added on top of that:
* Upload validation on `/ingest`: rejects non-`.txt`, oversized, empty, or non-UTF8 files instead of crashing
* Ingestion runs as a single atomic transaction (rollback and connection cleanup on failure, no partial writes)
* Duplicate document detection by content hash
* `/reset` requires a `?confirm=true` query parameter
* Ollama calls retry on transient failures
* Logging in place of print statements
* CORS enabled so a local frontend can call the API directly

### Minimal Frontend
A single-page frontend is available at `web/index.html`. With the backend running (`uvicorn main:app --port 8000`), open the file directly in a browser. It lets you:
* Upload a `.txt` file
* Ask a question
* See the matched text chunks and the graph connections for that query, drawn as a simple diagram

The page has a spot reserved for a generated answer. It will show automatically once the `/query` response includes an `answer` field (or the endpoint producing that is wired in), and shows a placeholder message until then.

### Still Open
* A generation endpoint that turns retrieved and traversed chunks into a final answer
* A full automated test suite
