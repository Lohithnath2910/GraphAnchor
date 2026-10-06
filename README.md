# GraphAnchor

A local-first hybrid RAG engine. It ingests documents one at a time, extracts a knowledge graph with a small on-device
language model, and answers questions with vector search plus graph traversal. Multi-hop questions are answered in steps.
Everything runs offline through [Ollama](https://ollama.com/); no API keys, no cloud.

## Results

On questions that chain facts across documents, GraphAnchor answers **59.5%** correctly against **38.0%** for standard vector
RAG (+21.5 points) and retrieves the right documents **88.3%** of the time against **51.9%**. On single-fact questions it
matches or slightly beats vector RAG. Full tables, method and limitations: [results.md](results.md) ([PDF](results.pdf)).

## How it works

1. **Ingest** (`POST /ingest`): text is split into overlapping token chunks; each chunk is embedded and stored in ChromaDB;
   a local LLM extracts (entity, relation, entity) triples, entities are merged by exact match, then embedding similarity,
   with spaCy clean-up, and the triples go to SQLite. Ingestion is atomic and de-duplicated by content hash.
2. **Retrieve** (`GET /query`): vector search finds entry chunks; a breadth-first walk of the graph (up to 4 hops) pulls in
   linked chunks, which are re-ranked by similarity and question-word overlap.
3. **Route**: chained or list-style questions use the graph; single-fact questions are answered by vector search alone.
4. **Reason**: a multi-hop question is split into 2 or 3 steps, each retrieved and answered in turn; the final answer uses the
   merged evidence and the facts established along the way. The response includes `reasoning_steps` and
   `graph_traversal.paths`.
5. **Answer**: a grounded prompt answers only from the supplied context and says so when the answer is not there.

## Quick start

**Requirements:** [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com/), Python 3.11+.

```bash
ollama pull llama3.2
ollama pull snowflake-arctic-embed2:568m
uv sync                                   # installs everything, including the pinned spaCy model
uv run uvicorn main:app --port 8000       # then open http://127.0.0.1:8000  (API docs at /docs)
```

**Docker** (starts Ollama, downloads both models on first run, then the API):

```bash
docker compose up --build
```

The graph database, vector store and embedding cache live in `./data` and survive restarts. For an NVIDIA GPU, uncomment
the `deploy` block in `docker-compose.yml`.

## API

| Endpoint | Purpose |
|---|---|
| `POST /ingest` | upload a `.txt`, `.md` or `.pdf` file |
| `GET /query?q=...&k=3&enable_graph=true` | retrieval plus answer; `answer=false` for retrieval only; `decompose=true/false` overrides the multi-step setting |
| `GET /answer`, `POST /answer` | just the answer text |
| `GET /answer/stream` | answer as server-sent events |
| `GET /documents`, `GET /documents/{id}`, `DELETE /documents/{id}` | list, inspect, delete |
| `GET /graph/stats`, `GET /graph/all` | graph size, whole graph |
| `DELETE /reset?confirm=true` | wipe all data |

## Configuration

`config.yaml` holds models, paths and chunking. Behaviour switches have defaults in `src/config.py` and can be overridden in
`config.yaml`:

| Setting | Default | Meaning |
|---|---|---|
| `llm_model`, `embed_model` | `llama3.2`, `snowflake-arctic-embed2:568m` | Ollama models |
| `decompose_multihop` | on | answer multi-hop questions in steps (about 12 s per question) |
| `graph_routing` | on | single-fact questions skip the graph |
| `prompt_v2` | on | stricter, grounded answer prompt |
| `graph_lexical_weight`, `graph_same_doc_discount` | 0.5, 0.6 | graph chunk selection |
| `graph_hygiene`, `graph_min_coverage_gain` | off | experimental, not adopted |

Set `OLLAMA_HOST` to point at a remote Ollama (the Docker setup does this).

## Benchmark

`benchmark/` holds the 76-document corpus and 148 questions; `eval/run_benchmark.py` compares vector with hybrid and writes
CSVs to `eval/results/`. See [benchmark/README.md](benchmark/README.md). The report's reproduce section has the commands.

## Layout

| Path | Contents |
|---|---|
| `main.py`, `config.yaml`, `src/` | API server, retrieval, graph, generation |
| `web/` | browser front end |
| `benchmark/` | corpus, questions, builder, ingest script |
| `eval/` | benchmark runner and recorded results |
| `docs/` | project report and codebase explanation |
| `tests/` | API tests (they reset the database: run them only against a scratch copy) |
| `results.md`, `results.pdf` | the evaluation report |
