# Local, Graph-Structured Storage Engine for RAG with On-Device SLM-Driven Ingestion

*Working title — rename as you like*

---

## Abstract

Standard RAG systems retrieve document chunks purely by embedding similarity, which fails on queries that require understanding *relationships* between pieces of information — exceptions, dependencies, contradictions. The existing fix, GraphRAG-style systems, solves this by building a relationship graph at ingestion time, but requires a large cloud LLM, which is slow, expensive, and not private.

This project builds a **local, incrementally-growing storage engine** for personal RAG use. Documents are added one at a time (not as a full corpus upfront). Each new document is processed by a **small on-device model (1–3B parameters)** that extracts structured `{entity, relation, target_chunk_id}` triples in a fixed schema. These are stored alongside standard vector embeddings in a **single local file** (SQLite/DuckDB-backed), split into a vector index and a graph-edge store, kept in sync. New documents attach into the *existing* graph via a cheap entity-lookup step with an embedding-similarity fallback — never a full-corpus rebuild. At query time, vector similarity finds an entry point, and graph traversal (1–2 hops) expands outward to pull in structurally related chunks that similarity search alone would miss.

---

## 1. Problem Statement

1. **Relational blindness** — plain vector RAG answers "what sounds like the query," not "what's structurally related to the answer." Queries like "what are the exceptions to X" or "what depends on Y" are answered poorly.
2. **The existing fix is expensive and not private** — GraphRAG-style graph construction depends on a large cloud LLM at ingestion time: high token cost, high latency, and data leaving the device.
3. **No existing system builds this structure locally, incrementally, with a small on-device model, at the point of ingestion** — as opposed to a full-corpus batch job run in the cloud.

---

## 2. Current Competition

| System | Structure builder | Runs | Lifecycle | Core weakness |
|---|---|---|---|---|
| **Plain vector RAG** | None | Anywhere | N/A | No relational awareness at all |
| **GraphRAG** | Large cloud LLM | Cloud | Batch, full-corpus rebuild | Expensive, not private, static index |
| **LiteSemRAG** | None — graph built from embedding similarity only | Local | Batch-ish | Graph is just similarity clusters; no real relation semantics |
| **Contextual Retrieval** | Large LLM enriches chunks before embedding | Cloud | Batch | No explicit traversable graph structure |
| **This project** | Small on-device SLM | Fully local | Incremental, per-document | — (this is the gap being targeted) |

**Where this project differs from GraphRAG specifically — not just "same idea, smaller model":**
- **Lifecycle**: GraphRAG is batch build → static index → query. This is a living graph that grows with each ingestion event — no full-corpus reprocessing when a new document arrives.
- **No expensive entity resolution**: GraphRAG normalizes entities into deduplicated canonical nodes (an LLM-heavy step). This project skips that — entities are edge labels pointing at chunks, not separately resolved nodes.
- **Confidence as a stored value**: edges carry a confidence score (high for exact entity matches, scaled for embedding-fallback matches), so a small, occasionally-noisy model's uncertainty is explicit and usable at query time — GraphRAG has no comparable need since it trusts a large model's extractions.
- **Single unified local file**: GraphRAG's output is fragmented (separate parquet tables for entities/relationships/communities, usually paired with a separate vector store). This project keeps vectors + graph edges in one local engine.

---

## 3. Summary

A private, on-device knowledge base that grows as you feed it documents. Each document is chunked, embedded normally, and separately passed through a small local model that extracts entity/relation triples describing how it connects to what's already stored. New chunks attach to the existing graph cheaply (exact entity match first, embedding-similarity fallback second, unconnected "island" as the safe default) — no reprocessing of prior documents. At query time, retrieval is hybrid: vector search finds a starting chunk, and graph traversal walks outward from there, so an answer can be assembled from chunks that are *structurally* related even if they don't read as similar to the question.

---

## 4. Our Solution, Point-Wise

- Chunk documents with standard methods (no novel chunking claimed).
- Embed each chunk normally, using a sentence-transformer model — stored in a vector index.
- Pass each chunk through a small on-device model (1–3B, via Ollama or similar) that emits a **fixed-schema** output: `{entity, relation, target_chunk_id}` — constrained extraction, not open-ended graph generation, because small models are far more reliable at the former.
- Store the extracted triples as graph edges, in the **same local file** as the vector index (SQLite/DuckDB), not a separate store.
- On each new document, attach its chunks into the existing graph incrementally:
  - Cheap exact-match lookup against an inverted entity index first.
  - Embedding-similarity fallback if no exact match, attached with a lower confidence score.
  - Unconnected chunks are left as "islands" rather than blocked or forced to connect.
- Store a confidence value per edge, usable later to threshold or weight traversal.
- At query time: embed the query → vector search finds entry-point chunk(s) → graph traversal (1–2 hops, confidence-thresholded) expands to structurally related chunks → dedupe and rank the combined set → pass to the generation model.
- Everything — embedding, SLM extraction, storage, traversal — runs fully offline, on-device.

---

## 5. Stage-Wise Implementation

**Stage 1 — Ingestion**
1. Document input (plain text, arbitrary length)
2. Chunking module (fixed-size or sentence-based)
3. Branches into:
   - Chunk embedding (sentence-transformer)
   - Small on-device LLM → fixed-schema extraction (entity, relation, target_chunk)
4. Both outputs land in the local storage layer

**Stage 2 — Storage**
5. Local storage layer, single file (SQLite/DuckDB):
   - Vector index (cosine similarity, top-k)
   - Graph edges (entity, relation, chunk, confidence)
6. Placement logic for new chunks: exact entity match → embedding-similarity fallback → unconnected island as default

**Stage 3 — Query & retrieval**
7. User query → query embedding (same model as ingestion)
8. Vector index search → entry-point chunk(s)
9. Entry-point chunks → look up connected graph edges
10. Graph traversal (1–2 hops, confidence-thresholded)
11. Combine and dedupe entry-point chunks + traversed chunks, ranked by score

**Stage 4 — Generation**
12. Combined, ranked chunks passed to a generation model (local or larger) → grounded answer

**Stage 5 — Evaluation** *(post-build)*
13. Build a 20–30 question test set: simple factual lookups vs. relational/multi-hop questions
14. Compare three pipelines: plain vector RAG, this graph-hybrid engine, optionally a large-LLM GraphRAG baseline
15. Measure retrieval recall/precision, answer correctness, latency, and resource/cost footprint
16. Expected finding: minimal difference on simple lookups, meaningful gap in favor of this engine on relational/multi-hop questions

---

## 6. Still Open (decide during implementation, not before)

- Merge/ranking logic for combining vector-entry-point results with graph-traversal results before generation
- The exact SLM extraction prompt template
- Concrete confidence-threshold values (tune empirically during evaluation)
- Final tech stack: specific embedding model, specific Ollama model, chunk size in tokens
- Target document domain and the actual 20–30 evaluation questions

## 7. Honesty Notes Carried Through

- This is a **storage engine/library layer built on SQLite/DuckDB**, not a from-scratch DBMS — durability, crash recovery, and concurrency are inherited from the underlying engine, not built new.
- Patentability (India, Section 3(k)): a pure algorithm/ranking claim won't survive examination on its own. The stronger basis here is the specific data structure (unified local vector+graph schema) and method (on-device model populating it), tied to a measurable technical effect (offline operation, reduced cost vs. cloud alternatives) — still requires confirmation from an institution's patent cell before investing effort in a filing.
- Reference list in the original document reflects a working search pass, not an exhaustive prior-art search — a dedicated Google Patents / Scholar / Semantic Scholar check is recommended before committing significant effort.

---

## 8. Deliverables

1. Working local prototype — ingestion pipeline, storage schema, hybrid query engine, running fully offline
2. Evaluation report comparing the three pipelines, with numbers
3. Short paper/write-up: problem, related work, method, evaluation, results, limitations
4. Optional: a scoped patent draft, reviewed by the institution's patent cell
