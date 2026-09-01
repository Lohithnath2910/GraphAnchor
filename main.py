import os
import sys
import uuid
import hashlib
import logging

# Ensure root directory is always on python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple

from src.config import config
from src.ingestion import chunk_text
from src.storage import (
    db_cursor,
    get_chunks_collection,
    get_entities_collection,
    reset_all_data
)
from src.retrieval import get_embedding
from src.generation import extract_graph_from_chunk, generate_answer, GraphExtraction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("graphanchor")


app = FastAPI(title="GraphAnchor")

# Allow the local frontend (opened as a file, or served from any port) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class IngestResponse(BaseModel):
    doc_id: str
    chunks_processed: int
    entities_extracted: int
    edges_added: int
    extraction_failures: int = 0
    duplicate: bool = False

INVALID_ENTITIES = {
    "is", "are", "was", "were", "has", "have", "had", "the", "a", "an",
    "this", "that", "these", "those", "it", "its", "it's"
}

PRONOUNS = {
    "he", "she", "they", "him", "her", "his", "their", "them", "my", "i", "me", "we", "us"
}

def place_entity(entity_str: str, staged_entities: Optional[List[str]] = None, chunk_context: str = "") -> Tuple[str, float]:
    """Returns canonical entity name and confidence.
    Filters out invalid verbs/pronouns and performs fuzzy embedding deduplication."""
    if not entity_str:
        return "", 0.0

    entity_clean = entity_str.strip()
    if not entity_clean or len(entity_clean) <= 1:
        return "", 0.0

    # 1. Filter out verbs / articles
    if entity_clean.lower() in INVALID_ENTITIES:
        return "", 0.0

    # 2. Resolve pronoun to primary subject of chunk if available
    if entity_clean.lower() in PRONOUNS:
        if chunk_context:
            import re
            words = re.findall(r'\b([A-Z][a-z]+|[a-z]{3,})\b', chunk_context)
            for w in words:
                if w.lower() not in INVALID_ENTITIES and w.lower() not in PRONOUNS and w.lower() not in {'most', 'very', 'person', 'good', 'smart', 'grace', 'planet', 'partner', 'project'}:
                    entity_clean = w
                    break
        if entity_clean.lower() in PRONOUNS:
            return "", 0.0

    entities_col = get_entities_collection()

    # 3. Exact match lookup
    try:
        res = entities_col.get(ids=[entity_clean])
        if res and res.get('ids') and len(res['ids']) > 0:
            return entity_clean, 1.0
    except Exception as e:
        logger.warning(f"Exact-match lookup failed for '{entity_clean}': {e}")

    # 4. Embedding similarity fallback (for fuzzy matching like 'lohtih' -> 'Lohith')
    emb = get_embedding(entity_clean)

    if entities_col.count() > 0:
        search_res = entities_col.query(
            query_embeddings=[emb],
            n_results=1
        )

        if search_res and search_res.get('distances') and len(search_res['distances'][0]) > 0:
            dist = search_res['distances'][0][0]
            sim = 1.0 - dist
            if sim >= config.similarity_threshold:
                canonical = search_res['ids'][0][0]
                return canonical, float(sim)

    # 5. Insert as new canonical entity
    entities_col.add(
        ids=[entity_clean],
        embeddings=[emb],
        documents=[entity_clean]
    )
    if staged_entities is not None:
        staged_entities.append(entity_clean)
    return entity_clean, 1.0

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=415, detail="Only .txt files are supported.")

    content = await file.read()

    max_bytes = int(config.max_file_size_mb * 1024 * 1024)
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds max size of {config.max_file_size_mb} MB.")

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be valid UTF-8 text.")

    if not text.strip():
        raise HTTPException(status_code=400, detail="File is empty.")

    content_hash = hashlib.sha256(content).hexdigest()

    # Dedup check - skip reprocessing a document we've already ingested
    with db_cursor() as cursor:
        cursor.execute("SELECT doc_id FROM documents WHERE content_hash = ?", (content_hash,))
        existing = cursor.fetchone()
    if existing:
        return IngestResponse(
            doc_id=existing[0],
            chunks_processed=0,
            entities_extracted=0,
            edges_added=0,
            duplicate=True
        )

    doc_id = str(uuid.uuid4())
    chunks = chunk_text(text)

    total_entities = 0
    total_edges = 0
    extraction_failures = 0

    staged_chunk_ids: List[str] = []
    staged_entity_ids: List[str] = []
    chunks_col = get_chunks_collection()
    entities_col = get_entities_collection()

    try:
        with db_cursor() as cursor:
            for i, chunk_text_content in enumerate(chunks):
                chunk_id = f"{doc_id}_{i}"

                # Embed chunk
                chunk_emb = get_embedding(chunk_text_content)

                # Add to ChromaDB chunks
                chunks_col.add(
                    ids=[chunk_id],
                    embeddings=[chunk_emb],
                    documents=[chunk_text_content],
                    metadatas=[{"chunk_id": chunk_id, "doc_id": doc_id}]
                )
                staged_chunk_ids.append(chunk_id)

                # Add to SQLite chunks
                cursor.execute(
                    "INSERT INTO chunks (chunk_id, doc_id, text) VALUES (?, ?, ?)",
                    (chunk_id, doc_id, chunk_text_content)
                )

                # Extract graph
                try:
                    extraction = extract_graph_from_chunk(chunk_text_content)
                except Exception as e:
                    logger.error(f"Extraction failed for chunk {chunk_id}: {e}")
                    extraction_failures += 1
                    continue

                total_entities += len(extraction.entities)

                for rel in extraction.relations:
                    src_raw = rel.entity.strip()
                    tgt_raw = rel.target_entity.strip()

                    src_canonical, src_conf = place_entity(src_raw, staged_entities=staged_entity_ids, chunk_context=chunk_text_content)
                    tgt_canonical, tgt_conf = place_entity(tgt_raw, staged_entities=staged_entity_ids, chunk_context=chunk_text_content)

                    if not src_canonical or not tgt_canonical or src_canonical.lower() == tgt_canonical.lower():
                        continue

                    edge_conf = min(src_conf, tgt_conf)

                    cursor.execute("""
                        INSERT INTO edges (source_entity, relation, target_entity, chunk_id, confidence)
                        VALUES (?, ?, ?, ?, ?)
                    """, (src_canonical, rel.relation, tgt_canonical, chunk_id, edge_conf))
                    total_edges += 1

            cursor.execute(
                "INSERT INTO documents (doc_id, content_hash, filename) VALUES (?, ?, ?)",
                (doc_id, content_hash, file.filename)
            )
    except Exception as e:
        logger.error(f"Ingest failed for doc {doc_id}: {e}. Rolling back staged ChromaDB entries.")
        # Roll back ChromaDB chunks and entities to maintain 100% sync with SQLite transaction rollback
        if staged_chunk_ids:
            try:
                chunks_col.delete(ids=staged_chunk_ids)
            except Exception as rollback_err:
                logger.warning(f"Error rolling back ChromaDB chunks: {rollback_err}")
        if staged_entity_ids:
            try:
                entities_col.delete(ids=staged_entity_ids)
            except Exception as rollback_err:
                logger.warning(f"Error rolling back ChromaDB entities: {rollback_err}")
        raise HTTPException(status_code=500, detail="Ingestion failed; no partial data was committed.")

    return IngestResponse(
        doc_id=doc_id,
        chunks_processed=len(chunks),
        entities_extracted=total_entities,
        edges_added=total_edges,
        extraction_failures=extraction_failures
    )

@app.delete("/reset")
def reset_databases(confirm: bool = False):
    """Clear all data from SQLite and ChromaDB. Pass ?confirm=true to proceed."""
    if not confirm:
        raise HTTPException(status_code=400, detail="This permanently deletes all data. Pass ?confirm=true to proceed.")

    try:
        reset_all_data()
        return {"message": "All databases cleared successfully."}
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset databases: {e}")

@app.get("/graph/stats")
def get_stats():
    try:
        with db_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM edges")
            edge_count = cursor.fetchone()[0]

        entities_count = get_entities_collection().count()

        return {
            "chunk_count": chunk_count,
            "edge_count": edge_count,
            "total_entities_known": entities_count
        }
    except Exception as e:
        logger.error(f"Stats query failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to read graph stats.")

@app.get("/graph/all")
def get_entire_graph():
    """Returns all nodes and edges currently in the Knowledge Graph for upfront visualization."""
    try:
        with db_cursor() as cursor:
            cursor.execute("""
                SELECT source_entity, relation, target_entity, confidence, chunk_id
                FROM edges
                ORDER BY id ASC
            """)
            rows = cursor.fetchall()
            edges = [
                {
                    "source": r[0],
                    "relation": r[1],
                    "target": r[2],
                    "confidence": r[3],
                    "chunk_id": r[4]
                }
                for r in rows
            ]
            
            node_set = set()
            for e in edges:
                if e["source"]: node_set.add(e["source"])
                if e["target"]: node_set.add(e["target"])
                
            nodes = [{"id": n, "label": n} for n in sorted(node_set)]
            
        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": edges
        }
    except Exception as e:
        logger.error(f"Failed to fetch full graph: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch entire graph.")

def find_query_anchor_entities(query_text: str, query_emb: List[float], max_anchors: int = 3) -> List[Tuple[str, float]]:
    """Identify starting graph entities using exact token substring matching and vector similarity."""
    anchor_scores: Dict[str, float] = {}

    # 1. Substring / Token matching against known entities from SQLite
    try:
        with db_cursor() as cursor:
            cursor.execute("""
                SELECT DISTINCT source_entity FROM edges
                UNION
                SELECT DISTINCT target_entity FROM edges
            """)
            known_entities = [row[0] for row in cursor.fetchall() if row[0]]

        q_lower = query_text.lower()
        for ent in known_entities:
            if ent.lower() in q_lower:
                anchor_scores[ent] = 1.0
    except Exception as e:
        logger.warning(f"Error querying known entities for substring match: {e}")

    # 2. Vector search against entities collection
    entities_col = get_entities_collection()
    if entities_col.count() > 0:
        try:
            n_search = min(5, entities_col.count())
            entity_res = entities_col.query(
                query_embeddings=[query_emb],
                n_results=n_search
            )
            if entity_res and entity_res.get('ids') and len(entity_res['ids'][0]) > 0:
                for idx, ent_id in enumerate(entity_res['ids'][0]):
                    dist = entity_res['distances'][0][idx] if entity_res.get('distances') else 0.0
                    sim = 1.0 - dist
                    # Include if reasonably close or if we have few anchors
                    if sim >= 0.35:
                        if ent_id not in anchor_scores or sim > anchor_scores[ent_id]:
                            anchor_scores[ent_id] = float(sim)
        except Exception as e:
            logger.warning(f"Error during vector entity query: {e}")

    sorted_anchors = sorted(anchor_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_anchors[:max_anchors]

@app.get("/query")
def query_chunks(
    q: str = Query(..., min_length=1),
    k: int = Query(3, ge=1, le=20),
    enable_graph: bool = Query(True, description="Toggle graph traversal augmentation")
):
    try:
        emb = get_embedding(q)
    except Exception as e:
        logger.error(f"Query embedding failed: {e}")
        raise HTTPException(status_code=503, detail="Embedding service unavailable. Is Ollama running?")

    try:
        # 1. Pure Vector Search (Semantic Chunk Search)
        chunks_col = get_chunks_collection()
        res = chunks_col.query(
            query_embeddings=[emb],
            n_results=k
        )

        vector_results = []
        retrieved_chunk_ids = set()
        if res and res.get('ids') and len(res['ids']) > 0 and res['ids'][0]:
            for idx in range(len(res['ids'][0])):
                chunk_id = res['ids'][0][idx]
                text = res['documents'][0][idx] if res.get('documents') and res['documents'][0] else None
                distance = res['distances'][0][idx] if res.get('distances') and res['distances'][0] else None
                metadata = res['metadatas'][0][idx] if res.get('metadatas') and res['metadatas'][0] else None

                vector_results.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "distance": distance,
                    "metadata": metadata
                })
                retrieved_chunk_ids.add(chunk_id)

        # 2. Multi-Entity Graph Traversal (2 Hops, if enabled)
        graph_results = []
        connected_chunks = []

        if enable_graph:
            anchor_entities = find_query_anchor_entities(q, emb, max_anchors=3)
            traversed_edges_set = set()
            traversed_chunk_ids = set()

            if anchor_entities:
                anchor_names = {a[0] for a in anchor_entities}
                with db_cursor() as cursor:
                    # Hop 1 traversal
                    hop1_neighbors = set()
                    for ent, score in anchor_entities:
                        cursor.execute("""
                            SELECT source_entity, relation, target_entity, confidence, chunk_id
                            FROM edges
                            WHERE source_entity = ? OR target_entity = ?
                            ORDER BY confidence DESC
                            LIMIT 10
                        """, (ent, ent))
                        for row in cursor.fetchall():
                            edge_key = (row[0], row[1], row[2])
                            if edge_key not in traversed_edges_set:
                                traversed_edges_set.add(edge_key)
                                graph_results.append({
                                    "source": row[0],
                                    "relation": row[1],
                                    "target": row[2],
                                    "confidence": row[3],
                                    "found_in_chunk": row[4]
                                })
                                if row[4]:
                                    traversed_chunk_ids.add(row[4])
                            # Collect next hops
                            neighbor = row[2] if row[0] == ent else row[0]
                            hop1_neighbors.add(neighbor)

                    # Hop 2 traversal (expand top neighbor entities, avoiding immediate starting anchors)
                    next_hop_candidates = [n for n in hop1_neighbors if n not in anchor_names]
                    for neighbor in next_hop_candidates[:3]:
                        cursor.execute("""
                            SELECT source_entity, relation, target_entity, confidence, chunk_id
                            FROM edges
                            WHERE source_entity = ? OR target_entity = ?
                            ORDER BY confidence DESC
                            LIMIT 5
                        """, (neighbor, neighbor))
                        for row in cursor.fetchall():
                            edge_key = (row[0], row[1], row[2])
                            if edge_key not in traversed_edges_set:
                                traversed_edges_set.add(edge_key)
                                graph_results.append({
                                    "source": row[0],
                                    "relation": row[1],
                                    "target": row[2],
                                    "confidence": row[3],
                                    "found_in_chunk": row[4]
                                })
                                if row[4]:
                                    traversed_chunk_ids.add(row[4])

                primary_anchor = anchor_entities[0][0] if anchor_entities else None
                graph_metadata = {
                    "traversed_from_entity": primary_anchor,
                    "anchor_entities": [a[0] for a in anchor_entities],
                    "edge_count": len(graph_results)
                }
            else:
                graph_metadata = {"message": "No relevant entities found in graph for this query."}

            # 3. Fetch connected chunks' text from SQLite (for chunks not already retrieved via vector search)
            needed_chunk_ids = list(traversed_chunk_ids - retrieved_chunk_ids)
            if needed_chunk_ids:
                with db_cursor() as cursor:
                    placeholders = ",".join("?" * len(needed_chunk_ids))
                    cursor.execute(f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({placeholders})", needed_chunk_ids)
                    for row in cursor.fetchall():
                        connected_chunks.append({
                            "chunk_id": row[0],
                            "text": row[1]
                        })
        else:
            graph_metadata = {"message": "Graph traversal is disabled."}

        # 4. Generate grounded LLM answer using combined context
        try:
            answer = generate_answer(
                query=q,
                vector_chunks=vector_results,
                graph_edges=graph_results,
                traversed_chunks=connected_chunks
            )
        except Exception as gen_err:
            logger.error(f"Answer generation failed: {gen_err}")
            answer = "Error generating answer from LLM. Please check Ollama connection."

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail="Query failed while searching vectors/graph.")

    return {
        "query": q,
        "answer": answer,
        "vector_search_results": vector_results,
        "graph_traversal": {
            "metadata": graph_metadata,
            "edges": graph_results,
            "connected_chunks": connected_chunks
        }
    }

class QueryRequest(BaseModel):
    query: str
    k: int = 3
    enable_graph: bool = True

class AnswerResponse(BaseModel):
    query: str
    answer: str

@app.get("/answer", response_model=AnswerResponse)
def get_answer_endpoint(
    q: str = Query(..., min_length=1, description="Question to answer"),
    k: int = Query(3, ge=1, le=20),
    enable_graph: bool = Query(True, description="Toggle graph traversal")
):
    """Direct answer endpoint returning only the synthesized grounded answer."""
    full_result = query_chunks(q=q, k=k, enable_graph=enable_graph)
    return AnswerResponse(query=full_result["query"], answer=full_result["answer"])

@app.post("/answer", response_model=AnswerResponse)
def post_answer_endpoint(req: QueryRequest):
    """Direct POST answer endpoint returning only the synthesized grounded answer."""
    full_result = query_chunks(q=req.query, k=req.k, enable_graph=req.enable_graph)
    return AnswerResponse(query=full_result["query"], answer=full_result["answer"])

# Mount static files so opening http://localhost:8000/ directly serves the web UI
if os.path.exists("web"):
    app.mount("/", StaticFiles(directory="web", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


