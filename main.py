import os
import sys
import uuid
import hashlib
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple

from src.config import config
from src.ingestion import chunk_text, extract_text_from_file
from src.storage import (
    db_cursor,
    get_chunks_collection,
    get_entities_collection,
    reset_all_data,
    list_documents,
    delete_document
)
from src.retrieval import get_embedding
from src.generation import (
    extract_graph_from_chunk,
    generate_answer,
    stream_answer,
    GraphExtraction
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("graphanchor")


app = FastAPI(title="GraphAnchor")

@app.get("/docs/", include_in_schema=False)
def redirect_docs():
    return RedirectResponse(url="/docs")

@app.get("/redoc/", include_in_schema=False)
def redirect_redoc():
    return RedirectResponse(url="/redoc")

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
    # Returns canonical entity name and confidence by filtering invalid words and performing fuzzy deduplication.
    if not entity_str:
        return "", 0.0

    entity_clean = entity_str.strip()
    if not entity_clean or len(entity_clean) <= 1:
        return "", 0.0

    if entity_clean.lower() in INVALID_ENTITIES:
        return "", 0.0

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

    try:
        res = entities_col.get(ids=[entity_clean])
        if res and res.get('ids') and len(res['ids']) > 0:
            return entity_clean, 1.0
    except Exception as e:
        logger.warning(f"Exact-match lookup failed for '{entity_clean}': {e}")

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
    # Handles uploading a document, extracting its text, generating graph triples, and storing the data.
    allowed_exts = {".txt", ".md", ".markdown", ".pdf"}
    fname = file.filename or ""
    file_ext = os.path.splitext(fname.lower())[1]
    if file_ext not in allowed_exts:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file format '{file_ext}'. Supported formats: {', '.join(sorted(allowed_exts))}"
        )

    content = await file.read()

    max_bytes = int(config.max_file_size_mb * 1024 * 1024)
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds max size of {config.max_file_size_mb} MB.")

    try:
        text = extract_text_from_file(fname, content)
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as parse_err:
        raise HTTPException(status_code=400, detail=f"Failed to read file content: {parse_err}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="File is empty or contains no extractable text.")

    content_hash = hashlib.sha256(content).hexdigest()

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

                chunk_emb = get_embedding(chunk_text_content)

                chunks_col.add(
                    ids=[chunk_id],
                    embeddings=[chunk_emb],
                    documents=[chunk_text_content],
                    metadatas=[{"chunk_id": chunk_id, "doc_id": doc_id, "filename": fname}]
                )
                staged_chunk_ids.append(chunk_id)

                cursor.execute(
                    "INSERT INTO chunks (chunk_id, doc_id, text) VALUES (?, ?, ?)",
                    (chunk_id, doc_id, chunk_text_content)
                )

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
    # Clears all data from SQLite and ChromaDB permanently.
    if not confirm:
        raise HTTPException(status_code=400, detail="This permanently deletes all data. Pass ?confirm=true to proceed.")

    try:
        reset_all_data()
        return {"message": "All databases cleared successfully."}
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset databases: {e}")

@app.get("/documents")
def get_documents_endpoint():
    # Lists all ingested documents with chunk and graph edge counts.
    try:
        docs = list_documents()
        return {"documents": docs, "total_documents": len(docs)}
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents.")

@app.get("/documents/{doc_id}")
def get_single_document_endpoint(doc_id: str):
    # Retrieves metadata and stored chunks for a specific document.
    try:
        with db_cursor() as cursor:
            cursor.execute("SELECT doc_id, filename, content_hash, created_at FROM documents WHERE doc_id = ?", (doc_id,))
            doc = cursor.fetchone()
            if not doc:
                raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
            
            cursor.execute("SELECT chunk_id, text FROM chunks WHERE doc_id = ? ORDER BY chunk_id ASC", (doc_id,))
            chunks = [{"chunk_id": r[0], "text": r[1]} for r in cursor.fetchall()]

            cursor.execute("""
                SELECT source_entity, relation, target_entity, confidence, chunk_id
                FROM edges
                WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id = ?)
            """, (doc_id,))
            edges = [{"source": r[0], "relation": r[1], "target": r[2], "confidence": r[3], "chunk_id": r[4]} for r in cursor.fetchall()]

        return {
            "doc_id": doc[0],
            "filename": doc[1],
            "content_hash": doc[2],
            "created_at": doc[3],
            "chunk_count": len(chunks),
            "edge_count": len(edges),
            "chunks": chunks,
            "edges": edges
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch document.")

@app.delete("/documents/{doc_id}")
def delete_single_document_endpoint(doc_id: str):
    # Selectively deletes a document and all its associated chunks and edges.
    try:
        success = delete_document(doc_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
        return {"message": f"Document '{doc_id}' and all associated chunks and edges deleted successfully."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {e}")

@app.get("/graph/stats")
def get_stats():
    # Returns aggregate statistics for the entire database.
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
    # Returns all nodes and edges currently in the Knowledge Graph.
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
    # Identifies starting graph entities using exact token matching and vector similarity.
    anchor_scores: Dict[str, float] = {}

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
                    if sim >= config.vector_anchor_confidence:
                        if ent_id not in anchor_scores or sim > anchor_scores[ent_id]:
                            anchor_scores[ent_id] = float(sim)
        except Exception as e:
            logger.warning(f"Error during vector entity query: {e}")

    sorted_anchors = sorted(anchor_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_anchors[:max_anchors]

def rank_and_fuse_chunks(
    vector_results: List[Dict],
    connected_chunks: List[Dict],
    graph_edges: List[Dict]
) -> List[Dict]:
    # Combines vector similarity search results and graph traversal chunks using hybrid score fusion.
    fused: Dict[str, Dict] = {}

    for rank, v in enumerate(vector_results):
        cid = v["chunk_id"]
        dist = v.get("distance")
        vec_score = max(0.0, 1.0 - dist) if dist is not None else 0.8
        fused[cid] = {
            "chunk_id": cid,
            "text": v.get("text", ""),
            "metadata": v.get("metadata") or {},
            "vector_score": round(vec_score, 4),
            "graph_score": 0.0,
            "composite_score": round(vec_score, 4),
            "source_type": "vector",
            "hop_distance": 0
        }

    edge_by_chunk: Dict[str, List[Dict]] = {}
    for e in graph_edges:
        found_cid = e.get("found_in_chunk")
        if found_cid:
            edge_by_chunk.setdefault(found_cid, []).append(e)

    for g in connected_chunks:
        cid = g["chunk_id"]
        edges = edge_by_chunk.get(cid, [])
        max_conf = max([e.get("confidence", 1.0) for e in edges], default=config.graph_edge_base_confidence)
        hop = 1 if cid in edge_by_chunk else 2
        hop_decay = 0.8 ** hop
        graph_score = max_conf * hop_decay

        if cid in fused:
            v_score = fused[cid]["vector_score"]
            composite = config.hybrid_fusion_alpha * v_score + (1.0 - config.hybrid_fusion_alpha) * graph_score + 0.15
            fused[cid]["graph_score"] = round(graph_score, 4)
            fused[cid]["composite_score"] = round(min(1.0, composite), 4)
            fused[cid]["source_type"] = "hybrid"
            fused[cid]["hop_distance"] = hop
        else:
            composite = graph_score * 0.9
            fused[cid] = {
                "chunk_id": cid,
                "text": g.get("text", ""),
                "metadata": g.get("metadata") or {},
                "vector_score": 0.0,
                "graph_score": round(graph_score, 4),
                "composite_score": round(composite, 4),
                "source_type": "graph",
                "hop_distance": hop
            }

    ranked = sorted(fused.values(), key=lambda x: x["composite_score"], reverse=True)
    return ranked

@app.get("/query")
def query_chunks(
    q: str = Query(..., min_length=1),
    k: int = Query(3, ge=1, le=20),
    enable_graph: bool = Query(True, description="Toggle graph traversal augmentation")
):
    # Executes hybrid search combining vector retrieval and graph traversal to generate an answer.
    try:
        emb = get_embedding(q)
    except Exception as e:
        logger.error(f"Query embedding failed: {e}")
        raise HTTPException(status_code=503, detail="Embedding service unavailable. Is Ollama running?")

    try:
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

        graph_results = []
        connected_chunks = []

        if enable_graph:
            anchor_entities = find_query_anchor_entities(q, emb, max_anchors=5)
            traversed_edges_set = set()
            traversed_chunk_ids = set()

            if anchor_entities:
                visited_nodes = set()
                current_frontier = {a[0] for a in anchor_entities}

                with db_cursor() as cursor:
                    for depth in range(4):
                        if not current_frontier or len(traversed_edges_set) >= 35:
                            break

                        next_frontier = set()
                        for node in current_frontier:
                            if node in visited_nodes:
                                continue
                            visited_nodes.add(node)

                            cursor.execute("""
                                SELECT source_entity, relation, target_entity, confidence, chunk_id
                                FROM edges
                                WHERE source_entity = ? OR target_entity = ?
                                ORDER BY confidence DESC
                                LIMIT 10
                            """, (node, node))

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

                                neighbor = row[2] if row[0] == node else row[0]
                                if neighbor not in visited_nodes:
                                    next_frontier.add(neighbor)

                        current_frontier = next_frontier

                primary_anchor = anchor_entities[0][0] if anchor_entities else None
                graph_metadata = {
                    "traversed_from_entity": primary_anchor,
                    "anchor_entities": [a[0] for a in anchor_entities],
                    "edge_count": len(graph_results)
                }
            else:
                graph_metadata = {"message": "No relevant entities found in graph for this query."}

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

        ranked_chunks = rank_and_fuse_chunks(
            vector_results=vector_results,
            connected_chunks=connected_chunks,
            graph_edges=graph_results
        )

        try:
            answer = generate_answer(
                query=q,
                vector_chunks=ranked_chunks,
                graph_edges=graph_results
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
        },
        "ranking_breakdown": ranked_chunks
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
    # Direct GET endpoint returning only the synthesized grounded answer.
    full_result = query_chunks(q=q, k=k, enable_graph=enable_graph)
    return AnswerResponse(query=full_result["query"], answer=full_result["answer"])

@app.post("/answer", response_model=AnswerResponse)
def post_answer_endpoint(req: QueryRequest):
    # Direct POST endpoint returning only the synthesized grounded answer.
    full_result = query_chunks(q=req.query, k=req.k, enable_graph=req.enable_graph)
    return AnswerResponse(query=full_result["query"], answer=full_result["answer"])

@app.get("/answer/stream")
def get_answer_stream_endpoint(
    q: str = Query(..., min_length=1, description="Question to answer"),
    k: int = Query(3, ge=1, le=20),
    enable_graph: bool = Query(True, description="Toggle graph traversal")
):
    # Server-Sent Events (SSE) endpoint streaming grounded answer tokens in real time.
    full_result = query_chunks(q=q, k=k, enable_graph=enable_graph)
    ranked_chunks = full_result.get("ranking_breakdown", [])
    graph_edges = full_result.get("graph_traversal", {}).get("edges", [])

    def event_stream():
        meta = {
            "query": full_result["query"],
            "anchors": full_result.get("graph_traversal", {}).get("metadata", {}).get("anchor_entities", []),
            "chunk_count": len(ranked_chunks),
            "edge_count": len(graph_edges)
        }
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
        for token in stream_answer(query=q, vector_chunks=ranked_chunks, graph_edges=graph_edges):
            yield f"event: token\ndata: {json.dumps(token)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/answer/stream")
def post_answer_stream_endpoint(req: QueryRequest):
    # Server-Sent Events (SSE) POST endpoint streaming grounded answer tokens in real time.
    return get_answer_stream_endpoint(q=req.query, k=req.k, enable_graph=req.enable_graph)

if os.path.exists("web"):
    app.mount("/", StaticFiles(directory="web", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


