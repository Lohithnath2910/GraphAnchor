import os
import uuid
import chromadb
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List, Dict

from src.config import config
from src.ingestion import chunk_text
from src.storage import get_db_connection, collection, chroma_client
from src.retrieval import get_embedding
from src.generation import extract_graph_from_chunk, GraphExtraction

app = FastAPI(title="GraphAnchor")

# Ensure entities collection uses cosine distance for proper thresholding
entities_collection = chroma_client.get_or_create_collection(name="entities", metadata={"hnsw:space": "cosine"})

class IngestResponse(BaseModel):
    doc_id: str
    chunks_processed: int
    entities_extracted: int
    edges_added: int

def place_entity(entity_str: str) -> tuple[str, float]:
    """Returns canonical entity name and confidence"""
    if not entity_str:
        return "", 0.0
        
    # 1. Exact match
    try:
        res = entities_collection.get(ids=[entity_str])
        if res and res.get('ids') and len(res['ids']) > 0:
            return entity_str, 1.0
    except Exception:
        pass
        
    # 2. Embedding similarity fallback
    emb = get_embedding(entity_str)
    
    # If DB is empty, just insert
    if entities_collection.count() > 0:
        search_res = entities_collection.query(
            query_embeddings=[emb],
            n_results=1
        )
        
        if search_res and search_res.get('distances') and len(search_res['distances'][0]) > 0:
            dist = search_res['distances'][0][0]
            # Chroma cosine distance is 1 - cosine_similarity
            sim = 1.0 - dist
            if sim >= config.similarity_threshold:
                canonical = search_res['ids'][0][0]
                return canonical, float(sim)
            
    # 3. Insert as unconnected island
    entities_collection.add(
        ids=[entity_str],
        embeddings=[emb],
        documents=[entity_str]
    )
    return entity_str, 1.0

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())
    content = await file.read()
    text = content.decode("utf-8")
    
    chunks = chunk_text(text)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    total_entities = 0
    total_edges = 0
    
    for i, chunk_text_content in enumerate(chunks):
        chunk_id = f"{doc_id}_{i}"
        
        # Embed chunk
        chunk_emb = get_embedding(chunk_text_content)
        
        # Add to ChromaDB chunks
        collection.add(
            ids=[chunk_id],
            embeddings=[chunk_emb],
            documents=[chunk_text_content],
            metadatas=[{"chunk_id": chunk_id, "doc_id": doc_id}]
        )
        
        # Add to SQLite chunks
        cursor.execute("INSERT INTO chunks (chunk_id, doc_id, text) VALUES (?, ?, ?)", 
                      (chunk_id, doc_id, chunk_text_content))
                      
        # Extract graph
        try:
            extraction = extract_graph_from_chunk(chunk_text_content)
        except Exception as e:
            print(f"Extraction failed for chunk {chunk_id}: {e}")
            continue
            
        total_entities += len(extraction.entities)
        
        for rel in extraction.relations:
            src_canonical, src_conf = place_entity(rel.entity)
            tgt_canonical, tgt_conf = place_entity(rel.target_entity)
            
            if not src_canonical or not tgt_canonical:
                continue
                
            edge_conf = min(src_conf, tgt_conf)
            
            cursor.execute("""
                INSERT INTO edges (source_entity, relation, target_entity, chunk_id, confidence)
                VALUES (?, ?, ?, ?, ?)
            """, (src_canonical, rel.relation, tgt_canonical, chunk_id, edge_conf))
            total_edges += 1
            
    conn.commit()
    conn.close()
    
    return IngestResponse(
        doc_id=doc_id,
        chunks_processed=len(chunks),
        entities_extracted=total_entities,
        edges_added=total_edges
    )

@app.delete("/reset")
def reset_databases():
    """Clear all data from SQLite and ChromaDB"""
    global entities_collection, collection
    
    # 1. Clear SQLite
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chunks")
    cursor.execute("DELETE FROM edges")
    conn.commit()
    conn.close()
    
    # 2. Clear ChromaDB by deleting and recreating collections
    try:
        chroma_client.delete_collection("chunks")
    except Exception:
        pass
        
    try:
        chroma_client.delete_collection("entities")
    except Exception:
        pass
        
    # Recreate them so they are ready for the next ingest
    collection = chroma_client.get_or_create_collection(name="chunks")
    entities_collection = chroma_client.get_or_create_collection(name="entities", metadata={"hnsw:space": "cosine"})
    
    return {"message": "All databases cleared successfully."}

@app.get("/graph/stats")
def get_stats():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM chunks")
    chunk_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM edges")
    edge_count = cursor.fetchone()[0]
    
    entities_count = entities_collection.count()
    
    return {
        "chunk_count": chunk_count,
        "edge_count": edge_count,
        "total_entities_known": entities_count
    }

@app.get("/query")
def query_chunks(q: str, k: int = 3):
    emb = get_embedding(q)
    
    # 1. Pure Vector Search (Standard RAG)
    res = collection.query(
        query_embeddings=[emb],
        n_results=k
    )
    
    vector_results = []
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
            
    # 2. Graph Traversal Search (The Novelty Claim!)
    graph_results = []
    
    # Find the most relevant entity to the user's query
    entity_res = entities_collection.query(
        query_embeddings=[emb],
        n_results=1
    )
    
    if entity_res and entity_res.get('ids') and len(entity_res['ids'][0]) > 0:
        top_entity = entity_res['ids'][0][0]
        
        # Safely get the distance, default to 0.0 if not available
        distances = entity_res.get('distances')
        if distances is not None and len(distances) > 0 and len(distances[0]) > 0:
            entity_distance = distances[0][0]
        else:
            entity_distance = 0.0
        
        # Only traverse if the entity is reasonably relevant to the query
        if entity_distance < 0.5:  
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Fetch all edges connected to this entity
            cursor.execute("""
                SELECT source_entity, relation, target_entity, confidence, chunk_id 
                FROM edges 
                WHERE source_entity = ? OR target_entity = ?
                ORDER BY confidence DESC
            """, (top_entity, top_entity))
            
            edges = cursor.fetchall()
            conn.close()
            
            for edge in edges:
                graph_results.append({
                    "source": edge[0],
                    "relation": edge[1],
                    "target": edge[2],
                    "confidence": edge[3],
                    "found_in_chunk": edge[4]
                })
                
            # Attach the top entity we traversed from
            graph_metadata = {
                "traversed_from_entity": top_entity,
                "entity_relevance_distance": entity_distance
            }
        else:
            graph_metadata = {"message": "No highly relevant entities found in graph for this query."}
    else:
        graph_metadata = {"message": "Graph is empty."}

    return {
        "query": q, 
        "vector_search_results": vector_results,
        "graph_traversal": {
            "metadata": graph_metadata,
            "edges": graph_results
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
