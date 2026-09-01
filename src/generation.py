import ollama
import logging
import time
from pydantic import BaseModel, Field
from typing import List, Optional
try:
    from .config import config
except ImportError:
    from src.config import config

logger = logging.getLogger("graphanchor")

class Relation(BaseModel):
    entity: str = Field(description="The source entity")
    relation: str = Field(description="The relationship between source and target")
    target_entity: str = Field(description="The target entity")

class GraphExtraction(BaseModel):
    entities: List[str] = Field(description="List of all unique entities extracted from the text")
    relations: List[Relation] = Field(description="List of relationships between entities")

def extract_graph_from_chunk(text: str) -> GraphExtraction:
    """Extract entities and relationships from text using Ollama JSON mode.
    Retries on transient failures (connection issues or malformed JSON) instead
    of failing the whole ingest on the first hiccup."""
    prompt = f"Extract all entities and relationships from the following text:\n\n{text}"

    last_err = None
    for attempt in range(config.ollama_max_retries + 1):
        try:
            response = ollama.chat(
                model=config.llm_model,
                messages=[
                    {"role": "system", "content": "You are a graph extraction AI. Always respond in JSON matching the requested schema. Do not add any conversational text."},
                    {"role": "user", "content": prompt}
                ],
                format=GraphExtraction.model_json_schema(),
                options={"num_ctx": 4096}
            )
            return GraphExtraction.model_validate_json(response['message']['content'])
        except Exception as e:
            last_err = e
            logger.warning(f"Graph extraction attempt {attempt + 1} failed: {e}")
            if attempt < config.ollama_max_retries:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Graph extraction failed after {config.ollama_max_retries + 1} attempt(s): {last_err}") from last_err

def generate_answer(
    query: str,
    vector_chunks: Optional[List[dict]] = None,
    graph_edges: Optional[List[dict]] = None,
    traversed_chunks: Optional[List[dict]] = None
) -> str:
    """Generate grounded answer from retrieved text chunks and knowledge graph facts."""
    vector_chunks = vector_chunks or []
    graph_edges = graph_edges or []
    traversed_chunks = traversed_chunks or []

    # If completely empty context
    if not vector_chunks and not graph_edges and not traversed_chunks:
        return "I could not find any relevant information in the knowledge base to answer your question."

    # Collect and deduplicate unique text chunks
    seen_texts = set()
    text_contexts = []

    for item in vector_chunks:
        txt = item.get("text")
        if txt and txt.strip() and txt not in seen_texts:
            seen_texts.add(txt)
            text_contexts.append(txt.strip())

    for item in traversed_chunks:
        txt = item.get("text")
        if txt and txt.strip() and txt not in seen_texts:
            seen_texts.add(txt)
            text_contexts.append(txt.strip())

    # Format graph triples cleanly without confidence scores
    edge_contexts = []
    for edge in graph_edges:
        src = edge.get("source")
        rel = edge.get("relation")
        tgt = edge.get("target")
        if src and rel and tgt:
            edge_contexts.append(f"- {src} -> {rel} -> {tgt}")

    context_parts = []
    if text_contexts:
        passages = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(text_contexts))
        context_parts.append(f"### Relevant Text Passages:\n{passages}")
    if edge_contexts:
        triples = "\n".join(edge_contexts)
        context_parts.append(f"### Knowledge Graph Relationships:\n{triples}")

    full_context = "\n\n".join(context_parts)

    system_prompt = (
        "You are GraphAnchor, a concise and direct factual assistant. "
        "Answer the user's question directly, clearly, and naturally using ONLY the provided context and knowledge graph facts.\n\n"
        "Strict Guidelines:\n"
        "1. Answer directly in 1-3 sentences without repeating the question.\n"
        "2. Do NOT include preambles, boilerplate, or meta-commentary (never say 'Based on the provided context', 'According to the knowledge graph', 'Therefore the answer is', or 'I can answer the question').\n"
        "3. Synthesize facts across different passages and relationships smoothly into natural sentences.\n"
        "4. If the context does not contain enough information, simply state what is unknown."
    )

    user_prompt = f"Context:\n{full_context}\n\nQuestion: {query}\n\nAnswer:"

    last_err = None
    for attempt in range(config.ollama_max_retries + 1):
        try:
            response = ollama.chat(
                model=config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={"num_ctx": 4096, "temperature": 0.1}
            )
            return response['message']['content'].strip()
        except Exception as e:
            last_err = e
            logger.warning(f"Answer generation attempt {attempt + 1} failed: {e}")
            if attempt < config.ollama_max_retries:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Answer generation failed after {config.ollama_max_retries + 1} attempt(s): {last_err}") from last_err

if __name__ == "__main__":
    sample_text = "Apple was founded by Steve Jobs and Steve Wozniak in Cupertino, California."
    print("Extracting graph...")
    res = extract_graph_from_chunk(sample_text)
    print(res.model_dump_json(indent=2))

