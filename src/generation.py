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
    Resolves coreferences, eliminates pronouns/verbs as nodes, and runs deterministically (temp=0.0)."""
    system_prompt = (
        "You are the GraphAnchor Knowledge Graph Extraction Engine, a specialized system for converting unstructured text into structured (Entity, Relation, Target) triples.\n"
        "Your objective is to extract high-precision facts, relationships, attributes, roles, and dependencies from the provided text.\n\n"
        "Strict Extraction Rules:\n"
        "1. Entity Recognition & Canonicalization:\n"
        "   - Identify clear, distinct named entities: People, Projects, Organizations, Facilities, Components, Technologies, Conditions, Chemicals, Locations, and Roles.\n"
        "   - Use proper canonical casing and exact names (e.g., 'Dr. Aris Thorne', 'Munich Foundry', 'Inhibitor-Z').\n"
        "2. Coreference & Pronoun Resolution:\n"
        "   - ALWAYS resolve anaphoric pronouns ('he', 'she', 'they', 'it', 'his', 'her', 'their', 'its') to the primary named entity referenced in the text.\n"
        "   - NEVER create entity nodes with pronoun names like 'He', 'She', 'It', or 'They'.\n"
        "3. First-Person & Possessive Normalization:\n"
        "   - Strip first-person possessives from roles and targets (convert 'my project partner' -> 'project partner', 'our lead engineer' -> 'lead engineer').\n"
        "   - NEVER output 'I', 'me', 'my', or 'we' as entity names.\n"
        "4. Relation Formatting:\n"
        "   - Use concise, meaningful verb phrases (e.g., 'leads', 'specializes_in', 'manufactured_by', 'partner_of', 'located_in', 'reports_to', 'authenticates_with', 'secured_by', 'is').\n"
        "   - NEVER output standalone verbs ('is', 'has', 'was') as entity names.\n"
        "5. Output Schema:\n"
        "   - Return strictly valid JSON containing the list of unique 'entities' and 'relations' matching the requested schema."
    )

    user_prompt = f"Extract all factual entities and relationships from the following text:\n\n{text}"

    last_err = None
    for attempt in range(config.ollama_max_retries + 1):
        try:
            response = ollama.chat(
                model=config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                format=GraphExtraction.model_json_schema(),
                options={"num_ctx": 4096, "temperature": 0.0}
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

    # Format graph triples cleanly
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
        "You are GraphAnchor, an advanced factual question-answering and multi-hop reasoning engine.\n"
        "Your task is to provide an accurate, completely grounded, and objective answer to the user's question using the provided context.\n\n"
        "Reasoning & Synthesis Instructions:\n"
        "1. Objective Third-Person Voice:\n"
        "   - ALWAYS formulate your response using an objective, neutral third-person perspective.\n"
        "   - NEVER use first-person pronouns ('I', 'me', 'my', 'we', 'our') or second-person pronouns ('you', 'your'), even if the source document was written in the first person.\n"
        "2. Cross-Document Multi-Hop Bridging:\n"
        "   - When answering questions that require traversing multiple facts, explicitly state the relational bridge linking the starting entity, intermediate components, and the final answer entity.\n"
        "   - Example format: '[Entity A], which [relates to Entity B], is [connected/managed/secured by Entity C].'\n"
        "3. Strict Evidence Grounding:\n"
        "   - Rely ONLY on the facts explicitly stated in the provided text passages and knowledge graph relationships.\n"
        "   - Do NOT extrapolate, hallucinate, or assume facts not present in the evidence.\n"
        "4. Tone and Style:\n"
        "   - Be direct, articulate, and complete (1-3 well-constructed sentences).\n"
        "   - Avoid conversational filler, introductory preambles (e.g., 'Based on the provided text', 'According to the graph'), disclaimers, or meta-commentary."
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

