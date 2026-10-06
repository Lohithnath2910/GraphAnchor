import logging
import re
import time

import ollama
from pydantic import BaseModel, Field

try:
    from .config import config
except ImportError:
    from src.config import config

logger = logging.getLogger("graphanchor")

# Extraction client with a hard timeout so one looping generation can't hang ingestion.
_extract_client = ollama.Client(timeout=90)
# Answer client: timeout so a stuck generation cannot hang a request forever.
_answer_client = ollama.Client(timeout=120)
# Answers are 1-3 sentences; the cap stops a looping generation from running until the context fills.
ANSWER_OPTIONS = {"num_ctx": 8192, "temperature": 0.1, "num_predict": 300}

class Relation(BaseModel):
    entity: str = Field(description="The source entity")
    relation: str = Field(description="The relationship between source and target")
    target_entity: str = Field(description="The target entity")

class GraphExtraction(BaseModel):
    entities: list[str] = Field(description="List of all unique entities extracted from the text")
    relations: list[Relation] = Field(description="List of relationships between entities")

def extract_graph_from_chunk(text: str) -> GraphExtraction:
    # Extracts factual entities and relationships from text using Ollama JSON mode with a fixed schema.
    system_prompt = (
        "You are the GraphAnchor Knowledge Graph Extraction Engine, a specialized system for converting unstructured text into structured (Entity, Relation, Target) triples.\n"
        "Your objective is to extract high-precision facts, relationships, attributes, roles, and dependencies from the provided text.\n\n"
        "Strict Extraction Rules:\n"
        "1. Entity Recognition & Canonicalization:\n"
        "   - Identify clear, distinct named entities: People, Projects, Organizations, Facilities, Components, Technologies, Conditions, Chemicals, Locations, and Roles.\n"
        "   - Use proper canonical casing and exact names exactly as written in the text. Never use names that do not appear in the text.\n"
        "2. Coreference & Pronoun Resolution:\n"
        "   - ALWAYS resolve anaphoric pronouns ('he', 'she', 'they', 'it', 'his', 'her', 'their', 'its') to the primary named entity referenced in the text.\n"
        "   - NEVER create entity nodes with pronoun names like 'He', 'She', 'It', or 'They'.\n"
        "3. First-Person & Possessive Normalization:\n"
        "   - Strip first-person possessives from roles and targets (convert 'my project partner' -> 'project partner', 'our lead engineer' -> 'lead engineer').\n"
        "   - NEVER output 'I', 'me', 'my', or 'we' as entity names.\n"
        "4. Relation Formatting:\n"
        "   - Use concise, meaningful verb phrases (e.g., 'leads', 'specializes_in', 'manufactured_by', 'partner_of', 'located_in', 'reports_to', 'authenticates_with', 'secured_by', 'is').\n"
        "   - Complex actions or intentions MUST be captured entirely in the relation (e.g. 'plans_to_destroy', 'wants_to_use', 'intends_to_sabotage').\n"
        "   - NEVER output standalone verbs ('is', 'has', 'was') as entity names.\n"
        "   - Entities MUST be concrete nouns or names, NOT verbs or actions like 'misuse of'.\n"
        "5. Output Schema:\n"
        "   - Return strictly valid JSON containing the list of unique 'entities' and 'relations' matching the requested schema.\n"
        "6. Brevity:\n"
        f"   - Output at most {config.extraction_max_relations} entities and {config.extraction_max_relations} relations: only the most important facts. Do not enumerate every term in the text."
    )

    user_prompt = f"Extract all factual entities and relationships from the following text:\n\n{text}"

    last_err = None
    for attempt in range(config.ollama_max_retries + 1):
        try:
            response = _extract_client.chat(
                model=config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                format=GraphExtraction.model_json_schema(),
                # repeat_penalty stops the model looping on its entity list; temperature rises on retries so a retry isn't identical
                options={"num_ctx": 4096, "temperature": 0.3 * attempt, "repeat_penalty": 1.3, "num_predict": 1400}
            )
            return GraphExtraction.model_validate_json(response['message']['content'])
        except Exception as e:
            last_err = e
            logger.warning(f"Graph extraction attempt {attempt + 1} failed: {e}")
            if attempt < config.ollama_max_retries:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Graph extraction failed after {config.ollama_max_retries + 1} attempt(s): {last_err}") from last_err

ANSWER_RULES_V1 = (
    "4. Tone and Style:\n"
    "   - Be direct, articulate, concise, and complete (1-3 well-constructed sentences).\n"
    "   - Avoid conversational filler, introductory preambles (e.g., 'Based on the provided text'), disclaimers, or meta-commentary."
)

# V2 adds three rules that target failure modes seen in the benchmark: naming the wrong person from the same passages,
# stopping a list early, and being misled by loosely related passages.
ANSWER_RULES_V2 = (
    "4. Precise, Grounded Answers Only:\n"
    "   - Answer only the question asked, using only facts stated in the provided passages, facts and chains. Never use outside knowledge, "
    "never guess, and never invent a name, number, date, or relationship that is not written in the context.\n"
    "   - Give just the requested fact(s) in one short sentence. Do not add background, explanations, or extra details nobody asked for.\n"
    "   - If the context does not contain the answer, reply exactly: 'The provided information does not contain the answer.' "
    "A partial answer is fine only if every part of it is stated in the context.\n"
    "   - Do not treat a person or entity as the answer merely because it appears nearby; check that the context states the exact "
    "relationship the question describes.\n"
    "5. Answer Exactly What Was Asked:\n"
    "   - Give the specific role or item the question names. If it asks for a deputy, chief engineer, head of a department, or similar, "
    "answer with that person, not with a different person such as the project director who appears in the same passages.\n"
    "   - If the question asks 'Which ...' and expects several items, list every matching item that appears in any passage; do not stop at the first one or two.\n"
    "   - If passages disagree, prefer the passage whose wording matches the question most directly.\n"
    "6. Tone and Style:\n"
    "   - Be direct and concise (one sentence; for list questions, name every item in one sentence).\n"
    "   - Avoid conversational filler, introductory preambles (e.g., 'Based on the provided text'), disclaimers, or meta-commentary."
)

def _build_rag_prompts(
    query: str,
    vector_chunks: list[dict] | None = None,
    graph_edges: list[dict] | None = None,
    traversed_chunks: list[dict] | None = None,
    graph_paths: list[str] | None = None,
    step_facts: list[str] | None = None
) -> tuple[str | None, str | None]:
    # Constructs grounded context and system/user prompts incorporating both text chunks and graph relationships.
    vector_chunks = vector_chunks or []
    graph_edges = graph_edges or []
    traversed_chunks = traversed_chunks or []
    graph_paths = graph_paths or []
    step_facts = step_facts or []

    if not vector_chunks and not graph_edges and not traversed_chunks:
        return None, None

    seen_texts = set()
    text_contexts = []
    extra_contexts = []

    # Combined candidate chunks (ranked vector + traversed)
    combined = list(vector_chunks) + list(traversed_chunks)
    for i, item in enumerate(combined):
        txt = item.get("text")
        if txt and txt.strip() and txt not in seen_texts:
            seen_texts.add(txt)
            meta = item.get("metadata") or {}
            fname = meta.get("filename", "")
            header = f"[Source: {fname}]" if fname else f"[Passage {i+1}]"
            # v2 separates the passages vector search ranked highest from those only the graph added
            target = extra_contexts if (config.prompt_v2 and item.get("source_type") == "graph") else text_contexts
            target.append(f"{header}\n{txt.strip()}")

    edge_contexts = []
    for edge in graph_edges:
        src = edge.get("source")
        rel = edge.get("relation")
        tgt = edge.get("target")
        if src and rel and tgt:
            edge_contexts.append(f"- {src} -> {rel} -> {tgt}")

    context_parts = []
    if step_facts:
        facts = "\n".join(f"- {f}" for f in step_facts)
        context_parts.append(f"### Facts Established Step By Step (each answers part of the question):\n{facts}")
    if graph_paths and config.prompt_v2:
        chains = "\n".join(f"- {p}" for p in graph_paths)
        context_parts.append(f"### Connection Chains (how entities in the passages link to the question):\n{chains}")
    if edge_contexts:
        triples = "\n".join(edge_contexts)
        context_parts.append(f"### Knowledge Graph Relationships:\n{triples}")
    if text_contexts:
        passages = "\n\n".join(text_contexts)
        title = "Primary Passages (most similar to the question)" if extra_contexts else "Relevant Evidence Passages"
        context_parts.append(f"### {title}:\n{passages}")
    if extra_contexts:
        extra = "\n\n".join(extra_contexts)
        context_parts.append("### Additional Linked Passages (found through the knowledge graph; use them only for facts "
                             f"the primary passages do not contain):\n{extra}")

    full_context = "\n\n".join(context_parts)

    system_prompt = (
        "You are GraphAnchor, an advanced factual question-answering and multi-hop reasoning engine.\n"
        "Your task is to provide an accurate, completely grounded, and objective answer to the user's question using the provided context.\n\n"
        "Reasoning & Synthesis Instructions:\n"
        "1. Objective Third-Person Voice:\n"
        "   - ALWAYS formulate your response using an objective, neutral third-person perspective.\n"
        "   - NEVER use first-person pronouns ('I', 'me', 'my', 'we', 'our') or second-person pronouns ('you', 'your'), even if the source document was written in the first person.\n"
        "2. Cross-Document Multi-Hop Bridging:\n"
        "   - When answering questions that require traversing multiple facts, clearly connect the entities.\n"
        "   - Example: '[Entity A], who [relates to Entity B], is [connected/managed/praised by Entity C].'\n"
        "3. Evidence Grounding:\n"
        "   - Rely ONLY on facts explicitly present in the provided evidence passages and graph relationships.\n"
        "   - Formulate natural, fluent, and grammatical sentences.\n"
        "   - NEVER insert raw IDs, UUIDs, or awkward bracketed tags like '[Chunk: ...]' into your text.\n"
        "   - Do NOT extrapolate, hallucinate, or assume facts not present in the evidence.\n"
        + (ANSWER_RULES_V2 if config.prompt_v2 else ANSWER_RULES_V1)
    )

    user_prompt = f"Context:\n{full_context}\n\nQuestion: {query}\n\nAnswer:"
    return system_prompt, user_prompt

def generate_answer(
    query: str,
    vector_chunks: list[dict] | None = None,
    graph_edges: list[dict] | None = None,
    traversed_chunks: list[dict] | None = None,
    graph_paths: list[str] | None = None,
    step_facts: list[str] | None = None
) -> str:
    # Generates a grounded answer from retrieved chunks and graph facts using Ollama.
    system_prompt, user_prompt = _build_rag_prompts(query, vector_chunks, graph_edges, traversed_chunks, graph_paths, step_facts)
    if not system_prompt or not user_prompt:
        return "I could not find any relevant information in the knowledge base to answer your question."

    last_err = None
    for attempt in range(config.ollama_max_retries + 1):
        try:
            response = _answer_client.chat(
                model=config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options=ANSWER_OPTIONS
            )
            return response['message']['content'].strip()
        except Exception as e:
            last_err = e
            logger.warning(f"Answer generation attempt {attempt + 1} failed: {e}")
            if attempt < config.ollama_max_retries:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Answer generation failed after {config.ollama_max_retries + 1} attempt(s): {last_err}") from last_err

class SubQuestions(BaseModel):
    steps: list[str] = Field(description="Sub-questions in the order they must be answered")

DECOMPOSE_SYSTEM = (
    "You split a multi-hop question into simple sub-questions that are answered one after another.\n"
    "Rules: write 2 or 3 steps. Each step asks for exactly one fact. A later step may refer to the answer of an earlier step with "
    "{1} or {2} (the answer of step 1 or step 2). Keep every name from the question exactly as written. "
    "If the question needs only one fact, return it unchanged as the single step.\n"
    "Example question: Who is the chief executive of the company that made the rotor hubs of the Tidewrack Tidal Array?\n"
    'Example steps: ["Which company made the rotor hubs of the Tidewrack Tidal Array?", "Who is the chief executive of {1}?"]\n'
    "Example question: Who chairs the bank that financed the wind farm whose cable fault caused the November 2023 brownout?\n"
    'Example steps: ["Which wind farm had a cable fault that caused the November 2023 brownout?", "Which bank financed {1}?", "Who chairs {2}?"]'
)

def decompose_question(query: str) -> list[str]:
    # Returns the ordered sub-questions, or [] if the model output is unusable or the question is a single hop.
    try:
        response = _extract_client.chat(
            model=config.llm_model,
            messages=[{"role": "system", "content": DECOMPOSE_SYSTEM}, {"role": "user", "content": query}],
            format=SubQuestions.model_json_schema(),
            options={"num_ctx": 2048, "temperature": 0, "num_predict": 220}
        )
        steps = [x.strip() for x in SubQuestions.model_validate_json(response['message']['content']).steps if x.strip()]
    except Exception as e:
        logger.warning(f"Question decomposition failed: {e}")
        return []
    if not (2 <= len(steps) <= config.decompose_max_steps):
        return []
    # {n} may only point at an earlier step
    for i, step in enumerate(steps):
        if any(int(n) > i for n in re.findall(r"\{(\d+)\}", step)):
            return []
    return steps

def short_answer(query: str, vector_chunks: list[dict], graph_edges: list[dict] | None = None) -> str:
    # Answers one intermediate step with only the name or short phrase, so it can be substituted into the next step.
    system_prompt, user_prompt = _build_rag_prompts(query, vector_chunks, graph_edges)
    if not system_prompt or not user_prompt:
        return ""
    system_prompt += "\nIMPORTANT: reply with only the name or short phrase that answers the question, no full sentence."
    try:
        response = _answer_client.chat(
            model=config.llm_model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            options={"num_ctx": 8192, "temperature": 0, "num_predict": 30}
        )
        return response['message']['content'].strip().strip('."\'').split("\n")[0]
    except Exception as e:
        logger.warning(f"Intermediate answer failed: {e}")
        return ""

def stream_answer(
    query: str,
    vector_chunks: list[dict] | None = None,
    graph_edges: list[dict] | None = None,
    traversed_chunks: list[dict] | None = None,
    graph_paths: list[str] | None = None,
    step_facts: list[str] | None = None
):
    # Streams a grounded answer token-by-token from Ollama for real-time UI display.
    system_prompt, user_prompt = _build_rag_prompts(query, vector_chunks, graph_edges, traversed_chunks, graph_paths, step_facts)
    if not system_prompt or not user_prompt:
        yield "I could not find any relevant information in the knowledge base to answer your question."
        return

    try:
        response_stream = _answer_client.chat(
            model=config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=True,
            options=ANSWER_OPTIONS
        )
        for chunk in response_stream:
            content = chunk.get('message', {}).get('content', '')
            if content:
                yield content
    except Exception as e:
        logger.error(f"Streaming failed: {e}")
        yield f"\n[Error generating streamed answer: {e}]"

if __name__ == "__main__":
    sample_text = "Apple was founded by Steve Jobs and Steve Wozniak in Cupertino, California."
    print("Extracting graph...")
    res = extract_graph_from_chunk(sample_text)
    print(res.model_dump_json(indent=2))

