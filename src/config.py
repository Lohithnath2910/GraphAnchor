from pathlib import Path

import yaml
from pydantic import BaseModel, Field

# Loads configuration parameters for the RAG engine, falling back to defaults if config.yaml is missing.

class AppConfig(BaseModel):
    llm_model: str = Field(default="llama3.2", description="Ollama model for generation and graph extraction")
    embed_model: str
    db_path: str
    chroma_path: str
    chunk_size: int
    chunk_overlap: int
    similarity_threshold: float
    max_file_size_mb: float = 5.0
    ollama_max_retries: int = 2
    extraction_max_relations: int = 20  # cap on relations/entities extracted per chunk (was a hardcoded 12)
    hybrid_fusion_alpha: float = 0.55
    vector_anchor_confidence: float = 0.75
    graph_edge_base_confidence: float = 0.85
    graph_lexical_weight: float = 0.5  # share of graph-chunk selection score from question-word overlap (rest: embedding similarity)
    graph_same_doc_discount: float = 0.6  # multiplier per chunk already selected from the same document

def load_config(path: str = "config.yaml") -> AppConfig:
    if not Path(path).exists():
        return AppConfig(
            llm_model="llama3.2",
            embed_model="snowflake-arctic-embed2:568m",
            db_path="./data/graph.db",
            chroma_path="./data/chroma",
            chunk_size=300,
            chunk_overlap=50,
            similarity_threshold=0.7
        )
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return AppConfig(**data)

config = load_config()
