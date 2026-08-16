import ollama
import json
import logging
import time
from pydantic import BaseModel, Field
from typing import List, Optional
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

if __name__ == "__main__":
    sample_text = "Apple was founded by Steve Jobs and Steve Wozniak in Cupertino, California."
    print("Extracting graph...")
    res = extract_graph_from_chunk(sample_text)
    print(res.model_dump_json(indent=2))
