import ollama
import logging
import time
from src.config import config
from typing import List

logger = logging.getLogger("graphanchor")

def get_embedding(text: str) -> List[float]:
    """Get embeddings using Ollama. Retries on transient failures so a single
    dropped connection doesn't fail an entire ingest or query."""
    last_err = None
    for attempt in range(config.ollama_max_retries + 1):
        try:
            response = ollama.embeddings(
                model=config.embed_model,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            last_err = e
            logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
            if attempt < config.ollama_max_retries:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Embedding failed after {config.ollama_max_retries + 1} attempt(s): {last_err}") from last_err

if __name__ == "__main__":
    sample_text = "This is a test chunk for embedding."
    emb = get_embedding(sample_text)
    print(f"Embedding shape: {len(emb)}-dim")
