import logging
import time
from functools import lru_cache

import ollama

try:
    from .config import config
except ImportError:
    from src.config import config

logger = logging.getLogger("graphanchor")

_embed_client = ollama.Client(timeout=60)

@lru_cache(maxsize=4096)  # repeated questions and entity names skip the ~2 s Ollama call; callers never mutate the list
def get_embedding(text: str) -> list[float]:
    # Retrieves vector embeddings using the configured Ollama model, with automatic retries for transient failures.
    last_err = None
    for attempt in range(config.ollama_max_retries + 1):
        try:
            response = _embed_client.embeddings(
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
