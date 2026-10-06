import json
import logging
import os
import sqlite3
import time
from functools import lru_cache

import ollama

try:
    from .config import config
except ImportError:
    from src.config import config

logger = logging.getLogger("graphanchor")

_embed_client = ollama.Client(timeout=60)

# Short texts (questions, entity names) are also cached on disk, so a server restart or a repeated benchmark run does
# not pay the ~2 s Ollama call again. Chunks are long and are not cached. The cache is keyed by model and text.
_CACHE_PATH = os.path.join(os.path.dirname(config.db_path) or ".", "embed_cache.db")
_CACHE_MAX_CHARS = 600


def _disk_cache():
    conn = sqlite3.connect(_CACHE_PATH, timeout=10.0)
    conn.execute("CREATE TABLE IF NOT EXISTS emb (model TEXT, text TEXT, vec TEXT, PRIMARY KEY (model, text))")
    return conn


def _disk_get(text: str):
    try:
        with _disk_cache() as conn:
            row = conn.execute("SELECT vec FROM emb WHERE model = ? AND text = ?", (config.embed_model, text)).fetchone()
        return json.loads(row[0]) if row else None
    except Exception as e:
        logger.warning(f"Embedding disk cache read failed: {e}")
        return None


def _disk_put(text: str, vec: list[float]):
    try:
        with _disk_cache() as conn:
            conn.execute("INSERT OR REPLACE INTO emb VALUES (?, ?, ?)", (config.embed_model, text, json.dumps(vec)))
    except Exception as e:
        logger.warning(f"Embedding disk cache write failed: {e}")


@lru_cache(maxsize=4096)  # callers never mutate the returned list
def get_embedding(text: str) -> list[float]:
    # Retrieves vector embeddings using the configured Ollama model, with automatic retries for transient failures.
    cacheable = len(text) <= _CACHE_MAX_CHARS
    if cacheable:
        cached = _disk_get(text)
        if cached is not None:
            return cached
    last_err = None
    for attempt in range(config.ollama_max_retries + 1):
        try:
            response = _embed_client.embeddings(
                model=config.embed_model,
                prompt=text
            )
            vec = response['embedding']
            if cacheable:
                _disk_put(text, vec)
            return vec
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
