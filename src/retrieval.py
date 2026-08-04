import ollama
from src.config import config
from typing import List

def get_embedding(text: str) -> List[float]:
    """Get embeddings using Ollama"""
    response = ollama.embeddings(
        model=config.embed_model,
        prompt=text
    )
    return response['embedding']

if __name__ == "__main__":
    sample_text = "This is a test chunk for embedding."
    emb = get_embedding(sample_text)
    print(f"Embedding shape: {len(emb)}-dim")
