import tiktoken
from typing import List

try:
    from .config import config
except ImportError:
    from src.config import config

def chunk_text(text: str) -> List[str]:
    """Token-based sliding window splitter"""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    
    chunks = []
    chunk_size = config.chunk_size
    overlap = config.chunk_overlap
    
    if not tokens:
        return []
        
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        
        if end == len(tokens):
            break
            
        start += step
        
    return chunks

if __name__ == "__main__":
    sample_text = "This is a sample document. " * 50
    chunks = chunk_text(sample_text)
    print(f"Total chunks: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"Chunk {i} size: {len(tiktoken.get_encoding('cl100k_base').encode(c))} tokens")
