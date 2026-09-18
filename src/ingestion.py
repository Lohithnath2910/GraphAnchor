import io
import tiktoken
from typing import List

try:
    from .config import config
except ImportError:
    from src.config import config

def extract_text_from_file(filename: str, content: bytes) -> str:
    # Extracts clean plain text from raw file bytes based on the file extension (.txt, .md, .pdf).
    ext = filename.lower().split('.')[-1] if '.' in filename else ''
    
    if ext in {"txt", "md", "markdown"}:
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return content.decode("latin-1")
            except Exception as err:
                raise ValueError("Could not decode text file as UTF-8 or Latin-1.") from err

    elif ext == "pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            extracted_pages = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    extracted_pages.append(text.strip())
            if not extracted_pages:
                raise ValueError("PDF contains no extractable text (it might be scanned/image-only).")
            return "\n\n".join(extracted_pages)
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Failed to parse PDF document: {e}") from e

    else:
        raise ValueError(f"Unsupported file format: '.{ext}'. Supported formats: .txt, .md, .pdf")

def chunk_text(text: str) -> List[str]:
    # Splits text into overlapping token-based chunks using the cl100k_base tokenizer.
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
