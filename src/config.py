import yaml
from pydantic import BaseModel
from pathlib import Path

class AppConfig(BaseModel):
    llm_model: str
    embed_model: str
    db_path: str
    chroma_path: str
    chunk_size: int
    chunk_overlap: int
    similarity_threshold: float

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
