import ollama
import json
from pydantic import BaseModel, Field
from typing import List, Optional
from src.config import config

class Relation(BaseModel):
    entity: str = Field(description="The source entity")
    relation: str = Field(description="The relationship between source and target")
    target_entity: str = Field(description="The target entity")

class GraphExtraction(BaseModel):
    entities: List[str] = Field(description="List of all unique entities extracted from the text")
    relations: List[Relation] = Field(description="List of relationships between entities")

def extract_graph_from_chunk(text: str) -> GraphExtraction:
    """Extract entities and relationships from text using Ollama JSON mode"""
    prompt = f"Extract all entities and relationships from the following text:\n\n{text}"
    
    response = ollama.chat(
        model=config.llm_model,
        messages=[
            {"role": "system", "content": "You are a graph extraction AI. Always respond in JSON matching the requested schema. Do not add any conversational text."},
            {"role": "user", "content": prompt}
        ],
        format=GraphExtraction.model_json_schema(),
        options={"num_ctx": 4096}
    )
    
    try:
        return GraphExtraction.model_validate_json(response['message']['content'])
    except Exception as e:
        print(f"Failed to parse LLM response: {response['message']['content']}")
        raise e

if __name__ == "__main__":
    sample_text = "Apple was founded by Steve Jobs and Steve Wozniak in Cupertino, California."
    print("Extracting graph...")
    res = extract_graph_from_chunk(sample_text)
    print(res.model_dump_json(indent=2))
