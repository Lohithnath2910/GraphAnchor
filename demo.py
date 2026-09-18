import requests
import os
import json

API_URL = "http://localhost:8000"

# Demonstrates the end-to-end ingestion and compares Vector-only vs Graph-augmented search.

def upload_file(filepath):
    print(f"\nUploading {filepath}...")
    if not os.path.exists(filepath):
        print(f"File {filepath} not found!")
        return
    with open(filepath, "rb") as f:
        res = requests.post(f"{API_URL}/ingest", files={"file": (os.path.basename(filepath), f, "text/plain")})
    if res.status_code == 200:
        print("Success:", res.json())
    else:
        print("Failed:", res.status_code, res.text)

def get_stats():
    res = requests.get(f"{API_URL}/graph/stats")
    if res.status_code == 200:
        print("Stats:", res.json())
    else:
        print("Failed:", res.status_code, res.text)

def query(q: str, enable_graph: bool):
    print(f"\n--- Querying: '{q}' (Graph Enabled: {enable_graph}) ---")
    res = requests.get(f"{API_URL}/query", params={"q": q, "k": 3, "enable_graph": enable_graph})
    if res.status_code == 200:
        data = res.json()
        print("\n[ANSWER]")
        print(data["answer"])
        
        print("\n[RETRIEVED CHUNKS]")
        for i, chunk in enumerate(data.get("ranking_breakdown", [])):
            source = chunk.get("source_type", "unknown")
            text = chunk.get("text", "").replace("\n", " ")[:100]
            print(f"  {i+1}. [{source.upper()}] {text}...")
    else:
        print("Query Failed:", res.status_code, res.text)

if __name__ == "__main__":
    docs = [
        "docs/lohith.txt",
        "docs/shayla.txt",
        "docs/sir.txt"
    ]

    print("--- Initial Stats ---")
    get_stats()

    for doc in docs:
        upload_file(doc)

    print("\n--- Final Stats ---")
    get_stats()

    print("\n===========================================")
    print("COMPARISON: VECTOR SEARCH VS GRAPH SEARCH")
    print("===========================================")
    
    test_query = "Who is Dr. Aris Thorne and what company did his former employee start?"
    
    print("\n>>> Running Plain Vector Search (No Graph) <<<")
    query(test_query, enable_graph=False)
    
    print("\n>>> Running Graph-Hybrid Search <<<")
    query(test_query, enable_graph=True)
