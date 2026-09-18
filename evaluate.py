import os
import requests
import time

API_URL = "http://localhost:8000"
DOCS_DIR = "test_dataset_complex"

# Dedicated evaluation script to benchmark Vector vs Graph RAG performance on the generated dataset.

def upload_all_docs():
    print(f"--- Uploading all documents from {DOCS_DIR}/ ---")
    files = os.listdir(DOCS_DIR)
    for file in files:
        filepath = os.path.join(DOCS_DIR, file)
        if os.path.isfile(filepath):
            print(f"Uploading {filepath}...")
            with open(filepath, "rb") as f:
                res = requests.post(f"{API_URL}/ingest", files={"file": (file, f, "application/octet-stream")})
            if res.status_code == 200:
                print(f"  Success: {res.json()}")
            else:
                print(f"  Failed: {res.status_code} - {res.text}")

def evaluate_query(query: str):
    print(f"\n==================================================")
    print(f"QUERY: {query}")
    print(f"==================================================")
    
    # 1. Plain Vector Search
    print("\n[SCENARIO 1: Plain Vector Search (Graph Disabled)]")
    res_vec = requests.get(f"{API_URL}/query", params={"q": query, "k": 5, "enable_graph": False})
    if res_vec.status_code == 200:
        data = res_vec.json()
        print(f"Answer: {data['answer']}")
        print("Retrieved Chunks:")
        for i, chunk in enumerate(data.get("ranking_breakdown", [])):
            print(f"  {i+1}. [{chunk.get('source_type')}] {chunk.get('text')[:80]}...")
    else:
        print("Failed to run vector query.")
        
    # 2. Graph-Hybrid Search
    print("\n[SCENARIO 2: Graph-Hybrid Search (Graph Enabled)]")
    res_graph = requests.get(f"{API_URL}/query", params={"q": query, "k": 5, "enable_graph": True})
    if res_graph.status_code == 200:
        data = res_graph.json()
        print(f"Answer: {data['answer']}")
        print("Retrieved Chunks:")
        for i, chunk in enumerate(data.get("ranking_breakdown", [])):
            print(f"  {i+1}. [{chunk.get('source_type')}] {chunk.get('text')[:80]}...")
        edges = data.get("graph_traversal", {}).get("edges", [])
        print(f"Graph Edges Traversed: {len(edges)}")
        for e in edges[:5]:
            print(f"  - {e['source']} -> {e['relation']} -> {e['target']}")
    else:
        print("Failed to run graph query.")

if __name__ == "__main__":
    print("WARNING: Make sure the FastAPI server (main.py) is running on port 8000.")
    print("Resetting database before evaluation...")
    requests.delete(f"{API_URL}/reset?confirm=true")
    time.sleep(1)
    
    upload_all_docs()
    
    test_queries = [
        "Where is the secret project overseen by Alice Vance located?",
        "Who manages the facility where Project Chimera is being developed?"
    ]
    
    for q in test_queries:
        evaluate_query(q)
