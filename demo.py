import requests
import os

API_URL = "http://localhost:8000"

def upload_file(filepath):
    print(f"\nUploading {filepath}...")
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

def query(q: str):
    res = requests.get(f"{API_URL}/query", params={"q": q, "k": 2})
    if res.status_code == 200:
        print(f"\nQuery '{q}':", res.json())
    else:
        print("Query Failed:", res.status_code, res.text)

if __name__ == "__main__":
    docs = [
        "test/lohith.txt",
        "test/shayla.txt",
        "test/sir.txt"
    ]

    print("--- Initial Stats ---")
    get_stats()

    for doc in docs:
        upload_file(doc)

    print("\n--- Final Stats ---")
    get_stats()

    print("\n--- Stretch Goal: Vector Search ---")
    query("lohith")
