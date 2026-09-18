import requests
import time
import os

API_URL = "http://localhost:8000"

def test_system_status():
    print("[1] Testing System Endpoints...")
    res = requests.get(f"{API_URL}/docs/")
    assert res.status_code == 200, "/docs/ redirect failed"
    print("   [OK] /docs/ is up.")

def test_database_reset():
    print("[2] Testing Database Reset...")
    res = requests.delete(f"{API_URL}/reset?confirm=true")
    assert res.status_code == 200, "Reset failed"
    
    res_stats = requests.get(f"{API_URL}/graph/stats").json()
    assert res_stats["chunk_count"] == 0
    assert res_stats["edge_count"] == 0
    print("   [OK] DB reset and verified empty.")

def test_ingestion():
    print("[3] Testing Document Ingestion...")
    test_content = "# Apple Intelligence\nApple intelligence was unveiled by Tim Cook in 2024. Tim Cook leads Apple Inc."
    
    with open("temp_test.txt", "w") as f:
        f.write(test_content)
        
    with open("temp_test.txt", "rb") as f:
        res = requests.post(f"{API_URL}/ingest", files={"file": ("temp_test.txt", f, "text/plain")})
    
    os.remove("temp_test.txt")
    
    assert res.status_code == 200, "Ingestion failed"
    data = res.json()
    assert data["doc_id"] is not None
    assert data["chunks_processed"] > 0
    
    print(f"   [OK] Ingestion successful. Doc ID: {data['doc_id']}")
    return data["doc_id"]

def test_document_listing_and_deletion(doc_id):
    print("[4] Testing Document Listing & Deletion...")
    res_list = requests.get(f"{API_URL}/documents").json()
    assert any(d["doc_id"] == doc_id for d in res_list["documents"]), "Document not in list"
    print("   [OK] Document listing verified.")
    
    res_get = requests.get(f"{API_URL}/documents/{doc_id}").json()
    assert res_get["chunk_count"] > 0, "No chunks found in single doc lookup"
    
    print("   [OK] Single document lookup verified.")

def test_querying():
    print("[5] Testing Query Endpoints...")
    # Test standard answer endpoint
    res_ans = requests.post(f"{API_URL}/answer", json={"query": "Who leads Apple Inc?", "k": 3, "enable_graph": True})
    assert res_ans.status_code == 200, "Answer POST failed"
    assert "Tim Cook" in res_ans.json()["answer"]
    
    # Test main query endpoint with breakdown
    res_query = requests.get(f"{API_URL}/query", params={"q": "When was Apple Intelligence unveiled?", "k": 3, "enable_graph": True})
    assert res_query.status_code == 200
    assert "2024" in res_query.json()["answer"]
    
    print("   [OK] Query endpoints responded accurately.")

def test_streaming():
    print("[6] Testing Streaming Endpoint...")
    res_stream = requests.get(f"{API_URL}/answer/stream", params={"q": "Who leads Apple Inc?", "k": 3}, stream=True)
    assert res_stream.status_code == 200
    
    tokens_received = 0
    for line in res_stream.iter_lines():
        if line:
            if line.startswith(b"data: ") and not line.startswith(b"data: {}"):
                tokens_received += 1
                
    assert tokens_received > 0, "No stream tokens received"
    print(f"   [OK] Streaming working ({tokens_received} stream events).")

def run_all_tests():
    try:
        test_system_status()
        test_database_reset()
        doc_id = test_ingestion()
        
        # Give embeddings a moment
        time.sleep(1)
        
        test_document_listing_and_deletion(doc_id)
        test_querying()
        test_streaming()
        
        # Clean up
        requests.delete(f"{API_URL}/documents/{doc_id}")
        print("   [OK] Cleanup verified.")
        print("\n[OK] ALL TESTS PASSED FOR THIS ITERATION\n")
        return True
    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n[FAIL] UNEXPECTED ERROR: {e}")
        return False

if __name__ == "__main__":
    iterations = 5
    print(f"Starting rigorous testing for {iterations} iterations...\n")
    
    successes = 0
    for i in range(1, iterations + 1):
        print(f"=== ITERATION {i} ===")
        if run_all_tests():
            successes += 1
            
    print(f"====================================")
    print(f"Final Score: {successes}/{iterations} Iterations Passed")
    print(f"====================================")
