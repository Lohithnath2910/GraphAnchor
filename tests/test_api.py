import os
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time

from main import app

client = TestClient(app)

# Mock Ollama Responses
def mock_embeddings(*args, **kwargs):
    return {"embedding": [0.1] * 568}

def mock_chat(*args, **kwargs):
    messages = kwargs.get("messages", [])
    system_prompt = messages[0]["content"] if messages else ""
    
    # Check if this is a graph extraction call (JSON schema request)
    if "Extract all factual" in kwargs.get("messages", [{}])[-1].get("content", "") or "Strict Extraction Rules" in system_prompt:
        return {
            "message": {
                "content": '{"entities": ["Alice", "Acme Corp", "San Francisco"], "relations": [{"entity": "Alice", "relation": "is CEO of", "target_entity": "Acme Corp"}, {"entity": "Acme Corp", "relation": "located in", "target_entity": "San Francisco"}]}'
            }
        }
    
    # Otherwise it is a generation call
    return {
        "message": {
            "content": "Alice is the CEO of Acme Corp, which is located in San Francisco."
        }
    }


@pytest.fixture(autouse=True)
def reset_db_before_and_after():
    # Clear databases before test
    client.delete("/reset?confirm=true")
    yield
    # Clear databases after test
    client.delete("/reset?confirm=true")

@patch('ollama.embeddings', side_effect=mock_embeddings)
@patch('ollama.chat', side_effect=mock_chat)
def test_easy_stats(mock_chat, mock_embed):
    # Easy: Test stats endpoint on empty database
    response = client.get("/graph/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["chunk_count"] == 0
    assert data["edge_count"] == 0
    assert data["total_entities_known"] == 0

@patch('ollama.embeddings', side_effect=mock_embeddings)
@patch('ollama.chat', side_effect=mock_chat)
def test_medium_upload_and_query(mock_chat, mock_embed, tmp_path):
    # Medium: Upload a single document, verify chunks, simple query
    # Create a temporary text file
    test_file = tmp_path / "test_doc.txt"
    content = "Alice is the CEO of Acme Corp. Acme Corp is located in San Francisco."
    test_file.write_text(content)
    
    with open(test_file, "rb") as f:
        response = client.post("/ingest", files={"file": ("test_doc.txt", f, "text/plain")})
        
    assert response.status_code == 200
    data = response.json()
    assert data["chunks_processed"] > 0
    assert data["entities_extracted"] >= 0
    
    # Query vector search (graph traversal disabled)
    query_res = client.get("/query?q=Who is the CEO?&k=2&enable_graph=false")
    assert query_res.status_code == 200
    q_data = query_res.json()
    assert "answer" in q_data
    assert len(q_data["vector_search_results"]) > 0

@patch('ollama.embeddings', side_effect=mock_embeddings)
@patch('ollama.chat', side_effect=mock_chat)
def test_hard_complex_graph_query(mock_chat, mock_embed, tmp_path):
    # Hard: Upload complex file, query with graph traversal enabled and check wait/LLM generation
    test_file = tmp_path / "complex_doc.txt"
    # Provide a multi-hop story
    content = """Dr. Aris Thorne discovered Inhibitor-Z. Inhibitor-Z is a rare chemical. 
The rare chemical cures the Blue Plague. The Blue Plague devastated the Munich Foundry. 
The Munich Foundry is located in Germany."""
    test_file.write_text(content)
    
    with open(test_file, "rb") as f:
        response = client.post("/ingest", files={"file": ("complex_doc.txt", f, "text/plain")})
    assert response.status_code == 200
    
    # Wait briefly to ensure any async db flushing is complete (though SQLite is sync here)
    time.sleep(1)
    
    # Check graph traversal query
    query_res = client.get("/query?q=What cures the disease that devastated the Munich Foundry?&k=3&enable_graph=true")
    assert query_res.status_code == 200
    
    q_data = query_res.json()
    assert "answer" in q_data
    assert "graph_traversal" in q_data
    
    assert isinstance(q_data["answer"], str)
    assert len(q_data["answer"]) > 5
    
    # Check /answer POST endpoint
    post_res = client.post("/answer", json={"query": "Who discovered Inhibitor-Z?", "k": 2, "enable_graph": True})
    assert post_res.status_code == 200
    p_data = post_res.json()
    assert "answer" in p_data
    assert len(p_data["answer"]) > 5

@patch('ollama.embeddings', side_effect=mock_embeddings)
@patch('ollama.chat', side_effect=mock_chat)
def test_document_deletion(mock_chat, mock_embed, tmp_path):
    test_file = tmp_path / "del_doc.txt"
    test_file.write_text("Alice works at Acme.")
    
    with open(test_file, "rb") as f:
        res = client.post("/ingest", files={"file": ("del_doc.txt", f, "text/plain")})
    
    doc_id = res.json()["doc_id"]
    
    # Verify it exists
    assert client.get(f"/documents/{doc_id}").status_code == 200
    
    # Delete it
    del_res = client.delete(f"/documents/{doc_id}")
    assert del_res.status_code == 200
    
    # Verify it's gone
    assert client.get(f"/documents/{doc_id}").status_code == 404
    
    # Verify stats reflect deletion
    stats = client.get("/graph/stats").json()
    assert stats["chunk_count"] == 0
    assert stats["edge_count"] == 0

def test_unsupported_file(tmp_path):
    test_file = tmp_path / "bad.csv"
    test_file.write_text("name,age\nalice,30")
    
    with open(test_file, "rb") as f:
        res = client.post("/ingest", files={"file": ("bad.csv", f, "text/csv")})
    assert res.status_code == 415

def test_empty_file(tmp_path):
    test_file = tmp_path / "empty.txt"
    test_file.write_text("   ")
    
    with open(test_file, "rb") as f:
        res = client.post("/ingest", files={"file": ("empty.txt", f, "text/plain")})
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()

@patch('ollama.embeddings', side_effect=mock_embeddings)
def test_extraction_failure_resilience(mock_embed, tmp_path):
    # If extraction fails, chunk should still be ingested but extraction_failures should be > 0
    def mock_chat_fail(*args, **kwargs):
        raise RuntimeError("Ollama crashed")
        
    with patch('ollama.chat', side_effect=mock_chat_fail):
        test_file = tmp_path / "fail_doc.txt"
        test_file.write_text("This is some text.")
        
        with open(test_file, "rb") as f:
            res = client.post("/ingest", files={"file": ("fail_doc.txt", f, "text/plain")})
            
        assert res.status_code == 200
        data = res.json()
        assert data["chunks_processed"] > 0
        assert data["extraction_failures"] > 0
        assert data["edges_added"] == 0
