import requests
import json

API_URL = "http://localhost:8000"

def evaluate_query(query: str):
    print(f"\n==================================================")
    print(f"QUERY: {query}")
    print(f"==================================================")
    
    results = {}
    
    # 1. Plain Vector Search
    print("Testing Vector RAG...")
    res_vec = requests.get(f"{API_URL}/query", params={"q": query, "k": 5, "enable_graph": False})
    if res_vec.status_code == 200:
        data = res_vec.json()
        results["vector"] = data
    else:
        print(f"Vector failed: {res_vec.status_code}")
    
    # 2. Graph-Hybrid Search
    print("Testing Hybrid Graph RAG...")
    res_graph = requests.get(f"{API_URL}/query", params={"q": query, "k": 5, "enable_graph": True})
    if res_graph.status_code == 200:
        data = res_graph.json()
        results["hybrid"] = data
    else:
        print(f"Hybrid failed: {res_graph.status_code}")
        
    return results

if __name__ == "__main__":
    test_queries = [
        "Where are Xenotium Crystals mined?",
        "Who manages the island where Xenotium Crystals are mined?",
        "Does Victor Vance secretly fund the energy weapon initiative?",
        "Which island is associated with the Phantom Protocol?"
    ]
    
    final_report = "# Evaluation Report: Graph BFS Optimized\n\n"
    
    for q in test_queries:
        res = evaluate_query(q)
        
        vec_ans = res.get("vector", {}).get("answer", "ERROR")
        hyb_ans = res.get("hybrid", {}).get("answer", "ERROR")
        edges = res.get("hybrid", {}).get("graph_traversal", {}).get("edges", [])
        
        final_report += f"## Query: {q}\n"
        final_report += f"- **Standard Vector Result:** {vec_ans}\n"
        final_report += f"- **Hybrid Graph Result:** {hyb_ans}\n"
        final_report += f"- **Graph Edges Traversed:** {len(edges)}\n\n"
        
    # Write report
    with open(r"c:\Users\lohit\Desktop\Project I\data\evaluation_report_optimized.md", "w") as f:
        f.write(final_report)
        
    print("Evaluation complete. Report saved to data/evaluation_report_optimized.md")
