import os
import requests
import time
from fpdf import FPDF

API_URL = "http://localhost:8000"
DOCS_DIR = r"c:\Users\lohit\Desktop\Project I\data\massive_dataset"

CORE_QUERIES = [
    "What is the connection between Aris Thorne and Project Chimera?",
    "How does the Phantom Protocol relate to Aetherium?",
    "What is the purpose of Facility 42?",
    "Who leads Lumina Corporation and what is their secret project?",
    "What links Echo Vanguard to the events in Facility 42?",
    "What is the true nature of Aetherium?",
]

def upload_docs_and_query():
    print("Resetting database before massive ingestion...")
    requests.delete(f"{API_URL}/reset?confirm=true")
    time.sleep(2)
    
    print(f"--- Uploading 30 documents from {DOCS_DIR}/ ---")
    files = sorted(os.listdir(DOCS_DIR))
    
    # Take the 6 core documents (which have names like doc_001_..., doc_003_..., etc)
    # Actually, all files in the directory are named doc_001 to doc_100.
    # Let's just pick the first 30 documents.
    files_to_upload = files[:30]
    total = len(files_to_upload)
    
    for i, file in enumerate(files_to_upload):
        filepath = os.path.join(DOCS_DIR, file)
        if os.path.isfile(filepath):
            with open(filepath, "rb") as f:
                res = requests.post(f"{API_URL}/ingest", files={"file": (file, f, "text/plain")})
            if res.status_code == 200:
                data = res.json()
                print(f"[{i+1}/{total}] Uploaded {file} (Chunks: {data.get('chunks_processed')})")
            else:
                print(f"[{i+1}/{total}] Failed {file}: {res.status_code}")
                
            # Wait to ensure database and LLM are ready and not overloaded
            time.sleep(1)

    print("\n--- Waiting for Background Graph Extraction to Complete ---")
    prev_edges = -1
    stable_count = 0
    while True:
        try:
            stats = requests.get(f"{API_URL}/graph/stats").json()
            curr_edges = stats.get("edge_count", 0)
            print(f"Current Stats -> Chunks: {stats.get('chunk_count')}, Edges: {curr_edges}")
            
            if curr_edges > 0 and curr_edges == prev_edges:
                stable_count += 1
            else:
                stable_count = 0
                
            if stable_count >= 3:
                print("Graph extraction appears to have completed and stabilized.")
                break
                
            prev_edges = curr_edges
        except Exception as e:
            print(f"Waiting for API... {e}")
            
        time.sleep(5)
        
    print(f"Final Graph DB Stats: {stats}")
    
    # 24 extra generic queries for the other docs
    extra_queries = []
    for i in range(7, 31):
        extra_queries.append(f"What are the main details discussed in document {i} regarding its specific topic?")
        
    ALL_QUERIES = CORE_QUERIES + extra_queries

    print(f"\n--- Starting execution of {len(ALL_QUERIES)} queries ---")
    
    from fpdf import FPDF
    class PDFReport(FPDF):
        def header(self):
            self.set_font('helvetica', 'B', 15)
            self.cell(0, 10, 'GraphAnchor: RAG Evaluation Report (30 Queries)', 0, 1, 'C')
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('helvetica', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

        def chapter_title(self, num, title):
            self.set_font('helvetica', 'B', 12)
            self.set_fill_color(200, 220, 255)
            self.cell(0, 10, f'Query {num}: {title}', 0, 1, 'L', 1)
            self.ln(4)

        def chapter_body(self, title, standard_ans, hybrid_ans):
            self.set_font('helvetica', 'B', 11)
            self.cell(0, 8, 'Standard Vector RAG:', 0, 1, 'L')
            self.set_font('helvetica', '', 10)
            self.multi_cell(0, 6, standard_ans)
            self.ln(4)
            
            self.set_font('helvetica', 'B', 11)
            self.set_text_color(0, 100, 0)
            self.cell(0, 8, 'Hybrid Graph RAG:', 0, 1, 'L')
            self.set_text_color(0, 0, 0)
            self.set_font('helvetica', '', 10)
            self.multi_cell(0, 6, hybrid_ans)
            self.ln(10)

    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font('helvetica', '', 11)
    pdf.multi_cell(0, 6, "This document contains a side-by-side comparison of 30 different queries executed against both the Standard Vector RAG and the Hybrid Graph RAG models. It explicitly ensures successful document ingestion.")
    pdf.ln(10)

    for idx, q in enumerate(ALL_QUERIES, 1):
        print(f"[{idx}/30] Query: {q}")
        
        try:
            res_std = requests.get(f"{API_URL}/query", params={"q": q, "k": 3, "enable_graph": False}, timeout=60)
            std_ans = res_std.json().get("answer", "Error/Timeout")
        except Exception as e:
            std_ans = f"Failed to get response: {e}"
            
        try:
            res_hyb = requests.get(f"{API_URL}/query", params={"q": q, "k": 3, "enable_graph": True}, timeout=60)
            hyb_ans = res_hyb.json().get("answer", "Error/Timeout")
        except Exception as e:
            hyb_ans = f"Failed to get response: {e}"

        std_ans = std_ans.encode('latin-1', 'replace').decode('latin-1')
        hyb_ans = hyb_ans.encode('latin-1', 'replace').decode('latin-1')
        clean_q = q.encode('latin-1', 'replace').decode('latin-1')

        pdf.chapter_title(idx, clean_q)
        pdf.chapter_body(clean_q, std_ans, hyb_ans)
        
    output_path = os.path.abspath(r"c:\Users\lohit\Desktop\Project I\data\GraphAnchor_30_Queries_Report.pdf")
    pdf.output(output_path)
    print(f"\nPDF Report successfully generated at: {output_path}")

if __name__ == "__main__":
    upload_docs_and_query()
