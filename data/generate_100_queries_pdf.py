import requests
import json
import time
from fpdf import FPDF
import os

API_URL = "http://localhost:8000"

# Pre-defined multi-hop and specific queries based on the core story
CORE_QUERIES = [
    "What is the connection between Aris Thorne and Project Chimera?",
    "How does the Phantom Protocol relate to Aetherium?",
    "What is the purpose of Facility 42?",
    "Who leads Lumina Corporation and what is their secret project?",
    "What links Echo Vanguard to the events in Facility 42?",
    "What is the true nature of Aetherium?",
    "Describe the timeline of the Phantom Protocol.",
    "Is there any evidence that Lumina Corporation is aware of Aetherium?",
    "How is Aris Thorne connected to Echo Vanguard?",
    "What are the security measures at Facility 42?"
]

# Generate remaining 90 queries to make it 100
# We can ask generic questions about the other 90 files
extra_queries = []
for i in range(11, 101):
    extra_queries.append(f"What are the main details discussed in document {i} regarding its specific topic?")

ALL_QUERIES = CORE_QUERIES + extra_queries

class PDFReport(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'GraphAnchor: RAG Evaluation Report (100 Queries)', 0, 1, 'C')
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
        # Standard RAG
        self.set_font('helvetica', 'B', 11)
        self.cell(0, 8, 'Standard Vector RAG:', 0, 1, 'L')
        self.set_font('helvetica', '', 10)
        self.multi_cell(0, 6, standard_ans)
        self.ln(4)
        
        # Hybrid Graph RAG
        self.set_font('helvetica', 'B', 11)
        self.set_text_color(0, 100, 0) # Dark green
        self.cell(0, 8, 'Hybrid Graph RAG:', 0, 1, 'L')
        self.set_text_color(0, 0, 0)
        self.set_font('helvetica', '', 10)
        self.multi_cell(0, 6, hybrid_ans)
        self.ln(10)

def run_evaluation():
    pdf = PDFReport()
    pdf.add_page()
    
    pdf.set_font('helvetica', '', 11)
    pdf.multi_cell(0, 6, "This document contains a side-by-side comparison of 100 different queries executed against both the Standard Vector RAG and the Hybrid Graph RAG models. It mimics the 'Compare Models' feature of the frontend.")
    pdf.ln(10)

    print(f"Starting execution of {len(ALL_QUERIES)} queries...")
    
    # We will limit to 100 queries
    for idx, q in enumerate(ALL_QUERIES[:100], 1):
        print(f"[{idx}/100] Query: {q}")
        
        # Standard RAG
        try:
            res_std = requests.get(f"{API_URL}/query", params={"q": q, "k": 3, "enable_graph": False}, timeout=30)
            std_ans = res_std.json().get("answer", "Error/Timeout")
        except Exception as e:
            std_ans = f"Failed to get response: {e}"
            
        # Hybrid Graph RAG
        try:
            res_hyb = requests.get(f"{API_URL}/query", params={"q": q, "k": 3, "enable_graph": True}, timeout=30)
            hyb_ans = res_hyb.json().get("answer", "Error/Timeout")
        except Exception as e:
            hyb_ans = f"Failed to get response: {e}"

        # Clean unicode for fpdf
        std_ans = std_ans.encode('latin-1', 'replace').decode('latin-1')
        hyb_ans = hyb_ans.encode('latin-1', 'replace').decode('latin-1')
        clean_q = q.encode('latin-1', 'replace').decode('latin-1')

        pdf.chapter_title(idx, clean_q)
        pdf.chapter_body(clean_q, std_ans, hyb_ans)
        
    output_path = os.path.abspath("data/GraphAnchor_100_Queries_Report.pdf")
    pdf.output(output_path)
    print(f"\nPDF Report successfully generated at: {output_path}")

if __name__ == "__main__":
    run_evaluation()
