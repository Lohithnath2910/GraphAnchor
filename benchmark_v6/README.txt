benchmark_v6 - vector (standard) RAG vs hybrid (vector + knowledge graph) RAG

WHAT IS NEW
  Same corpus as benchmark_v5 (the 76 documents already ingested; nothing to re-ingest). New question set.
  gold_qa.json    148 questions, built by build_gold.py (edit the lists there and re-run it)
  Run from the project root:  python benchmark_v6/build_gold.py

WHY A V6
  The v5 audit found problems that made the comparison blurry:
  1. Eight multi-hop questions ended at the same person (Kallenbach, Demir, Tavares, Osterhout ...), so one
     missing document sank many questions at once. In v6 no answer is used by more than 3 multi-hop questions.
  2. Those names recur in many documents, so the keyword check could be satisfied by accident ("retrieval 100%,
     answer wrong"). New v6 questions end at facts that live in a single document (a deputy's or department
     head's name, a batch code), and the runner now also reports a strict document-level retrieval score.
  3. Some wording was ambiguous ("company" / "group" / "plant" for Brannock Foundry Group / Ashgill Works).
  4. Some aggregation gold lists were incomplete (v5 missed Lumen Grid as a Lodestar customer).
  Every new question carries an `evidence` list: (document number, exact text) pairs establishing each hop.
  build_gold.py refuses to build if any evidence string is missing from its document.

QUESTION MIX
  real_single_hop 47   your real course documents (control group, unchanged)
  single_hop      12   one fact in one document (control group, unchanged)
  multi_hop_2     46   two documents
  multi_hop_3     23   three documents
  multi_hop_4     10   four or more documents
  aggregation     10   lists gathered from one or several documents, every list re-verified

HOW TO REPORT IT
  Headline = the multi-hop and aggregation questions (89 of 148). Single-hop and real-text questions are the control
  group: both systems should tie there, and hybrid must not be worse. The runner prints each category separately.

METRICS (eval/run_benchmark.py)
  keyword-retr   share of expected keywords found in the retrieved chunks (lenient: recurring names)
  doc-retr       share of the documents the question needs that appear among the retrieved chunks (strict)
  answer         share of expected keywords found in the generated answer (0 if a forbidden phrase appears)

COMMANDS
  1. ollama serve
  2. .venv\Scripts\python.exe -m uvicorn main:app --port 8000
  3. fast retrieval-only check (no LLM, a few minutes):
       .venv\Scripts\python.exe eval\run_benchmark.py --gold benchmark_v6\gold_qa.json --out eval\v6_retrieval.csv --retrieval-only
  4. full run with LLM answers (about 20 minutes):
       .venv\Scripts\python.exe eval\run_benchmark.py --gold benchmark_v6\gold_qa.json --out eval\v6_results.csv
     If it is interrupted, rerun the same command with --resume (it refuses to overwrite an existing results file).

HONEST CAVEATS
  - The corpus is synthetic and its questions were written to need several documents: it is a stress test of
    cross-document retrieval, not a sample of typical queries. Single-hop questions are included to show that
    hybrid does not hurt where the graph is not needed.
  - Hybrid sends about 11 chunks to the LLM and vector sends 3, so part of any gain can be extra context. The
    runner's earlier vector_matched mode was removed at the author's request.
  - 148 questions, one run: differences of a few questions are not statistically solid.
