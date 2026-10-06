# Benchmark

The corpus GraphAnchor was ingested with, and the questions used to compare vector RAG with hybrid graph RAG.
Results are in [`../results.md`](../results.md); every recorded run is in `../eval/results/`.

| File | Contents |
|---|---|
| `corpus/` | the 76 ingested documents: 30 real course documents (`real_*.txt`) and 46 synthetic "Calder Reach" documents (`01_*` to `46_*`), 705 chunks |
| `gold_qa.json` | the 148 questions with expected keywords and, for most, the documents they need (`source_docs`) |
| `build_gold.py` | builds `gold_qa.json`; refuses to build if any `evidence` string is missing from its document |
| `legacy_gold_v5.json` | the earlier 126-question set that `build_gold.py` keeps a selection of questions from |
| `ingest.py` | ingests a folder of `.txt` files into a running server |

## Question mix

| Category | n | Role |
|---|---|---|
| `real_single_hop` | 47 | control: one fact in one real document |
| `single_hop` | 12 | control: one fact in one synthetic document |
| `multi_hop_2` | 46 | two documents must be combined |
| `multi_hop_3` | 23 | three documents chained |
| `multi_hop_4` | 10 | four or more documents chained |
| `aggregation` | 10 | a list gathered from several documents (excluded from `../results.md`) |

## Design rules

- No answer is used by more than 3 multi-hop questions, so one hard document cannot sink many questions at once.
- New multi-hop questions end at facts that appear in a single document (a deputy's or department head's name, a batch
  code), not at names that recur across the corpus.
- Every new question has an `evidence` list: (document number, exact text) pairs establishing each hop.

## Commands

```powershell
ollama serve
.venv\Scripts\python.exe -m uvicorn main:app --port 8000
# only on an empty database (3 to 6 hours):
.venv\Scripts\python.exe benchmark\ingest.py
# benchmark (from the project root)
.venv\Scripts\python.exe eval\run_benchmark.py --out eval\results\my_run.csv
.venv\Scripts\python.exe eval\run_benchmark.py --out eval\results\my_retrieval.csv --retrieval-only
```

The corpus is synthetic and its multi-hop questions were written to need several documents, so this is a stress test of
cross-document retrieval, not a sample of typical queries.
