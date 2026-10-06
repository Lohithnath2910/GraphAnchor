# GraphAnchor: Final Summary (V6)

Standard vector RAG versus GraphAnchor's hybrid RAG (vector search plus knowledge-graph traversal), measured on the same
ingested corpus with the same local LLM for both modes. Raw per-question results are in `eval/v6_results.csv`.

## Setup

| | |
|---|---|
| Corpus | 76 documents, 705 chunks, 2,555 graph edges, 2,308 entities (30 real course documents plus 46 synthetic "Calder Reach" documents) |
| Questions | 148 (`benchmark_v6/gold_qa.json`), built by `benchmark_v6/build_gold.py` |
| Models | `llama3.2` (3B) for answers and graph extraction, `snowflake-arctic-embed2:568m` for embeddings, both through Ollama |
| Vector mode | top 3 chunks (k=3) |
| Hybrid mode | top 3 chunks plus up to 8 chunks found by walking the graph, about 11 chunks in total |
| Errors / stalls | none (148 questions x 2 modes) |

### Question mix

| Category | n | Role |
|---|---|---|
| `real_single_hop` | 47 | control: one fact in one real course document |
| `single_hop` | 12 | control: one fact in one synthetic document |
| `multi_hop_2` | 46 | two documents must be combined |
| `multi_hop_3` | 23 | three documents chained |
| `multi_hop_4` | 10 | four or more documents chained |
| `aggregation` | 10 | a list gathered from one or several documents |

The **headline subset** is the multi-hop and aggregation questions (89 of 148). The single-hop questions are the control
group: hybrid should tie there and must not be worse.

### Metrics

- **doc-retr** (strict): share of the documents a question needs (`source_docs`) that appear among the retrieved chunks.
- **keyword-retr** (lenient): share of the expected keywords found in the retrieved chunks. Names that recur across
  documents can satisfy it by accident.
- **answer**: share of the expected keywords found in the generated answer (0 if a forbidden phrase appears).

## Results

### Headline: multi-hop and aggregation (89 questions)

| | Vector | Hybrid | Difference |
|---|---|---|---|
| Document retrieval | 55.6% | 87.8% | **+32.2 pts** |
| Answer correctness | 40.2% | 51.6% | **+11.4 pts** |

### All 148 questions

| | Vector (k=3) | Hybrid | Difference |
|---|---|---|---|
| Document retrieval | 70.6% | 90.6% | +20.0 pts |
| Keyword retrieval | 69.0% | 80.4% | +11.4 pts |
| Answer correctness | 58.5% | 65.9% | +7.4 pts |
| Fully-correct answers | 80 / 148 | 92 / 148 | +12 |
| Latency, mean / median / p95 | 3.0 / 2.9 / 3.8 s | 3.8 / 3.7 / 4.9 s | +0.8 s |

Per question (answer correctness): hybrid better on **20**, same on **120**, worse on **8**.

### By category

| Category | n | Doc retrieval: vector | hybrid | Answer: vector | hybrid | Fully correct: vector | hybrid |
|---|---|---|---|---|---|---|---|
| `real_single_hop` | 47 | 91.5% | 93.6% | 82.6% | 84.0% | 34 | 36 |
| `single_hop` | 12 | 100% | 100% | 100% | 100% | 12 | 12 |
| `multi_hop_2` | 46 | 58.7% | 94.6% | 39.1% | 50.0% | 18 | 23 |
| `multi_hop_3` | 23 | 49.3% | 78.3% | 43.5% | 60.9% | 10 | 14 |
| `multi_hop_4` | 10 | 26.8% | 76.5% | 10.0% | 30.0% | 1 | 3 |
| `aggregation` | 10 | 85.0% | 90.0% | 67.7% | 59.6% | 5 | 4 |

## What the results show

- **Retrieval is where the graph wins, and by a wide margin.** On two-hop questions hybrid finds the needed documents
  94.6% of the time against 58.7% for vector. At four hops it is 76.5% against 26.8%.
- **Answer gains are real but smaller than retrieval gains.** A 3B model has to reason over about 11 chunks, so part of
  the retrieval advantage is lost in generation.
- **On the control questions hybrid ties or is slightly ahead**, so the graph does not hurt where it is not needed overall.
- **Aggregation is not a win.** Hybrid is worse on answers there (59.6% against 67.7%), on only 10 questions.

## Where hybrid still loses (8 questions)

| Q | Category | What happened |
|---|---|---|
| 2 | `real_single_hop` | vector found the answer chunk; hybrid added unrelated chunks and answered "7 stages" instead of 8 |
| 43 | `real_single_hop` | extra chunks from another Spanish document; answered "nos enfermamos" instead of "tenemos" |
| 45 | `real_single_hop` | dropped one of three listed uses |
| 222 | `multi_hop_2` | vector had the answer; hybrid named the project director instead of the head of marine operations |
| 223 | `multi_hop_2` | same pattern: named the project director instead of the network chief |
| 246 | `aggregation` | listed fewer of the five projects |
| 250 | `aggregation` | listed none of the four directors |
| 253 | `aggregation` | missed one of four timing customers |

In every case that was traced, the vector top 3 already contained the answer and hybrid then appended up to 8 weakly
related graph chunks that confused the model. The cause is noise injection, not a retrieval miss.

## How the pipeline got here

The v5 benchmark (126 questions, original pipeline) gave vector 59.2% and hybrid 67.5% answer correctness. The audit of
that run found two things: graph chunks reached the LLM with no source filename, and hybrid sent mostly noise (about 1 of
11 chunks relevant on some questions). The traversal pool already held 93.6% of the needed documents, but the final
selection kept only 79.4% of them, so selection was the bottleneck.

Changes (all query-time; nothing was re-ingested):

1. Graph chunks are ranked by embedding similarity blended with IDF-weighted question-word overlap, with a discount for
   documents already selected.
2. Graph chunks carry their source filename.
3. The triples sent to the LLM come from the passages the LLM sees.
4. Query embeddings are cached.
5. `/query` accepts `answer=false` for retrieval-only runs, and `/answer/stream` no longer generates the answer twice.

Selection-weight sweep on v5 (retrieval only, hybrid doc-retr; vector scored 73.1%):

| lexical weight | same-document discount | hybrid doc-retr |
|---|---|---|
| (original selection) | | 88.9% |
| 0.5 | 1.0 | 89.7% |
| 0.7 | 0.8 | 89.9% |
| 0.3 | 0.8 | 91.4% |
| 0.5 | 0.8 | 91.9% |
| **0.5** | **0.6 (chosen)** | **93.0%** |

The weights were tuned on v5. v6 shares only 36 questions with v5, and hybrid document retrieval there was 90.6%, so the
result is not simply overfit to the tuning set.

## Why v6 exists

The v5 audit also found problems with the benchmark itself:

- Eight multi-hop questions ended at the same person, so one hard-to-find document sank many questions at once. In v6 no
  answer is used by more than three multi-hop questions.
- Recurring names let the keyword check pass by accident ("retrieval 100%, answer wrong"). New v6 questions end at facts
  that appear in a single document, and the runner now reports the strict document-level score.
- Some question wording was ambiguous and one aggregation list was incomplete (Lumen Grid was missing as a Lodestar
  customer).
- Every new question carries an `evidence` list of (document, exact text) pairs, and `build_gold.py` refuses to build if
  any of them is missing from its document.

## Caveats

- Hybrid sends about 11 chunks and vector sends 3, so part of the gain can come from extra context. An equal-chunk
  vector baseline was removed at the author's request and is not measured here.
- The corpus is synthetic and its multi-hop questions were written to need several documents. It is a stress test of
  cross-document retrieval, not a sample of typical queries.
- Single run on 148 questions: differences of a few questions are not statistically solid.
- Answer scoring is a keyword match, not a semantic judgement.

## Reproduce

```powershell
ollama serve
.venv\Scripts\python.exe -m uvicorn main:app --port 8000
# retrieval only, no LLM (a few minutes)
.venv\Scripts\python.exe eval\run_benchmark.py --gold benchmark_v6\gold_qa.json --out eval\v6_retrieval.csv --retrieval-only
# full run with LLM answers (about 20 minutes); add --resume to continue an interrupted run
.venv\Scripts\python.exe eval\run_benchmark.py --gold benchmark_v6\gold_qa.json --out eval\v6_results.csv
```

Files: `eval/v6_results.csv` (full run), `eval/v6_retrieval.csv` (retrieval only), `eval/v5_*.csv` (v5 baselines),
`benchmark_v6/README.txt` (benchmark design).
