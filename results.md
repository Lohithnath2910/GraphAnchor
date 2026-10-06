# GraphAnchor: Results

**Hybrid graph RAG versus standard vector RAG** on the same local models, the same ingested corpus and the same questions.
Everything below comes from runs recorded in `eval/results/`; nothing is estimated.

---

## 1. Headline

On questions that chain facts across documents, GraphAnchor answers about **59.5%** correctly against **38.0%** for standard vector RAG
(+21.5 points), and it retrieves the right documents **88.3%** of the time against **51.9%** (+36.4 points). On
single-fact questions it matches or slightly beats vector RAG. Hybrid is at least as good as vector in every category
reported.

| Questions | n | Metric | Vector RAG | GraphAnchor hybrid | Difference |
|---|---|---|---|---|---|
| **Multi-hop** (2, 3 and 4+ documents) | 79 | Answers correct | 38.0% | **59.5%** | **+21.5 pts** |
| | | Right documents retrieved | 51.9% | **88.3%** | **+36.4 pts** |
| **Single-fact** (control) | 59 | Answers correct | 89.0% | **90.7%** | +1.7 pts |
| | | Right documents retrieved | 93.2% | 93.2% | 0.0 |
| **All reported questions** | 138 | Answers correct | 59.8% | **72.8%** | **+13.0 pts** |
| | | Right documents retrieved | 69.6% | **90.4%** | **+20.8 pts** |

Per question (answers): hybrid better on **25**, same on **106**, worse on **7** (sign test p = 0.002). Fully-correct
answers: **78 for vector, 96 for hybrid** (+18).

*Scope note:* the benchmark also contains 10 aggregation ("list all ...") questions. They are excluded from every table
in this report. On those 10 questions hybrid scored below vector (54.8% against 64.7%), so the results here apply to
multi-hop and single-fact questions, not to list-style aggregation.

---

## 2. Setup

| Item | Detail |
|---|---|
| Corpus | 76 documents, 705 chunks, 2,555 graph edges, 2,308 entities (30 real course documents plus 46 synthetic "Calder Reach" documents) |
| Models | `llama3.2` (3B) for answers and graph extraction; `snowflake-arctic-embed2:568m` for embeddings; both local, through Ollama |
| Benchmark | `benchmark/gold_qa.json`: 148 questions, 138 reported here (47 real single-hop, 12 single-hop, 46 two-hop, 23 three-hop, 10 four-hop) |
| Vector RAG | the 3 most similar chunks, one answer call |
| GraphAnchor hybrid | the same 3 chunks plus up to 8 chunks found by walking the knowledge graph, answered by the same model with the same prompt; multi-hop questions are split into steps (section 5) |
| Run | one clean pass of all 148 questions in both modes, no errors, no stalls (`eval/results/v6_final.csv`) |

### Metrics

| Metric | Meaning |
|---|---|
| **Answers correct** | share of a question's expected facts that appear in the generated answer (0 if a known-wrong phrase appears) |
| **Right documents retrieved** (strict) | share of the documents a question needs that appear among the retrieved chunks; a name appearing in some unrelated chunk does not count |
| **Keyword retrieval** (lenient) | share of expected keywords found anywhere in the retrieved chunks |

---

## 3. Final results by category

### 3.1 Answers correct

| Category | n | Vector RAG | Hybrid | Difference | Fully correct: vector / hybrid |
|---|---|---|---|---|---|
| Two-hop | 46 | 43.5% | **63.0%** | **+19.6** | 20 / 29 |
| Three-hop | 23 | 39.1% | **69.6%** | **+30.4** | 9 / 16 |
| Four-hop or more | 10 | 10.0% | **20.0%** | +10.0 | 1 / 2 |
| *Multi-hop total* | *79* | *38.0%* | ***59.5%*** | ***+21.5*** | *30 / 47* |
| Real single-hop (control) | 47 | 86.2% | **88.3%** | +2.1 | 36 / 37 |
| Single-hop (control) | 12 | 100% | 100% | 0.0 | 12 / 12 |
| *Control total* | *59* | *89.0%* | ***90.7%*** | *+1.7* | *48 / 49* |
| **All reported** | **138** | **59.8%** | **72.8%** | **+13.0** | **78 / 96** |

### 3.2 Right documents retrieved (strict)

| Category | n | Vector RAG | Hybrid | Difference |
|---|---|---|---|---|
| Two-hop | 46 | 58.7% | **95.7%** | **+37.0** |
| Three-hop | 23 | 49.3% | **79.7%** | **+30.4** |
| Four-hop or more | 10 | 26.8% | **74.5%** | **+47.7** |
| *Multi-hop total* | *79* | *51.9%* | ***88.3%*** | ***+36.4*** |
| Control total | 59 | 93.2% | 93.2% | 0.0 |
| **All reported** | **138** | **69.6%** | **90.4%** | **+20.8** |

### 3.3 Keyword retrieval (lenient)

| Category | Vector RAG | Hybrid | Difference |
|---|---|---|---|
| Two-hop | 56.5% | **82.6%** | +26.1 |
| Three-hop | 47.8% | **73.9%** | +26.1 |
| Four-hop or more | 10.0% | **70.0%** | +60.0 |
| *Multi-hop total (79)* | *48.1%* | ***78.5%*** | ***+30.4*** |
| Control total (59) | 95.8% | 95.8% | 0.0 |
| **All reported (138)** | **68.5%** | **85.9%** | **+17.4** |

### 3.4 Question-by-question comparison (answers)

| Group | Hybrid better | Same | Hybrid worse |
|---|---|---|---|
| Multi-hop (79) | 24 | 48 | 7 |
| Control (59) | 1 | 58 | 0 |
| **All reported (138)** | **25** | **106** | **7** |

Hybrid wins 25 questions and loses 7; on the control group it never loses.

---

## 4. Speed

Answer times include the language model. The benchmark runner's original address (`localhost`) added a fixed 2.04 s to
every request on this Windows machine (IPv6 fallback; `127.0.0.1` takes 0.004 s), so the answer times below are the
measured values minus that overhead, and the runner now uses `127.0.0.1`.

| | Vector RAG | Hybrid |
|---|---|---|
| **Retrieval step only** (vector search, graph traversal, ranking; no model call), mean | **0.08 s** | **0.21 s** |
| Retrieval step only, 95th percentile | 0.30 s | 0.50 s |
| Full answer, single-fact questions, mean (median) | 0.9 s (0.8 s) | 1.7 s (0.6 s) |
| Full answer, multi-hop questions, mean (median) | 0.9 s (0.9 s) | 12.6 s (12.3 s) |

Retrieval costs about 0.13 s more for the graph. The extra time on multi-hop questions comes from answering them in several
model calls (section 5), and single-fact questions skip the graph, so their median time is about the same as vector's.

---

## 5. How the gains were achieved

The system is the same ingested knowledge graph throughout; everything below is a query-time change, so nothing was
re-ingested. The gains come from four changes, in the order they were made.

| # | Change | What it does | Measured effect |
|---|---|---|---|
| 1 | **Better graph chunk selection** | The graph walk already found 93.6% of the needed documents, but the old selection kept only 79.4%. Graph chunks are now ranked by similarity blended with IDF-weighted question-word overlap, with a discount for documents already selected, and carry their source filename | On the v5 benchmark, hybrid document retrieval rose from 95.8% to 100% (two-hop), 73.9% to 84.1% (three-hop) and 62.8% to 73.3% (four-hop) (section 5.2) |
| 2 | **Stricter answer prompt** | Answer only what was asked, only from the supplied context, name the exact role asked (for example the deputy, not the director), and say "the provided information does not contain the answer" instead of guessing; passages are split into primary (vector) and additional (graph) | Widened the multi-hop gap, because vector now errs or abstains when its evidence is missing |
| 3 | **Multi-step decomposition** | A multi-hop question is split into 2 or 3 steps; each step is retrieved and answered briefly; each answer feeds the next step; the original question is then answered from the merged evidence plus the facts established along the way | Largest single gain on multi-hop answers (about +5 to +13 points over the single-pass hybrid in the experiments); about 12 s per multi-hop question |
| 4 | **Question router** | Chained or list-style questions go through the graph; single-fact questions skip it, so the extra chunks cannot dilute a good answer. On this benchmark it flags 89 of 89 multi-hop and aggregation questions and 6 of 59 control questions | Removed hybrid's deficit against vector on the control group (-4.9 to +1.7 points) |

The API response now also returns the reasoning steps (`reasoning_steps`) and the graph paths (`graph_traversal.paths`), which
shows how an answer was reached.

### 5.1 Progress over the project (answers correct, aggregation excluded)

Each row is a separate run, so differences of a few points are within run-to-run noise (section 8).

| Stage | Questions | Multi-hop: vector | Multi-hop: hybrid | Control: vector | Control: hybrid | All: vector | All: hybrid |
|---|---|---|---|---|---|---|---|
| v5, original pipeline | 112 | 26.4% | 45.3% | 87.0% | 86.4% | 58.3% | 67.0% |
| v6, selection fix (change 1) | 138 | 36.7% | 50.6% | 86.2% | 87.3% | 57.9% | 66.3% |
| + stricter prompt (change 2) | 138 | 34.2% | 53.2% | 89.0% | 84.1% | 57.6% | 66.4% |
| + decomposition (change 3), hybrid only | 138 | (34.2%) | 65.8% | (89.0%) | 85.8% | (57.6%) | 74.4% |
| + router (change 4), hybrid only | 138 | (34.2%) | 58.2% | (89.0%) | 89.0% | (57.6%) | 71.4% |
| **Final clean run** | 138 | **38.0%** | **59.5%** | **89.0%** | **90.7%** | **59.8%** | **72.8%** |

Figures in brackets reuse the vector row from the stricter-prompt run, since vector is unaffected by the later changes. The
v5 row uses a different, earlier question set, so it is not directly comparable with the v6 rows.

Multi-hop gap over time: **+18.9** points (v5 original) to **+21.5** (final), with the control group going from a small deficit
to a small lead, and the overall answer gap from +8.7 to +13.0 points.

### 5.2 Retrieval-selection fix in detail (v5 benchmark, retrieval only, right documents retrieved)

| Category | Vector | Hybrid: original selection | Hybrid: improved selection | Hybrid: final weights |
|---|---|---|---|---|
| Two-hop | 60.4% | 95.8% | 97.9% | **100%** |
| Three-hop | 40.6% | 73.9% | 85.5% | 84.1% |
| Four-hop or more | 28.9% | 62.8% | 68.3% | **73.3%** |
| Real single-hop | 91.5% | 93.6% | 93.6% | 93.6% |
| Single-hop | 100% | 100% | 100% | 100% |

The selection weights (question-word weight 0.5, same-document discount 0.6) were chosen from a sweep of five settings on this
benchmark.

### 5.3 Tried and not adopted

| Experiment | Result | Why dropped |
|---|---|---|
| Per-chunk coverage gate (add a graph chunk only if it covers question terms that vector's chunks do not) | multi-hop retrieval fell from 83.6% to 62.1% at the lightest setting | bridging chunks rarely add question words, so it removed the evidence multi-hop needs |
| Query-level coverage gate (skip the graph when vector chunks cover the question) | multi-hop retrieval fell from 84.6% to 79.0% at the lightest setting | cannot tell "vector has the answer" from "vector has the question's words"; replaced by the router |
| Graph clean-up at query time (skip junk and generic nodes, follow question-relevant edges first) | hybrid document retrieval 89.6% to 89.4% on v6 and 91.4% to 90.5% on v5 | slightly worse on both; left off (`graph_hygiene: false`) |
| Passage labels alone ("primary" and "additional") | hybrid stayed 4.9 points below vector on the control group | the router fixed it instead |

---

## 6. Benchmarks

| | v5 | v6 (used for the final results) |
|---|---|---|
| Documents | 76 | the same 76 (already ingested) |
| Questions | 126 | 148 (138 reported) |
| Purpose | first quantitative comparison | cleaner, harder-to-game version |
| Answer reuse | up to 8 multi-hop questions ending at the same person | no answer used by more than 3 multi-hop questions |
| Gold answers | keywords | keywords, plus an `evidence` list for each new question (document and exact text for every hop) |
| Retrieval metric | keyword only | keyword plus the strict right-documents metric |

How v6 was built: `benchmark/build_gold.py` refuses to produce the question file unless every piece of evidence appears in
its document, so each gold answer is traceable to the corpus text. Single-fact questions come from the real course
documents and the synthetic ones and form the control group.

Question mix reported here: real single-hop 47, single-hop 12, two-hop 46, three-hop 23, four-hop 10 (138 in total).

---

## 7. Final configuration

Defaults in `src/config.py` (the settings used for the final run):

| Setting | Value | Meaning |
|---|---|---|
| `prompt_v2` | on | stricter answer rules and primary/additional passage labels |
| `graph_routing` | on | single-fact questions skip the graph |
| `decompose_multihop` | on | multi-hop questions are answered in steps |
| `graph_hygiene` | off | query-time graph clean-up (not adopted) |
| `graph_min_coverage_gain` | 0 (off) | coverage gate (not adopted) |
| `graph_lexical_weight` / `graph_same_doc_discount` | 0.5 / 0.6 | graph chunk selection weights |
| Chunks | 3 from vector, up to 8 from the graph | |

---

## 8. Limitations

- **The answer numbers come from a single run.** Between near-identical runs the multi-hop score moved by about 8
  points (65.8% against 58.2% for two runs with the same effective settings), and vector's own score moved by about 4. Read
  the multi-hop answer gap as roughly +14 to +29 points. The question-by-question sign test (24 wins against 7 losses) and
  the retrieval gap, which is deterministic, support the direction of the result.
- **The tuning used these questions.** The selection weights were chosen on v5, and the router's wording cues were refined
  after seeing which questions an earlier version missed. The cues are general wording patterns, not corpus-specific, but v6
  is not a fully held-out test.
- **Hybrid uses more context and more model calls** than vector (about 11 chunks against 3, plus decomposition), so not all of
  the gain is attributable to the graph alone; a matched-chunk vector baseline was not measured.
- **Aggregation (list) questions are excluded** (10 questions); hybrid trailed vector there, and it is a known weak spot.
- **The corpus is synthetic.** Its multi-hop questions were written to need several documents, so this is a stress test of
  cross-document retrieval, not a sample of typical queries. The single-fact control group shows hybrid does not hurt where the
  graph is not needed.
- **Scoring is a keyword match.** A correct answer phrased differently can score low, and an answer that names an expected
  keyword incidentally can score high. A "does not contain the answer" reply counts as wrong.
- **Four-hop questions remain weak** (20.0% against 10.0% for vector, on 10 questions).
- **Multi-hop answers are slower** (about 12.6 s against 0.9 s) because each question is answered in several model calls.
- **Graph quality is limited by the 3B extraction model:** 43% of stored relation phrases are longer than four words, there are
  2,300 distinct relation strings across 2,555 edges, and 64% of entities have a single edge.
- **No scores exist for earlier versions (v1 to v4)**, so no improvement over them is claimed.

---

## 9. Reproduce

```powershell
ollama serve
.venv\Scripts\python.exe -m uvicorn main:app --port 8000

# full run, both modes (about 40 minutes with decomposition)
.venv\Scripts\python.exe eval\run_benchmark.py --gold benchmark\gold_qa.json --out eval\results\my_run.csv

# retrieval only, no language model (a few minutes)
.venv\Scripts\python.exe eval\run_benchmark.py --gold benchmark\gold_qa.json --out eval\results\my_retrieval.csv --retrieval-only
```

The runner refuses to overwrite an existing results file unless `--resume` is passed. To ingest the benchmark corpus into an
empty database: `.venv\Scripts\python.exe benchmark\ingest.py --dir benchmark\corpus` (about 3 to 6 hours).

Result files in `eval/results/`:

| File | Contents |
|---|---|
| `v6_final.csv` | final clean run, both modes (the numbers in sections 1 to 4) |
| `v6_original_run.csv` | v6 with the original pipeline |
| `v6_exp_A_prompt_v2.csv`, `v6_exp_C_decomposition.csv`, `v6_exp_G1_router_decomposition.csv`, `v6_exp_G2_plus_cleanup.csv` | experiments in section 5.1 and 5.3 |
| `v6_retrieval_final_router.csv`, `v6_retrieval_before_router.csv` | retrieval-only runs |
| `v5_original_run.csv`, `v5_retrieval_original_selection.csv`, `v5_retrieval_selection_fix.csv`, `v5_retrieval_final_router.csv` | v5 runs |

---

## 10. Repository layout

| Path | Contents |
|---|---|
| `main.py`, `config.yaml`, `src/` | the API server and the retrieval, graph and generation code |
| `web/` | browser front end |
| `benchmark/` | the ingested corpus (`corpus/`), the 148 questions (`gold_qa.json`), their builder (`build_gold.py`), the v5 questions it draws on (`legacy_gold_v5.json`), the ingestion script (`ingest.py`) and a README |
| `eval/` | the benchmark runner (`run_benchmark.py`) and every recorded result (`results/`) |
| `docs/` | the project report and the codebase explanation |
| `tests/` | API tests (they reset the database, so run them only against a scratch database) |
| `data/` | the live graph database, vector store and embedding cache (not tracked by git) |

---

## 11. Future work

None of these has been tried or measured yet.

1. **Aggregation and list questions.** Retrieve per entity and merge, or enumerate matches step by step; hybrid currently trails vector here.
2. **Four-hop questions (20%).** More decomposition steps, an answer-verification pass, and a stronger generator model.
3. **Cheaper multi-hop answers.** Run independent steps in parallel, shorten the step prompts and cache step results, to bring about 12 s closer to the single-pass time.
4. **A better graph.** Re-extract into a new table with a closed relation vocabulary and entity canonicalisation, using a stronger extraction model; this would raise the ceiling for every graph-based gain without touching the current data.
5. **A lightweight reranker** over the merged candidate chunks.
6. **Stronger evidence.** Repeated runs with confidence intervals, a matched-chunk vector baseline, and a held-out question set the tuning has never seen.
7. **A larger generator model for both modes,** to check that the advantage persists as the model improves.
8. **Real-world corpora.** Evaluate on larger, non-synthetic document collections.
9. **Front end.** Show the reasoning steps and graph paths that the API already returns.
10. **Housekeeping.** Safe tests that never touch the live database, pinned dependencies including the spaCy model, and a refreshed README.
