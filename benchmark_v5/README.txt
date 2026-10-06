benchmark_v5 - large mixed benchmark: standard (vector) RAG vs hybrid (vector + graph) RAG

WHAT IS IN HERE
  corpus/          76 documents, ~700 chunks (chunk_size 300 tokens)
                     01-46  46 NEW documents, 9-13 chunks each (~1,900-2,500 words), set in an invented
                            region ("the Calder Reach"): 14 infrastructure projects, their incident
                            reports, 8 organizations and 10 people profiles. All names are invented, so
                            the LLM cannot answer from memory.
                     real_* your 30 real course documents (copied from real_corpus_v1/corpus, not moved)
  gold_qa.json     126 questions for eval/run_benchmark.py
  build_gold.py    source of gold_qa.json (edit the list and re-run to regenerate)

QUESTION MIX
  real_single_hop  47   from your real documents: facts that sit in one document (control group)
  single_hop       12   one fact in one new document
  multi_hop_2      24   two documents must be combined
  multi_hop_3      23   three documents chained
  multi_hop_4       6   four or more links
  aggregation      14   facts gathered from several documents (e.g. "which projects ...")
  Each item lists `source_docs` (the NN_ numbers of the documents needed) so you can audit it.

HOW THE WORLD IS CONNECTED (why the graph can matter)
  People move between organizations (a light rail director came from a freight company, a port director
  from a bank, a recycling director from a foundry). Projects share suppliers (one foundry group made
  hubs, casings, bearings, gates), power, timing and finance. Incident reports link the projects: one
  lightning strike caused three separate incidents; one cable fault caused a regional brownout that hit
  four more. Answers often sit in a different document from the entity the question starts from.

HOW TO RUN (clear data/ first; ingestion takes hours, see below)
  1. ollama serve
  2. Remove-Item -Recurse -Force data/graph.db, data/chroma
     .venv\Scripts\python.exe -m uvicorn main:app --port 8000
  3. .venv\Scripts\python.exe real_corpus_v1/ingest_corpus.py --dir benchmark_v5/corpus
     (76 files; prints one line per file with time, chunks, edges, extraction failures)
  4. Check http://localhost:8000/graph/stats : edge_count should be well over 1,500 (roughly 3+ edges per
     chunk). If it is far lower, stop and look at the extraction failures before benchmarking.
  5. .venv\Scripts\python.exe eval/run_benchmark.py --gold benchmark_v5/gold_qa.json --out eval/v5_results.csv
     (use --resume to continue an interrupted run)

EXPECTED TIME  (llama3.2 + snowflake-arctic-embed2 on your machine)
  Ingest:     ~700 chunks x roughly 15-30 s  =  about 3-6 hours.  Start it in the evening.
  Benchmark:  126 questions x 3 modes x ~4-6 s = about 30-45 minutes.

WHAT THE THREE MODES MEAN IN THE RESULTS
  vector          plain RAG, top-3 chunks
  hybrid          top-3 vector chunks + chunks found by walking the knowledge graph
  vector_matched  plain RAG given the same number of chunks hybrid ended up using (the fair baseline)
  If hybrid beats vector but not vector_matched, the gain came from retrieving more text, not from the graph.

BASELINE MEASURED OFFLINE (plain vector search, your embedding model, no graph, no LLM)
  retrieval recall = share of expected keywords found in the retrieved chunks
                      k=3     k=11
  real_single_hop     0.95    1.00
  single_hop          0.92    1.00
  multi_hop_2         0.46    0.71
  multi_hop_3         0.30    0.57
  multi_hop_4         0.00    0.67
  aggregation         0.79    0.90
  new questions only  0.51    0.74
  So there is real headroom for hybrid over vector at k=3, and less over vector at k=11.

HONEST CAVEATS
  - The new documents were written for this benchmark, and the multi-hop questions were written to need
    several documents. It is a stress test of cross-document retrieval, not a sample of typical queries.
  - Recurring names (a few chief executives and directors appear in many documents) make the keyword
    metric lenient: plain vector search can sometimes hit the answer's name without following the chain.
  - Whether hybrid wins depends on the graph the 3B model extracts. The extraction cap is now 20 relations
    per chunk (config.yaml: extraction_max_relations; set it to 12 to restore the old behaviour).
  - 126 questions, one run: differences of a few questions are not statistically solid.
