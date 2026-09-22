# Quantitative benchmark (vector RAG vs hybrid graph RAG)

This folder turns the qualitative 30-question comparison into real numbers:
retrieval recall, answer correctness, and latency, for both plain vector RAG
and the hybrid graph RAG pipeline.

## Files
- `gold_qa.json` - the 30 questions with expected keywords per question
  (facts a correct answer must contain) and, where relevant, forbidden
  phrases (patterns that mark a known-wrong answer, e.g. "not directly
  connected" on a question where the two ARE connected).
- `run_benchmark.py` - calls the running server for every question, in both
  modes, scores the results, and writes `results.csv` plus a console
  summary. No extra pip installs needed (stdlib only).

## How to run it
1. Start the server: `uvicorn main:app --port 8000`
2. Make sure the same massive_dataset corpus used for the qualitative
   report is ingested (reset with `DELETE /reset?confirm=true` first if you
   want a clean run, then re-ingest the docs).
3. Run: `python eval/run_benchmark.py`
4. Read the console summary, and open `eval/results.csv` for the per-question
   breakdown (useful for picking 2-3 concrete examples to put in the report).

## Important: some gold answers need a human check
A few questions (see the "confidence" field in gold_qa.json, and the list
printed at the end of a run) had genuinely unclear or contradictory answers
even in the qualitative run - these are marked `"confidence": "low"` with a
`note` field, and have no expected_keywords so they are excluded from
scoring automatically. Before quoting the final numbers anywhere, skim
those flagged questions against the actual source documents and either fill
in real expected_keywords or leave them excluded on purpose - don't let the
benchmark script's silence on them be mistaken for "we checked, it's fine."

## What the numbers mean
- Retrieval recall: of the key facts a question needs, what fraction showed
  up somewhere in the chunks the system retrieved. Low retrieval recall on
  vector RAG for a multi-hop question is exactly the failure mode this
  whole project targets.
- Answer correctness: same idea, but checked against the final generated
  answer text (and zeroed out if a forbidden/known-wrong phrase appears).
- Latency: wall-clock time per `/query` call, mean/median/p95, so you can
  also report the hybrid engine's speed cost honestly (it does more work,
  so it should be slower - the question is by how much).
