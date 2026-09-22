"""
Quantitative benchmark: plain vector RAG vs hybrid graph RAG.

What this does, in plain terms:
  For every question in gold_qa.json, it calls the running GraphAnchor server
  twice - once with graph traversal off (plain vector RAG) and once with it
  on (hybrid graph RAG) - and checks three things for each:
    1. Retrieval coverage: do the retrieved chunks actually contain the
       key facts needed to answer the question?
    2. Answer correctness: does the generated answer contain those same
       key facts, and does it avoid known-wrong phrases?
    3. Latency: how long did the /query call take, end to end?

Requirements before running:
  1. The GraphAnchor server must be running: `uvicorn main:app --port 8000`
  2. The same 30-document synthetic corpus used for the qualitative
     comparison must already be ingested into that running server
     (fresh DB via /reset, then re-ingest the corpus).
  3. Nothing beyond the Python standard library is required - this script
     deliberately avoids adding new dependencies to the project.

Usage:
  python eval/run_benchmark.py [--base http://localhost:8000] [--out eval/results.csv]

Output:
  - Prints a summary table to the console (overall + per-category, per mode).
  - Writes a detailed per-question CSV to eval/results.csv (or --out).
  - Prints a short list of low/medium-confidence questions that need a
    human to double-check the gold answer before the numbers are trusted.
"""

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from statistics import mean, median


def percentile(values, pct):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def call_query(base_url, question, enable_graph, timeout=120):
    params = urllib.parse.urlencode({"q": question, "enable_graph": str(enable_graph).lower()})
    url = f"{base_url}/query?{params}"
    start = time.perf_counter()
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    latency = time.perf_counter() - start
    return data, latency


def gather_retrieved_text(result):
    parts = []
    for c in result.get("ranking_breakdown", []) or []:
        parts.append(c.get("text", ""))
    return " ".join(parts).lower()


def score_entry(gold, retrieved_text, answer_text):
    answer_lower = (answer_text or "").lower()
    keywords = gold.get("expected_keywords", [])
    forbidden = gold.get("forbidden_phrases", [])

    if not keywords:
        # Nothing to score against (low-confidence / unresolved gold entry).
        return None, None

    retrieval_hits = sum(1 for kw in keywords if kw.lower() in retrieved_text)
    retrieval_recall = retrieval_hits / len(keywords)

    answer_hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    has_forbidden = any(fp.lower() in answer_lower for fp in forbidden)
    answer_correctness = (answer_hits / len(keywords)) if not has_forbidden else 0.0

    return retrieval_recall, answer_correctness


def run(base_url, gold_path, out_path):
    gold = json.loads(Path(gold_path).read_text())

    rows = []
    needs_review = []

    for g in gold:
        if g.get("confidence") in ("low",) or not g.get("expected_keywords"):
            needs_review.append((g["id"], g["question"], g.get("note", "")))

        for mode_name, enable_graph in (("vector", False), ("hybrid", True)):
            try:
                result, latency = call_query(base_url, g["question"], enable_graph)
                answer = result.get("answer", "")
                retrieved_text = gather_retrieved_text(result)
                retrieval_recall, answer_correctness = score_entry(g, retrieved_text, answer)
            except Exception as e:
                answer = f"[ERROR: {e}]"
                latency = None
                retrieval_recall = None
                answer_correctness = None

            rows.append({
                "id": g["id"],
                "category": g["category"],
                "confidence": g.get("confidence", ""),
                "mode": mode_name,
                "question": g["question"],
                "latency_sec": round(latency, 3) if latency is not None else "",
                "retrieval_recall": round(retrieval_recall, 3) if retrieval_recall is not None else "",
                "answer_correctness": round(answer_correctness, 3) if answer_correctness is not None else "",
                "answer": answer,
            })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDetailed per-question results written to: {out_path}\n")

    def summarize(subset, label):
        lat = [r["latency_sec"] for r in subset if r["latency_sec"] != ""]
        rec = [r["retrieval_recall"] for r in subset if r["retrieval_recall"] != ""]
        acc = [r["answer_correctness"] for r in subset if r["answer_correctness"] != ""]
        if not lat:
            print(f"  {label}: no scored data")
            return
        print(f"  {label}:")
        print(f"    n = {len(lat)}")
        print(f"    retrieval recall     = {mean(rec):.2%}" if rec else "    retrieval recall     = n/a")
        print(f"    answer correctness   = {mean(acc):.2%}" if acc else "    answer correctness   = n/a")
        print(f"    latency mean/median/p95 = {mean(lat):.2f}s / {median(lat):.2f}s / {percentile(lat, 95):.2f}s")

    print("=" * 60)
    print("OVERALL")
    print("=" * 60)
    for mode in ("vector", "hybrid"):
        summarize([r for r in rows if r["mode"] == mode], f"{mode} RAG")

    print()
    for cat in ("single_hop", "multi_hop"):
        print("=" * 60)
        print(cat.upper())
        print("=" * 60)
        for mode in ("vector", "hybrid"):
            summarize([r for r in rows if r["mode"] == mode and r["category"] == cat], f"{mode} RAG")
        print()

    if needs_review:
        print("=" * 60)
        print("QUESTIONS TO DOUBLE-CHECK BEFORE TRUSTING THE NUMBERS")
        print("=" * 60)
        for qid, question, note in needs_review:
            print(f"  Q{qid}: {question}")
            if note:
                print(f"        note: {note}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:8000")
    parser.add_argument("--gold", default=str(Path(__file__).parent / "gold_qa.json"))
    parser.add_argument("--out", default=str(Path(__file__).parent / "results.csv"))
    args = parser.parse_args()
    run(args.base, args.gold, args.out)
