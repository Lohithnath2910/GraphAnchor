"""
Quantitative benchmark: plain vector RAG vs hybrid graph RAG.

For every question in the gold file, the running GraphAnchor server is queried in two modes:
  vector          plain vector RAG, top-k chunks (k=3 by default)
  hybrid          vector top-k + graph traversal (what GraphAnchor adds)

Per query it records:
  retrieval recall   do the retrieved chunks contain the expected key facts?
  answer correctness does the generated answer contain them (and avoid forbidden phrases)?
  latency            end-to-end seconds for the /query call

Robustness: every query gets a generous timeout and automatic retries, results are written to the
CSV after every query (a crash or Ctrl+C loses nothing), and --resume skips finished queries.

Requirements: the server is running (`uvicorn main:app --port 8000`) with the corpus already
ingested. Standard library only.

Usage:
  python eval/run_benchmark.py --gold benchmark/gold_qa.json --out eval/results/my_run.csv
  python eval/run_benchmark.py ... --resume          # continue an interrupted run
  python eval/run_benchmark.py ... --timeout 1200    # seconds allowed per query (default 900)
"""

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from statistics import mean, median

MODES = ("vector", "hybrid")
FIELDS = ["id", "category", "confidence", "mode", "k_used", "question",
          "latency_sec", "retrieval_recall", "doc_recall", "answer_correctness", "answer"]


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


EXTRA_PARAMS = {}  # hybrid-only overrides from --min-gain / --decompose


def call_query(base_url, question, enable_graph, k, timeout, answer=True):
    p = {"q": question, "k": k, "enable_graph": str(enable_graph).lower(), "answer": str(answer).lower()}
    if enable_graph:
        p.update(EXTRA_PARAMS)
    params = urllib.parse.urlencode(p)
    url = f"{base_url}/query?{params}"
    start = time.perf_counter()
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    latency = time.perf_counter() - start
    return data, latency


def call_with_retries(base_url, question, enable_graph, k, timeout, retries, answer=True):
    # Retries transient failures (timeouts, dropped connections, Ollama hiccups) with a growing pause.
    last = None
    for attempt in range(retries + 1):
        try:
            return call_query(base_url, question, enable_graph, k, timeout, answer)
        except Exception as e:
            last = e
            if attempt < retries:
                print(f"      retry {attempt + 1}/{retries} after error: {e}", flush=True)
                time.sleep(5 * (attempt + 1))
    raise last


def gather_retrieved_text(result):
    return " ".join(c.get("text", "") for c in result.get("ranking_breakdown", []) or []).lower()


def doc_recall(gold, result):
    # Strict retrieval metric: share of the documents the question needs (gold source_docs) that appear among the
    # retrieved chunks. Unlike keyword recall it is not fooled by a name that recurs in many documents.
    need = gold.get("source_docs") or []
    if not need:
        return None
    files = [(c.get("metadata") or {}).get("filename") or "" for c in result.get("ranking_breakdown", []) or []]
    def hit(n):
        return any(f.startswith(f"{n:02d}_") for f in files) if isinstance(n, int) else any(n in f for f in files)
    return sum(1 for n in need if hit(n)) / len(need)


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


def fmt(v):
    return "n/a" if v in (None, "") else f"{float(v):.2f}"


def summarize(subset, label):
    lat = [float(r["latency_sec"]) for r in subset if r["latency_sec"] != ""]
    rec = [float(r["retrieval_recall"]) for r in subset if r["retrieval_recall"] != ""]
    acc = [float(r["answer_correctness"]) for r in subset if r["answer_correctness"] != ""]
    docs = [float(r["doc_recall"]) for r in subset if r.get("doc_recall") not in (None, "")]
    if not lat:
        print(f"  {label:15s} no scored data")
        return
    full = sum(1 for a in acc if a == 1.0)
    ans = f"{mean(acc):6.1%}" if acc else "   n/a"
    dr = f"{mean(docs):6.1%}" if docs else "   n/a"
    print(f"  {label:15s} n={len(lat):2d}  keyword-retr={mean(rec):6.1%}  doc-retr={dr}  answer={ans}  "
          f"fully-correct={full:2d}/{len(acc):<2d}  latency mean/med/p95={mean(lat):.1f}/{median(lat):.1f}/{percentile(lat, 95):.1f}s")


def head_to_head(rows, a, b):
    # Per-question answer-correctness comparison of mode a vs mode b.
    by_q = {}
    for r in rows:
        if r["answer_correctness"] != "":
            by_q.setdefault(r["id"], {})[r["mode"]] = float(r["answer_correctness"])
    win = tie = loss = 0
    for modes in by_q.values():
        if a in modes and b in modes:
            if modes[a] > modes[b]:
                win += 1
            elif modes[a] < modes[b]:
                loss += 1
            else:
                tie += 1
    print(f"  {a} vs {b}: {a} better on {win} questions, same on {tie}, worse on {loss}")


def print_summary(rows, needs_review):
    cats = list(dict.fromkeys(r["category"] for r in rows))
    print("\n" + "=" * 100)
    print("OVERALL")
    print("=" * 100)
    for mode in MODES:
        summarize([r for r in rows if r["mode"] == mode], mode)
    for cat in cats:
        print("\n" + "-" * 100)
        print(cat.upper())
        print("-" * 100)
        for mode in MODES:
            summarize([r for r in rows if r["mode"] == mode and r["category"] == cat], mode)
    print("\n" + "=" * 100)
    print("HEAD-TO-HEAD (answer correctness per question)")
    print("=" * 100)
    head_to_head(rows, "hybrid", "vector")
    if needs_review:
        print("\nQUESTIONS TO DOUBLE-CHECK BEFORE TRUSTING THE NUMBERS")
        for qid, question, note in needs_review:
            print(f"  Q{qid}: {question}" + (f"  (note: {note})" if note else ""))


def run(base_url, gold_path, out_path, k, timeout, retries, resume, retrieval_only=False):
    gold = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    out = Path(out_path)

    rows = []
    done = set()
    if resume and out.exists():
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        done = {(int(r["id"]), r["mode"]) for r in rows}
        print(f"Resuming: {len(done)} queries already in {out}", flush=True)

    if out.exists() and out.stat().st_size > 0 and not resume:
        raise SystemExit(f"{out} already has results. Use --resume to continue it, or choose a new --out.")
    new_file = not (resume and out.exists())
    f = open(out, "w" if new_file else "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()
        f.flush()

    needs_review = []
    consecutive_errors = 0
    t_all = time.perf_counter()
    try:
        for n, g in enumerate(gold, 1):
            if g.get("confidence") in ("low",) or not g.get("expected_keywords"):
                needs_review.append((g["id"], g["question"], g.get("note", "")))

            hybrid_k = k  # filled in once the hybrid call tells us how many chunks it used
            for mode in MODES:
                if (g["id"], mode) in done:
                    # Rebuild hybrid_k from the saved row so vector_matched stays consistent on resume.
                    if mode == "hybrid":
                        hybrid_k = int(next(r["k_used"] for r in rows if int(r["id"]) == g["id"] and r["mode"] == "hybrid") or k)
                    continue

                enable_graph = mode == "hybrid"
                k_used = hybrid_k if mode == "vector_matched" else k
                k_used = max(1, min(20, k_used))  # server limit is 20
                try:
                    result, latency = call_with_retries(base_url, g["question"], enable_graph, k_used, timeout, retries,
                                                        answer=not retrieval_only)
                    answer = result.get("answer", "")
                    retrieved_text = gather_retrieved_text(result)
                    if mode == "hybrid":
                        hybrid_k = len(result.get("ranking_breakdown", []) or []) or k
                    retrieval_recall, answer_correctness = score_entry(g, retrieved_text, answer)
                    d_recall = doc_recall(g, result)
                    if retrieval_only:
                        answer_correctness = None
                except Exception as e:
                    answer = f"[ERROR: {e}]"
                    latency = None
                    retrieval_recall = None
                    answer_correctness = None
                    d_recall = None

                row = {
                    "id": g["id"],
                    "category": g["category"],
                    "confidence": g.get("confidence", ""),
                    "mode": mode,
                    "k_used": hybrid_k if mode == "hybrid" else k_used,
                    "question": g["question"],
                    "latency_sec": round(latency, 3) if latency is not None else "",
                    "retrieval_recall": round(retrieval_recall, 3) if retrieval_recall is not None else "",
                    "doc_recall": round(d_recall, 3) if d_recall is not None else "",
                    "answer_correctness": round(answer_correctness, 3) if answer_correctness is not None else "",
                    "answer": answer.replace("\n", " "),
                }
                rows.append(row)
                writer.writerow(row)
                f.flush()

                consecutive_errors = consecutive_errors + 1 if latency is None else 0
                if consecutive_errors >= 3:
                    # A stalled server fails every query; stop instead of burning hours of timeouts.
                    raise SystemExit("3 queries in a row failed: the server looks stalled. Restart uvicorn, "
                                     "then rerun the same command with --resume (failed rows are kept; delete them "
                                     "from the CSV first if you want them retried).")

                print(f"[{n}/{len(gold)}] {mode:14s} {g['category']:12s} k={row['k_used']:<2} "
                      f"{'ERR' if latency is None else f'{latency:6.1f}s'} "
                      f"retr={fmt(retrieval_recall)} ans={fmt(answer_correctness)} "
                      f"(elapsed {(time.perf_counter() - t_all) / 60:.1f} min)", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted. Partial results are saved; rerun with --resume to continue.", flush=True)
    finally:
        f.close()

    print(f"\nDetailed per-question results: {out}")
    if rows:
        print_summary(rows, needs_review)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--gold", default=str(Path(__file__).parent.parent / "benchmark" / "gold_qa.json"))
    parser.add_argument("--out", default=str(Path(__file__).parent / "results" / "run.csv"))
    parser.add_argument("--k", type=int, default=3, help="vector chunks per query (default 3)")
    parser.add_argument("--timeout", type=int, default=180, help="seconds allowed per query (default 180)")
    parser.add_argument("--retries", type=int, default=1, help="retries per failed query (default 2)")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run from --out")
    parser.add_argument("--min-gain", type=float, default=None, help="hybrid only: override graph_min_coverage_gain")
    parser.add_argument("--decompose", choices=["on", "off"], default=None, help="hybrid only: force multi-step decomposition on or off")
    parser.add_argument("--modes", default="vector,hybrid", help="comma-separated subset of vector,hybrid to run")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="skip the LLM answer: measures retrieval only, about 10x faster")
    args = parser.parse_args()
    MODES = tuple(m for m in args.modes.split(",") if m in ("vector", "hybrid"))
    if args.min_gain is not None:
        EXTRA_PARAMS["min_gain"] = args.min_gain
    if args.decompose:
        EXTRA_PARAMS["decompose"] = str(args.decompose == "on").lower()
    run(args.base, args.gold, args.out, args.k, args.timeout, args.retries, args.resume, args.retrieval_only)
