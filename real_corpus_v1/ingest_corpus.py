"""
Ingest all 30 .txt files from the real corpus into a running GraphAnchor server.

Usage:
    python ingest_corpus.py --dir corpus --base http://localhost:8000
"""
import argparse
import json
import os
import time
import urllib.request


def ingest_file(base_url, path):
    boundary = "----graphanchorboundary"
    filename = os.path.basename(path)
    with open(path, "rb") as f:
        content = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: text/plain\r\n\r\n"
    ).encode("utf-8") + content + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/ingest",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=6000) as resp:
        return resp.read().decode("utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="corpus")
    parser.add_argument("--base", default="http://localhost:8000")
    args = parser.parse_args()

    files = sorted(f for f in os.listdir(args.dir) if f.endswith(".txt"))
    print(f"Found {len(files)} files to ingest.\n", flush=True)

    t_all = time.time()
    for i, fname in enumerate(files, 1):
        path = os.path.join(args.dir, fname)
        print(f"[{i}/{len(files)}] ...  {fname}", flush=True)
        t0 = time.time()
        try:
            result = json.loads(ingest_file(args.base, path))
            print(f"[{i}/{len(files)}] OK   {fname}  {time.time() - t0:.0f}s  "
                  f"chunks={result['chunks_processed']} edges={result['edges_added']} "
                  f"failures={result.get('extraction_failures', 0)}  "
                  f"(elapsed {(time.time() - t_all) / 60:.1f} min)", flush=True)
        except Exception as e:
            print(f"[{i}/{len(files)}] FAIL {fname}: {e}", flush=True)
        time.sleep(0.5)

    print("\nDone. Check /graph/stats on the server to confirm counts.", flush=True)
