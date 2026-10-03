#!/usr/bin/env bash
# Ingests every .txt file in a corpus folder into a running GraphAnchor server.
# Usage: ./ingest_corpus.sh <corpus_dir> <base_url>
# Example: ./ingest_corpus.sh ./corpus http://localhost:8000

set -e
CORPUS_DIR="${1:-./corpus}"
BASE_URL="${2:-http://localhost:8000}"

count=0
for f in "$CORPUS_DIR"/*.txt; do
    echo "Ingesting: $f"
    curl -sS -X POST "$BASE_URL/ingest" -F "file=@$f" -o /dev/null -w "  -> HTTP %{http_code}\n"
    count=$((count+1))
done
echo "Ingested $count documents into $BASE_URL"
