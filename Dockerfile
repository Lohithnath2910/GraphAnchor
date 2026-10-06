FROM python:3.11-slim

# uv, pinned to the version the lock file was made with
COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first so this layer is cached until pyproject.toml or uv.lock change. The spaCy model is a
# pinned wheel in the lock file, so nothing is downloaded when the server starts.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

# The tokenizer file is fetched once at build time, so the container also works without internet afterwards.
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY main.py config.yaml ./
COPY src ./src
COPY web ./web

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/graph/stats', timeout=4)" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
