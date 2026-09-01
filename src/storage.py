import sqlite3
import chromadb
from contextlib import contextmanager
try:
    from .config import config
except ImportError:
    from src.config import config
import os
import logging

logger = logging.getLogger("graphanchor")

# Ensure data directory exists
os.makedirs(os.path.dirname(config.chroma_path), exist_ok=True)

# Chroma Setup
chroma_client = chromadb.PersistentClient(path=config.chroma_path)

def get_chunks_collection():
    return chroma_client.get_or_create_collection(name="chunks")

def get_entities_collection():
    return chroma_client.get_or_create_collection(
        name="entities",
        metadata={"hnsw:space": "cosine"}
    )

# Backward-compatibility module aliases
collection = get_chunks_collection()
entities_collection = get_entities_collection()


# SQLite Setup
def get_db_connection():
    conn = sqlite3.connect(config.db_path, timeout=10.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT,
            text TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_entity TEXT,
            relation TEXT,
            target_entity TEXT,
            chunk_id TEXT,
            confidence REAL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            content_hash TEXT UNIQUE,
            filename TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()

@contextmanager
def db_cursor():
    """Context manager: yields a cursor, commits on success, rolls back and
    re-raises on any error, and always closes the connection."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def reset_all_data():
    """Wipes all data from SQLite and ChromaDB, and re-initializes clean schemas."""
    global collection, entities_collection

    # 1. Clear SQLite tables
    with db_cursor() as cursor:
        cursor.execute("DELETE FROM chunks")
        cursor.execute("DELETE FROM edges")
        cursor.execute("DELETE FROM documents")
        
    # 2. Reset Chroma collections
    try:
        chroma_client.delete_collection("chunks")
    except Exception:
        pass
    try:
        chroma_client.delete_collection("entities")
    except Exception:
        pass

    # Recreate fresh collections
    collection = get_chunks_collection()
    entities_collection = get_entities_collection()
    return True

init_db()


