import sqlite3
import chromadb
from contextlib import contextmanager
from src.config import config
import os

# Ensure data directory exists
os.makedirs(os.path.dirname(config.chroma_path), exist_ok=True)

# Chroma Setup
chroma_client = chromadb.PersistentClient(path=config.chroma_path)
collection = chroma_client.get_or_create_collection(name="chunks")

# SQLite Setup
def init_db():
    conn = sqlite3.connect(config.db_path)
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

def get_db_connection():
    return sqlite3.connect(config.db_path)

@contextmanager
def db_cursor():
    """Context manager: yields a cursor, commits on success, rolls back and
    re-raises on any error, and always closes the connection. Use this instead
    of get_db_connection() directly so a failure mid-request can't leave a
    connection open or a half-written transaction behind."""
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

init_db()
