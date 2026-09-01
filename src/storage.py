import sqlite3
import chromadb
from contextlib import contextmanager
try:
    from .config import config
except ImportError:
    from src.config import config
import os
import logging
import shutil

logger = logging.getLogger("graphanchor")

# Ensure data directory exists
os.makedirs(os.path.dirname(config.chroma_path), exist_ok=True)

# Chroma Setup
chroma_client = chromadb.PersistentClient(
    path=config.chroma_path,
    settings=chromadb.Settings(allow_reset=True)
)

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
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                content_hash TEXT UNIQUE,
                filename TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT,
                text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_entity TEXT,
                relation TEXT,
                target_entity TEXT,
                chunk_id TEXT,
                confidence REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_entity)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_entity)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash)")
        conn.commit()

@contextmanager
def db_cursor():
    """Context manager for SQLite operations that automatically commits,
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
    """Wipes all data from SQLite and ChromaDB, removes disk segment folders, and re-initializes clean schemas."""
    global collection, entities_collection

    # 1. Clear SQLite tables
    with db_cursor() as cursor:
        cursor.execute("DELETE FROM chunks")
        cursor.execute("DELETE FROM edges")
        cursor.execute("DELETE FROM documents")
        
    # 2. Reset ChromaDB via client reset
    try:
        chroma_client.reset()
    except Exception as e:
        logger.warning(f"ChromaDB client.reset() encountered: {e}")
        try:
            chroma_client.delete_collection("chunks")
        except Exception:
            pass
        try:
            chroma_client.delete_collection("entities")
        except Exception:
            pass

    # 3. Clean up orphaned UUID segment directories on disk
    if os.path.exists(config.chroma_path):
        for item in os.listdir(config.chroma_path):
            item_path = os.path.join(config.chroma_path, item)
            if os.path.isdir(item_path):
                try:
                    shutil.rmtree(item_path, ignore_errors=True)
                except Exception as clean_err:
                    logger.debug(f"Could not remove segment directory {item}: {clean_err}")

    # 4. Recreate fresh clean collections
    collection = get_chunks_collection()
    entities_collection = get_entities_collection()
    return True

init_db()



