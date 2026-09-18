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

os.makedirs(os.path.dirname(config.chroma_path), exist_ok=True)

chroma_client = chromadb.PersistentClient(
    path=config.chroma_path,
    settings=chromadb.Settings(allow_reset=True)
)

# Manages the unified local storage layer combining SQLite for graph edges and ChromaDB for vector embeddings.

def get_chunks_collection():
    return chroma_client.get_or_create_collection(name="chunks")

def get_entities_collection():
    return chroma_client.get_or_create_collection(
        name="entities",
        metadata={"hnsw:space": "cosine"}
    )

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
        
        for tbl, col, col_type in [
            ("chunks", "created_at", "TIMESTAMP DEFAULT '1970-01-01 00:00:00'"),
            ("edges", "created_at", "TIMESTAMP DEFAULT '1970-01-01 00:00:00'"),
            ("documents", "filename", "TEXT"),
            ("documents", "created_at", "TIMESTAMP DEFAULT '1970-01-01 00:00:00'")
        ]:
            try:
                conn.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {col_type}")
            except sqlite3.OperationalError as e:
                err_msg = str(e).lower()
                if "duplicate column name" not in err_msg and "non-constant default" not in err_msg:
                    logger.error(f"Failed to migrate table {tbl}: {e}")
                    raise

        conn.commit()

@contextmanager
def db_cursor():
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
    global collection, entities_collection

    with db_cursor() as cursor:
        cursor.execute("DELETE FROM chunks")
        cursor.execute("DELETE FROM edges")
        cursor.execute("DELETE FROM documents")
        
    try:
        chroma_client.reset()
    except Exception as e:
        logger.warning(f"ChromaDB client.reset() encountered: {e}")
        try:
            chroma_client.delete_collection("chunks")
            chroma_client.delete_collection("entities")
        except Exception:
            pass

    if os.path.exists(config.chroma_path):
        for item in os.listdir(config.chroma_path):
            item_path = os.path.join(config.chroma_path, item)
            if os.path.isdir(item_path):
                try:
                    shutil.rmtree(item_path, ignore_errors=True)
                except Exception as clean_err:
                    logger.debug(f"Could not remove segment directory {item}: {clean_err}")

    collection = get_chunks_collection()
    entities_collection = get_entities_collection()
    return True

def list_documents():
    with db_cursor() as cursor:
        cursor.execute("""
            SELECT 
                d.doc_id, 
                d.filename, 
                d.content_hash, 
                d.created_at,
                COUNT(DISTINCT c.chunk_id) as chunk_count,
                COUNT(DISTINCT e.id) as edge_count
            FROM documents d
            LEFT JOIN chunks c ON d.doc_id = c.doc_id
            LEFT JOIN edges e ON c.chunk_id = e.chunk_id
            GROUP BY d.doc_id, d.filename, d.content_hash, d.created_at
            ORDER BY d.created_at DESC
        """)
        rows = cursor.fetchall()
        return [
            {
                "doc_id": r[0],
                "filename": r[1],
                "content_hash": r[2],
                "created_at": r[3],
                "chunk_count": r[4],
                "edge_count": r[5]
            }
            for r in rows
        ]

def delete_document(doc_id: str) -> bool:
    chunks_col = get_chunks_collection()
    entities_col = get_entities_collection()

    with db_cursor() as cursor:
        cursor.execute("SELECT doc_id FROM documents WHERE doc_id = ?", (doc_id,))
        if not cursor.fetchone():
            return False

        cursor.execute("SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,))
        chunk_ids = [row[0] for row in cursor.fetchall()]

        entities_to_check = set()
        if chunk_ids:
            placeholders = ",".join("?" * len(chunk_ids))
            cursor.execute(f"SELECT source_entity, target_entity FROM edges WHERE chunk_id IN ({placeholders})", chunk_ids)
            for s, t in cursor.fetchall():
                if s: entities_to_check.add(s)
                if t: entities_to_check.add(t)

            cursor.execute(f"DELETE FROM edges WHERE chunk_id IN ({placeholders})", chunk_ids)
            cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

        cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

        orphaned_entities = []
        for ent in entities_to_check:
            cursor.execute(
                "SELECT 1 FROM edges WHERE source_entity = ? OR target_entity = ? LIMIT 1",
                (ent, ent)
            )
            if not cursor.fetchone():
                orphaned_entities.append(ent)

    if chunk_ids:
        try:
            chunks_col.delete(ids=chunk_ids)
        except Exception as e:
            logger.warning(f"Failed to delete chunks from ChromaDB for doc {doc_id}: {e}")

    if orphaned_entities:
        try:
            entities_col.delete(ids=orphaned_entities)
        except Exception as e:
            logger.warning(f"Failed to delete orphaned entities from ChromaDB: {e}")

    return True

init_db()



