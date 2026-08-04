import sqlite3
import chromadb
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
    
    conn.commit()
    conn.close()

def get_db_connection():
    return sqlite3.connect(config.db_path)

init_db()
