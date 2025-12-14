"""
Basic sqlite backed memory DB for storing short messages and metadata.
Includes a tiny API to save and fetch recent memory.
"""
import sqlite3
from ..config import CONFIG
from pathlib import Path
import json
import time

DB_PATH = Path(CONFIG["db_path"])

def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL,
        source TEXT,
        text TEXT,
        meta TEXT
    )
    """)
    conn.commit()
    return conn

def add_memory(text: str, source: str = "user", meta: dict = None):
    conn = _connect()
    c = conn.cursor()
    c.execute("INSERT INTO memories (timestamp, source, text, meta) VALUES (?, ?, ?, ?)",
              (time.time(), source, text, json.dumps(meta or {})))
    conn.commit()
    conn.close()

def recent(n=20):
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT id, timestamp, source, text, meta FROM memories ORDER BY id DESC LIMIT ?", (n,))
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "timestamp": r[1], "source": r[2], "text": r[3], "meta": json.loads(r[4])} for r in rows]
