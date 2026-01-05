"""
SQLite-backed memory database for storing short messages and metadata.
Provides both legacy API and new class-based interface with proper connection management.
"""
import sqlite3
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
import threading
from pathlib import Path
import json
import time
import logging

from ..config import CONFIG

logger = logging.getLogger(__name__)

DB_PATH = Path(CONFIG["db_path"])


class MemoryDatabase:
    """SQLite-backed memory database with proper connection management."""
    
    _local = threading.local()
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize memory database.
        
        Args:
            db_path: Path to SQLite database file (uses default if None)
        """
        self.db_path = db_path or DB_PATH
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use a direct connection for initialization
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    source TEXT,
                    text TEXT,
                    meta TEXT
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON memories(timestamp DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_source 
                ON memories(source)
            """)
            
            conn.commit()
        finally:
            conn.close()
    
    @contextmanager
    def get_connection(self):
        """
        Get a thread-local database connection.
        
        Yields:
            sqlite3.Connection: Database connection
        """
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False
            )
            # Enable row factory for dict-like access
            self._local.conn.row_factory = sqlite3.Row
        
        try:
            yield self._local.conn
        except Exception:
            # Rollback on error
            if self._local.conn:
                self._local.conn.rollback()
            raise
    
    def close(self):
        """Close all thread-local connections."""
        if hasattr(self._local, 'conn') and self._local.conn:
            try:
                self._local.conn.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
            finally:
                self._local.conn = None
    
    def add_memory(
        self,
        text: str,
        source: str = "user",
        meta: Optional[Dict] = None
    ) -> int:
        """
        Add a memory and return its ID.
        
        Args:
            text: Memory content
            source: Source of the memory (e.g., 'user', 'assistant')
            meta: Optional metadata dictionary
            
        Returns:
            ID of the inserted memory
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO memories (timestamp, source, text, meta) VALUES (?, ?, ?, ?)",
                (time.time(), source, text, json.dumps(meta or {}))
            )
            conn.commit()
            return cursor.lastrowid
    
    def get_recent(self, n: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent memories.
        
        Args:
            n: Number of memories to retrieve
            
        Returns:
            List of memory dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, timestamp, source, text, meta FROM memories ORDER BY id DESC LIMIT ?",
                (n,)
            )
            rows = cursor.fetchall()
            
            return [
                {
                    "id": row[0],
                    "timestamp": row[1],
                    "source": row[2],
                    "text": row[3],
                    "meta": json.loads(row[4])
                }
                for row in rows
            ]
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search memories by text content.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching memory dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, timestamp, source, text, meta 
                FROM memories 
                WHERE text LIKE ? 
                ORDER BY timestamp DESC 
                LIMIT ?
                """,
                (f"%{query}%", limit)
            )
            rows = cursor.fetchall()
            
            return [
                {
                    "id": row[0],
                    "timestamp": row[1],
                    "source": row[2],
                    "text": row[3],
                    "meta": json.loads(row[4])
                }
                for row in rows
            ]
    
    def get_by_id(self, id_: int) -> Optional[Dict[str, Any]]:
        """
        Get a memory by ID.
        
        Args:
            id_: Memory ID
            
        Returns:
            Memory dictionary or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, timestamp, source, text, meta FROM memories WHERE id = ?",
                (id_,)
            )
            row = cursor.fetchone()
            
            if row:
                return {
                    "id": row[0],
                    "timestamp": row[1],
                    "source": row[2],
                    "text": row[3],
                    "meta": json.loads(row[4])
                }
            return None
    
    def delete(self, id_: int) -> bool:
        """
        Delete a memory by ID.
        
        Args:
            id_: Memory ID
            
        Returns:
            True if deleted, False if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memories WHERE id = ?", (id_,))
            conn.commit()
            return cursor.rowcount > 0
    
    def count(self) -> int:
        """
        Get total number of memories.
        
        Returns:
            Count of memories
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM memories")
            return cursor.fetchone()[0]
    
    def clear_all(self):
        """Clear all memories (use with caution!)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memories")
            conn.commit()
            logger.warning("All memories cleared from database")


# Global instance for legacy API
_default_db = None


def _get_default_db() -> MemoryDatabase:
    """Get or create default database instance."""
    global _default_db
    if _default_db is None:
        _default_db = MemoryDatabase()
    return _default_db


# Legacy API - maintained for backwards compatibility
def _connect():
    """
    Legacy function: Create a new database connection.
    
    Deprecated: Use MemoryDatabase class instead.
    
    Returns:
        sqlite3.Connection: Database connection
    """
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
    """
    Legacy function: Add a memory to the database.
    
    Deprecated: Use MemoryDatabase class instead.
    
    Args:
        text: Memory content
        source: Source of the memory
        meta: Optional metadata dictionary
    """
    db = _get_default_db()
    db.add_memory(text, source, meta)


def recent(n=20):
    """
    Legacy function: Get recent memories.
    
    Deprecated: Use MemoryDatabase class instead.
    
    Args:
        n: Number of memories to retrieve
        
    Returns:
        List of memory dictionaries
    """
    db = _get_default_db()
    return db.get_recent(n)
