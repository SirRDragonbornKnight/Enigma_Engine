"""
Async Memory Operations for Enigma Engine
Provides async/await interface for memory database and vector database operations.
"""
import asyncio
import json
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class AsyncMemoryDatabase:
    """Async SQLite-backed memory database."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize async memory database.
        
        Args:
            db_path: Path to SQLite database file
        """
        from ..config import CONFIG
        self.db_path = db_path or Path(CONFIG["db_path"])
        self._initialized = False
    
    async def _init_db(self):
        """Initialize database schema."""
        try:
            import aiosqlite
        except ImportError:
            raise ImportError(
                "aiosqlite not installed. Install with: pip install aiosqlite"
            )
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    source TEXT,
                    text TEXT,
                    meta TEXT
                )
            """)
            
            # Create indexes
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON memories(timestamp DESC)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_source 
                ON memories(source)
            """)
            
            await db.commit()
        
        self._initialized = True
    
    async def add_memory(
        self,
        text: str,
        source: str = "user",
        meta: Optional[Dict] = None
    ) -> int:
        """
        Add a memory asynchronously.
        
        Args:
            text: Memory content
            source: Source of the memory
            meta: Optional metadata
            
        Returns:
            ID of the inserted memory
        """
        if not self._initialized:
            await self._init_db()
        
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "INSERT INTO memories (timestamp, source, text, meta) VALUES (?, ?, ?, ?)",
                (time.time(), source, text, json.dumps(meta or {}))
            )
            await db.commit()
            return cursor.lastrowid
    
    async def get_recent(self, n: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent memories asynchronously.
        
        Args:
            n: Number of memories to retrieve
            
        Returns:
            List of memory dictionaries
        """
        if not self._initialized:
            await self._init_db()
        
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT id, timestamp, source, text, meta FROM memories ORDER BY id DESC LIMIT ?",
                (n,)
            ) as cursor:
                rows = await cursor.fetchall()
                
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
    
    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search memories by text content asynchronously.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching memory dictionaries
        """
        if not self._initialized:
            await self._init_db()
        
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT id, timestamp, source, text, meta 
                FROM memories 
                WHERE text LIKE ? 
                ORDER BY timestamp DESC 
                LIMIT ?
                """,
                (f"%{query}%", limit)
            ) as cursor:
                rows = await cursor.fetchall()
                
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
    
    async def get_by_id(self, id_: int) -> Optional[Dict[str, Any]]:
        """
        Get a memory by ID asynchronously.
        
        Args:
            id_: Memory ID
            
        Returns:
            Memory dictionary or None if not found
        """
        if not self._initialized:
            await self._init_db()
        
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT id, timestamp, source, text, meta FROM memories WHERE id = ?",
                (id_,)
            ) as cursor:
                row = await cursor.fetchone()
                
                if row:
                    return {
                        "id": row[0],
                        "timestamp": row[1],
                        "source": row[2],
                        "text": row[3],
                        "meta": json.loads(row[4])
                    }
                return None
    
    async def delete(self, id_: int) -> bool:
        """
        Delete a memory by ID asynchronously.
        
        Args:
            id_: Memory ID
            
        Returns:
            True if deleted, False if not found
        """
        if not self._initialized:
            await self._init_db()
        
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("DELETE FROM memories WHERE id = ?", (id_,))
            await db.commit()
            return cursor.rowcount > 0
    
    async def count(self) -> int:
        """
        Get total number of memories asynchronously.
        
        Returns:
            Count of memories
        """
        if not self._initialized:
            await self._init_db()
        
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM memories") as cursor:
                row = await cursor.fetchone()
                return row[0]


class AsyncVectorDB:
    """Async wrapper for vector database operations."""
    
    def __init__(self, vector_db):
        """
        Initialize async vector DB wrapper.
        
        Args:
            vector_db: Underlying vector database instance
        """
        self.vector_db = vector_db
    
    async def add(
        self,
        vectors: np.ndarray,
        ids: List[str],
        metadata: Optional[List[Dict]] = None
    ) -> None:
        """
        Add vectors asynchronously.
        
        Args:
            vectors: Vectors to add
            ids: IDs for the vectors
            metadata: Optional metadata
        """
        # Run sync operation in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self.vector_db.add,
            vectors,
            ids,
            metadata
        )
    
    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5
    ) -> List:
        """
        Search for similar vectors asynchronously.
        
        Args:
            query_vector: Query vector
            top_k: Number of results
            
        Returns:
            List of (id, score, metadata) tuples
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.vector_db.search,
            query_vector,
            top_k
        )
    
    async def delete(self, ids: List[str]) -> None:
        """
        Delete vectors asynchronously.
        
        Args:
            ids: IDs to delete
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self.vector_db.delete,
            ids
        )
    
    async def save(self, path: Path) -> None:
        """
        Save vector DB asynchronously.
        
        Args:
            path: Path to save to
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self.vector_db.save,
            path
        )
    
    async def load(self, path: Path) -> None:
        """
        Load vector DB asynchronously.
        
        Args:
            path: Path to load from
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self.vector_db.load,
            path
        )
    
    async def count(self) -> int:
        """
        Get count asynchronously.
        
        Returns:
            Number of vectors
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.vector_db.count
        )
    
    @property
    def dim(self) -> int:
        """Get vector dimension."""
        return self.vector_db.dim
