"""
Advanced Memory Search for Enigma Engine
Provides full-text, semantic, and hybrid search capabilities.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .memory_db import MemoryDatabase
from .vector_db import VectorDBInterface
from .categorization import Memory, MemoryType, MemoryCategorization

logger = logging.getLogger(__name__)


class MemorySearch:
    """Advanced memory search capabilities."""
    
    def __init__(
        self,
        memory_db: Optional[MemoryDatabase] = None,
        memory_system: Optional[MemoryCategorization] = None,
        vector_db: Optional[VectorDBInterface] = None,
        embedding_generator=None
    ):
        """
        Initialize memory search.
        
        Args:
            memory_db: Memory database for SQL-based search
            memory_system: Memory categorization system
            vector_db: Vector database for semantic search
            embedding_generator: Generator for query embeddings
        """
        self.memory_db = memory_db or MemoryDatabase()
        self.memory_system = memory_system
        self.vector_db = vector_db
        self.embedding_generator = embedding_generator
        
        # Initialize FTS5 if needed
        if memory_db:
            self._init_fts5()
    
    def _init_fts5(self):
        """Initialize FTS5 full-text search table."""
        try:
            with self.memory_db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if FTS5 table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='memories_fts'
                """)
                
                if not cursor.fetchone():
                    # Create FTS5 virtual table
                    cursor.execute("""
                        CREATE VIRTUAL TABLE memories_fts USING fts5(
                            text,
                            content='memories',
                            content_rowid='id'
                        )
                    """)
                    
                    # Populate FTS5 table from existing memories
                    cursor.execute("""
                        INSERT INTO memories_fts(rowid, text)
                        SELECT id, text FROM memories
                    """)
                    
                    # Create triggers to keep FTS5 in sync
                    cursor.execute("""
                        CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
                            INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
                        END
                    """)
                    
                    cursor.execute("""
                        CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
                            DELETE FROM memories_fts WHERE rowid = old.id;
                        END
                    """)
                    
                    cursor.execute("""
                        CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
                            UPDATE memories_fts SET text = new.text WHERE rowid = new.id;
                        END
                    """)
                    
                    conn.commit()
                    logger.info("Initialized FTS5 full-text search")
        
        except Exception as e:
            logger.warning(f"Could not initialize FTS5: {e}")
    
    def full_text_search(self, query: str, limit: int = 10) -> List[Memory]:
        """
        SQLite FTS5 full-text search.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of Memory objects
        """
        try:
            with self.memory_db.get_connection() as conn:
                cursor = conn.cursor()
                
                # FTS5 query
                cursor.execute("""
                    SELECT m.id, m.timestamp, m.source, m.text, m.meta, 
                           bm25(memories_fts) as score
                    FROM memories_fts
                    JOIN memories m ON memories_fts.rowid = m.id
                    WHERE memories_fts MATCH ?
                    ORDER BY score
                    LIMIT ?
                """, (query, limit))
                
                rows = cursor.fetchall()
                
                memories = []
                for row in rows:
                    import json
                    meta = json.loads(row[4])
                    meta['fts_score'] = row[5]
                    
                    # Try to get from memory system first
                    if self.memory_system:
                        memory = self.memory_system.get_memory(str(row[0]))
                        if memory:
                            memory.metadata['fts_score'] = row[5]
                            memories.append(memory)
                            continue
                    
                    # Create basic Memory object
                    memory = Memory(
                        id=str(row[0]),
                        content=row[3],
                        memory_type=MemoryType.SHORT_TERM,
                        timestamp=row[1],
                        metadata=meta
                    )
                    memories.append(memory)
                
                return memories
        
        except Exception as e:
            logger.error(f"Full-text search error: {e}")
            # Fallback to basic LIKE search
            return self._fallback_text_search(query, limit)
    
    def _fallback_text_search(self, query: str, limit: int) -> List[Memory]:
        """Fallback text search using LIKE."""
        results = self.memory_db.search(query, limit)
        
        memories = []
        for result in results:
            memory = Memory(
                id=str(result['id']),
                content=result['text'],
                memory_type=MemoryType.SHORT_TERM,
                timestamp=result['timestamp'],
                metadata=result.get('meta', {})
            )
            memories.append(memory)
        
        return memories
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Memory]:
        """
        Vector similarity search.
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            List of Memory objects
        """
        if not self.vector_db:
            logger.warning("Vector DB not available for semantic search")
            return []
        
        # Generate query embedding
        if self.embedding_generator:
            query_embedding = self.embedding_generator.embed(query)
        else:
            # Use simple fallback
            import hashlib
            import numpy as np
            hash_obj = hashlib.sha256(query.encode())
            hash_bytes = hash_obj.digest()
            query_embedding = np.frombuffer(
                hash_bytes[:self.vector_db.dim * 4],
                dtype=np.float32
            )
            if len(query_embedding) < self.vector_db.dim:
                query_embedding = np.pad(
                    query_embedding,
                    (0, self.vector_db.dim - len(query_embedding))
                )
            else:
                query_embedding = query_embedding[:self.vector_db.dim]
        
        # Search vector DB
        results = self.vector_db.search(query_embedding, top_k)
        
        memories = []
        for mem_id, score, metadata in results:
            # Try to get from memory system
            if self.memory_system:
                memory = self.memory_system.get_memory(mem_id)
                if memory:
                    memory.metadata['semantic_score'] = score
                    memories.append(memory)
                    continue
            
            # Create from metadata
            content = metadata.get('content', '')
            memory = Memory(
                id=mem_id,
                content=content,
                memory_type=MemoryType.SHORT_TERM,
                timestamp=metadata.get('timestamp', 0),
                metadata={'semantic_score': score, **metadata}
            )
            memories.append(memory)
        
        return memories
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5
    ) -> List[Memory]:
        """
        Combine full-text and semantic search.
        
        Args:
            query: Query text
            top_k: Number of results
            alpha: Weight for semantic vs full-text (0=text only, 1=semantic only)
            
        Returns:
            List of Memory objects ranked by combined score
        """
        # Get results from both methods
        fts_results = self.full_text_search(query, limit=top_k * 2)
        semantic_results = self.semantic_search(query, top_k=top_k * 2)
        
        # Combine and score
        memory_scores: Dict[str, tuple] = {}  # id -> (memory, score)
        
        # Add FTS results
        for i, mem in enumerate(fts_results):
            fts_score = mem.metadata.get('fts_score', 0)
            # Normalize rank-based score (higher is better)
            rank_score = 1.0 - (i / len(fts_results))
            combined_score = (1 - alpha) * rank_score
            memory_scores[mem.id] = (mem, combined_score)
        
        # Add semantic results
        for i, mem in enumerate(semantic_results):
            semantic_score = mem.metadata.get('semantic_score', 0)
            # Normalize rank-based score
            rank_score = 1.0 - (i / len(semantic_results))
            combined_score = alpha * rank_score
            
            if mem.id in memory_scores:
                # Combine scores
                existing_mem, existing_score = memory_scores[mem.id]
                memory_scores[mem.id] = (existing_mem, existing_score + combined_score)
            else:
                memory_scores[mem.id] = (mem, combined_score)
        
        # Sort by combined score
        sorted_results = sorted(
            memory_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [mem for mem, score in sorted_results[:top_k]]
    
    def search_by_date_range(
        self,
        start: datetime,
        end: datetime,
        memory_type: Optional[MemoryType] = None
    ) -> List[Memory]:
        """
        Search memories within a date range.
        
        Args:
            start: Start datetime
            end: End datetime
            memory_type: Optional memory type filter
            
        Returns:
            List of Memory objects
        """
        if self.memory_system:
            # Use memory system
            memories = self.memory_system.get_all_memories()
            
            filtered = [
                mem for mem in memories
                if start.timestamp() <= mem.timestamp <= end.timestamp()
            ]
            
            if memory_type:
                filtered = [m for m in filtered if m.memory_type == memory_type]
            
            return sorted(filtered, key=lambda m: m.timestamp)
        
        else:
            # Use database
            with self.memory_db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, timestamp, source, text, meta
                    FROM memories
                    WHERE timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                """, (start.timestamp(), end.timestamp()))
                
                rows = cursor.fetchall()
                
                memories = []
                for row in rows:
                    import json
                    memory = Memory(
                        id=str(row[0]),
                        content=row[3],
                        memory_type=MemoryType.SHORT_TERM,
                        timestamp=row[1],
                        metadata=json.loads(row[4])
                    )
                    memories.append(memory)
                
                return memories
    
    def search_by_type(
        self,
        memory_type: MemoryType,
        query: Optional[str] = None,
        limit: int = 10
    ) -> List[Memory]:
        """
        Search within a specific memory type.
        
        Args:
            memory_type: Memory type to search
            query: Optional search query
            limit: Maximum number of results
            
        Returns:
            List of Memory objects
        """
        if not self.memory_system:
            logger.warning("Memory system not available for type-based search")
            return []
        
        memories = self.memory_system.get_memories_by_type(memory_type)
        
        if query:
            # Filter by query
            query_lower = query.lower()
            memories = [
                m for m in memories
                if query_lower in m.content.lower()
            ]
        
        # Sort by recency
        memories.sort(key=lambda m: m.timestamp, reverse=True)
        
        return memories[:limit]
