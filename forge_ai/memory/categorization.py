"""
Memory Categorization and TTL Management
Implements short-term and long-term memory with automatic pruning.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory."""
    SHORT_TERM = "short_term"  # Recent conversations, expires quickly
    LONG_TERM = "long_term"    # Important facts, persists
    WORKING = "working"         # Current context, cleared after session
    EPISODIC = "episodic"       # Specific events/experiences
    SEMANTIC = "semantic"       # General knowledge/facts


@dataclass
class Memory:
    """A single memory entry."""
    id: str
    content: str
    memory_type: MemoryType
    timestamp: float
    ttl: Optional[float] = None  # Time to live in seconds (None = permanent)
    importance: float = 0.5  # 0.0 to 1.0
    metadata: Dict[str, Any] = None
    access_count: int = 0
    last_accessed: float = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.last_accessed is None:
            self.last_accessed = self.timestamp
    
    def is_expired(self) -> bool:
        """Check if memory has expired."""
        if self.ttl is None:
            return False
        return (time.time() - self.timestamp) > self.ttl
    
    def access(self):
        """Mark memory as accessed."""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'content': self.content,
            'memory_type': self.memory_type.value,
            'timestamp': self.timestamp,
            'ttl': self.ttl,
            'importance': self.importance,
            'metadata': self.metadata,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create from dictionary."""
        data['memory_type'] = MemoryType(data['memory_type'])
        return cls(**data)


class MemoryCategory:
    """Manages memories of a specific type."""
    
    def __init__(self, memory_type: MemoryType, default_ttl: Optional[float] = None):
        """
        Initialize memory category.
        
        Args:
            memory_type: Type of memory
            default_ttl: Default time to live in seconds
        """
        self.memory_type = memory_type
        self.default_ttl = default_ttl
        self.memories: Dict[str, Memory] = {}
    
    def add(
        self,
        id_: str,
        content: str,
        ttl: Optional[float] = None,
        importance: float = 0.5,
        metadata: Optional[Dict] = None
    ) -> Memory:
        """Add a memory."""
        ttl = ttl if ttl is not None else self.default_ttl
        
        memory = Memory(
            id=id_,
            content=content,
            memory_type=self.memory_type,
            timestamp=time.time(),
            ttl=ttl,
            importance=importance,
            metadata=metadata or {}
        )
        
        self.memories[id_] = memory
        return memory
    
    def get(self, id_: str) -> Optional[Memory]:
        """Get a memory by ID."""
        memory = self.memories.get(id_)
        if memory:
            memory.access()
        return memory
    
    def remove(self, id_: str) -> bool:
        """Remove a memory."""
        if id_ in self.memories:
            del self.memories[id_]
            return True
        return False
    
    def prune_expired(self) -> int:
        """Remove expired memories. Returns count of removed memories."""
        to_remove = [
            id_ for id_, mem in self.memories.items()
            if mem.is_expired()
        ]
        
        for id_ in to_remove:
            del self.memories[id_]
        
        return len(to_remove)
    
    def get_all(self, include_expired: bool = False) -> List[Memory]:
        """Get all memories."""
        if include_expired:
            return list(self.memories.values())
        return [mem for mem in self.memories.values() if not mem.is_expired()]
    
    def count(self, include_expired: bool = False) -> int:
        """Count memories."""
        if include_expired:
            return len(self.memories)
        return sum(1 for mem in self.memories.values() if not mem.is_expired())


class MemoryCategorization:
    """Manages all memory categories with TTL-based pruning."""
    
    # Default TTL values (in seconds)
    DEFAULT_TTLS = {
        MemoryType.WORKING: 3600,           # 1 hour
        MemoryType.SHORT_TERM: 86400,       # 1 day
        MemoryType.EPISODIC: 604800,        # 1 week
        MemoryType.SEMANTIC: None,          # Permanent
        MemoryType.LONG_TERM: None,         # Permanent
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize memory categorization system.
        
        Args:
            config: Configuration dictionary with TTL overrides
        """
        self.config = config or {}
        self.categories: Dict[MemoryType, MemoryCategory] = {}
        
        # Initialize categories
        for mem_type in MemoryType:
            ttl = self.config.get(f'{mem_type.value}_ttl', self.DEFAULT_TTLS.get(mem_type))
            self.categories[mem_type] = MemoryCategory(mem_type, ttl)
        
        self.last_prune_time = time.time()
        self.prune_interval = self.config.get('prune_interval', 3600)  # 1 hour
    
    def add_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        ttl: Optional[float] = None,
        importance: float = 0.5,
        metadata: Optional[Dict] = None,
        id_: Optional[str] = None
    ) -> Memory:
        """
        Add a memory to the appropriate category.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            ttl: Time to live (overrides default)
            importance: Importance score (0.0 to 1.0)
            metadata: Additional metadata
            id_: Optional ID (generated if not provided)
            
        Returns:
            Created Memory object
        """
        if id_ is None:
            id_ = f"{memory_type.value}_{time.time()}_{hash(content) % 100000}"
        
        category = self.categories[memory_type]
        return category.add(id_, content, ttl, importance, metadata)
    
    def get_memory(self, id_: str, memory_type: Optional[MemoryType] = None) -> Optional[Memory]:
        """Get a memory by ID, optionally specifying type."""
        if memory_type:
            return self.categories[memory_type].get(id_)
        
        # Search all categories
        for category in self.categories.values():
            memory = category.get(id_)
            if memory:
                return memory
        
        return None
    
    def remove_memory(self, id_: str, memory_type: Optional[MemoryType] = None) -> bool:
        """Remove a memory."""
        if memory_type:
            return self.categories[memory_type].remove(id_)
        
        # Try all categories
        for category in self.categories.values():
            if category.remove(id_):
                return True
        
        return False
    
    def prune_all(self) -> Dict[MemoryType, int]:
        """Prune expired memories from all categories."""
        results = {}
        for mem_type, category in self.categories.items():
            count = category.prune_expired()
            results[mem_type] = count
        
        self.last_prune_time = time.time()
        return results
    
    def auto_prune_if_needed(self):
        """Automatically prune if interval has passed."""
        if time.time() - self.last_prune_time > self.prune_interval:
            results = self.prune_all()
            total_pruned = sum(results.values())
            if total_pruned > 0:
                logger.info(f"Auto-pruned {total_pruned} expired memories")
    
    def get_memories_by_type(
        self,
        memory_type: MemoryType,
        include_expired: bool = False
    ) -> List[Memory]:
        """Get all memories of a specific type."""
        return self.categories[memory_type].get_all(include_expired)
    
    def get_all_memories(self, include_expired: bool = False) -> List[Memory]:
        """Get all memories across all categories."""
        all_memories = []
        for category in self.categories.values():
            all_memories.extend(category.get_all(include_expired))
        return all_memories
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = {
            'total': 0,
            'by_type': {},
            'last_prune': self.last_prune_time
        }
        
        for mem_type, category in self.categories.items():
            count = category.count(include_expired=False)
            stats['by_type'][mem_type.value] = count
            stats['total'] += count
        
        return stats
    
    def promote_to_long_term(self, id_: str) -> bool:
        """
        Promote a memory to long-term (make permanent).
        
        Args:
            id_: Memory ID
            
        Returns:
            True if successful
        """
        # Find the memory
        memory = self.get_memory(id_)
        if not memory:
            return False
        
        # If already long-term, just remove TTL
        if memory.memory_type == MemoryType.LONG_TERM:
            memory.ttl = None
            return True
        
        # Create copy in long-term category
        long_term_cat = self.categories[MemoryType.LONG_TERM]
        long_term_cat.add(
            id_=f"lt_{id_}",
            content=memory.content,
            ttl=None,
            importance=max(memory.importance, 0.8),  # Boost importance
            metadata=memory.metadata
        )
        
        # Remove from original category
        self.remove_memory(id_, memory.memory_type)
        
        return True
    
    def save(self, path: Path):
        """Save all memories to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'categories': {},
            'config': self.config,
            'last_prune_time': self.last_prune_time
        }
        
        for mem_type, category in self.categories.items():
            data['categories'][mem_type.value] = [
                mem.to_dict() for mem in category.get_all(include_expired=True)
            ]
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: Path):
        """Load memories from disk."""
        if not path.exists():
            logger.warning(f"Memory file not found: {path}")
            return
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.config = data.get('config', {})
        self.last_prune_time = data.get('last_prune_time', time.time())
        
        # Load memories into categories
        for mem_type_str, memories in data.get('categories', {}).items():
            mem_type = MemoryType(mem_type_str)
            category = self.categories[mem_type]
            
            for mem_dict in memories:
                memory = Memory.from_dict(mem_dict)
                category.memories[memory.id] = memory
        
        logger.info(f"Loaded {len(self.get_all_memories())} memories from {path}")
    
    def clear_working_memory(self):
        """Clear working memory (for new session)."""
        self.categories[MemoryType.WORKING].memories.clear()
        logger.info("Working memory cleared")
    
    def clear_all(self):
        """Clear all memories (use with caution!)."""
        for category in self.categories.values():
            category.memories.clear()
        logger.warning("All memories cleared")
