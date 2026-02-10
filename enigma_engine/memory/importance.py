"""
Memory Importance Scoring for Enigma AI Engine

Score and prioritize memories for retention.

Features:
- Importance scoring
- Decay over time
- Relevance scoring
- Pruning strategies
- Priority queues

Usage:
    from enigma_engine.memory.importance import MemoryScorer, ImportanceManager
    
    scorer = MemoryScorer()
    
    # Score a memory
    score = scorer.score(memory_text, context="user query")
    
    # Manage memories with importance
    manager = ImportanceManager()
    manager.add_memory("Hello", importance=0.8)
    
    # Get most important memories
    top = manager.get_top_memories(k=10)
"""

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ImportanceMetric(Enum):
    """Importance metrics to consider."""
    RECENCY = "recency"  # How recent
    FREQUENCY = "frequency"  # How often accessed
    RELEVANCE = "relevance"  # Relevance to context
    EMOTIONAL = "emotional"  # Emotional weight
    FACTUAL = "factual"  # Factual importance
    USER_MARKED = "user_marked"  # User-marked as important


@dataclass
class MemoryEntry:
    """A memory with importance metadata."""
    id: str
    content: str
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    
    # Importance scores
    base_importance: float = 0.5  # Initial importance
    computed_importance: float = 0.5  # Current computed score
    
    # Metrics
    recency_score: float = 1.0
    frequency_score: float = 0.0
    relevance_score: float = 0.0
    emotional_score: float = 0.0
    user_score: float = 0.0  # 0 = not marked, 1 = marked important
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    source: str = ""  # conversation, document, etc.
    embedding: Optional[List[float]] = None


@dataclass
class ScoringConfig:
    """Configuration for importance scoring."""
    # Weight for each metric
    weights: Dict[ImportanceMetric, float] = field(default_factory=lambda: {
        ImportanceMetric.RECENCY: 0.2,
        ImportanceMetric.FREQUENCY: 0.15,
        ImportanceMetric.RELEVANCE: 0.3,
        ImportanceMetric.EMOTIONAL: 0.15,
        ImportanceMetric.FACTUAL: 0.1,
        ImportanceMetric.USER_MARKED: 0.1,
    })
    
    # Decay settings
    recency_half_life: float = 7 * 24 * 3600  # 7 days in seconds
    min_importance: float = 0.01  # Below this, memory can be pruned
    
    # Thresholds
    high_importance_threshold: float = 0.7
    low_importance_threshold: float = 0.3


class MemoryScorer:
    """Score memory importance."""
    
    # Keywords indicating emotional content
    EMOTIONAL_KEYWORDS = {
        "happy", "sad", "angry", "excited", "worried", "love", "hate",
        "fear", "joy", "anxious", "grateful", "thank", "sorry", "please",
        "amazing", "terrible", "wonderful", "awful", "great", "horrible"
    }
    
    # Keywords indicating factual importance
    FACTUAL_KEYWORDS = {
        "name", "date", "birthday", "address", "password", "key", "number",
        "phone", "email", "remember", "important", "deadline", "meeting",
        "appointment", "schedule", "always", "never", "rule", "preference"
    }
    
    def __init__(self, config: Optional[ScoringConfig] = None):
        """
        Initialize scorer.
        
        Args:
            config: Scoring configuration
        """
        self.config = config or ScoringConfig()
    
    def score(
        self,
        memory: MemoryEntry,
        context: str = "",
        current_time: Optional[float] = None
    ) -> float:
        """
        Score memory importance.
        
        Args:
            memory: Memory to score
            context: Current context for relevance
            current_time: Current timestamp
            
        Returns:
            Importance score 0-1
        """
        if current_time is None:
            current_time = time.time()
        
        # Calculate individual scores
        memory.recency_score = self._score_recency(memory, current_time)
        memory.frequency_score = self._score_frequency(memory)
        memory.relevance_score = self._score_relevance(memory, context)
        memory.emotional_score = self._score_emotional(memory)
        factual_score = self._score_factual(memory)
        
        # Weighted combination
        weights = self.config.weights
        total_weight = sum(weights.values())
        
        score = (
            weights[ImportanceMetric.RECENCY] * memory.recency_score +
            weights[ImportanceMetric.FREQUENCY] * memory.frequency_score +
            weights[ImportanceMetric.RELEVANCE] * memory.relevance_score +
            weights[ImportanceMetric.EMOTIONAL] * memory.emotional_score +
            weights[ImportanceMetric.FACTUAL] * factual_score +
            weights[ImportanceMetric.USER_MARKED] * memory.user_score
        ) / total_weight
        
        # Combine with base importance
        score = 0.5 * memory.base_importance + 0.5 * score
        
        memory.computed_importance = score
        return score
    
    def _score_recency(self, memory: MemoryEntry, current_time: float) -> float:
        """Score based on how recent the memory is."""
        age = current_time - memory.last_accessed
        half_life = self.config.recency_half_life
        
        # Exponential decay
        score = math.pow(0.5, age / half_life)
        return max(0.0, min(1.0, score))
    
    def _score_frequency(self, memory: MemoryEntry) -> float:
        """Score based on access frequency."""
        # Log scale for access count
        if memory.access_count == 0:
            return 0.0
        
        score = math.log10(memory.access_count + 1) / 3.0  # Normalize
        return max(0.0, min(1.0, score))
    
    def _score_relevance(self, memory: MemoryEntry, context: str) -> float:
        """Score based on relevance to context."""
        if not context:
            return 0.5
        
        # Simple word overlap
        memory_words = set(memory.content.lower().split())
        context_words = set(context.lower().split())
        
        overlap = len(memory_words & context_words)
        total = len(memory_words | context_words)
        
        if total == 0:
            return 0.0
        
        return overlap / total
    
    def _score_emotional(self, memory: MemoryEntry) -> float:
        """Score based on emotional content."""
        words = memory.content.lower().split()
        emotional_count = sum(1 for w in words if w in self.EMOTIONAL_KEYWORDS)
        
        if len(words) == 0:
            return 0.0
        
        score = emotional_count / len(words) * 5  # Scale up
        return max(0.0, min(1.0, score))
    
    def _score_factual(self, memory: MemoryEntry) -> float:
        """Score based on factual importance."""
        words = memory.content.lower().split()
        factual_count = sum(1 for w in words if w in self.FACTUAL_KEYWORDS)
        
        if len(words) == 0:
            return 0.0
        
        score = factual_count / len(words) * 5  # Scale up
        return max(0.0, min(1.0, score))


class ImportanceManager:
    """Manage memories with importance scoring."""
    
    def __init__(
        self,
        scorer: Optional[MemoryScorer] = None,
        max_memories: int = 10000,
        storage_path: Optional[str] = None
    ):
        """
        Initialize manager.
        
        Args:
            scorer: Memory scorer
            max_memories: Maximum memories to keep
            storage_path: Path to persist memories
        """
        self.scorer = scorer or MemoryScorer()
        self.max_memories = max_memories
        self.storage_path = Path(storage_path) if storage_path else None
        
        # Memory storage
        self._memories: Dict[str, MemoryEntry] = {}
        
        # Load from storage
        if self.storage_path and self.storage_path.exists():
            self._load()
        
        logger.info(f"ImportanceManager initialized with {len(self._memories)} memories")
    
    def add_memory(
        self,
        content: str,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        source: str = "conversation",
        embedding: Optional[List[float]] = None
    ) -> MemoryEntry:
        """
        Add a new memory.
        
        Args:
            content: Memory content
            importance: Initial importance
            tags: Memory tags
            source: Memory source
            embedding: Vector embedding
            
        Returns:
            Created memory entry
        """
        # Generate ID
        memory_id = hashlib.sha256(
            f"{content}{time.time()}".encode()
        ).hexdigest()[:16]
        
        entry = MemoryEntry(
            id=memory_id,
            content=content,
            base_importance=importance,
            tags=tags or [],
            source=source,
            embedding=embedding
        )
        
        # Score the memory
        self.scorer.score(entry)
        
        self._memories[memory_id] = entry
        
        # Prune if over limit
        if len(self._memories) > self.max_memories:
            self._prune()
        
        self._save()
        
        return entry
    
    def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get memory by ID."""
        memory = self._memories.get(memory_id)
        
        if memory:
            # Update access stats
            memory.last_accessed = time.time()
            memory.access_count += 1
            self._save()
        
        return memory
    
    def get_top_memories(
        self,
        k: int = 10,
        context: str = "",
        min_importance: float = 0.0,
        tags: Optional[List[str]] = None
    ) -> List[MemoryEntry]:
        """
        Get top memories by importance.
        
        Args:
            k: Number of memories to return
            context: Context for relevance scoring
            min_importance: Minimum importance threshold
            tags: Filter by tags
            
        Returns:
            List of top memories
        """
        # Re-score all memories with context
        for memory in self._memories.values():
            self.scorer.score(memory, context)
        
        # Filter
        memories = list(self._memories.values())
        
        if min_importance > 0:
            memories = [m for m in memories if m.computed_importance >= min_importance]
        
        if tags:
            memories = [m for m in memories if any(t in m.tags for t in tags)]
        
        # Sort by importance
        memories.sort(key=lambda m: m.computed_importance, reverse=True)
        
        return memories[:k]
    
    def search(
        self,
        query: str,
        k: int = 10,
        use_embedding: bool = True
    ) -> List[MemoryEntry]:
        """
        Search memories.
        
        Args:
            query: Search query
            k: Number of results
            use_embedding: Use embedding similarity if available
            
        Returns:
            List of matching memories
        """
        # Simple keyword search
        query_words = set(query.lower().split())
        
        results = []
        for memory in self._memories.values():
            memory_words = set(memory.content.lower().split())
            overlap = len(query_words & memory_words)
            
            if overlap > 0:
                score = overlap / len(query_words)
                results.append((memory, score))
        
        # Sort by match score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return [m for m, _ in results[:k]]
    
    def mark_important(self, memory_id: str, important: bool = True):
        """Mark memory as important by user."""
        memory = self._memories.get(memory_id)
        if memory:
            memory.user_score = 1.0 if important else 0.0
            self.scorer.score(memory)
            self._save()
    
    def forget(self, memory_id: str):
        """Remove a memory."""
        if memory_id in self._memories:
            del self._memories[memory_id]
            self._save()
    
    def decay_all(self):
        """Apply time decay to all memories."""
        current_time = time.time()
        
        for memory in self._memories.values():
            self.scorer.score(memory, current_time=current_time)
        
        self._save()
    
    def _prune(self, keep_ratio: float = 0.8):
        """Prune low-importance memories."""
        # Re-score all
        for memory in self._memories.values():
            self.scorer.score(memory)
        
        # Sort by importance
        sorted_memories = sorted(
            self._memories.values(),
            key=lambda m: m.computed_importance,
            reverse=True
        )
        
        # Keep top memories
        keep_count = int(self.max_memories * keep_ratio)
        kept_ids = {m.id for m in sorted_memories[:keep_count]}
        
        # Remove pruned
        pruned = [
            mid for mid in self._memories
            if mid not in kept_ids
        ]
        
        for mid in pruned:
            del self._memories[mid]
        
        logger.info(f"Pruned {len(pruned)} memories")
    
    def _save(self):
        """Save to storage."""
        if not self.storage_path:
            return
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = []
        for memory in self._memories.values():
            data.append({
                "id": memory.id,
                "content": memory.content,
                "created_at": memory.created_at,
                "last_accessed": memory.last_accessed,
                "access_count": memory.access_count,
                "base_importance": memory.base_importance,
                "computed_importance": memory.computed_importance,
                "tags": memory.tags,
                "source": memory.source,
                "user_score": memory.user_score
            })
        
        self.storage_path.write_text(json.dumps(data, indent=2))
    
    def _load(self):
        """Load from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        try:
            data = json.loads(self.storage_path.read_text())
            
            for entry_data in data:
                memory = MemoryEntry(
                    id=entry_data["id"],
                    content=entry_data["content"],
                    created_at=entry_data.get("created_at", time.time()),
                    last_accessed=entry_data.get("last_accessed", time.time()),
                    access_count=entry_data.get("access_count", 0),
                    base_importance=entry_data.get("base_importance", 0.5),
                    computed_importance=entry_data.get("computed_importance", 0.5),
                    tags=entry_data.get("tags", []),
                    source=entry_data.get("source", ""),
                    user_score=entry_data.get("user_score", 0.0)
                )
                self._memories[memory.id] = memory
                
        except Exception as e:
            logger.error(f"Failed to load memories: {e}")
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        if not self._memories:
            return {"count": 0}
        
        importances = [m.computed_importance for m in self._memories.values()]
        
        return {
            "count": len(self._memories),
            "avg_importance": sum(importances) / len(importances),
            "high_importance_count": sum(
                1 for i in importances
                if i >= self.scorer.config.high_importance_threshold
            ),
            "low_importance_count": sum(
                1 for i in importances
                if i <= self.scorer.config.low_importance_threshold
            )
        }


# Global instance
_importance_manager: Optional[ImportanceManager] = None


def get_importance_manager() -> ImportanceManager:
    """Get or create global ImportanceManager."""
    global _importance_manager
    if _importance_manager is None:
        _importance_manager = ImportanceManager(
            storage_path="memory/importance.json"
        )
    return _importance_manager
