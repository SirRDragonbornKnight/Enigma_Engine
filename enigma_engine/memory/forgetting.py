"""
Memory Forgetting Mechanism for Enigma AI Engine

Implements intelligent memory pruning and forgetting.

Features:
- Time-based decay
- Access-frequency decay
- Importance-based retention
- Consolidation (merge similar memories)
- Selective forgetting

Usage:
    from enigma_engine.memory.forgetting import ForgettingManager, get_forgetting_manager
    
    manager = get_forgetting_manager()
    
    # Add memories
    manager.add_memory("key1", "value1", importance=0.8)
    
    # Run forgetting cycle
    forgotten = manager.forget_cycle()
    
    # Get remaining memories
    memories = manager.get_all()
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class DecayStrategy(Enum):
    """Strategies for memory decay."""
    TIME_BASED = "time_based"  # Forget based on age
    ACCESS_BASED = "access_based"  # Forget rarely accessed
    IMPORTANCE = "importance"  # Forget low importance
    HYBRID = "hybrid"  # Combine strategies
    NONE = "none"  # No decay


class ConsolidationStrategy(Enum):
    """Strategies for memory consolidation."""
    MERGE_SIMILAR = "merge_similar"  # Merge similar memories
    SUMMARIZE = "summarize"  # Create summary memories
    HIERARCHICAL = "hierarchical"  # Create memory hierarchy
    NONE = "none"


@dataclass
class ForgettingConfig:
    """Configuration for forgetting mechanism."""
    # Decay
    decay_strategy: DecayStrategy = DecayStrategy.HYBRID
    time_decay_rate: float = 0.1  # Per hour
    access_decay_rate: float = 0.05  # Per hour without access
    
    # Thresholds
    forget_threshold: float = 0.2  # Below this, forget
    max_memories: int = 10000  # Hard limit
    min_memories: int = 100  # Keep at least this many
    
    # Importance
    base_importance: float = 0.5
    importance_boost_on_access: float = 0.1
    
    # Consolidation
    consolidation_strategy: ConsolidationStrategy = ConsolidationStrategy.MERGE_SIMILAR
    similarity_threshold: float = 0.85
    
    # Timing
    forget_interval_hours: float = 1.0  # Run forget cycle


@dataclass
class MemoryItem:
    """A memory item with metadata."""
    key: str
    value: Any
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    
    # Scores
    importance: float = 0.5
    retention_score: float = 1.0
    
    # Flags
    pinned: bool = False  # Never forget
    consolidated: bool = False  # Result of consolidation
    source_keys: List[str] = field(default_factory=list)  # For consolidated memories


class DecayCalculator:
    """Calculate memory decay."""
    
    def __init__(self, config: ForgettingConfig):
        self._config = config
    
    def calculate_time_decay(
        self,
        memory: MemoryItem,
        current_time: Optional[float] = None
    ) -> float:
        """
        Calculate time-based decay.
        
        Returns:
            Decay amount (subtract from retention)
        """
        current_time = current_time or time.time()
        hours_since_creation = (current_time - memory.created_at) / 3600
        
        # Exponential decay
        decay = self._config.time_decay_rate * hours_since_creation
        return min(decay, 1.0)
    
    def calculate_access_decay(
        self,
        memory: MemoryItem,
        current_time: Optional[float] = None
    ) -> float:
        """
        Calculate access-based decay.
        
        Returns:
            Decay amount
        """
        current_time = current_time or time.time()
        hours_since_access = (current_time - memory.last_accessed) / 3600
        
        # More decay if not accessed
        decay = self._config.access_decay_rate * hours_since_access
        
        # Mitigate with access count
        access_factor = 1.0 / (1 + memory.access_count * 0.1)
        
        return min(decay * access_factor, 1.0)
    
    def calculate_retention(
        self,
        memory: MemoryItem,
        current_time: Optional[float] = None
    ) -> float:
        """
        Calculate overall retention score.
        
        Returns:
            Retention score (0-1, higher = keep)
        """
        if memory.pinned:
            return 1.0
        
        strategy = self._config.decay_strategy
        
        if strategy == DecayStrategy.TIME_BASED:
            decay = self.calculate_time_decay(memory, current_time)
        elif strategy == DecayStrategy.ACCESS_BASED:
            decay = self.calculate_access_decay(memory, current_time)
        elif strategy == DecayStrategy.IMPORTANCE:
            decay = (1.0 - memory.importance) * 0.5
        elif strategy == DecayStrategy.HYBRID:
            time_decay = self.calculate_time_decay(memory, current_time)
            access_decay = self.calculate_access_decay(memory, current_time)
            importance_factor = memory.importance
            
            # Weighted combination
            decay = (0.3 * time_decay + 0.4 * access_decay) * (1 - importance_factor * 0.5)
        else:
            decay = 0.0
        
        return max(0.0, 1.0 - decay)


class MemoryConsolidator:
    """Consolidate similar memories."""
    
    def __init__(
        self,
        config: ForgettingConfig,
        similarity_fn: Optional[Callable[[Any, Any], float]] = None
    ):
        self._config = config
        self._similarity_fn = similarity_fn or self._default_similarity
    
    def _default_similarity(self, value1: Any, value2: Any) -> float:
        """Default similarity using string comparison."""
        str1 = str(value1).lower()
        str2 = str(value2).lower()
        
        if str1 == str2:
            return 1.0
        
        # Simple word overlap
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def find_similar(
        self,
        memories: List[MemoryItem],
        threshold: Optional[float] = None
    ) -> List[List[MemoryItem]]:
        """
        Find groups of similar memories.
        
        Returns:
            List of similar memory groups
        """
        threshold = threshold or self._config.similarity_threshold
        groups: List[List[MemoryItem]] = []
        used: Set[str] = set()
        
        for i, mem1 in enumerate(memories):
            if mem1.key in used:
                continue
            
            group = [mem1]
            used.add(mem1.key)
            
            for mem2 in memories[i+1:]:
                if mem2.key in used:
                    continue
                
                sim = self._similarity_fn(mem1.value, mem2.value)
                if sim >= threshold:
                    group.append(mem2)
                    used.add(mem2.key)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def consolidate_group(
        self,
        group: List[MemoryItem],
        merge_fn: Optional[Callable[[List[Any]], Any]] = None
    ) -> MemoryItem:
        """
        Consolidate a group of similar memories.
        
        Returns:
            Consolidated memory item
        """
        if not group:
            raise ValueError("Empty group")
        
        if len(group) == 1:
            return group[0]
        
        # Merge values
        if merge_fn:
            merged_value = merge_fn([m.value for m in group])
        else:
            # Default: keep the most accessed
            best = max(group, key=lambda m: m.access_count)
            merged_value = best.value
        
        # Combine metadata
        total_access = sum(m.access_count for m in group)
        max_importance = max(m.importance for m in group)
        earliest_created = min(m.created_at for m in group)
        latest_accessed = max(m.last_accessed for m in group)
        
        # Create consolidated memory
        consolidated = MemoryItem(
            key=f"consolidated_{group[0].key}",
            value=merged_value,
            created_at=earliest_created,
            last_accessed=latest_accessed,
            access_count=total_access,
            importance=max_importance,
            consolidated=True,
            source_keys=[m.key for m in group]
        )
        
        return consolidated


class ForgettingManager:
    """Manages memory forgetting and consolidation."""
    
    def __init__(
        self,
        config: Optional[ForgettingConfig] = None,
        similarity_fn: Optional[Callable[[Any, Any], float]] = None
    ):
        """
        Initialize forgetting manager.
        
        Args:
            config: Forgetting configuration
            similarity_fn: Custom similarity function
        """
        self._config = config or ForgettingConfig()
        self._memories: Dict[str, MemoryItem] = {}
        
        self._decay = DecayCalculator(self._config)
        self._consolidator = MemoryConsolidator(self._config, similarity_fn)
        
        self._last_forget_time = 0.0
        self._forgetting_callbacks: List[Callable[[MemoryItem], None]] = []
    
    def add_memory(
        self,
        key: str,
        value: Any,
        importance: Optional[float] = None,
        pinned: bool = False
    ):
        """
        Add a memory.
        
        Args:
            key: Memory key
            value: Memory value
            importance: Initial importance (0-1)
            pinned: If True, never forget
        """
        importance = importance or self._config.base_importance
        
        self._memories[key] = MemoryItem(
            key=key,
            value=value,
            importance=importance,
            pinned=pinned
        )
        
        # Check if we need to forget
        if len(self._memories) > self._config.max_memories:
            self.forget_cycle()
    
    def get_memory(self, key: str) -> Optional[Any]:
        """
        Get a memory (updates access time).
        
        Returns:
            Memory value or None
        """
        if key not in self._memories:
            return None
        
        memory = self._memories[key]
        memory.last_accessed = time.time()
        memory.access_count += 1
        memory.importance = min(
            1.0,
            memory.importance + self._config.importance_boost_on_access
        )
        
        return memory.value
    
    def get_all(self) -> Dict[str, Any]:
        """Get all memories."""
        return {k: m.value for k, m in self._memories.items()}
    
    def pin_memory(self, key: str):
        """Pin a memory (never forget)."""
        if key in self._memories:
            self._memories[key].pinned = True
    
    def unpin_memory(self, key: str):
        """Unpin a memory."""
        if key in self._memories:
            self._memories[key].pinned = False
    
    def set_importance(self, key: str, importance: float):
        """Set memory importance."""
        if key in self._memories:
            self._memories[key].importance = max(0.0, min(1.0, importance))
    
    def update_retention_scores(self):
        """Update retention scores for all memories."""
        current_time = time.time()
        
        for memory in self._memories.values():
            memory.retention_score = self._decay.calculate_retention(
                memory, current_time
            )
    
    def forget_cycle(self) -> List[str]:
        """
        Run a forgetting cycle.
        
        Returns:
            List of forgotten memory keys
        """
        self.update_retention_scores()
        
        forgotten: List[str] = []
        
        # Get memories below threshold
        to_forget = [
            key for key, mem in self._memories.items()
            if mem.retention_score < self._config.forget_threshold
            and not mem.pinned
        ]
        
        # Keep minimum memories
        keep_count = len(self._memories) - len(to_forget)
        if keep_count < self._config.min_memories:
            # Sort by retention and keep the best
            to_forget = sorted(
                to_forget,
                key=lambda k: self._memories[k].retention_score
            )
            to_forget = to_forget[:len(self._memories) - self._config.min_memories]
        
        # Forget
        for key in to_forget:
            memory = self._memories[key]
            
            # Call callbacks
            for callback in self._forgetting_callbacks:
                try:
                    callback(memory)
                except Exception as e:
                    logger.error(f"Forgetting callback failed: {e}")
            
            del self._memories[key]
            forgotten.append(key)
        
        self._last_forget_time = time.time()
        
        if forgotten:
            logger.info(f"Forgot {len(forgotten)} memories")
        
        return forgotten
    
    def consolidate(self) -> List[MemoryItem]:
        """
        Consolidate similar memories.
        
        Returns:
            List of new consolidated memories
        """
        if self._config.consolidation_strategy == ConsolidationStrategy.NONE:
            return []
        
        memories = list(self._memories.values())
        groups = self._consolidator.find_similar(memories)
        
        new_consolidated: List[MemoryItem] = []
        
        for group in groups:
            consolidated = self._consolidator.consolidate_group(group)
            
            # Remove original memories
            for mem in group:
                if mem.key in self._memories:
                    del self._memories[mem.key]
            
            # Add consolidated memory
            self._memories[consolidated.key] = consolidated
            new_consolidated.append(consolidated)
        
        if new_consolidated:
            logger.info(f"Consolidated {len(new_consolidated)} memory groups")
        
        return new_consolidated
    
    def on_forget(self, callback: Callable[[MemoryItem], None]):
        """Register callback for when memories are forgotten."""
        self._forgetting_callbacks.append(callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if not self._memories:
            return {
                "total_memories": 0,
                "pinned_count": 0,
                "avg_retention": 0.0,
                "avg_importance": 0.0
            }
        
        return {
            "total_memories": len(self._memories),
            "pinned_count": sum(1 for m in self._memories.values() if m.pinned),
            "consolidated_count": sum(1 for m in self._memories.values() if m.consolidated),
            "avg_retention": sum(m.retention_score for m in self._memories.values()) / len(self._memories),
            "avg_importance": sum(m.importance for m in self._memories.values()) / len(self._memories),
            "avg_access_count": sum(m.access_count for m in self._memories.values()) / len(self._memories)
        }
    
    def export_memories(self) -> List[Dict[str, Any]]:
        """Export all memories with metadata."""
        return [
            {
                "key": m.key,
                "value": m.value,
                "created_at": m.created_at,
                "last_accessed": m.last_accessed,
                "access_count": m.access_count,
                "importance": m.importance,
                "retention_score": m.retention_score,
                "pinned": m.pinned,
                "consolidated": m.consolidated
            }
            for m in self._memories.values()
        ]
    
    def import_memories(self, data: List[Dict[str, Any]]):
        """Import memories from export."""
        for item in data:
            memory = MemoryItem(
                key=item["key"],
                value=item["value"],
                created_at=item.get("created_at", time.time()),
                last_accessed=item.get("last_accessed", time.time()),
                access_count=item.get("access_count", 0),
                importance=item.get("importance", self._config.base_importance),
                retention_score=item.get("retention_score", 1.0),
                pinned=item.get("pinned", False),
                consolidated=item.get("consolidated", False)
            )
            self._memories[memory.key] = memory


# Global instance
_manager: Optional[ForgettingManager] = None


def get_forgetting_manager(
    config: Optional[ForgettingConfig] = None
) -> ForgettingManager:
    """Get or create global forgetting manager."""
    global _manager
    if _manager is None:
        _manager = ForgettingManager(config)
    return _manager
