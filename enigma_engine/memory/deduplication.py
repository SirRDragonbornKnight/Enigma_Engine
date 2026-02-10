"""
Memory Deduplication for Enigma AI Engine
Detects and removes duplicate or near-duplicate memories.
"""
import hashlib
import logging
from typing import Optional

from .categorization import Memory, MemoryCategorization

logger = logging.getLogger(__name__)


class MemoryDeduplicator:
    """Detect and handle duplicate memories."""
    
    def __init__(self, memory_system: MemoryCategorization):
        """
        Initialize memory deduplicator.
        
        Args:
            memory_system: Memory categorization system
        """
        self.memory_system = memory_system
        self._hash_cache: dict[str, str] = {}  # hash -> memory_id
        self._build_hash_cache()
    
    def _build_hash_cache(self):
        """Build hash cache from existing memories."""
        memories = self.memory_system.get_all_memories()
        
        for memory in memories:
            content_hash = self.compute_hash(memory.content)
            if content_hash not in self._hash_cache:
                self._hash_cache[content_hash] = memory.id
    
    def compute_hash(self, content: str) -> str:
        """
        Compute content hash for deduplication.
        
        Args:
            content: Content to hash
            
        Returns:
            Hash string
        """
        # Normalize content
        normalized = content.strip().lower()
        
        # Use SHA-256
        hash_obj = hashlib.sha256(normalized.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def is_duplicate(self, content: str) -> Optional[str]:
        """
        Check if content is a duplicate.
        
        Args:
            content: Content to check
            
        Returns:
            Existing memory ID if duplicate, None otherwise
        """
        content_hash = self.compute_hash(content)
        return self._hash_cache.get(content_hash)
    
    def find_duplicates(self) -> list[tuple[str, str]]:
        """
        Find all duplicate pairs in the system.
        
        Returns:
            List of (id1, id2) tuples representing duplicates
        """
        memories = self.memory_system.get_all_memories()
        
        # Group by hash
        hash_groups: dict[str, list[Memory]] = {}
        
        for memory in memories:
            content_hash = self.compute_hash(memory.content)
            if content_hash not in hash_groups:
                hash_groups[content_hash] = []
            hash_groups[content_hash].append(memory)
        
        # Find groups with multiple memories
        duplicates = []
        for content_hash, group in hash_groups.items():
            if len(group) > 1:
                # Create pairs
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        duplicates.append((group[i].id, group[j].id))
        
        logger.info(f"Found {len(duplicates)} duplicate pairs")
        return duplicates
    
    def remove_duplicates(self, keep: str = "first") -> int:
        """
        Remove duplicate memories.
        
        Args:
            keep: Which duplicate to keep:
                  'first' - keep oldest
                  'last' - keep newest
                  'most_accessed' - keep most accessed
                  'highest_importance' - keep highest importance
                  
        Returns:
            Number of memories removed
        """
        memories = self.memory_system.get_all_memories()
        
        # Group by hash
        hash_groups: dict[str, list[Memory]] = {}
        
        for memory in memories:
            content_hash = self.compute_hash(memory.content)
            if content_hash not in hash_groups:
                hash_groups[content_hash] = []
            hash_groups[content_hash].append(memory)
        
        removed_count = 0
        
        for content_hash, group in hash_groups.items():
            if len(group) <= 1:
                continue
            
            # Determine which to keep
            if keep == "first":
                to_keep = min(group, key=lambda m: m.timestamp)
            elif keep == "last":
                to_keep = max(group, key=lambda m: m.timestamp)
            elif keep == "most_accessed":
                to_keep = max(group, key=lambda m: m.access_count)
            elif keep == "highest_importance":
                to_keep = max(group, key=lambda m: m.importance)
            else:
                logger.warning(f"Unknown keep strategy: {keep}, using 'first'")
                to_keep = min(group, key=lambda m: m.timestamp)
            
            # Remove others
            for memory in group:
                if memory.id != to_keep.id:
                    self.memory_system.remove_memory(memory.id, memory.memory_type)
                    removed_count += 1
            
            # Update hash cache
            self._hash_cache[content_hash] = to_keep.id
        
        logger.info(f"Removed {removed_count} duplicate memories")
        return removed_count
    
    def find_near_duplicates(
        self,
        similarity_threshold: float = 0.95
    ) -> list[tuple[str, str, float]]:
        """
        Find near-duplicate memories using similarity.
        
        Args:
            similarity_threshold: Similarity threshold (0-1)
            
        Returns:
            List of (id1, id2, similarity) tuples
        """
        memories = self.memory_system.get_all_memories()
        
        near_duplicates = []
        
        # Compare all pairs (O(n²) - could be optimized)
        for i, mem1 in enumerate(memories):
            for mem2 in memories[i+1:]:
                similarity = self._calculate_similarity(mem1.content, mem2.content)
                
                if similarity >= similarity_threshold:
                    near_duplicates.append((mem1.id, mem2.id, similarity))
        
        logger.info(f"Found {len(near_duplicates)} near-duplicate pairs")
        return near_duplicates
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using multiple methods.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # Normalize
        text1 = text1.strip().lower()
        text2 = text2.strip().lower()
        
        # Exact match
        if text1 == text2:
            return 1.0
        
        # Jaccard similarity (word-level)
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        jaccard = intersection / union if union > 0 else 0.0
        
        # Character-level similarity (Levenshtein-like)
        # Simple approximation: length ratio
        len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
        
        # Combine scores
        similarity = (jaccard * 0.7) + (len_ratio * 0.3)
        
        return similarity
    
    def remove_near_duplicates(
        self,
        similarity_threshold: float = 0.95,
        keep: str = "first"
    ) -> int:
        """
        Remove near-duplicate memories.
        
        Args:
            similarity_threshold: Similarity threshold
            keep: Which duplicate to keep
            
        Returns:
            Number of memories removed
        """
        near_duplicates = self.find_near_duplicates(similarity_threshold)
        
        if not near_duplicates:
            return 0
        
        # Build groups of similar memories
        groups: dict[str, set] = {}  # representative_id -> set of similar ids
        
        for id1, id2, similarity in near_duplicates:
            # Find if either is already in a group
            found_group = None
            for rep_id, group in groups.items():
                if id1 in group or id2 in group:
                    found_group = rep_id
                    break
            
            if found_group:
                groups[found_group].add(id1)
                groups[found_group].add(id2)
            else:
                # Create new group
                groups[id1] = {id1, id2}
        
        removed_count = 0
        
        # Remove from each group
        for rep_id, group_ids in groups.items():
            group_memories = []
            for mem_id in group_ids:
                memory = self.memory_system.get_memory(mem_id)
                if memory:
                    group_memories.append(memory)
            
            if len(group_memories) <= 1:
                continue
            
            # Determine which to keep
            if keep == "first":
                to_keep = min(group_memories, key=lambda m: m.timestamp)
            elif keep == "last":
                to_keep = max(group_memories, key=lambda m: m.timestamp)
            elif keep == "most_accessed":
                to_keep = max(group_memories, key=lambda m: m.access_count)
            elif keep == "highest_importance":
                to_keep = max(group_memories, key=lambda m: m.importance)
            else:
                to_keep = group_memories[0]
            
            # Remove others
            for memory in group_memories:
                if memory.id != to_keep.id:
                    self.memory_system.remove_memory(memory.id, memory.memory_type)
                    removed_count += 1
        
        logger.info(f"Removed {removed_count} near-duplicate memories")
        return removed_count
    
    def get_duplicate_statistics(self) -> dict[str, any]:
        """
        Get statistics about duplicates.
        
        Returns:
            Dictionary with duplicate statistics
        """
        exact_duplicates = self.find_duplicates()
        near_duplicates_95 = self.find_near_duplicates(0.95)
        near_duplicates_90 = self.find_near_duplicates(0.90)
        near_duplicates_85 = self.find_near_duplicates(0.85)
        
        total_memories = len(self.memory_system.get_all_memories())
        
        return {
            'total_memories': total_memories,
            'exact_duplicate_pairs': len(exact_duplicates),
            'near_duplicates_95': len(near_duplicates_95),
            'near_duplicates_90': len(near_duplicates_90),
            'near_duplicates_85': len(near_duplicates_85),
            'cache_size': len(self._hash_cache)
        }
