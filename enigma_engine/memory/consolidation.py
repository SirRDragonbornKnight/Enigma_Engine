"""
Memory Consolidation System for Enigma AI Engine
Periodically summarizes and merges old memories to reduce storage and improve retrieval.
"""
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Callable, Optional

from .categorization import Memory, MemoryCategorization, MemoryType

logger = logging.getLogger(__name__)


class MemoryConsolidator:
    """Consolidate and summarize old memories."""
    
    def __init__(
        self,
        memory_system: MemoryCategorization,
        summarizer: Optional[Callable[[list[Memory]], str]] = None
    ):
        """
        Initialize memory consolidator.
        
        Args:
            memory_system: Memory categorization system
            summarizer: Optional function to summarize memories (text input -> summary output)
        """
        self.memory_system = memory_system
        self.summarizer = summarizer or self._default_summarizer
        self._scheduler_thread = None
        self._running = False
    
    def _default_summarizer(self, memories: list[Memory]) -> str:
        """
        Default summarizer: concatenates and truncates.
        
        Args:
            memories: List of memories to summarize
            
        Returns:
            Summary text
        """
        if not memories:
            return ""
        
        # Simple concatenation with timestamps
        parts = []
        for mem in memories:
            timestamp = datetime.fromtimestamp(mem.timestamp).strftime("%Y-%m-%d %H:%M")
            parts.append(f"[{timestamp}] {mem.content[:200]}")
        
        summary = "\n".join(parts)
        
        # Truncate if too long
        max_length = 2000
        if len(summary) > max_length:
            summary = summary[:max_length] + f"\n... (summarized {len(memories)} memories)"
        
        return summary
    
    def consolidate_old_memories(
        self,
        age_threshold_days: int = 7,
        min_count: int = 10,
        memory_type: MemoryType = MemoryType.SHORT_TERM
    ) -> Optional[Memory]:
        """
        Consolidate old memories into a summary.
        
        Args:
            age_threshold_days: Age threshold in days
            min_count: Minimum number of memories to consolidate
            memory_type: Type of memory to consolidate
            
        Returns:
            Consolidated memory if created, None otherwise
        """
        # Get memories older than threshold
        threshold_time = time.time() - (age_threshold_days * 86400)
        
        category = self.memory_system.categories[memory_type]
        old_memories = [
            mem for mem in category.get_all()
            if mem.timestamp < threshold_time
        ]
        
        if len(old_memories) < min_count:
            logger.info(f"Not enough old memories to consolidate ({len(old_memories)} < {min_count})")
            return None
        
        # Sort by timestamp
        old_memories.sort(key=lambda m: m.timestamp)
        
        # Generate summary
        summary_text = self.summarizer(old_memories)
        
        # Calculate average importance
        avg_importance = sum(m.importance for m in old_memories) / len(old_memories)
        
        # Create consolidated memory
        consolidated_id = f"consolidated_{memory_type.value}_{int(time.time())}"
        consolidated = self.memory_system.add_memory(
            content=summary_text,
            memory_type=MemoryType.LONG_TERM,  # Store as long-term
            importance=min(avg_importance + 0.1, 1.0),  # Boost importance slightly
            metadata={
                'consolidated': True,
                'original_count': len(old_memories),
                'original_type': memory_type.value,
                'date_range': {
                    'start': old_memories[0].timestamp,
                    'end': old_memories[-1].timestamp
                }
            },
            id_=consolidated_id
        )
        
        # Remove original memories
        for mem in old_memories:
            category.remove(mem.id)
        
        logger.info(
            f"Consolidated {len(old_memories)} {memory_type.value} memories "
            f"into long-term memory"
        )
        
        return consolidated
    
    def merge_similar_memories(
        self,
        similarity_threshold: float = 0.9,
        memory_type: Optional[MemoryType] = None
    ) -> int:
        """
        Merge memories that are very similar.
        
        Args:
            similarity_threshold: Similarity threshold (0-1)
            memory_type: Type of memory to check (None = all types)
            
        Returns:
            Number of memories merged
        """
        # Get memories to check
        if memory_type:
            memories = self.memory_system.get_memories_by_type(memory_type)
        else:
            memories = self.memory_system.get_all_memories()
        
        if len(memories) < 2:
            return 0
        
        # Find similar pairs (simple text comparison)
        merged_count = 0
        processed = set()
        
        for i, mem1 in enumerate(memories):
            if mem1.id in processed:
                continue
            
            similar_group = [mem1]
            
            for mem2 in memories[i+1:]:
                if mem2.id in processed:
                    continue
                
                # Calculate similarity (simple Jaccard for now)
                similarity = self._calculate_similarity(mem1.content, mem2.content)
                
                if similarity >= similarity_threshold:
                    similar_group.append(mem2)
                    processed.add(mem2.id)
            
            # Merge if we found similar memories
            if len(similar_group) > 1:
                self._merge_memory_group(similar_group)
                merged_count += len(similar_group) - 1
        
        logger.info(f"Merged {merged_count} similar memories")
        return merged_count
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (simple Jaccard)."""
        # Tokenize
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_memory_group(self, memories: list[Memory]) -> Memory:
        """Merge a group of similar memories."""
        if not memories:
            return None
        
        # Keep the most important or most recent
        primary = max(memories, key=lambda m: (m.importance, m.timestamp))
        
        # Combine content
        combined_content = primary.content
        if len(memories) > 1:
            combined_content += f"\n(Merged from {len(memories)} similar memories)"
        
        # Update primary memory
        primary.content = combined_content
        primary.access_count = sum(m.access_count for m in memories)
        primary.importance = min(
            max(m.importance for m in memories) + 0.05,
            1.0
        )
        
        # Remove duplicates
        for mem in memories:
            if mem.id != primary.id:
                self.memory_system.remove_memory(mem.id, mem.memory_type)
        
        return primary
    
    def create_daily_summary(self, date: Optional[datetime] = None) -> Optional[Memory]:
        """
        Create a summary of memories from a specific day.
        
        Args:
            date: Date to summarize (None = yesterday)
            
        Returns:
            Summary memory if created
        """
        if date is None:
            date = datetime.now() - timedelta(days=1)
        
        # Get start and end of day
        start_time = datetime(date.year, date.month, date.day, 0, 0, 0).timestamp()
        end_time = datetime(date.year, date.month, date.day, 23, 59, 59).timestamp()
        
        # Get memories from that day
        all_memories = self.memory_system.get_all_memories()
        day_memories = [
            mem for mem in all_memories
            if start_time <= mem.timestamp <= end_time
        ]
        
        if not day_memories:
            logger.info(f"No memories found for {date.strftime('%Y-%m-%d')}")
            return None
        
        # Sort by timestamp
        day_memories.sort(key=lambda m: m.timestamp)
        
        # Generate summary
        summary_text = self.summarizer(day_memories)
        
        # Create summary memory
        summary_id = f"daily_summary_{date.strftime('%Y%m%d')}"
        summary = self.memory_system.add_memory(
            content=f"Daily Summary for {date.strftime('%Y-%m-%d')}:\n\n{summary_text}",
            memory_type=MemoryType.EPISODIC,
            importance=0.7,
            metadata={
                'summary': True,
                'date': date.strftime('%Y-%m-%d'),
                'memory_count': len(day_memories)
            },
            id_=summary_id
        )
        
        logger.info(f"Created daily summary for {date.strftime('%Y-%m-%d')} ({len(day_memories)} memories)")
        return summary
    
    def schedule_consolidation(
        self,
        interval_hours: int = 24,
        age_threshold_days: int = 7
    ):
        """
        Schedule periodic consolidation.
        
        Args:
            interval_hours: Hours between consolidation runs
            age_threshold_days: Age threshold for consolidation
        """
        if self._running:
            logger.warning("Consolidation already scheduled")
            return
        
        self._running = True
        
        def consolidation_loop():
            while self._running:
                try:
                    # Run consolidation
                    logger.info("Running scheduled memory consolidation")
                    
                    # Consolidate short-term memories
                    self.consolidate_old_memories(
                        age_threshold_days=age_threshold_days,
                        memory_type=MemoryType.SHORT_TERM
                    )
                    
                    # Create daily summary for yesterday
                    self.create_daily_summary()
                    
                    # Merge similar memories
                    self.merge_similar_memories(similarity_threshold=0.95)
                    
                except Exception as e:
                    logger.error(f"Error in consolidation loop: {e}", exc_info=True)
                
                # Sleep until next run
                time.sleep(interval_hours * 3600)
        
        self._scheduler_thread = threading.Thread(
            target=consolidation_loop,
            daemon=True,
            name="MemoryConsolidation"
        )
        self._scheduler_thread.start()
        
        logger.info(f"Scheduled memory consolidation every {interval_hours} hours")
    
    def stop_scheduling(self):
        """Stop scheduled consolidation."""
        if self._running:
            self._running = False
            logger.info("Stopped memory consolidation scheduling")
