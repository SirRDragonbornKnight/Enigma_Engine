"""
Working Memory System

Short-term scratchpad for AI reasoning during response generation.
Allows the model to store intermediate thoughts and recall them.

FILE: enigma_engine/memory/working_memory.py
TYPE: Memory System
MAIN CLASSES: WorkingMemory, MemorySlot, WorkingMemoryManager
"""

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SlotType(Enum):
    """Types of working memory slots."""
    SCRATCH = "scratch"          # General scratchpad
    REASONING = "reasoning"      # Reasoning steps
    CONTEXT = "context"          # Retrieved context
    VARIABLE = "variable"        # Named variables
    INTERMEDIATE = "intermediate"  # Intermediate computations
    GOAL = "goal"                # Current goals
    CONSTRAINT = "constraint"    # Active constraints


@dataclass
class MemorySlot:
    """A slot in working memory."""
    id: str
    slot_type: SlotType
    content: Any
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 1
    priority: float = 1.0  # Higher = more important
    ttl: Optional[float] = None  # Time to live in seconds
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def touch(self):
        """Update access time and count."""
        self.accessed_at = time.time()
        self.access_count += 1
        
    def is_expired(self) -> bool:
        """Check if slot has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.slot_type.value,
            "content": self.content,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
            "priority": self.priority,
            "tags": self.tags
        }


class WorkingMemory:
    """Short-term memory for single reasoning session."""
    
    def __init__(self, 
                 max_slots: int = 20,
                 max_content_size: int = 4096):
        """
        Initialize working memory.
        
        Args:
            max_slots: Maximum number of slots
            max_content_size: Maximum content size per slot
        """
        self._max_slots = max_slots
        self._max_content_size = max_content_size
        self._slots: OrderedDict[str, MemorySlot] = OrderedDict()
        self._by_type: dict[SlotType, list[str]] = {}
        self._by_tag: dict[str, list[str]] = {}
        self._lock = threading.RLock()
        self._counter = 0
        
    def store(self,
              content: Any,
              slot_type: SlotType = SlotType.SCRATCH,
              slot_id: Optional[str] = None,
              priority: float = 1.0,
              ttl: Optional[float] = None,
              tags: Optional[list[str]] = None) -> str:
        """
        Store content in working memory.
        
        Args:
            content: Content to store
            slot_type: Type of slot
            slot_id: Optional ID (auto-generated if None)
            priority: Priority (higher = more important)
            ttl: Time to live in seconds
            tags: Tags for retrieval
            
        Returns:
            Slot ID
        """
        with self._lock:
            # Generate ID if needed
            if slot_id is None:
                self._counter += 1
                slot_id = f"slot_{self._counter}"
                
            # Truncate content if needed
            if isinstance(content, str) and len(content) > self._max_content_size:
                content = content[:self._max_content_size] + "..."
                
            # Create slot
            slot = MemorySlot(
                id=slot_id,
                slot_type=slot_type,
                content=content,
                priority=priority,
                ttl=ttl,
                tags=tags or []
            )
            
            # Check capacity
            self._cleanup_expired()
            while len(self._slots) >= self._max_slots:
                self._evict_lowest_priority()
                
            # Store slot
            self._slots[slot_id] = slot
            self._slots.move_to_end(slot_id)
            
            # Index by type
            if slot_type not in self._by_type:
                self._by_type[slot_type] = []
            self._by_type[slot_type].append(slot_id)
            
            # Index by tags
            for tag in slot.tags:
                if tag not in self._by_tag:
                    self._by_tag[tag] = []
                self._by_tag[tag].append(slot_id)
                
            logger.debug(f"Stored in working memory: {slot_id} ({slot_type.value})")
            return slot_id
            
    def retrieve(self, slot_id: str) -> Optional[Any]:
        """
        Retrieve content from working memory.
        
        Args:
            slot_id: Slot ID
            
        Returns:
            Content or None
        """
        with self._lock:
            slot = self._slots.get(slot_id)
            if slot is None:
                return None
            if slot.is_expired():
                self._remove_slot(slot_id)
                return None
            slot.touch()
            return slot.content
            
    def get_by_type(self, slot_type: SlotType) -> list[tuple[str, Any]]:
        """Get all slots of a type."""
        with self._lock:
            result = []
            slot_ids = self._by_type.get(slot_type, [])
            for sid in slot_ids:
                slot = self._slots.get(sid)
                if slot and not slot.is_expired():
                    slot.touch()
                    result.append((sid, slot.content))
            return result
            
    def get_by_tag(self, tag: str) -> list[tuple[str, Any]]:
        """Get all slots with a tag."""
        with self._lock:
            result = []
            slot_ids = self._by_tag.get(tag, [])
            for sid in slot_ids:
                slot = self._slots.get(sid)
                if slot and not slot.is_expired():
                    slot.touch()
                    result.append((sid, slot.content))
            return result
            
    def update(self, slot_id: str, content: Any) -> bool:
        """Update slot content."""
        with self._lock:
            slot = self._slots.get(slot_id)
            if slot is None or slot.is_expired():
                return False
            
            if isinstance(content, str) and len(content) > self._max_content_size:
                content = content[:self._max_content_size] + "..."
                
            slot.content = content
            slot.touch()
            return True
            
    def delete(self, slot_id: str) -> bool:
        """Delete a slot."""
        with self._lock:
            return self._remove_slot(slot_id)
            
    def clear(self, slot_type: Optional[SlotType] = None):
        """Clear memory (optionally by type)."""
        with self._lock:
            if slot_type is None:
                self._slots.clear()
                self._by_type.clear()
                self._by_tag.clear()
            else:
                slot_ids = self._by_type.get(slot_type, []).copy()
                for sid in slot_ids:
                    self._remove_slot(sid)
                    
    def _remove_slot(self, slot_id: str) -> bool:
        """Remove a slot and update indices."""
        slot = self._slots.get(slot_id)
        if slot is None:
            return False
            
        # Remove from type index
        if slot.slot_type in self._by_type:
            try:
                self._by_type[slot.slot_type].remove(slot_id)
            except ValueError:
                pass  # Intentionally silent
                
        # Remove from tag indices
        for tag in slot.tags:
            if tag in self._by_tag:
                try:
                    self._by_tag[tag].remove(slot_id)
                except ValueError:
                    pass  # Intentionally silent
                    
        del self._slots[slot_id]
        return True
        
    def _cleanup_expired(self):
        """Remove expired slots."""
        expired = [sid for sid, slot in self._slots.items() if slot.is_expired()]
        for sid in expired:
            self._remove_slot(sid)
            
    def _evict_lowest_priority(self):
        """Evict the lowest priority slot."""
        if not self._slots:
            return
            
        # Score slots: lower priority * older access = higher eviction score
        scores = {}
        now = time.time()
        for sid, slot in self._slots.items():
            age = now - slot.accessed_at
            scores[sid] = (1.0 / max(slot.priority, 0.01)) * (age / 60.0)
            
        # Evict highest score
        evict_id = max(scores, key=scores.get)
        self._remove_slot(evict_id)
        logger.debug(f"Evicted slot: {evict_id}")
        
    def get_summary(self) -> str:
        """Get a text summary of working memory contents."""
        with self._lock:
            lines = [f"Working Memory ({len(self._slots)}/{self._max_slots} slots):"]
            for slot in self._slots.values():
                if slot.is_expired():
                    continue
                content_preview = str(slot.content)[:50]
                if len(str(slot.content)) > 50:
                    content_preview += "..."
                lines.append(f"  [{slot.slot_type.value}] {slot.id}: {content_preview}")
            return "\n".join(lines)
            
    def to_context_string(self) -> str:
        """Convert to a string for inclusion in prompts."""
        with self._lock:
            self._cleanup_expired()
            if not self._slots:
                return ""
                
            lines = ["<working_memory>"]
            for slot in self._slots.values():
                lines.append(f"  <{slot.slot_type.value} id='{slot.id}'>")
                lines.append(f"    {slot.content}")
                lines.append(f"  </{slot.slot_type.value}>")
            lines.append("</working_memory>")
            return "\n".join(lines)
            
    def __len__(self) -> int:
        return len(self._slots)


class WorkingMemoryManager:
    """Manages working memory instances for multiple sessions."""
    
    def __init__(self, max_sessions: int = 10):
        """
        Initialize manager.
        
        Args:
            max_sessions: Maximum concurrent sessions
        """
        self._sessions: dict[str, WorkingMemory] = {}
        self._max_sessions = max_sessions
        self._session_times: dict[str, float] = {}
        self._lock = threading.Lock()
        
    def get_session(self, session_id: str) -> WorkingMemory:
        """Get or create a working memory session."""
        with self._lock:
            if session_id not in self._sessions:
                # Check capacity
                while len(self._sessions) >= self._max_sessions:
                    self._evict_oldest_session()
                    
                self._sessions[session_id] = WorkingMemory()
                
            self._session_times[session_id] = time.time()
            return self._sessions[session_id]
            
    def close_session(self, session_id: str):
        """Close a working memory session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
            if session_id in self._session_times:
                del self._session_times[session_id]
                
    def _evict_oldest_session(self):
        """Evict the oldest session."""
        if not self._session_times:
            return
        oldest = min(self._session_times, key=self._session_times.get)
        self.close_session(oldest)
        logger.debug(f"Evicted session: {oldest}")


# Singleton manager
_wm_manager: Optional[WorkingMemoryManager] = None


def get_working_memory_manager() -> WorkingMemoryManager:
    """Get the working memory manager singleton."""
    global _wm_manager
    if _wm_manager is None:
        _wm_manager = WorkingMemoryManager()
    return _wm_manager


def get_working_memory(session_id: str = "default") -> WorkingMemory:
    """Get working memory for a session."""
    return get_working_memory_manager().get_session(session_id)


# Convenience functions
def remember(content: Any, 
             slot_type: SlotType = SlotType.SCRATCH,
             session_id: str = "default",
             **kwargs) -> str:
    """Store something in working memory."""
    wm = get_working_memory(session_id)
    return wm.store(content, slot_type, **kwargs)


def recall(slot_id: str, session_id: str = "default") -> Optional[Any]:
    """Recall from working memory."""
    wm = get_working_memory(session_id)
    return wm.retrieve(slot_id)


def forget(slot_id: str, session_id: str = "default") -> bool:
    """Forget from working memory."""
    wm = get_working_memory(session_id)
    return wm.delete(slot_id)


__all__ = [
    'WorkingMemory',
    'WorkingMemoryManager',
    'MemorySlot',
    'SlotType',
    'get_working_memory',
    'get_working_memory_manager',
    'remember',
    'recall',
    'forget'
]
