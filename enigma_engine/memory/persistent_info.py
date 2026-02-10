"""
Persistent Information Storage - Save and retrieve important information.

Features:
- User notes and reminders
- AI-saved facts and context
- Tagged organization
- Search and filtering
- Import/export

This allows both the user and AI to save information that persists
across sessions and can be recalled when needed.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# INFO TYPES
# =============================================================================

class InfoType(Enum):
    """Types of stored information."""
    NOTE = "note"           # User notes/reminders
    FACT = "fact"           # AI-learned facts about user
    PREFERENCE = "pref"     # User preferences
    CONTEXT = "context"     # Conversation context to remember
    TASK = "task"           # Task/todo items
    BOOKMARK = "bookmark"   # Bookmarked content
    SNIPPET = "snippet"     # Code/text snippets
    CONTACT = "contact"     # Contact information
    CUSTOM = "custom"       # Custom type


class InfoPriority(Enum):
    """Priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


# =============================================================================
# STORED INFORMATION
# =============================================================================

@dataclass
class StoredInfo:
    """A piece of stored information."""
    id: str
    title: str
    content: str
    info_type: InfoType
    source: str  # "user" or "ai"
    tags: list[str] = field(default_factory=list)
    priority: InfoPriority = InfoPriority.MEDIUM
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    expires_at: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    pinned: bool = False
    archived: bool = False
    
    def is_expired(self) -> bool:
        """Check if info has expired."""
        if not self.expires_at:
            return False
        try:
            exp = datetime.fromisoformat(self.expires_at)
            return datetime.now() > exp
        except Exception:
            return False
    
    def matches_search(self, query: str) -> bool:
        """Check if info matches search query."""
        query_lower = query.lower()
        return (
            query_lower in self.title.lower() or
            query_lower in self.content.lower() or
            any(query_lower in tag.lower() for tag in self.tags)
        )
    
    def age_formatted(self) -> str:
        """Get human-readable age."""
        try:
            created = datetime.fromisoformat(self.created_at)
            delta = datetime.now() - created
            
            if delta.days > 365:
                return f"{delta.days // 365}y ago"
            elif delta.days > 30:
                return f"{delta.days // 30}mo ago"
            elif delta.days > 0:
                return f"{delta.days}d ago"
            elif delta.seconds > 3600:
                return f"{delta.seconds // 3600}h ago"
            elif delta.seconds > 60:
                return f"{delta.seconds // 60}m ago"
            return "just now"
        except Exception:
            return "unknown"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "info_type": self.info_type.value,
            "source": self.source,
            "tags": self.tags,
            "priority": self.priority.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "metadata": self.metadata,
            "pinned": self.pinned,
            "archived": self.archived
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StoredInfo":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            title=data["title"],
            content=data["content"],
            info_type=InfoType(data["info_type"]),
            source=data["source"],
            tags=data.get("tags", []),
            priority=InfoPriority(data.get("priority", 2)),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            expires_at=data.get("expires_at"),
            metadata=data.get("metadata", {}),
            pinned=data.get("pinned", False),
            archived=data.get("archived", False)
        )


# =============================================================================
# USER PREFERENCES STORE
# =============================================================================

class UserPreferences:
    """
    Store user preferences learned by the AI.
    
    Examples:
    - Prefers dark mode
    - Likes detailed explanations
    - Primary programming language: Python
    """
    
    def __init__(self, storage: "PersistentInfoStorage"):
        """Initialize with parent storage."""
        self._storage = storage
    
    def set(self, key: str, value: Any, source: str = "ai"):
        """Set a preference."""
        # Check if exists
        existing = self._storage.search(
            query=key,
            info_type=InfoType.PREFERENCE
        )
        
        for item in existing:
            if item.title.lower() == key.lower():
                # Update existing
                item.content = json.dumps(value) if not isinstance(value, str) else value
                item.updated_at = datetime.now().isoformat()
                item.source = source
                self._storage._save()
                return
        
        # Create new
        self._storage.add(
            title=key,
            content=json.dumps(value) if not isinstance(value, str) else value,
            info_type=InfoType.PREFERENCE,
            source=source,
            tags=["preference", "auto-learned"]
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a preference."""
        items = self._storage.search(
            query=key,
            info_type=InfoType.PREFERENCE
        )
        
        for item in items:
            if item.title.lower() == key.lower():
                try:
                    return json.loads(item.content)
                except Exception:
                    return item.content
        
        return default
    
    def all(self) -> dict[str, Any]:
        """Get all preferences."""
        result = {}
        for item in self._storage.list(info_type=InfoType.PREFERENCE):
            try:
                result[item.title] = json.loads(item.content)
            except Exception:
                result[item.title] = item.content
        return result


# =============================================================================
# AI FACTS STORE
# =============================================================================

class AIFacts:
    """
    Store facts the AI has learned about the user.
    
    Examples:
    - User's name is John
    - Works as a software developer
    - Has a dog named Max
    """
    
    def __init__(self, storage: "PersistentInfoStorage"):
        """Initialize with parent storage."""
        self._storage = storage
    
    def remember(self, fact: str, category: str = "general"):
        """Remember a fact."""
        # Avoid duplicates
        existing = self._storage.search(query=fact, info_type=InfoType.FACT)
        for item in existing:
            if item.content.lower() == fact.lower():
                return  # Already stored
        
        self._storage.add(
            title=f"Fact: {category}",
            content=fact,
            info_type=InfoType.FACT,
            source="ai",
            tags=["fact", category]
        )
    
    def recall(self, query: str, limit: int = 10) -> list[str]:
        """Recall facts matching query."""
        items = self._storage.search(query=query, info_type=InfoType.FACT, limit=limit)
        return [item.content for item in items]
    
    def forget(self, fact_content: str) -> bool:
        """Forget a specific fact."""
        for item in self._storage.list(info_type=InfoType.FACT):
            if item.content.lower() == fact_content.lower():
                self._storage.delete(item.id)
                return True
        return False
    
    def all_facts(self) -> list[str]:
        """Get all stored facts."""
        return [item.content for item in self._storage.list(info_type=InfoType.FACT)]


# =============================================================================
# CONTEXT MEMORY
# =============================================================================

class ContextMemory:
    """
    Store important context that should persist across conversations.
    
    Examples:
    - Current project being worked on
    - Ongoing topics of discussion
    - Important deadlines
    """
    
    def __init__(self, storage: "PersistentInfoStorage"):
        """Initialize with parent storage."""
        self._storage = storage
    
    def save_context(
        self,
        key: str,
        value: str,
        expires_hours: Optional[int] = None
    ):
        """Save a context value."""
        expires_at = None
        if expires_hours:
            from datetime import timedelta
            expires_at = (datetime.now() + timedelta(hours=expires_hours)).isoformat()
        
        # Update if exists
        for item in self._storage.list(info_type=InfoType.CONTEXT):
            if item.title == key:
                item.content = value
                item.updated_at = datetime.now().isoformat()
                item.expires_at = expires_at
                self._storage._save()
                return
        
        # Create new
        self._storage.add(
            title=key,
            content=value,
            info_type=InfoType.CONTEXT,
            source="ai",
            tags=["context"],
            expires_at=expires_at
        )
    
    def get_context(self, key: str) -> Optional[str]:
        """Get a context value."""
        for item in self._storage.list(info_type=InfoType.CONTEXT):
            if item.title == key and not item.is_expired():
                return item.content
        return None
    
    def get_all_context(self) -> dict[str, str]:
        """Get all active context."""
        result = {}
        for item in self._storage.list(info_type=InfoType.CONTEXT):
            if not item.is_expired():
                result[item.title] = item.content
        return result
    
    def clear_expired(self) -> int:
        """Clear expired context items."""
        count = 0
        for item in self._storage.list(info_type=InfoType.CONTEXT):
            if item.is_expired():
                self._storage.delete(item.id)
                count += 1
        return count


# =============================================================================
# MAIN STORAGE CLASS
# =============================================================================

class PersistentInfoStorage:
    """
    Main persistent information storage.
    
    Features:
    - Multiple info types (notes, facts, preferences, context)
    - Tag-based organization
    - Full-text search
    - Priority and pinning
    - Archive/restore
    - Import/export
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize persistent storage.
        
        Args:
            storage_path: Path to storage file
        """
        self.storage_path = storage_path or Path("data/persistent_info.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._items: dict[str, StoredInfo] = {}
        self._lock = threading.Lock()
        self._id_counter = 0
        
        # Sub-stores
        self.preferences = UserPreferences(self)
        self.facts = AIFacts(self)
        self.context = ContextMemory(self)
        
        # Load existing data
        self._load()
    
    def _generate_id(self) -> str:
        """Generate unique ID."""
        self._id_counter += 1
        timestamp = int(time.time() * 1000)
        return f"info_{timestamp}_{self._id_counter}"
    
    def add(
        self,
        title: str,
        content: str,
        info_type: InfoType = InfoType.NOTE,
        source: str = "user",
        tags: Optional[list[str]] = None,
        priority: InfoPriority = InfoPriority.MEDIUM,
        expires_at: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None
    ) -> str:
        """
        Add new information.
        
        Args:
            title: Short title/summary
            content: Full content
            info_type: Type of information
            source: "user" or "ai"
            tags: Optional tags
            priority: Priority level
            expires_at: Optional expiration ISO timestamp
            metadata: Additional metadata
            
        Returns:
            ID of created item
        """
        with self._lock:
            info = StoredInfo(
                id=self._generate_id(),
                title=title,
                content=content,
                info_type=info_type,
                source=source,
                tags=tags or [],
                priority=priority,
                expires_at=expires_at,
                metadata=metadata or {}
            )
            
            self._items[info.id] = info
            self._save()
            
            return info.id
    
    def get(self, item_id: str) -> Optional[StoredInfo]:
        """Get item by ID."""
        return self._items.get(item_id)
    
    def update(
        self,
        item_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        tags: Optional[list[str]] = None,
        priority: Optional[InfoPriority] = None,
        pinned: Optional[bool] = None
    ) -> bool:
        """Update an item."""
        with self._lock:
            item = self._items.get(item_id)
            if not item:
                return False
            
            if title is not None:
                item.title = title
            if content is not None:
                item.content = content
            if tags is not None:
                item.tags = tags
            if priority is not None:
                item.priority = priority
            if pinned is not None:
                item.pinned = pinned
            
            item.updated_at = datetime.now().isoformat()
            self._save()
            return True
    
    def delete(self, item_id: str) -> bool:
        """Delete an item."""
        with self._lock:
            if item_id in self._items:
                del self._items[item_id]
                self._save()
                return True
            return False
    
    def archive(self, item_id: str) -> bool:
        """Archive an item."""
        with self._lock:
            item = self._items.get(item_id)
            if item:
                item.archived = True
                item.updated_at = datetime.now().isoformat()
                self._save()
                return True
            return False
    
    def unarchive(self, item_id: str) -> bool:
        """Unarchive an item."""
        with self._lock:
            item = self._items.get(item_id)
            if item:
                item.archived = False
                item.updated_at = datetime.now().isoformat()
                self._save()
                return True
            return False
    
    def pin(self, item_id: str) -> bool:
        """Pin an item."""
        return self.update(item_id, pinned=True)
    
    def unpin(self, item_id: str) -> bool:
        """Unpin an item."""
        return self.update(item_id, pinned=False)
    
    def list(
        self,
        info_type: Optional[InfoType] = None,
        source: Optional[str] = None,
        tags: Optional[list[str]] = None,
        include_archived: bool = False,
        pinned_only: bool = False
    ) -> list[StoredInfo]:
        """
        List items with optional filtering.
        
        Args:
            info_type: Filter by type
            source: Filter by source (user/ai)
            tags: Filter by tags (any match)
            include_archived: Include archived items
            pinned_only: Only pinned items
            
        Returns:
            Filtered list of items
        """
        result = []
        
        for item in self._items.values():
            # Skip expired
            if item.is_expired():
                continue
            
            # Skip archived unless requested
            if item.archived and not include_archived:
                continue
            
            # Filter by type
            if info_type and item.info_type != info_type:
                continue
            
            # Filter by source
            if source and item.source != source:
                continue
            
            # Filter by tags
            if tags and not any(t in item.tags for t in tags):
                continue
            
            # Filter by pinned
            if pinned_only and not item.pinned:
                continue
            
            result.append(item)
        
        # Sort: pinned first, then by priority, then by date
        result.sort(key=lambda x: (
            not x.pinned,
            -x.priority.value,
            x.updated_at
        ), reverse=True)
        
        return result
    
    def search(
        self,
        query: str,
        info_type: Optional[InfoType] = None,
        limit: int = 20
    ) -> list[StoredInfo]:
        """
        Search items by query.
        
        Args:
            query: Search query
            info_type: Filter by type
            limit: Max results
            
        Returns:
            Matching items
        """
        if not query:
            return self.list(info_type=info_type)[:limit]
        
        results = []
        query_lower = query.lower()
        
        for item in self._items.values():
            if item.is_expired() or item.archived:
                continue
            
            if info_type and item.info_type != info_type:
                continue
            
            if item.matches_search(query):
                # Score relevance
                score = 0
                if query_lower in item.title.lower():
                    score += 10
                if query_lower in item.content.lower():
                    score += 5
                for tag in item.tags:
                    if query_lower in tag.lower():
                        score += 3
                
                results.append((score, item))
        
        # Sort by score
        results.sort(key=lambda x: x[0], reverse=True)
        
        return [item for _, item in results[:limit]]
    
    def get_tags(self) -> list[str]:
        """Get all unique tags."""
        tags = set()
        for item in self._items.values():
            if not item.archived:
                tags.update(item.tags)
        return sorted(tags)
    
    def get_by_tag(self, tag: str) -> list[StoredInfo]:
        """Get all items with a specific tag."""
        return [
            item for item in self._items.values()
            if tag in item.tags and not item.archived and not item.is_expired()
        ]
    
    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        active = [i for i in self._items.values() if not i.archived and not i.is_expired()]
        
        by_type = {}
        for item in active:
            by_type[item.info_type.value] = by_type.get(item.info_type.value, 0) + 1
        
        by_source = {"user": 0, "ai": 0}
        for item in active:
            by_source[item.source] = by_source.get(item.source, 0) + 1
        
        return {
            "total_items": len(self._items),
            "active_items": len(active),
            "archived_items": len(self._items) - len(active),
            "by_type": by_type,
            "by_source": by_source,
            "unique_tags": len(self.get_tags()),
            "pinned_count": sum(1 for i in active if i.pinned)
        }
    
    def export_json(self) -> str:
        """Export all data as JSON."""
        data = {
            "version": 1,
            "exported_at": datetime.now().isoformat(),
            "items": [item.to_dict() for item in self._items.values()]
        }
        return json.dumps(data, indent=2)
    
    def import_json(self, json_str: str, merge: bool = True) -> int:
        """
        Import data from JSON.
        
        Args:
            json_str: JSON string to import
            merge: If True, merge with existing. If False, replace.
            
        Returns:
            Number of items imported
        """
        try:
            data = json.loads(json_str)
            items = data.get("items", [])
            
            with self._lock:
                if not merge:
                    self._items.clear()
                
                count = 0
                for item_data in items:
                    try:
                        item = StoredInfo.from_dict(item_data)
                        # Generate new ID if merging to avoid conflicts
                        if merge and item.id in self._items:
                            item.id = self._generate_id()
                        self._items[item.id] = item
                        count += 1
                    except Exception as e:
                        logger.warning(f"Failed to import item: {e}")
                
                self._save()
                return count
                
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return 0
    
    def _save(self):
        """Save to disk."""
        try:
            data = {
                "version": 1,
                "saved_at": datetime.now().isoformat(),
                "id_counter": self._id_counter,
                "items": {k: v.to_dict() for k, v in self._items.items()}
            }
            
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save persistent info: {e}")
    
    def _load(self):
        """Load from disk."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
            
            self._id_counter = data.get("id_counter", 0)
            
            for item_id, item_data in data.get("items", {}).items():
                try:
                    self._items[item_id] = StoredInfo.from_dict(item_data)
                except Exception as e:
                    logger.warning(f"Failed to load item {item_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to load persistent info: {e}")


# =============================================================================
# QUICK ACCESS FUNCTIONS
# =============================================================================

# Singleton instance
_storage: Optional[PersistentInfoStorage] = None


def get_storage() -> PersistentInfoStorage:
    """Get or create the persistent storage instance."""
    global _storage
    if _storage is None:
        _storage = PersistentInfoStorage()
    return _storage


def remember(content: str, title: Optional[str] = None, source: str = "ai") -> str:
    """Quick function to remember something."""
    storage = get_storage()
    return storage.add(
        title=title or content[:50],
        content=content,
        info_type=InfoType.FACT if source == "ai" else InfoType.NOTE,
        source=source
    )


def recall(query: str, limit: int = 5) -> list[str]:
    """Quick function to recall information."""
    storage = get_storage()
    results = storage.search(query, limit=limit)
    return [f"{r.title}: {r.content}" for r in results]


def save_note(title: str, content: str, tags: Optional[list[str]] = None) -> str:
    """Quick function to save a user note."""
    storage = get_storage()
    return storage.add(
        title=title,
        content=content,
        info_type=InfoType.NOTE,
        source="user",
        tags=tags
    )


def set_preference(key: str, value: Any) -> None:
    """Quick function to set a preference."""
    storage = get_storage()
    storage.preferences.set(key, value, source="ai")


def get_preference(key: str, default: Any = None) -> Any:
    """Quick function to get a preference."""
    storage = get_storage()
    return storage.preferences.get(key, default)
