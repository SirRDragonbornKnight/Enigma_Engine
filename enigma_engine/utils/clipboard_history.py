"""
================================================================================
Clipboard History - Track and manage clipboard contents.
================================================================================

Clipboard management features:
- Track clipboard history
- Search past clipboard items
- Pin important items
- Categorize by type (text, code, URL, etc.)
- Sync across sessions

USAGE:
    from enigma_engine.utils.clipboard_history import ClipboardHistory, get_clipboard_history
    
    history = get_clipboard_history()
    
    # Start monitoring clipboard
    history.start_monitoring()
    
    # Get recent items
    items = history.get_recent(limit=10)
    
    # Search history
    results = history.search("function")
    
    # Copy from history
    history.copy_to_clipboard(item_id)
    
    # Pin an item
    history.pin(item_id)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ContentType(str, Enum):
    """Types of clipboard content."""
    TEXT = "text"
    CODE = "code"
    URL = "url"
    EMAIL = "email"
    PATH = "path"
    JSON = "json"
    NUMBER = "number"
    UNKNOWN = "unknown"


@dataclass
class ClipboardItem:
    """A clipboard history item."""
    id: str
    content: str
    content_type: ContentType
    created_at: str
    
    # Metadata
    source_app: str = ""
    char_count: int = 0
    word_count: int = 0
    line_count: int = 0
    
    # User data
    pinned: bool = False
    favorite: bool = False
    tags: list[str] = field(default_factory=list)
    note: str = ""
    
    # Usage tracking
    copy_count: int = 1
    last_used: str = ""
    
    def __post_init__(self):
        self.char_count = len(self.content)
        self.word_count = len(self.content.split())
        self.line_count = self.content.count('\n') + 1
        if not self.last_used:
            self.last_used = self.created_at
    
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["content_type"] = self.content_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClipboardItem:
        if "content_type" in data:
            data["content_type"] = ContentType(data["content_type"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @property
    def preview(self) -> str:
        """Get a preview of the content."""
        preview = self.content[:100].replace('\n', ' ').strip()
        if len(self.content) > 100:
            preview += "..."
        return preview


def detect_content_type(content: str) -> ContentType:
    """Detect the type of clipboard content."""
    content = content.strip()
    
    # URL detection
    if re.match(r'^https?://', content, re.IGNORECASE):
        return ContentType.URL
    
    # Email detection
    if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', content):
        return ContentType.EMAIL
    
    # Path detection (Windows or Unix)
    if re.match(r'^([A-Za-z]:\\|/|~/).+', content):
        return ContentType.PATH
    
    # JSON detection
    if (content.startswith('{') and content.endswith('}')) or \
       (content.startswith('[') and content.endswith(']')):
        try:
            json.loads(content)
            return ContentType.JSON
        except (json.JSONDecodeError, ValueError):
            pass  # Intentionally silent
    
    # Code detection (heuristic)
    code_indicators = [
        r'\bdef\s+\w+\s*\(',  # Python function
        r'\bfunction\s+\w+\s*\(',  # JavaScript function
        r'\bclass\s+\w+',  # Class definition
        r'\bimport\s+',  # Import statement
        r'\breturn\s+',  # Return statement
        r'^\s*[{}]\s*$',  # Braces on own line
        r';\s*$',  # Semicolon at end
        r'=>',  # Arrow function
        r'const\s+\w+\s*=',  # Const declaration
        r'let\s+\w+\s*=',  # Let declaration
        r'var\s+\w+\s*=',  # Var declaration
    ]
    
    for pattern in code_indicators:
        if re.search(pattern, content, re.MULTILINE):
            return ContentType.CODE
    
    # Number detection
    if re.match(r'^-?[\d,]+\.?\d*$', content.replace(',', '')):
        return ContentType.NUMBER
    
    return ContentType.TEXT


class ClipboardHistory:
    """
    Manage clipboard history with persistence.
    """
    
    def __init__(
        self,
        data_path: Path | None = None,
        max_items: int = 1000,
        max_content_length: int = 100000
    ):
        """
        Initialize clipboard history.
        
        Args:
            data_path: Path to store history
            max_items: Maximum items to keep
            max_content_length: Max content size per item
        """
        self._data_path = data_path or Path("data/clipboard")
        self._data_path.mkdir(parents=True, exist_ok=True)
        
        self._history_file = self._data_path / "history.json"
        self._max_items = max_items
        self._max_content_length = max_content_length
        
        self._items: dict[str, ClipboardItem] = {}
        self._order: list[str] = []  # Most recent first
        
        self._monitoring = False
        self._monitor_thread: threading.Thread | None = None
        self._last_content = ""
        
        self._callbacks: list[Callable[[ClipboardItem], None]] = []
        
        self._load_history()
    
    def _load_history(self) -> None:
        """Load history from disk."""
        if self._history_file.exists():
            try:
                with open(self._history_file, encoding='utf-8') as f:
                    data = json.load(f)
                    for item_data in data.get("items", []):
                        item = ClipboardItem.from_dict(item_data)
                        self._items[item.id] = item
                        self._order.append(item.id)
                logger.info(f"Loaded {len(self._items)} clipboard items")
            except Exception as e:
                logger.error(f"Failed to load clipboard history: {e}")
    
    def _save_history(self) -> None:
        """Save history to disk."""
        try:
            data = {
                "items": [self._items[id].to_dict() for id in self._order if id in self._items],
                "last_updated": datetime.now().isoformat()
            }
            with open(self._history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save clipboard history: {e}")
    
    def _generate_id(self, content: str) -> str:
        """Generate a unique ID for content."""
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _get_clipboard(self) -> str | None:
        """Get current clipboard content."""
        try:
            # Try pyperclip first
            try:
                import pyperclip
                return pyperclip.paste()
            except ImportError:
                pass  # Intentionally silent
            
            # Try tkinter
            try:
                import tkinter as tk
                root = tk.Tk()
                root.withdraw()
                content = root.clipboard_get()
                root.destroy()
                return content
            except (ImportError, tk.TclError) as e:
                logger.debug(f"tkinter clipboard read failed: {e}")
            
            # Platform-specific fallbacks
            import sys
            if sys.platform == 'win32':
                import ctypes
                
                CF_UNICODETEXT = 13
                user32 = ctypes.windll.user32
                kernel32 = ctypes.windll.kernel32
                
                if not user32.OpenClipboard(None):
                    return None
                
                try:
                    handle = user32.GetClipboardData(CF_UNICODETEXT)
                    if handle:
                        data = ctypes.c_wchar_p(handle)
                        return data.value
                finally:
                    user32.CloseClipboard()
            
        except Exception as e:
            logger.debug(f"Clipboard read error: {e}")
        
        return None
    
    def _set_clipboard(self, content: str) -> bool:
        """Set clipboard content."""
        try:
            # Try pyperclip first
            try:
                import pyperclip
                pyperclip.copy(content)
                return True
            except ImportError:
                pass  # Intentionally silent
            
            # Try tkinter
            try:
                import tkinter as tk
                root = tk.Tk()
                root.withdraw()
                root.clipboard_clear()
                root.clipboard_append(content)
                root.update()
                root.destroy()
                return True
            except (ImportError, tk.TclError) as e:
                logger.debug(f"tkinter clipboard write failed: {e}")
            
            # Windows fallback
            import sys
            if sys.platform == 'win32':
                import subprocess
                proc = subprocess.Popen(['clip'], stdin=subprocess.PIPE)
                proc.communicate(content.encode('utf-16-le'), timeout=5)
                return True
            
        except Exception as e:
            logger.error(f"Clipboard write error: {e}")
        
        return False
    
    def add(self, content: str, source_app: str = "") -> ClipboardItem | None:
        """
        Add content to history.
        
        Args:
            content: Content to add
            source_app: Source application name
            
        Returns:
            ClipboardItem or None if filtered
        """
        # Filter
        content = content.strip()
        if not content or len(content) > self._max_content_length:
            return None
        
        item_id = self._generate_id(content)
        
        # Check for duplicate
        if item_id in self._items:
            # Update existing item
            item = self._items[item_id]
            item.copy_count += 1
            item.last_used = datetime.now().isoformat()
            
            # Move to front
            if item_id in self._order:
                self._order.remove(item_id)
            self._order.insert(0, item_id)
        else:
            # Create new item
            item = ClipboardItem(
                id=item_id,
                content=content,
                content_type=detect_content_type(content),
                created_at=datetime.now().isoformat(),
                source_app=source_app
            )
            
            self._items[item_id] = item
            self._order.insert(0, item_id)
            
            # Trim history
            while len(self._order) > self._max_items:
                old_id = self._order.pop()
                if old_id in self._items and not self._items[old_id].pinned:
                    del self._items[old_id]
        
        self._save_history()
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(item)
            except Exception as e:
                logger.error(f"Clipboard callback error: {e}")
        
        return item
    
    def get(self, item_id: str) -> ClipboardItem | None:
        """Get an item by ID."""
        return self._items.get(item_id)
    
    def get_recent(self, limit: int = 20) -> list[ClipboardItem]:
        """Get recent clipboard items."""
        items = []
        for item_id in self._order[:limit]:
            if item_id in self._items:
                items.append(self._items[item_id])
        return items
    
    def get_pinned(self) -> list[ClipboardItem]:
        """Get pinned items."""
        return [i for i in self._items.values() if i.pinned]
    
    def search(
        self,
        query: str,
        content_type: ContentType | None = None,
        limit: int = 20
    ) -> list[ClipboardItem]:
        """
        Search clipboard history.
        
        Args:
            query: Search query
            content_type: Filter by type
            limit: Max results
            
        Returns:
            Matching items
        """
        query_lower = query.lower()
        results = []
        
        for item_id in self._order:
            if item_id not in self._items:
                continue
            
            item = self._items[item_id]
            
            # Type filter
            if content_type and item.content_type != content_type:
                continue
            
            # Search in content, tags, and note
            searchable = f"{item.content} {' '.join(item.tags)} {item.note}".lower()
            if query_lower in searchable:
                results.append(item)
                
                if len(results) >= limit:
                    break
        
        return results
    
    def copy_to_clipboard(self, item_id: str) -> bool:
        """
        Copy an item back to clipboard.
        
        Args:
            item_id: Item ID
            
        Returns:
            True if successful
        """
        item = self._items.get(item_id)
        if not item:
            return False
        
        if self._set_clipboard(item.content):
            item.copy_count += 1
            item.last_used = datetime.now().isoformat()
            self._last_content = item.content  # Prevent re-adding
            self._save_history()
            return True
        
        return False
    
    def pin(self, item_id: str, pinned: bool = True) -> bool:
        """Pin or unpin an item."""
        item = self._items.get(item_id)
        if item:
            item.pinned = pinned
            self._save_history()
            return True
        return False
    
    def favorite(self, item_id: str, favorited: bool = True) -> bool:
        """Favorite or unfavorite an item."""
        item = self._items.get(item_id)
        if item:
            item.favorite = favorited
            self._save_history()
            return True
        return False
    
    def add_tag(self, item_id: str, tag: str) -> bool:
        """Add a tag to an item."""
        item = self._items.get(item_id)
        if item and tag not in item.tags:
            item.tags.append(tag)
            self._save_history()
            return True
        return False
    
    def set_note(self, item_id: str, note: str) -> bool:
        """Set a note on an item."""
        item = self._items.get(item_id)
        if item:
            item.note = note
            self._save_history()
            return True
        return False
    
    def delete(self, item_id: str) -> bool:
        """Delete an item."""
        if item_id in self._items:
            del self._items[item_id]
            if item_id in self._order:
                self._order.remove(item_id)
            self._save_history()
            return True
        return False
    
    def clear(self, keep_pinned: bool = True) -> int:
        """
        Clear history.
        
        Args:
            keep_pinned: Keep pinned items
            
        Returns:
            Number of items removed
        """
        removed = 0
        
        for item_id in list(self._items.keys()):
            item = self._items[item_id]
            if keep_pinned and item.pinned:
                continue
            del self._items[item_id]
            if item_id in self._order:
                self._order.remove(item_id)
            removed += 1
        
        self._save_history()
        return removed
    
    def start_monitoring(self, interval: float = 0.5) -> None:
        """
        Start monitoring clipboard for changes.
        
        Args:
            interval: Check interval in seconds
        """
        if self._monitoring:
            return
        
        self._monitoring = True
        self._last_content = self._get_clipboard() or ""
        
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Started clipboard monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring clipboard."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Stopped clipboard monitoring")
    
    def _monitor_loop(self, interval: float) -> None:
        """Monitoring loop."""
        while self._monitoring:
            time.sleep(interval)
            
            try:
                content = self._get_clipboard()
                if content and content != self._last_content:
                    self._last_content = content
                    self.add(content)
            except Exception as e:
                logger.debug(f"Monitor loop error: {e}")
    
    def add_callback(self, callback: Callable[[ClipboardItem], None]) -> None:
        """Add a callback for new clipboard items."""
        self._callbacks.append(callback)
    
    def get_stats(self) -> dict[str, Any]:
        """Get clipboard statistics."""
        type_counts = {}
        for item in self._items.values():
            type_name = item.content_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        return {
            "total_items": len(self._items),
            "pinned_items": len(self.get_pinned()),
            "type_distribution": type_counts,
            "total_characters": sum(i.char_count for i in self._items.values()),
            "most_copied": sorted(
                self._items.values(),
                key=lambda x: x.copy_count,
                reverse=True
            )[:5]
        }


# Singleton instance
_clipboard_instance: ClipboardHistory | None = None


def get_clipboard_history(data_path: Path | None = None) -> ClipboardHistory:
    """Get or create the singleton clipboard history."""
    global _clipboard_instance
    if _clipboard_instance is None:
        _clipboard_instance = ClipboardHistory(data_path)
    return _clipboard_instance


# Convenience functions
def add_to_history(content: str) -> ClipboardItem | None:
    """Add content to clipboard history."""
    return get_clipboard_history().add(content)


def get_recent_clips(limit: int = 20) -> list[ClipboardItem]:
    """Get recent clipboard items."""
    return get_clipboard_history().get_recent(limit)


def search_clips(query: str) -> list[ClipboardItem]:
    """Search clipboard history."""
    return get_clipboard_history().search(query)
