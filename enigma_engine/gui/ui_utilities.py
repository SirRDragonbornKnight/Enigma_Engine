"""
UI Utilities - Clipboard history, emoji picker, bookmarks, tab management.

User interface enhancements:
- Clipboard history tracking and paste
- Emoji picker with categories and search
- Message bookmarking with notes
- Tab history and color coding

Part of the Enigma AI Engine GUI enhancement suite.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# CLIPBOARD HISTORY
# =============================================================================

@dataclass
class ClipboardEntry:
    """A clipboard history entry."""
    content: str
    timestamp: float = field(default_factory=time.time)
    content_type: str = "text"  # "text", "image", "file"
    source: str = ""  # Application that copied
    pinned: bool = False
    
    def preview(self, max_length: int = 50) -> str:
        """Get preview of content."""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length - 3] + "..."
    
    def age_formatted(self) -> str:
        """Get human-readable age."""
        age = time.time() - self.timestamp
        if age < 60:
            return "just now"
        elif age < 3600:
            return f"{int(age / 60)}m ago"
        elif age < 86400:
            return f"{int(age / 3600)}h ago"
        return f"{int(age / 86400)}d ago"


class ClipboardHistory:
    """
    Track clipboard history for easy paste into chat.
    
    Features:
    - Store recent clipboard entries
    - Pin important entries
    - Search history
    - Auto-cleanup old entries
    """
    
    def __init__(
        self,
        max_entries: int = 100,
        max_age_days: int = 7,
        persist_path: Optional[Path] = None
    ):
        """
        Initialize clipboard history.
        
        Args:
            max_entries: Maximum entries to keep
            max_age_days: Auto-delete entries older than this
            persist_path: Path to save history (optional)
        """
        self.max_entries = max_entries
        self.max_age_seconds = max_age_days * 86400
        self.persist_path = persist_path
        
        self._entries: list[ClipboardEntry] = []
        self._last_content: str = ""
        self._callbacks: list[Callable[[ClipboardEntry], None]] = []
        
        if persist_path and persist_path.exists():
            self._load()
    
    def add(self, content: str, content_type: str = "text", source: str = ""):
        """Add entry to history."""
        # Skip duplicates of last entry
        if content == self._last_content:
            return
        
        self._last_content = content
        
        entry = ClipboardEntry(
            content=content,
            content_type=content_type,
            source=source
        )
        
        self._entries.insert(0, entry)
        self._cleanup()
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(entry)
            except Exception as e:
                logger.warning(f"Clipboard callback error: {e}")
        
        self._save()
    
    def get_history(self, limit: int = 50) -> list[ClipboardEntry]:
        """Get recent history."""
        return self._entries[:limit]
    
    def get_pinned(self) -> list[ClipboardEntry]:
        """Get pinned entries."""
        return [e for e in self._entries if e.pinned]
    
    def pin(self, index: int, pinned: bool = True):
        """Pin/unpin an entry."""
        if 0 <= index < len(self._entries):
            self._entries[index].pinned = pinned
            self._save()
    
    def delete(self, index: int):
        """Delete an entry."""
        if 0 <= index < len(self._entries):
            del self._entries[index]
            self._save()
    
    def clear(self, keep_pinned: bool = True):
        """Clear history."""
        if keep_pinned:
            self._entries = [e for e in self._entries if e.pinned]
        else:
            self._entries = []
        self._save()
    
    def search(self, query: str) -> list[ClipboardEntry]:
        """Search history."""
        query_lower = query.lower()
        return [e for e in self._entries if query_lower in e.content.lower()]
    
    def on_add(self, callback: Callable[[ClipboardEntry], None]):
        """Register callback for new entries."""
        self._callbacks.append(callback)
    
    def _cleanup(self):
        """Remove old entries over limit."""
        cutoff = time.time() - self.max_age_seconds
        
        # Keep pinned regardless of age
        self._entries = [
            e for e in self._entries
            if e.pinned or e.timestamp > cutoff
        ][:self.max_entries]
    
    def _save(self):
        """Save to disk."""
        if not self.persist_path:
            return
        
        try:
            data = [
                {
                    "content": e.content,
                    "timestamp": e.timestamp,
                    "content_type": e.content_type,
                    "source": e.source,
                    "pinned": e.pinned
                }
                for e in self._entries
            ]
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self.persist_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save clipboard history: {e}")
    
    def _load(self):
        """Load from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return
        
        try:
            data = json.loads(self.persist_path.read_text())
            self._entries = [
                ClipboardEntry(
                    content=d["content"],
                    timestamp=d.get("timestamp", time.time()),
                    content_type=d.get("content_type", "text"),
                    source=d.get("source", ""),
                    pinned=d.get("pinned", False)
                )
                for d in data
            ]
        except Exception as e:
            logger.warning(f"Failed to load clipboard history: {e}")


# =============================================================================
# EMOJI PICKER
# =============================================================================

class EmojiCategory(Enum):
    """Emoji categories."""
    SMILEYS = "Smileys & Emotion"
    PEOPLE = "People & Body"
    ANIMALS = "Animals & Nature"
    FOOD = "Food & Drink"
    TRAVEL = "Travel & Places"
    ACTIVITIES = "Activities"
    OBJECTS = "Objects"
    SYMBOLS = "Symbols"
    FLAGS = "Flags"


@dataclass
class Emoji:
    """An emoji with metadata."""
    char: str
    name: str
    keywords: list[str] = field(default_factory=list)
    category: EmojiCategory = EmojiCategory.SMILEYS
    
    def matches(self, query: str) -> bool:
        """Check if emoji matches search query."""
        query_lower = query.lower()
        if query_lower in self.name.lower():
            return True
        return any(query_lower in k.lower() for k in self.keywords)


class EmojiPicker:
    """
    Emoji selection UI with categories and search.
    
    Features:
    - Categorized emoji lists
    - Keyword search
    - Recent/favorite emojis
    - Skin tone variants
    """
    
    # Common emojis by category (subset for demonstration)
    DEFAULT_EMOJIS = {
        EmojiCategory.SMILEYS: [
            Emoji("😀", "grinning face", ["happy", "smile"]),
            Emoji("😂", "face with tears of joy", ["laugh", "lol", "funny"]),
            Emoji("🥹", "face holding back tears", ["emotional", "touched"]),
            Emoji("😊", "smiling face with smiling eyes", ["happy", "blush"]),
            Emoji("😍", "smiling face with heart-eyes", ["love", "heart"]),
            Emoji("🤔", "thinking face", ["think", "hmm", "consider"]),
            Emoji("😢", "crying face", ["sad", "cry", "tear"]),
            Emoji("😎", "smiling face with sunglasses", ["cool", "sunglasses"]),
            Emoji("🤯", "exploding head", ["mind blown", "shocked"]),
            Emoji("😴", "sleeping face", ["sleep", "tired", "zzz"]),
        ],
        EmojiCategory.PEOPLE: [
            Emoji("👍", "thumbs up", ["like", "yes", "ok", "approve"]),
            Emoji("👎", "thumbs down", ["dislike", "no", "disapprove"]),
            Emoji("👋", "waving hand", ["wave", "hello", "bye"]),
            Emoji("🤝", "handshake", ["deal", "agree", "partnership"]),
            Emoji("🙏", "folded hands", ["please", "pray", "thanks"]),
            Emoji("👏", "clapping hands", ["applause", "clap", "bravo"]),
            Emoji("💪", "flexed biceps", ["strong", "muscle", "power"]),
            Emoji("🧠", "brain", ["think", "smart", "intelligence"]),
        ],
        EmojiCategory.ANIMALS: [
            Emoji("🐶", "dog face", ["dog", "puppy", "pet"]),
            Emoji("🐱", "cat face", ["cat", "kitten", "pet"]),
            Emoji("🦊", "fox", ["fox", "firefox"]),
            Emoji("🐻", "bear", ["bear", "teddy"]),
            Emoji("🐼", "panda", ["panda", "cute"]),
            Emoji("🦁", "lion", ["lion", "king"]),
            Emoji("🐸", "frog", ["frog", "kermit"]),
            Emoji("🦄", "unicorn", ["unicorn", "magic"]),
        ],
        EmojiCategory.OBJECTS: [
            Emoji("💻", "laptop", ["computer", "laptop", "code"]),
            Emoji("📱", "mobile phone", ["phone", "mobile", "smartphone"]),
            Emoji("⌨️", "keyboard", ["keyboard", "type"]),
            Emoji("🖥️", "desktop computer", ["computer", "desktop", "monitor"]),
            Emoji("🔧", "wrench", ["tool", "fix", "repair"]),
            Emoji("⚙️", "gear", ["settings", "config", "cog"]),
            Emoji("💡", "light bulb", ["idea", "light", "bright"]),
            Emoji("📝", "memo", ["note", "write", "document"]),
            Emoji("📚", "books", ["book", "read", "library"]),
            Emoji("🎯", "direct hit", ["target", "goal", "bullseye"]),
        ],
        EmojiCategory.SYMBOLS: [
            Emoji("✅", "check mark button", ["done", "complete", "yes"]),
            Emoji("❌", "cross mark", ["no", "wrong", "delete"]),
            Emoji("⚠️", "warning", ["warning", "caution", "alert"]),
            Emoji("❗", "exclamation mark", ["important", "attention"]),
            Emoji("❓", "question mark", ["question", "help", "ask"]),
            Emoji("💯", "hundred points", ["perfect", "100", "score"]),
            Emoji("🔥", "fire", ["hot", "fire", "lit", "trending"]),
            Emoji("⭐", "star", ["star", "favorite", "best"]),
            Emoji("❤️", "red heart", ["love", "heart", "like"]),
            Emoji("🚀", "rocket", ["launch", "ship", "fast", "go"]),
        ],
    }
    
    def __init__(self, recent_limit: int = 20):
        """
        Initialize emoji picker.
        
        Args:
            recent_limit: Max recent emojis to store
        """
        self.recent_limit = recent_limit
        self._recent: list[Emoji] = []
        self._favorites: list[Emoji] = []
        self._emojis = self.DEFAULT_EMOJIS.copy()
    
    def get_by_category(self, category: EmojiCategory) -> list[Emoji]:
        """Get emojis by category."""
        return self._emojis.get(category, [])
    
    def get_categories(self) -> list[EmojiCategory]:
        """Get all available categories."""
        return list(self._emojis.keys())
    
    def get_recent(self) -> list[Emoji]:
        """Get recently used emojis."""
        return self._recent.copy()
    
    def get_favorites(self) -> list[Emoji]:
        """Get favorited emojis."""
        return self._favorites.copy()
    
    def search(self, query: str) -> list[Emoji]:
        """Search emojis by name/keywords."""
        if not query:
            return []
        
        results = []
        for emojis in self._emojis.values():
            for emoji in emojis:
                if emoji.matches(query):
                    results.append(emoji)
        
        return results
    
    def use(self, emoji: Emoji):
        """Mark emoji as used (adds to recent)."""
        # Remove if already in recent
        self._recent = [e for e in self._recent if e.char != emoji.char]
        # Add to front
        self._recent.insert(0, emoji)
        # Trim to limit
        self._recent = self._recent[:self.recent_limit]
    
    def toggle_favorite(self, emoji: Emoji):
        """Toggle favorite status."""
        for i, e in enumerate(self._favorites):
            if e.char == emoji.char:
                del self._favorites[i]
                return False
        self._favorites.append(emoji)
        return True
    
    def is_favorite(self, emoji: Emoji) -> bool:
        """Check if emoji is favorited."""
        return any(e.char == emoji.char for e in self._favorites)
    
    def add_custom_emoji(self, emoji: Emoji, category: EmojiCategory):
        """Add custom emoji to a category."""
        if category not in self._emojis:
            self._emojis[category] = []
        self._emojis[category].append(emoji)


# =============================================================================
# MESSAGE BOOKMARKS
# =============================================================================

@dataclass
class Bookmark:
    """A bookmarked message."""
    message_id: str
    conversation_id: str
    content_preview: str
    note: str = ""
    created_at: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    color: str = "#FFD700"  # Gold default
    
    def created_formatted(self) -> str:
        """Get formatted creation date."""
        dt = datetime.fromtimestamp(self.created_at)
        return dt.strftime("%Y-%m-%d %H:%M")


class BookmarkManager:
    """
    Message bookmarking with notes.
    
    Features:
    - Bookmark messages with custom notes
    - Tag and color-code bookmarks
    - Search and filter bookmarks
    - Export bookmarks
    """
    
    DEFAULT_COLORS = [
        "#FFD700",  # Gold
        "#FF6B6B",  # Red
        "#4ECDC4",  # Teal
        "#45B7D1",  # Blue
        "#96CEB4",  # Green
        "#FFEAA7",  # Yellow
        "#DDA0DD",  # Plum
        "#F0E68C",  # Khaki
    ]
    
    def __init__(self, persist_path: Optional[Path] = None):
        """
        Initialize bookmark manager.
        
        Args:
            persist_path: Path to save bookmarks
        """
        self.persist_path = persist_path
        self._bookmarks: dict[str, Bookmark] = {}  # message_id -> Bookmark
        self._callbacks: list[Callable[[str, Optional[Bookmark]], None]] = []
        
        if persist_path and persist_path.exists():
            self._load()
    
    def add(
        self,
        message_id: str,
        conversation_id: str,
        content: str,
        note: str = "",
        tags: Optional[list[str]] = None,
        color: Optional[str] = None
    ) -> Bookmark:
        """Add a bookmark."""
        bookmark = Bookmark(
            message_id=message_id,
            conversation_id=conversation_id,
            content_preview=content[:200],
            note=note,
            tags=tags or [],
            color=color or self.DEFAULT_COLORS[0]
        )
        
        self._bookmarks[message_id] = bookmark
        self._notify(message_id, bookmark)
        self._save()
        
        return bookmark
    
    def remove(self, message_id: str):
        """Remove a bookmark."""
        if message_id in self._bookmarks:
            del self._bookmarks[message_id]
            self._notify(message_id, None)
            self._save()
    
    def get(self, message_id: str) -> Optional[Bookmark]:
        """Get bookmark by message ID."""
        return self._bookmarks.get(message_id)
    
    def is_bookmarked(self, message_id: str) -> bool:
        """Check if message is bookmarked."""
        return message_id in self._bookmarks
    
    def get_all(self) -> list[Bookmark]:
        """Get all bookmarks sorted by date."""
        return sorted(
            self._bookmarks.values(),
            key=lambda b: b.created_at,
            reverse=True
        )
    
    def get_by_conversation(self, conversation_id: str) -> list[Bookmark]:
        """Get bookmarks for a conversation."""
        return [
            b for b in self._bookmarks.values()
            if b.conversation_id == conversation_id
        ]
    
    def get_by_tag(self, tag: str) -> list[Bookmark]:
        """Get bookmarks with a specific tag."""
        return [
            b for b in self._bookmarks.values()
            if tag in b.tags
        ]
    
    def search(self, query: str) -> list[Bookmark]:
        """Search bookmarks by content or note."""
        query_lower = query.lower()
        return [
            b for b in self._bookmarks.values()
            if query_lower in b.content_preview.lower()
            or query_lower in b.note.lower()
        ]
    
    def update_note(self, message_id: str, note: str):
        """Update bookmark note."""
        if message_id in self._bookmarks:
            self._bookmarks[message_id].note = note
            self._notify(message_id, self._bookmarks[message_id])
            self._save()
    
    def update_tags(self, message_id: str, tags: list[str]):
        """Update bookmark tags."""
        if message_id in self._bookmarks:
            self._bookmarks[message_id].tags = tags
            self._save()
    
    def update_color(self, message_id: str, color: str):
        """Update bookmark color."""
        if message_id in self._bookmarks:
            self._bookmarks[message_id].color = color
            self._save()
    
    def get_all_tags(self) -> list[str]:
        """Get all unique tags."""
        tags = set()
        for bookmark in self._bookmarks.values():
            tags.update(bookmark.tags)
        return sorted(tags)
    
    def on_change(self, callback: Callable[[str, Optional[Bookmark]], None]):
        """Register callback for bookmark changes."""
        self._callbacks.append(callback)
    
    def export_markdown(self) -> str:
        """Export bookmarks as markdown."""
        lines = ["# Bookmarks", ""]
        
        for bookmark in self.get_all():
            lines.append(f"## {bookmark.created_formatted()}")
            if bookmark.tags:
                lines.append(f"Tags: {', '.join(bookmark.tags)}")
            lines.append("")
            lines.append(f"> {bookmark.content_preview}")
            if bookmark.note:
                lines.append("")
                lines.append(f"**Note:** {bookmark.note}")
            lines.append("")
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)
    
    def _notify(self, message_id: str, bookmark: Optional[Bookmark]):
        """Notify callbacks."""
        for callback in self._callbacks:
            try:
                callback(message_id, bookmark)
            except Exception as e:
                logger.warning(f"Bookmark callback error: {e}")
    
    def _save(self):
        """Save to disk."""
        if not self.persist_path:
            return
        
        try:
            data = {
                msg_id: {
                    "message_id": b.message_id,
                    "conversation_id": b.conversation_id,
                    "content_preview": b.content_preview,
                    "note": b.note,
                    "created_at": b.created_at,
                    "tags": b.tags,
                    "color": b.color
                }
                for msg_id, b in self._bookmarks.items()
            }
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self.persist_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save bookmarks: {e}")
    
    def _load(self):
        """Load from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return
        
        try:
            data = json.loads(self.persist_path.read_text())
            for msg_id, d in data.items():
                self._bookmarks[msg_id] = Bookmark(
                    message_id=d["message_id"],
                    conversation_id=d["conversation_id"],
                    content_preview=d["content_preview"],
                    note=d.get("note", ""),
                    created_at=d.get("created_at", time.time()),
                    tags=d.get("tags", []),
                    color=d.get("color", "#FFD700")
                )
        except Exception as e:
            logger.warning(f"Failed to load bookmarks: {e}")


# =============================================================================
# TAB MANAGEMENT
# =============================================================================

@dataclass
class TabInfo:
    """Information about a tab."""
    id: str
    title: str
    tab_type: str  # "chat", "image", "code", etc.
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    color: Optional[str] = None
    icon: Optional[str] = None
    pinned: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class TabHistory:
    """
    Track tab history and recently closed tabs.
    
    Features:
    - Track recently closed tabs
    - Restore closed tabs
    - Tab access history
    """
    
    def __init__(self, max_history: int = 20):
        """
        Initialize tab history.
        
        Args:
            max_history: Max closed tabs to remember
        """
        self.max_history = max_history
        self._closed_tabs: list[TabInfo] = []
        self._access_history: list[str] = []  # Tab IDs
    
    def record_close(self, tab: TabInfo):
        """Record a tab being closed."""
        self._closed_tabs.insert(0, tab)
        self._closed_tabs = self._closed_tabs[:self.max_history]
    
    def get_closed_tabs(self) -> list[TabInfo]:
        """Get recently closed tabs."""
        return self._closed_tabs.copy()
    
    def pop_last_closed(self) -> Optional[TabInfo]:
        """Get and remove the most recently closed tab."""
        if self._closed_tabs:
            return self._closed_tabs.pop(0)
        return None
    
    def clear_history(self):
        """Clear closed tab history."""
        self._closed_tabs = []
    
    def record_access(self, tab_id: str):
        """Record tab access."""
        # Remove if already in history
        self._access_history = [t for t in self._access_history if t != tab_id]
        # Add to front
        self._access_history.insert(0, tab_id)
        # Limit size
        self._access_history = self._access_history[:50]
    
    def get_access_history(self) -> list[str]:
        """Get tab access history (most recent first)."""
        return self._access_history.copy()


class TabColorManager:
    """
    Manage custom tab colors.
    
    Features:
    - Set tab colors
    - Color presets
    - Auto-color by type
    """
    
    # Default colors by tab type
    TYPE_COLORS = {
        "chat": "#4A90D9",      # Blue
        "image": "#9B59B6",     # Purple
        "code": "#2ECC71",      # Green
        "audio": "#E74C3C",     # Red
        "video": "#F39C12",     # Orange
        "embeddings": "#1ABC9C", # Teal
        "settings": "#95A5A6",  # Gray
    }
    
    PRESET_COLORS = [
        "#E74C3C",  # Red
        "#E67E22",  # Orange
        "#F1C40F",  # Yellow
        "#2ECC71",  # Green
        "#1ABC9C",  # Teal
        "#3498DB",  # Blue
        "#9B59B6",  # Purple
        "#E91E63",  # Pink
        "#795548",  # Brown
        "#607D8B",  # Blue Gray
    ]
    
    def __init__(self):
        """Initialize tab color manager."""
        self._tab_colors: dict[str, str] = {}
    
    def set_color(self, tab_id: str, color: str):
        """Set custom color for a tab."""
        self._tab_colors[tab_id] = color
    
    def get_color(self, tab_id: str, tab_type: Optional[str] = None) -> str:
        """Get color for a tab."""
        if tab_id in self._tab_colors:
            return self._tab_colors[tab_id]
        if tab_type and tab_type in self.TYPE_COLORS:
            return self.TYPE_COLORS[tab_type]
        return "#808080"  # Default gray
    
    def clear_color(self, tab_id: str):
        """Remove custom color."""
        self._tab_colors.pop(tab_id, None)
    
    def get_presets(self) -> list[str]:
        """Get preset colors."""
        return self.PRESET_COLORS.copy()


class TabManager:
    """
    Combined tab management.
    
    Features:
    - Tab history tracking
    - Color coding
    - Pin tabs
    - Tab groups (future)
    """
    
    def __init__(self, max_history: int = 20):
        """
        Initialize tab manager.
        
        Args:
            max_history: Max closed tabs to remember
        """
        self.history = TabHistory(max_history)
        self.colors = TabColorManager()
        self._tabs: dict[str, TabInfo] = {}
        self._callbacks: list[Callable[[str, str], None]] = []  # (tab_id, event)
    
    def register_tab(self, tab: TabInfo):
        """Register a tab."""
        self._tabs[tab.id] = tab
        self._notify(tab.id, "registered")
    
    def unregister_tab(self, tab_id: str):
        """Unregister a tab (usually on close)."""
        if tab_id in self._tabs:
            tab = self._tabs.pop(tab_id)
            self.history.record_close(tab)
            self._notify(tab_id, "closed")
    
    def get_tab(self, tab_id: str) -> Optional[TabInfo]:
        """Get tab by ID."""
        return self._tabs.get(tab_id)
    
    def get_all_tabs(self) -> list[TabInfo]:
        """Get all registered tabs."""
        return list(self._tabs.values())
    
    def access_tab(self, tab_id: str):
        """Mark tab as accessed."""
        if tab_id in self._tabs:
            self._tabs[tab_id].last_accessed = time.time()
            self.history.record_access(tab_id)
            self._notify(tab_id, "accessed")
    
    def pin_tab(self, tab_id: str, pinned: bool = True):
        """Pin/unpin a tab."""
        if tab_id in self._tabs:
            self._tabs[tab_id].pinned = pinned
            self._notify(tab_id, "pinned" if pinned else "unpinned")
    
    def set_tab_color(self, tab_id: str, color: str):
        """Set tab color."""
        self.colors.set_color(tab_id, color)
        if tab_id in self._tabs:
            self._tabs[tab_id].color = color
            self._notify(tab_id, "color_changed")
    
    def restore_last_closed(self) -> Optional[TabInfo]:
        """Restore last closed tab."""
        tab = self.history.pop_last_closed()
        if tab:
            self._notify(tab.id, "restore_requested")
        return tab
    
    def on_change(self, callback: Callable[[str, str], None]):
        """Register callback for tab events."""
        self._callbacks.append(callback)
    
    def _notify(self, tab_id: str, event: str):
        """Notify callbacks."""
        for callback in self._callbacks:
            try:
                callback(tab_id, event)
            except Exception as e:
                logger.warning(f"Tab callback error: {e}")


# =============================================================================
# COMBINED UI UTILITIES
# =============================================================================

class UIUtilities:
    """
    Combined UI utilities manager.
    
    Combines:
    - Clipboard history
    - Emoji picker
    - Bookmark manager
    - Tab manager
    """
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        clipboard_max: int = 100,
        tab_history_max: int = 20
    ):
        """
        Initialize UI utilities.
        
        Args:
            data_dir: Directory for persistence
            clipboard_max: Max clipboard entries
            tab_history_max: Max closed tabs to remember
        """
        clipboard_path = data_dir / "clipboard_history.json" if data_dir else None
        bookmarks_path = data_dir / "bookmarks.json" if data_dir else None
        
        self.clipboard = ClipboardHistory(
            max_entries=clipboard_max,
            persist_path=clipboard_path
        )
        self.emoji = EmojiPicker()
        self.bookmarks = BookmarkManager(persist_path=bookmarks_path)
        self.tabs = TabManager(max_history=tab_history_max)


# Singleton
_ui_utilities: Optional[UIUtilities] = None


def get_ui_utilities(data_dir: Optional[Path] = None) -> UIUtilities:
    """Get or create UI utilities."""
    global _ui_utilities
    if _ui_utilities is None:
        _ui_utilities = UIUtilities(data_dir)
    return _ui_utilities
