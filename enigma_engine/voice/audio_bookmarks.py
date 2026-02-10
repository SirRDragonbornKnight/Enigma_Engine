"""
Audio Bookmarks for Enigma AI Engine

Mark important moments in voice chat.

Features:
- Timestamp marking
- Tag system
- Quick navigation
- Export/import
- Search functionality

Usage:
    from enigma_engine.voice.audio_bookmarks import AudioBookmarkManager
    
    manager = AudioBookmarkManager()
    
    # Add bookmark
    manager.add_bookmark("Important point", timestamp=120.5, tags=["key", "decision"])
    
    # Get bookmarks
    bookmarks = manager.get_bookmarks(session_id="abc123")
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class AudioBookmark:
    """A bookmark in audio/voice chat."""
    id: str
    session_id: str
    timestamp: float  # seconds into recording
    title: str
    description: str = ""
    tags: Set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    audio_file: str = ""
    transcript: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "title": self.title,
            "description": self.description,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "audio_file": self.audio_file,
            "transcript": self.transcript,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "AudioBookmark":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            timestamp=data["timestamp"],
            title=data["title"],
            description=data.get("description", ""),
            tags=set(data.get("tags", [])),
            created_at=data.get("created_at", time.time()),
            audio_file=data.get("audio_file", ""),
            transcript=data.get("transcript", ""),
            metadata=data.get("metadata", {})
        )


class AudioBookmarkManager:
    """Manage audio/voice chat bookmarks."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize bookmark manager.
        
        Args:
            storage_path: Path to store bookmarks
        """
        self.storage_path = storage_path or Path("data/bookmarks")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory bookmark storage
        self._bookmarks: Dict[str, AudioBookmark] = {}
        
        # Session tracking
        self._current_session: Optional[str] = None
        self._session_start: float = 0
        
        # Load existing bookmarks
        self._load_bookmarks()
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new bookmark session.
        
        Args:
            session_id: Optional session ID
            
        Returns:
            Session ID
        """
        self._current_session = session_id or str(uuid.uuid4())[:8]
        self._session_start = time.time()
        
        logger.info(f"Started bookmark session: {self._current_session}")
        return self._current_session
    
    def add_bookmark(
        self,
        title: str,
        timestamp: Optional[float] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        audio_file: str = "",
        transcript: str = ""
    ) -> AudioBookmark:
        """
        Add a bookmark.
        
        Args:
            title: Bookmark title
            timestamp: Timestamp in seconds (auto if None)
            description: Optional description
            tags: Optional tags
            session_id: Session ID (current if None)
            audio_file: Associated audio file
            transcript: Transcript at this point
            
        Returns:
            Created bookmark
        """
        bookmark_id = str(uuid.uuid4())[:8]
        session = session_id or self._current_session or "default"
        
        # Calculate timestamp relative to session start
        if timestamp is None:
            timestamp = time.time() - self._session_start if self._session_start else 0
        
        bookmark = AudioBookmark(
            id=bookmark_id,
            session_id=session,
            timestamp=timestamp,
            title=title,
            description=description,
            tags=set(tags) if tags else set(),
            audio_file=audio_file,
            transcript=transcript
        )
        
        self._bookmarks[bookmark_id] = bookmark
        self._save_bookmarks()
        
        logger.info(f"Added bookmark: {title} at {timestamp:.1f}s")
        return bookmark
    
    def quick_bookmark(self, label: str = "Quick Mark") -> AudioBookmark:
        """Add a quick bookmark at current time."""
        return self.add_bookmark(
            title=label,
            tags=["quick"]
        )
    
    def get_bookmark(self, bookmark_id: str) -> Optional[AudioBookmark]:
        """Get a bookmark by ID."""
        return self._bookmarks.get(bookmark_id)
    
    def get_bookmarks(
        self,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[AudioBookmark]:
        """
        Get bookmarks with optional filters.
        
        Args:
            session_id: Filter by session
            tags: Filter by tags (any match)
            start_time: Filter after this time
            end_time: Filter before this time
            
        Returns:
            List of matching bookmarks
        """
        results = list(self._bookmarks.values())
        
        # Filter by session
        if session_id:
            results = [b for b in results if b.session_id == session_id]
        
        # Filter by tags
        if tags:
            tag_set = set(tags)
            results = [b for b in results if b.tags & tag_set]
        
        # Filter by time range
        if start_time is not None:
            results = [b for b in results if b.timestamp >= start_time]
        
        if end_time is not None:
            results = [b for b in results if b.timestamp <= end_time]
        
        # Sort by timestamp
        results.sort(key=lambda b: b.timestamp)
        
        return results
    
    def search_bookmarks(self, query: str) -> List[AudioBookmark]:
        """
        Search bookmarks by title/description.
        
        Args:
            query: Search query
            
        Returns:
            Matching bookmarks
        """
        query_lower = query.lower()
        
        results = []
        for bookmark in self._bookmarks.values():
            if (query_lower in bookmark.title.lower() or
                query_lower in bookmark.description.lower() or
                query_lower in bookmark.transcript.lower()):
                results.append(bookmark)
        
        return sorted(results, key=lambda b: b.timestamp)
    
    def update_bookmark(
        self,
        bookmark_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[AudioBookmark]:
        """Update a bookmark."""
        bookmark = self._bookmarks.get(bookmark_id)
        
        if not bookmark:
            return None
        
        if title is not None:
            bookmark.title = title
        
        if description is not None:
            bookmark.description = description
        
        if tags is not None:
            bookmark.tags = set(tags)
        
        self._save_bookmarks()
        return bookmark
    
    def delete_bookmark(self, bookmark_id: str) -> bool:
        """Delete a bookmark."""
        if bookmark_id in self._bookmarks:
            del self._bookmarks[bookmark_id]
            self._save_bookmarks()
            return True
        return False
    
    def get_all_tags(self) -> Set[str]:
        """Get all unique tags."""
        tags = set()
        for bookmark in self._bookmarks.values():
            tags.update(bookmark.tags)
        return tags
    
    def get_sessions(self) -> List[str]:
        """Get all session IDs."""
        return list(set(b.session_id for b in self._bookmarks.values()))
    
    def export_bookmarks(
        self,
        output_path: Path,
        session_id: Optional[str] = None
    ) -> int:
        """
        Export bookmarks to file.
        
        Args:
            output_path: Output file path
            session_id: Filter by session
            
        Returns:
            Number of exported bookmarks
        """
        bookmarks = self.get_bookmarks(session_id=session_id)
        
        data = [b.to_dict() for b in bookmarks]
        
        output_path.write_text(json.dumps(data, indent=2))
        
        return len(bookmarks)
    
    def import_bookmarks(self, input_path: Path) -> int:
        """
        Import bookmarks from file.
        
        Args:
            input_path: Input file path
            
        Returns:
            Number of imported bookmarks
        """
        try:
            data = json.loads(input_path.read_text())
            
            count = 0
            for item in data:
                bookmark = AudioBookmark.from_dict(item)
                self._bookmarks[bookmark.id] = bookmark
                count += 1
            
            self._save_bookmarks()
            return count
            
        except Exception as e:
            logger.error(f"Failed to import bookmarks: {e}")
            return 0
    
    def generate_timeline(
        self,
        session_id: str,
        include_transcript: bool = False
    ) -> str:
        """
        Generate a text timeline of bookmarks.
        
        Args:
            session_id: Session ID
            include_transcript: Include transcript excerpts
            
        Returns:
            Formatted timeline
        """
        bookmarks = self.get_bookmarks(session_id=session_id)
        
        if not bookmarks:
            return "No bookmarks found."
        
        lines = [f"Timeline for session: {session_id}", "=" * 40]
        
        for bookmark in bookmarks:
            # Format timestamp as MM:SS
            minutes = int(bookmark.timestamp // 60)
            seconds = int(bookmark.timestamp % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"
            
            lines.append(f"[{time_str}] {bookmark.title}")
            
            if bookmark.description:
                lines.append(f"         {bookmark.description}")
            
            if bookmark.tags:
                lines.append(f"         Tags: {', '.join(bookmark.tags)}")
            
            if include_transcript and bookmark.transcript:
                lines.append(f"         \"{bookmark.transcript[:100]}...\"")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _save_bookmarks(self):
        """Save bookmarks to disk."""
        data = [b.to_dict() for b in self._bookmarks.values()]
        
        (self.storage_path / "bookmarks.json").write_text(
            json.dumps(data, indent=2)
        )
    
    def _load_bookmarks(self):
        """Load bookmarks from disk."""
        bookmarks_file = self.storage_path / "bookmarks.json"
        
        if bookmarks_file.exists():
            try:
                data = json.loads(bookmarks_file.read_text())
                
                for item in data:
                    bookmark = AudioBookmark.from_dict(item)
                    self._bookmarks[bookmark.id] = bookmark
                    
            except Exception as e:
                logger.error(f"Failed to load bookmarks: {e}")


# Global instance
_manager: Optional[AudioBookmarkManager] = None


def get_bookmark_manager() -> AudioBookmarkManager:
    """Get or create global bookmark manager."""
    global _manager
    if _manager is None:
        _manager = AudioBookmarkManager()
    return _manager


# Convenience functions
def add_bookmark(title: str, **kwargs) -> AudioBookmark:
    """Add a bookmark."""
    return get_bookmark_manager().add_bookmark(title, **kwargs)


def quick_mark(label: str = "Quick Mark") -> AudioBookmark:
    """Add a quick bookmark."""
    return get_bookmark_manager().quick_bookmark(label)
