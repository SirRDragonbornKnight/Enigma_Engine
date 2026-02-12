"""
Cross-Session Context for Enigma AI Engine

Maintain context and memory across conversation sessions.

Features:
- Session state persistence
- Context summarization
- Memory retrieval
- Session linking
- Topic continuity

Usage:
    from enigma_engine.memory.cross_session import CrossSessionContext
    
    ctx = CrossSessionContext()
    
    # Save session context
    ctx.end_session(session_id="123", summary="Discussed Python")
    
    # Start new session with context
    context = ctx.start_session()
    print(context.previous_topics)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class SessionSummary:
    """Summary of a conversation session."""
    session_id: str
    start_time: float
    end_time: float
    topics: List[str]
    key_facts: List[str]
    user_preferences: Dict[str, Any]
    message_count: int
    summary_text: str


@dataclass
class UserProfile:
    """Long-term user profile."""
    user_id: str
    name: str = ""
    preferences: Dict[str, Any] = field(default_factory=dict)
    interests: Set[str] = field(default_factory=set)
    facts: Dict[str, str] = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


@dataclass
class SessionContext:
    """Context for starting a new session."""
    user_id: str
    user_name: str
    previous_sessions: List[SessionSummary]
    previous_topics: List[str]
    user_preferences: Dict[str, Any]
    relevant_facts: List[str]
    last_session_summary: str
    context_prompt: str


class CrossSessionContext:
    """Manage context across conversation sessions."""
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize cross-session context.
        
        Args:
            storage_dir: Directory to store session data
        """
        self.storage_dir = storage_dir or Path("memory/sessions")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Current session
        self._current_session_id: Optional[str] = None
        self._session_start: float = 0
        self._session_messages: List[Dict] = []
        self._session_topics: Set[str] = set()
        
        # User profiles
        self._users: Dict[str, UserProfile] = {}
        self._current_user: Optional[str] = None
        
        # Session index
        self._session_index: Dict[str, List[str]] = {}  # user_id -> session_ids
        
        # Load existing data
        self._load_data()
    
    def set_user(self, user_id: str, name: str = ""):
        """
        Set current user.
        
        Args:
            user_id: User identifier
            name: User display name
        """
        self._current_user = user_id
        
        if user_id not in self._users:
            self._users[user_id] = UserProfile(
                user_id=user_id,
                name=name
            )
        else:
            self._users[user_id].last_seen = time.time()
            if name:
                self._users[user_id].name = name
        
        if user_id not in self._session_index:
            self._session_index[user_id] = []
    
    def start_session(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> SessionContext:
        """
        Start a new session with retrieved context.
        
        Args:
            user_id: User identifier
            session_id: Optional session ID
            
        Returns:
            Context for the new session
        """
        if user_id:
            self.set_user(user_id)
        
        user_id = self._current_user or "default"
        
        # Generate session ID
        self._current_session_id = session_id or f"session_{int(time.time())}"
        self._session_start = time.time()
        self._session_messages = []
        self._session_topics = set()
        
        # Get previous sessions for this user
        previous_summaries = self._get_user_sessions(user_id, limit=5)
        
        # Extract topics from recent sessions
        recent_topics = []
        for summary in previous_summaries:
            recent_topics.extend(summary.topics[:3])
        recent_topics = list(dict.fromkeys(recent_topics))[:10]  # Dedupe, limit
        
        # Get user profile
        user = self._users.get(user_id, UserProfile(user_id=user_id))
        
        # Get relevant facts
        relevant_facts = list(user.facts.values())[:10]
        
        # Build last session summary
        last_summary = ""
        if previous_summaries:
            last = previous_summaries[0]
            last_summary = last.summary_text
        
        # Build context prompt
        context_prompt = self._build_context_prompt(
            user=user,
            previous_summaries=previous_summaries,
            recent_topics=recent_topics
        )
        
        return SessionContext(
            user_id=user_id,
            user_name=user.name,
            previous_sessions=previous_summaries,
            previous_topics=recent_topics,
            user_preferences=user.preferences,
            relevant_facts=relevant_facts,
            last_session_summary=last_summary,
            context_prompt=context_prompt
        )
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ):
        """
        Add a message to the current session.
        
        Args:
            role: Message role (user/assistant)
            content: Message content
            metadata: Additional metadata
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        self._session_messages.append(message)
        
        # Extract topics (simple keyword extraction)
        words = content.lower().split()
        for word in words:
            if len(word) > 5 and word.isalpha():
                self._session_topics.add(word)
    
    def add_topic(self, topic: str):
        """Add a topic to current session."""
        self._session_topics.add(topic.lower())
    
    def learn_fact(self, key: str, value: str):
        """
        Learn a fact about the user.
        
        Args:
            key: Fact key (e.g., "favorite_color")
            value: Fact value
        """
        user_id = self._current_user or "default"
        
        if user_id not in self._users:
            self._users[user_id] = UserProfile(user_id=user_id)
        
        self._users[user_id].facts[key] = value
    
    def update_preference(self, key: str, value: Any):
        """
        Update a user preference.
        
        Args:
            key: Preference key
            value: Preference value
        """
        user_id = self._current_user or "default"
        
        if user_id not in self._users:
            self._users[user_id] = UserProfile(user_id=user_id)
        
        self._users[user_id].preferences[key] = value
    
    def end_session(
        self,
        summary_text: Optional[str] = None,
        key_facts: Optional[List[str]] = None
    ) -> SessionSummary:
        """
        End the current session and save summary.
        
        Args:
            summary_text: Optional manual summary
            key_facts: Optional key facts from session
            
        Returns:
            Session summary
        """
        session_id = self._current_session_id or f"session_{int(time.time())}"
        user_id = self._current_user or "default"
        
        # Generate summary if not provided
        if not summary_text:
            summary_text = self._generate_summary()
        
        # Create summary
        summary = SessionSummary(
            session_id=session_id,
            start_time=self._session_start,
            end_time=time.time(),
            topics=list(self._session_topics)[:20],
            key_facts=key_facts or [],
            user_preferences=self._users.get(user_id, UserProfile(user_id=user_id)).preferences,
            message_count=len(self._session_messages),
            summary_text=summary_text
        )
        
        # Save to disk
        self._save_session(user_id, summary)
        
        # Update index
        if user_id not in self._session_index:
            self._session_index[user_id] = []
        self._session_index[user_id].insert(0, session_id)
        
        # Reset state
        self._current_session_id = None
        self._session_messages = []
        self._session_topics = set()
        
        # Save data
        self._save_data()
        
        return summary
    
    def get_user_facts(self, user_id: Optional[str] = None) -> Dict[str, str]:
        """Get all known facts about a user."""
        user_id = user_id or self._current_user or "default"
        user = self._users.get(user_id)
        return user.facts if user else {}
    
    def get_user_preferences(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get user preferences."""
        user_id = user_id or self._current_user or "default"
        user = self._users.get(user_id)
        return user.preferences if user else {}
    
    def get_recent_topics(
        self,
        user_id: Optional[str] = None,
        limit: int = 20
    ) -> List[str]:
        """Get recently discussed topics."""
        user_id = user_id or self._current_user or "default"
        sessions = self._get_user_sessions(user_id, limit=5)
        
        topics = []
        for session in sessions:
            topics.extend(session.topics)
        
        return list(dict.fromkeys(topics))[:limit]
    
    def _get_user_sessions(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[SessionSummary]:
        """Get user's recent sessions."""
        session_ids = self._session_index.get(user_id, [])[:limit]
        
        summaries = []
        for session_id in session_ids:
            summary = self._load_session(user_id, session_id)
            if summary:
                summaries.append(summary)
        
        return summaries
    
    def _generate_summary(self) -> str:
        """Generate automatic session summary."""
        if not self._session_messages:
            return "Empty session"
        
        # Simple summary: topics discussed
        topics = list(self._session_topics)[:5]
        message_count = len(self._session_messages)
        
        if topics:
            return f"Discussed {', '.join(topics)} ({message_count} messages)"
        else:
            return f"Conversation with {message_count} messages"
    
    def _build_context_prompt(
        self,
        user: UserProfile,
        previous_summaries: List[SessionSummary],
        recent_topics: List[str]
    ) -> str:
        """Build context prompt for new session."""
        parts = []
        
        # User info
        if user.name:
            parts.append(f"User: {user.name}")
        
        # Previous session
        if previous_summaries:
            last = previous_summaries[0]
            parts.append(f"Previous session: {last.summary_text}")
        
        # Topics
        if recent_topics:
            parts.append(f"Recent topics: {', '.join(recent_topics[:5])}")
        
        # Facts
        if user.facts:
            facts = [f"{k}: {v}" for k, v in list(user.facts.items())[:5]]
            parts.append(f"Known facts: {'; '.join(facts)}")
        
        # Preferences
        if user.preferences:
            prefs = [f"{k}: {v}" for k, v in list(user.preferences.items())[:3]]
            parts.append(f"Preferences: {'; '.join(prefs)}")
        
        return "\n".join(parts) if parts else ""
    
    def _save_session(self, user_id: str, summary: SessionSummary):
        """Save session summary to disk."""
        user_dir = self.storage_dir / user_id
        user_dir.mkdir(exist_ok=True)
        
        session_file = user_dir / f"{summary.session_id}.json"
        
        data = {
            "session_id": summary.session_id,
            "start_time": summary.start_time,
            "end_time": summary.end_time,
            "topics": summary.topics,
            "key_facts": summary.key_facts,
            "user_preferences": summary.user_preferences,
            "message_count": summary.message_count,
            "summary_text": summary.summary_text
        }
        
        session_file.write_text(json.dumps(data, indent=2))
    
    def _load_session(self, user_id: str, session_id: str) -> Optional[SessionSummary]:
        """Load session summary from disk."""
        session_file = self.storage_dir / user_id / f"{session_id}.json"
        
        if not session_file.exists():
            return None
        
        try:
            data = json.loads(session_file.read_text())
            
            return SessionSummary(
                session_id=data["session_id"],
                start_time=data["start_time"],
                end_time=data["end_time"],
                topics=data.get("topics", []),
                key_facts=data.get("key_facts", []),
                user_preferences=data.get("user_preferences", {}),
                message_count=data.get("message_count", 0),
                summary_text=data.get("summary_text", "")
            )
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return None
    
    def _save_data(self):
        """Save index and user profiles."""
        # Save index
        index_file = self.storage_dir / "session_index.json"
        index_file.write_text(json.dumps(self._session_index, indent=2))
        
        # Save user profiles
        users_file = self.storage_dir / "user_profiles.json"
        users_data = {}
        for user_id, user in self._users.items():
            users_data[user_id] = {
                "user_id": user.user_id,
                "name": user.name,
                "preferences": user.preferences,
                "interests": list(user.interests),
                "facts": user.facts,
                "created": user.created,
                "last_seen": user.last_seen
            }
        users_file.write_text(json.dumps(users_data, indent=2))
    
    def _load_data(self):
        """Load index and user profiles."""
        # Load index
        index_file = self.storage_dir / "session_index.json"
        if index_file.exists():
            try:
                self._session_index = json.loads(index_file.read_text())
            except Exception:
                self._session_index = {}
        
        # Load user profiles
        users_file = self.storage_dir / "user_profiles.json"
        if users_file.exists():
            try:
                users_data = json.loads(users_file.read_text())
                for user_id, data in users_data.items():
                    self._users[user_id] = UserProfile(
                        user_id=data["user_id"],
                        name=data.get("name", ""),
                        preferences=data.get("preferences", {}),
                        interests=set(data.get("interests", [])),
                        facts=data.get("facts", {}),
                        created=data.get("created", time.time()),
                        last_seen=data.get("last_seen", time.time())
                    )
            except Exception:
                pass  # Intentionally silent


# Global instance
_cross_session: Optional[CrossSessionContext] = None


def get_cross_session_context() -> CrossSessionContext:
    """Get or create global cross-session context."""
    global _cross_session
    if _cross_session is None:
        _cross_session = CrossSessionContext()
    return _cross_session
