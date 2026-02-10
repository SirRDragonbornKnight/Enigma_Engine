"""
Session Manager - Handle user sessions and offline support.

Features:
- Session state management
- Offline mode detection
- Session persistence
- Activity tracking
- Auto-save on idle

Part of the Enigma AI Engine core infrastructure.
"""

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# SESSION STATES
# =============================================================================

class SessionState(Enum):
    """Session state."""
    INACTIVE = "inactive"
    ACTIVE = "active"
    IDLE = "idle"
    OFFLINE = "offline"
    RECONNECTING = "reconnecting"


class ConnectionState(Enum):
    """Network connection state."""
    ONLINE = "online"
    OFFLINE = "offline"
    LIMITED = "limited"
    UNKNOWN = "unknown"


# =============================================================================
# SESSION DATA
# =============================================================================

@dataclass
class SessionData:
    """Data associated with a session."""
    session_id: str
    user_id: str = "default"
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_activity: str = field(default_factory=lambda: datetime.now().isoformat())
    state: SessionState = SessionState.INACTIVE
    connection: ConnectionState = ConnectionState.UNKNOWN
    data: dict[str, Any] = field(default_factory=dict)
    activity_count: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    
    def duration_seconds(self) -> float:
        """Get session duration in seconds."""
        try:
            started = datetime.fromisoformat(self.started_at)
            return (datetime.now() - started).total_seconds()
        except Exception:
            return 0.0
    
    def duration_formatted(self) -> str:
        """Get human-readable duration."""
        secs = self.duration_seconds()
        if secs < 60:
            return f"{int(secs)}s"
        elif secs < 3600:
            return f"{int(secs // 60)}m {int(secs % 60)}s"
        else:
            hours = int(secs // 3600)
            mins = int((secs % 3600) // 60)
            return f"{hours}h {mins}m"
    
    def idle_seconds(self) -> float:
        """Get seconds since last activity."""
        try:
            last = datetime.fromisoformat(self.last_activity)
            return (datetime.now() - last).total_seconds()
        except Exception:
            return 0.0
    
    def touch(self):
        """Update last activity time."""
        self.last_activity = datetime.now().isoformat()
        self.activity_count += 1
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "started_at": self.started_at,
            "last_activity": self.last_activity,
            "state": self.state.value,
            "connection": self.connection.value,
            "data": self.data,
            "activity_count": self.activity_count,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionData":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            user_id=data.get("user_id", "default"),
            started_at=data.get("started_at", datetime.now().isoformat()),
            last_activity=data.get("last_activity", datetime.now().isoformat()),
            state=SessionState(data.get("state", "inactive")),
            connection=ConnectionState(data.get("connection", "unknown")),
            data=data.get("data", {}),
            activity_count=data.get("activity_count", 0),
            messages_sent=data.get("messages_sent", 0),
            messages_received=data.get("messages_received", 0)
        )


# =============================================================================
# OFFLINE QUEUE
# =============================================================================

@dataclass
class QueuedAction:
    """An action queued for when online."""
    id: str
    action_type: str
    payload: dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    retries: int = 0
    max_retries: int = 3


class OfflineQueue:
    """
    Queue actions to execute when connection is restored.
    
    Features:
    - Persist queue to disk
    - Priority ordering
    - Retry handling
    - Callback on sync
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize offline queue."""
        self.storage_path = storage_path or Path("data/offline_queue.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._queue: list[QueuedAction] = []
        self._processing = False
        self._lock = threading.Lock()
        self._callbacks: dict[str, Callable[[dict], Any]] = {}
        
        self._load()
    
    def add(
        self,
        action_type: str,
        payload: dict[str, Any],
        priority: bool = False
    ) -> str:
        """
        Add action to queue.
        
        Args:
            action_type: Type of action
            payload: Action data
            priority: Add to front of queue
            
        Returns:
            Action ID
        """
        action_id = f"action_{int(time.time() * 1000)}_{len(self._queue)}"
        
        action = QueuedAction(
            id=action_id,
            action_type=action_type,
            payload=payload
        )
        
        with self._lock:
            if priority:
                self._queue.insert(0, action)
            else:
                self._queue.append(action)
            self._save()
        
        return action_id
    
    def register_handler(self, action_type: str, handler: Callable[[dict], Any]):
        """Register handler for action type."""
        self._callbacks[action_type] = handler
    
    def process(self) -> int:
        """
        Process queued actions.
        
        Returns:
            Number of actions processed
        """
        if self._processing:
            return 0
        
        self._processing = True
        processed = 0
        
        try:
            with self._lock:
                remaining = []
                
                for action in self._queue:
                    handler = self._callbacks.get(action.action_type)
                    
                    if not handler:
                        logger.warning(f"No handler for action type: {action.action_type}")
                        remaining.append(action)
                        continue
                    
                    try:
                        handler(action.payload)
                        processed += 1
                    except Exception as e:
                        logger.error(f"Queue action failed: {e}")
                        action.retries += 1
                        
                        if action.retries < action.max_retries:
                            remaining.append(action)
                
                self._queue = remaining
                self._save()
                
        finally:
            self._processing = False
        
        return processed
    
    def clear(self):
        """Clear the queue."""
        with self._lock:
            self._queue = []
            self._save()
    
    def count(self) -> int:
        """Get queue size."""
        return len(self._queue)
    
    def _save(self):
        """Save queue to disk."""
        try:
            data = [
                {
                    "id": a.id,
                    "action_type": a.action_type,
                    "payload": a.payload,
                    "created_at": a.created_at,
                    "retries": a.retries
                }
                for a in self._queue
            ]
            
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save offline queue: {e}")
    
    def _load(self):
        """Load queue from disk."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
            
            self._queue = [
                QueuedAction(
                    id=item["id"],
                    action_type=item["action_type"],
                    payload=item["payload"],
                    created_at=item.get("created_at", datetime.now().isoformat()),
                    retries=item.get("retries", 0)
                )
                for item in data
            ]
            
        except Exception as e:
            logger.error(f"Failed to load offline queue: {e}")


# =============================================================================
# SESSION MANAGER
# =============================================================================

class SessionManager:
    """
    Manage user sessions with offline support.
    
    Features:
    - Session lifecycle management
    - Offline mode detection
    - Auto-save state
    - Activity tracking
    - Session restoration
    """
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        idle_timeout: int = 300,  # 5 minutes
        auto_save_interval: int = 60  # 1 minute
    ):
        """
        Initialize session manager.
        
        Args:
            storage_path: Path to session storage
            idle_timeout: Seconds until idle state
            auto_save_interval: Auto-save interval in seconds
        """
        self.storage_path = storage_path or Path("data/sessions")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.idle_timeout = idle_timeout
        self.auto_save_interval = auto_save_interval
        
        self._current_session: Optional[SessionData] = None
        self._offline_queue = OfflineQueue()
        self._lock = threading.Lock()
        self._callbacks: dict[str, list[Callable]] = {
            "state_change": [],
            "connection_change": [],
            "idle": [],
            "activity": []
        }
        
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def start_session(self, user_id: str = "default") -> SessionData:
        """
        Start a new session.
        
        Args:
            user_id: User identifier
            
        Returns:
            New session data
        """
        session_id = self._generate_session_id(user_id)
        
        session = SessionData(
            session_id=session_id,
            user_id=user_id,
            state=SessionState.ACTIVE,
            connection=self._check_connection()
        )
        
        with self._lock:
            # End previous session if exists
            if self._current_session:
                self._save_session(self._current_session)
            
            self._current_session = session
            self._save_session(session)
        
        # Start monitoring
        self._start_monitor()
        
        self._notify("state_change", session)
        return session
    
    def end_session(self):
        """End current session."""
        with self._lock:
            if self._current_session:
                self._current_session.state = SessionState.INACTIVE
                self._save_session(self._current_session)
                self._notify("state_change", self._current_session)
                self._current_session = None
        
        self._stop_monitor()
    
    def get_session(self) -> Optional[SessionData]:
        """Get current session."""
        return self._current_session
    
    def record_activity(self, activity_type: str = "general"):
        """Record user activity."""
        with self._lock:
            if self._current_session:
                self._current_session.touch()
                
                if self._current_session.state == SessionState.IDLE:
                    self._current_session.state = SessionState.ACTIVE
                    self._notify("state_change", self._current_session)
                
                self._notify("activity", {
                    "session": self._current_session,
                    "type": activity_type
                })
    
    def record_message(self, direction: str = "sent"):
        """Record message sent/received."""
        with self._lock:
            if self._current_session:
                self.record_activity("message")
                
                if direction == "sent":
                    self._current_session.messages_sent += 1
                else:
                    self._current_session.messages_received += 1
    
    def set_data(self, key: str, value: Any):
        """Set session data."""
        with self._lock:
            if self._current_session:
                self._current_session.data[key] = value
    
    def get_data(self, key: str, default: Any = None) -> Any:
        """Get session data."""
        if self._current_session:
            return self._current_session.data.get(key, default)
        return default
    
    def go_offline(self):
        """Manually set offline mode."""
        with self._lock:
            if self._current_session:
                self._current_session.connection = ConnectionState.OFFLINE
                self._current_session.state = SessionState.OFFLINE
                self._notify("connection_change", self._current_session)
    
    def go_online(self):
        """Manually set online mode and process queue."""
        with self._lock:
            if self._current_session:
                self._current_session.connection = ConnectionState.ONLINE
                self._current_session.state = SessionState.ACTIVE
                self._notify("connection_change", self._current_session)
        
        # Process offline queue
        processed = self._offline_queue.process()
        if processed > 0:
            logger.info(f"Processed {processed} queued actions")
    
    def queue_action(self, action_type: str, payload: dict[str, Any]) -> str:
        """Queue action for offline processing."""
        return self._offline_queue.add(action_type, payload)
    
    def register_queue_handler(self, action_type: str, handler: Callable):
        """Register handler for queued actions."""
        self._offline_queue.register_handler(action_type, handler)
    
    def get_queued_count(self) -> int:
        """Get number of queued actions."""
        return self._offline_queue.count()
    
    def on(self, event: str, callback: Callable):
        """Register event callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def restore_session(self, session_id: str) -> Optional[SessionData]:
        """
        Restore a previous session.
        
        Args:
            session_id: Session ID to restore
            
        Returns:
            Restored session or None
        """
        file_path = self.storage_path / f"{session_id}.json"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            session = SessionData.from_dict(data)
            session.state = SessionState.ACTIVE
            session.touch()
            
            with self._lock:
                if self._current_session:
                    self._save_session(self._current_session)
                self._current_session = session
            
            self._start_monitor()
            return session
            
        except Exception as e:
            logger.error(f"Failed to restore session: {e}")
            return None
    
    def list_sessions(self, user_id: Optional[str] = None) -> list[dict[str, Any]]:
        """
        List saved sessions.
        
        Args:
            user_id: Filter by user ID
            
        Returns:
            List of session summaries
        """
        sessions = []
        
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                
                if user_id and data.get("user_id") != user_id:
                    continue
                
                sessions.append({
                    "session_id": data["session_id"],
                    "user_id": data.get("user_id", "default"),
                    "started_at": data.get("started_at"),
                    "last_activity": data.get("last_activity"),
                    "messages_sent": data.get("messages_sent", 0),
                    "messages_received": data.get("messages_received", 0)
                })
                
            except Exception as e:
                logger.warning(f"Failed to load session {file_path}: {e}")
        
        # Sort by last activity
        sessions.sort(key=lambda s: s.get("last_activity", ""), reverse=True)
        return sessions
    
    def _generate_session_id(self, user_id: str) -> str:
        """Generate unique session ID."""
        timestamp = int(time.time() * 1000)
        data = f"{user_id}_{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _check_connection(self) -> ConnectionState:
        """Check network connection status."""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return ConnectionState.ONLINE
        except OSError:
            return ConnectionState.OFFLINE
    
    def _save_session(self, session: SessionData):
        """Save session to disk."""
        try:
            file_path = self.storage_path / f"{session.session_id}.json"
            with open(file_path, "w") as f:
                json.dump(session.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
    
    def _start_monitor(self):
        """Start background monitoring thread."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
    
    def _stop_monitor(self):
        """Stop background monitoring."""
        self._running = False
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        last_save = time.time()
        
        while self._running:
            try:
                with self._lock:
                    if self._current_session:
                        # Check for idle
                        idle_secs = self._current_session.idle_seconds()
                        
                        if (self._current_session.state == SessionState.ACTIVE and 
                            idle_secs >= self.idle_timeout):
                            self._current_session.state = SessionState.IDLE
                            self._notify("idle", self._current_session)
                        
                        # Check connection
                        new_conn = self._check_connection()
                        if new_conn != self._current_session.connection:
                            old_conn = self._current_session.connection
                            self._current_session.connection = new_conn
                            
                            if new_conn == ConnectionState.ONLINE:
                                self._current_session.state = SessionState.ACTIVE
                                # Process offline queue
                                self._offline_queue.process()
                            elif new_conn == ConnectionState.OFFLINE:
                                self._current_session.state = SessionState.OFFLINE
                            
                            self._notify("connection_change", self._current_session)
                        
                        # Auto-save
                        if time.time() - last_save >= self.auto_save_interval:
                            self._save_session(self._current_session)
                            last_save = time.time()
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
            
            time.sleep(5)  # Check every 5 seconds
    
    def _notify(self, event: str, data: Any):
        """Notify event callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.warning(f"Session callback error: {e}")


# =============================================================================
# SINGLETON
# =============================================================================

_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get or create session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def start_session(user_id: str = "default") -> SessionData:
    """Quick function to start a session."""
    return get_session_manager().start_session(user_id)


def end_session():
    """Quick function to end current session."""
    get_session_manager().end_session()


def record_activity():
    """Quick function to record activity."""
    get_session_manager().record_activity()
