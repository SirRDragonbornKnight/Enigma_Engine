"""
Gaming Session Stats for Enigma AI Engine

Track gaming time, achievements, and progress.

Usage:
    from enigma_engine.tools.game_stats import GameSessionTracker, get_session_tracker
    
    tracker = get_session_tracker()
    
    # Start a session
    session_id = tracker.start_session("Minecraft")
    
    # Log events
    tracker.log_event(session_id, "death", {"cause": "creeper"})
    tracker.log_event(session_id, "achievement", {"name": "Getting Wood"})
    
    # End session
    tracker.end_session(session_id)
    
    # Get stats
    stats = tracker.get_game_stats("Minecraft")
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of gaming events."""
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    DEATH = "death"
    KILL = "kill"
    ACHIEVEMENT = "achievement"
    LEVEL_UP = "level_up"
    ITEM_ACQUIRED = "item_acquired"
    QUEST_COMPLETE = "quest_complete"
    BOSS_DEFEATED = "boss_defeated"
    SAVE = "save"
    CUSTOM = "custom"


@dataclass
class GameEvent:
    """A single gaming event."""
    timestamp: str
    event_type: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class GameSession:
    """A gaming session record."""
    id: str
    game: str
    started_at: str
    ended_at: Optional[str] = None
    duration_minutes: float = 0.0
    events: List[GameEvent] = field(default_factory=list)
    
    # Quick stats
    deaths: int = 0
    kills: int = 0
    achievements: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "game": self.game,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_minutes": self.duration_minutes,
            "events": [asdict(e) for e in self.events],
            "deaths": self.deaths,
            "kills": self.kills,
            "achievements": self.achievements
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameSession':
        """Create from dictionary."""
        events = [GameEvent(**e) for e in data.get("events", [])]
        data["events"] = events
        return cls(**data)


@dataclass
class GameStats:
    """Aggregate stats for a game."""
    game: str
    total_sessions: int = 0
    total_playtime_hours: float = 0.0
    total_deaths: int = 0
    total_kills: int = 0
    total_achievements: int = 0
    first_played: Optional[str] = None
    last_played: Optional[str] = None
    average_session_length_minutes: float = 0.0
    
    # Achievement list
    achievements_unlocked: List[str] = field(default_factory=list)


class GameSessionTracker:
    """
    Track gaming sessions and statistics.
    
    Features:
    - Session timing
    - Event logging
    - Per-game statistics
    - Achievement tracking
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        from ..config import CONFIG
        
        self.storage_path = storage_path or (
            Path(CONFIG.get("data_dir", "data")) / "game_sessions"
        )
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._active_sessions: Dict[str, GameSession] = {}
        self._load_sessions()
    
    def _get_session_file(self, game: str) -> Path:
        """Get session file path for a game."""
        safe_name = "".join(c for c in game if c.isalnum() or c in "._- ").strip()
        return self.storage_path / f"{safe_name}_sessions.json"
    
    def _load_sessions(self):
        """Load all session data."""
        # Sessions are loaded per-game as needed
    
    def _load_game_sessions(self, game: str) -> List[GameSession]:
        """Load sessions for a specific game."""
        session_file = self._get_session_file(game)
        
        if not session_file.exists():
            return []
        
        try:
            with open(session_file) as f:
                data = json.load(f)
            return [GameSession.from_dict(s) for s in data.get("sessions", [])]
        except Exception as e:
            logger.warning(f"Could not load sessions for {game}: {e}")
            return []
    
    def _save_game_sessions(self, game: str, sessions: List[GameSession]):
        """Save sessions for a game."""
        session_file = self._get_session_file(game)
        
        try:
            with open(session_file, 'w') as f:
                json.dump({
                    "game": game,
                    "sessions": [s.to_dict() for s in sessions]
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save sessions for {game}: {e}")
    
    def start_session(self, game: str) -> str:
        """
        Start a new gaming session.
        
        Args:
            game: Name of the game
            
        Returns:
            Session ID
        """
        import uuid
        session_id = str(uuid.uuid4())[:8]
        
        session = GameSession(
            id=session_id,
            game=game,
            started_at=datetime.now().isoformat()
        )
        
        session.events.append(GameEvent(
            timestamp=datetime.now().isoformat(),
            event_type=EventType.SESSION_START.value
        ))
        
        self._active_sessions[session_id] = session
        logger.info(f"Started gaming session {session_id} for {game}")
        
        return session_id
    
    def end_session(self, session_id: str) -> Optional[GameSession]:
        """
        End a gaming session.
        
        Args:
            session_id: Session to end
            
        Returns:
            Completed session or None if not found
        """
        session = self._active_sessions.pop(session_id, None)
        
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return None
        
        # Finalize session
        session.ended_at = datetime.now().isoformat()
        
        # Calculate duration
        started = datetime.fromisoformat(session.started_at)
        ended = datetime.fromisoformat(session.ended_at)
        session.duration_minutes = (ended - started).total_seconds() / 60
        
        session.events.append(GameEvent(
            timestamp=datetime.now().isoformat(),
            event_type=EventType.SESSION_END.value,
            data={"duration_minutes": session.duration_minutes}
        ))
        
        # Save to storage
        sessions = self._load_game_sessions(session.game)
        sessions.append(session)
        self._save_game_sessions(session.game, sessions)
        
        logger.info(f"Ended session {session_id}: {session.duration_minutes:.1f} minutes")
        return session
    
    def log_event(
        self,
        session_id: str,
        event_type: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """
        Log an event in a session.
        
        Args:
            session_id: Active session ID
            event_type: Type of event (death, kill, achievement, etc.)
            data: Additional event data
        """
        session = self._active_sessions.get(session_id)
        if not session:
            logger.warning(f"Cannot log event: session {session_id} not found")
            return
        
        event = GameEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            data=data or {}
        )
        session.events.append(event)
        
        # Update quick stats
        if event_type == "death" or event_type == EventType.DEATH.value:
            session.deaths += 1
        elif event_type == "kill" or event_type == EventType.KILL.value:
            session.kills += 1
        elif event_type == "achievement" or event_type == EventType.ACHIEVEMENT.value:
            session.achievements += 1
        
        logger.debug(f"Logged {event_type} in session {session_id}")
    
    def get_active_session(self, session_id: str) -> Optional[GameSession]:
        """Get an active session."""
        return self._active_sessions.get(session_id)
    
    def get_game_stats(self, game: str) -> GameStats:
        """
        Get aggregate statistics for a game.
        
        Args:
            game: Game name
            
        Returns:
            Aggregate stats
        """
        sessions = self._load_game_sessions(game)
        
        if not sessions:
            return GameStats(game=game)
        
        total_time = sum(s.duration_minutes for s in sessions) / 60  # hours
        total_deaths = sum(s.deaths for s in sessions)
        total_kills = sum(s.kills for s in sessions)
        total_achievements = sum(s.achievements for s in sessions)
        
        # Collect achievement names
        achievements = set()
        for session in sessions:
            for event in session.events:
                if event.event_type == "achievement":
                    name = event.data.get("name")
                    if name:
                        achievements.add(name)
        
        avg_session = total_time * 60 / len(sessions) if sessions else 0
        
        stats = GameStats(
            game=game,
            total_sessions=len(sessions),
            total_playtime_hours=total_time,
            total_deaths=total_deaths,
            total_kills=total_kills,
            total_achievements=total_achievements,
            first_played=sessions[0].started_at if sessions else None,
            last_played=sessions[-1].started_at if sessions else None,
            average_session_length_minutes=avg_session,
            achievements_unlocked=sorted(achievements)
        )
        
        return stats
    
    def get_all_games(self) -> List[str]:
        """Get list of all tracked games."""
        games = []
        for file in self.storage_path.glob("*_sessions.json"):
            try:
                with open(file) as f:
                    data = json.load(f)
                    games.append(data.get("game", file.stem))
            except Exception:
                pass
        return sorted(set(games))
    
    def get_recent_sessions(self, limit: int = 10) -> List[GameSession]:
        """Get most recent sessions across all games."""
        all_sessions = []
        
        for game in self.get_all_games():
            sessions = self._load_game_sessions(game)
            all_sessions.extend(sessions)
        
        all_sessions.sort(key=lambda s: s.started_at, reverse=True)
        return all_sessions[:limit]
    
    def get_total_playtime_hours(self) -> float:
        """Get total playtime across all games."""
        total = 0.0
        for game in self.get_all_games():
            stats = self.get_game_stats(game)
            total += stats.total_playtime_hours
        return total
    
    def get_summary(self) -> Dict[str, Any]:
        """Get overall gaming summary."""
        games = self.get_all_games()
        total_hours = self.get_total_playtime_hours()
        
        # Most played game
        most_played = None
        most_hours = 0
        for game in games:
            stats = self.get_game_stats(game)
            if stats.total_playtime_hours > most_hours:
                most_hours = stats.total_playtime_hours
                most_played = game
        
        recent = self.get_recent_sessions(5)
        
        return {
            "total_games": len(games),
            "total_playtime_hours": total_hours,
            "most_played": most_played,
            "most_played_hours": most_hours,
            "recent_sessions": [
                {"game": s.game, "date": s.started_at, "minutes": s.duration_minutes}
                for s in recent
            ]
        }


# Global instance
_session_tracker: Optional[GameSessionTracker] = None


def get_session_tracker() -> GameSessionTracker:
    """Get or create global session tracker."""
    global _session_tracker
    if _session_tracker is None:
        _session_tracker = GameSessionTracker()
    return _session_tracker
