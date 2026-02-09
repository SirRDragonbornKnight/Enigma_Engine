"""
Player Statistics Tracking

Track player performance, sessions, and statistics across games.
Provides analytics, trends, and improvement suggestions.

FILE: enigma_engine/game/stats.py
TYPE: Game
MAIN CLASSES: PlayerStats, SessionTracker, StatsAnalyzer
"""

import json
import logging
import sqlite3
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatType(Enum):
    """Types of statistics."""
    COUNTER = "counter"       # Incrementing count
    GAUGE = "gauge"           # Current value
    RATE = "rate"             # Per-unit rate
    DURATION = "duration"     # Time duration
    PERCENTAGE = "percentage" # Percentage value


@dataclass
class StatDefinition:
    """Definition of a tracked statistic."""
    name: str
    stat_type: StatType
    display_name: str
    description: str = ""
    unit: str = ""
    higher_is_better: bool = True
    aggregation: str = "sum"  # sum, avg, max, min, last


@dataclass
class GameSession:
    """A single gaming session."""
    session_id: str
    game_id: str
    start_time: float
    end_time: Optional[float] = None
    duration_minutes: float = 0
    stats: dict[str, float] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def is_active(self) -> bool:
        return self.end_time is None


@dataclass
class PlayerProfile:
    """Player profile with aggregate statistics."""
    player_id: str
    display_name: str
    created_at: float = field(default_factory=time.time)
    total_playtime_minutes: float = 0
    total_sessions: int = 0
    lifetime_stats: dict[str, float] = field(default_factory=dict)
    per_game_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    achievements: list[str] = field(default_factory=list)


# Common game statistics definitions
COMMON_STATS = {
    "kills": StatDefinition("kills", StatType.COUNTER, "Kills", "Total enemy kills"),
    "deaths": StatDefinition("deaths", StatType.COUNTER, "Deaths", "Total deaths", higher_is_better=False),
    "assists": StatDefinition("assists", StatType.COUNTER, "Assists", "Total assists"),
    "damage_dealt": StatDefinition("damage_dealt", StatType.COUNTER, "Damage Dealt", unit="dmg"),
    "damage_taken": StatDefinition("damage_taken", StatType.COUNTER, "Damage Taken", unit="dmg", higher_is_better=False),
    "healing_done": StatDefinition("healing_done", StatType.COUNTER, "Healing Done", unit="hp"),
    "gold_earned": StatDefinition("gold_earned", StatType.COUNTER, "Gold Earned", unit="gold"),
    "xp_earned": StatDefinition("xp_earned", StatType.COUNTER, "XP Earned", unit="xp"),
    "objectives_completed": StatDefinition("objectives_completed", StatType.COUNTER, "Objectives Completed"),
    "win": StatDefinition("win", StatType.COUNTER, "Wins"),
    "loss": StatDefinition("loss", StatType.COUNTER, "Losses", higher_is_better=False),
    "accuracy": StatDefinition("accuracy", StatType.PERCENTAGE, "Accuracy", unit="%", aggregation="avg"),
    "kda": StatDefinition("kda", StatType.RATE, "K/D/A Ratio", aggregation="avg"),
    "score": StatDefinition("score", StatType.GAUGE, "Score"),
}


class SessionTracker:
    """
    Track individual gaming sessions.
    """
    
    def __init__(self):
        self._active_session: Optional[GameSession] = None
        self._sessions: list[GameSession] = []
        self._max_sessions = 100  # Prevent unbounded memory growth
    
    def start_session(self, game_id: str, metadata: dict[str, Any] = None) -> GameSession:
        """
        Start a new gaming session.
        
        Args:
            game_id: Game identifier
            metadata: Optional session metadata
        
        Returns:
            Created GameSession
        """
        # End any active session
        if self._active_session:
            self.end_session()
        
        session = GameSession(
            session_id=f"{game_id}_{int(time.time())}",
            game_id=game_id,
            start_time=time.time(),
            metadata=metadata or {}
        )
        
        self._active_session = session
        logger.info(f"Started session for {game_id}")
        
        return session
    
    def end_session(self) -> Optional[GameSession]:
        """End the current session."""
        if not self._active_session:
            return None
        
        session = self._active_session
        session.end_time = time.time()
        session.duration_minutes = (session.end_time - session.start_time) / 60
        
        self._sessions.append(session)
        # Trim old sessions to prevent unbounded growth
        if len(self._sessions) > self._max_sessions:
            self._sessions = self._sessions[-self._max_sessions:]
        self._active_session = None
        
        logger.info(f"Ended session {session.session_id} ({session.duration_minutes:.1f} min)")
        
        return session
    
    def record_stat(self, name: str, value: float, increment: bool = True):
        """
        Record a statistic in the current session.
        
        Args:
            name: Stat name
            value: Stat value
            increment: If True, add to existing value
        """
        if not self._active_session:
            return
        
        if increment and name in self._active_session.stats:
            self._active_session.stats[name] += value
        else:
            self._active_session.stats[name] = value
    
    def record_event(self, event_type: str, data: dict[str, Any] = None):
        """Record an event in the current session."""
        if not self._active_session:
            return
        
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "data": data or {}
        }
        self._active_session.events.append(event)
    
    def get_active_session(self) -> Optional[GameSession]:
        """Get the current active session."""
        return self._active_session
    
    def get_session_stats(self) -> dict[str, float]:
        """Get stats from current session."""
        if self._active_session:
            return self._active_session.stats.copy()
        return {}
    
    def get_recent_sessions(self, count: int = 10) -> list[GameSession]:
        """Get recent completed sessions."""
        return self._sessions[-count:]


class StatsAnalyzer:
    """
    Analyze player statistics and provide insights.
    """
    
    def __init__(self):
        self._stat_definitions = COMMON_STATS.copy()
    
    def add_stat_definition(self, stat: StatDefinition):
        """Add a custom stat definition."""
        self._stat_definitions[stat.name] = stat
    
    def calculate_kda(self, kills: int, deaths: int, assists: int) -> float:
        """Calculate K/D/A ratio."""
        if deaths == 0:
            return kills + assists * 0.5
        return (kills + assists * 0.5) / deaths
    
    def calculate_winrate(self, wins: int, losses: int) -> float:
        """Calculate win rate percentage."""
        total = wins + losses
        if total == 0:
            return 0.0
        return (wins / total) * 100
    
    def analyze_sessions(
        self,
        sessions: list[GameSession]
    ) -> dict[str, Any]:
        """
        Analyze a list of sessions.
        
        Args:
            sessions: List of sessions to analyze
        
        Returns:
            Analysis results
        """
        if not sessions:
            return {"error": "No sessions to analyze"}
        
        # Aggregate stats
        totals: dict[str, float] = {}
        for session in sessions:
            for stat, value in session.stats.items():
                if stat not in totals:
                    totals[stat] = 0
                totals[stat] += value
        
        # Calculate averages
        count = len(sessions)
        averages = {stat: value / count for stat, value in totals.items()}
        
        # Time analysis
        total_time = sum(s.duration_minutes for s in sessions)
        avg_session_length = total_time / count if count > 0 else 0
        
        # Trend analysis
        trends = self._calculate_trends(sessions)
        
        # Performance metrics
        metrics = self._calculate_performance_metrics(sessions)
        
        return {
            "session_count": count,
            "total_playtime_minutes": total_time,
            "avg_session_length_minutes": avg_session_length,
            "totals": totals,
            "averages": averages,
            "trends": trends,
            "performance": metrics
        }
    
    def _calculate_trends(
        self,
        sessions: list[GameSession]
    ) -> dict[str, str]:
        """Calculate stat trends over sessions."""
        if len(sessions) < 3:
            return {}
        
        trends = {}
        
        # Split into halves
        mid = len(sessions) // 2
        first_half = sessions[:mid]
        second_half = sessions[mid:]
        
        # Compare averages
        for stat in first_half[0].stats.keys():
            first_avg = sum(s.stats.get(stat, 0) for s in first_half) / len(first_half)
            second_avg = sum(s.stats.get(stat, 0) for s in second_half) / len(second_half)
            
            if first_avg == 0:
                continue
            
            change = (second_avg - first_avg) / first_avg * 100
            
            if abs(change) < 5:
                trends[stat] = "stable"
            elif change > 0:
                trends[stat] = "improving" if self._is_higher_better(stat) else "declining"
            else:
                trends[stat] = "declining" if self._is_higher_better(stat) else "improving"
        
        return trends
    
    def _is_higher_better(self, stat_name: str) -> bool:
        """Check if higher values are better for a stat."""
        if stat_name in self._stat_definitions:
            return self._stat_definitions[stat_name].higher_is_better
        return True  # Default assumption
    
    def _calculate_performance_metrics(
        self,
        sessions: list[GameSession]
    ) -> dict[str, Any]:
        """Calculate overall performance metrics."""
        metrics = {}
        
        # KDA calculation
        total_kills = sum(s.stats.get("kills", 0) for s in sessions)
        total_deaths = sum(s.stats.get("deaths", 0) for s in sessions)
        total_assists = sum(s.stats.get("assists", 0) for s in sessions)
        
        if total_kills > 0 or total_deaths > 0 or total_assists > 0:
            metrics["overall_kda"] = self.calculate_kda(
                int(total_kills), int(total_deaths), int(total_assists)
            )
        
        # Win rate
        wins = sum(s.stats.get("win", 0) for s in sessions)
        losses = sum(s.stats.get("loss", 0) for s in sessions)
        
        if wins > 0 or losses > 0:
            metrics["win_rate"] = self.calculate_winrate(int(wins), int(losses))
        
        # Best session
        if sessions:
            best_idx = max(
                range(len(sessions)),
                key=lambda i: sessions[i].stats.get("score", 0)
            )
            metrics["best_session"] = sessions[best_idx].session_id
        
        return metrics
    
    def get_improvement_suggestions(
        self,
        analysis: dict[str, Any]
    ) -> list[str]:
        """
        Generate improvement suggestions based on analysis.
        
        Args:
            analysis: Analysis results from analyze_sessions
        
        Returns:
            List of suggestion strings
        """
        suggestions = []
        
        trends = analysis.get("trends", {})
        averages = analysis.get("averages", {})
        performance = analysis.get("performance", {})
        
        # Check KDA
        kda = performance.get("overall_kda", 0)
        if kda < 1.0:
            suggestions.append("Focus on staying alive - your deaths are outpacing your contributions")
        elif kda > 3.0:
            suggestions.append("Excellent KDA! Consider being more aggressive to maximize impact")
        
        # Check win rate
        win_rate = performance.get("win_rate", 50)
        if win_rate < 45:
            suggestions.append("Win rate trending low - review replays to identify common issues")
        elif win_rate > 55:
            suggestions.append("Strong win rate! You might benefit from increasing difficulty")
        
        # Check trends
        for stat, trend in trends.items():
            if trend == "declining":
                stat_display = self._stat_definitions.get(stat, StatDefinition(stat, StatType.GAUGE, stat)).display_name
                suggestions.append(f"Your {stat_display} is trending downward - consider reviewing your approach")
        
        # Session length
        avg_length = analysis.get("avg_session_length_minutes", 0)
        if avg_length > 180:
            suggestions.append("Long sessions detected - remember to take breaks for optimal performance")
        
        return suggestions


class PlayerStats:
    """
    Main interface for player statistics management.
    Includes persistence and comprehensive tracking.
    """
    
    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir or "data/stats")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self._db_path = self.data_dir / "player_stats.db"
        self._init_database()
        
        self._tracker = SessionTracker()
        self._analyzer = StatsAnalyzer()
        self._profile: Optional[PlayerProfile] = None
    
    def _init_database(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                game_id TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL,
                duration_minutes REAL,
                stats TEXT,
                events TEXT,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                player_id TEXT PRIMARY KEY,
                display_name TEXT,
                created_at REAL,
                total_playtime_minutes REAL,
                total_sessions INTEGER,
                lifetime_stats TEXT,
                per_game_stats TEXT,
                achievements TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def load_profile(self, player_id: str = "default") -> PlayerProfile:
        """Load or create a player profile."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM profiles WHERE player_id = ?", (player_id,))
        row = cursor.fetchone()
        
        if row:
            self._profile = PlayerProfile(
                player_id=row[0],
                display_name=row[1],
                created_at=row[2],
                total_playtime_minutes=row[3],
                total_sessions=row[4],
                lifetime_stats=json.loads(row[5] or "{}"),
                per_game_stats=json.loads(row[6] or "{}"),
                achievements=json.loads(row[7] or "[]")
            )
        else:
            self._profile = PlayerProfile(
                player_id=player_id,
                display_name=player_id
            )
            self._save_profile()
        
        conn.close()
        return self._profile
    
    def _save_profile(self):
        """Save current profile to database."""
        if not self._profile:
            return
        
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO profiles VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self._profile.player_id,
            self._profile.display_name,
            self._profile.created_at,
            self._profile.total_playtime_minutes,
            self._profile.total_sessions,
            json.dumps(self._profile.lifetime_stats),
            json.dumps(self._profile.per_game_stats),
            json.dumps(self._profile.achievements)
        ))
        
        conn.commit()
        conn.close()
    
    def start_session(self, game_id: str) -> GameSession:
        """Start a new game session."""
        return self._tracker.start_session(game_id)
    
    def end_session(self):
        """End current session and update stats."""
        session = self._tracker.end_session()
        
        if session and self._profile:
            # Update profile stats
            self._profile.total_sessions += 1
            self._profile.total_playtime_minutes += session.duration_minutes
            
            # Update lifetime stats
            for stat, value in session.stats.items():
                if stat not in self._profile.lifetime_stats:
                    self._profile.lifetime_stats[stat] = 0
                self._profile.lifetime_stats[stat] += value
            
            # Update per-game stats
            if session.game_id not in self._profile.per_game_stats:
                self._profile.per_game_stats[session.game_id] = {}
            
            game_stats = self._profile.per_game_stats[session.game_id]
            for stat, value in session.stats.items():
                if stat not in game_stats:
                    game_stats[stat] = 0
                game_stats[stat] += value
            
            # Save to database
            self._save_session(session)
            self._save_profile()
    
    def _save_session(self, session: GameSession):
        """Save session to database."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session.session_id,
            session.game_id,
            session.start_time,
            session.end_time,
            session.duration_minutes,
            json.dumps(session.stats),
            json.dumps(session.events),
            json.dumps(session.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def record_stat(self, name: str, value: float, increment: bool = True):
        """Record a stat in current session."""
        self._tracker.record_stat(name, value, increment)
    
    def record_event(self, event_type: str, data: dict[str, Any] = None):
        """Record an event in current session."""
        self._tracker.record_event(event_type, data)
    
    def get_stats_summary(self, game_id: str = None) -> dict[str, Any]:
        """
        Get statistics summary.
        
        Args:
            game_id: Filter by game (None for all)
        
        Returns:
            Stats summary dictionary
        """
        if not self._profile:
            return {}
        
        if game_id:
            stats = self._profile.per_game_stats.get(game_id, {})
        else:
            stats = self._profile.lifetime_stats
        
        return {
            "total_playtime_minutes": self._profile.total_playtime_minutes,
            "total_sessions": self._profile.total_sessions,
            "stats": stats,
            "achievements": self._profile.achievements
        }
    
    def get_session_history(
        self,
        game_id: str = None,
        limit: int = 20
    ) -> list[GameSession]:
        """Get session history from database."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        
        if game_id:
            cursor.execute(
                "SELECT * FROM sessions WHERE game_id = ? ORDER BY start_time DESC LIMIT ?",
                (game_id, limit)
            )
        else:
            cursor.execute(
                "SELECT * FROM sessions ORDER BY start_time DESC LIMIT ?",
                (limit,)
            )
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append(GameSession(
                session_id=row[0],
                game_id=row[1],
                start_time=row[2],
                end_time=row[3],
                duration_minutes=row[4],
                stats=json.loads(row[5] or "{}"),
                events=json.loads(row[6] or "[]"),
                metadata=json.loads(row[7] or "{}")
            ))
        
        conn.close()
        return sessions
    
    def analyze_performance(self, game_id: str = None) -> dict[str, Any]:
        """Analyze player performance."""
        sessions = self.get_session_history(game_id, limit=50)
        return self._analyzer.analyze_sessions(sessions)
    
    def get_suggestions(self, game_id: str = None) -> list[str]:
        """Get improvement suggestions."""
        analysis = self.analyze_performance(game_id)
        return self._analyzer.get_improvement_suggestions(analysis)


def get_player_stats(data_dir: str = None) -> PlayerStats:
    """Get or create PlayerStats singleton."""
    if not hasattr(get_player_stats, '_instance'):
        get_player_stats._instance = PlayerStats(data_dir)
    return get_player_stats._instance
