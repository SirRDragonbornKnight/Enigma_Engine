"""
Achievement Tracking for Enigma AI Engine

AI congratulates on milestones.

Features:
- Achievement detection
- Progress tracking
- Celebration responses
- Statistics
- Leaderboards

Usage:
    from enigma_engine.tools.achievement_tracker import AchievementTracker
    
    tracker = AchievementTracker()
    
    # Register achievements
    tracker.register("First Kill", "Defeat your first enemy", icon="sword")
    tracker.register("Millionaire", "Earn 1,000,000 gold", icon="gold")
    
    # Unlock achievement
    tracker.unlock("First Kill")
    
    # Track progress
    tracker.update_progress("Millionaire", 50000)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Achievement:
    """An achievement."""
    id: str
    name: str
    description: str
    icon: str = "trophy"
    points: int = 10
    
    # Progress tracking
    target: float = 1.0
    current: float = 0.0
    
    # Status
    unlocked: bool = False
    unlocked_at: Optional[float] = None
    
    # Rarity
    hidden: bool = False
    rarity: str = "common"  # common, uncommon, rare, epic, legendary
    
    # Category
    category: str = "general"


@dataclass
class AchievementEvent:
    """Achievement event for history."""
    achievement_id: str
    event_type: str  # unlocked, progress
    timestamp: float
    value: float = 0.0
    message: str = ""


class AchievementTracker:
    """Track achievements and milestones."""
    
    CELEBRATION_MESSAGES = {
        "common": [
            "Nice! You unlocked: {name}!",
            "Achievement unlocked: {name}!",
            "Great job! {name} is yours!"
        ],
        "uncommon": [
            "Well done! You earned: {name}!",
            "Impressive! {name} unlocked!",
            "You're doing great! {name} achieved!"
        ],
        "rare": [
            "Excellent! Rare achievement unlocked: {name}!",
            "Amazing work! You got the rare {name}!",
            "Few players have earned: {name}!"
        ],
        "epic": [
            "EPIC! You unlocked: {name}!",
            "Incredible achievement: {name}!",
            "You're a legend! {name} is yours!"
        ],
        "legendary": [
            "LEGENDARY! {name} has been unlocked!",
            "You are one of the few to earn: {name}!",
            "The ultimate achievement: {name}!"
        ]
    }
    
    def __init__(self, save_path: Optional[Path] = None):
        """
        Initialize tracker.
        
        Args:
            save_path: Path to save achievements
        """
        self.save_path = save_path or Path("data/achievements.json")
        self._achievements: Dict[str, Achievement] = {}
        self._history: List[AchievementEvent] = []
        self._listeners: List[Callable[[Achievement, str], None]] = []
        
        # Statistics
        self._stats = {
            "total_unlocked": 0,
            "total_points": 0,
            "categories": {}
        }
        
        # Load saved achievements
        self._load()
    
    def register(
        self,
        id: str,
        name: str,
        description: str,
        target: float = 1.0,
        points: int = 10,
        icon: str = "trophy",
        category: str = "general",
        rarity: str = "common",
        hidden: bool = False
    ) -> Achievement:
        """
        Register a new achievement.
        
        Args:
            id: Unique ID
            name: Display name
            description: Description
            target: Target value for progress-based
            points: Point value
            icon: Icon name
            category: Category name
            rarity: Rarity level
            hidden: Whether hidden until unlocked
            
        Returns:
            Created achievement
        """
        achievement = Achievement(
            id=id,
            name=name,
            description=description,
            target=target,
            points=points,
            icon=icon,
            category=category,
            rarity=rarity,
            hidden=hidden
        )
        
        self._achievements[id] = achievement
        
        if category not in self._stats["categories"]:
            self._stats["categories"][category] = {"total": 0, "unlocked": 0}
        self._stats["categories"][category]["total"] += 1
        
        return achievement
    
    def unlock(self, achievement_id: str) -> tuple[bool, str]:
        """
        Unlock an achievement.
        
        Args:
            achievement_id: Achievement ID
            
        Returns:
            Tuple of (was_newly_unlocked, celebration_message)
        """
        achievement = self._achievements.get(achievement_id)
        
        if not achievement:
            return False, f"Unknown achievement: {achievement_id}"
        
        if achievement.unlocked:
            return False, f"Already unlocked: {achievement.name}"
        
        # Unlock
        achievement.unlocked = True
        achievement.unlocked_at = time.time()
        achievement.current = achievement.target
        
        # Update stats
        self._stats["total_unlocked"] += 1
        self._stats["total_points"] += achievement.points
        self._stats["categories"][achievement.category]["unlocked"] += 1
        
        # Record event
        event = AchievementEvent(
            achievement_id=achievement_id,
            event_type="unlocked",
            timestamp=time.time(),
            value=achievement.target
        )
        self._history.append(event)
        
        # Generate celebration message
        import random
        messages = self.CELEBRATION_MESSAGES.get(achievement.rarity, self.CELEBRATION_MESSAGES["common"])
        message = random.choice(messages).format(name=achievement.name)
        
        # Notify listeners
        for listener in self._listeners:
            try:
                listener(achievement, message)
            except Exception as e:
                logger.error(f"Listener error: {e}")
        
        # Save
        self._save()
        
        logger.info(f"Achievement unlocked: {achievement.name}")
        
        return True, message
    
    def update_progress(self, achievement_id: str, value: float) -> tuple[bool, str]:
        """
        Update achievement progress.
        
        Args:
            achievement_id: Achievement ID
            value: New progress value
            
        Returns:
            Tuple of (was_unlocked, message)
        """
        achievement = self._achievements.get(achievement_id)
        
        if not achievement:
            return False, f"Unknown achievement: {achievement_id}"
        
        if achievement.unlocked:
            return False, ""
        
        # Update progress
        old_progress = achievement.current
        achievement.current = min(value, achievement.target)
        
        # Record progress event
        event = AchievementEvent(
            achievement_id=achievement_id,
            event_type="progress",
            timestamp=time.time(),
            value=value
        )
        self._history.append(event)
        
        # Check if complete
        if achievement.current >= achievement.target:
            return self.unlock(achievement_id)
        
        # Progress milestone notifications (25%, 50%, 75%)
        old_pct = old_progress / achievement.target * 100
        new_pct = achievement.current / achievement.target * 100
        
        for milestone in [25, 50, 75]:
            if old_pct < milestone <= new_pct:
                return False, f"{milestone}% progress on {achievement.name}!"
        
        self._save()
        return False, ""
    
    def increment_progress(self, achievement_id: str, amount: float = 1.0) -> tuple[bool, str]:
        """Increment achievement progress."""
        achievement = self._achievements.get(achievement_id)
        if achievement:
            return self.update_progress(achievement_id, achievement.current + amount)
        return False, ""
    
    def get_achievement(self, achievement_id: str) -> Optional[Achievement]:
        """Get an achievement by ID."""
        return self._achievements.get(achievement_id)
    
    def get_all_achievements(self, include_hidden: bool = False) -> List[Achievement]:
        """Get all achievements."""
        return [
            a for a in self._achievements.values()
            if include_hidden or not a.hidden or a.unlocked
        ]
    
    def get_unlocked(self) -> List[Achievement]:
        """Get all unlocked achievements."""
        return [a for a in self._achievements.values() if a.unlocked]
    
    def get_by_category(self, category: str) -> List[Achievement]:
        """Get achievements by category."""
        return [a for a in self._achievements.values() if a.category == category]
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """
        Get progress summary.
        
        Returns:
            Summary dict
        """
        total = len(self._achievements)
        unlocked = len(self.get_unlocked())
        
        return {
            "total_achievements": total,
            "unlocked_achievements": unlocked,
            "completion_percentage": (unlocked / total * 100) if total > 0 else 0,
            "total_points": self._stats["total_points"],
            "categories": {
                cat: {
                    "total": stats["total"],
                    "unlocked": stats["unlocked"],
                    "percentage": (stats["unlocked"] / stats["total"] * 100) if stats["total"] > 0 else 0
                }
                for cat, stats in self._stats["categories"].items()
            }
        }
    
    def add_listener(self, callback: Callable[[Achievement, str], None]):
        """Add achievement listener."""
        self._listeners.append(callback)
    
    def get_recent_unlocks(self, limit: int = 10) -> List[Achievement]:
        """Get recently unlocked achievements."""
        unlocked = [a for a in self._achievements.values() if a.unlocked]
        sorted_achievements = sorted(unlocked, key=lambda a: a.unlocked_at or 0, reverse=True)
        return sorted_achievements[:limit]
    
    def get_history(self) -> List[AchievementEvent]:
        """Get achievement history."""
        return list(self._history)
    
    def _load(self):
        """Load saved achievements."""
        if not self.save_path.exists():
            return
        
        try:
            with open(self.save_path) as f:
                data = json.load(f)
            
            for ach_data in data.get("achievements", []):
                ach_id = ach_data["id"]
                if ach_id in self._achievements:
                    ach = self._achievements[ach_id]
                    ach.current = ach_data.get("current", 0)
                    ach.unlocked = ach_data.get("unlocked", False)
                    ach.unlocked_at = ach_data.get("unlocked_at")
                else:
                    # Recreate achievement
                    self._achievements[ach_id] = Achievement(**ach_data)
            
            self._stats = data.get("stats", self._stats)
            
        except Exception as e:
            logger.error(f"Failed to load achievements: {e}")
    
    def _save(self):
        """Save achievements."""
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            data = {
                "achievements": [
                    {
                        "id": a.id,
                        "name": a.name,
                        "description": a.description,
                        "target": a.target,
                        "current": a.current,
                        "unlocked": a.unlocked,
                        "unlocked_at": a.unlocked_at,
                        "points": a.points,
                        "category": a.category,
                        "rarity": a.rarity
                    }
                    for a in self._achievements.values()
                ],
                "stats": self._stats
            }
            
            with open(self.save_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save achievements: {e}")
    
    def render_achievement(self, achievement: Achievement) -> str:
        """Render achievement as text."""
        status = "[X]" if achievement.unlocked else "[ ]"
        progress = f" ({achievement.current:.0f}/{achievement.target:.0f})" if achievement.target > 1 else ""
        
        return f"{status} {achievement.name}{progress} - {achievement.description} [{achievement.points}pts]"


# Built-in achievement presets
def register_gaming_achievements(tracker: AchievementTracker):
    """Register common gaming achievements."""
    # Time-based
    tracker.register("First Hour", "Play for 1 hour", target=3600, category="time", icon="clock")
    tracker.register("Dedicated", "Play for 10 hours", target=36000, category="time", icon="clock", rarity="uncommon")
    tracker.register("No Life", "Play for 100 hours", target=360000, category="time", icon="clock", rarity="rare")
    
    # Combat
    tracker.register("First Blood", "Defeat your first enemy", category="combat", icon="sword")
    tracker.register("Hunter", "Defeat 100 enemies", target=100, category="combat", icon="sword", rarity="uncommon")
    tracker.register("Slayer", "Defeat 1000 enemies", target=1000, category="combat", icon="sword", rarity="rare")
    
    # Collection
    tracker.register("Hoarder", "Collect 1000 items", target=1000, category="collection", icon="chest")
    tracker.register("Completionist", "Find all collectibles", category="collection", icon="star", rarity="epic", hidden=True)
    
    # Social
    tracker.register("Friendly", "Help another player", category="social", icon="heart")
    tracker.register("Leader", "Form a party", category="social", icon="users")


# Convenience functions
def create_tracker() -> AchievementTracker:
    """Create a new achievement tracker."""
    return AchievementTracker()


def quick_celebrate(name: str) -> str:
    """Generate a celebration message."""
    import random
    messages = AchievementTracker.CELEBRATION_MESSAGES["common"]
    return random.choice(messages).format(name=name)
