"""
Game Mode - Zero Lag Gaming with AI Companion

BACKWARDS COMPATIBILITY MODULE
==============================
This module provides backwards-compatible aliases to gaming_mode.py.
All new code should use gaming_mode.py directly.

Mapping:
    GameMode -> GamingMode
    get_game_mode() -> get_gaming_mode()
    GameModeWatcher -> (use GamingMode's built-in monitoring)
"""

import logging
from typing import Any, Callable, Optional

# Import everything from gaming_mode (the main implementation)
from .gaming_mode import (
    GamingMode,
    GamingPriority,
    GamingProfile,
    ResourceLimits,
    get_gaming_mode,
)

logger = logging.getLogger(__name__)


class GameMode(GamingMode):
    """
    DEPRECATED: Use GamingMode instead.
    
    Backwards-compatible wrapper around GamingMode.
    Provides the old API for existing code.
    """
    
    def __init__(self):
        """Initialize game mode with default settings."""
        super().__init__()
        self._aggressive = False
        self._on_game_detected_callbacks: list[Callable[[str], None]] = []
        self._on_game_ended_callbacks: list[Callable[[], None]] = []
        self._on_limits_changed_callbacks: list[Callable[[ResourceLimits], None]] = []
    
    def enable(self, aggressive: bool = False):
        """
        Enable game mode.
        
        Args:
            aggressive: If True, maximum performance mode (maps to BACKGROUND priority)
                       If False, balanced mode (maps to MEDIUM priority)
        """
        self._aggressive = aggressive
        super().enable()
        logger.info(f"Game mode enabled (aggressive={aggressive})")
    
    def is_enabled(self) -> bool:
        """Check if game mode is enabled."""
        return self._enabled
    
    def is_active(self) -> bool:
        """Check if a game is currently detected."""
        return self._active_game is not None
    
    def auto_detect_game(self) -> bool:
        """Detect if a game is running."""
        return self._active_game is not None
    
    def get_resource_limits(self) -> ResourceLimits:
        """Get current resource limits."""
        return self._current_limits
    
    def on_game_detected(self, callback: Callable[[str], None]):
        """Register callback for when game is detected."""
        self._on_game_detected_callbacks.append(callback)
    
    def on_game_ended(self, callback: Callable[[], None]):
        """Register callback for when game ends."""
        self._on_game_ended_callbacks.append(callback)
    
    def on_limits_changed(self, callback: Callable[[ResourceLimits], None]):
        """Register callback for when resource limits change."""
        self._on_limits_changed_callbacks.append(callback)


class GameModeWatcher:
    """
    DEPRECATED: Use GamingMode's built-in monitoring instead.
    
    This class is kept for backwards compatibility.
    GamingMode handles its own monitoring internally.
    """
    
    def __init__(self, game_mode: GameMode):
        """Initialize watcher (no-op - GamingMode handles monitoring)."""
        self.game_mode = game_mode
        logger.warning("GameModeWatcher is deprecated. GamingMode handles monitoring internally.")
    
    def start(self):
        """Start watching (no-op - GamingMode handles this)."""
        pass
    
    def stop(self):
        """Stop watching (no-op - GamingMode handles this)."""
        pass


# Global instance using the same singleton as gaming_mode
_game_mode: Optional[GameMode] = None


def get_game_mode() -> GameMode:
    """
    Get or create global GameMode instance.
    
    DEPRECATED: Use get_gaming_mode() instead.
    """
    global _game_mode
    if _game_mode is None:
        _game_mode = GameMode()
    return _game_mode


__all__ = [
    'GameMode',
    'GameModeWatcher', 
    'get_game_mode',
    # Re-export from gaming_mode for convenience
    'GamingMode',
    'GamingPriority',
    'GamingProfile',
    'ResourceLimits',
    'get_gaming_mode',
]
