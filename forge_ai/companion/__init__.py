"""
Companion Package - Always-on AI desktop companion.

The companion watches the screen, comments on what's happening,
offers help proactively, and behaves like a lifelike assistant.
"""

from .companion_mode import (
    CompanionMode,
    CompanionConfig,
    CompanionState,
    get_companion,
    start_companion,
    stop_companion,
)

__all__ = [
    "CompanionMode",
    "CompanionConfig", 
    "CompanionState",
    "get_companion",
    "start_companion",
    "stop_companion",
]
