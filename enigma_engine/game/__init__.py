"""
Game Integration

Gaming features including overlay, stats, and streaming.

Provides:
- GameOverlay: In-game overlay UI
- GameStats: Performance and statistics tracking
- GameAdvice: AI gaming tips and analysis
- StreamManager: Streaming platform integration
"""

from .advice import Advice, AdviceCategory, AdvicePriority, AdviceTrigger
from .overlay import OverlayConfig, OverlayMode, OverlayPosition
from .profiles import (
    AIBehavior,
    GameDetector,
    GameGenre,
    GameProfile,
    GameProfileManager,
    OverlaySettings,
)
from .stats import (
    GameSession,
    PlayerProfile,
    PlayerStats,
    SessionTracker,
    StatDefinition,
    StatsAnalyzer,
    StatType,
)
from .streaming import (
    ChatCommand,
    ChatMessage,
    MessageType,
    StreamConfig,
    StreamManager,
    StreamPlatform,
    TwitchIntegration,
    YouTubeIntegration,
)

__all__ = [
    # Overlay
    "OverlayConfig",
    "OverlayMode",
    "OverlayPosition",
    # Stats
    "GameSession",
    "PlayerProfile",
    "PlayerStats",
    "SessionTracker",
    "StatsAnalyzer",
    "StatDefinition",
    "StatType",
    # Advice
    "Advice",
    "AdviceCategory",
    "AdvicePriority",
    "AdviceTrigger",
    # Profiles
    "GameProfile",
    "GameProfileManager",
    "GameDetector",
    "GameGenre",
    "AIBehavior",
    "OverlaySettings",
    # Streaming
    "StreamManager",
    "StreamPlatform",
    "StreamConfig",
    "TwitchIntegration",
    "YouTubeIntegration",
    "ChatMessage",
    "ChatCommand",
    "MessageType",
]
