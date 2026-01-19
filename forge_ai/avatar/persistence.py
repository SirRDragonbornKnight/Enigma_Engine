"""
Avatar Persistence System

Handles saving and loading avatar settings including:
- Current avatar selection
- Screen position (desktop overlay)
- Color customization
- Expression states
"""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

from ..config import CONFIG


SETTINGS_FILE = Path(CONFIG.get("data_dir", "data")) / "avatar" / "avatar_settings.json"


@dataclass
class AvatarSettings:
    """Persistent avatar settings."""
    
    # Current avatar
    current_avatar: Optional[str] = None
    avatar_type: str = "PNG_BOUNCE"  # PNG_BOUNCE, ANIMATED_2D, SKELETAL_3D
    
    # Desktop position (for overlay)
    screen_position: Tuple[int, int] = (100, 100)
    overlay_size: int = 200
    
    # 3D specific
    overlay_3d_size: int = 250
    
    # Appearance
    primary_color: str = "#6b8afd"
    secondary_color: str = "#4a6fd9"
    accent_color: str = "#ffc107"
    
    # Expression
    current_expression: str = "neutral"
    
    # Behavior
    resize_enabled: bool = False
    auto_emotion: bool = True
    
    # Custom emotion mappings
    emotion_mappings: Dict[str, str] = field(default_factory=dict)
    
    # Last modified
    last_modified: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["last_modified"] = datetime.now().isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AvatarSettings":
        """Create from dictionary."""
        # Handle tuple conversion
        if "screen_position" in data and isinstance(data["screen_position"], list):
            data["screen_position"] = tuple(data["screen_position"])
        
        # Remove unknown fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        
        return cls(**filtered)


class AvatarPersistence:
    """
    Manages avatar settings persistence.
    
    Usage:
        persistence = AvatarPersistence()
        
        # Load settings
        settings = persistence.load()
        
        # Modify
        settings.screen_position = (200, 300)
        settings.current_avatar = "robot"
        
        # Save
        persistence.save(settings)
    """
    
    def __init__(self, settings_path: Optional[Path] = None):
        self.settings_path = Path(settings_path) if settings_path else SETTINGS_FILE
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
    
    def load(self) -> AvatarSettings:
        """Load settings from disk."""
        if self.settings_path.exists():
            try:
                data = json.loads(self.settings_path.read_text(encoding="utf-8"))
                return AvatarSettings.from_dict(data)
            except Exception:
                pass
        return AvatarSettings()
    
    def save(self, settings: AvatarSettings) -> bool:
        """Save settings to disk."""
        try:
            data = settings.to_dict()
            self.settings_path.write_text(
                json.dumps(data, indent=2),
                encoding="utf-8"
            )
            return True
        except Exception:
            return False
    
    def update(self, **kwargs) -> AvatarSettings:
        """Update specific settings."""
        settings = self.load()
        for key, value in kwargs.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        self.save(settings)
        return settings
    
    def get_position(self) -> Tuple[int, int]:
        """Get saved screen position."""
        return self.load().screen_position
    
    def set_position(self, x: int, y: int) -> None:
        """Save screen position."""
        self.update(screen_position=(x, y))
    
    def get_colors(self) -> Dict[str, str]:
        """Get color settings."""
        settings = self.load()
        return {
            "primary": settings.primary_color,
            "secondary": settings.secondary_color,
            "accent": settings.accent_color,
        }
    
    def set_colors(self, primary: str = None, secondary: str = None, accent: str = None) -> None:
        """Save color settings."""
        updates = {}
        if primary:
            updates["primary_color"] = primary
        if secondary:
            updates["secondary_color"] = secondary
        if accent:
            updates["accent_color"] = accent
        if updates:
            self.update(**updates)


# Global instance
_persistence: Optional[AvatarPersistence] = None


def get_persistence() -> AvatarPersistence:
    """Get the global persistence instance."""
    global _persistence
    if _persistence is None:
        _persistence = AvatarPersistence()
    return _persistence


def save_position(x: int, y: int) -> None:
    """Quick function to save position."""
    get_persistence().set_position(x, y)


def load_position() -> Tuple[int, int]:
    """Quick function to load position."""
    return get_persistence().get_position()


def save_avatar_settings(**kwargs) -> None:
    """Quick function to save avatar settings."""
    get_persistence().update(**kwargs)


def load_avatar_settings() -> AvatarSettings:
    """Quick function to load avatar settings."""
    return get_persistence().load()
