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
    
    # Manual rotation (Shift+drag on popup)
    overlay_rotation: float = 0.0  # 2D overlay rotation
    overlay_3d_yaw: float = 0.0  # 3D overlay yaw rotation
    
    # Per-avatar size overrides (avatar_path -> size)
    per_avatar_sizes: Dict[str, int] = field(default_factory=dict)
    per_avatar_positions: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    
    # Reposition mode (allows dragging to move avatar)
    reposition_enabled: bool = False
    
    # Appearance
    primary_color: str = "#6b8afd"
    secondary_color: str = "#4a6fd9"
    accent_color: str = "#ffc107"
    
    # Expression
    current_expression: str = "neutral"
    
    # Behavior
    resize_enabled: bool = False
    auto_emotion: bool = True
    
    # Display settings (for 3D preview)
    wireframe_mode: bool = False
    show_grid: bool = True
    light_intensity: float = 1.0
    ambient_strength: float = 0.15
    
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
        
        # Handle per_avatar_positions tuple conversion
        if "per_avatar_positions" in data:
            converted = {}
            for k, v in data["per_avatar_positions"].items():
                if isinstance(v, list):
                    converted[k] = tuple(v)
                else:
                    converted[k] = v
            data["per_avatar_positions"] = converted
        
        # Remove unknown fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        
        return cls(**filtered)
    
    def get_size_for_avatar(self, avatar_path: str) -> int:
        """Get the saved size for a specific avatar, or default."""
        if avatar_path and avatar_path in self.per_avatar_sizes:
            return self.per_avatar_sizes[avatar_path]
        return self.overlay_size
    
    def set_size_for_avatar(self, avatar_path: str, size: int) -> None:
        """Set the size for a specific avatar."""
        if avatar_path:
            self.per_avatar_sizes[avatar_path] = size
    
    def get_position_for_avatar(self, avatar_path: str) -> Tuple[int, int]:
        """Get the saved position for a specific avatar, or default."""
        if avatar_path and avatar_path in self.per_avatar_positions:
            return self.per_avatar_positions[avatar_path]
        return self.screen_position
    
    def set_position_for_avatar(self, avatar_path: str, x: int, y: int) -> None:
        """Set the position for a specific avatar."""
        if avatar_path:
            self.per_avatar_positions[avatar_path] = (x, y)


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


def get_avatar_state_for_ai() -> Dict[str, Any]:
    """
    Get complete avatar state for AI awareness.
    
    Returns a dict containing:
    - position: (x, y) screen coordinates
    - size: overlay size in pixels
    - facing: which direction avatar is facing (based on model yaw)
    - expression: current expression
    - screen_region: which part of screen (top-left, center, bottom-right, etc.)
    - visible: whether overlay is shown
    """
    import json
    from pathlib import Path
    
    settings = get_persistence().load()
    x, y = settings.screen_position
    size = settings.overlay_3d_size or settings.overlay_size
    
    # Determine screen region
    try:
        from PyQt5.QtWidgets import QApplication
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            screen_w, screen_h = geo.width(), geo.height()
        else:
            screen_w, screen_h = 1920, 1080  # Fallback
    except:
        screen_w, screen_h = 1920, 1080
    
    # Determine region (3x3 grid)
    third_w, third_h = screen_w // 3, screen_h // 3
    col = min(2, x // third_w) if third_w > 0 else 1
    row = min(2, y // third_h) if third_h > 0 else 1
    
    regions = [
        ["top-left", "top-center", "top-right"],
        ["middle-left", "center", "middle-right"],
        ["bottom-left", "bottom-center", "bottom-right"]
    ]
    screen_region = regions[row][col]
    
    # Get facing direction from model orientation
    facing = "forward"  # Default
    try:
        orient_path = Path("data/avatar/model_orientations.json")
        if orient_path.exists():
            with open(orient_path, 'r') as f:
                orientations = json.load(f)
            # Get current model's orientation
            current_avatar = settings.current_avatar
            if current_avatar:
                model_key = Path(current_avatar).name
                if model_key in orientations:
                    yaw = orientations[model_key].get('yaw', 0)
                    # Convert yaw to facing direction
                    yaw = yaw % 360
                    if 315 <= yaw or yaw < 45:
                        facing = "forward"
                    elif 45 <= yaw < 135:
                        facing = "right"
                    elif 135 <= yaw < 225:
                        facing = "backward"
                    else:
                        facing = "left"
    except:
        pass
    
    return {
        "position": {"x": x, "y": y},
        "size": size,
        "facing": facing,
        "expression": settings.current_expression,
        "screen_region": screen_region,
        "screen_size": {"width": screen_w, "height": screen_h},
        "resize_enabled": settings.resize_enabled,
        "avatar_type": settings.avatar_type,
        "current_avatar": settings.current_avatar,
    }


def write_avatar_state_for_ai() -> None:
    """Write avatar state to a JSON file that AI can read."""
    import json
    from pathlib import Path
    
    state = get_avatar_state_for_ai()
    state_path = Path("data/avatar/ai_avatar_state.json")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)
