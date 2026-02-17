"""Voice profile stub - simplified voice configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import CONFIG

# Profiles directory
PROFILES_DIR = Path(CONFIG.voices_dir if hasattr(CONFIG, 'voices_dir') else "voices")


@dataclass
class VoiceProfile:
    """Voice configuration profile."""
    
    name: str = "default"
    pitch: float = 1.0
    speed: float = 1.0
    volume: float = 1.0
    voice: str = "default"
    effects: List[str] = field(default_factory=list)
    description: str = ""
    samples_dir: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "pitch": self.pitch,
            "speed": self.speed,
            "volume": self.volume,
            "voice": self.voice,
            "effects": self.effects,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceProfile":
        return cls(
            name=data.get("name", "default"),
            pitch=data.get("pitch", 1.0),
            speed=data.get("speed", 1.0),
            volume=data.get("volume", 1.0),
            voice=data.get("voice", "default"),
            effects=data.get("effects", []),
            description=data.get("description", ""),
        )


# Preset voice profiles
PRESET_PROFILES = {
    "default": VoiceProfile(name="default", description="Default voice"),
    "assistant": VoiceProfile(name="assistant", pitch=1.0, speed=1.0, description="Standard assistant"),
    "friendly": VoiceProfile(name="friendly", pitch=1.1, speed=1.05, description="Friendly tone"),
    "calm": VoiceProfile(name="calm", pitch=0.95, speed=0.9, description="Calm and relaxed"),
}


def get_engine():
    """Stub - return None for disabled engine."""
    return None


def list_presets() -> list:
    """Return preset profile names."""
    return list(PRESET_PROFILES.keys())


def list_custom_profiles() -> list:
    """Stub - return empty list."""
    return []


def list_system_voices() -> list:
    """Stub - return empty list."""
    return []
