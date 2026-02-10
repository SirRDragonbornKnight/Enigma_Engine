"""
Avatar Preset System

Save, load, and share complete avatar configurations.
Presets include colors, expressions, style, and accessories.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .avatar_identity import AvatarAppearance


@dataclass
class AvatarPreset:
    """A complete avatar configuration preset."""
    
    name: str
    description: str = ""
    author: str = "user"
    
    # Appearance settings
    appearance: Optional[AvatarAppearance] = None
    
    # Metadata
    created_at: str = ""
    version: str = "1.0"
    tags: list[str] = field(default_factory=list)
    
    # Preview image (base64 or path)
    preview_image: str = ""
    
    def __post_init__(self):
        if not self.tags:
            self.tags = []
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if self.appearance is None:
            self.appearance = AvatarAppearance()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "appearance": self.appearance.to_dict() if self.appearance else {},
            "created_at": self.created_at,
            "version": self.version,
            "tags": self.tags,
            "preview_image": self.preview_image,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'AvatarPreset':
        """Create from dictionary."""
        appearance_data = data.get("appearance", {})
        appearance = AvatarAppearance.from_dict(appearance_data) if appearance_data else AvatarAppearance()
        
        return cls(
            name=data.get("name", "Unnamed"),
            description=data.get("description", ""),
            author=data.get("author", "user"),
            appearance=appearance,
            created_at=data.get("created_at", ""),
            version=data.get("version", "1.0"),
            tags=data.get("tags", []),
            preview_image=data.get("preview_image", ""),
        )


class PresetManager:
    """
    Manage avatar presets - save, load, import, export.
    
    Usage:
        manager = PresetManager()
        
        # Save current avatar as preset
        manager.save_preset("MyAvatar", avatar.get_identity().appearance)
        
        # Load preset
        preset = manager.load_preset("MyAvatar")
        avatar.set_appearance(preset.appearance)
        
        # List presets
        for name in manager.list_presets():
            print(name)
    """
    
    # Built-in presets
    BUILTIN_PRESETS = {
        "default": {
            "name": "Default",
            "description": "Standard Enigma AI Engine avatar",
            "author": "Enigma AI Engine",
            "tags": ["default", "neutral"],
            "appearance": {
                "style": "default",
                "primary_color": "#6366f1",
                "secondary_color": "#8b5cf6",
                "accent_color": "#10b981",
                "shape": "rounded",
                "size": "medium",
                "default_expression": "neutral",
                "eye_style": "normal",
                "idle_animation": "breathe",
            }
        },
        "friendly_helper": {
            "name": "Friendly Helper",
            "description": "Warm, approachable assistant avatar",
            "author": "Enigma AI Engine",
            "tags": ["friendly", "warm", "helper"],
            "appearance": {
                "style": "default",
                "primary_color": "#f59e0b",
                "secondary_color": "#fbbf24",
                "accent_color": "#22c55e",
                "shape": "rounded",
                "size": "medium",
                "default_expression": "friendly",
                "eye_style": "cute",
                "idle_animation": "breathe",
            }
        },
        "professional": {
            "name": "Professional",
            "description": "Formal, business-like appearance",
            "author": "Enigma AI Engine",
            "tags": ["professional", "business", "formal"],
            "appearance": {
                "style": "minimal",
                "primary_color": "#1e293b",
                "secondary_color": "#475569",
                "accent_color": "#64748b",
                "shape": "angular",
                "size": "medium",
                "default_expression": "neutral",
                "eye_style": "sharp",
                "idle_animation": "still",
                "accessories": ["tie"],
            }
        },
        "creative_spark": {
            "name": "Creative Spark",
            "description": "Vibrant, artistic personality",
            "author": "Enigma AI Engine",
            "tags": ["creative", "artistic", "colorful"],
            "appearance": {
                "style": "abstract",
                "primary_color": "#8b5cf6",
                "secondary_color": "#ec4899",
                "accent_color": "#06b6d4",
                "shape": "mixed",
                "size": "medium",
                "default_expression": "excited",
                "eye_style": "cute",
                "idle_animation": "float",
                "accessories": ["creative_element", "sparkles"],
            }
        },
        "night_owl": {
            "name": "Night Owl",
            "description": "Dark theme for late night sessions",
            "author": "Enigma AI Engine",
            "tags": ["dark", "night", "minimal"],
            "appearance": {
                "style": "minimal",
                "primary_color": "#0f172a",
                "secondary_color": "#1e293b",
                "accent_color": "#6366f1",
                "shape": "rounded",
                "size": "small",
                "default_expression": "neutral",
                "eye_style": "normal",
                "idle_animation": "breathe",
            }
        },
        "energetic": {
            "name": "Energetic",
            "description": "Bouncy, playful avatar",
            "author": "Enigma AI Engine",
            "tags": ["playful", "energetic", "fun"],
            "appearance": {
                "style": "anime",
                "primary_color": "#22d3ee",
                "secondary_color": "#a855f7",
                "accent_color": "#fbbf24",
                "shape": "rounded",
                "size": "large",
                "default_expression": "excited",
                "eye_style": "cute",
                "idle_animation": "bounce",
                "accessories": ["hat"],
            }
        },
        "ocean_calm": {
            "name": "Ocean Calm",
            "description": "Serene, calming blue tones",
            "author": "Enigma AI Engine",
            "tags": ["calm", "ocean", "relaxing"],
            "appearance": {
                "style": "default",
                "primary_color": "#0ea5e9",
                "secondary_color": "#06b6d4",
                "accent_color": "#3b82f6",
                "shape": "rounded",
                "size": "medium",
                "default_expression": "friendly",
                "eye_style": "normal",
                "idle_animation": "float",
            }
        },
        "sunset_warmth": {
            "name": "Sunset Warmth",
            "description": "Warm sunset colors",
            "author": "Enigma AI Engine",
            "tags": ["warm", "sunset", "cozy"],
            "appearance": {
                "style": "default",
                "primary_color": "#f97316",
                "secondary_color": "#ef4444",
                "accent_color": "#fbbf24",
                "shape": "rounded",
                "size": "medium",
                "default_expression": "happy",
                "eye_style": "cute",
                "idle_animation": "breathe",
            }
        },
    }
    
    def __init__(self, presets_dir: Optional[str] = None):
        """
        Initialize preset manager.
        
        Args:
            presets_dir: Directory to store user presets
        """
        if presets_dir:
            self.presets_dir = Path(presets_dir)
        else:
            from ..config import CONFIG
            self.presets_dir = Path(CONFIG.get("data_dir", "data")) / "avatar" / "presets"
        
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, AvatarPreset] = {}
    
    def list_presets(self, include_builtin: bool = True) -> list[str]:
        """
        List available preset names.
        
        Args:
            include_builtin: Include built-in presets
            
        Returns:
            List of preset names
        """
        presets = []
        
        # Built-in presets
        if include_builtin:
            presets.extend([f"[builtin] {name}" for name in self.BUILTIN_PRESETS.keys()])
        
        # User presets
        for preset_file in self.presets_dir.glob("*.json"):
            presets.append(preset_file.stem)
        
        return sorted(presets)
    
    def get_preset(self, name: str) -> Optional[AvatarPreset]:
        """
        Get a preset by name.
        
        Args:
            name: Preset name
            
        Returns:
            AvatarPreset or None
        """
        # Check cache
        if name in self._cache:
            return self._cache[name]
        
        # Check if builtin
        builtin_name = name.replace("[builtin] ", "")
        if builtin_name in self.BUILTIN_PRESETS:
            preset_data = self.BUILTIN_PRESETS[builtin_name]
            preset = AvatarPreset.from_dict(preset_data)
            self._cache[name] = preset
            return preset
        
        # Load from file
        preset_path = self.presets_dir / f"{name}.json"
        if preset_path.exists():
            try:
                with open(preset_path, encoding='utf-8') as f:
                    data = json.load(f)
                preset = AvatarPreset.from_dict(data)
                self._cache[name] = preset
                return preset
            except Exception as e:
                print(f"[PresetManager] Error loading preset {name}: {e}")
        
        return None
    
    def save_preset(
        self, 
        name: str, 
        appearance: AvatarAppearance,
        description: str = "",
        tags: Optional[list[str]] = None
    ) -> bool:
        """
        Save an appearance as a preset.
        
        Args:
            name: Preset name
            appearance: AvatarAppearance to save
            description: Optional description
            tags: Optional tags for categorization
            
        Returns:
            True if saved successfully
        """
        preset = AvatarPreset(
            name=name,
            description=description,
            appearance=appearance,
            tags=tags or [],
        )
        
        try:
            preset_path = self.presets_dir / f"{name}.json"
            with open(preset_path, 'w', encoding='utf-8') as f:
                json.dump(preset.to_dict(), f, indent=2)
            
            # Update cache
            self._cache[name] = preset
            
            print(f"[PresetManager] Saved preset: {name}")
            return True
            
        except Exception as e:
            print(f"[PresetManager] Error saving preset {name}: {e}")
            return False
    
    def delete_preset(self, name: str) -> bool:
        """
        Delete a user preset.
        
        Args:
            name: Preset name
            
        Returns:
            True if deleted
        """
        if name.startswith("[builtin]"):
            print("[PresetManager] Cannot delete built-in presets")
            return False
        
        preset_path = self.presets_dir / f"{name}.json"
        if preset_path.exists():
            try:
                preset_path.unlink()
                if name in self._cache:
                    del self._cache[name]
                print(f"[PresetManager] Deleted preset: {name}")
                return True
            except Exception as e:
                print(f"[PresetManager] Error deleting preset: {e}")
        
        return False
    
    def export_preset(self, name: str, output_path: str) -> bool:
        """
        Export a preset to a file.
        
        Args:
            name: Preset name
            output_path: Path to export to
            
        Returns:
            True if exported
        """
        preset = self.get_preset(name)
        if not preset:
            return False
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(preset.to_dict(), f, indent=2)
            print(f"[PresetManager] Exported preset to: {output_path}")
            return True
        except Exception as e:
            print(f"[PresetManager] Export error: {e}")
            return False
    
    def import_preset(self, filepath: str, name: Optional[str] = None) -> bool:
        """
        Import a preset from a file.
        
        Args:
            filepath: Path to import from
            name: Optional override name
            
        Returns:
            True if imported
        """
        try:
            with open(filepath, encoding='utf-8') as f:
                data = json.load(f)
            
            preset = AvatarPreset.from_dict(data)
            
            if name:
                preset.name = name
            
            if preset.appearance is None:
                print("[PresetManager] Preset has no appearance data")
                return False
            return self.save_preset(
                preset.name,
                preset.appearance,
                preset.description,
                preset.tags
            )
        except Exception as e:
            print(f"[PresetManager] Import error: {e}")
            return False
    
    def get_presets_by_tag(self, tag: str) -> list[AvatarPreset]:
        """
        Get presets with a specific tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of matching presets
        """
        matches = []
        
        for name in self.list_presets():
            preset = self.get_preset(name)
            if preset and tag.lower() in [t.lower() for t in preset.tags]:
                matches.append(preset)
        
        return matches


# Global preset manager instance
_preset_manager: Optional[PresetManager] = None


def get_preset_manager() -> PresetManager:
    """Get or create preset manager."""
    global _preset_manager
    if _preset_manager is None:
        _preset_manager = PresetManager()
    return _preset_manager
