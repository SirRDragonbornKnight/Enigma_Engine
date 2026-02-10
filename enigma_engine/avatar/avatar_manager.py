"""
Multi-Avatar Manager

Manages multiple avatars, each potentially associated with a different persona.
This allows different AI personas to have different visual representations.

Usage:
    from enigma_engine.avatar.avatar_manager import get_avatar_manager
    
    manager = get_avatar_manager()
    
    # Create avatar for a persona
    manager.create_avatar("assistant", appearance={"style": "friendly", "color": "#89b4fa"})
    manager.create_avatar("coder", appearance={"style": "tech", "color": "#a6e3a1"})
    
    # Switch active avatar when persona changes
    manager.set_active_avatar("coder")
    
    # Get active avatar for controlling
    avatar = manager.get_active()
    avatar.speak("Hello!")
    
    # Or directly control named avatar
    manager.get_avatar("assistant").set_emotion("happy")
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .controller import AvatarController

try:
    from PyQt5.QtCore import QObject, pyqtSignal
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    QObject = object
    pyqtSignal = lambda *args: None

from ..config import CONFIG

logger = logging.getLogger(__name__)


@dataclass
class AvatarProfile:
    """Profile for a single avatar."""
    name: str  # Unique identifier, often matches persona name
    display_name: str = ""  # Friendly name for UI
    appearance: Dict[str, Any] = field(default_factory=dict)
    model_path: Optional[str] = None  # Path to 3D model or animation set
    voice_id: Optional[str] = None  # Voice ID for TTS
    enabled: bool = True
    
    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name.replace("_", " ").title()


class AvatarManager(QObject if HAS_PYQT else object):
    """
    Manager for multiple avatars with persona association.
    
    Supports:
    - Creating/deleting avatars
    - Associating avatars with personas
    - Switching active avatar
    - Syncing avatar state with persona changes
    - Persisting avatar configurations
    """
    
    # Signals
    if HAS_PYQT:
        avatar_changed = pyqtSignal(str)  # Emitted when active avatar changes
        avatar_created = pyqtSignal(str)  # Emitted when new avatar created
        avatar_deleted = pyqtSignal(str)  # Emitted when avatar deleted
    
    def __init__(self):
        if HAS_PYQT:
            super().__init__()
        
        self._avatars: Dict[str, "AvatarController"] = {}
        self._profiles: Dict[str, AvatarProfile] = {}
        self._active_name: Optional[str] = None
        self._persona_mapping: Dict[str, str] = {}  # persona_name -> avatar_name
        
        # Path for storing avatar configs
        self._config_path = Path(CONFIG.get("data_dir", "data")) / "avatars"
        self._config_path.mkdir(parents=True, exist_ok=True)
        
        # Load saved configurations
        self._load_configs()
    
    # =========================================================================
    # AVATAR LIFECYCLE
    # =========================================================================
    
    def create_avatar(
        self,
        name: str,
        display_name: str = "",
        appearance: Dict[str, Any] = None,
        model_path: str = None,
        voice_id: str = None,
        make_active: bool = False
    ) -> "AvatarController":
        """
        Create a new avatar.
        
        Args:
            name: Unique identifier
            display_name: Friendly name for UI
            appearance: Visual configuration
            model_path: Path to 3D model or animation set
            voice_id: Voice ID for TTS
            make_active: Whether to make this the active avatar
            
        Returns:
            The created AvatarController
        """
        from .controller import AvatarController
        
        if name in self._avatars:
            logger.warning(f"Avatar '{name}' already exists")
            return self._avatars[name]
        
        # Create profile
        profile = AvatarProfile(
            name=name,
            display_name=display_name or name.replace("_", " ").title(),
            appearance=appearance or {},
            model_path=model_path,
            voice_id=voice_id
        )
        self._profiles[name] = profile
        
        # Create controller
        controller = AvatarController(name=name)
        
        # Apply appearance settings
        if appearance:
            for key, value in appearance.items():
                controller.set_property(key, value)
        
        if model_path:
            controller.load_model(model_path)
        
        self._avatars[name] = controller
        
        # Set as active if requested or if first avatar
        if make_active or not self._active_name:
            self.set_active_avatar(name)
        
        # Emit signal
        if HAS_PYQT:
            self.avatar_created.emit(name)
        
        # Save config
        self._save_configs()
        
        logger.info(f"Created avatar: {name}")
        return controller
    
    def delete_avatar(self, name: str) -> bool:
        """
        Delete an avatar.
        
        Args:
            name: Avatar identifier
            
        Returns:
            True if deleted, False if not found
        """
        if name not in self._avatars:
            return False
        
        # Disable avatar first
        self._avatars[name].disable()
        
        # Remove from dictionaries
        del self._avatars[name]
        if name in self._profiles:
            del self._profiles[name]
        
        # Update persona mapping
        self._persona_mapping = {
            k: v for k, v in self._persona_mapping.items() if v != name
        }
        
        # Switch active if needed
        if self._active_name == name:
            self._active_name = next(iter(self._avatars), None)
            if HAS_PYQT and self._active_name:
                self.avatar_changed.emit(self._active_name)
        
        # Emit signal
        if HAS_PYQT:
            self.avatar_deleted.emit(name)
        
        # Save config
        self._save_configs()
        
        logger.info(f"Deleted avatar: {name}")
        return True
    
    def get_avatar(self, name: str) -> Optional["AvatarController"]:
        """Get avatar by name."""
        return self._avatars.get(name)
    
    def get_active(self) -> Optional["AvatarController"]:
        """Get the currently active avatar."""
        if self._active_name:
            return self._avatars.get(self._active_name)
        return None
    
    def list_avatars(self) -> List[str]:
        """List all avatar names."""
        return list(self._avatars.keys())
    
    def list_profiles(self) -> List[AvatarProfile]:
        """List all avatar profiles."""
        return list(self._profiles.values())
    
    # =========================================================================
    # ACTIVE AVATAR MANAGEMENT
    # =========================================================================
    
    def set_active_avatar(self, name: str) -> bool:
        """
        Set the active avatar.
        
        Args:
            name: Avatar identifier
            
        Returns:
            True if changed, False if not found
        """
        if name not in self._avatars:
            # Try to create default avatar if none exist
            if not self._avatars:
                self.create_avatar(name, make_active=True)
                return True
            return False
        
        old_active = self._active_name
        self._active_name = name
        
        # Swap visibility - hide old, show new
        if old_active and old_active in self._avatars and old_active != name:
            self._avatars[old_active].disable()
        
        # Enable new active
        self._avatars[name].enable()
        
        # Emit signal
        if HAS_PYQT:
            self.avatar_changed.emit(name)
        
        logger.info(f"Active avatar changed: {old_active} -> {name}")
        return True
    
    def get_active_name(self) -> Optional[str]:
        """Get the name of the active avatar."""
        return self._active_name
    
    # =========================================================================
    # PERSONA INTEGRATION
    # =========================================================================
    
    def map_persona_to_avatar(self, persona_name: str, avatar_name: str):
        """
        Associate a persona with an avatar.
        
        When the AI switches to this persona, the corresponding avatar
        will automatically become active.
        
        Args:
            persona_name: Name of the persona
            avatar_name: Name of the avatar to use
        """
        if avatar_name not in self._avatars:
            logger.warning(f"Avatar '{avatar_name}' not found")
            return
        
        self._persona_mapping[persona_name] = avatar_name
        self._save_configs()
        logger.info(f"Mapped persona '{persona_name}' to avatar '{avatar_name}'")
    
    def on_persona_changed(self, persona_name: str):
        """
        Handle persona change event.
        
        Automatically switches to the mapped avatar if one exists.
        
        Args:
            persona_name: Name of the new active persona
        """
        avatar_name = self._persona_mapping.get(persona_name)
        
        if avatar_name and avatar_name in self._avatars:
            self.set_active_avatar(avatar_name)
            logger.info(f"Switched to avatar '{avatar_name}' for persona '{persona_name}'")
        elif not avatar_name:
            # Create new avatar for unmapped persona
            logger.info(f"No avatar mapped for persona '{persona_name}', creating default")
            self.create_avatar(persona_name, make_active=True)
            self._persona_mapping[persona_name] = persona_name
            self._save_configs()
    
    def get_avatar_for_persona(self, persona_name: str) -> Optional["AvatarController"]:
        """Get the avatar associated with a persona."""
        avatar_name = self._persona_mapping.get(persona_name, persona_name)
        return self._avatars.get(avatar_name)
    
    # =========================================================================
    # BULK OPERATIONS
    # =========================================================================
    
    def enable_all(self):
        """Enable all avatars."""
        for avatar in self._avatars.values():
            avatar.enable()
    
    def disable_all(self):
        """Disable all avatars."""
        for avatar in self._avatars.values():
            avatar.disable()
    
    def set_emotion_all(self, emotion: str):
        """Set emotion on all avatars."""
        for avatar in self._avatars.values():
            avatar.set_emotion(emotion)
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def _save_configs(self):
        """Save avatar configurations to disk."""
        try:
            config = {
                "profiles": {
                    name: {
                        "name": p.name,
                        "display_name": p.display_name,
                        "appearance": p.appearance,
                        "model_path": p.model_path,
                        "voice_id": p.voice_id,
                        "enabled": p.enabled
                    }
                    for name, p in self._profiles.items()
                },
                "persona_mapping": self._persona_mapping,
                "active": self._active_name
            }
            
            config_file = self._config_path / "avatar_manager.json"
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save avatar configs: {e}")
    
    def _load_configs(self):
        """Load avatar configurations from disk."""
        try:
            config_file = self._config_path / "avatar_manager.json"
            if not config_file.exists():
                return
            
            with open(config_file) as f:
                config = json.load(f)
            
            # Load profiles and create avatars
            for name, profile_data in config.get("profiles", {}).items():
                profile = AvatarProfile(
                    name=profile_data.get("name", name),
                    display_name=profile_data.get("display_name", name),
                    appearance=profile_data.get("appearance", {}),
                    model_path=profile_data.get("model_path"),
                    voice_id=profile_data.get("voice_id"),
                    enabled=profile_data.get("enabled", True)
                )
                self._profiles[name] = profile
                
                # Create controller (but don't enable yet)
                from .controller import AvatarController
                controller = AvatarController(name=name)
                if profile.appearance:
                    for key, value in profile.appearance.items():
                        controller.set_property(key, value)
                self._avatars[name] = controller
            
            # Load persona mapping
            self._persona_mapping = config.get("persona_mapping", {})
            
            # Set active avatar
            active = config.get("active")
            if active and active in self._avatars:
                self._active_name = active
            elif self._avatars:
                self._active_name = next(iter(self._avatars))
                
        except Exception as e:
            logger.error(f"Failed to load avatar configs: {e}")
    
    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================
    
    def speak(self, text: str, avatar_name: str = None):
        """Make an avatar speak."""
        avatar = self.get_avatar(avatar_name) if avatar_name else self.get_active()
        if avatar:
            avatar.speak(text)
    
    def set_emotion(self, emotion: str, avatar_name: str = None):
        """Set avatar emotion."""
        avatar = self.get_avatar(avatar_name) if avatar_name else self.get_active()
        if avatar:
            avatar.set_emotion(emotion)
    
    def gesture(self, gesture: str, avatar_name: str = None):
        """Trigger avatar gesture."""
        avatar = self.get_avatar(avatar_name) if avatar_name else self.get_active()
        if avatar:
            avatar.gesture(gesture)


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_manager: Optional[AvatarManager] = None


def get_avatar_manager() -> AvatarManager:
    """Get or create the global avatar manager."""
    global _manager
    if _manager is None:
        _manager = AvatarManager()
    return _manager


def create_avatar(
    name: str,
    **kwargs
) -> "AvatarController":
    """Convenience function to create an avatar."""
    return get_avatar_manager().create_avatar(name, **kwargs)


def get_avatar_for_persona(persona_name: str) -> Optional["AvatarController"]:
    """Get avatar for a specific persona."""
    return get_avatar_manager().get_avatar_for_persona(persona_name)


def switch_avatar_for_persona(persona_name: str):
    """Switch to the avatar for a persona."""
    get_avatar_manager().on_persona_changed(persona_name)
