# avatar package
"""
ForgeAI Avatar System

Provides a controllable avatar that can:
  - Display on screen (on/off toggle, default OFF)
  - Move around the desktop
  - Express emotions
  - "Interact" with windows and files
  - Control a 3D model (when renderer implemented)
  - AI self-design based on personality
  - User customization
  - Load VRM/Live2D models (stubs available)
  - Use preset appearances

USAGE:
    from forge_ai.avatar import get_avatar, toggle_avatar
    
    avatar = get_avatar()
    avatar.enable()  # Turn on (default is off!)
    
    # Link to AI personality for auto-design
    from forge_ai.core.personality import load_personality
    personality = load_personality("my_model")
    avatar.link_personality(personality)
    
    # Let AI design its own appearance
    appearance = avatar.auto_design()
    
    # Or customize manually
    customizer = avatar.get_customizer()
    customizer.set_style("anime")
    customizer.apply_color_preset("sunset")
    
    # Or use presets
    from forge_ai.avatar.presets import get_preset_manager
    manager = get_preset_manager()
    preset = manager.get_preset("[builtin] friendly_helper")
    avatar.set_appearance(preset.appearance)
    
    avatar.move_to(500, 300)
    avatar.speak("Hello!")
    avatar.interact_with_window("My Document")
    avatar.disable()
"""

from .controller import (
    AvatarController,
    AvatarConfig,
    AvatarState,
    AvatarPosition,
    get_avatar,
    enable_avatar,
    disable_avatar,
    toggle_avatar,
)

from .avatar_identity import (
    AIAvatarIdentity,
    AvatarAppearance,
    PERSONALITY_TO_APPEARANCE,
)

from .emotion_sync import EmotionExpressionSync
from .lip_sync import LipSync
from .customizer import AvatarCustomizer

# Autonomous avatar system
from .autonomous import (
    AutonomousAvatar,
    AvatarMood,
    AutonomousConfig,
    ScreenRegion,
    get_autonomous_avatar,
)

# Preset system
from .presets import (
    AvatarPreset,
    PresetManager,
    get_preset_manager,
)

# Desktop pet (DesktopMate-style)
from .desktop_pet import (
    DesktopPet,
    PetState,
    PetConfig,
    get_desktop_pet,
)

# Blender integration
from .blender_bridge import (
    BlenderBridge,
    BlenderBridgeConfig,
    BlenderModelInfo,
    get_blender_bridge,
    save_blender_addon,
)

__all__ = [
    # Controller
    "AvatarController",
    "AvatarConfig",
    "AvatarState",
    "AvatarPosition",
    "get_avatar",
    "enable_avatar",
    "disable_avatar",
    "toggle_avatar",
    
    # Identity & Appearance
    "AIAvatarIdentity",
    "AvatarAppearance",
    "PERSONALITY_TO_APPEARANCE",
    
    # Sync & Animation
    "EmotionExpressionSync",
    "LipSync",
    
    # Customization
    "AvatarCustomizer",
    
    # Autonomous
    "AutonomousAvatar",
    "AvatarMood",
    "AutonomousConfig",
    "ScreenRegion",
    "get_autonomous_avatar",
    
    # Presets
    "AvatarPreset",
    "PresetManager",
    "get_preset_manager",
    
    # Desktop Pet
    "DesktopPet",
    "PetState",
    "PetConfig",
    "get_desktop_pet",
    
    # Blender Bridge
    "BlenderBridge",
    "BlenderBridgeConfig",
    "BlenderModelInfo",
    "get_blender_bridge",
    "save_blender_addon",
]
