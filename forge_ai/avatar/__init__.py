# avatar package
"""
ForgeAI Avatar System

Provides a controllable avatar with AI-driven animations like Cortana or JARVIS.

Animation System:
    from forge_ai.avatar.animation_system import AvatarAnimator, AIAvatarController, AnimationState
    
    # Create animator and load animations (GIF, sprite sheet, or image sequence)
    animator = AvatarAnimator()
    animator.load_animation("idle", "data/avatar/animations/idle.gif")
    animator.load_animation("talking", "data/avatar/animations/talking.gif")
    animator.load_animation("wave", "data/avatar/animations/wave.gif", loop=False)
    
    # Map states to animations
    animator.map_state_to_animation(AnimationState.IDLE, "idle")
    animator.map_state_to_animation(AnimationState.TALKING, "talking")
    
    # Connect to display
    animator.frame_changed.connect(my_display.set_pixmap)
    
    # AI controls via controller
    controller = AIAvatarController(animator)
    controller.start_talking()  # When AI speaks
    controller.stop_talking()   # When done
    controller.set_emotion("happy")  # Based on response
    controller.gesture("wave")  # One-shot gesture

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
    execute_action,
)

from .avatar_identity import (
    AIAvatarIdentity,
    AvatarAppearance,
    PERSONALITY_TO_APPEARANCE,
)

from .emotion_sync import EmotionExpressionSync
from .lip_sync import LipSync
from .customizer import AvatarCustomizer

# Adaptive animation system (works with ANY model)
from .adaptive_animator import (
    AdaptiveAnimator,
    ModelCapabilities,
    AnimationStrategy,
    AnimationState,
)

# AI-controlled animation systems (Cortana/JARVIS style)
# 2D Animation (GIF, sprite sheets, image sequences)
from .animation_system import (
    AvatarAnimator,
    AIAvatarController,
    AnimationState as AnimationState2D,
    Animation,
)

# 3D Animation (real-time skeletal animation with Panda3D)
try:
    from .animation_3d import (
        Avatar3DAnimator,
        AI3DAvatarController,
        Animation3DState,
        create_3d_avatar,
    )
    HAS_3D_ANIMATION = True
except ImportError:
    HAS_3D_ANIMATION = False

# Speech synchronization (voice + lip sync)
from .speech_sync import (
    SpeechSync,
    SpeechSyncConfig,
    get_speech_sync,
    sync_speak,
    set_avatar_for_sync,
    create_voice_avatar_bridge,
)

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
    "execute_action",
    
    # Identity & Appearance
    "AIAvatarIdentity",
    "AvatarAppearance",
    "PERSONALITY_TO_APPEARANCE",
    
    # Sync & Animation
    "EmotionExpressionSync",
    "LipSync",
    
    # Adaptive Animation (universal)
    "AdaptiveAnimator",
    "ModelCapabilities", 
    "AnimationStrategy",
    "AnimationState",
    
    # Speech Sync (voice + avatar)
    "SpeechSync",
    "SpeechSyncConfig",
    "get_speech_sync",
    "sync_speak",
    "set_avatar_for_sync",
    "create_voice_avatar_bridge",
    
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
