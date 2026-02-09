# avatar package
"""
Enigma AI Engine Avatar System

Provides a controllable avatar with AI-driven animations like Cortana or JARVIS.

Animation System:
    from enigma_engine.avatar.animation_system import AvatarAnimator, AIAvatarController, AnimationState
    
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
    from enigma_engine.avatar import get_avatar, toggle_avatar
    
    avatar = get_avatar()
    avatar.enable()  # Turn on (default is off!)
    
    # Link to AI personality for auto-design
    from enigma_engine.core.personality import load_personality
    personality = load_personality("my_model")
    avatar.link_personality(personality)
    
    # Let AI design its own appearance
    appearance = avatar.auto_design()
    
    # Or customize manually
    customizer = avatar.get_customizer()
    customizer.set_style("anime")
    customizer.apply_color_preset("sunset")
    
    # Or use presets
    from enigma_engine.avatar.presets import get_preset_manager
    manager = get_preset_manager()
    preset = manager.get_preset("[builtin] friendly_helper")
    avatar.set_appearance(preset.appearance)
    
    avatar.move_to(500, 300)
    avatar.speak("Hello!")
    avatar.interact_with_window("My Document")
    avatar.disable()
"""

# Adaptive animation system (works with ANY model)
from .adaptive_animator import (
    AdaptiveAnimator,
    AnimationState,
    AnimationStrategy,
    ModelCapabilities,
)

# 3D Animation (real-time skeletal animation - NO EXTERNAL DEPENDENCIES)
# Uses only PyQt5's built-in OpenGL support
from .animation_3d_native import (
    AI3DController,
    Animation3DState,
    Avatar3DWidget,
    GLTFLoader,
    NativeAvatar3D,
)

# AI-controlled animation systems (Cortana/JARVIS style)
# 2D Animation (GIF, sprite sheets, image sequences)
from .animation_system import (
    AIAvatarController,
    Animation,
)
from .animation_system import AnimationState as AnimationState2D
from .animation_system import (
    AvatarAnimator,
)
from .avatar_identity import (
    PERSONALITY_TO_APPEARANCE,
    AIAvatarIdentity,
    AvatarAppearance,
)

# Multi-avatar manager for persona-specific avatars
from .avatar_manager import (
    AvatarManager,
    AvatarProfile,
    create_avatar,
    get_avatar_for_persona,
    get_avatar_manager,
    switch_avatar_for_persona,
)
from .controller import (
    AvatarConfig,
    AvatarController,
    AvatarPosition,
    AvatarState,
    ControlPriority,
    disable_avatar,
    enable_avatar,
    execute_action,
    get_avatar,
    toggle_avatar,
)
from .customizer import AvatarCustomizer
from .emotion_sync import EmotionExpressionSync
from .lip_sync import LipSync

# Sentiment analyzer for emotion recognition
from .sentiment_analyzer import (
    EmotionType,
    SentimentAnalyzer,
    SentimentResult,
    analyze_for_avatar,
    get_sentiment_analyzer,
)

# Aliases for compatibility
Avatar3DAnimator = NativeAvatar3D
AI3DAvatarController = AI3DController
HAS_3D_ANIMATION = True

# AI-Avatar Bridge (AI controls avatar during chat)
# Two control modes:
# 1. AUTOMATIC: Detects emotions from text via SentimentAnalyzer
# 2. EXPLICIT: AI uses [emotion:happy], [gesture:wave], etc. commands
from .ai_bridge import (  # Explicit command system; Convenience functions
    AIAvatarBridge,
    AvatarChatIntegration,
    AvatarCommand,
    EmotionKeywords,
    ExplicitCommands,
    GestureKeywords,
    create_avatar_bridge,
    get_avatar_command_prompt,
    get_command_reference,
    integrate_avatar_with_chat,
    list_avatar_commands,
    parse_explicit_commands,
    process_ai_response,
)

# AI Control System
from .ai_control import (
    AIAvatarControl,
    BoneCommand,
    get_ai_avatar_control,
    parse_bone_commands,
    process_ai_response,
)

# Autonomous avatar system
from .autonomous import (
    AutonomousAvatar,
    AutonomousConfig,
    AvatarMood,
    ScreenRegion,
    get_autonomous_avatar,
)

# Avatar Bundle System
from .avatar_bundle import (
    AvatarBundle,
    AvatarBundleCreator,
    AvatarManifest,
    install_avatar_bundle,
    list_installed_avatars,
)

# Blender integration
from .blender_bridge import (
    BlenderBridge,
    BlenderBridgeConfig,
    BlenderModelInfo,
    get_blender_bridge,
    save_blender_addon,
)

# Desktop pet (DesktopMate-style)
from .desktop_pet import (
    DesktopPet,
    PetConfig,
    PetState,
    get_desktop_pet,
)

# Persistence
from .persistence import (
    AvatarPersistence,
    AvatarSettings,
    get_persistence,
    load_avatar_settings,
    load_position,
    save_avatar_settings,
    save_position,
)

# Preset system
from .presets import (
    AvatarPreset,
    PresetManager,
    get_preset_manager,
)

# Sample Avatars
from .sample_avatars import (
    SampleAvatarGenerator,
    generate_sample_avatars,
)

# Spawnable objects system (avatar can spawn items on screen)
from .spawnable_objects import (
    AttachPoint,
    ObjectSpawner,
    ObjectType,
    ObjectWindow,
    SpawnedObject,
    get_spawner,
)

# Speech synchronization (voice + lip sync)
from .speech_sync import (
    SpeechSync,
    SpeechSyncConfig,
    create_voice_avatar_bridge,
    get_speech_sync,
    set_avatar_for_sync,
    sync_speak,
)

# Template Generator
from .template_generator import (
    EXTENDED_EMOTIONS,
    STANDARD_EMOTIONS,
    SpriteTemplateGenerator,
    TemplateConfig,
    generate_template,
)

# Unified Avatar System (combines all modes)
# - PNG_BOUNCE: DougDoug style simple PNG with bounce
# - ANIMATED_2D: Desktop Mate style GIF/sprite animations
# - SKELETAL_3D: Cortana style 3D skeletal animation
from .unified_avatar import (  # Convenience functions
    AvatarConfig,
    AvatarMode,
    AvatarType,
    EmotionMapping,
    PNGBounceWidget,
    UnifiedAvatar,
    create_2d_avatar,
    create_3d_avatar,
    create_animal_avatar,
    create_png_avatar,
    create_robot_avatar,
)

# Part Editor (real-time part-by-part editing with morph transitions)
from .part_editor import (
    AvatarPart,
    AvatarPartEditor,
    MorphTransition,
    PartLayer,
    PartType,
    get_part_editor,
)

# Mesh Manipulation (vertex-level editing, morph targets, blend shapes)
from .mesh_manipulation import (
    BlendShape,
    Face,
    MeshManipulator,
    MeshRegion,
    MorphTarget,
    RegionDefinition,
    Vertex,
    get_mesh_manipulator,
)

__all__ = [
    # Controller
    "AvatarController",
    "AvatarConfig",
    "AvatarState",
    "AvatarPosition",    "ControlPriority",    "get_avatar",
    "enable_avatar",
    "disable_avatar",
    "toggle_avatar",
    "execute_action",
    
    # AI Control
    "AIAvatarControl",
    "BoneCommand",
    "get_ai_avatar_control",
    "parse_bone_commands",
    "process_ai_response",
    
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
    
    # AI-Avatar Bridge
    "AIAvatarBridge",
    "AvatarChatIntegration",
    "EmotionKeywords",
    "GestureKeywords",
    "AvatarCommand",
    "ExplicitCommands",
    "parse_explicit_commands",
    "get_command_reference",
    "create_avatar_bridge",
    "integrate_avatar_with_chat",
    "get_avatar_command_prompt",
    "process_ai_response",
    "list_avatar_commands",
    
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
    
    # Avatar Bundle System
    "AvatarBundle",
    "AvatarManifest",
    "AvatarBundleCreator",
    "install_avatar_bundle",
    "list_installed_avatars",
    
    # Multi-Avatar Manager
    "AvatarManager",
    "AvatarProfile",
    "get_avatar_manager",
    "create_avatar",
    "get_avatar_for_persona",
    "switch_avatar_for_persona",
    
    # Sentiment Analysis for Avatar Emotions
    "SentimentAnalyzer",
    "SentimentResult",
    "EmotionType",
    "analyze_for_avatar",
    "get_sentiment_analyzer",
    
    # Sample Avatars
    "SampleAvatarGenerator",
    "generate_sample_avatars",
    
    # Persistence
    "AvatarSettings",
    "AvatarPersistence",
    "get_persistence",
    "save_position",
    "load_position",
    "save_avatar_settings",
    "load_avatar_settings",
    
    # Template Generator
    "SpriteTemplateGenerator",
    "TemplateConfig",
    "generate_template",
    "STANDARD_EMOTIONS",
    "EXTENDED_EMOTIONS",
    
    # Part Editor
    "AvatarPartEditor",
    "AvatarPart",
    "PartLayer",
    "PartType",
    "MorphTransition",
    "get_part_editor",
    
    # Mesh Manipulation
    "MeshManipulator",
    "Vertex",
    "Face",
    "MeshRegion",
    "RegionDefinition",
    "MorphTarget",
    "BlendShape",
    "get_mesh_manipulator",
]
