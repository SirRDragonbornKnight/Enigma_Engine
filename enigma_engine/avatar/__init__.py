# avatar package
"""
Enigma AI Engine Avatar System - CORE ONLY

Provides AI-controlled avatar with direct bone control.

USAGE:
    from enigma_engine.avatar import get_avatar, get_bone_controller
    
    avatar = get_avatar()
    avatar.enable()
    
    # Direct bone control
    bones = get_bone_controller()
    bones.move_bone("head", pitch=10, yaw=-5)
    
    # AI outputs [MOVE: bone=angle] in responses
    # Parsed automatically by ai_bridge

Core Files:
- bone_control.py: Unified bone manipulation (no limits)
- ai_bridge.py: [MOVE:] parser + execution  
- animation_3d_native.py: 3D GLB/GLTF rendering
- controller.py: Main avatar controller
- desktop_pet.py: Desktop pet display
"""

# 3D Animation (real-time skeletal - NO EXTERNAL DEPENDENCIES)
from .animation_3d_native import (
    AI3DController,
    NativeAvatar3D,
)

# Main controller
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

# AI-Avatar Bridge (continuous bone control)
from .ai_bridge import (
    AIAvatarBridge,
    AvatarChatIntegration,
    AvatarCommand,
    ExplicitCommands,
    MoveBoneCommand,
    create_avatar_bridge,
    enable_smooth_streaming,
    execute_bone_commands,
    get_avatar_command_prompt,
    get_command_reference,
    integrate_avatar_with_chat,
    list_avatar_commands,
    parse_explicit_commands,
    parse_move_commands,
    process_ai_response,
)

# Bone control (core)
from .bone_control import (
    BoneController,
    BoneLimits,
    BoneState,
    FINGER_POSES,
    get_bone_controller,
)

# Desktop pet
from .desktop_pet import (
    DesktopPet,
    PetConfig,
    PetState,
    get_desktop_pet,
)

# Aliases
Avatar3DAnimator = NativeAvatar3D
AI3DAvatarController = AI3DController
HAS_3D_ANIMATION = True

__all__ = [
    # Controller
    "AvatarController",
    "AvatarConfig",
    "AvatarState",
    "AvatarPosition",
    "ControlPriority",
    "get_avatar",
    "enable_avatar",
    "disable_avatar",
    "toggle_avatar",
    "execute_action",
    
    # 3D Animation
    "NativeAvatar3D",
    "AI3DController",
    "Avatar3DAnimator",
    "AI3DAvatarController",
    "HAS_3D_ANIMATION",
    
    # AI-Avatar Bridge
    "AIAvatarBridge",
    "AvatarChatIntegration",
    "AvatarCommand",
    "ExplicitCommands",
    "MoveBoneCommand",
    "parse_explicit_commands",
    "parse_move_commands",
    "execute_bone_commands",
    "enable_smooth_streaming",
    "get_command_reference",
    "create_avatar_bridge",
    "integrate_avatar_with_chat",
    "get_avatar_command_prompt",
    "process_ai_response",
    "list_avatar_commands",
    
    # Bone Control
    "BoneController",
    "BoneLimits",
    "BoneState",
    "FINGER_POSES",
    "get_bone_controller",
    
    # Desktop Pet
    "DesktopPet",
    "PetState",
    "PetConfig",
    "get_desktop_pet",
]
