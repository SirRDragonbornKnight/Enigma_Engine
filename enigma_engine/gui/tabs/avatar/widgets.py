"""
Avatar Widget Aliases - Convenient imports for avatar widgets.

This module provides convenient access to all avatar display widgets
without needing to know their exact location.

Usage:
    from enigma_engine.gui.tabs.avatar.widgets import (
        AvatarOverlayWindow,
        DragBarWidget,
        FloatingDragBar,
        AvatarHitLayer,
        BoneHitRegion,
        BoneHitManager,
        Avatar3DOverlayWindow,
        AvatarPreviewWidget,
    )

For the OpenGL 3D rendering widget:
    from enigma_engine.gui.tabs.avatar.avatar_rendering import OpenGL3DWidget
"""

# Re-export all avatar display widgets for convenience
from .avatar_display import (
    AvatarOverlayWindow,
    DragBarWidget,
    FloatingDragBar,
    AvatarHitLayer,
    BoneHitRegion,
    BoneHitManager,  
    Avatar3DOverlayWindow,
    AvatarPreviewWidget,
    # Helper functions
    _log_avatar_activity,
    # Constants
    IMAGE_EXTENSIONS,
    MODEL_3D_EXTENSIONS,
    ALL_AVATAR_EXTENSIONS,
    AVATAR_CONFIG_DIR,
    AVATAR_MODELS_DIR,
    AVATAR_IMAGES_DIR,
)

# Re-export OpenGL widget
from .avatar_rendering import OpenGL3DWidget, HAS_TRIMESH, HAS_OPENGL

__all__ = [
    # Overlay windows
    'AvatarOverlayWindow',
    'Avatar3DOverlayWindow',
    # Drag widgets
    'DragBarWidget',
    'FloatingDragBar',
    # Hit detection
    'AvatarHitLayer',
    'BoneHitRegion',
    'BoneHitManager',
    # Preview
    'AvatarPreviewWidget',
    # OpenGL
    'OpenGL3DWidget',
    'HAS_TRIMESH',
    'HAS_OPENGL',
    # Helpers
    '_log_avatar_activity',
    # Constants
    'IMAGE_EXTENSIONS',
    'MODEL_3D_EXTENSIONS', 
    'ALL_AVATAR_EXTENSIONS',
    'AVATAR_CONFIG_DIR',
    'AVATAR_MODELS_DIR',
    'AVATAR_IMAGES_DIR',
]
