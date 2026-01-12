"""
Avatar Renderer Package

Provides different rendering backends for the avatar system:
- SpriteRenderer: 2D sprite-based rendering (default, lightweight)
- QtAvatarRenderer: PyQt5 overlay window
- WebAvatarRenderer: Web dashboard integration
"""

from .base import BaseRenderer
from .sprite_renderer import SpriteRenderer
from .qt_renderer import QtAvatarRenderer
from .web_renderer import WebAvatarRenderer
from .default_sprites import generate_sprite, SPRITE_TEMPLATES

__all__ = [
    "BaseRenderer",
    "SpriteRenderer",
    "QtAvatarRenderer",
    "WebAvatarRenderer",
    "generate_sprite",
    "SPRITE_TEMPLATES",
]
