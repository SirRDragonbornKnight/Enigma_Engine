"""
Enigma Addons System
====================

Addons extend Enigma's capabilities beyond text generation.
Each addon provides a specific AI capability:

- Image Generation (DALL-E, Stable Diffusion, local models)
- Code Generation (specialized code models)
- Video Generation (Runway, Pika, local)
- Audio Generation (music, sound effects)
- Speech Synthesis (advanced TTS)
- Translation (multi-language)
- And any custom capability you want!

Addons can connect to:
- Local models (run on your hardware)
- Remote APIs (OpenAI, Anthropic, Stability, etc.)
- Custom servers (your own deployments)
- Other Enigma instances (distributed)
"""

from .base import Addon, AddonType, AddonProvider, AddonConfig
from .manager import AddonManager
from .registry import ADDON_REGISTRY, register_addon, get_addon

__all__ = [
    'Addon',
    'AddonType',
    'AddonProvider', 
    'AddonConfig',
    'AddonManager',
    'ADDON_REGISTRY',
    'register_addon',
    'get_addon',
]
