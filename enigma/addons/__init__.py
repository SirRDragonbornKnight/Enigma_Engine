"""
Enigma Addons System
====================

AI generation capabilities integrated with the Enigma module system.
Enable addons through the Module Manager - each addon is a toggleable module.

ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────┐
│                      MODULE MANAGER                              │
│  (enigma/modules/manager.py - central control for everything)    │
├─────────────────────────────────────────────────────────────────┤
│                      MODULE REGISTRY                             │
│  (enigma/modules/registry.py - all modules including generation) │
├─────────────────┬───────────────────────────────────────────────┤
│  Core Modules   │  Generation Modules (wrap addons below)       │
│  - model        │  - image_gen_local  → StableDiffusionLocal    │
│  - tokenizer    │  - image_gen_api    → OpenAIImage/Replicate   │
│  - training     │  - code_gen_local   → EnigmaCode              │
│  - inference    │  - code_gen_api     → OpenAICode              │
│                 │  - video_gen_local  → LocalVideo              │
│                 │  - video_gen_api    → ReplicateVideo          │
│                 │  - audio_gen_local  → LocalTTS                │
│                 │  - audio_gen_api    → ElevenLabsTTS           │
│                 │  - embedding_local  → LocalEmbedding          │
│                 │  - embedding_api    → OpenAIEmbedding         │
├─────────────────┴───────────────────────────────────────────────┤
│                    ADDON IMPLEMENTATIONS                         │
│  (enigma/addons/builtin.py - actual AI generation code)          │
└─────────────────────────────────────────────────────────────────┘

USAGE:
    # Through Module Manager (RECOMMENDED - no conflicts):
    from enigma.modules import ModuleManager
    
    manager = ModuleManager()
    manager.load('image_gen_local')  # Load Stable Diffusion
    
    module = manager.get('image_gen_local')
    result = module.generate("a sunset", width=512, height=512)

ADDON TYPES:
    - IMAGE: Generate images (Stable Diffusion, DALL-E)
    - CODE: Generate code (Enigma model, GPT-4)
    - VIDEO: Generate videos (AnimateDiff, Replicate)
    - AUDIO: TTS and music (pyttsx3, ElevenLabs, MusicGen)
    - EMBEDDING: Vector embeddings (sentence-transformers, OpenAI)

LOCAL vs API:
    - Local: Free, private, needs GPU/hardware
    - API: Cloud-based, needs API key, pay per use
"""

from .base import (
    Addon, AddonType, AddonProvider, AddonConfig, AddonResult,
    ImageAddon, CodeAddon, VideoAddon, AudioAddon, EmbeddingAddon
)
from .manager import AddonManager


def get_builtin_addons():
    """Get dictionary of all built-in addon classes."""
    from .builtin import BUILTIN_ADDONS
    return BUILTIN_ADDONS


def get_addon_class(name: str):
    """Get a specific addon class by name."""
    from .builtin import BUILTIN_ADDONS
    return BUILTIN_ADDONS.get(name)


def create_addon(name: str, **kwargs):
    """Create an addon instance by name."""
    cls = get_addon_class(name)
    if cls:
        return cls(**kwargs)
    raise ValueError(f"Unknown addon: {name}")


__all__ = [
    # Core
    'Addon', 'AddonType', 'AddonProvider', 'AddonConfig', 'AddonResult',
    # Specialized bases
    'ImageAddon', 'CodeAddon', 'VideoAddon', 'AudioAddon', 'EmbeddingAddon',
    # Manager
    'AddonManager',
    # Helpers
    'get_builtin_addons', 'get_addon_class', 'create_addon',
]
