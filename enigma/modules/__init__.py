"""
Enigma Modules System
=====================

EVERYTHING in Enigma is a toggleable module. This prevents conflicts
and allows mixing capabilities as needed.

Module Categories:
┌────────────────┬───────────────────────────────────────────────┐
│ CORE           │ model, tokenizer, training, inference         │
│ GENERATION     │ image_gen, code_gen, video_gen, audio_gen     │
│ MEMORY         │ memory, embedding_local, embedding_api        │
│ PERCEPTION     │ voice_input, vision                           │
│ OUTPUT         │ voice_output, avatar                          │
│ TOOLS          │ web_tools, file_tools                         │
│ NETWORK        │ api_server, network (multi-device)            │
│ INTERFACE      │ gui                                           │
└────────────────┴───────────────────────────────────────────────┘

Usage:
    from enigma.modules import ModuleManager

    # Create manager (singleton recommended)
    manager = ModuleManager()

    # Load specific modules
    manager.load('model')
    manager.load('inference')
    manager.load('image_gen_local')  # Generation addon as module

    # Use modules
    model = manager.get('model')
    image_gen = manager.get('image_gen_local')

    # Unload when done
    manager.unload('image_gen_local')

    # Prevent conflicts - manager ensures:
    # - Only one module with same 'provides' loaded at a time
    # - Dependencies loaded first
    # - Resources freed on unload

Conflict Prevention:
    - image_gen_local and image_gen_api both provide 'image_generation'
    - Only ONE should be loaded at a time
    - Manager warns if you try to load both

    Same for:
    - code_gen_local / code_gen_api
    - video_gen_local / video_gen_api
    - audio_gen_local / audio_gen_api
    - embedding_local / embedding_api
"""

from .manager import ModuleManager, Module, ModuleState, ModuleCategory, ModuleInfo
from .registry import (
    MODULE_REGISTRY,
    get_module,
    list_modules,
    list_by_category,
    register_all
)

__all__ = [
    # Core classes
    'ModuleManager',
    'Module',
    'ModuleState',
    'ModuleCategory',
    'ModuleInfo',
    # Registry
    'MODULE_REGISTRY',
    'get_module',
    'list_modules',
    'list_by_category',
    'register_all',
]
