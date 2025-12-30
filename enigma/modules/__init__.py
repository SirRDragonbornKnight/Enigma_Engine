"""
Enigma Modules System
=====================

Every capability in Enigma is a module that can be:
- Enabled or disabled independently
- Combined with other modules
- Configured for different hardware profiles
- Distributed across devices

Module Categories:
- Core: Model, Tokenizer, Training, Inference
- Memory: Conversations, Vector DB, Long-term Memory
- Interface: GUI, API, CLI, Web
- Perception: Vision, Voice Input, Sensors
- Output: Voice Output, Avatar, Display
- Tools: Web, Files, Documents, Code
- Network: Multi-device, Distributed, Cloud
"""

from .manager import ModuleManager, Module, ModuleState
from .registry import MODULE_REGISTRY, get_module, list_modules

__all__ = [
    'ModuleManager',
    'Module', 
    'ModuleState',
    'MODULE_REGISTRY',
    'get_module',
    'list_modules',
]
