"""
Addon Registry (DEPRECATED)
===========================

NOTE: Addons are now managed through the Module system.
Use enigma.modules.ModuleManager instead.

This file provides backwards compatibility.
"""

from typing import Dict, List, Optional, Type
from .base import Addon, AddonType


def get_addon(name: str) -> Optional[Type[Addon]]:
    """Get addon class by name. DEPRECATED: Use ModuleManager."""
    from .builtin import BUILTIN_ADDONS
    return BUILTIN_ADDONS.get(name)


def list_addons(addon_type: Optional[AddonType] = None) -> List[str]:
    """List addon names. DEPRECATED: Use ModuleManager.list_modules()."""
    from .builtin import BUILTIN_ADDONS
    if addon_type is None:
        return list(BUILTIN_ADDONS.keys())
    from .base import ImageAddon, CodeAddon, VideoAddon, AudioAddon, EmbeddingAddon
    type_map = {
        AddonType.IMAGE: ImageAddon,
        AddonType.CODE: CodeAddon,
        AddonType.VIDEO: VideoAddon,
        AddonType.AUDIO: AudioAddon,
        AddonType.EMBEDDING: EmbeddingAddon,
    }
    base_class = type_map.get(addon_type)
    if not base_class:
        return []
    return [name for name, cls in BUILTIN_ADDONS.items() if issubclass(cls, base_class)]


def register_addon(name: str, addon_class: Type[Addon]):
    """Register custom addon. DEPRECATED: Use ModuleManager.register()."""
    from .builtin import BUILTIN_ADDONS
    BUILTIN_ADDONS[name] = addon_class


# Backwards compatibility
ADDON_REGISTRY: Dict[str, Type[Addon]] = {}

def _init_registry():
    from .builtin import BUILTIN_ADDONS
    ADDON_REGISTRY.update(BUILTIN_ADDONS)

try:
    _init_registry()
except ImportError:
    pass
