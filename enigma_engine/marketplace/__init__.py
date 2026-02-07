"""
================================================================================
PLUGIN MARKETPLACE - SHARE AND DISCOVER MODULES
================================================================================

A marketplace system for sharing Enigma AI Engine modules with the community.
Download modules from others, share your creations, rate and review.

FILE: enigma_engine/marketplace/__init__.py
TYPE: Package Initialization
MAIN CLASSES: Marketplace, PluginInfo, PluginCategory

FEATURES:
    - Browse community modules
    - Download and install modules
    - Share your modules
    - Rate and review
    - Version management
    - Dependency resolution

USAGE:
    from enigma_engine.marketplace import Marketplace
    
    market = Marketplace()
    
    # Browse plugins
    plugins = market.search("image generation")
    
    # Install a plugin
    market.install("cool-image-filter")
    
    # Share your plugin
    market.publish("my-plugin", "/path/to/plugin")
"""

from .installer import (
    DependencyResolver,
    PluginInstaller,
)
from .marketplace import (
    InstallResult,
    Marketplace,
    PluginCategory,
    PluginInfo,
    PluginVersion,
)
from .repository import (
    LocalRepository,
    PluginRepository,
    RemoteRepository,
)

__all__ = [
    'Marketplace',
    'PluginInfo',
    'PluginCategory',
    'PluginVersion',
    'InstallResult',
    'PluginRepository',
    'LocalRepository',
    'RemoteRepository',
    'PluginInstaller',
    'DependencyResolver',
]
