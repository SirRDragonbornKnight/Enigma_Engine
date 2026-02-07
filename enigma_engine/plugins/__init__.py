"""
Enigma AI Engine Plugin System

Provides extensibility through loadable plugins for tools, tabs, themes,
and text processors.

SUBMODULES:
- templates: Base classes and scaffolding for creating plugins
"""

from .templates import (
    PLUGIN_TEMPLATES,
    PluginBase,
    PluginMetadata,
    PluginScaffold,
    ProcessorPlugin,
    TabPlugin,
    ThemePlugin,
    ToolPlugin,
    create_plugin,
)

__all__ = [
    'PluginBase',
    'PluginMetadata', 
    'ToolPlugin',
    'TabPlugin',
    'ThemePlugin',
    'ProcessorPlugin',
    'PluginScaffold',
    'create_plugin',
    'PLUGIN_TEMPLATES'
]
