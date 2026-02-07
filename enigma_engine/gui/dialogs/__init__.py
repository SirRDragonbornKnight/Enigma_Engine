# type: ignore
"""
GUI Dialogs Package - Modal dialogs extracted from enhanced_window.py

This package contains:
- loading.py: ModelLoadingDialog for showing loading progress
- model_manager.py: ModelManagerDialog for managing AI models
- command_palette.py: VS Code-style command palette (Ctrl+K)
- theme_editor.py: Visual theme editor for custom themes
"""

from .command_palette import (
    Command,
    CommandPaletteDialog,
    CommandRegistry,
    get_command_registry,
    register_default_commands,
    setup_command_palette_shortcut,
)
from .loading import ModelLoadingDialog
from .model_manager import ModelManagerDialog
from .theme_editor import ThemeEditorDialog, show_theme_editor

__all__ = [
    'ModelLoadingDialog',
    'ModelManagerDialog',
    'CommandPaletteDialog',
    'CommandRegistry',
    'Command',
    'get_command_registry',
    'register_default_commands',
    'setup_command_palette_shortcut',
    'ThemeEditorDialog',
    'show_theme_editor',
]
