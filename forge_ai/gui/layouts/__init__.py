"""
Layouts package - Window layout management and presets.

This package contains:
- presets.py: Layout preset manager for saving/loading window layouts
"""

from .presets import (
    DockState,
    LayoutPreset,
    LayoutPresetManager,
    SplitterState,
    TabState,
    WindowGeometry,
    get_layout_manager,
    load_layout,
    save_layout,
)

__all__ = [
    'LayoutPreset',
    'LayoutPresetManager',
    'WindowGeometry',
    'SplitterState',
    'TabState',
    'DockState',
    'get_layout_manager',
    'save_layout',
    'load_layout',
]
