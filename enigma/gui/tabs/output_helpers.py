"""
Output helpers for generation tabs.

Provides common utilities for:
  - Opening files in explorer
  - Opening files in default application
  - Auto-open checkbox creation
"""

import os
import sys
import subprocess
from pathlib import Path

try:
    from PyQt5.QtWidgets import QCheckBox, QHBoxLayout
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False


def open_file_in_explorer(path: str):
    """Open file explorer with the file selected."""
    path = Path(path)
    if not path.exists():
        return
        
    if sys.platform == 'darwin':
        subprocess.run(['open', '-R', str(path)])
    elif sys.platform == 'win32':
        subprocess.run(['explorer', '/select,', str(path)])
    else:
        # Linux - open containing folder
        subprocess.run(['xdg-open', str(path.parent)])


def open_in_default_viewer(path: str):
    """Open file in the default application."""
    path = Path(path)
    if not path.exists():
        return
        
    if sys.platform == 'darwin':
        subprocess.run(['open', str(path)])
    elif sys.platform == 'win32':
        os.startfile(str(path))
    else:
        subprocess.run(['xdg-open', str(path)])


def open_folder(folder_path):
    """Open a folder in the file manager."""
    folder = Path(folder_path)
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
    
    if sys.platform == 'darwin':
        subprocess.run(['open', str(folder)])
    elif sys.platform == 'win32':
        subprocess.run(['explorer', str(folder)])
    else:
        subprocess.run(['xdg-open', str(folder)])


def create_auto_open_options(parent):
    """
    Create auto-open checkboxes for generation tabs.
    
    Returns tuple of (layout, file_checkbox, viewer_checkbox)
    
    Usage:
        layout, file_cb, viewer_cb = create_auto_open_options(self)
        main_layout.addLayout(layout)
        self.auto_open_file_cb = file_cb
        self.auto_open_viewer_cb = viewer_cb
    """
    if not HAS_PYQT:
        return None, None, None
    
    auto_layout = QHBoxLayout()
    
    file_cb = QCheckBox("Auto-open file in explorer")
    file_cb.setChecked(True)
    file_cb.setToolTip("Open the generated file in your file explorer when done")
    auto_layout.addWidget(file_cb)
    
    viewer_cb = QCheckBox("Auto-open in default app")
    viewer_cb.setChecked(False)
    viewer_cb.setToolTip("Open the file in your default application")
    auto_layout.addWidget(viewer_cb)
    
    auto_layout.addStretch()
    
    return auto_layout, file_cb, viewer_cb


def handle_generation_complete(path: str, auto_open_file: bool = True, 
                                auto_open_viewer: bool = False):
    """
    Handle auto-open after generation completes.
    
    Args:
        path: Path to generated file
        auto_open_file: Whether to open in file explorer
        auto_open_viewer: Whether to open in default viewer
    """
    if not path or not Path(path).exists():
        return
    
    if auto_open_file:
        open_file_in_explorer(path)
    
    if auto_open_viewer:
        open_in_default_viewer(path)
