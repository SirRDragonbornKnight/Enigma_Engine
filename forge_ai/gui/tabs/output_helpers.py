"""
Output helpers for generation tabs.

Provides common utilities for:
  - Opening files in explorer
  - Opening files in default application
  - Auto-open checkbox creation
"""

import os
import sys
from pathlib import Path
from typing import Union, Optional, Tuple, Any

try:
    from PyQt5.QtWidgets import QCheckBox, QHBoxLayout  # type: ignore[import]
    from PyQt5.QtCore import QUrl  # type: ignore[import]
    from PyQt5.QtGui import QDesktopServices  # type: ignore[import]
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False


def open_file_in_explorer(path: Union[str, Path]) -> None:
    """Open file explorer with the file selected."""
    file_path = Path(path)
    if not file_path.exists():
        return
    
    # Use Qt's cross-platform file opening (internal, no external tools)
    if HAS_PYQT:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(file_path.parent)))
    elif sys.platform == 'win32':
        os.startfile(str(file_path.parent))  # type: ignore[attr-defined]


def open_in_default_viewer(path: Union[str, Path]) -> None:
    """Open file in the default application."""
    file_path = Path(path)
    if not file_path.exists():
        return
    
    # Use Qt's cross-platform file opening (internal, no external tools)
    if HAS_PYQT:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(file_path)))
    elif sys.platform == 'win32':
        os.startfile(str(file_path))  # type: ignore[attr-defined]


def open_folder(folder_path: Union[str, Path]) -> None:
    """Open a folder in the file manager."""
    folder = Path(folder_path)
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
    
    # Use platform-specific methods for reliability
    if sys.platform == 'win32':
        os.startfile(str(folder))  # type: ignore[attr-defined]
    elif sys.platform == 'darwin':
        import subprocess
        subprocess.run(['open', str(folder)])
    else:
        # Linux/other - try Qt first, then xdg-open
        if HAS_PYQT:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        else:
            import subprocess
            subprocess.run(['xdg-open', str(folder)])


def create_auto_open_options(parent: Any) -> Tuple[Any, Any, Any]:
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
    
    auto_layout = QHBoxLayout()  # type: ignore[possibly-unbound]
    
    file_cb = QCheckBox("Auto-open file in explorer")  # type: ignore[possibly-unbound]
    file_cb.setChecked(True)
    file_cb.setToolTip("Open the generated file in your file explorer when done")
    auto_layout.addWidget(file_cb)
    
    viewer_cb = QCheckBox("Auto-open in default app")  # type: ignore[possibly-unbound]
    viewer_cb.setChecked(False)
    viewer_cb.setToolTip("Open the file in your default application")
    auto_layout.addWidget(viewer_cb)
    
    auto_layout.addStretch()
    
    return auto_layout, file_cb, viewer_cb


def handle_generation_complete(path: Union[str, Path], auto_open_file: bool = True, 
                                auto_open_viewer: bool = False) -> None:
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
