# type: ignore
"""
GUI Dialogs Package - Modal dialogs extracted from enhanced_window.py

This package contains:
- loading.py: ModelLoadingDialog for showing loading progress
- model_manager.py: ModelManagerDialog for managing AI models
"""

from .loading import ModelLoadingDialog
from .model_manager import ModelManagerDialog

__all__ = ['ModelLoadingDialog', 'ModelManagerDialog']
