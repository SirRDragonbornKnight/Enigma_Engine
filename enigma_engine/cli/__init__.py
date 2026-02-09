"""
Forge CLI package.

Provides Ollama-style command-line interface.

Usage:
    forge pull forge-small
    forge run forge-small
    forge serve
    forge list
    forge chat  # Interactive chat mode
"""

from .main import main
from .chat import CLIChat

__all__ = ["main", "CLIChat"]

