"""
Forge CLI package.

Provides Ollama-style command-line interface.

Usage:
    forge pull forge-small
    forge run forge-small
    forge serve
    forge list
"""

from .main import main

__all__ = ["main"]
