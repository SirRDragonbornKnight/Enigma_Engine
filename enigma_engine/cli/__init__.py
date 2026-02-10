"""
Forge CLI package.

Provides Ollama-style command-line interface.

Usage:
    forge pull forge-small
    forge run forge-small
    forge serve
    forge list
    forge chat  # Interactive chat mode

Commands module provides reusable command handlers:
    from enigma_engine.cli.commands import cmd_train, cmd_gui, cmd_serve
"""

from .main import main
from .chat import CLIChat
from .commands import (
    cmd_train, cmd_build, cmd_serve, cmd_tunnel,
    cmd_run_cli, cmd_gui, cmd_background, cmd_web
)

__all__ = [
    "main", "CLIChat",
    "cmd_train", "cmd_build", "cmd_serve", "cmd_tunnel",
    "cmd_run_cli", "cmd_gui", "cmd_background", "cmd_web"
]

