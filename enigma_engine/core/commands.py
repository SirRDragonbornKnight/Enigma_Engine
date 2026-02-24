from __future__ import annotations
"""
Command Registry - AI-controlled command system.

The AI emits [CMD]command arg1 arg2[/CMD] blocks.
This module parses and executes them.

Commands are registered by category (config, model, system, file, etc.)
and can be executed by name. CLI-only - no GUI dependencies.
"""

from typing import Callable, Any, Optional
from dataclasses import dataclass
import json
import re


@dataclass
class CommandResult:
    """Result of a command execution."""
    success: bool
    message: str
    data: Any = None


@dataclass
class Command:
    """A registered command."""
    name: str
    handler: Callable
    description: str
    usage: str


class CommandRegistry:
    """
    Central registry for all commands.
    
    Commands are registered by category (gui, config, model, etc.)
    and can be executed by name.
    """
    
    def __init__(self):
        self._commands: dict[str, Command] = {}
        self._context: dict[str, Any] = {}
        self._history: list[str] = []  # Track executed commands
    
    def register(self, name: str, handler: Callable, description: str = "", usage: str = "") -> None:
        """Register a command."""
        self._commands[name] = Command(
            name=name,
            handler=handler,
            description=description or f"Execute {name}",
            usage=usage or name
        )
    
    def set_context(self, key: str, value: Any) -> None:
        """Set context value (engine, window, etc.)."""
        self._context[key] = value
    
    def get_context(self, key: str) -> Any:
        """Get context value."""
        return self._context.get(key)
    
    def execute(self, command_str: str) -> CommandResult:
        """
        Execute a command string.
        
        Args:
            command_str: Full command like "config.set temperature 0.7"
            
        Returns:
            CommandResult with success/failure and message
        """
        parts = command_str.strip().split()
        if not parts:
            return CommandResult(False, "[ERROR] Empty command")
        
        cmd_name = parts[0]
        args = parts[1:] if len(parts) > 1 else []
        
        if cmd_name not in self._commands:
            return CommandResult(False, f"[ERROR] Unknown command: {cmd_name}")
        
        cmd = self._commands[cmd_name]
        
        # Track in history (skip history command itself to avoid recursion)
        if cmd_name != "history":
            self._history.append(command_str.strip())
            # Keep only last 100 commands
            if len(self._history) > 100:
                self._history = self._history[-100:]
        
        try:
            result = cmd.handler(args, self._context)
            if isinstance(result, CommandResult):
                return result
            return CommandResult(True, f"[OK] {result}")
        except Exception as e:
            return CommandResult(False, f"[ERROR] {cmd_name}: {e}")
    
    def get_history(self, count: int = 20) -> list[str]:
        """Get recent command history."""
        return self._history[-count:]
    
    def list_commands(self, prefix: str = "") -> list[Command]:
        """List all commands, optionally filtered by prefix."""
        if prefix:
            return [c for c in self._commands.values() if c.name.startswith(prefix)]
        return list(self._commands.values())
    
    def get_help(self, cmd_name: str = "") -> str:
        """Get help text for a command or all commands."""
        if cmd_name:
            if cmd_name in self._commands:
                cmd = self._commands[cmd_name]
                return f"{cmd.name}\n  {cmd.description}\n  Usage: {cmd.usage}"
            return f"Unknown command: {cmd_name}"
        
        # All commands grouped by category
        categories: dict[str, list[Command]] = {}
        for cmd in self._commands.values():
            cat = cmd.name.split(".")[0] if "." in cmd.name else "general"
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(cmd)
        
        lines = ["Available commands:"]
        for cat in sorted(categories.keys()):
            lines.append(f"\n{cat.upper()}")
            for cmd in sorted(categories[cat], key=lambda c: c.name):
                lines.append(f"  {cmd.usage:<30} - {cmd.description}")
        
        return "\n".join(lines)


def parse_commands(text: str) -> tuple:
    """
    Extract [CMD]...[/CMD] blocks from AI response.
    
    Args:
        text: Full AI response text
        
    Returns:
        (clean_text, commands_list) - text without command blocks, list of commands
    """
    commands = re.findall(r'\[CMD\](.*?)\[/CMD\]', text, re.DOTALL)
    commands = [cmd.strip() for cmd in commands]
    clean = re.sub(r'\[CMD\].*?\[/CMD\]', '', text, flags=re.DOTALL).strip()
    return clean, commands


# Global registry instance
_registry: Optional[CommandRegistry] = None


def get_registry() -> CommandRegistry:
    """Get the global command registry."""
    global _registry
    if _registry is None:
        _registry = CommandRegistry()
        from .builtin_commands import register_builtin_commands
        register_builtin_commands(_registry)
    return _registry
