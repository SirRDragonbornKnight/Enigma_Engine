"""
Quick API - Simplified access to common Enigma AI Engine operations.

This module provides single-function shortcuts for the most common tasks:

    from enigma_engine.quick import chat, search, screenshot, read, write

    # Chat with AI
    response = chat("Tell me a joke")
    
    # Search the web
    results = search("python tutorials")
    
    # Take a screenshot
    path = screenshot()
    
    # Read a file
    content = read("README.md")
    
    # Write a file
    write("notes.txt", "Hello world")

All functions return simple values on success or raise QuickError on failure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


class QuickError(Exception):
    """Error from quick API operations."""
    
    def __init__(self, message: str, operation: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.operation = operation
        self.details = details or {}
    
    def __str__(self) -> str:
        return f"[{self.operation}] {self.message}"


def _ensure_preset():
    """Ensure full tool preset is applied."""
    try:
        from .tools.tool_manager import get_tool_manager
        manager = get_tool_manager()
        if not manager.enabled_tools:
            manager.apply_preset("full")
    except Exception:
        pass  # Intentionally silent


def _execute(tool_name: str, **kwargs) -> dict[str, Any]:
    """Execute a tool and return result."""
    _ensure_preset()
    from .tools import execute_tool
    result = execute_tool(tool_name, **kwargs)
    if not result.get("success", False):
        raise QuickError(
            result.get("error", "Unknown error"),
            tool_name,
            result
        )
    return result


# =============================================================================
# Chat & AI
# =============================================================================

def chat(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """
    Chat with the AI and get a response.
    
    Args:
        prompt: Your message to the AI
        model: Model name (default: uses configured model)
        max_tokens: Maximum response length
        temperature: Creativity (0.0-1.0)
        
    Returns:
        AI's response text
        
    Example:
        >>> response = chat("What is 2 + 2?")
        >>> print(response)
        "2 + 2 equals 4."
    """
    try:
        from .core.inference import EnigmaEngine
        engine = EnigmaEngine(model_path=model)
        return engine.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
    except Exception as e:
        raise QuickError(str(e), "chat", {"prompt": prompt[:100]})


def complete(prompt: str, max_tokens: int = 256) -> str:
    """
    Complete a text prompt (no chat formatting).
    
    Args:
        prompt: Text to complete
        max_tokens: Maximum completion length
        
    Returns:
        Completion text
    """
    return chat(prompt, max_tokens=max_tokens, temperature=0.3)


# =============================================================================
# Web & Search
# =============================================================================

def search(query: str, limit: int = 5) -> list[dict[str, str]]:
    """
    Search the web.
    
    Args:
        query: Search query
        limit: Max results
        
    Returns:
        List of {title, url, snippet} dicts
        
    Example:
        >>> results = search("python tutorials")
        >>> for r in results:
        ...     print(r["title"], r["url"])
    """
    result = _execute("web_search", query=query)
    return result.get("results", [])[:limit]


def fetch(url: str) -> str:
    """
    Fetch content from a URL.
    
    Args:
        url: Web URL to fetch
        
    Returns:
        Page content as text
    """
    result = _execute("fetch_webpage", url=url)
    return result.get("content", "")


# =============================================================================
# Files
# =============================================================================

def read(path: Union[str, Path]) -> str:
    """
    Read a file's contents.
    
    Args:
        path: File path
        
    Returns:
        File content as string
    """
    result = _execute("read_file", path=str(path))
    return result.get("content", "")


def write(path: Union[str, Path], content: str, append: bool = False) -> Path:
    """
    Write content to a file.
    
    Args:
        path: File path
        content: Content to write
        append: Append instead of overwrite
        
    Returns:
        Path to written file
    """
    result = _execute("write_file", path=str(path), content=content, append=append)
    return Path(result.get("path", str(path)))


def ls(path: Union[str, Path] = ".") -> list[str]:
    """
    List directory contents.
    
    Args:
        path: Directory path (default: current)
        
    Returns:
        List of filenames
    """
    result = _execute("list_directory", path=str(path))
    items = result.get("items", result.get("files", []))
    return [f["name"] if isinstance(f, dict) else f for f in items]


# =============================================================================
# System
# =============================================================================

def screenshot(path: Optional[str] = None) -> Path:
    """
    Take a screenshot.
    
    Args:
        path: Save path (default: auto-generated)
        
    Returns:
        Path to screenshot file
    """
    kwargs = {}
    if path:
        kwargs["save_path"] = path
    result = _execute("screenshot", **kwargs)
    return Path(result.get("path", result.get("save_path", "screenshot.png")))


def sysinfo() -> dict[str, Any]:
    """
    Get system information.
    
    Returns:
        Dict with cpu, memory, disk, gpu info
    """
    result = _execute("get_system_info")
    # Unwrap nested 'info' dict if present
    if "info" in result and isinstance(result["info"], dict):
        return result["info"]
    return {k: v for k, v in result.items() if k != "success"}


def run(command: str, timeout: int = 30) -> str:
    """
    Run a shell command.
    
    Args:
        command: Command to run
        timeout: Max execution time (seconds)
        
    Returns:
        Command output
    """
    result = _execute("run_command", command=command, timeout=timeout)
    return result.get("output", "")


# =============================================================================
# Memory
# =============================================================================

def remember(fact: str, category: str = "world") -> bool:
    """
    Remember a fact.
    
    Args:
        fact: The fact to remember
        category: "user", "self", or "world"
        
    Returns:
        True if remembered
    """
    result = _execute("remember_fact", fact=fact, category=category)
    return result.get("success", False)


def recall(category: Optional[str] = None, limit: int = 10) -> list[str]:
    """
    Recall remembered facts.
    
    Args:
        category: Filter by category ("user", "self", "world", or None for all)
        limit: Max facts to return
        
    Returns:
        List of remembered facts
    """
    kwargs: dict[str, Any] = {"limit": limit}
    if category:
        kwargs["category"] = category
    result = _execute("recall_facts", **kwargs)
    return result.get("facts", [])


def search_memory(query: str, limit: int = 10) -> list[dict]:
    """
    Search conversation history.
    
    Args:
        query: Search text
        limit: Max results
        
    Returns:
        List of matching messages
    """
    result = _execute("search_memory", query=query, limit=limit)
    return result.get("results", [])


# =============================================================================
# Knowledge
# =============================================================================

def wiki(query: str, sentences: int = 3) -> str:
    """
    Search Wikipedia.
    
    Args:
        query: Search term
        sentences: Sentences to return
        
    Returns:
        Wikipedia summary text
    """
    result = _execute("wikipedia_search", query=query, sentences=sentences)
    return result.get("summary", result.get("content", ""))


def translate(text: str, to_lang: str = "en", from_lang: str = "auto") -> str:
    """
    Translate text.
    
    Args:
        text: Text to translate
        to_lang: Target language code
        from_lang: Source language code (default: auto-detect)
        
    Returns:
        Translated text
    """
    result = _execute("translate_text", text=text, target_lang=to_lang, source_lang=from_lang)
    return result.get("translated", result.get("translation", text))


# =============================================================================
# Convenience exports
# =============================================================================

__all__ = [
    "QuickError",
    # Chat
    "chat",
    "complete",
    # Web
    "search",
    "fetch",
    # Files
    "read",
    "write",
    "ls",
    # System
    "screenshot",
    "sysinfo",
    "run",
    # Memory
    "remember",
    "recall",
    "search_memory",
    # Knowledge
    "wiki",
    "translate",
]
