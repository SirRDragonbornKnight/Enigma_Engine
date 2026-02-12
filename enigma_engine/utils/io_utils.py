"""
I/O Utility Functions for Enigma AI Engine

Provides safe, reusable functions for file operations with proper error handling.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


def safe_load_json(
    path: str | Path,
    default: T | None = None,
    log_errors: bool = True
) -> dict[str, Any] | list[Any] | T:
    """
    Safely load a JSON file with proper error handling.
    
    Args:
        path: Path to the JSON file
        default: Default value to return if loading fails (default: empty dict)
        log_errors: Whether to log errors (default: True)
        
    Returns:
        Loaded JSON data or default value on failure
        
    Example:
        >>> config = safe_load_json("config.json", default={})
        >>> tasks = safe_load_json("tasks.json", default=[])
    """
    if default is None:
        default = {}
    
    path = Path(path)
    
    if not path.exists():
        return default
    
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        if log_errors:
            logger.warning(f"Invalid JSON in {path}: {e}")
        return default
    except OSError as e:
        if log_errors:
            logger.warning(f"Failed to read {path}: {e}")
        return default
    except Exception as e:
        if log_errors:
            logger.error(f"Unexpected error loading {path}: {e}")
        return default


def safe_save_json(
    path: str | Path,
    data: Any,
    indent: int = 2,
    create_dirs: bool = True
) -> bool:
    """
    Safely save data to a JSON file.
    
    Args:
        path: Path to save the JSON file
        data: Data to serialize to JSON
        indent: JSON indentation level (default: 2)
        create_dirs: Create parent directories if needed (default: True)
        
    Returns:
        True if successful, False otherwise
        
    Example:
        >>> success = safe_save_json("config.json", {"key": "value"})
    """
    path = Path(path)
    
    try:
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, default=str)
        return True
    except (OSError, TypeError) as e:
        logger.error(f"Failed to save {path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving {path}: {e}")
        return False


def safe_read_text(
    path: str | Path,
    default: str = "",
    encoding: str = "utf-8"
) -> str:
    """
    Safely read text from a file.
    
    Args:
        path: Path to the text file
        default: Default value if reading fails
        encoding: File encoding (default: utf-8)
        
    Returns:
        File contents or default value
    """
    path = Path(path)
    
    if not path.exists():
        return default
    
    try:
        with open(path, encoding=encoding) as f:
            return f.read()
    except (OSError, UnicodeDecodeError) as e:
        logger.warning(f"Failed to read {path}: {e}")
        return default


def safe_write_text(
    path: str | Path,
    content: str,
    encoding: str = "utf-8",
    create_dirs: bool = True
) -> bool:
    """
    Safely write text to a file.
    
    Args:
        path: Path to save the file
        content: Text content to write
        encoding: File encoding (default: utf-8)
        create_dirs: Create parent directories if needed
        
    Returns:
        True if successful, False otherwise
    """
    path = Path(path)
    
    try:
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding=encoding) as f:
            f.write(content)
        return True
    except OSError as e:
        logger.error(f"Failed to write {path}: {e}")
        return False


def atomic_save_json(
    path: str | Path,
    data: Any,
    indent: int = 2,
    create_dirs: bool = True
) -> bool:
    """
    Atomically save data to a JSON file using write-to-temp-then-rename pattern.
    
    This prevents file corruption if the process is interrupted during write.
    The file is either completely written or not modified at all.
    
    Args:
        path: Path to save the JSON file
        data: Data to serialize to JSON
        indent: JSON indentation level (default: 2)
        create_dirs: Create parent directories if needed (default: True)
        
    Returns:
        True if successful, False otherwise
        
    Example:
        >>> success = atomic_save_json("config.json", {"key": "value"})
    """
    import os
    import tempfile
    
    path = Path(path)
    
    try:
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temp file in same directory (ensures same filesystem for rename)
        fd, temp_path = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp"
        )
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, default=str)
            
            # Atomic rename (on POSIX) or replace (on Windows)
            temp_file = Path(temp_path)
            temp_file.replace(path)
            return True
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except OSError:
                pass  # Intentionally silent
            raise
    except (OSError, TypeError) as e:
        logger.error(f"Failed to save {path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving {path}: {e}")
        return False
