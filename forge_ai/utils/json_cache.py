"""
JSON Configuration Cache
========================

Provides cached JSON file reading to avoid repeated disk I/O.
Files are only re-read if they've been modified.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache
import threading

# Thread-safe cache for JSON files
_json_cache: Dict[str, tuple] = {}  # path -> (mtime, data)
_cache_lock = threading.Lock()


def read_json_cached(path: str | Path, default: Any = None) -> Any:
    """
    Read a JSON file with caching.
    
    The file is only re-read if it has been modified since the last read.
    This significantly improves performance for config files that are
    read multiple times during a session.
    
    Args:
        path: Path to the JSON file
        default: Default value if file doesn't exist or can't be parsed
        
    Returns:
        Parsed JSON data, or default value on error
    """
    path = Path(path)
    path_str = str(path.resolve())
    
    try:
        if not path.exists():
            return default
            
        mtime = os.path.getmtime(path_str)
        
        with _cache_lock:
            # Check if we have a cached version that's still valid
            if path_str in _json_cache:
                cached_mtime, cached_data = _json_cache[path_str]
                if cached_mtime == mtime:
                    return cached_data
            
            # Read and cache
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            _json_cache[path_str] = (mtime, data)
            return data
            
    except (json.JSONDecodeError, OSError, IOError):
        return default


def write_json_cached(path: str | Path, data: Any, indent: int = 2) -> bool:
    """
    Write a JSON file and update the cache.
    
    Args:
        path: Path to the JSON file
        data: Data to serialize to JSON
        indent: JSON indentation level
        
    Returns:
        True if successful, False on error
    """
    path = Path(path)
    path_str = str(path.resolve())
    
    try:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent)
        
        # Update cache
        with _cache_lock:
            mtime = os.path.getmtime(path_str)
            _json_cache[path_str] = (mtime, data)
        
        return True
        
    except (OSError, IOError, TypeError):
        return False


def invalidate_cache(path: Optional[str | Path] = None) -> None:
    """
    Invalidate cached JSON data.
    
    Args:
        path: Specific file to invalidate, or None to clear all cache
    """
    with _cache_lock:
        if path is None:
            _json_cache.clear()
        else:
            path_str = str(Path(path).resolve())
            _json_cache.pop(path_str, None)


def get_cache_stats() -> Dict[str, int]:
    """Get cache statistics."""
    with _cache_lock:
        return {
            'cached_files': len(_json_cache),
            'total_size': sum(
                len(json.dumps(data)) 
                for _, data in _json_cache.values()
            )
        }


# Convenience function for common ForgeAI config files
@lru_cache(maxsize=1)
def get_config_paths():
    """Get common config file paths (cached)."""
    from ..config import CONFIG
    return {
        'gui_settings': CONFIG.get('DATA_DIR', Path('data')) / 'gui_settings.json',
        'tool_routing': CONFIG.get('DATA_DIR', Path('data')) / 'tool_routing.json',
        'module_config': Path('forge_ai/modules/module_config.json'),
    }


__all__ = [
    'read_json_cached',
    'write_json_cached', 
    'invalidate_cache',
    'get_cache_stats',
    'get_config_paths',
]
