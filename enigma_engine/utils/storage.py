"""
JSON Storage Utilities for Enigma AI Engine.

Provides a mixin class for persistent JSON storage, eliminating
duplicate load/save implementations across the codebase.
"""

import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any, Optional

logger = logging.getLogger(__name__)


class JsonStorageMixin:
    """
    Mixin class providing JSON file persistence.
    
    Classes using this mixin should set `_storage_path` attribute
    to the Path where data should be stored.
    
    Usage:
        class MyConfig(JsonStorageMixin):
            def __init__(self, path: Path):
                self._storage_path = path
                self._data = self._load_json()
            
            def save(self):
                self._save_json(self._data)
    """
    
    _storage_path: Optional[Path] = None
    _storage_lock: Lock = Lock()
    
    def _load_json(self, default: Optional[dict] = None) -> dict[str, Any]:
        """
        Load data from JSON file.
        
        Args:
            default: Default value if file doesn't exist or is invalid
            
        Returns:
            Loaded data or default value
        """
        if default is None:
            default = {}
        
        if self._storage_path is None:
            logger.warning("Storage path not set, returning default")
            return default
        
        path = Path(self._storage_path)
        
        if not path.exists():
            return default
        
        try:
            with self._storage_lock:
                with open(path, encoding='utf-8') as f:
                    data = json.load(f)
                    if not isinstance(data, dict):
                        logger.warning(f"Invalid data type in {path}, expected dict")
                        return default
                    return data
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in {path}: {e}")
            return default
        except OSError as e:
            logger.warning(f"Could not read {path}: {e}")
            return default
        except Exception as e:
            logger.error(f"Unexpected error loading {path}: {e}")
            return default
    
    def _save_json(self, data: dict[str, Any], indent: int = 2) -> bool:
        """
        Save data to JSON file.
        
        Args:
            data: Data to save
            indent: JSON indentation level
            
        Returns:
            True if saved successfully, False otherwise
        """
        if self._storage_path is None:
            logger.warning("Storage path not set, cannot save")
            return False
        
        path = Path(self._storage_path)
        
        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with self._storage_lock:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=indent, ensure_ascii=False)
            return True
        except OSError as e:
            logger.error(f"Could not write {path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving {path}: {e}")
            return False


def load_json_file(path: Path, default: Optional[dict] = None) -> dict[str, Any]:
    """
    Standalone function to load a JSON file.
    
    Args:
        path: Path to JSON file
        default: Default value if file doesn't exist or is invalid
        
    Returns:
        Loaded data or default value
    """
    if default is None:
        default = {}
    
    path = Path(path)
    
    if not path.exists():
        return default
    
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
            return data if isinstance(data, dict) else default
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not load {path}: {e}")
        return default


def save_json_file(path: Path, data: dict[str, Any], indent: int = 2) -> bool:
    """
    Standalone function to save data to a JSON file.
    
    Args:
        path: Path to JSON file
        data: Data to save
        indent: JSON indentation level
        
    Returns:
        True if saved successfully
    """
    path = Path(path)
    
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except (OSError, TypeError) as e:
        logger.error(f"Could not save {path}: {e}")
        return False
