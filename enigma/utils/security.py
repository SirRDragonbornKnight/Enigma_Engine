"""
Security utilities for Enigma Engine.

Provides:
  - Path blocking: Prevent AI from accessing certain files
  - The blocked_paths and blocked_patterns settings CANNOT be modified by the AI
"""

import fnmatch
from pathlib import Path
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# These are loaded ONCE at startup and cached
# The AI cannot modify these at runtime
_BLOCKED_PATHS: List[str] = []
_BLOCKED_PATTERNS: List[str] = []
_INITIALIZED = False


def _initialize_blocks():
    """Load blocked paths from config. Called once at startup."""
    global _BLOCKED_PATHS, _BLOCKED_PATTERNS, _INITIALIZED
    
    if _INITIALIZED:
        return
    
    try:
        from ..config import CONFIG
        _BLOCKED_PATHS = list(CONFIG.get("blocked_paths", []))
        _BLOCKED_PATTERNS = list(CONFIG.get("blocked_patterns", []))
        _INITIALIZED = True
        
        if _BLOCKED_PATHS or _BLOCKED_PATTERNS:
            logger.info(f"Security: Loaded {len(_BLOCKED_PATHS)} blocked paths, {len(_BLOCKED_PATTERNS)} patterns")
    except Exception as e:
        logger.warning(f"Could not load security config: {e}")
        _INITIALIZED = True


def is_path_blocked(path: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a path is blocked from AI access.
    
    Args:
        path: The path to check
        
    Returns:
        Tuple of (is_blocked, reason)
        If blocked, reason explains why
    """
    _initialize_blocks()
    
    if not path:
        return False, None
    
    try:
        # Check BOTH resolved and unresolved paths to prevent symlink bypass
        raw_path = Path(path).expanduser()
        resolved_path = raw_path.resolve()
        
        # Check both the raw path and resolved path
        paths_to_check = [
            (str(raw_path), raw_path.name),
            (str(resolved_path), resolved_path.name),
        ]
        
        for path_str, name in paths_to_check:
            path_lower = path_str.lower()
            name_lower = name.lower()
            
            # Check explicit blocked paths
            for blocked in _BLOCKED_PATHS:
                if not blocked:
                    continue
                blocked_path = Path(blocked).expanduser().resolve()
                blocked_str = str(blocked_path).lower()
                
                # Check if path is the blocked path or inside it
                sep = "/" if "/" in path_lower else "\\"
                if path_lower == blocked_str or path_lower.startswith(blocked_str + sep):
                    return True, f"Path is in blocked location: {blocked}"
            
            # Check patterns against filename and full path
            for pattern in _BLOCKED_PATTERNS:
                if not pattern:
                    continue
                pattern_lower = pattern.lower()
                
                # Check filename
                if fnmatch.fnmatch(name_lower, pattern_lower):
                    return True, f"Filename matches blocked pattern: {pattern}"
                
                # Check full path
                if fnmatch.fnmatch(path_lower, pattern_lower):
                    return True, f"Path matches blocked pattern: {pattern}"
        
        return False, None
        
    except Exception as e:
        logger.warning(f"Error checking path security: {e}")
        # On error, default to blocking for safety
        return True, f"Security check failed: {e}"


def get_blocked_paths() -> List[str]:
    """Get list of blocked paths (read-only copy)."""
    _initialize_blocks()
    return list(_BLOCKED_PATHS)


def get_blocked_patterns() -> List[str]:
    """Get list of blocked patterns (read-only copy)."""
    _initialize_blocks()
    return list(_BLOCKED_PATTERNS)


def add_blocked_path(path: str, save: bool = True) -> bool:
    """
    Add a path to the blocked list.
    
    Note: This can only be called from user code, not from AI tools.
    The AI cannot call this function to unblock paths.
    
    Args:
        path: Path to block
        save: Whether to save to config file
        
    Returns:
        True if added successfully
    """
    global _BLOCKED_PATHS
    _initialize_blocks()
    
    if not path:
        return False
    
    # Normalize
    norm_path = str(Path(path).expanduser().resolve())
    
    if norm_path not in _BLOCKED_PATHS:
        _BLOCKED_PATHS.append(norm_path)
        
        if save:
            _save_to_config()
        
        logger.info(f"Security: Added blocked path: {norm_path}")
        return True
    
    return False


def add_blocked_pattern(pattern: str, save: bool = True) -> bool:
    """
    Add a pattern to the blocked list.
    
    Args:
        pattern: Glob pattern to block (e.g., "*.exe", "*password*")
        save: Whether to save to config file
        
    Returns:
        True if added successfully
    """
    global _BLOCKED_PATTERNS
    _initialize_blocks()
    
    if not pattern:
        return False
    
    if pattern not in _BLOCKED_PATTERNS:
        _BLOCKED_PATTERNS.append(pattern)
        
        if save:
            _save_to_config()
        
        logger.info(f"Security: Added blocked pattern: {pattern}")
        return True
    
    return False


def remove_blocked_path(path: str, save: bool = True) -> bool:
    """Remove a path from the blocked list."""
    global _BLOCKED_PATHS
    _initialize_blocks()
    
    norm_path = str(Path(path).expanduser().resolve())
    
    if norm_path in _BLOCKED_PATHS:
        _BLOCKED_PATHS.remove(norm_path)
        if save:
            _save_to_config()
        logger.info(f"Security: Removed blocked path: {norm_path}")
        return True
    
    return False


def remove_blocked_pattern(pattern: str, save: bool = True) -> bool:
    """Remove a pattern from the blocked list."""
    global _BLOCKED_PATTERNS
    _initialize_blocks()
    
    if pattern in _BLOCKED_PATTERNS:
        _BLOCKED_PATTERNS.remove(pattern)
        if save:
            _save_to_config()
        logger.info(f"Security: Removed blocked pattern: {pattern}")
        return True
    
    return False


def _save_to_config():
    """Save current blocks to config file."""
    try:
        import json
        from ..config import CONFIG
        
        # Find config file
        config_path = Path(CONFIG.get("root", ".")) / "enigma_config.json"
        
        # Load existing or create new
        if config_path.exists():
            with open(config_path, "r") as f:
                config_data = json.load(f)
        else:
            config_data = {}
        
        # Update blocks
        config_data["blocked_paths"] = _BLOCKED_PATHS
        config_data["blocked_patterns"] = _BLOCKED_PATTERNS
        
        # Save
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Security: Saved blocks to {config_path}")
        
    except Exception as e:
        logger.warning(f"Could not save security config: {e}")


# Decorator to protect functions from AI modification
def ai_cannot_call(func):
    """
    Decorator that marks a function as not callable by AI.
    
    Usage:
        @ai_cannot_call
        def sensitive_function():
            pass
    """
    func._ai_blocked = True
    return func
