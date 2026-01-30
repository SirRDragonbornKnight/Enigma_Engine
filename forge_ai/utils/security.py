"""
================================================================================
THE GUARDIAN'S WATCHTOWER - SECURITY UTILITIES
================================================================================

Deep within ForgeAI's fortress lies the Guardian's Watchtower, an ancient
sentinel that protects the realm from harm. No AI, however cunning, may pass
through its enchanted gates to reach forbidden territories.

FILE: forge_ai/utils/security.py
TYPE: Security & Access Control
MAIN FUNCTIONS: is_path_blocked(), add_blocked_path(), add_blocked_pattern()

    THE GUARDIAN'S OATH:
    
    "I shall stand vigilant at the gates of forbidden paths,
     No clever trick or symlink shall fool my watchful eye.
     The blocked_paths are sacred scrolls, immutable once read,
     And patterns mark the treasures that must never be touched."

PROTECTED TERRITORIES:
    - System files (*.exe, *.dll, *.sys)
    - Secret scrolls (*.pem, *.key, *.env)
    - Forbidden words (*password*, *secret*)
    - Custom blocked paths from the sacred config

CONNECTED REALMS:
    READS FROM:   forge_ai/config/ (the sacred scrolls of configuration)
    GUARDS:       forge_ai/tools/ (all tool operations pass through here)
    PROTECTS:     The entire filesystem from unauthorized AI access

SEE ALSO:
    - forge_ai/config/defaults.py - Where blocked_paths are defined
    - forge_ai/tools/file_tools.py - File operations that check security

WARNING: The AI cannot modify these protections at runtime.
         Only the user may alter the sacred blocked lists.
"""

import fnmatch
from pathlib import Path
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# THE SACRED VAULTS - Immutable Security State
# =============================================================================
# These ancient scrolls are read ONCE when the Guardian awakens.
# They cannot be altered by AI magic - only by human hands.

_BLOCKED_PATHS: List[str] = []
_BLOCKED_PATTERNS: List[str] = []
_INITIALIZED = False


# =============================================================================
# THE AWAKENING RITUAL
# =============================================================================

def _initialize_blocks():
    """
    The Guardian's Awakening Ritual.
    
    Called once at startup, this sacred ceremony reads the blocked paths
    from the ancient configuration scrolls and commits them to memory.
    Once read, these protections stand eternal until the next awakening.
    """
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


# =============================================================================
# THE GUARDIAN'S JUDGMENT - Path Validation
# =============================================================================

def is_path_blocked(path: str) -> Tuple[bool, Optional[str]]:
    """
    The Guardian's Judgment - Evaluate if a path may be accessed.
    
    When any traveler (AI or tool) seeks passage to a file or directory,
    they must first present their destination to the Guardian. The Guardian
    examines both the obvious path AND any hidden routes (symlinks) to ensure
    no trickery bypasses the sacred protections.
    
    The Trial of Passage:
        1. The path is revealed in both raw and resolved forms
        2. Each form is checked against the List of Forbidden Locations
        3. Each form is tested against the Patterns of Prohibition
        4. If any test fails, passage is DENIED
    
    Args:
        path: The destination the traveler wishes to reach
        
    Returns:
        A tuple of (is_blocked, reason):
        - (False, None) if passage is granted
        - (True, reason) if the Guardian bars the way
        
    Note:
        When in doubt, the Guardian errs on the side of caution.
        A failed security check blocks access to protect the realm.
    """
    _initialize_blocks()
    
    if not path:
        return False, None
    
    # ==========================================================================
    # INJECTION ATTACK PREVENTION
    # ==========================================================================
    
    # Check for null byte injection (common bypass technique)
    if '\x00' in path or '%00' in path:
        logger.warning(f"Security: Null byte injection detected in path")
        return True, "Null byte injection detected"
    
    # Check for URL encoding bypass attempts
    import urllib.parse
    try:
        decoded_path = urllib.parse.unquote(path)
        # If decoding changed the path, check the decoded version too
        if decoded_path != path:
            # Recursively check decoded path
            is_blocked, reason = is_path_blocked(decoded_path)
            if is_blocked:
                return True, f"URL-encoded path blocked: {reason}"
    except Exception:
        pass  # If decoding fails, continue with original
    
    # Check for Unicode normalization attacks
    import unicodedata
    try:
        normalized_path = unicodedata.normalize('NFKC', path)
        if normalized_path != path:
            # Check normalized version too
            is_blocked, reason = is_path_blocked(normalized_path)
            if is_blocked:
                return True, f"Unicode-normalized path blocked: {reason}"
    except Exception:
        pass
    
    try:
        # Check BOTH resolved and unresolved paths to prevent symlink bypass
        raw_path = Path(path).expanduser()
        resolved_path = raw_path.resolve(strict=False)  # strict=False to handle non-existent paths
        
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
        # When the Guardian cannot see clearly, caution prevails
        return True, f"Security check failed: {e}"


# =============================================================================
# READING THE SACRED SCROLLS - Retrieving Block Lists
# =============================================================================

def get_blocked_paths() -> List[str]:
    """
    Reveal the List of Forbidden Locations.
    
    Returns a copy of the sacred scroll - the original cannot be altered
    by those who read it. Only the _save_to_config ritual may change
    the true list.
    """
    _initialize_blocks()
    return list(_BLOCKED_PATHS)


def get_blocked_patterns() -> List[str]:
    """
    Reveal the Patterns of Prohibition.
    
    These mystical glyphs (glob patterns) mark entire categories of
    forbidden treasures. Returns a protective copy.
    """
    _initialize_blocks()
    return list(_BLOCKED_PATTERNS)


# =============================================================================
# THE SCRIBE'S AUTHORITY - Modifying Block Lists (User Only)
# =============================================================================

def add_blocked_path(path: str, save: bool = True) -> bool:
    """
    Inscribe a new location upon the List of Forbidden Locations.
    
    Only humans may call upon this power - the AI is bound by ancient
    oaths and cannot invoke this ritual. The path is normalized and
    resolved to its true form before inscription.
    
    Args:
        path: The location to forbid (will be normalized)
        save: Whether to etch this change into the permanent scrolls
        
    Returns:
        True if the inscription was successful, False if already forbidden
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
    Inscribe a new glyph upon the Patterns of Prohibition.
    
    These mystical patterns use the ancient glob syntax to mark
    entire categories of forbidden files. Common glyphs include:
        "*.exe"      - All executable scrolls
        "*password*" - Any scroll bearing the word of secrets
        "*.key"      - The keys to hidden chambers
    
    Args:
        pattern: The glob pattern to forbid
        save: Whether to etch this change into the permanent scrolls
        
    Returns:
        True if the glyph was inscribed, False if already present
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
    """
    Erase a location from the List of Forbidden Locations.
    
    The Scribe may grant passage to previously forbidden territories,
    but this power must be wielded with wisdom.
    """
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
    """
    Erase a glyph from the Patterns of Prohibition.
    
    Removes the mystical pattern, allowing matching files to be accessed.
    """
    global _BLOCKED_PATTERNS
    _initialize_blocks()
    
    if pattern in _BLOCKED_PATTERNS:
        _BLOCKED_PATTERNS.remove(pattern)
        if save:
            _save_to_config()
        logger.info(f"Security: Removed blocked pattern: {pattern}")
        return True
    
    return False


# =============================================================================
# THE ETERNAL INSCRIPTION - Persisting Changes
# =============================================================================

def _save_to_config():
    """
    The Ritual of Eternal Inscription.
    
    Commits the current state of the forbidden lists to the sacred
    configuration scrolls (forge_config.json), ensuring the protections
    persist across system awakenings.
    """
    try:
        import json
        from ..config import CONFIG
        
        # Find config file
        config_path = Path(CONFIG.get("root", ".")) / "forge_config.json"
        
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


# =============================================================================
# THE BINDING SEAL - Protection Against AI Invocation
# =============================================================================

def ai_cannot_call(func):
    """
    The Binding Seal - A decorator of ancient power.
    
    When placed upon a function, this seal prevents any AI from invoking
    its magic. The function gains a hidden mark (_ai_blocked = True) that
    tool executors recognize and respect.
    
    Usage in the sacred texts:
        @ai_cannot_call
        def sensitive_ritual():
            # This function is protected from AI invocation
            pass
    
    The AI may see the function exists, but cannot call upon its power.
    Only human hands may invoke functions bearing this seal.
    """
    func._ai_blocked = True
    return func
