"""
================================================================================
THE GUARDIAN'S WATCHTOWER - SECURITY UTILITIES
================================================================================

Deep within Enigma AI Engine's fortress lies the Guardian's Watchtower, an ancient
sentinel that protects the realm from harm. No AI, however cunning, may pass
through its enchanted gates to reach forbidden territories.

FILE: enigma_engine/utils/security.py
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
    READS FROM:   enigma_engine/config/ (the sacred scrolls of configuration)
    GUARDS:       enigma_engine/tools/ (all tool operations pass through here)
    PROTECTS:     The entire filesystem from unauthorized AI access

SEE ALSO:
    - enigma_engine/config/defaults.py - Where blocked_paths are defined
    - enigma_engine/tools/file_tools.py - File operations that check security

WARNING: The AI cannot modify these protections at runtime.
         Only the user may alter the sacred blocked lists.
"""

import fnmatch
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# THE SACRED VAULTS - Immutable Security State
# =============================================================================
# These ancient scrolls are read ONCE when the Guardian awakens.
# They cannot be altered by AI magic - only by human hands.

_BLOCKED_PATHS: list[str] = []
_BLOCKED_PATTERNS: list[str] = []
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

def is_path_blocked(path: str) -> tuple[bool, Optional[str]]:
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
        pass  # Intentionally silent
    
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

def get_blocked_paths() -> list[str]:
    """
    Reveal the List of Forbidden Locations.
    
    Returns a copy of the sacred scroll - the original cannot be altered
    by those who read it. Only the _save_to_config ritual may change
    the true list.
    """
    _initialize_blocks()
    return list(_BLOCKED_PATHS)


def get_blocked_patterns() -> list[str]:
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
            with open(config_path) as f:
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


# =============================================================================
# THE VAULT OF SAFE LOADING - Secure Deserialization
# =============================================================================

class SecureLoader:
    """
    The Vault of Safe Loading - Protection against malicious serialized data.
    
    Pickle files and torch checkpoints can execute arbitrary code when loaded.
    This guardian validates and safely loads serialized data, protecting
    against poisoned models and data files.
    
    THREATS GUARDED AGAINST:
    - Pickle deserialization attacks (arbitrary code execution)
    - Malicious torch checkpoints
    - Tampered model files
    - Path traversal in archives
    """
    
    # File extensions that are known to be dangerous
    DANGEROUS_EXTENSIONS = {'.pkl', '.pickle', '.pt', '.pth', '.ckpt', '.bin'}
    
    # Hash algorithms for integrity verification
    HASH_ALGORITHMS = {'sha256', 'sha512', 'blake2b'}
    
    @staticmethod
    def compute_hash(path: Path, algorithm: str = 'sha256') -> str:
        """
        Compute cryptographic hash of a file.
        
        Args:
            path: Path to file
            algorithm: Hash algorithm (sha256, sha512, blake2b)
            
        Returns:
            Hex digest of file hash
        """
        import hashlib
        
        if algorithm not in SecureLoader.HASH_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        h = hashlib.new(algorithm)
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()
    
    @staticmethod
    def verify_hash(path: Path, expected_hash: str, algorithm: str = 'sha256') -> bool:
        """
        Verify file integrity against expected hash.
        
        Args:
            path: Path to file
            expected_hash: Expected hash value
            algorithm: Hash algorithm used
            
        Returns:
            True if hash matches, False otherwise
        """
        actual_hash = SecureLoader.compute_hash(path, algorithm)
        return actual_hash.lower() == expected_hash.lower()
    
    @staticmethod
    def safe_torch_load(
        path: Path,
        map_location: str = 'cpu',
        weights_only: bool = True,
        expected_hash: Optional[str] = None
    ) -> Any:
        """
        Safely load a torch checkpoint.
        
        Args:
            path: Path to checkpoint
            map_location: Device to map tensors to
            weights_only: If True, only load weights (safer). If False, allows
                         full checkpoint loading (required for some legacy models)
            expected_hash: Optional hash to verify before loading
            
        Returns:
            Loaded checkpoint data
            
        Raises:
            SecurityError: If hash verification fails
            RuntimeError: If loading fails
        """
        import torch
        
        path = Path(path)
        
        # Verify hash if provided
        if expected_hash:
            if not SecureLoader.verify_hash(path, expected_hash):
                logger.error(f"Security: Hash mismatch for {path}")
                raise SecurityError(f"File integrity check failed: {path}")
        
        # Log warning for unsafe loading
        if not weights_only:
            logger.warning(
                f"Security: Loading {path} with weights_only=False. "
                "This allows arbitrary code execution. Only do this for trusted files."
            )
        
        try:
            return torch.load(path, map_location=map_location, weights_only=weights_only)
        except Exception as e:
            # If weights_only fails, it might be a legacy checkpoint
            if weights_only and "weights_only" in str(e):
                logger.warning(
                    f"weights_only=True failed for {path}. "
                    "This is a legacy checkpoint that requires unsafe loading."
                )
                raise RuntimeError(
                    f"Cannot safely load {path}. It requires weights_only=False. "
                    "If you trust this file, load it explicitly with weights_only=False."
                ) from e
            raise
    
    @staticmethod
    def safe_pickle_load(
        path: Path,
        expected_hash: Optional[str] = None,
        allowed_modules: Optional[set] = None
    ) -> Any:
        """
        Load pickle file with restricted unpickler for safety.
        
        Args:
            path: Path to pickle file
            expected_hash: Optional hash to verify before loading
            allowed_modules: Set of allowed module names (default: safe set)
            
        Returns:
            Unpickled data
            
        Raises:
            SecurityError: If hash verification fails or dangerous class found
        """
        import pickle
        
        path = Path(path)
        
        # Verify hash if provided
        if expected_hash:
            if not SecureLoader.verify_hash(path, expected_hash):
                logger.error(f"Security: Hash mismatch for pickle file {path}")
                raise SecurityError(f"File integrity check failed: {path}")
        
        # Default allowed modules (safe data types only)
        if allowed_modules is None:
            allowed_modules = {
                'builtins', 'collections', 'datetime', 'numpy', 'numpy.core',
                'numpy.core.multiarray', 'torch', 'torch._utils'
            }
        
        class RestrictedUnpickler(pickle.Unpickler):
            """Unpickler that restricts which classes can be loaded."""
            
            def find_class(self, module, name):
                # Allow only safe modules
                module_base = module.split('.')[0]
                if module_base not in allowed_modules and module not in allowed_modules:
                    logger.warning(f"Security: Blocked unpickle of {module}.{name}")
                    raise SecurityError(
                        f"Pickle contains untrusted class: {module}.{name}"
                    )
                return super().find_class(module, name)
        
        with open(path, 'rb') as f:
            return RestrictedUnpickler(f).load()
    
    @staticmethod
    def is_safe_to_load(path: Path) -> tuple[bool, str]:
        """
        Check if a file is safe to load.
        
        Returns:
            (is_safe, reason) - True if likely safe, or False with reason
        """
        path = Path(path)
        
        # Check if file exists
        if not path.exists():
            return False, "File does not exist"
        
        # Check for dangerous extensions
        if path.suffix.lower() in SecureLoader.DANGEROUS_EXTENSIONS:
            return False, f"File type {path.suffix} can execute code when loaded"
        
        # Check file size (suspiciously small might be malicious)
        if path.stat().st_size < 100:
            return False, "File suspiciously small"
        
        return True, "File appears safe"


class SecurityError(Exception):
    """Raised when a security check fails."""


# =============================================================================
# THE LOG SENTINEL - Sensitive Data Sanitization
# =============================================================================

class LogSanitizer:
    """
    The Log Sentinel - Prevents sensitive information from being logged.
    
    Scans log messages for sensitive patterns (API keys, passwords, paths)
    and redacts them before they reach log files.
    """
    
    # Patterns to redact
    SENSITIVE_PATTERNS = [
        (r'(api[_-]?key|apikey)["\s:=]+["\']?([a-zA-Z0-9_-]{20,})["\']?', r'\1=***REDACTED***'),
        (r'(password|passwd|pwd)["\s:=]+["\']?([^\s"\']+)["\']?', r'\1=***REDACTED***'),
        (r'(secret|token)["\s:=]+["\']?([a-zA-Z0-9_-]{10,})["\']?', r'\1=***REDACTED***'),
        (r'(sk-[a-zA-Z0-9]{20,})', '***API_KEY***'),  # OpenAI key pattern
        (r'(hf_[a-zA-Z0-9]{20,})', '***HF_TOKEN***'),  # HuggingFace token
        (r'(bearer\s+[a-zA-Z0-9_-]{20,})', 'Bearer ***REDACTED***'),  # Bearer tokens
    ]
    
    @classmethod
    def sanitize(cls, message: str) -> str:
        """
        Sanitize a log message by redacting sensitive information.
        
        Args:
            message: Log message to sanitize
            
        Returns:
            Sanitized message
        """
        import re
        
        result = message
        for pattern, replacement in cls.SENSITIVE_PATTERNS:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
    
    @classmethod
    def sanitize_exception(cls, exc: Exception) -> str:
        """
        Sanitize exception message for safe logging/display.
        
        Args:
            exc: Exception to sanitize
            
        Returns:
            Sanitized error message
        """
        return cls.sanitize(str(exc))


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

def safe_load_model(path: Path, **kwargs) -> Any:
    """Convenience wrapper for SecureLoader.safe_torch_load()"""
    return SecureLoader.safe_torch_load(path, **kwargs)

def safe_load_pickle(path: Path, **kwargs) -> Any:
    """Convenience wrapper for SecureLoader.safe_pickle_load()"""
    return SecureLoader.safe_pickle_load(path, **kwargs)

def sanitize_log(message: str) -> str:
    """Convenience wrapper for LogSanitizer.sanitize()"""
    return LogSanitizer.sanitize(message)
