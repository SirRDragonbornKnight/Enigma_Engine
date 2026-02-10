"""
API Key Management and Rotation

Secure storage and rotation of API keys for external services.
Supports automatic rotation, expiration tracking, and secure storage.

Usage:
    from enigma_engine.utils.api_keys import APIKeyManager, get_key_manager
    
    # Get singleton manager
    manager = get_key_manager()
    
    # Store a key
    manager.set_key("openai", "sk-...")
    
    # Retrieve a key
    key = manager.get_key("openai")
    
    # Rotate a key
    manager.rotate_key("openai", "sk-new-key...")
    
    # Check key status
    status = manager.get_key_status("openai")
"""

import base64
import hashlib
import json
import logging
import os
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

# Default key storage location
DEFAULT_KEY_PATH = Path.home() / ".enigma_engine" / "keys.enc"
SALT_PATH = Path.home() / ".enigma_engine" / ".salt"


@dataclass
class APIKeyInfo:
    """Information about a stored API key."""
    service: str
    created_at: str
    last_rotated: Optional[str] = None
    rotation_count: int = 0
    expires_at: Optional[str] = None
    last_used: Optional[str] = None
    use_count: int = 0
    masked_key: str = ""  # e.g., "sk-...xyz"
    
    def is_expired(self) -> bool:
        """Check if the key has expired."""
        if not self.expires_at:
            return False
        return datetime.fromisoformat(self.expires_at) < datetime.now()
    
    def days_until_expiry(self) -> Optional[int]:
        """Get days until key expires, or None if no expiry."""
        if not self.expires_at:
            return None
        delta = datetime.fromisoformat(self.expires_at) - datetime.now()
        return max(0, delta.days)


@dataclass
class KeyRotationPolicy:
    """Policy for automatic key rotation."""
    enabled: bool = False
    rotation_days: int = 90  # Rotate every 90 days
    warn_days_before: int = 7  # Warn 7 days before rotation needed
    require_manual_rotation: bool = True  # Don't auto-rotate, just warn


class APIKeyManager:
    """
    Secure API key storage and rotation manager.
    
    Features:
    - Encrypted storage using Fernet (AES-128-CBC)
    - Automatic key rotation reminders
    - Key usage tracking
    - Expiration management
    """
    
    # Supported services with their key patterns
    KNOWN_SERVICES = {
        "openai": {"prefix": "sk-", "display": "OpenAI"},
        "anthropic": {"prefix": "sk-ant-", "display": "Anthropic"},
        "replicate": {"prefix": "r8_", "display": "Replicate"},
        "elevenlabs": {"prefix": "", "display": "ElevenLabs"},
        "huggingface": {"prefix": "hf_", "display": "HuggingFace"},
        "stability": {"prefix": "sk-", "display": "Stability AI"},
        "custom": {"prefix": "", "display": "Custom"},
    }
    
    def __init__(
        self,
        key_path: Optional[Path] = None,
        master_password: Optional[str] = None
    ):
        """
        Initialize the API key manager.
        
        Args:
            key_path: Path to encrypted key storage file
            master_password: Password for encryption (uses machine ID if not provided)
        """
        self._key_path = key_path or DEFAULT_KEY_PATH
        self._salt_path = SALT_PATH
        self._keys: dict[str, str] = {}
        self._metadata: dict[str, APIKeyInfo] = {}
        self._policies: dict[str, KeyRotationPolicy] = {}
        self._fernet: Optional[Fernet] = None
        self._master_password = master_password
        
        # Initialize encryption
        self._init_encryption()
        
        # Load existing keys
        self._load()
    
    def _get_machine_id(self) -> str:
        """Get a unique machine identifier for default encryption."""
        import platform
        import uuid

        # Combine various system identifiers
        node = uuid.getnode()
        system = platform.system()
        machine = platform.machine()
        
        combined = f"{node}-{system}-{machine}-Enigma AI Engine"
        return hashlib.sha256(combined.encode()).hexdigest()[:32]
    
    def _init_encryption(self):
        """Initialize Fernet encryption with derived key."""
        # Get or create salt
        self._salt_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self._salt_path.exists():
            salt = self._salt_path.read_bytes()
        else:
            salt = secrets.token_bytes(16)
            self._salt_path.write_bytes(salt)
            # Secure the salt file
            try:
                os.chmod(self._salt_path, 0o600)
            except (OSError, AttributeError):
                pass  # Windows doesn't support chmod the same way
        
        # Derive key from password
        password = (self._master_password or self._get_machine_id()).encode()
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self._fernet = Fernet(key)
    
    def _mask_key(self, key: str, service: str) -> str:
        """Create a masked version of the key for display."""
        if len(key) < 8:
            return "***"
        
        # Keep prefix if known
        prefix_len = 0
        if service in self.KNOWN_SERVICES:
            prefix = self.KNOWN_SERVICES[service]["prefix"]
            if key.startswith(prefix):
                prefix_len = len(prefix)
        
        # Show prefix + first 2 chars + last 4 chars
        visible_start = prefix_len + 2
        if len(key) > visible_start + 4:
            return key[:visible_start] + "..." + key[-4:]
        return key[:4] + "..." + key[-2:]
    
    def _load(self):
        """Load encrypted keys from storage."""
        if not self._key_path.exists():
            return
        
        try:
            encrypted_data = self._key_path.read_bytes()
            decrypted = self._fernet.decrypt(encrypted_data)
            data = json.loads(decrypted.decode())
            
            self._keys = data.get("keys", {})
            
            # Load metadata
            for service, meta in data.get("metadata", {}).items():
                self._metadata[service] = APIKeyInfo(**meta)
            
            # Load policies
            for service, policy in data.get("policies", {}).items():
                self._policies[service] = KeyRotationPolicy(**policy)
                
            logger.debug(f"Loaded {len(self._keys)} API keys")
            
        except Exception as e:
            logger.warning(f"Failed to load API keys: {e}")
            self._keys = {}
            self._metadata = {}
    
    def _save(self):
        """Save encrypted keys to storage."""
        try:
            self._key_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "keys": self._keys,
                "metadata": {k: asdict(v) for k, v in self._metadata.items()},
                "policies": {k: asdict(v) for k, v in self._policies.items()},
            }
            
            encrypted = self._fernet.encrypt(json.dumps(data).encode())
            self._key_path.write_bytes(encrypted)
            
            # Secure the file
            try:
                os.chmod(self._key_path, 0o600)
            except (OSError, AttributeError):
                pass
                
            logger.debug("API keys saved")
            
        except Exception as e:
            logger.error(f"Failed to save API keys: {e}")
            raise
    
    def set_key(
        self,
        service: str,
        key: str,
        expires_in_days: Optional[int] = None
    ):
        """
        Store an API key.
        
        Args:
            service: Service identifier (e.g., "openai", "replicate")
            key: The API key value
            expires_in_days: Optional number of days until key expires
        """
        now = datetime.now().isoformat()
        
        # Create or update metadata
        if service in self._metadata:
            self._metadata[service].last_rotated = now
            self._metadata[service].rotation_count += 1
        else:
            self._metadata[service] = APIKeyInfo(
                service=service,
                created_at=now,
            )
        
        self._metadata[service].masked_key = self._mask_key(key, service)
        
        if expires_in_days:
            expires = datetime.now() + timedelta(days=expires_in_days)
            self._metadata[service].expires_at = expires.isoformat()
        
        self._keys[service] = key
        self._save()
        
        logger.info(f"API key for {service} stored")
    
    def get_key(self, service: str) -> Optional[str]:
        """
        Retrieve an API key.
        
        Args:
            service: Service identifier
            
        Returns:
            The API key or None if not found
        """
        key = self._keys.get(service)
        
        if key and service in self._metadata:
            # Update usage tracking
            self._metadata[service].last_used = datetime.now().isoformat()
            self._metadata[service].use_count += 1
            self._save()
        
        return key
    
    def rotate_key(
        self,
        service: str,
        new_key: str,
        expires_in_days: Optional[int] = None
    ):
        """
        Rotate an API key to a new value.
        
        Args:
            service: Service identifier
            new_key: The new API key
            expires_in_days: Optional new expiration
        """
        if service not in self._keys:
            raise ValueError(f"No existing key for service: {service}")
        
        old_count = self._metadata[service].rotation_count
        self.set_key(service, new_key, expires_in_days)
        
        # Restore and increment rotation count
        self._metadata[service].rotation_count = old_count + 1
        self._save()
        
        logger.info(f"API key for {service} rotated (rotation #{old_count + 1})")
    
    def delete_key(self, service: str):
        """Remove an API key."""
        if service in self._keys:
            del self._keys[service]
        if service in self._metadata:
            del self._metadata[service]
        if service in self._policies:
            del self._policies[service]
        self._save()
        
        logger.info(f"API key for {service} deleted")
    
    def get_key_status(self, service: str) -> Optional[APIKeyInfo]:
        """Get status information about a key (without the actual key)."""
        return self._metadata.get(service)
    
    def list_services(self) -> list[str]:
        """List all services with stored keys."""
        return list(self._keys.keys())
    
    def set_rotation_policy(
        self,
        service: str,
        rotation_days: int = 90,
        warn_days: int = 7,
        auto_rotate: bool = False
    ):
        """
        Set rotation policy for a service.
        
        Args:
            service: Service identifier
            rotation_days: Days between rotations
            warn_days: Days before rotation to start warning
            auto_rotate: If True, system will attempt auto-rotation
        """
        self._policies[service] = KeyRotationPolicy(
            enabled=True,
            rotation_days=rotation_days,
            warn_days_before=warn_days,
            require_manual_rotation=not auto_rotate,
        )
        self._save()
    
    def get_keys_needing_rotation(self) -> list[dict[str, Any]]:
        """
        Get list of keys that need rotation based on policy.
        
        Returns:
            List of dicts with service, reason, and days info
        """
        needs_rotation = []
        
        for service, info in self._metadata.items():
            policy = self._policies.get(service)
            
            # Check expiration
            if info.is_expired():
                needs_rotation.append({
                    "service": service,
                    "reason": "expired",
                    "days": 0,
                    "urgent": True,
                })
                continue
            
            # Check days until expiry
            days_left = info.days_until_expiry()
            if days_left is not None:
                warn_days = policy.warn_days_before if policy else 7
                if days_left <= warn_days:
                    needs_rotation.append({
                        "service": service,
                        "reason": "expiring_soon",
                        "days": days_left,
                        "urgent": days_left <= 1,
                    })
                    continue
            
            # Check rotation policy
            if policy and policy.enabled:
                last_rotated = info.last_rotated or info.created_at
                last_rotated_dt = datetime.fromisoformat(last_rotated)
                days_since = (datetime.now() - last_rotated_dt).days
                
                if days_since >= policy.rotation_days:
                    needs_rotation.append({
                        "service": service,
                        "reason": "policy",
                        "days": days_since,
                        "urgent": days_since > policy.rotation_days + 7,
                    })
                elif days_since >= policy.rotation_days - policy.warn_days_before:
                    needs_rotation.append({
                        "service": service,
                        "reason": "policy_upcoming",
                        "days": policy.rotation_days - days_since,
                        "urgent": False,
                    })
        
        return needs_rotation
    
    def export_keys(self, password: str) -> bytes:
        """
        Export all keys encrypted with a custom password.
        For backup/transfer purposes.
        
        Args:
            password: Password to encrypt the export
            
        Returns:
            Encrypted bytes that can be imported later
        """
        # Create export-specific encryption
        salt = secrets.token_bytes(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        fernet = Fernet(key)
        
        data = {
            "keys": self._keys,
            "metadata": {k: asdict(v) for k, v in self._metadata.items()},
            "exported_at": datetime.now().isoformat(),
        }
        
        encrypted = fernet.encrypt(json.dumps(data).encode())
        
        # Prepend salt
        return salt + encrypted
    
    def import_keys(self, encrypted_data: bytes, password: str, merge: bool = True):
        """
        Import keys from encrypted export.
        
        Args:
            encrypted_data: Data from export_keys()
            password: Password used during export
            merge: If True, merge with existing keys; if False, replace all
        """
        # Extract salt and decrypt
        salt = encrypted_data[:16]
        encrypted = encrypted_data[16:]
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        fernet = Fernet(key)
        
        decrypted = fernet.decrypt(encrypted)
        data = json.loads(decrypted.decode())
        
        if not merge:
            self._keys = {}
            self._metadata = {}
        
        # Import keys
        for service, key_value in data.get("keys", {}).items():
            self._keys[service] = key_value
        
        for service, meta in data.get("metadata", {}).items():
            self._metadata[service] = APIKeyInfo(**meta)
        
        self._save()
        logger.info(f"Imported {len(data.get('keys', {}))} API keys")


# Singleton instance
_manager: Optional[APIKeyManager] = None


def get_key_manager() -> APIKeyManager:
    """Get the singleton API key manager instance."""
    global _manager
    if _manager is None:
        _manager = APIKeyManager()
    return _manager


def get_api_key(service: str) -> Optional[str]:
    """Convenience function to get an API key."""
    return get_key_manager().get_key(service)


def set_api_key(service: str, key: str):
    """Convenience function to set an API key."""
    get_key_manager().set_key(service, key)
