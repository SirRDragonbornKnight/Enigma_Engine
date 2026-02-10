"""
API Key Encryption for Enigma AI Engine.

Provides secure storage and retrieval of API keys using
encryption at rest. Keys are never stored in plaintext.
"""
import base64
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Try to import cryptography for stronger encryption
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography not installed - using fallback encryption")


@dataclass
class KeyMetadata:
    """Metadata for a stored API key."""
    service: str
    description: str = ""
    created_at: str = ""
    last_used: str = ""
    masked_value: str = ""  # e.g., sk-...8x4f


class SecureKeyStorage:
    """
    Secure storage for API keys with encryption at rest.
    
    Uses Fernet symmetric encryption (AES-128-CBC with HMAC)
    when cryptography is available, with a fallback to XOR
    obfuscation for basic protection.
    
    Usage:
        storage = SecureKeyStorage()
        
        # Store a key
        storage.store_key("openai", "sk-abc123...")
        
        # Retrieve a key
        key = storage.get_key("openai")
        
        # List stored keys (shows metadata only, not actual keys)
        keys = storage.list_keys()
    """
    
    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        master_password: Optional[str] = None
    ):
        """
        Initialize secure key storage.
        
        Args:
            storage_dir: Directory for encrypted key storage
            master_password: Optional master password for key derivation
        """
        self.storage_dir = storage_dir or Path("data/.secrets")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.keys_file = self.storage_dir / "api_keys.enc"
        self.metadata_file = self.storage_dir / "api_keys_meta.json"
        
        # Initialize encryption
        self._master_password = master_password
        self._fernet = self._init_encryption()
        
        # Load existing keys
        self._keys: dict[str, str] = {}
        self._metadata: dict[str, KeyMetadata] = {}
        self._load_keys()
    
    def _init_encryption(self) -> Optional[object]:
        """Initialize Fernet encryption if available."""
        if not CRYPTO_AVAILABLE:
            return None
        
        # Get or generate salt
        salt_file = self.storage_dir / ".salt"
        if salt_file.exists():
            salt = salt_file.read_bytes()
        else:
            salt = os.urandom(16)
            salt_file.write_bytes(salt)
            # Make salt file hidden on Windows
            try:
                import ctypes
                ctypes.windll.kernel32.SetFileAttributesW(str(salt_file), 0x02)
            except (AttributeError, OSError) as e:
                logger.debug(f"Could not hide salt file: {e}")
        
        # Derive key from password or machine-specific data
        password = self._master_password or self._get_machine_key()
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        
        return Fernet(key)
    
    def _get_machine_key(self) -> str:
        """Get a machine-specific key for encryption."""
        # Combine multiple machine-specific values
        components = []
        
        # Machine ID (varies by OS)
        try:
            import platform
            components.append(platform.node())
            components.append(platform.machine())
        except Exception as e:
            logger.debug(f"Could not get platform info: {e}")
        
        # User info
        try:
            import getpass
            components.append(getpass.getuser())
        except Exception as e:
            logger.debug(f"Could not get user info: {e}")
        
        # Fallback
        if not components:
            components.append("Enigma AI Engine-default")
        
        return hashlib.sha256(":".join(components).encode()).hexdigest()
    
    def _encrypt(self, data: str) -> bytes:
        """Encrypt data."""
        if self._fernet:
            return self._fernet.encrypt(data.encode())
        else:
            # Fallback: XOR obfuscation (not secure, but better than plaintext)
            return self._xor_obfuscate(data.encode())
    
    def _decrypt(self, data: bytes) -> str:
        """Decrypt data."""
        if self._fernet:
            return self._fernet.decrypt(data).decode()
        else:
            return self._xor_obfuscate(data).decode()
    
    def _xor_obfuscate(self, data: bytes) -> bytes:
        """Simple XOR obfuscation (fallback when cryptography not available)."""
        key = self._get_machine_key().encode()
        return bytes(a ^ key[i % len(key)] for i, a in enumerate(data))
    
    def _load_keys(self):
        """Load encrypted keys from storage."""
        # Load metadata
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    meta_data = json.load(f)
                    for service, meta in meta_data.items():
                        self._metadata[service] = KeyMetadata(**meta)
            except Exception as e:
                logger.error(f"Failed to load key metadata: {e}")
        
        # Load encrypted keys
        if self.keys_file.exists():
            try:
                encrypted = self.keys_file.read_bytes()
                decrypted = self._decrypt(encrypted)
                self._keys = json.loads(decrypted)
            except Exception as e:
                logger.error(f"Failed to load encrypted keys: {e}")
                self._keys = {}
    
    def _save_keys(self):
        """Save encrypted keys to storage."""
        try:
            # Save encrypted keys
            encrypted = self._encrypt(json.dumps(self._keys))
            self.keys_file.write_bytes(encrypted)
            
            # Save metadata
            meta_data = {k: vars(v) for k, v in self._metadata.items()}
            with open(self.metadata_file, 'w') as f:
                json.dump(meta_data, f, indent=2)
            
            logger.info("API keys saved securely")
        except Exception as e:
            logger.error(f"Failed to save keys: {e}")
    
    def _mask_key(self, key: str) -> str:
        """Create a masked version of the key for display."""
        if len(key) <= 8:
            return "*" * len(key)
        return f"{key[:4]}...{key[-4:]}"
    
    def store_key(
        self,
        service: str,
        key: str,
        description: str = ""
    ) -> bool:
        """
        Store an API key securely.
        
        Args:
            service: Service name (e.g., "openai", "anthropic")
            key: The API key to store
            description: Optional description
            
        Returns:
            True if successful
        """
        try:
            from datetime import datetime
            
            self._keys[service] = key
            self._metadata[service] = KeyMetadata(
                service=service,
                description=description,
                created_at=datetime.now().isoformat(),
                last_used="",
                masked_value=self._mask_key(key)
            )
            
            self._save_keys()
            logger.info(f"Stored API key for service: {service}")
            return True
        except Exception as e:
            logger.error(f"Failed to store key: {e}")
            return False
    
    def get_key(self, service: str) -> Optional[str]:
        """
        Retrieve an API key.
        
        Args:
            service: Service name
            
        Returns:
            The API key or None if not found
        """
        key = self._keys.get(service)
        
        if key:
            # Update last used time
            from datetime import datetime
            if service in self._metadata:
                self._metadata[service].last_used = datetime.now().isoformat()
                self._save_keys()
        
        return key
    
    def delete_key(self, service: str) -> bool:
        """
        Delete an API key.
        
        Args:
            service: Service name
            
        Returns:
            True if deleted, False if not found
        """
        if service in self._keys:
            del self._keys[service]
            if service in self._metadata:
                del self._metadata[service]
            self._save_keys()
            logger.info(f"Deleted API key for service: {service}")
            return True
        return False
    
    def list_keys(self) -> dict[str, KeyMetadata]:
        """
        List all stored keys (metadata only).
        
        Returns:
            Dict mapping service names to KeyMetadata
        """
        return self._metadata.copy()
    
    def has_key(self, service: str) -> bool:
        """Check if a key exists for a service."""
        return service in self._keys
    
    def update_description(self, service: str, description: str) -> bool:
        """Update the description for a stored key."""
        if service in self._metadata:
            self._metadata[service].description = description
            self._save_keys()
            return True
        return False
    
    def export_keys(self, filepath: Path, password: str) -> bool:
        """
        Export keys to an encrypted file for backup.
        
        Args:
            filepath: Path to save the export
            password: Password to encrypt the export
            
        Returns:
            True if successful
        """
        try:
            if not CRYPTO_AVAILABLE:
                logger.error("Cannot export: cryptography not available")
                return False
            
            # Create export data
            export_data = {
                "keys": self._keys,
                "metadata": {k: vars(v) for k, v in self._metadata.items()}
            }
            
            # Encrypt with provided password
            salt = os.urandom(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=480000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            fernet = Fernet(key)
            
            encrypted = fernet.encrypt(json.dumps(export_data).encode())
            
            # Save with salt prefix
            with open(filepath, 'wb') as f:
                f.write(salt + encrypted)
            
            logger.info(f"Exported keys to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def import_keys(self, filepath: Path, password: str) -> bool:
        """
        Import keys from an encrypted backup.
        
        Args:
            filepath: Path to the import file
            password: Password to decrypt the import
            
        Returns:
            True if successful
        """
        try:
            if not CRYPTO_AVAILABLE:
                logger.error("Cannot import: cryptography not available")
                return False
            
            # Read file
            with open(filepath, 'rb') as f:
                data = f.read()
            
            # Extract salt and encrypted data
            salt = data[:16]
            encrypted = data[16:]
            
            # Decrypt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=480000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            fernet = Fernet(key)
            
            decrypted = fernet.decrypt(encrypted)
            import_data = json.loads(decrypted)
            
            # Merge with existing keys
            self._keys.update(import_data.get("keys", {}))
            for service, meta in import_data.get("metadata", {}).items():
                self._metadata[service] = KeyMetadata(**meta)
            
            self._save_keys()
            logger.info(f"Imported keys from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return False


# Global instance
_storage_instance: Optional[SecureKeyStorage] = None


def get_key_storage() -> SecureKeyStorage:
    """Get the global secure key storage instance."""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = SecureKeyStorage()
    return _storage_instance


def get_api_key(service: str) -> Optional[str]:
    """
    Convenience function to get an API key.
    
    First checks environment variables, then secure storage.
    
    Args:
        service: Service name (e.g., "openai", "anthropic")
        
    Returns:
        API key or None if not found
    """
    # Check environment variables first
    env_var_names = [
        f"{service.upper()}_API_KEY",
        f"{service.upper()}_KEY",
        f"{service}_api_key",
    ]
    
    for env_var in env_var_names:
        key = os.environ.get(env_var)
        if key:
            return key
    
    # Fall back to secure storage
    storage = get_key_storage()
    return storage.get_key(service)


def store_api_key(service: str, key: str, description: str = "") -> bool:
    """
    Convenience function to store an API key.
    
    Args:
        service: Service name
        key: API key
        description: Optional description
        
    Returns:
        True if successful
    """
    storage = get_key_storage()
    return storage.store_key(service, key, description)
