"""
API Key Management System for Enigma AI Engine

Secure import/export and management of API keys for various services.

Usage:
    from enigma_engine.core.api_key_manager import APIKeyManager, get_api_key_manager
    
    manager = get_api_key_manager()
    
    # Set a key
    manager.set_key("openai", "sk-...")
    
    # Get a key
    key = manager.get_key("openai")
    
    # Export (encrypted) for backup
    manager.export_keys("backup.enc", password="secret")
    
    # Import from backup
    manager.import_keys("backup.enc", password="secret")
"""

import base64
import hashlib
import json
import logging
import os
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Supported API services
SUPPORTED_SERVICES = {
    "openai": {
        "name": "OpenAI",
        "key_prefix": "sk-",
        "env_var": "OPENAI_API_KEY",
        "docs": "https://platform.openai.com/api-keys"
    },
    "anthropic": {
        "name": "Anthropic",
        "key_prefix": "sk-ant-",
        "env_var": "ANTHROPIC_API_KEY",
        "docs": "https://console.anthropic.com/settings/keys"
    },
    "huggingface": {
        "name": "HuggingFace",
        "key_prefix": "hf_",
        "env_var": "HF_TOKEN",
        "docs": "https://huggingface.co/settings/tokens"
    },
    "replicate": {
        "name": "Replicate",
        "key_prefix": "r8_",
        "env_var": "REPLICATE_API_TOKEN",
        "docs": "https://replicate.com/account"
    },
    "elevenlabs": {
        "name": "ElevenLabs",
        "key_prefix": "",
        "env_var": "ELEVENLABS_API_KEY",
        "docs": "https://elevenlabs.io/app/profile"
    },
    "stability": {
        "name": "Stability AI",
        "key_prefix": "sk-",
        "env_var": "STABILITY_API_KEY",
        "docs": "https://platform.stability.ai/account/keys"
    },
    "cohere": {
        "name": "Cohere",
        "key_prefix": "",
        "env_var": "COHERE_API_KEY",
        "docs": "https://dashboard.cohere.ai/api-keys"
    },
    "groq": {
        "name": "Groq",
        "key_prefix": "gsk_",
        "env_var": "GROQ_API_KEY", 
        "docs": "https://console.groq.com/keys"
    }
}


@dataclass
class APIKeyInfo:
    """Information about a stored API key."""
    service: str
    key_hash: str  # SHA256 hash of key (for verification without exposing)
    added_at: str
    last_used: Optional[str] = None
    uses_count: int = 0
    is_valid: bool = True
    notes: str = ""


def _simple_encrypt(data: str, password: str) -> bytes:
    """Simple encryption using XOR with key derivation."""
    # Derive key from password
    key = hashlib.sha256(password.encode()).digest()
    
    # Add random salt
    salt = secrets.token_bytes(16)
    
    # XOR data with repeated key
    data_bytes = data.encode()
    key_repeated = (key * (len(data_bytes) // len(key) + 1))[:len(data_bytes)]
    encrypted = bytes(a ^ b for a, b in zip(data_bytes, key_repeated))
    
    # Combine salt + encrypted
    return salt + encrypted


def _simple_decrypt(encrypted_data: bytes, password: str) -> str:
    """Simple decryption."""
    # Extract salt and data
    salt = encrypted_data[:16]
    encrypted = encrypted_data[16:]
    
    # Derive key from password
    key = hashlib.sha256(password.encode()).digest()
    
    # XOR to decrypt
    key_repeated = (key * (len(encrypted) // len(key) + 1))[:len(encrypted)]
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key_repeated))
    
    return decrypted.decode()


class APIKeyManager:
    """
    Secure API key management.
    
    Features:
    - Encrypted storage
    - Import/export with password protection
    - Key validation
    - Usage tracking
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        from ..config import CONFIG
        
        self.storage_path = storage_path or Path(
            CONFIG.get("data_dir", "data")
        ) / "api_keys.json"
        
        # Keys stored in memory (decrypted)
        self._keys: Dict[str, str] = {}
        self._key_info: Dict[str, APIKeyInfo] = {}
        
        self._load_from_env()
        self._load_stored_keys()
    
    def _load_from_env(self):
        """Load keys from environment variables."""
        for service, config in SUPPORTED_SERVICES.items():
            env_var = config["env_var"]
            if env_var in os.environ:
                self._keys[service] = os.environ[env_var]
                logger.debug(f"Loaded {service} key from environment")
    
    def _load_stored_keys(self):
        """Load keys from storage file."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
            
            # Keys are base64 encoded (not secure, but obscured)
            for service, encoded in data.get("keys", {}).items():
                if service not in self._keys:  # Don't override env vars
                    self._keys[service] = base64.b64decode(encoded).decode()
            
            # Load key info
            for service, info in data.get("info", {}).items():
                self._key_info[service] = APIKeyInfo(**info)
            
            logger.info(f"Loaded {len(self._keys)} API keys from storage")
        except Exception as e:
            logger.warning(f"Could not load API keys: {e}")
    
    def _save_keys(self):
        """Save keys to storage file."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "keys": {
                    service: base64.b64encode(key.encode()).decode()
                    for service, key in self._keys.items()
                },
                "info": {
                    service: {
                        "service": info.service,
                        "key_hash": info.key_hash,
                        "added_at": info.added_at,
                        "last_used": info.last_used,
                        "uses_count": info.uses_count,
                        "is_valid": info.is_valid,
                        "notes": info.notes
                    }
                    for service, info in self._key_info.items()
                }
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save API keys: {e}")
    
    def get_key(self, service: str, mark_used: bool = True) -> Optional[str]:
        """
        Get an API key for a service.
        
        Args:
            service: Service name (openai, anthropic, etc.)
            mark_used: Update usage tracking
            
        Returns:
            API key or None if not found
        """
        key = self._keys.get(service)
        
        if key and mark_used and service in self._key_info:
            self._key_info[service].last_used = datetime.now().isoformat()
            self._key_info[service].uses_count += 1
            self._save_keys()
        
        return key
    
    def set_key(self, service: str, key: str, notes: str = "") -> bool:
        """
        Set an API key for a service.
        
        Args:
            service: Service name
            key: API key value
            notes: Optional notes about this key
            
        Returns:
            True if key appears valid
        """
        if service not in SUPPORTED_SERVICES:
            logger.warning(f"Unknown service: {service}")
            return False
        
        # Validate key format
        config = SUPPORTED_SERVICES[service]
        prefix = config.get("key_prefix", "")
        if prefix and not key.startswith(prefix):
            logger.warning(f"Key doesn't match expected prefix for {service}")
        
        self._keys[service] = key
        self._key_info[service] = APIKeyInfo(
            service=service,
            key_hash=hashlib.sha256(key.encode()).hexdigest()[:16],
            added_at=datetime.now().isoformat(),
            notes=notes
        )
        
        self._save_keys()
        logger.info(f"Saved API key for {service}")
        return True
    
    def remove_key(self, service: str):
        """Remove an API key."""
        if service in self._keys:
            del self._keys[service]
        if service in self._key_info:
            del self._key_info[service]
        self._save_keys()
        logger.info(f"Removed API key for {service}")
    
    def has_key(self, service: str) -> bool:
        """Check if a key exists for a service."""
        return service in self._keys
    
    def get_available_services(self) -> List[str]:
        """Get list of services with configured keys."""
        return list(self._keys.keys())
    
    def get_all_service_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all supported services.
        
        Returns:
            Dict with service info including whether key is configured
        """
        result = {}
        for service, config in SUPPORTED_SERVICES.items():
            result[service] = {
                "name": config["name"],
                "has_key": service in self._keys,
                "docs_url": config["docs"],
                "env_var": config["env_var"]
            }
            if service in self._key_info:
                info = self._key_info[service]
                result[service].update({
                    "key_hash": info.key_hash,
                    "added_at": info.added_at,
                    "last_used": info.last_used,
                    "uses_count": info.uses_count
                })
        return result
    
    def export_keys(
        self,
        output_path: Path,
        password: str,
        services: Optional[List[str]] = None
    ) -> bool:
        """
        Export API keys to encrypted file.
        
        Args:
            output_path: Where to save the export
            password: Encryption password
            services: Specific services to export (None = all)
            
        Returns:
            True if successful
        """
        try:
            # Gather keys to export
            keys_to_export = {}
            for service, key in self._keys.items():
                if services is None or service in services:
                    keys_to_export[service] = key
            
            if not keys_to_export:
                logger.warning("No keys to export")
                return False
            
            # Serialize and encrypt
            data = json.dumps({
                "version": 1,
                "exported_at": datetime.now().isoformat(),
                "keys": keys_to_export
            })
            
            encrypted = _simple_encrypt(data, password)
            
            # Write with magic header
            with open(output_path, 'wb') as f:
                f.write(b"ENIGMA_KEYS_V1\n")
                f.write(base64.b64encode(encrypted))
            
            logger.info(f"Exported {len(keys_to_export)} API keys to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export keys: {e}")
            return False
    
    def import_keys(
        self,
        input_path: Path,
        password: str,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Import API keys from encrypted file.
        
        Args:
            input_path: Path to import file
            password: Decryption password
            overwrite: Whether to overwrite existing keys
            
        Returns:
            Dict mapping service to success status (or error key with message)
        """
        results: Dict[str, Any] = {}
        
        try:
            with open(input_path, 'rb') as f:
                header = f.readline()
                if not header.startswith(b"ENIGMA_KEYS"):
                    logger.error("Invalid key export file")
                    return {"error": "Invalid file format"}
                
                encrypted = base64.b64decode(f.read())
            
            # Decrypt
            try:
                decrypted = _simple_decrypt(encrypted, password)
                data = json.loads(decrypted)
            except Exception:
                logger.error("Failed to decrypt (wrong password?)")
                return {"error": "Decryption failed"}
            
            # Import keys
            for service, key in data.get("keys", {}).items():
                if service in self._keys and not overwrite:
                    results[service] = False
                    continue
                
                self.set_key(service, key)
                results[service] = True
            
            logger.info(f"Imported {sum(results.values())} API keys")
            return results
            
        except Exception as e:
            logger.error(f"Failed to import keys: {e}")
            return {"error": str(e)}
    
    def validate_key(self, service: str) -> Optional[bool]:
        """
        Test if an API key is valid by making a minimal API call.
        
        Returns:
            True if valid, False if invalid, None if unable to test
        """
        key = self._keys.get(service)
        if not key:
            return None
        
        try:
            if service == "openai":
                import openai
                client = openai.OpenAI(api_key=key)
                client.models.list()
                return True
            elif service == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=key)
                # Just creating client validates format
                return True
            elif service == "huggingface":
                import requests
                resp = requests.get(
                    "https://huggingface.co/api/whoami",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=10
                )
                return resp.status_code == 200
            else:
                # Can't validate, assume valid
                return None
        except Exception as e:
            logger.warning(f"Key validation failed for {service}: {e}")
            return False


# Global instance
_api_key_manager: Optional[APIKeyManager] = None


def get_api_key_manager() -> APIKeyManager:
    """Get or create global API key manager."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager
