"""
Conversation Encryption for Enigma AI Engine

End-to-end encryption for stored conversations.

Features:
- AES-256 encryption
- Key derivation from password
- Encrypted storage
- Secure key management
- Transparent encryption/decryption

Usage:
    from enigma_engine.utils.encryption import ConversationEncryption
    
    enc = ConversationEncryption()
    enc.set_password("my_secret_password")
    
    # Encrypt a message
    encrypted = enc.encrypt("Hello, this is private!")
    
    # Decrypt
    message = enc.decrypt(encrypted)
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EncryptedData:
    """Container for encrypted data."""
    ciphertext: bytes
    iv: bytes
    salt: bytes
    tag: Optional[bytes] = None
    version: int = 1


@dataclass
class EncryptionConfig:
    """Encryption configuration."""
    algorithm: str = "AES-256-GCM"
    key_derivation: str = "PBKDF2"
    iterations: int = 100000
    salt_length: int = 16
    iv_length: int = 12  # For GCM


class KeyManager:
    """Manage encryption keys."""
    
    def __init__(self, key_file: Optional[Path] = None):
        """
        Initialize key manager.
        
        Args:
            key_file: Path to store encrypted master key
        """
        self.key_file = key_file or Path("memory/.keystore")
        self._master_key: Optional[bytes] = None
        self._password_hash: Optional[bytes] = None
    
    def derive_key(
        self,
        password: str,
        salt: bytes,
        iterations: int = 100000
    ) -> bytes:
        """
        Derive encryption key from password.
        
        Args:
            password: User password
            salt: Random salt
            iterations: PBKDF2 iterations
            
        Returns:
            32-byte key
        """
        return hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            iterations,
            dklen=32
        )
    
    def generate_key(self) -> bytes:
        """Generate a random encryption key."""
        return secrets.token_bytes(32)
    
    def set_password(self, password: str):
        """
        Set up encryption with a password.
        
        Args:
            password: User password
        """
        # Generate salt
        salt = secrets.token_bytes(16)
        
        # Derive key from password
        self._master_key = self.derive_key(password, salt)
        
        # Store password hash (for verification)
        self._password_hash = hashlib.sha256(password.encode()).digest()
        
        # Save salt (not the key!)
        self._save_salt(salt)
    
    def verify_password(self, password: str) -> bool:
        """Verify a password."""
        if self._password_hash is None:
            return False
        
        test_hash = hashlib.sha256(password.encode()).digest()
        return hmac.compare_digest(test_hash, self._password_hash)
    
    def get_key(self) -> Optional[bytes]:
        """Get the current encryption key."""
        return self._master_key
    
    def _save_salt(self, salt: bytes):
        """Save salt to file."""
        self.key_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "salt": base64.b64encode(salt).decode(),
            "version": 1
        }
        
        self.key_file.write_text(json.dumps(data))
    
    def _load_salt(self) -> Optional[bytes]:
        """Load salt from file."""
        if not self.key_file.exists():
            return None
        
        try:
            data = json.loads(self.key_file.read_text())
            return base64.b64decode(data["salt"])
        except Exception:
            return None


class ConversationEncryption:
    """Encrypt and decrypt conversations."""
    
    def __init__(self, config: Optional[EncryptionConfig] = None):
        """
        Initialize encryption.
        
        Args:
            config: Encryption configuration
        """
        self.config = config or EncryptionConfig()
        self._key_manager = KeyManager()
        self._initialized = False
        
        # Try to import cryptography
        self._crypto_available = False
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            self._crypto_available = True
        except ImportError:
            logger.warning("cryptography library not installed - using fallback")
    
    def set_password(self, password: str):
        """
        Set up encryption with password.
        
        Args:
            password: User password
        """
        self._key_manager.set_password(password)
        self._initialized = True
    
    def is_initialized(self) -> bool:
        """Check if encryption is initialized."""
        return self._initialized and self._key_manager.get_key() is not None
    
    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt text.
        
        Args:
            plaintext: Text to encrypt
            
        Returns:
            Base64-encoded encrypted data
        """
        if not self.is_initialized():
            raise ValueError("Encryption not initialized. Call set_password() first.")
        
        key = self._key_manager.get_key()
        
        if self._crypto_available:
            return self._encrypt_aes_gcm(plaintext, key)
        else:
            return self._encrypt_fallback(plaintext, key)
    
    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt text.
        
        Args:
            ciphertext: Base64-encoded encrypted data
            
        Returns:
            Decrypted text
        """
        if not self.is_initialized():
            raise ValueError("Encryption not initialized. Call set_password() first.")
        
        key = self._key_manager.get_key()
        
        if self._crypto_available:
            return self._decrypt_aes_gcm(ciphertext, key)
        else:
            return self._decrypt_fallback(ciphertext, key)
    
    def encrypt_conversation(self, messages: List[Dict]) -> str:
        """
        Encrypt a conversation (list of messages).
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Encrypted JSON string
        """
        plaintext = json.dumps(messages)
        return self.encrypt(plaintext)
    
    def decrypt_conversation(self, encrypted: str) -> List[Dict]:
        """
        Decrypt a conversation.
        
        Args:
            encrypted: Encrypted conversation data
            
        Returns:
            List of message dictionaries
        """
        plaintext = self.decrypt(encrypted)
        return json.loads(plaintext)
    
    def _encrypt_aes_gcm(self, plaintext: str, key: bytes) -> str:
        """Encrypt using AES-GCM."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        
        # Generate IV
        iv = secrets.token_bytes(self.config.iv_length)
        
        # Encrypt
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(iv, plaintext.encode('utf-8'), None)
        
        # Combine IV + ciphertext
        combined = iv + ciphertext
        
        # Base64 encode
        return base64.b64encode(combined).decode('utf-8')
    
    def _decrypt_aes_gcm(self, ciphertext: str, key: bytes) -> str:
        """Decrypt using AES-GCM."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        
        # Base64 decode
        combined = base64.b64decode(ciphertext)
        
        # Extract IV and ciphertext
        iv = combined[:self.config.iv_length]
        encrypted = combined[self.config.iv_length:]
        
        # Decrypt
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(iv, encrypted, None)
        
        return plaintext.decode('utf-8')
    
    def _encrypt_fallback(self, plaintext: str, key: bytes) -> str:
        """Simple XOR-based fallback encryption (NOT SECURE - for testing only)."""
        logger.warning("Using fallback encryption - install 'cryptography' for security")
        
        # Simple XOR (not secure!)
        data = plaintext.encode('utf-8')
        encrypted = bytes([b ^ key[i % len(key)] for i, b in enumerate(data)])
        
        return base64.b64encode(encrypted).decode('utf-8')
    
    def _decrypt_fallback(self, ciphertext: str, key: bytes) -> str:
        """Simple XOR-based fallback decryption."""
        encrypted = base64.b64decode(ciphertext)
        decrypted = bytes([b ^ key[i % len(key)] for i, b in enumerate(encrypted)])
        
        return decrypted.decode('utf-8')


class EncryptedStorage:
    """Encrypted file storage for conversations."""
    
    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        encryption: Optional[ConversationEncryption] = None
    ):
        """
        Initialize encrypted storage.
        
        Args:
            storage_dir: Directory for encrypted files
            encryption: Encryption instance
        """
        self.storage_dir = storage_dir or Path("memory/encrypted")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self._encryption = encryption or ConversationEncryption()
    
    def set_password(self, password: str):
        """Set encryption password."""
        self._encryption.set_password(password)
    
    def save_conversation(
        self,
        conversation_id: str,
        messages: List[Dict],
        metadata: Optional[Dict] = None
    ):
        """
        Save an encrypted conversation.
        
        Args:
            conversation_id: Unique conversation ID
            messages: List of messages
            metadata: Optional metadata
        """
        if not self._encryption.is_initialized():
            raise ValueError("Storage not initialized. Call set_password() first.")
        
        # Prepare data
        data = {
            "id": conversation_id,
            "messages": messages,
            "metadata": metadata or {},
            "saved_at": time.time()
        }
        
        # Encrypt
        encrypted = self._encryption.encrypt(json.dumps(data))
        
        # Save
        file_path = self.storage_dir / f"{conversation_id}.enc"
        file_path.write_text(encrypted)
    
    def load_conversation(self, conversation_id: str) -> Optional[Dict]:
        """
        Load an encrypted conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Decrypted conversation data, or None if not found
        """
        if not self._encryption.is_initialized():
            raise ValueError("Storage not initialized. Call set_password() first.")
        
        file_path = self.storage_dir / f"{conversation_id}.enc"
        
        if not file_path.exists():
            return None
        
        try:
            encrypted = file_path.read_text()
            decrypted = self._encryption.decrypt(encrypted)
            return json.loads(decrypted)
        except Exception as e:
            logger.error(f"Failed to decrypt conversation: {e}")
            return None
    
    def list_conversations(self) -> List[str]:
        """List all encrypted conversation IDs."""
        return [
            f.stem
            for f in self.storage_dir.glob("*.enc")
        ]
    
    def delete_conversation(self, conversation_id: str):
        """Delete an encrypted conversation."""
        file_path = self.storage_dir / f"{conversation_id}.enc"
        
        if file_path.exists():
            # Secure delete - overwrite before removing
            file_path.write_bytes(secrets.token_bytes(file_path.stat().st_size))
            file_path.unlink()


# Global instances
_encryption: Optional[ConversationEncryption] = None
_storage: Optional[EncryptedStorage] = None


def get_encryption() -> ConversationEncryption:
    """Get or create global encryption instance."""
    global _encryption
    if _encryption is None:
        _encryption = ConversationEncryption()
    return _encryption


def get_encrypted_storage() -> EncryptedStorage:
    """Get or create global encrypted storage."""
    global _storage
    if _storage is None:
        _storage = EncryptedStorage()
    return _storage


def encrypt_text(text: str, password: str) -> str:
    """Quick encrypt function."""
    enc = ConversationEncryption()
    enc.set_password(password)
    return enc.encrypt(text)


def decrypt_text(ciphertext: str, password: str) -> str:
    """Quick decrypt function."""
    enc = ConversationEncryption()
    enc.set_password(password)
    return enc.decrypt(ciphertext)
