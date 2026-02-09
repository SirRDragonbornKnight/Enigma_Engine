"""
Conversation Encryption for Enigma AI Engine

End-to-end encryption for stored conversations.

Features:
- AES-256 encryption
- Key derivation (PBKDF2)
- Encrypted storage
- Secure key management

Usage:
    from enigma_engine.utils.conversation_encryption import ConversationEncryption
    
    enc = ConversationEncryption()
    enc.set_password("my_secure_password")
    
    # Encrypt conversation
    encrypted = enc.encrypt_conversation(messages)
    
    # Decrypt
    messages = enc.decrypt_conversation(encrypted)
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EncryptedPayload:
    """Encrypted data container."""
    ciphertext: bytes
    salt: bytes
    iv: bytes
    tag: Optional[bytes] = None  # For GCM mode
    version: int = 1


class AESCipher:
    """AES encryption using cryptography library."""
    
    def __init__(self):
        self._key: Optional[bytes] = None
        self._backend_available = False
        
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.backends import default_backend
            self._backend_available = True
        except ImportError:
            logger.warning("cryptography library not available, using fallback")
    
    def derive_key(
        self,
        password: str,
        salt: bytes,
        iterations: int = 100000
    ) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        if self._backend_available:
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.backends import default_backend
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=iterations,
                backend=default_backend()
            )
            return kdf.derive(password.encode('utf-8'))
        else:
            # Fallback using hashlib
            return hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt,
                iterations,
                dklen=32
            )
    
    def encrypt(
        self,
        plaintext: bytes,
        key: bytes
    ) -> EncryptedPayload:
        """Encrypt data using AES-GCM."""
        iv = secrets.token_bytes(12)  # 96-bit IV for GCM
        
        if self._backend_available:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.backends import default_backend
            
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(plaintext) + encryptor.finalize()
            
            return EncryptedPayload(
                ciphertext=ciphertext,
                salt=b'',  # Salt handled separately
                iv=iv,
                tag=encryptor.tag
            )
        else:
            # Simple XOR fallback (NOT SECURE - for testing only)
            logger.warning("Using insecure XOR fallback - install cryptography!")
            ciphertext = bytes(p ^ key[i % len(key)] for i, p in enumerate(plaintext))
            
            return EncryptedPayload(
                ciphertext=ciphertext,
                salt=b'',
                iv=iv
            )
    
    def decrypt(
        self,
        payload: EncryptedPayload,
        key: bytes
    ) -> bytes:
        """Decrypt data."""
        if self._backend_available:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.backends import default_backend
            
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(payload.iv, payload.tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            return decryptor.update(payload.ciphertext) + decryptor.finalize()
        else:
            # XOR fallback
            return bytes(c ^ key[i % len(key)] for i, c in enumerate(payload.ciphertext))


class ConversationEncryption:
    """Encrypt and decrypt conversations."""
    
    def __init__(self):
        """Initialize encryption."""
        self._cipher = AESCipher()
        self._key: Optional[bytes] = None
        self._salt: Optional[bytes] = None
        
        logger.info("ConversationEncryption initialized")
    
    def set_password(self, password: str):
        """
        Set encryption password.
        
        Args:
            password: User password for encryption
        """
        self._salt = secrets.token_bytes(16)
        self._key = self._cipher.derive_key(password, self._salt)
        logger.info("Encryption password set")
    
    def set_key_from_file(self, key_file: str, password: str):
        """
        Load encryption key from file.
        
        Args:
            key_file: Path to key file
            password: Password to decrypt key file
        """
        path = Path(key_file)
        if not path.exists():
            raise FileNotFoundError(f"Key file not found: {key_file}")
        
        with open(path, 'rb') as f:
            data = json.loads(f.read())
        
        self._salt = base64.b64decode(data['salt'])
        self._key = self._cipher.derive_key(password, self._salt)
    
    def save_key_file(self, key_file: str):
        """
        Save encryption salt to file.
        
        Args:
            key_file: Path to save key file
        """
        if not self._salt:
            raise ValueError("No salt set - call set_password first")
        
        data = {
            'salt': base64.b64encode(self._salt).decode('utf-8'),
            'version': 1
        }
        
        path = Path(key_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"Key file saved to {key_file}")
    
    def encrypt_conversation(
        self,
        messages: List[Dict[str, Any]]
    ) -> str:
        """
        Encrypt conversation messages.
        
        Args:
            messages: List of message dicts
            
        Returns:
            Base64-encoded encrypted data
        """
        if not self._key:
            raise ValueError("No encryption key set")
        
        # Serialize
        plaintext = json.dumps(messages).encode('utf-8')
        
        # Encrypt
        payload = self._cipher.encrypt(plaintext, self._key)
        
        # Encode for storage
        result = {
            'ciphertext': base64.b64encode(payload.ciphertext).decode('utf-8'),
            'iv': base64.b64encode(payload.iv).decode('utf-8'),
            'salt': base64.b64encode(self._salt).decode('utf-8'),
            'version': payload.version
        }
        
        if payload.tag:
            result['tag'] = base64.b64encode(payload.tag).decode('utf-8')
        
        return base64.b64encode(json.dumps(result).encode('utf-8')).decode('utf-8')
    
    def decrypt_conversation(
        self,
        encrypted_data: str
    ) -> List[Dict[str, Any]]:
        """
        Decrypt conversation messages.
        
        Args:
            encrypted_data: Base64-encoded encrypted data
            
        Returns:
            List of message dicts
        """
        if not self._key:
            raise ValueError("No encryption key set")
        
        # Decode
        data = json.loads(base64.b64decode(encrypted_data))
        
        payload = EncryptedPayload(
            ciphertext=base64.b64decode(data['ciphertext']),
            salt=base64.b64decode(data['salt']),
            iv=base64.b64decode(data['iv']),
            tag=base64.b64decode(data['tag']) if 'tag' in data else None,
            version=data.get('version', 1)
        )
        
        # Decrypt
        plaintext = self._cipher.decrypt(payload, self._key)
        
        return json.loads(plaintext.decode('utf-8'))
    
    def encrypt_file(
        self,
        input_path: str,
        output_path: str
    ):
        """
        Encrypt a file.
        
        Args:
            input_path: Path to input file
            output_path: Path to save encrypted file
        """
        if not self._key:
            raise ValueError("No encryption key set")
        
        with open(input_path, 'rb') as f:
            plaintext = f.read()
        
        payload = self._cipher.encrypt(plaintext, self._key)
        
        data = {
            'ciphertext': base64.b64encode(payload.ciphertext).decode('utf-8'),
            'iv': base64.b64encode(payload.iv).decode('utf-8'),
            'salt': base64.b64encode(self._salt).decode('utf-8'),
            'version': payload.version
        }
        
        if payload.tag:
            data['tag'] = base64.b64encode(payload.tag).decode('utf-8')
        
        with open(output_path, 'w') as f:
            json.dump(data, f)
    
    def decrypt_file(
        self,
        input_path: str,
        output_path: str
    ):
        """
        Decrypt a file.
        
        Args:
            input_path: Path to encrypted file
            output_path: Path to save decrypted file
        """
        if not self._key:
            raise ValueError("No encryption key set")
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        payload = EncryptedPayload(
            ciphertext=base64.b64decode(data['ciphertext']),
            salt=base64.b64decode(data['salt']),
            iv=base64.b64decode(data['iv']),
            tag=base64.b64decode(data['tag']) if 'tag' in data else None,
            version=data.get('version', 1)
        )
        
        plaintext = self._cipher.decrypt(payload, self._key)
        
        with open(output_path, 'wb') as f:
            f.write(plaintext)


class EncryptedConversationStore:
    """Store conversations with encryption."""
    
    def __init__(self, storage_dir: str):
        """
        Initialize store.
        
        Args:
            storage_dir: Directory for encrypted conversations
        """
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        
        self._encryption = ConversationEncryption()
        self._unlocked = False
    
    def unlock(self, password: str) -> bool:
        """
        Unlock store with password.
        
        Args:
            password: Encryption password
            
        Returns:
            True if unlocked
        """
        key_file = self._storage_dir / '.key'
        
        if key_file.exists():
            try:
                self._encryption.set_key_from_file(str(key_file), password)
                self._unlocked = True
                return True
            except Exception as e:
                logger.error(f"Failed to unlock: {e}")
                return False
        else:
            # First time - create key
            self._encryption.set_password(password)
            self._encryption.save_key_file(str(key_file))
            self._unlocked = True
            return True
    
    def lock(self):
        """Lock store."""
        self._unlocked = False
        self._encryption = ConversationEncryption()
    
    def save_conversation(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]]
    ):
        """Save encrypted conversation."""
        if not self._unlocked:
            raise ValueError("Store is locked")
        
        encrypted = self._encryption.encrypt_conversation(messages)
        
        path = self._storage_dir / f"{conversation_id}.enc"
        with open(path, 'w') as f:
            f.write(encrypted)
    
    def load_conversation(
        self,
        conversation_id: str
    ) -> List[Dict[str, Any]]:
        """Load and decrypt conversation."""
        if not self._unlocked:
            raise ValueError("Store is locked")
        
        path = self._storage_dir / f"{conversation_id}.enc"
        if not path.exists():
            return []
        
        with open(path, 'r') as f:
            encrypted = f.read()
        
        return self._encryption.decrypt_conversation(encrypted)
    
    def list_conversations(self) -> List[str]:
        """List all conversation IDs."""
        return [
            p.stem for p in self._storage_dir.glob("*.enc")
        ]
    
    def delete_conversation(self, conversation_id: str):
        """Delete conversation."""
        path = self._storage_dir / f"{conversation_id}.enc"
        if path.exists():
            path.unlink()
