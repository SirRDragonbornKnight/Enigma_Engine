"""
Memory Encryption for Enigma AI Engine
Provides encryption/decryption for sensitive memories using Fernet (AES-128).
"""
import base64
import logging
from pathlib import Path
from typing import Optional

from .categorization import Memory, MemoryCategory, MemoryType

logger = logging.getLogger(__name__)


class MemoryEncryption:
    """Encrypt/decrypt sensitive memories."""
    
    def __init__(self, key: Optional[bytes] = None):
        """
        Initialize memory encryption.
        
        Args:
            key: Encryption key (generates new one if None)
        """
        try:
            from cryptography.fernet import Fernet
            self.Fernet = Fernet
        except ImportError:
            raise ImportError(
                "cryptography not installed. Install with: pip install cryptography"
            )
        
        if key is None:
            self.key = Fernet.generate_key()
        else:
            self.key = key
        
        self._fernet = Fernet(self.key)
    
    def encrypt(self, content: str) -> bytes:
        """
        Encrypt memory content.
        
        Args:
            content: Content to encrypt
            
        Returns:
            Encrypted bytes
        """
        if not content:
            return b''
        
        # Encode to bytes and encrypt
        content_bytes = content.encode('utf-8')
        encrypted = self._fernet.encrypt(content_bytes)
        
        return encrypted
    
    def decrypt(self, encrypted: bytes) -> str:
        """
        Decrypt memory content.
        
        Args:
            encrypted: Encrypted bytes
            
        Returns:
            Decrypted string
        """
        if not encrypted:
            return ''
        
        # Decrypt and decode
        decrypted_bytes = self._fernet.decrypt(encrypted)
        content = decrypted_bytes.decode('utf-8')
        
        return content
    
    def save_key(self, path: Path, password: Optional[str] = None):
        """
        Save encryption key to file.
        
        Args:
            path: Path to save key
            password: Optional password protection
            
        Warning:
            Password protection uses simple XOR obfuscation, NOT cryptographically
            secure. For production use, implement proper key derivation with PBKDF2
            or Argon2. This feature is provided for basic protection only.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        key_data = self.key
        
        if password:
            # WARNING: Simple password-based XOR obfuscation - NOT SECURE!
            # For production, use proper KDF like PBKDF2 or Argon2
            logger.critical(
                "Password protection uses simple obfuscation, NOT suitable for production! "
                "Implement proper key derivation for sensitive data."
            )
            
            import hashlib
            password_hash = hashlib.sha256(password.encode()).digest()
            
            # XOR key with password hash (repeating as needed)
            obfuscated = bytearray()
            for i, byte in enumerate(key_data):
                obfuscated.append(byte ^ password_hash[i % len(password_hash)])
            
            key_data = bytes(obfuscated)
        
        # Save to file
        with open(path, 'wb') as f:
            f.write(key_data)
        
        logger.info(f"Encryption key saved to {path}")
    
    def load_key(self, path: Path, password: Optional[str] = None):
        """
        Load encryption key from file.
        
        Args:
            path: Path to key file
            password: Optional password if key is protected
        """
        if not path.exists():
            raise FileNotFoundError(f"Key file not found: {path}")
        
        with open(path, 'rb') as f:
            key_data = f.read()
        
        if password:
            # Deobfuscate with password
            import hashlib
            password_hash = hashlib.sha256(password.encode()).digest()
            
            deobfuscated = bytearray()
            for i, byte in enumerate(key_data):
                deobfuscated.append(byte ^ password_hash[i % len(password_hash)])
            
            key_data = bytes(deobfuscated)
        
        self.key = key_data
        self._fernet = self.Fernet(self.key)
        
        logger.info(f"Encryption key loaded from {path}")
    
    @staticmethod
    def generate_key() -> bytes:
        """
        Generate a new encryption key.
        
        Returns:
            New Fernet key
        """
        from cryptography.fernet import Fernet
        return Fernet.generate_key()


class EncryptedMemoryCategory(MemoryCategory):
    """Memory category with automatic encryption."""
    
    def __init__(
        self,
        memory_type: MemoryType,
        encryption: MemoryEncryption,
        **kwargs
    ):
        """
        Initialize encrypted memory category.
        
        Args:
            memory_type: Type of memory
            encryption: Encryption instance
            **kwargs: Additional arguments for MemoryCategory
        """
        super().__init__(memory_type, **kwargs)
        self.encryption = encryption
    
    def add(
        self,
        id_: str,
        content: str,
        ttl: Optional[float] = None,
        importance: float = 0.5,
        metadata: Optional[dict] = None
    ) -> Memory:
        """
        Add encrypted memory.
        
        Args:
            id_: Memory ID
            content: Content to encrypt and store
            ttl: Time to live
            importance: Importance score
            metadata: Additional metadata
            
        Returns:
            Memory object (with encrypted content)
        """
        # Encrypt content
        encrypted_content = self.encryption.encrypt(content)
        
        # Store encrypted content as base64 string
        encrypted_str = base64.b64encode(encrypted_content).decode('utf-8')
        
        # Mark as encrypted in metadata
        metadata = metadata or {}
        metadata['encrypted'] = True
        
        # Add to parent category
        memory = super().add(
            id_=id_,
            content=encrypted_str,
            ttl=ttl,
            importance=importance,
            metadata=metadata
        )
        
        return memory
    
    def get(self, id_: str) -> Optional[Memory]:
        """
        Get and decrypt memory.
        
        Args:
            id_: Memory ID
            
        Returns:
            Memory with decrypted content
        """
        memory = super().get(id_)
        
        if memory and memory.metadata.get('encrypted'):
            # Decrypt content
            try:
                encrypted_bytes = base64.b64decode(memory.content)
                decrypted_content = self.encryption.decrypt(encrypted_bytes)
                
                # Create new memory with decrypted content
                memory = Memory(
                    id=memory.id,
                    content=decrypted_content,
                    memory_type=memory.memory_type,
                    timestamp=memory.timestamp,
                    ttl=memory.ttl,
                    importance=memory.importance,
                    metadata=memory.metadata,
                    access_count=memory.access_count,
                    last_accessed=memory.last_accessed
                )
                
                # Mark as accessed
                memory.access()
                
            except Exception as e:
                logger.error(f"Failed to decrypt memory {id_}: {e}")
                # Return encrypted version if decryption fails
        
        return memory
    
    def get_all(self, include_expired: bool = False, decrypt: bool = True) -> list:
        """
        Get all memories, optionally decrypted.
        
        Args:
            include_expired: Include expired memories
            decrypt: Decrypt contents
            
        Returns:
            List of memories
        """
        memories = super().get_all(include_expired)
        
        if not decrypt:
            return memories
        
        # Decrypt all memories
        decrypted = []
        for memory in memories:
            if memory.metadata.get('encrypted'):
                try:
                    encrypted_bytes = base64.b64decode(memory.content)
                    decrypted_content = self.encryption.decrypt(encrypted_bytes)
                    
                    decrypted_memory = Memory(
                        id=memory.id,
                        content=decrypted_content,
                        memory_type=memory.memory_type,
                        timestamp=memory.timestamp,
                        ttl=memory.ttl,
                        importance=memory.importance,
                        metadata=memory.metadata,
                        access_count=memory.access_count,
                        last_accessed=memory.last_accessed
                    )
                    
                    decrypted.append(decrypted_memory)
                    
                except Exception as e:
                    logger.error(f"Failed to decrypt memory {memory.id}: {e}")
                    decrypted.append(memory)  # Add encrypted version
            else:
                decrypted.append(memory)
        
        return decrypted
