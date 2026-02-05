"""
Storage Abstraction - Re-exports from storage_backends.py

DEPRECATED: Use forge_ai.utils.storage_backends directly.
This file maintains backward compatibility.
"""

from enum import Enum
from typing import Dict, Any, Optional, Union, BinaryIO

# Re-export everything from the main module
from forge_ai.utils.storage_backends import (
    StorageObject,
    StorageBackend,
    LocalStorage,
    MemoryStorage,
    S3Storage,
    AzureStorage,
    get_storage,
    get_local_storage,
    get_memory_storage,
)


# Exceptions - kept for backward compatibility
class StorageError(Exception):
    """Base exception for storage errors."""
    pass


class ObjectNotFoundError(StorageError):
    """Raised when object is not found."""
    pass


class PermissionDeniedError(StorageError):
    """Raised when permission is denied."""
    pass


# Enum for storage types
class StorageType(Enum):
    """Types of storage backends."""
    LOCAL = "local"
    MEMORY = "memory"
    S3 = "s3"
    AZURE = "azure"
    GCS = "gcs"


# Alias for backward compatibility
StorageMetadata = StorageObject
ListResult = list  # Simplified


class StorageManager:
    """
    Manager for multiple storage backends.
    
    Usage:
        manager = StorageManager()
        manager.add_backend("local", LocalStorage("/data"))
        manager.add_backend("s3", S3Storage("my-bucket"))
        manager.put_all("shared/file.txt", b"data")
    """
    
    def __init__(self):
        self._backends: Dict[str, StorageBackend] = {}
        self._default: Optional[str] = None
    
    def add_backend(self, name: str, backend: StorageBackend, default: bool = False):
        """Add a storage backend."""
        self._backends[name] = backend
        if default or self._default is None:
            self._default = name
    
    def get(self, name: Optional[str] = None) -> StorageBackend:
        """Get a backend by name (or default)."""
        name = name or self._default
        if name not in self._backends:
            raise StorageError(f"Unknown storage backend: {name}")
        return self._backends[name]
    
    def put_all(self, key: str, data: Union[bytes, BinaryIO, str], **kwargs) -> Dict[str, Any]:
        """Put object to all backends."""
        results = {}
        data_bytes = data.encode('utf-8') if isinstance(data, str) else (data.read() if hasattr(data, 'read') else data)
        
        for name, backend in self._backends.items():
            try:
                results[name] = backend.put(key, data_bytes, **kwargs)
            except Exception as e:
                results[name] = str(e)
        return results
    
    def delete_all(self, key: str) -> Dict[str, bool]:
        """Delete object from all backends."""
        return {name: backend.delete(key) for name, backend in self._backends.items()}


# Convenience function using enum
def create_storage(storage_type: StorageType = StorageType.LOCAL, **kwargs) -> StorageBackend:
    """Get a storage backend by type enum."""
    if storage_type == StorageType.LOCAL:
        return LocalStorage(kwargs.get('base_path', './storage'))
    elif storage_type == StorageType.MEMORY:
        return MemoryStorage()
    elif storage_type == StorageType.S3:
        return S3Storage(**kwargs)
    elif storage_type == StorageType.AZURE:
        return AzureStorage(**kwargs)
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")


# For backward compatibility
local_storage = get_local_storage
memory_storage = get_memory_storage
