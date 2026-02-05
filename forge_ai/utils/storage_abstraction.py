"""
Storage Abstraction - Unified interface for file and blob storage.

Provides a consistent API for different storage backends:
- Local filesystem
- Memory (for testing)
- S3-compatible (AWS S3, MinIO, etc.)
- Azure Blob Storage
- Google Cloud Storage

Part of the ForgeAI infrastructure.
"""

import os
import io
import json
import shutil
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, BinaryIO, Union, Iterator
from pathlib import Path
from datetime import datetime
from enum import Enum


class StorageError(Exception):
    """Base exception for storage errors."""
    pass


class ObjectNotFoundError(StorageError):
    """Raised when object is not found."""
    pass


class PermissionDeniedError(StorageError):
    """Raised when permission is denied."""
    pass


class StorageType(Enum):
    """Types of storage backends."""
    LOCAL = "local"
    MEMORY = "memory"
    S3 = "s3"
    AZURE = "azure"
    GCS = "gcs"


@dataclass
class StorageMetadata:
    """Metadata for a stored object."""
    key: str
    size: int
    content_type: Optional[str] = None
    last_modified: Optional[datetime] = None
    etag: Optional[str] = None
    custom_metadata: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "size": self.size,
            "content_type": self.content_type,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "etag": self.etag,
            "custom_metadata": self.custom_metadata
        }


@dataclass
class ListResult:
    """Result of listing objects."""
    objects: List[StorageMetadata]
    prefixes: List[str] = field(default_factory=list)  # Common prefixes (directories)
    is_truncated: bool = False
    next_token: Optional[str] = None


class StorageBackend(ABC):
    """
    Abstract base class for storage backends.
    
    All storage implementations must extend this class.
    """
    
    @abstractmethod
    def put(
        self,
        key: str,
        data: Union[bytes, BinaryIO, str],
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> StorageMetadata:
        """
        Store an object.
        
        Args:
            key: Object key/path
            data: Data to store (bytes, file-like object, or string)
            content_type: MIME type
            metadata: Custom metadata
            
        Returns:
            Object metadata
        """
        pass
    
    @abstractmethod
    def get(self, key: str) -> bytes:
        """
        Retrieve an object.
        
        Args:
            key: Object key/path
            
        Returns:
            Object data as bytes
            
        Raises:
            ObjectNotFoundError: If object doesn't exist
        """
        pass
    
    @abstractmethod
    def get_stream(self, key: str) -> BinaryIO:
        """
        Get object as a stream.
        
        Args:
            key: Object key/path
            
        Returns:
            File-like object for reading
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete an object.
        
        Args:
            key: Object key/path
            
        Returns:
            True if deleted, False if didn't exist
        """
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if an object exists.
        
        Args:
            key: Object key/path
            
        Returns:
            True if exists
        """
        pass
    
    @abstractmethod
    def get_metadata(self, key: str) -> StorageMetadata:
        """
        Get object metadata without downloading.
        
        Args:
            key: Object key/path
            
        Returns:
            Object metadata
        """
        pass
    
    @abstractmethod
    def list_objects(
        self,
        prefix: str = "",
        delimiter: Optional[str] = None,
        max_keys: int = 1000,
        continuation_token: Optional[str] = None
    ) -> ListResult:
        """
        List objects.
        
        Args:
            prefix: Filter by prefix
            delimiter: Group by delimiter (e.g., "/" for directories)
            max_keys: Maximum number of keys to return
            continuation_token: Token for pagination
            
        Returns:
            ListResult with objects and pagination info
        """
        pass
    
    @abstractmethod
    def copy(self, source_key: str, dest_key: str) -> StorageMetadata:
        """
        Copy an object.
        
        Args:
            source_key: Source object key
            dest_key: Destination object key
            
        Returns:
            New object metadata
        """
        pass
    
    def move(self, source_key: str, dest_key: str) -> StorageMetadata:
        """
        Move an object.
        
        Args:
            source_key: Source object key
            dest_key: Destination object key
            
        Returns:
            New object metadata
        """
        metadata = self.copy(source_key, dest_key)
        self.delete(source_key)
        return metadata
    
    def put_json(
        self,
        key: str,
        data: Any,
        metadata: Optional[Dict[str, str]] = None
    ) -> StorageMetadata:
        """Store JSON data."""
        json_bytes = json.dumps(data, indent=2).encode('utf-8')
        return self.put(key, json_bytes, 'application/json', metadata)
    
    def get_json(self, key: str) -> Any:
        """Retrieve and parse JSON data."""
        data = self.get(key)
        return json.loads(data.decode('utf-8'))
    
    def put_text(
        self,
        key: str,
        text: str,
        encoding: str = 'utf-8',
        metadata: Optional[Dict[str, str]] = None
    ) -> StorageMetadata:
        """Store text data."""
        return self.put(key, text.encode(encoding), 'text/plain', metadata)
    
    def get_text(self, key: str, encoding: str = 'utf-8') -> str:
        """Retrieve text data."""
        return self.get(key).decode(encoding)


class LocalStorage(StorageBackend):
    """
    Local filesystem storage backend.
    
    Usage:
        storage = LocalStorage("/path/to/storage")
        storage.put("data/file.txt", b"Hello World")
        data = storage.get("data/file.txt")
    """
    
    def __init__(self, base_path: str):
        """
        Initialize local storage.
        
        Args:
            base_path: Base directory for storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _full_path(self, key: str) -> Path:
        """Get full path for key."""
        # Sanitize key to prevent path traversal
        clean_key = key.lstrip('/').replace('..', '')
        return self.base_path / clean_key
    
    def _calculate_etag(self, data: bytes) -> str:
        """Calculate ETag (MD5 hash)."""
        return hashlib.md5(data).hexdigest()
    
    def put(
        self,
        key: str,
        data: Union[bytes, BinaryIO, str],
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> StorageMetadata:
        """Store an object to local filesystem."""
        path = self._full_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to bytes
        data_bytes: bytes
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif hasattr(data, 'read'):
            data_bytes = data.read()  # type: ignore
        elif isinstance(data, (bytearray, memoryview)):
            data_bytes = bytes(data)
        else:
            data_bytes = data
        
        path.write_bytes(data_bytes)
        
        # Store metadata in sidecar file
        if metadata or content_type:
            meta_path = Path(str(path) + '.meta')
            meta_data = {
                'content_type': content_type,
                'metadata': metadata or {}
            }
            meta_path.write_text(json.dumps(meta_data))
        
        return StorageMetadata(
            key=key,
            size=len(data_bytes),
            content_type=content_type,
            last_modified=datetime.fromtimestamp(path.stat().st_mtime),
            etag=self._calculate_etag(data_bytes),
            custom_metadata=metadata or {}
        )
    
    def get(self, key: str) -> bytes:
        """Retrieve object from local filesystem."""
        path = self._full_path(key)
        
        if not path.exists():
            raise ObjectNotFoundError(f"Object not found: {key}")
        
        return path.read_bytes()
    
    def get_stream(self, key: str) -> BinaryIO:
        """Get object as stream."""
        path = self._full_path(key)
        
        if not path.exists():
            raise ObjectNotFoundError(f"Object not found: {key}")
        
        return open(path, 'rb')
    
    def delete(self, key: str) -> bool:
        """Delete object from local filesystem."""
        path = self._full_path(key)
        meta_path = Path(str(path) + '.meta')
        
        if not path.exists():
            return False
        
        path.unlink()
        if meta_path.exists():
            meta_path.unlink()
        
        return True
    
    def exists(self, key: str) -> bool:
        """Check if object exists."""
        return self._full_path(key).exists()
    
    def get_metadata(self, key: str) -> StorageMetadata:
        """Get object metadata."""
        path = self._full_path(key)
        
        if not path.exists():
            raise ObjectNotFoundError(f"Object not found: {key}")
        
        stat = path.stat()
        
        # Load sidecar metadata
        meta_path = Path(str(path) + '.meta')
        content_type = None
        custom_metadata = {}
        
        if meta_path.exists():
            meta_data = json.loads(meta_path.read_text())
            content_type = meta_data.get('content_type')
            custom_metadata = meta_data.get('metadata', {})
        
        return StorageMetadata(
            key=key,
            size=stat.st_size,
            content_type=content_type,
            last_modified=datetime.fromtimestamp(stat.st_mtime),
            etag=self._calculate_etag(path.read_bytes()),
            custom_metadata=custom_metadata
        )
    
    def list_objects(
        self,
        prefix: str = "",
        delimiter: Optional[str] = None,
        max_keys: int = 1000,
        continuation_token: Optional[str] = None
    ) -> ListResult:
        """List objects in local filesystem."""
        prefix_path = self._full_path(prefix) if prefix else self.base_path
        objects = []
        prefixes = set()
        
        # Handle prefix as directory or partial match
        if prefix and prefix_path.is_dir():
            search_path = prefix_path
            search_prefix = ""
        else:
            search_path = prefix_path.parent if prefix else self.base_path
            search_prefix = prefix_path.name if prefix else ""
        
        if not search_path.exists():
            return ListResult(objects=[], prefixes=[])
        
        for item in search_path.rglob('*'):
            if item.suffix == '.meta':
                continue
            
            if not item.is_file():
                continue
            
            # Get relative key
            try:
                rel_path = item.relative_to(self.base_path)
                key = str(rel_path).replace('\\', '/')
            except ValueError:
                continue
            
            # Check prefix
            if prefix and not key.startswith(prefix):
                continue
            
            # Handle delimiter (for directory-like listing)
            if delimiter:
                after_prefix = key[len(prefix):]
                if delimiter in after_prefix:
                    common_prefix = prefix + after_prefix.split(delimiter)[0] + delimiter
                    prefixes.add(common_prefix)
                    continue
            
            stat = item.stat()
            objects.append(StorageMetadata(
                key=key,
                size=stat.st_size,
                last_modified=datetime.fromtimestamp(stat.st_mtime)
            ))
            
            if len(objects) >= max_keys:
                break
        
        return ListResult(
            objects=objects,
            prefixes=list(prefixes),
            is_truncated=len(objects) >= max_keys
        )
    
    def copy(self, source_key: str, dest_key: str) -> StorageMetadata:
        """Copy object."""
        source_path = self._full_path(source_key)
        dest_path = self._full_path(dest_key)
        
        if not source_path.exists():
            raise ObjectNotFoundError(f"Object not found: {source_key}")
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)
        
        # Copy metadata sidecar
        source_meta = Path(str(source_path) + '.meta')
        if source_meta.exists():
            dest_meta = Path(str(dest_path) + '.meta')
            shutil.copy2(source_meta, dest_meta)
        
        return self.get_metadata(dest_key)


class MemoryStorage(StorageBackend):
    """
    In-memory storage backend (useful for testing).
    
    Usage:
        storage = MemoryStorage()
        storage.put("test/file.txt", b"Hello")
        data = storage.get("test/file.txt")
    """
    
    def __init__(self):
        """Initialize memory storage."""
        self._objects: Dict[str, bytes] = {}
        self._metadata: Dict[str, StorageMetadata] = {}
    
    def put(
        self,
        key: str,
        data: Union[bytes, BinaryIO, str],
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> StorageMetadata:
        """Store object in memory."""
        data_bytes: bytes
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif hasattr(data, 'read'):
            data_bytes = data.read()  # type: ignore
        elif isinstance(data, (bytearray, memoryview)):
            data_bytes = bytes(data)
        else:
            data_bytes = data
        
        self._objects[key] = data_bytes
        
        meta = StorageMetadata(
            key=key,
            size=len(data_bytes),
            content_type=content_type,
            last_modified=datetime.now(),
            etag=hashlib.md5(data_bytes).hexdigest(),
            custom_metadata=metadata or {}
        )
        self._metadata[key] = meta
        
        return meta
    
    def get(self, key: str) -> bytes:
        """Retrieve object from memory."""
        if key not in self._objects:
            raise ObjectNotFoundError(f"Object not found: {key}")
        return self._objects[key]
    
    def get_stream(self, key: str) -> BinaryIO:
        """Get object as stream."""
        return io.BytesIO(self.get(key))
    
    def delete(self, key: str) -> bool:
        """Delete object from memory."""
        if key not in self._objects:
            return False
        
        del self._objects[key]
        del self._metadata[key]
        return True
    
    def exists(self, key: str) -> bool:
        """Check if object exists."""
        return key in self._objects
    
    def get_metadata(self, key: str) -> StorageMetadata:
        """Get object metadata."""
        if key not in self._metadata:
            raise ObjectNotFoundError(f"Object not found: {key}")
        return self._metadata[key]
    
    def list_objects(
        self,
        prefix: str = "",
        delimiter: Optional[str] = None,
        max_keys: int = 1000,
        continuation_token: Optional[str] = None
    ) -> ListResult:
        """List objects in memory."""
        objects = []
        prefixes = set()
        
        for key, meta in self._metadata.items():
            if prefix and not key.startswith(prefix):
                continue
            
            if delimiter:
                after_prefix = key[len(prefix):]
                if delimiter in after_prefix:
                    common_prefix = prefix + after_prefix.split(delimiter)[0] + delimiter
                    prefixes.add(common_prefix)
                    continue
            
            objects.append(meta)
            
            if len(objects) >= max_keys:
                break
        
        return ListResult(
            objects=objects,
            prefixes=list(prefixes),
            is_truncated=len(objects) >= max_keys
        )
    
    def copy(self, source_key: str, dest_key: str) -> StorageMetadata:
        """Copy object."""
        data = self.get(source_key)
        source_meta = self._metadata[source_key]
        
        return self.put(
            dest_key,
            data,
            source_meta.content_type,
            source_meta.custom_metadata.copy()
        )
    
    def clear(self):
        """Clear all objects."""
        self._objects.clear()
        self._metadata.clear()


class S3Storage(StorageBackend):
    """
    S3-compatible storage backend.
    
    Requires boto3: pip install boto3
    
    Usage:
        storage = S3Storage(
            bucket="my-bucket",
            access_key="...",
            secret_key="...",
            endpoint_url="http://localhost:9000"  # For MinIO
        )
    """
    
    def __init__(
        self,
        bucket: str,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: str = "us-east-1",
        endpoint_url: Optional[str] = None
    ):
        """Initialize S3 storage."""
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 required for S3Storage: pip install boto3")
        
        self.bucket = bucket
        
        session_kwargs = {}
        if access_key and secret_key:
            session_kwargs['aws_access_key_id'] = access_key
            session_kwargs['aws_secret_access_key'] = secret_key
        
        client_kwargs = {'region_name': region}
        if endpoint_url:
            client_kwargs['endpoint_url'] = endpoint_url
        
        self._client = boto3.client('s3', **session_kwargs, **client_kwargs)
    
    def put(
        self,
        key: str,
        data: Union[bytes, BinaryIO, str],
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> StorageMetadata:
        """Store object in S3."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        kwargs = {'Bucket': self.bucket, 'Key': key}
        
        if hasattr(data, 'read'):
            kwargs['Body'] = data
        else:
            kwargs['Body'] = data
        
        if content_type:
            kwargs['ContentType'] = content_type
        
        if metadata:
            kwargs['Metadata'] = metadata
        
        response = self._client.put_object(**kwargs)
        
        size = len(data) if isinstance(data, bytes) else 0
        
        return StorageMetadata(
            key=key,
            size=size,
            content_type=content_type,
            last_modified=datetime.now(),
            etag=response.get('ETag', '').strip('"'),
            custom_metadata=metadata or {}
        )
    
    def get(self, key: str) -> bytes:
        """Retrieve object from S3."""
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
            return response['Body'].read()
        except self._client.exceptions.NoSuchKey:
            raise ObjectNotFoundError(f"Object not found: {key}")
    
    def get_stream(self, key: str) -> BinaryIO:
        """Get object as stream."""
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
            return response['Body']
        except self._client.exceptions.NoSuchKey:
            raise ObjectNotFoundError(f"Object not found: {key}")
    
    def delete(self, key: str) -> bool:
        """Delete object from S3."""
        try:
            self._client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False
    
    def exists(self, key: str) -> bool:
        """Check if object exists."""
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False
    
    def get_metadata(self, key: str) -> StorageMetadata:
        """Get object metadata."""
        try:
            response = self._client.head_object(Bucket=self.bucket, Key=key)
            
            return StorageMetadata(
                key=key,
                size=response.get('ContentLength', 0),
                content_type=response.get('ContentType'),
                last_modified=response.get('LastModified'),
                etag=response.get('ETag', '').strip('"'),
                custom_metadata=response.get('Metadata', {})
            )
        except self._client.exceptions.NoSuchKey:
            raise ObjectNotFoundError(f"Object not found: {key}")
    
    def list_objects(
        self,
        prefix: str = "",
        delimiter: Optional[str] = None,
        max_keys: int = 1000,
        continuation_token: Optional[str] = None
    ) -> ListResult:
        """List objects in S3."""
        kwargs = {
            'Bucket': self.bucket,
            'MaxKeys': max_keys
        }
        
        if prefix:
            kwargs['Prefix'] = prefix
        if delimiter:
            kwargs['Delimiter'] = delimiter
        if continuation_token:
            kwargs['ContinuationToken'] = continuation_token
        
        response = self._client.list_objects_v2(**kwargs)
        
        objects = []
        for obj in response.get('Contents', []):
            objects.append(StorageMetadata(
                key=obj['Key'],
                size=obj['Size'],
                last_modified=obj['LastModified'],
                etag=obj.get('ETag', '').strip('"')
            ))
        
        prefixes = [p['Prefix'] for p in response.get('CommonPrefixes', [])]
        
        return ListResult(
            objects=objects,
            prefixes=prefixes,
            is_truncated=response.get('IsTruncated', False),
            next_token=response.get('NextContinuationToken')
        )
    
    def copy(self, source_key: str, dest_key: str) -> StorageMetadata:
        """Copy object in S3."""
        copy_source = {'Bucket': self.bucket, 'Key': source_key}
        self._client.copy_object(
            CopySource=copy_source,
            Bucket=self.bucket,
            Key=dest_key
        )
        return self.get_metadata(dest_key)


class StorageManager:
    """
    Manager for multiple storage backends.
    
    Usage:
        manager = StorageManager()
        manager.add_backend("local", LocalStorage("/data"))
        manager.add_backend("s3", S3Storage("my-bucket"))
        
        # Use specific backend
        manager.get("local").put("file.txt", b"data")
        
        # Mirror to multiple backends
        manager.put_all("shared/file.txt", b"data")
    """
    
    def __init__(self):
        """Initialize storage manager."""
        self._backends: Dict[str, StorageBackend] = {}
        self._default: Optional[str] = None
    
    def add_backend(
        self,
        name: str,
        backend: StorageBackend,
        default: bool = False
    ) -> None:
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
    
    def put_all(
        self,
        key: str,
        data: Union[bytes, BinaryIO, str],
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, StorageMetadata]:
        """Put object to all backends."""
        results = {}
        
        # Ensure we have bytes
        data_bytes: bytes
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif hasattr(data, 'read'):
            data_bytes = data.read()  # type: ignore
        elif isinstance(data, (bytearray, memoryview)):
            data_bytes = bytes(data)
        else:
            data_bytes = data
        
        for name, backend in self._backends.items():
            try:
                results[name] = backend.put(key, data_bytes, content_type, metadata)
            except Exception as e:
                results[name] = str(e)
        
        return results
    
    def delete_all(self, key: str) -> Dict[str, bool]:
        """Delete object from all backends."""
        results = {}
        for name, backend in self._backends.items():
            try:
                results[name] = backend.delete(key)
            except Exception:
                results[name] = False
        return results


# Convenience functions
def get_storage(
    storage_type: StorageType = StorageType.LOCAL,
    **kwargs
) -> StorageBackend:
    """
    Get a storage backend.
    
    Args:
        storage_type: Type of storage
        **kwargs: Backend-specific arguments
        
    Returns:
        Storage backend instance
    """
    if storage_type == StorageType.LOCAL:
        return LocalStorage(kwargs.get('base_path', './storage'))
    elif storage_type == StorageType.MEMORY:
        return MemoryStorage()
    elif storage_type == StorageType.S3:
        return S3Storage(**kwargs)
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")


def local_storage(base_path: str = "./storage") -> LocalStorage:
    """Get local filesystem storage."""
    return LocalStorage(base_path)


def memory_storage() -> MemoryStorage:
    """Get in-memory storage."""
    return MemoryStorage()
