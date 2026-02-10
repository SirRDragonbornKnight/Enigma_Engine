"""
================================================================================
Storage Backends - Unified file/blob storage interface.
================================================================================

Provides a consistent API for storing files across different backends:
- Local filesystem
- Amazon S3
- Azure Blob Storage
- Google Cloud Storage
- MinIO
- In-memory (for testing)

USAGE:
    from enigma_engine.utils.storage_backends import StorageBackend, get_storage
    
    # Get default storage (local)
    storage = get_storage()
    
    # Store a file
    storage.put("models/my_model.pt", model_bytes)
    
    # Retrieve a file
    data = storage.get("models/my_model.pt")
    
    # List files
    files = storage.list("models/")
    
    # Use S3
    s3_storage = get_storage("s3://my-bucket")
"""

from __future__ import annotations

import io
import json
import logging
import mimetypes
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class StorageObject:
    """Metadata about a stored object."""
    key: str
    size: int
    modified_at: datetime | None = None
    created_at: datetime | None = None
    content_type: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    
    @property
    def name(self) -> str:
        """Get the filename part of the key."""
        return Path(self.key).name
    
    @property
    def extension(self) -> str:
        """Get the file extension."""
        return Path(self.key).suffix


class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def get(self, key: str) -> bytes:
        """
        Retrieve an object.
        
        Args:
            key: Object key/path
            
        Returns:
            Object content as bytes
            
        Raises:
            FileNotFoundError: If object doesn't exist
        """
    
    @abstractmethod
    def put(self, key: str, data: bytes | BinaryIO, **kwargs) -> StorageObject:
        """
        Store an object.
        
        Args:
            key: Object key/path
            data: Content to store
            **kwargs: Additional options (content_type, metadata, etc.)
            
        Returns:
            StorageObject with metadata
        """
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete an object.
        
        Args:
            key: Object key/path
            
        Returns:
            True if deleted, False if not found
        """
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if an object exists."""
    
    @abstractmethod
    def list(self, prefix: str = "", recursive: bool = True) -> list[StorageObject]:
        """
        List objects.
        
        Args:
            prefix: Key prefix to filter
            recursive: Include nested objects
            
        Returns:
            List of StorageObject
        """
    
    @abstractmethod
    def info(self, key: str) -> StorageObject | None:
        """Get object metadata without downloading."""
    
    def get_text(self, key: str, encoding: str = "utf-8") -> str:
        """Get object as text."""
        return self.get(key).decode(encoding)
    
    def put_text(self, key: str, text: str, encoding: str = "utf-8", **kwargs) -> StorageObject:
        """Store text content."""
        return self.put(key, text.encode(encoding), **kwargs)
    
    def get_json(self, key: str) -> Any:
        """Get object as JSON."""
        return json.loads(self.get_text(key))
    
    def put_json(self, key: str, data: Any, **kwargs) -> StorageObject:
        """Store JSON content."""
        return self.put_text(key, json.dumps(data, indent=2), content_type="application/json", **kwargs)
    
    def copy(self, src_key: str, dst_key: str) -> StorageObject:
        """Copy an object."""
        data = self.get(src_key)
        return self.put(dst_key, data)
    
    def move(self, src_key: str, dst_key: str) -> StorageObject:
        """Move an object."""
        result = self.copy(src_key, dst_key)
        self.delete(src_key)
        return result
    
    def get_stream(self, key: str) -> BinaryIO:
        """Get object as a seekable stream."""
        return io.BytesIO(self.get(key))


class LocalStorage(StorageBackend):
    """Local filesystem storage backend."""
    
    def __init__(self, base_path: str | Path = "data/storage"):
        """
        Initialize local storage.
        
        Args:
            base_path: Base directory for storage
        """
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
    
    def _full_path(self, key: str) -> Path:
        """Get full path for a key."""
        key = key.lstrip("/")
        full = (self._base_path / key).resolve()
        
        # Security check
        if not str(full).startswith(str(self._base_path.resolve())):
            raise ValueError(f"Invalid key: {key}")
        
        return full
    
    def get(self, key: str) -> bytes:
        path = self._full_path(key)
        if not path.exists():
            raise FileNotFoundError(f"Object not found: {key}")
        return path.read_bytes()
    
    def put(self, key: str, data: bytes | BinaryIO, **kwargs) -> StorageObject:
        path = self._full_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, bytes):
            path.write_bytes(data)
            size = len(data)
        else:
            content = data.read()
            path.write_bytes(content)
            size = len(content)
        
        content_type = kwargs.get('content_type') or mimetypes.guess_type(key)[0]
        
        return StorageObject(
            key=key,
            size=size,
            modified_at=datetime.fromtimestamp(path.stat().st_mtime),
            content_type=content_type,
            metadata=kwargs.get('metadata', {})
        )
    
    def delete(self, key: str) -> bool:
        path = self._full_path(key)
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            return True
        return False
    
    def exists(self, key: str) -> bool:
        return self._full_path(key).exists()
    
    def list(self, prefix: str = "", recursive: bool = True) -> list[StorageObject]:
        base = self._full_path(prefix) if prefix else self._base_path
        
        if not base.exists():
            return []
        
        results = []
        pattern = base.rglob("*") if recursive and base.is_dir() else (base.glob("*") if base.is_dir() else [base])
        
        for path in pattern:
            if path.is_file():
                rel_path = path.relative_to(self._base_path)
                stat = path.stat()
                
                results.append(StorageObject(
                    key=str(rel_path).replace("\\", "/"),
                    size=stat.st_size,
                    modified_at=datetime.fromtimestamp(stat.st_mtime),
                    created_at=datetime.fromtimestamp(stat.st_ctime),
                    content_type=mimetypes.guess_type(str(path))[0]
                ))
        
        return results
    
    def info(self, key: str) -> StorageObject | None:
        path = self._full_path(key)
        if not path.exists():
            return None
        
        stat = path.stat()
        return StorageObject(
            key=key,
            size=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            created_at=datetime.fromtimestamp(stat.st_ctime),
            content_type=mimetypes.guess_type(key)[0]
        )


class MemoryStorage(StorageBackend):
    """In-memory storage for testing."""
    
    def __init__(self):
        self._data: dict[str, bytes] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
    
    def get(self, key: str) -> bytes:
        if key not in self._data:
            raise FileNotFoundError(f"Object not found: {key}")
        return self._data[key]
    
    def put(self, key: str, data: bytes | BinaryIO, **kwargs) -> StorageObject:
        content = data if isinstance(data, bytes) else data.read()
        
        self._data[key] = content
        self._metadata[key] = {
            "modified_at": datetime.now(),
            "content_type": kwargs.get("content_type"),
            "metadata": kwargs.get("metadata", {})
        }
        
        return StorageObject(
            key=key,
            size=len(content),
            modified_at=self._metadata[key]["modified_at"],
            content_type=self._metadata[key]["content_type"],
            metadata=self._metadata[key]["metadata"]
        )
    
    def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            del self._metadata[key]
            return True
        return False
    
    def exists(self, key: str) -> bool:
        return key in self._data
    
    def list(self, prefix: str = "", recursive: bool = True) -> list[StorageObject]:
        results = []
        for key in self._data:
            if key.startswith(prefix):
                meta = self._metadata[key]
                results.append(StorageObject(
                    key=key,
                    size=len(self._data[key]),
                    modified_at=meta["modified_at"],
                    content_type=meta["content_type"],
                    metadata=meta["metadata"]
                ))
        return results
    
    def info(self, key: str) -> StorageObject | None:
        if key not in self._data:
            return None
        
        meta = self._metadata[key]
        return StorageObject(
            key=key,
            size=len(self._data[key]),
            modified_at=meta["modified_at"],
            content_type=meta["content_type"],
            metadata=meta["metadata"]
        )
    
    def clear(self) -> None:
        """Clear all data."""
        self._data.clear()
        self._metadata.clear()


class S3Storage(StorageBackend):
    """Amazon S3 / MinIO storage backend."""
    
    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        endpoint_url: str | None = None,
        region_name: str = "us-east-1",
        access_key: str | None = None,
        secret_key: str | None = None
    ):
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._available = False
        
        try:
            import boto3
            
            config = {"region_name": region_name}
            if endpoint_url:
                config["endpoint_url"] = endpoint_url
            if access_key and secret_key:
                config["aws_access_key_id"] = access_key
                config["aws_secret_access_key"] = secret_key
            
            self._client = boto3.client("s3", **config)
            self._available = True
        except ImportError:
            logger.warning("boto3 not installed, S3 storage unavailable")
            self._client = None
    
    def _full_key(self, key: str) -> str:
        if self._prefix:
            return f"{self._prefix}/{key.lstrip('/')}"
        return key.lstrip("/")
    
    def get(self, key: str) -> bytes:
        if not self._available:
            raise RuntimeError("S3 not available")
        
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=self._full_key(key))
            return response["Body"].read()
        except Exception as e:
            if "NoSuchKey" in str(e):
                raise FileNotFoundError(f"Object not found: {key}")
            raise
    
    def put(self, key: str, data: bytes | BinaryIO, **kwargs) -> StorageObject:
        if not self._available:
            raise RuntimeError("S3 not available")
        
        full_key = self._full_key(key)
        body = data if isinstance(data, bytes) else data.read()
        size = len(body) if isinstance(body, bytes) else 0
        
        extra_args = {}
        if kwargs.get("content_type"):
            extra_args["ContentType"] = kwargs["content_type"]
        if kwargs.get("metadata"):
            extra_args["Metadata"] = kwargs["metadata"]
        
        self._client.put_object(Bucket=self._bucket, Key=full_key, Body=body, **extra_args)
        
        return StorageObject(
            key=key, size=size, modified_at=datetime.now(),
            content_type=kwargs.get("content_type"), metadata=kwargs.get("metadata", {})
        )
    
    def delete(self, key: str) -> bool:
        if not self._available:
            return False
        try:
            self._client.delete_object(Bucket=self._bucket, Key=self._full_key(key))
            return True
        except Exception as e:
            logger.debug(f"S3 delete failed for {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        if not self._available:
            return False
        try:
            self._client.head_object(Bucket=self._bucket, Key=self._full_key(key))
            return True
        except Exception as e:
            logger.debug(f"S3 exists check failed for {key}: {e}")
            return False
    
    def list(self, prefix: str = "", recursive: bool = True) -> list[StorageObject]:
        if not self._available:
            return []
        
        full_prefix = self._full_key(prefix) if prefix else self._prefix
        results = []
        paginator = self._client.get_paginator("list_objects_v2")
        
        for page in paginator.paginate(Bucket=self._bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if self._prefix:
                    key = key[len(self._prefix) + 1:]
                results.append(StorageObject(key=key, size=obj["Size"], modified_at=obj["LastModified"]))
        
        return results
    
    def info(self, key: str) -> StorageObject | None:
        if not self._available:
            return None
        try:
            response = self._client.head_object(Bucket=self._bucket, Key=self._full_key(key))
            return StorageObject(
                key=key, size=response["ContentLength"], modified_at=response["LastModified"],
                content_type=response.get("ContentType"), metadata=response.get("Metadata", {})
            )
        except Exception as e:
            logger.debug(f"S3 info failed for {key}: {e}")
            return None


class AzureStorage(StorageBackend):
    """Azure Blob Storage backend."""
    
    def __init__(self, container: str, connection_string: str | None = None, account_url: str | None = None):
        self._container = container
        self._available = False
        
        try:
            from azure.storage.blob import BlobServiceClient
            
            if connection_string:
                self._client = BlobServiceClient.from_connection_string(connection_string)
            elif account_url:
                from azure.identity import DefaultAzureCredential
                self._client = BlobServiceClient(account_url, credential=DefaultAzureCredential())
            else:
                conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
                if conn_str:
                    self._client = BlobServiceClient.from_connection_string(conn_str)
                else:
                    raise ValueError("No Azure credentials provided")
            
            self._container_client = self._client.get_container_client(container)
            self._available = True
        except ImportError:
            logger.warning("azure-storage-blob not installed")
        except Exception as e:
            logger.warning(f"Azure storage init failed: {e}")
    
    def get(self, key: str) -> bytes:
        if not self._available:
            raise RuntimeError("Azure storage not available")
        blob_client = self._container_client.get_blob_client(key)
        try:
            return blob_client.download_blob().readall()
        except Exception as e:
            raise FileNotFoundError(f"Object not found: {key}") from e
    
    def put(self, key: str, data: bytes | BinaryIO, **kwargs) -> StorageObject:
        if not self._available:
            raise RuntimeError("Azure storage not available")
        blob_client = self._container_client.get_blob_client(key)
        content = data if isinstance(data, bytes) else data.read()
        blob_client.upload_blob(content, overwrite=True)
        return StorageObject(key=key, size=len(content), modified_at=datetime.now())
    
    def delete(self, key: str) -> bool:
        if not self._available:
            return False
        try:
            self._container_client.get_blob_client(key).delete_blob()
            return True
        except Exception as e:
            logger.debug(f"Azure delete failed for {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        if not self._available:
            return False
        return self._container_client.get_blob_client(key).exists()
    
    def list(self, prefix: str = "", recursive: bool = True) -> list[StorageObject]:
        if not self._available:
            return []
        results = []
        for blob in self._container_client.list_blobs(name_starts_with=prefix):
            results.append(StorageObject(key=blob.name, size=blob.size, modified_at=blob.last_modified))
        return results
    
    def info(self, key: str) -> StorageObject | None:
        if not self._available:
            return None
        try:
            props = self._container_client.get_blob_client(key).get_blob_properties()
            return StorageObject(key=key, size=props.size, modified_at=props.last_modified)
        except Exception as e:
            logger.debug(f"Azure info failed for {key}: {e}")
            return None


# Storage factory
_storage_instances: dict[str, StorageBackend] = {}


def get_storage(uri: str = "local://data/storage") -> StorageBackend:
    """
    Get a storage backend by URI.
    
    Args:
        uri: Storage URI:
            - "local://path" or just "path" for local filesystem
            - "s3://bucket/prefix" for S3
            - "azure://container" for Azure
            - "memory://" for in-memory
            
    Returns:
        StorageBackend instance
    """
    if uri in _storage_instances:
        return _storage_instances[uri]
    
    parsed = urlparse(uri)
    scheme = parsed.scheme or "local"
    
    if scheme in ("local", "file", ""):
        path = parsed.path or parsed.netloc or uri
        backend = LocalStorage(path)
    elif scheme == "s3":
        backend = S3Storage(bucket=parsed.netloc, prefix=parsed.path.lstrip("/"))
    elif scheme == "azure":
        backend = AzureStorage(container=parsed.netloc)
    elif scheme == "memory":
        backend = MemoryStorage()
    else:
        raise ValueError(f"Unknown storage scheme: {scheme}")
    
    _storage_instances[uri] = backend
    return backend


def get_local_storage(path: str = "data/storage") -> LocalStorage:
    """Get local filesystem storage."""
    return LocalStorage(path)


def get_memory_storage() -> MemoryStorage:
    """Get in-memory storage for testing."""
    return MemoryStorage()
