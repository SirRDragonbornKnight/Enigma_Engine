"""
Dataset Versioning

Version control for training datasets, tracking changes,
diffs, and maintaining reproducible data lineage.

FILE: enigma_engine/data/versioning.py
TYPE: Data Management
MAIN CLASSES: DatasetVersion, VersionManager, DataDiff
"""

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of changes between versions."""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


@dataclass
class FileInfo:
    """Information about a file in a version."""
    path: str
    size: int
    hash: str
    modified_at: float
    
    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "size": self.size,
            "hash": self.hash,
            "modified_at": self.modified_at
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FileInfo':
        return cls(
            path=data["path"],
            size=data["size"],
            hash=data["hash"],
            modified_at=data.get("modified_at", 0)
        )


@dataclass
class DataChange:
    """A single change between versions."""
    change_type: ChangeType
    file_info: FileInfo
    old_info: Optional[FileInfo] = None
    
    def to_dict(self) -> dict:
        data = {
            "change_type": self.change_type.value,
            "file": self.file_info.to_dict()
        }
        if self.old_info:
            data["old_file"] = self.old_info.to_dict()
        return data


@dataclass
class DataDiff:
    """Difference between two versions."""
    from_version: str
    to_version: str
    changes: list[DataChange] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    
    @property
    def added_count(self) -> int:
        return sum(1 for c in self.changes if c.change_type == ChangeType.ADDED)
    
    @property
    def removed_count(self) -> int:
        return sum(1 for c in self.changes if c.change_type == ChangeType.REMOVED)
    
    @property
    def modified_count(self) -> int:
        return sum(1 for c in self.changes if c.change_type == ChangeType.MODIFIED)
    
    def summary(self) -> str:
        return f"+{self.added_count} -{self.removed_count} ~{self.modified_count}"
    
    def to_dict(self) -> dict:
        return {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "changes": [c.to_dict() for c in self.changes],
            "created_at": self.created_at
        }


@dataclass
class DatasetVersion:
    """A specific version of a dataset."""
    version_id: str
    name: str
    description: str = ""
    created_at: float = field(default_factory=time.time)
    parent_version: Optional[str] = None
    files: list[FileInfo] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    
    @property
    def total_size(self) -> int:
        return sum(f.size for f in self.files)
    
    @property
    def file_count(self) -> int:
        return len(self.files)
    
    def to_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "parent_version": self.parent_version,
            "files": [f.to_dict() for f in self.files],
            "metadata": self.metadata,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DatasetVersion':
        return cls(
            version_id=data["version_id"],
            name=data["name"],
            description=data.get("description", ""),
            created_at=data.get("created_at", time.time()),
            parent_version=data.get("parent_version"),
            files=[FileInfo.from_dict(f) for f in data.get("files", [])],
            metadata=data.get("metadata", {}),
            tags=data.get("tags", [])
        )


class VersionManager:
    """Manages dataset versions."""
    
    def __init__(self, 
                 data_dir: Path,
                 versions_dir: Optional[Path] = None):
        """
        Initialize version manager.
        
        Args:
            data_dir: Directory containing datasets
            versions_dir: Directory for version storage
        """
        self._data_dir = Path(data_dir)
        self._versions_dir = versions_dir or (self._data_dir / ".versions")
        self._versions_dir.mkdir(parents=True, exist_ok=True)
        
        self._versions: dict[str, DatasetVersion] = {}
        self._current_version: Optional[str] = None
        
        self._load_versions()
    
    def create_version(self,
                       name: str,
                       description: str = "",
                       source_dir: Optional[Path] = None,
                       tags: list[str] = None,
                       metadata: dict = None) -> DatasetVersion:
        """
        Create a new dataset version.
        
        Args:
            name: Version name
            description: Version description
            source_dir: Directory to version (uses data_dir if None)
            tags: Version tags
            metadata: Additional metadata
            
        Returns:
            Created version
        """
        source = source_dir or self._data_dir
        
        # Generate version ID
        version_id = self._generate_version_id()
        
        # Collect file info
        files = self._scan_directory(source)
        
        # Create version
        version = DatasetVersion(
            version_id=version_id,
            name=name,
            description=description,
            parent_version=self._current_version,
            files=files,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Store version
        self._versions[version_id] = version
        self._current_version = version_id
        
        # Save version data
        self._save_version(version)
        
        # Optionally snapshot files
        self._create_snapshot(version, source)
        
        logger.info(f"Created version {version_id}: {name}")
        return version
    
    def get_version(self, version_id: str) -> Optional[DatasetVersion]:
        """Get a specific version."""
        return self._versions.get(version_id)
    
    def list_versions(self) -> list[DatasetVersion]:
        """List all versions, newest first."""
        versions = list(self._versions.values())
        versions.sort(key=lambda v: v.created_at, reverse=True)
        return versions
    
    def get_latest_version(self) -> Optional[DatasetVersion]:
        """Get the most recent version."""
        versions = self.list_versions()
        return versions[0] if versions else None
    
    def diff_versions(self, 
                      from_version: str, 
                      to_version: str) -> DataDiff:
        """
        Compare two versions.
        
        Args:
            from_version: Source version ID
            to_version: Target version ID
            
        Returns:
            Diff between versions
        """
        v_from = self._versions.get(from_version)
        v_to = self._versions.get(to_version)
        
        if not v_from or not v_to:
            raise ValueError("Version not found")
        
        changes = []
        
        # Index files by path
        from_files = {f.path: f for f in v_from.files}
        to_files = {f.path: f for f in v_to.files}
        
        all_paths = set(from_files.keys()) | set(to_files.keys())
        
        for path in all_paths:
            in_from = path in from_files
            in_to = path in to_files
            
            if in_from and in_to:
                if from_files[path].hash != to_files[path].hash:
                    changes.append(DataChange(
                        change_type=ChangeType.MODIFIED,
                        file_info=to_files[path],
                        old_info=from_files[path]
                    ))
            elif in_to:
                changes.append(DataChange(
                    change_type=ChangeType.ADDED,
                    file_info=to_files[path]
                ))
            else:
                changes.append(DataChange(
                    change_type=ChangeType.REMOVED,
                    file_info=from_files[path]
                ))
        
        return DataDiff(
            from_version=from_version,
            to_version=to_version,
            changes=changes
        )
    
    def checkout_version(self, 
                         version_id: str,
                         target_dir: Optional[Path] = None) -> bool:
        """
        Restore a version to a directory.
        
        Args:
            version_id: Version to restore
            target_dir: Target directory (uses data_dir if None)
            
        Returns:
            True if successful
        """
        version = self._versions.get(version_id)
        if not version:
            logger.error(f"Version not found: {version_id}")
            return False
        
        target = target_dir or self._data_dir
        snapshot_dir = self._versions_dir / version_id / "snapshot"
        
        if not snapshot_dir.exists():
            logger.error(f"Snapshot not found for version: {version_id}")
            return False
        
        # Clear target
        if target.exists():
            for item in target.iterdir():
                if item.name != ".versions":
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
        
        # Copy from snapshot
        shutil.copytree(snapshot_dir, target, dirs_exist_ok=True)
        
        self._current_version = version_id
        logger.info(f"Checked out version {version_id}")
        return True
    
    def tag_version(self, version_id: str, tag: str):
        """Add a tag to a version."""
        version = self._versions.get(version_id)
        if version:
            if tag not in version.tags:
                version.tags.append(tag)
            self._save_version(version)
    
    def find_by_tag(self, tag: str) -> list[DatasetVersion]:
        """Find versions with a specific tag."""
        return [v for v in self._versions.values() if tag in v.tags]
    
    def get_lineage(self, version_id: str) -> list[DatasetVersion]:
        """Get the full lineage of a version."""
        lineage = []
        current = version_id
        
        while current:
            version = self._versions.get(current)
            if version:
                lineage.append(version)
                current = version.parent_version
            else:
                break
        
        return lineage
    
    def _generate_version_id(self) -> str:
        """Generate a unique version ID."""
        timestamp = int(time.time() * 1000)
        return f"v{timestamp}"
    
    def _scan_directory(self, directory: Path) -> list[FileInfo]:
        """Scan directory and collect file info."""
        files = []
        
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                # Skip version directory
                if ".versions" in str(file_path):
                    continue
                
                rel_path = file_path.relative_to(directory)
                stat = file_path.stat()
                
                files.append(FileInfo(
                    path=str(rel_path),
                    size=stat.st_size,
                    hash=self._compute_hash(file_path),
                    modified_at=stat.st_mtime
                ))
        
        return files
    
    def _compute_hash(self, file_path: Path, chunk_size: int = 8192) -> str:
        """Compute SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        
        return hasher.hexdigest()[:16]  # Abbreviated hash
    
    def _create_snapshot(self, version: DatasetVersion, source: Path):
        """Create a snapshot of the current state."""
        snapshot_dir = self._versions_dir / version.version_id / "snapshot"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        for file_info in version.files:
            src = source / file_info.path
            dst = snapshot_dir / file_info.path
            
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    
    def _save_version(self, version: DatasetVersion):
        """Save version metadata."""
        version_dir = self._versions_dir / version.version_id
        version_dir.mkdir(parents=True, exist_ok=True)
        
        meta_path = version_dir / "version.json"
        with open(meta_path, 'w') as f:
            json.dump(version.to_dict(), f, indent=2)
        
        # Also save to index
        self._save_index()
    
    def _save_index(self):
        """Save versions index."""
        index_path = self._versions_dir / "index.json"
        
        index = {
            "current_version": self._current_version,
            "versions": list(self._versions.keys())
        }
        
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
    
    def _load_versions(self):
        """Load all versions from storage."""
        index_path = self._versions_dir / "index.json"
        
        if not index_path.exists():
            return
        
        with open(index_path) as f:
            index = json.load(f)
        
        self._current_version = index.get("current_version")
        
        for version_id in index.get("versions", []):
            version_dir = self._versions_dir / version_id
            meta_path = version_dir / "version.json"
            
            if meta_path.exists():
                with open(meta_path) as f:
                    data = json.load(f)
                    self._versions[version_id] = DatasetVersion.from_dict(data)
    
    def cleanup_old_versions(self, keep_count: int = 10):
        """Remove old versions, keeping the most recent."""
        versions = self.list_versions()
        
        if len(versions) <= keep_count:
            return
        
        to_remove = versions[keep_count:]
        
        for version in to_remove:
            # Don't remove tagged versions
            if version.tags:
                continue
            
            version_dir = self._versions_dir / version.version_id
            if version_dir.exists():
                shutil.rmtree(version_dir)
            
            del self._versions[version.version_id]
        
        self._save_index()
        logger.info(f"Cleaned up {len(to_remove)} old versions")


# Factory function
def create_version_manager(data_dir: Path) -> VersionManager:
    """Create a dataset version manager."""
    return VersionManager(data_dir)


__all__ = [
    'VersionManager',
    'DatasetVersion',
    'DataDiff',
    'DataChange',
    'ChangeType',
    'FileInfo',
    'create_version_manager'
]
