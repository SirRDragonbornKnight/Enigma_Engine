"""
Auto-Backup System for Enigma AI Engine

Automatic backup of models, conversations, and configurations.

Features:
- Scheduled backups
- Incremental backups
- Compression support
- Multiple backup targets (local, cloud)
- Restore functionality

Usage:
    from enigma_engine.utils.backup import BackupManager, get_backup_manager
    
    backup = get_backup_manager()
    
    # Configure
    backup.set_backup_dir("./backups")
    backup.set_schedule(interval_hours=24)
    
    # Manual backup
    backup.backup_all()
    
    # Restore
    backup.restore("backup_20240115_120000.zip")
"""

import hashlib
import json
import logging
import os
import shutil
import tarfile
import threading
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class BackupType(Enum):
    """Types of backup."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class CompressionType(Enum):
    """Compression formats."""
    NONE = "none"
    ZIP = "zip"
    TAR_GZ = "tar.gz"
    TAR_BZ2 = "tar.bz2"


@dataclass
class BackupTarget:
    """Something to backup."""
    name: str
    path: str
    patterns: List[str] = field(default_factory=lambda: ["*"])
    exclude: List[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class BackupRecord:
    """Record of a backup."""
    id: str
    timestamp: float
    backup_type: str
    targets: List[str]
    file_count: int
    total_size: int
    compressed_size: int
    location: str
    checksum: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "backup_type": self.backup_type,
            "targets": self.targets,
            "file_count": self.file_count,
            "total_size": self.total_size,
            "compressed_size": self.compressed_size,
            "location": self.location,
            "checksum": self.checksum
        }


class BackupManager:
    """
    Manages automatic backups.
    """
    
    def __init__(
        self,
        backup_dir: Optional[str] = None,
        max_backups: int = 10,
        compression: CompressionType = CompressionType.ZIP
    ):
        """
        Initialize backup manager.
        
        Args:
            backup_dir: Directory to store backups
            max_backups: Maximum backups to keep
            compression: Compression format
        """
        self._backup_dir = Path(backup_dir) if backup_dir else Path("./backups")
        self._max_backups = max_backups
        self._compression = compression
        
        # Backup targets
        self._targets: Dict[str, BackupTarget] = {}
        
        # Backup history
        self._history: List[BackupRecord] = []
        self._history_file = self._backup_dir / "backup_history.json"
        
        # Incremental tracking
        self._file_hashes: Dict[str, str] = {}
        self._hashes_file = self._backup_dir / "file_hashes.json"
        
        # Scheduling
        self._schedule_thread: Optional[threading.Thread] = None
        self._running = False
        self._interval_hours = 0
        
        # Callbacks
        self._callbacks: List[Callable[[BackupRecord], None]] = []
        
        # Create backup dir
        self._backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Load history
        self._load_history()
        self._load_hashes()
        
        # Register default targets
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default backup targets."""
        # Models
        self.add_target(BackupTarget(
            name="models",
            path="./models",
            patterns=["*.pth", "*.pt", "*.gguf", "*.bin", "config.json", "tokenizer.json"],
            exclude=["*.tmp", "*.lock"]
        ))
        
        # Conversations
        self.add_target(BackupTarget(
            name="conversations",
            path="./memory",
            patterns=["*.json", "*.db"]
        ))
        
        # Configuration
        self.add_target(BackupTarget(
            name="config",
            path="./",
            patterns=["forge_modules.json", "config.json", "*.yaml", "*.yml"],
            exclude=["node_modules/*", "__pycache__/*"]
        ))
    
    def add_target(self, target: BackupTarget):
        """Add a backup target."""
        self._targets[target.name] = target
        logger.info(f"Added backup target: {target.name}")
    
    def remove_target(self, name: str):
        """Remove a backup target."""
        if name in self._targets:
            del self._targets[name]
    
    def set_backup_dir(self, path: str):
        """Set backup directory."""
        self._backup_dir = Path(path)
        self._backup_dir.mkdir(parents=True, exist_ok=True)
        self._history_file = self._backup_dir / "backup_history.json"
        self._hashes_file = self._backup_dir / "file_hashes.json"
    
    def set_schedule(self, interval_hours: float):
        """
        Set backup schedule.
        
        Args:
            interval_hours: Hours between backups (0 to disable)
        """
        self._interval_hours = interval_hours
        
        if interval_hours > 0 and not self._running:
            self._start_scheduler()
        elif interval_hours <= 0 and self._running:
            self._stop_scheduler()
    
    def _start_scheduler(self):
        """Start the backup scheduler."""
        self._running = True
        self._schedule_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._schedule_thread.start()
        logger.info(f"Backup scheduler started (every {self._interval_hours}h)")
    
    def _stop_scheduler(self):
        """Stop the backup scheduler."""
        self._running = False
        if self._schedule_thread:
            self._schedule_thread.join(timeout=1.0)
    
    def _scheduler_loop(self):
        """Background scheduler loop."""
        while self._running:
            try:
                self.backup_all(BackupType.INCREMENTAL)
            except Exception as e:
                logger.error(f"Scheduled backup failed: {e}")
            
            # Sleep in small increments for responsiveness
            sleep_time = self._interval_hours * 3600
            for _ in range(int(sleep_time / 10)):
                if not self._running:
                    break
                time.sleep(10)
    
    def backup_all(
        self,
        backup_type: BackupType = BackupType.FULL,
        targets: Optional[List[str]] = None
    ) -> Optional[BackupRecord]:
        """
        Perform backup of all targets.
        
        Args:
            backup_type: Type of backup
            targets: Specific targets (None for all)
            
        Returns:
            Backup record if successful
        """
        targets = targets or list(self._targets.keys())
        enabled_targets = [
            t for t in targets
            if t in self._targets and self._targets[t].enabled
        ]
        
        if not enabled_targets:
            logger.warning("No enabled targets to backup")
            return None
        
        # Generate backup ID
        timestamp = datetime.now()
        backup_id = timestamp.strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Starting {backup_type.value} backup: {backup_id}")
        
        # Collect files
        files_to_backup: List[Tuple[str, Path]] = []
        
        for target_name in enabled_targets:
            target = self._targets[target_name]
            target_files = self._collect_files(target, backup_type)
            files_to_backup.extend([(target_name, f) for f in target_files])
        
        if not files_to_backup:
            logger.info("No files changed, skipping backup")
            return None
        
        # Create backup archive
        archive_name = f"backup_{backup_id}"
        if self._compression == CompressionType.ZIP:
            archive_path = self._backup_dir / f"{archive_name}.zip"
            total_size = self._create_zip(files_to_backup, archive_path)
        elif self._compression == CompressionType.TAR_GZ:
            archive_path = self._backup_dir / f"{archive_name}.tar.gz"
            total_size = self._create_tar(files_to_backup, archive_path, "gz")
        elif self._compression == CompressionType.TAR_BZ2:
            archive_path = self._backup_dir / f"{archive_name}.tar.bz2"
            total_size = self._create_tar(files_to_backup, archive_path, "bz2")
        else:
            archive_path = self._backup_dir / archive_name
            total_size = self._create_directory(files_to_backup, archive_path)
        
        # Calculate checksum
        checksum = self._calculate_checksum(archive_path)
        
        # Create record
        record = BackupRecord(
            id=backup_id,
            timestamp=timestamp.timestamp(),
            backup_type=backup_type.value,
            targets=enabled_targets,
            file_count=len(files_to_backup),
            total_size=total_size,
            compressed_size=os.path.getsize(archive_path) if archive_path.exists() else total_size,
            location=str(archive_path),
            checksum=checksum
        )
        
        # Update history
        self._history.append(record)
        self._save_history()
        
        # Update hashes for incremental
        if backup_type != BackupType.FULL:
            self._update_hashes(files_to_backup)
            self._save_hashes()
        
        # Cleanup old backups
        self._cleanup_old_backups()
        
        # Callbacks
        for callback in self._callbacks:
            try:
                callback(record)
            except Exception as e:
                logger.error(f"Backup callback error: {e}")
        
        logger.info(f"Backup complete: {len(files_to_backup)} files, {record.compressed_size} bytes")
        return record
    
    def _collect_files(
        self,
        target: BackupTarget,
        backup_type: BackupType
    ) -> List[Path]:
        """Collect files for a target."""
        target_path = Path(target.path)
        if not target_path.exists():
            return []
        
        files = []
        
        for pattern in target.patterns:
            if target_path.is_file():
                if self._matches_pattern(target_path.name, pattern):
                    files.append(target_path)
            else:
                for match in target_path.rglob(pattern):
                    if match.is_file():
                        # Check excludes
                        if not any(self._matches_pattern(str(match), ex) for ex in target.exclude):
                            # For incremental, only include changed files
                            if backup_type == BackupType.INCREMENTAL:
                                if self._file_changed(match):
                                    files.append(match)
                            else:
                                files.append(match)
        
        return files
    
    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches pattern."""
        import fnmatch
        return fnmatch.fnmatch(filename, pattern)
    
    def _file_changed(self, path: Path) -> bool:
        """Check if file has changed since last backup."""
        current_hash = self._hash_file(path)
        stored_hash = self._file_hashes.get(str(path))
        return current_hash != stored_hash
    
    def _hash_file(self, path: Path) -> str:
        """Calculate file hash."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _update_hashes(self, files: List[Tuple[str, Path]]):
        """Update file hashes."""
        for _, path in files:
            self._file_hashes[str(path)] = self._hash_file(path)
    
    def _create_zip(
        self,
        files: List[Tuple[str, Path]],
        archive_path: Path
    ) -> int:
        """Create ZIP archive."""
        total_size = 0
        
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for target_name, file_path in files:
                arcname = f"{target_name}/{file_path.name}"
                zf.write(file_path, arcname)
                total_size += file_path.stat().st_size
        
        return total_size
    
    def _create_tar(
        self,
        files: List[Tuple[str, Path]],
        archive_path: Path,
        compression: str
    ) -> int:
        """Create TAR archive."""
        total_size = 0
        mode = f"w:{compression}"
        
        with tarfile.open(archive_path, mode) as tf:
            for target_name, file_path in files:
                arcname = f"{target_name}/{file_path.name}"
                tf.add(file_path, arcname)
                total_size += file_path.stat().st_size
        
        return total_size
    
    def _create_directory(
        self,
        files: List[Tuple[str, Path]],
        backup_path: Path
    ) -> int:
        """Create uncompressed backup directory."""
        total_size = 0
        backup_path.mkdir(parents=True, exist_ok=True)
        
        for target_name, file_path in files:
            dest_dir = backup_path / target_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            dest_file = dest_dir / file_path.name
            shutil.copy2(file_path, dest_file)
            total_size += file_path.stat().st_size
        
        return total_size
    
    def _calculate_checksum(self, path: Path) -> str:
        """Calculate checksum of backup."""
        if path.is_file():
            return self._hash_file(path)
        else:
            # Directory - hash filenames
            hasher = hashlib.sha256()
            for f in sorted(path.rglob("*")):
                hasher.update(f.name.encode())
            return hasher.hexdigest()[:16]
    
    def _cleanup_old_backups(self):
        """Remove old backups exceeding max_backups."""
        if len(self._history) <= self._max_backups:
            return
        
        # Sort by timestamp
        sorted_history = sorted(self._history, key=lambda x: x.timestamp)
        
        # Remove oldest
        to_remove = sorted_history[:-self._max_backups]
        
        for record in to_remove:
            try:
                path = Path(record.location)
                if path.exists():
                    if path.is_file():
                        path.unlink()
                    else:
                        shutil.rmtree(path)
                    logger.info(f"Removed old backup: {record.id}")
            except Exception as e:
                logger.error(f"Failed to remove backup {record.id}: {e}")
        
        # Update history
        self._history = sorted_history[-self._max_backups:]
        self._save_history()
    
    def restore(
        self,
        backup_id: str,
        targets: Optional[List[str]] = None,
        restore_dir: Optional[str] = None
    ) -> bool:
        """
        Restore from a backup.
        
        Args:
            backup_id: Backup ID or path
            targets: Specific targets to restore
            restore_dir: Directory to restore to (None = original locations)
            
        Returns:
            True if successful
        """
        # Find backup record
        record = None
        for r in self._history:
            if r.id == backup_id or r.location.endswith(backup_id):
                record = r
                break
        
        if not record:
            # Try as direct path
            archive_path = Path(backup_id)
            if not archive_path.exists():
                logger.error(f"Backup not found: {backup_id}")
                return False
        else:
            archive_path = Path(record.location)
        
        if not archive_path.exists():
            logger.error(f"Backup file not found: {archive_path}")
            return False
        
        logger.info(f"Restoring from: {archive_path}")
        
        # Determine restore directory
        if restore_dir:
            restore_path = Path(restore_dir)
            restore_path.mkdir(parents=True, exist_ok=True)
        else:
            restore_path = None  # Will restore to original locations
        
        # Extract based on format
        if archive_path.suffix == ".zip":
            self._restore_zip(archive_path, targets, restore_path)
        elif ".tar" in archive_path.suffix:
            self._restore_tar(archive_path, targets, restore_path)
        else:
            self._restore_directory(archive_path, targets, restore_path)
        
        logger.info("Restore complete")
        return True
    
    def _restore_zip(
        self,
        archive_path: Path,
        targets: Optional[List[str]],
        restore_path: Optional[Path]
    ):
        """Restore from ZIP archive."""
        with zipfile.ZipFile(archive_path, "r") as zf:
            for info in zf.infolist():
                if targets:
                    target = info.filename.split("/")[0]
                    if target not in targets:
                        continue
                
                if restore_path:
                    zf.extract(info, restore_path)
                else:
                    # Extract to original location
                    target = info.filename.split("/")[0]
                    if target in self._targets:
                        dest = Path(self._targets[target].path) / info.filename.split("/", 1)[1]
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(info) as src, open(dest, "wb") as dst:
                            dst.write(src.read())
    
    def _restore_tar(
        self,
        archive_path: Path,
        targets: Optional[List[str]],
        restore_path: Optional[Path]
    ):
        """Restore from TAR archive."""
        with tarfile.open(archive_path, "r:*") as tf:
            for member in tf.getmembers():
                if targets:
                    target = member.name.split("/")[0]
                    if target not in targets:
                        continue
                
                if restore_path:
                    tf.extract(member, restore_path)
                else:
                    target = member.name.split("/")[0]
                    if target in self._targets:
                        dest = Path(self._targets[target].path) / member.name.split("/", 1)[1]
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with tf.extractfile(member) as src, open(dest, "wb") as dst:
                            dst.write(src.read())
    
    def _restore_directory(
        self,
        archive_path: Path,
        targets: Optional[List[str]],
        restore_path: Optional[Path]
    ):
        """Restore from directory backup."""
        for target_dir in archive_path.iterdir():
            if targets and target_dir.name not in targets:
                continue
            
            for file_path in target_dir.rglob("*"):
                if file_path.is_file():
                    if restore_path:
                        dest = restore_path / file_path.relative_to(archive_path)
                    else:
                        target = target_dir.name
                        if target in self._targets:
                            dest = Path(self._targets[target].path) / file_path.relative_to(target_dir)
                        else:
                            continue
                    
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, dest)
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all backups."""
        return [r.to_dict() for r in self._history]
    
    def get_backup(self, backup_id: str) -> Optional[Dict[str, Any]]:
        """Get backup details."""
        for r in self._history:
            if r.id == backup_id:
                return r.to_dict()
        return None
    
    def verify_backup(self, backup_id: str) -> bool:
        """Verify backup integrity."""
        record = None
        for r in self._history:
            if r.id == backup_id:
                record = r
                break
        
        if not record:
            return False
        
        path = Path(record.location)
        if not path.exists():
            return False
        
        current_checksum = self._calculate_checksum(path)
        return current_checksum == record.checksum
    
    def add_callback(self, callback: Callable[[BackupRecord], None]):
        """Add backup completion callback."""
        self._callbacks.append(callback)
    
    def _load_history(self):
        """Load backup history."""
        if self._history_file.exists():
            try:
                with open(self._history_file) as f:
                    data = json.load(f)
                
                self._history = [
                    BackupRecord(**r) for r in data
                ]
            except Exception as e:
                logger.error(f"Failed to load backup history: {e}")
    
    def _save_history(self):
        """Save backup history."""
        try:
            with open(self._history_file, "w") as f:
                json.dump([r.to_dict() for r in self._history], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save backup history: {e}")
    
    def _load_hashes(self):
        """Load file hashes."""
        if self._hashes_file.exists():
            try:
                with open(self._hashes_file) as f:
                    self._file_hashes = json.load(f)
            except Exception:
                pass
    
    def _save_hashes(self):
        """Save file hashes."""
        try:
            with open(self._hashes_file, "w") as f:
                json.dump(self._file_hashes, f)
        except Exception:
            pass


# Global instance
_backup_manager: Optional[BackupManager] = None


def get_backup_manager() -> BackupManager:
    """Get or create global backup manager."""
    global _backup_manager
    if _backup_manager is None:
        _backup_manager = BackupManager()
    return _backup_manager
