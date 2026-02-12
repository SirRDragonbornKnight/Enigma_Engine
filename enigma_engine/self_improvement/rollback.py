"""
Rollback Manager for Self-Improvement System

Provides safety mechanisms to revert to previous model states
if self-training degrades quality.

Features:
- Automatic backups before training
- Checkpoint management
- Quick rollback capability
- Retention policy for old backups
"""

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Backup:
    """Information about a backup."""
    id: str
    timestamp: str
    description: str
    path: str
    model_hash: str = ""
    size_bytes: int = 0
    quality_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "description": self.description,
            "path": self.path,
            "model_hash": self.model_hash,
            "size_bytes": self.size_bytes,
            "quality_score": self.quality_score,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Backup":
        return cls(
            id=data.get("id", ""),
            timestamp=data.get("timestamp", ""),
            description=data.get("description", ""),
            path=data.get("path", ""),
            model_hash=data.get("model_hash", ""),
            size_bytes=data.get("size_bytes", 0),
            quality_score=data.get("quality_score"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RollbackConfig:
    """Configuration for rollback manager."""
    backup_dir: str = ""
    model_dir: str = ""
    max_backups: int = 10
    retention_days: int = 30
    auto_backup_before_training: bool = True
    compress_backups: bool = True
    backup_lora_only: bool = True  # Only backup LoRA adapters (smaller)


class RollbackManager:
    """
    Manages model backups and rollbacks for safe self-improvement.
    
    Usage:
        manager = RollbackManager()
        
        # Create backup before training
        backup_id = manager.create_backup("before_self_training")
        
        # ... do training ...
        
        # If quality degraded, rollback
        if quality_dropped:
            manager.restore_backup(backup_id)
    """
    
    def __init__(self, config: Optional[RollbackConfig] = None):
        base_path = Path(__file__).parent.parent.parent
        
        if config is None:
            config = RollbackConfig()
        
        if not config.backup_dir:
            config.backup_dir = str(base_path / "backups" / "self_improvement")
        
        if not config.model_dir:
            config.model_dir = str(base_path / "models" / "enigma")
        
        self.config = config
        self.backup_dir = Path(config.backup_dir)
        self.model_dir = Path(config.model_dir)
        self._index_path = self.backup_dir / "backup_index.json"
        
        # Ensure directories exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Load backup index
        self.backups: Dict[str, Backup] = {}
        self._load_index()
    
    def _load_index(self):
        """Load backup index from disk."""
        if self._index_path.exists():
            try:
                with open(self._index_path, 'r') as f:
                    data = json.load(f)
                
                for backup_data in data.get("backups", []):
                    backup = Backup.from_dict(backup_data)
                    self.backups[backup.id] = backup
                
                logger.info(f"Loaded {len(self.backups)} backup entries")
            except Exception as e:
                logger.warning(f"Failed to load backup index: {e}")
    
    def _save_index(self):
        """Save backup index to disk."""
        try:
            data = {
                "last_updated": datetime.now().isoformat(),
                "backups": [b.to_dict() for b in self.backups.values()],
            }
            
            with open(self._index_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save backup index: {e}")
    
    def create_backup(
        self,
        description: str = "Manual backup",
        quality_score: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Create a backup of the current model state.
        
        Args:
            description: Human-readable description
            quality_score: Optional quality score at time of backup
            metadata: Additional metadata
            
        Returns:
            Backup ID
        """
        # Generate unique ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_id = f"backup_{timestamp}"
        
        # Create backup directory
        backup_path = self.backup_dir / backup_id
        backup_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating backup: {backup_id}")
        
        try:
            if self.config.backup_lora_only:
                # Backup only LoRA adapters (faster, smaller)
                size = self._backup_lora(backup_path)
            else:
                # Full model backup
                size = self._backup_full_model(backup_path)
            
            # Calculate model hash for verification
            model_hash = self._calculate_model_hash()
            
            # Create backup entry
            backup = Backup(
                id=backup_id,
                timestamp=timestamp,
                description=description,
                path=str(backup_path),
                model_hash=model_hash,
                size_bytes=size,
                quality_score=quality_score,
                metadata=metadata or {},
            )
            
            self.backups[backup_id] = backup
            self._save_index()
            
            # Enforce retention policy
            self._enforce_retention()
            
            logger.info(f"Backup created: {backup_id} ({size / 1024 / 1024:.1f} MB)")
            return backup_id
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            # Clean up failed backup
            if backup_path.exists():
                shutil.rmtree(backup_path, ignore_errors=True)
            raise
    
    def _backup_lora(self, backup_path: Path) -> int:
        """Backup only LoRA adapters."""
        total_size = 0
        
        # Find LoRA adapter files
        lora_patterns = ["lora*.pt", "lora*.json", "adapter*.pt", "adapter*.json"]
        
        for pattern in lora_patterns:
            for src_file in self.model_dir.rglob(pattern):
                dst_file = backup_path / src_file.relative_to(self.model_dir)
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                total_size += src_file.stat().st_size
        
        # Also backup checkpoint directory if exists
        checkpoint_dir = self.model_dir.parent / "checkpoints" / "self_training"
        if checkpoint_dir.exists():
            # Get most recent checkpoint
            checkpoints = sorted(checkpoint_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
            if checkpoints:
                most_recent = checkpoints[0]
                dst = backup_path / "checkpoint"
                shutil.copytree(most_recent, dst)
                total_size += sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
        
        # Save model config if exists
        config_files = ["config.json", "model_config.json", "forge_config.json"]
        for config_name in config_files:
            config_path = self.model_dir / config_name
            if config_path.exists():
                shutil.copy2(config_path, backup_path / config_name)
                total_size += config_path.stat().st_size
        
        return total_size
    
    def _backup_full_model(self, backup_path: Path) -> int:
        """Full model backup."""
        if not self.model_dir.exists():
            logger.warning(f"Model directory not found: {self.model_dir}")
            return 0
        
        # Copy entire model directory
        model_backup = backup_path / "model"
        shutil.copytree(self.model_dir, model_backup)
        
        # Calculate size
        total_size = sum(
            f.stat().st_size for f in model_backup.rglob("*") if f.is_file()
        )
        
        # Compress if configured
        if self.config.compress_backups:
            archive_path = backup_path / "model.zip"
            shutil.make_archive(str(archive_path)[:-4], 'zip', model_backup)
            shutil.rmtree(model_backup)
            total_size = archive_path.stat().st_size
        
        return total_size
    
    def _calculate_model_hash(self) -> str:
        """Calculate hash of model for verification."""
        if not self.model_dir.exists():
            return ""
        
        # Hash important files
        hash_md5 = hashlib.md5()
        
        important_files = ["model.pt", "config.json", "model.safetensors"]
        for filename in important_files:
            filepath = self.model_dir / filename
            if filepath.exists():
                with open(filepath, 'rb') as f:
                    # Read in chunks for large files
                    for chunk in iter(lambda: f.read(8192), b""):
                        hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def restore_backup(self, backup_id: str) -> bool:
        """
        Restore a backup.
        
        Args:
            backup_id: ID of backup to restore
            
        Returns:
            True if successful
        """
        if backup_id not in self.backups:
            logger.error(f"Backup not found: {backup_id}")
            return False
        
        backup = self.backups[backup_id]
        backup_path = Path(backup.path)
        
        if not backup_path.exists():
            logger.error(f"Backup directory not found: {backup_path}")
            return False
        
        logger.info(f"Restoring backup: {backup_id}")
        
        try:
            # Create safety backup of current state
            safety_id = self.create_backup("Pre-rollback safety backup")
            
            if self.config.backup_lora_only:
                self._restore_lora(backup_path)
            else:
                self._restore_full_model(backup_path)
            
            logger.info(f"Backup restored successfully: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False
    
    def _restore_lora(self, backup_path: Path):
        """Restore LoRA adapters from backup."""
        # Restore LoRA files
        for src_file in backup_path.rglob("*.pt"):
            if "checkpoint" not in str(src_file):
                dst_file = self.model_dir / src_file.relative_to(backup_path)
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
        
        for src_file in backup_path.rglob("*.json"):
            if "checkpoint" not in str(src_file) and src_file.name != "backup_index.json":
                dst_file = self.model_dir / src_file.relative_to(backup_path)
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
        
        # Restore checkpoint if exists
        checkpoint_src = backup_path / "checkpoint"
        if checkpoint_src.exists():
            checkpoint_dst = self.model_dir.parent / "checkpoints" / "self_training" / "restored"
            if checkpoint_dst.exists():
                shutil.rmtree(checkpoint_dst)
            shutil.copytree(checkpoint_src, checkpoint_dst)
    
    def _restore_full_model(self, backup_path: Path):
        """Restore full model from backup."""
        model_backup = backup_path / "model"
        archive_path = backup_path / "model.zip"
        
        if archive_path.exists():
            # Extract compressed backup
            shutil.unpack_archive(archive_path, backup_path / "model_extracted")
            model_backup = backup_path / "model_extracted"
        
        if model_backup.exists():
            # Remove current model
            if self.model_dir.exists():
                shutil.rmtree(self.model_dir)
            
            # Copy backup
            shutil.copytree(model_backup, self.model_dir)
    
    def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup."""
        if backup_id not in self.backups:
            return False
        
        backup = self.backups[backup_id]
        backup_path = Path(backup.path)
        
        try:
            if backup_path.exists():
                shutil.rmtree(backup_path)
            
            del self.backups[backup_id]
            self._save_index()
            
            logger.info(f"Deleted backup: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete backup: {e}")
            return False
    
    def _enforce_retention(self):
        """Enforce retention policy."""
        # Remove old backups
        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
        
        to_delete = []
        for backup_id, backup in self.backups.items():
            try:
                backup_date = datetime.strptime(backup.timestamp, "%Y%m%d_%H%M%S")
                if backup_date < cutoff_date:
                    to_delete.append(backup_id)
            except ValueError:
                pass
        
        # Keep at least max_backups most recent
        sorted_backups = sorted(
            self.backups.items(),
            key=lambda x: x[1].timestamp,
            reverse=True,
        )
        
        if len(sorted_backups) > self.config.max_backups:
            for backup_id, _ in sorted_backups[self.config.max_backups:]:
                if backup_id not in to_delete:
                    to_delete.append(backup_id)
        
        # Delete
        for backup_id in to_delete:
            logger.info(f"Removing old backup: {backup_id}")
            self.delete_backup(backup_id)
    
    def list_backups(self) -> List[Backup]:
        """List all backups, sorted by date (newest first)."""
        return sorted(
            self.backups.values(),
            key=lambda x: x.timestamp,
            reverse=True,
        )
    
    def get_backup(self, backup_id: str) -> Optional[Backup]:
        """Get backup by ID."""
        return self.backups.get(backup_id)
    
    def get_latest_backup(self) -> Optional[Backup]:
        """Get most recent backup."""
        backups = self.list_backups()
        return backups[0] if backups else None
    
    def get_best_backup(self) -> Optional[Backup]:
        """Get backup with highest quality score."""
        backups_with_score = [
            b for b in self.backups.values() if b.quality_score is not None
        ]
        
        if not backups_with_score:
            return None
        
        return max(backups_with_score, key=lambda x: x.quality_score or 0.0)
    
    def verify_backup(self, backup_id: str) -> bool:
        """Verify backup integrity."""
        if backup_id not in self.backups:
            return False
        
        backup = self.backups[backup_id]
        backup_path = Path(backup.path)
        
        # Check path exists
        if not backup_path.exists():
            logger.warning(f"Backup path not found: {backup_path}")
            return False
        
        # Check has files
        files = list(backup_path.rglob("*"))
        if not files:
            logger.warning(f"Backup is empty: {backup_id}")
            return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get backup statistics."""
        total_size = sum(b.size_bytes for b in self.backups.values())
        
        return {
            "total_backups": len(self.backups),
            "total_size_mb": total_size / 1024 / 1024,
            "oldest_backup": min(b.timestamp for b in self.backups.values()) if self.backups else None,
            "newest_backup": max(b.timestamp for b in self.backups.values()) if self.backups else None,
            "backup_dir": str(self.backup_dir),
        }


# CLI interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Rollback Manager")
    parser.add_argument("command", choices=["list", "create", "restore", "delete", "stats", "verify"],
                       help="Command to run")
    parser.add_argument("--id", help="Backup ID (for restore/delete/verify)")
    parser.add_argument("--description", default="CLI backup", help="Backup description")
    
    args = parser.parse_args()
    
    manager = RollbackManager()
    
    if args.command == "list":
        backups = manager.list_backups()
        if not backups:
            print("No backups found")
        else:
            print(f"{'ID':<30} {'Timestamp':<20} {'Size (MB)':<10} {'Description'}")
            print("-" * 80)
            for b in backups:
                size_mb = b.size_bytes / 1024 / 1024
                print(f"{b.id:<30} {b.timestamp:<20} {size_mb:<10.1f} {b.description}")
    
    elif args.command == "create":
        backup_id = manager.create_backup(args.description)
        print(f"Created backup: {backup_id}")
    
    elif args.command == "restore":
        if not args.id:
            print("Error: --id required for restore")
            return
        
        if manager.restore_backup(args.id):
            print(f"Restored backup: {args.id}")
        else:
            print(f"Failed to restore backup: {args.id}")
    
    elif args.command == "delete":
        if not args.id:
            print("Error: --id required for delete")
            return
        
        if manager.delete_backup(args.id):
            print(f"Deleted backup: {args.id}")
        else:
            print(f"Failed to delete backup: {args.id}")
    
    elif args.command == "stats":
        stats = manager.get_stats()
        print(json.dumps(stats, indent=2))
    
    elif args.command == "verify":
        if not args.id:
            print("Error: --id required for verify")
            return
        
        if manager.verify_backup(args.id):
            print(f"Backup {args.id} is valid")
        else:
            print(f"Backup {args.id} is invalid or corrupted")


if __name__ == "__main__":
    main()
