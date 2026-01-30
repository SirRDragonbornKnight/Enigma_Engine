"""
Memory Backup Scheduling for ForgeAI
Provides automatic backup and restoration of memories.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .categorization import MemoryCategorization
from .export_import import MemoryExporter, MemoryImporter

logger = logging.getLogger(__name__)


class MemoryBackupScheduler:
    """Schedule automatic memory backups."""
    
    def __init__(
        self,
        memory_system: MemoryCategorization,
        backup_dir: Path
    ):
        """
        Initialize backup scheduler.
        
        Args:
            memory_system: Memory categorization system
            backup_dir: Directory to store backups
        """
        self.memory_system = memory_system
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.exporter = MemoryExporter(memory_system)
        self.importer = MemoryImporter(memory_system)
        
        self._scheduler_thread = None
        self._running = False
        self._schedule_config = {}
    
    def create_backup(self, name: Optional[str] = None) -> Path:
        """
        Create a backup now.
        
        Args:
            name: Optional backup name (uses timestamp if None)
            
        Returns:
            Path to backup file
        """
        if name is None:
            name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Ensure unique filename
        backup_path = self.backup_dir / f"{name}.zip"
        counter = 1
        while backup_path.exists():
            backup_path = self.backup_dir / f"{name}_{counter}.zip"
            counter += 1
        
        # Create backup
        try:
            self.exporter.export_to_archive(backup_path, include_vectors=False)
            logger.info(f"Created backup: {backup_path}")
            return backup_path
        
        except Exception as e:
            logger.error(f"Failed to create backup: {e}", exc_info=True)
            raise
    
    def schedule_daily(self, time_str: str = "02:00"):
        """
        Schedule daily backups at specified time.
        
        Args:
            time_str: Time in HH:MM format (24-hour)
        """
        # Parse time
        try:
            hour, minute = map(int, time_str.split(':'))
            if not (0 <= hour < 24 and 0 <= minute < 60):
                raise ValueError("Invalid time")
        except Exception:
            raise ValueError(f"Invalid time format: {time_str}. Use HH:MM format.")
        
        self._schedule_config = {
            'type': 'daily',
            'hour': hour,
            'minute': minute
        }
        
        if not self._running:
            self.start()
        
        logger.info(f"Scheduled daily backups at {time_str}")
    
    def schedule_hourly(self):
        """Schedule hourly backups."""
        self._schedule_config = {
            'type': 'hourly',
            'interval': 3600  # seconds
        }
        
        if not self._running:
            self.start()
        
        logger.info("Scheduled hourly backups")
    
    def schedule_interval(self, hours: int):
        """
        Schedule backups at regular intervals.
        
        Args:
            hours: Hours between backups
        """
        self._schedule_config = {
            'type': 'interval',
            'interval': hours * 3600  # seconds
        }
        
        if not self._running:
            self.start()
        
        logger.info(f"Scheduled backups every {hours} hours")
    
    def _scheduler_loop(self):
        """Main scheduler loop (runs in thread)."""
        last_backup = 0
        
        while self._running:
            try:
                current_time = time.time()
                should_backup = False
                
                if self._schedule_config.get('type') == 'daily':
                    # Check if it's time for daily backup
                    now = datetime.now()
                    target_hour = self._schedule_config['hour']
                    target_minute = self._schedule_config['minute']
                    
                    # If we're past the target time and haven't backed up today
                    if (now.hour > target_hour or 
                        (now.hour == target_hour and now.minute >= target_minute)):
                        
                        # Check if last backup was yesterday or earlier
                        if last_backup == 0 or \
                           datetime.fromtimestamp(last_backup).date() < now.date():
                            should_backup = True
                
                elif self._schedule_config.get('type') in ['hourly', 'interval']:
                    # Check if interval has passed
                    interval = self._schedule_config['interval']
                    if current_time - last_backup >= interval:
                        should_backup = True
                
                if should_backup:
                    logger.info("Running scheduled backup...")
                    self.create_backup()
                    last_backup = current_time
                
                # Sleep for a bit before checking again
                time.sleep(60)  # Check every minute
            
            except Exception as e:
                logger.error(f"Error in backup scheduler: {e}", exc_info=True)
                time.sleep(300)  # Sleep 5 minutes on error
    
    def start(self):
        """Start the backup scheduler."""
        if self._running:
            logger.warning("Backup scheduler already running")
            return
        
        if not self._schedule_config:
            raise ValueError("No schedule configured. Call schedule_daily() or schedule_hourly() first.")
        
        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="MemoryBackupScheduler"
        )
        self._scheduler_thread.start()
        
        logger.info("Backup scheduler started")
    
    def stop(self):
        """Stop the backup scheduler."""
        if self._running:
            self._running = False
            if self._scheduler_thread:
                self._scheduler_thread.join(timeout=5)
            logger.info("Backup scheduler stopped")
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all available backups.
        
        Returns:
            List of backup information dictionaries
        """
        backups = []
        
        for backup_file in sorted(self.backup_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True):
            stat = backup_file.stat()
            backups.append({
                'name': backup_file.stem,
                'path': str(backup_file),
                'size_bytes': stat.st_size,
                'size_mb': stat.st_size / (1024 * 1024),
                'created': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'age_days': (time.time() - stat.st_mtime) / 86400
            })
        
        return backups
    
    def restore_from_backup(
        self,
        backup_path: Path,
        merge: bool = False
    ) -> bool:
        """
        Restore memories from a backup.
        
        Args:
            backup_path: Path to backup file
            merge: Merge with existing memories (vs replace)
            
        Returns:
            True if successful
        """
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")
        
        try:
            # Import from backup
            stats = self.importer.import_from_archive(
                backup_path,
                merge=merge,
                import_vectors=False
            )
            
            logger.info(
                f"Restored {stats['imported_count']} memories from backup: {backup_path}"
            )
            return True
        
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}", exc_info=True)
            return False
    
    def cleanup_old_backups(self, keep_days: int = 30, keep_minimum: int = 5):
        """
        Remove backups older than specified days.
        
        Args:
            keep_days: Keep backups newer than this many days
            keep_minimum: Always keep at least this many backups (newest)
        """
        backups = self.list_backups()
        
        if len(backups) <= keep_minimum:
            logger.info(f"Only {len(backups)} backups, keeping all (minimum: {keep_minimum})")
            return
        
        # Sort by age
        backups_sorted = sorted(backups, key=lambda b: b['age_days'])
        
        # Keep minimum newest backups
        to_keep = set(b['path'] for b in backups_sorted[:keep_minimum])
        
        # Also keep backups within the retention period
        cutoff_days = keep_days
        for backup in backups:
            if backup['age_days'] <= cutoff_days:
                to_keep.add(backup['path'])
        
        # Delete old backups
        deleted_count = 0
        for backup in backups:
            if backup['path'] not in to_keep:
                try:
                    Path(backup['path']).unlink()
                    logger.info(f"Deleted old backup: {backup['name']} (age: {backup['age_days']:.1f} days)")
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete backup {backup['name']}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old backups")
        else:
            logger.info("No old backups to clean up")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get scheduler status.
        
        Returns:
            Status information dictionary
        """
        backups = self.list_backups()
        
        status = {
            'running': self._running,
            'schedule': self._schedule_config,
            'backup_count': len(backups),
            'total_backup_size_mb': sum(b['size_mb'] for b in backups),
            'latest_backup': backups[0] if backups else None,
            'backup_dir': str(self.backup_dir)
        }
        
        return status
