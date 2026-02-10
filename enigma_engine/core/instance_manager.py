"""
Multi-Instance Manager for Forge

Manages multiple Forge instances running simultaneously.

Features:
- Lock files to prevent conflicts
- Shared memory pool (optional)
- Inter-instance communication
- Resource allocation

Usage:
    from enigma_engine.core.instance_manager import InstanceManager
    
    manager = InstanceManager()
    
    # Acquire model lock
    if manager.acquire_model_lock("my_model"):
        # Use the model
        ...
        manager.release_model_lock("my_model")
    
    # List running instances
    instances = manager.list_running_instances()
"""

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import psutil

from ..config import CONFIG  # noqa: F401 - imported for side effects

# Lock directory in user's home
LOCK_DIR = Path.home() / ".enigma_engine" / "locks"
LOCK_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class InstanceInfo:
    """Information about a running instance."""
    
    instance_id: str
    pid: int
    started_at: str
    model_name: Optional[str] = None
    host: str = "localhost"
    port: Optional[int] = None
    status: str = "running"
    locked_models: list[str] = None
    
    def __post_init__(self):
        if self.locked_models is None:
            self.locked_models = []
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'InstanceInfo':
        """Create from dictionary."""
        return cls(**data)


class InstanceManager:
    """
    Manage multiple Forge instances.
    
    Uses lock files to:
    - Prevent multiple instances from using the same model simultaneously
    - Track running instances
    - Enable inter-instance communication
    """
    
    def __init__(self, instance_id: Optional[str] = None):
        """
        Initialize instance manager.
        
        Args:
            instance_id: Unique identifier for this instance (auto-generated if None)
        """
        self.instance_id = instance_id or self._generate_instance_id()
        self.lock_file = LOCK_DIR / f"instance_{self.instance_id}.lock"
        self.model_locks: dict[str, Path] = {}
        self.info = InstanceInfo(
            instance_id=self.instance_id,
            pid=os.getpid(),
            started_at=datetime.now().isoformat()
        )
        
        # Register this instance
        self._register_instance()
    
    def _generate_instance_id(self) -> str:
        """Generate unique instance ID."""
        return str(uuid.uuid4())[:8]
    
    def _register_instance(self):
        """Register this instance with a lock file."""
        try:
            with open(self.lock_file, 'w') as f:
                json.dump(self.info.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Warning: Could not register instance: {e}")
    
    def _update_instance_info(self):
        """Update instance lock file with current info."""
        try:
            if self.lock_file.exists():
                with open(self.lock_file, 'w') as f:
                    json.dump(self.info.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Warning: Could not update instance info: {e}")
    
    def acquire_model_lock(self, model_name: str, timeout: float = 0.0) -> bool:
        """
        Try to get exclusive access to a model.
        
        Args:
            model_name: Name of the model to lock
            timeout: How long to wait for lock (0 = don't wait)
        
        Returns:
            True if lock acquired, False otherwise
        """
        lock_file = LOCK_DIR / f"model_{model_name}.lock"
        start_time = time.time()
        
        while True:
            # Check if lock file exists and is valid
            if lock_file.exists():
                try:
                    with open(lock_file) as f:
                        lock_info = json.load(f)
                    
                    # Check if the process holding the lock is still alive
                    lock_pid = lock_info.get('pid')
                    if lock_pid and not self._is_process_alive(lock_pid):
                        # Process is dead, remove stale lock
                        lock_file.unlink()
                    else:
                        # Lock is held by active process
                        if timeout <= 0:
                            return False
                        
                        # Wait and retry
                        elapsed = time.time() - start_time
                        if elapsed >= timeout:
                            return False
                        
                        time.sleep(0.1)
                        continue
                except Exception:
                    # Corrupted lock file, remove it
                    try:
                        lock_file.unlink()
                    except Exception:
                        pass
            
            # Try to acquire lock
            try:
                lock_data = {
                    'instance_id': self.instance_id,
                    'pid': os.getpid(),
                    'model_name': model_name,
                    'acquired_at': datetime.now().isoformat()
                }
                
                with open(lock_file, 'w') as f:
                    json.dump(lock_data, f, indent=2)
                
                self.model_locks[model_name] = lock_file
                self.info.locked_models.append(model_name)
                self._update_instance_info()
                
                return True
            except Exception as e:
                print(f"Failed to acquire lock for {model_name}: {e}")
                return False
    
    def release_model_lock(self, model_name: str):
        """
        Release model lock.
        
        Args:
            model_name: Name of the model to unlock
        """
        if model_name not in self.model_locks:
            return
        
        lock_file = self.model_locks[model_name]
        
        try:
            if lock_file.exists():
                lock_file.unlink()
            del self.model_locks[model_name]
            
            if model_name in self.info.locked_models:
                self.info.locked_models.remove(model_name)
                self._update_instance_info()
        except Exception as e:
            print(f"Warning: Could not release lock for {model_name}: {e}")
    
    def release_all_locks(self):
        """Release all locks held by this instance."""
        for model_name in list(self.model_locks.keys()):
            self.release_model_lock(model_name)
    
    def _is_process_alive(self, pid: int) -> bool:
        """Check if a process is alive."""
        try:
            process = psutil.Process(pid)
            return process.is_running()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    
    def list_running_instances(self) -> list[dict[str, Any]]:
        """
        List all running Forge instances.
        
        Returns:
            List of instance information dictionaries
        """
        instances = []
        
        # Clean up stale lock files first
        self._cleanup_stale_locks()
        
        # Read all instance lock files
        for lock_file in LOCK_DIR.glob("instance_*.lock"):
            try:
                with open(lock_file) as f:
                    info = json.load(f)
                
                # Verify process is still alive
                pid = info.get('pid')
                if pid and self._is_process_alive(pid):
                    instances.append(info)
                else:
                    # Stale lock, remove it
                    try:
                        lock_file.unlink()
                    except Exception:
                        pass
            except Exception as e:
                # Corrupted file, skip it
                print(f"Warning: Could not read {lock_file}: {e}")
        
        return instances
    
    def _cleanup_stale_locks(self):
        """Remove lock files for dead processes."""
        for lock_file in LOCK_DIR.glob("*.lock"):
            try:
                with open(lock_file) as f:
                    info = json.load(f)
                
                pid = info.get('pid')
                if pid and not self._is_process_alive(pid):
                    lock_file.unlink()
            except Exception:
                # Corrupted or inaccessible file
                try:
                    # If file is very old (>1 day), remove it
                    if lock_file.stat().st_mtime < time.time() - 86400:
                        lock_file.unlink()
                except Exception:
                    pass
    
    def send_to_instance(self, instance_id: str, message: str):
        """
        Send message to another instance.
        
        Note: This is a basic implementation using file-based messaging.
        For production, consider using proper IPC mechanisms.
        
        Args:
            instance_id: Target instance ID
            message: Message to send
        """
        messages_dir = LOCK_DIR / "messages"
        messages_dir.mkdir(exist_ok=True)
        
        message_file = messages_dir / f"{instance_id}_{int(time.time()*1000)}.msg"
        
        try:
            with open(message_file, 'w') as f:
                json.dump({
                    'from': self.instance_id,
                    'to': instance_id,
                    'message': message,
                    'timestamp': datetime.now().isoformat()
                }, f)
        except Exception as e:
            print(f"Failed to send message: {e}")
    
    def receive_messages(self) -> list[dict[str, Any]]:
        """
        Receive messages sent to this instance.
        
        Returns:
            List of message dictionaries
        """
        messages_dir = LOCK_DIR / "messages"
        if not messages_dir.exists():
            return []
        
        messages = []
        
        for msg_file in messages_dir.glob(f"{self.instance_id}_*.msg"):
            try:
                with open(msg_file) as f:
                    msg = json.load(f)
                messages.append(msg)
                
                # Remove processed message
                msg_file.unlink()
            except Exception as e:
                print(f"Warning: Could not read message {msg_file}: {e}")
        
        return messages
    
    def get_instance_info(self, instance_id: str) -> Optional[dict[str, Any]]:
        """
        Get information about a specific instance.
        
        Args:
            instance_id: Instance ID to query
        
        Returns:
            Instance info dictionary or None if not found
        """
        lock_file = LOCK_DIR / f"instance_{instance_id}.lock"
        
        if not lock_file.exists():
            return None
        
        try:
            with open(lock_file) as f:
                info = json.load(f)
            
            # Verify process is alive
            pid = info.get('pid')
            if pid and self._is_process_alive(pid):
                return info
            else:
                # Stale lock
                try:
                    lock_file.unlink()
                except Exception:
                    pass
                return None
        except Exception:
            return None
    
    def shutdown(self):
        """Shutdown this instance and cleanup."""
        # Release all locks
        self.release_all_locks()
        
        # Remove instance lock file
        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
        except Exception as e:
            print(f"Warning: Could not remove instance lock: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.shutdown()
        except Exception:
            pass


# Convenience functions
def get_active_instances() -> list[dict[str, Any]]:
    """Get list of all active Forge instances."""
    manager = InstanceManager()
    return manager.list_running_instances()


def cleanup_stale_locks():
    """Clean up stale lock files."""
    manager = InstanceManager()
    manager._cleanup_stale_locks()
