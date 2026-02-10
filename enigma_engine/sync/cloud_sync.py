"""
Cloud Sync Settings
====================

Sync user preferences and settings across devices using cloud storage.
Supports multiple backends (Firebase, REST API, local backup).

Usage:
    from enigma_engine.sync.cloud_sync import CloudSyncService
    
    # Initialize with Firebase
    sync = CloudSyncService(backend='firebase', user_id='user123')
    
    # Or with custom REST API
    sync = CloudSyncService(
        backend='rest',
        api_url='https://your-api.com/sync',
        api_key='your-key'
    )
    
    # Upload current settings
    await sync.upload_settings()
    
    # Download and apply settings
    await sync.download_settings(apply=True)
    
    # Auto-sync (watch for changes)
    sync.start_auto_sync(interval=300)  # Every 5 minutes
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class SyncSettings:
    """Settings to be synced across devices."""
    
    # UI Preferences
    theme: str = "dark"
    font_scale: float = 1.0
    window_geometry: dict = field(default_factory=dict)
    sidebar_collapsed: bool = False
    
    # Chat Settings
    system_prompt: str = ""
    ai_name: str = "Enigma"
    user_name: str = "User"
    temperature: float = 0.8
    max_tokens: int = 512
    
    # Model Settings
    default_model: str = ""
    model_device: str = "auto"
    precision: str = "float32"
    
    # Voice Settings
    voice_enabled: bool = False
    voice_model: str = "default"
    speech_rate: float = 1.0
    
    # Avatar Settings
    avatar_enabled: bool = False
    avatar_mode: str = "idle"
    
    # API Keys (encrypted)
    encrypted_api_keys: dict = field(default_factory=dict)
    
    # Custom Settings
    custom: dict = field(default_factory=dict)
    
    # Metadata
    last_modified: str = ""
    device_id: str = ""
    version: int = 1
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SyncSettings':
        """Create from dictionary."""
        # Filter to only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
    
    def merge_with(self, other: 'SyncSettings', strategy: str = 'newer') -> 'SyncSettings':
        """
        Merge with another settings object.
        
        Strategies:
        - 'newer': Take values from whichever was modified more recently
        - 'local': Prefer local values (keep self)
        - 'remote': Prefer remote values (keep other)
        """
        if strategy == 'local':
            return self
        elif strategy == 'remote':
            return other
        elif strategy == 'newer':
            # Compare timestamps
            self_time = self.last_modified or "0"
            other_time = other.last_modified or "0"
            return other if other_time > self_time else self
        else:
            return self


@dataclass
class SyncResult:
    """Result of a sync operation."""
    success: bool
    action: str  # 'upload', 'download', 'merge', 'conflict'
    timestamp: str
    error: Optional[str] = None
    changes: List[str] = field(default_factory=list)


class CloudSyncBackend:
    """Base class for cloud sync backends."""
    
    async def upload(self, user_id: str, settings: dict) -> bool:
        """Upload settings to cloud."""
        raise NotImplementedError
    
    async def download(self, user_id: str) -> Optional[dict]:
        """Download settings from cloud."""
        raise NotImplementedError
    
    async def get_last_modified(self, user_id: str) -> Optional[str]:
        """Get last modified timestamp."""
        raise NotImplementedError
    
    def is_available(self) -> bool:
        """Check if backend is available."""
        return True


class RestApiBackend(CloudSyncBackend):
    """REST API backend for cloud sync."""
    
    def __init__(self, api_url: str, api_key: Optional[str] = None):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self._session = None
    
    def _get_headers(self) -> dict:
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers
    
    async def upload(self, user_id: str, settings: dict) -> bool:
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.put(
                    f"{self.api_url}/settings/{user_id}",
                    json=settings,
                    headers=self._get_headers()
                ) as response:
                    return response.status in (200, 201)
                    
        except ImportError:
            # Fall back to synchronous requests
            import requests
            try:
                response = requests.put(
                    f"{self.api_url}/settings/{user_id}",
                    json=settings,
                    headers=self._get_headers(),
                    timeout=30
                )
                return response.status_code in (200, 201)
            except Exception as e:
                logger.error(f"Upload failed: {e}")
                return False
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False
    
    async def download(self, user_id: str) -> Optional[dict]:
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/settings/{user_id}",
                    headers=self._get_headers()
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
                    
        except ImportError:
            import requests
            try:
                response = requests.get(
                    f"{self.api_url}/settings/{user_id}",
                    headers=self._get_headers(),
                    timeout=30
                )
                if response.status_code == 200:
                    return response.json()
                return None
            except Exception as e:
                logger.error(f"Download failed: {e}")
                return None
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return None
    
    async def get_last_modified(self, user_id: str) -> Optional[str]:
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.head(
                    f"{self.api_url}/settings/{user_id}",
                    headers=self._get_headers()
                ) as response:
                    if response.status == 200:
                        return response.headers.get('Last-Modified')
                    return None
        except Exception:
            return None


class FirebaseBackend(CloudSyncBackend):
    """Firebase Realtime Database backend."""
    
    def __init__(self, project_id: str, api_key: Optional[str] = None):
        self.project_id = project_id
        self.api_key = api_key
        self.db_url = f"https://{project_id}.firebaseio.com"
    
    def _get_url(self, user_id: str) -> str:
        url = f"{self.db_url}/users/{user_id}/settings.json"
        if self.api_key:
            url += f"?auth={self.api_key}"
        return url
    
    async def upload(self, user_id: str, settings: dict) -> bool:
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.put(
                    self._get_url(user_id),
                    json=settings
                ) as response:
                    return response.status == 200
        except ImportError:
            import requests
            try:
                response = requests.put(
                    self._get_url(user_id),
                    json=settings,
                    timeout=30
                )
                return response.status_code == 200
            except Exception as e:
                logger.error(f"Firebase upload failed: {e}")
                return False
        except Exception as e:
            logger.error(f"Firebase upload failed: {e}")
            return False
    
    async def download(self, user_id: str) -> Optional[dict]:
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self._get_url(user_id)) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
        except ImportError:
            import requests
            try:
                response = requests.get(self._get_url(user_id), timeout=30)
                if response.status_code == 200:
                    return response.json()
                return None
            except Exception as e:
                logger.error(f"Firebase download failed: {e}")
                return None
        except Exception as e:
            logger.error(f"Firebase download failed: {e}")
            return None
    
    async def get_last_modified(self, user_id: str) -> Optional[str]:
        data = await self.download(user_id)
        if data and 'last_modified' in data:
            return data['last_modified']
        return None


class LocalBackupBackend(CloudSyncBackend):
    """Local file backup backend (for testing or offline use)."""
    
    def __init__(self, backup_dir: Optional[Path] = None):
        if backup_dir is None:
            backup_dir = Path.home() / '.enigma' / 'sync_backup'
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, user_id: str) -> Path:
        return self.backup_dir / f"{user_id}_settings.json"
    
    async def upload(self, user_id: str, settings: dict) -> bool:
        try:
            path = self._get_path(user_id)
            with open(path, 'w') as f:
                json.dump(settings, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Local backup upload failed: {e}")
            return False
    
    async def download(self, user_id: str) -> Optional[dict]:
        try:
            path = self._get_path(user_id)
            if path.exists():
                with open(path, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Local backup download failed: {e}")
            return None
    
    async def get_last_modified(self, user_id: str) -> Optional[str]:
        path = self._get_path(user_id)
        if path.exists():
            mtime = path.stat().st_mtime
            return datetime.fromtimestamp(mtime).isoformat()
        return None


class CloudSyncService:
    """
    Main cloud sync service.
    
    Handles syncing user preferences across devices with
    conflict resolution and automatic sync.
    """
    
    def __init__(
        self,
        backend: Union[str, CloudSyncBackend] = 'local',
        user_id: Optional[str] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        encryption_key: Optional[str] = None,
        auto_sync: bool = False,
        sync_interval: int = 300,
        on_settings_change: Optional[Callable[[SyncSettings], None]] = None,
    ):
        """
        Initialize cloud sync service.
        
        Args:
            backend: 'local', 'rest', 'firebase', or CloudSyncBackend instance
            user_id: User identifier (generated if not provided)
            api_url: REST API URL (for 'rest' backend)
            api_key: API key for authentication
            project_id: Firebase project ID (for 'firebase' backend)
            encryption_key: Key for encrypting sensitive data
            auto_sync: Enable automatic sync
            sync_interval: Seconds between auto-sync (default: 5 minutes)
            on_settings_change: Callback when settings change from sync
        """
        # Set up backend
        if isinstance(backend, CloudSyncBackend):
            self._backend = backend
        elif backend == 'rest':
            if not api_url:
                raise ValueError("api_url required for REST backend")
            self._backend = RestApiBackend(api_url, api_key)
        elif backend == 'firebase':
            if not project_id:
                raise ValueError("project_id required for Firebase backend")
            self._backend = FirebaseBackend(project_id, api_key)
        else:  # 'local' or default
            self._backend = LocalBackupBackend()
        
        # User ID (generate if not provided)
        self.user_id = user_id or self._generate_device_id()
        
        # Encryption
        self._encryption_key = encryption_key
        
        # Current settings
        self._local_settings: Optional[SyncSettings] = None
        self._last_sync: Optional[str] = None
        
        # Callbacks
        self.on_settings_change = on_settings_change
        
        # Auto-sync
        self._auto_sync = auto_sync
        self._sync_interval = sync_interval
        self._sync_task: Optional[asyncio.Task] = None
        
        # Load local settings
        self._load_local_settings()
    
    def _generate_device_id(self) -> str:
        """Generate a unique device ID."""
        import platform
        import uuid
        
        # Combine machine-specific info
        machine_info = f"{platform.node()}-{platform.machine()}-{uuid.getnode()}"
        return hashlib.sha256(machine_info.encode()).hexdigest()[:16]
    
    def _get_config_path(self) -> Path:
        """Get path to local config file."""
        from ..config import CONFIG
        return Path(CONFIG.get('data_dir', 'data')) / 'sync_settings.json'
    
    def _load_local_settings(self):
        """Load settings from local storage."""
        try:
            config_path = self._get_config_path()
            if config_path.exists():
                with open(config_path, 'r') as f:
                    data = json.load(f)
                self._local_settings = SyncSettings.from_dict(data)
            else:
                self._local_settings = SyncSettings(device_id=self.user_id)
        except Exception as e:
            logger.error(f"Failed to load local settings: {e}")
            self._local_settings = SyncSettings(device_id=self.user_id)
    
    def _save_local_settings(self):
        """Save settings to local storage."""
        try:
            config_path = self._get_config_path()
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            self._local_settings.last_modified = datetime.now().isoformat()
            self._local_settings.device_id = self.user_id
            
            with open(config_path, 'w') as f:
                json.dump(self._local_settings.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save local settings: {e}")
    
    @property
    def settings(self) -> SyncSettings:
        """Get current settings."""
        if self._local_settings is None:
            self._load_local_settings()
        return self._local_settings
    
    def update_setting(self, key: str, value: Any):
        """Update a single setting."""
        if hasattr(self._local_settings, key):
            setattr(self._local_settings, key, value)
        else:
            self._local_settings.custom[key] = value
        self._save_local_settings()
    
    def update_settings(self, updates: dict):
        """Update multiple settings."""
        for key, value in updates.items():
            if hasattr(self._local_settings, key):
                setattr(self._local_settings, key, value)
            else:
                self._local_settings.custom[key] = value
        self._save_local_settings()
    
    async def upload_settings(self) -> SyncResult:
        """Upload local settings to cloud."""
        try:
            self._local_settings.last_modified = datetime.now().isoformat()
            self._local_settings.device_id = self.user_id
            
            settings_dict = self._local_settings.to_dict()
            
            # Encrypt sensitive data if key provided
            if self._encryption_key:
                settings_dict = self._encrypt_sensitive(settings_dict)
            
            success = await self._backend.upload(self.user_id, settings_dict)
            
            if success:
                self._last_sync = datetime.now().isoformat()
                logger.info("Settings uploaded successfully")
                return SyncResult(
                    success=True,
                    action='upload',
                    timestamp=self._last_sync
                )
            else:
                return SyncResult(
                    success=False,
                    action='upload',
                    timestamp=datetime.now().isoformat(),
                    error="Upload failed"
                )
                
        except Exception as e:
            logger.error(f"Upload settings failed: {e}")
            return SyncResult(
                success=False,
                action='upload',
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )
    
    async def download_settings(
        self,
        apply: bool = False,
        merge_strategy: str = 'newer'
    ) -> SyncResult:
        """
        Download settings from cloud.
        
        Args:
            apply: Whether to apply downloaded settings
            merge_strategy: How to handle conflicts ('newer', 'local', 'remote')
        """
        try:
            remote_data = await self._backend.download(self.user_id)
            
            if remote_data is None:
                return SyncResult(
                    success=False,
                    action='download',
                    timestamp=datetime.now().isoformat(),
                    error="No remote settings found"
                )
            
            # Decrypt if needed
            if self._encryption_key:
                remote_data = self._decrypt_sensitive(remote_data)
            
            remote_settings = SyncSettings.from_dict(remote_data)
            
            # Find changes
            changes = self._find_changes(self._local_settings, remote_settings)
            
            if apply:
                # Merge settings
                merged = self._local_settings.merge_with(remote_settings, merge_strategy)
                self._local_settings = merged
                self._save_local_settings()
                
                # Notify callback
                if self.on_settings_change and changes:
                    self.on_settings_change(self._local_settings)
            
            self._last_sync = datetime.now().isoformat()
            
            return SyncResult(
                success=True,
                action='download',
                timestamp=self._last_sync,
                changes=changes
            )
            
        except Exception as e:
            logger.error(f"Download settings failed: {e}")
            return SyncResult(
                success=False,
                action='download',
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )
    
    async def sync(self, strategy: str = 'newer') -> SyncResult:
        """
        Perform bidirectional sync.
        
        Compares local and remote timestamps and syncs accordingly.
        """
        try:
            # Check remote timestamp
            remote_time = await self._backend.get_last_modified(self.user_id)
            local_time = self._local_settings.last_modified
            
            if remote_time is None:
                # No remote settings, upload local
                return await self.upload_settings()
            
            if local_time is None or local_time == "":
                # No local timestamp, download remote
                return await self.download_settings(apply=True, merge_strategy=strategy)
            
            # Compare timestamps
            if remote_time > local_time:
                # Remote is newer, download
                return await self.download_settings(apply=True, merge_strategy=strategy)
            elif local_time > remote_time:
                # Local is newer, upload
                return await self.upload_settings()
            else:
                # Same timestamp, no action needed
                return SyncResult(
                    success=True,
                    action='none',
                    timestamp=datetime.now().isoformat()
                )
                
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            return SyncResult(
                success=False,
                action='sync',
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )
    
    def _find_changes(self, local: SyncSettings, remote: SyncSettings) -> List[str]:
        """Find differences between local and remote settings."""
        changes = []
        local_dict = local.to_dict()
        remote_dict = remote.to_dict()
        
        for key in local_dict:
            if key in ('last_modified', 'device_id', 'version'):
                continue
            if local_dict.get(key) != remote_dict.get(key):
                changes.append(key)
        
        return changes
    
    def _encrypt_sensitive(self, settings: dict) -> dict:
        """Encrypt sensitive fields."""
        # Simple XOR encryption for demonstration
        # In production, use proper encryption (Fernet, AES, etc.)
        if 'encrypted_api_keys' in settings and settings['encrypted_api_keys']:
            try:
                import base64
                key_bytes = self._encryption_key.encode()
                for api_name, api_key in settings['encrypted_api_keys'].items():
                    encrypted = bytes([
                        ord(c) ^ key_bytes[i % len(key_bytes)]
                        for i, c in enumerate(api_key)
                    ])
                    settings['encrypted_api_keys'][api_name] = base64.b64encode(encrypted).decode()
            except Exception as e:
                logger.warning(f"Encryption failed: {e}")
        return settings
    
    def _decrypt_sensitive(self, settings: dict) -> dict:
        """Decrypt sensitive fields."""
        if 'encrypted_api_keys' in settings and settings['encrypted_api_keys']:
            try:
                import base64
                key_bytes = self._encryption_key.encode()
                for api_name, encrypted in settings['encrypted_api_keys'].items():
                    encrypted_bytes = base64.b64decode(encrypted)
                    decrypted = ''.join([
                        chr(b ^ key_bytes[i % len(key_bytes)])
                        for i, b in enumerate(encrypted_bytes)
                    ])
                    settings['encrypted_api_keys'][api_name] = decrypted
            except Exception as e:
                logger.warning(f"Decryption failed: {e}")
        return settings
    
    def start_auto_sync(self, interval: Optional[int] = None):
        """Start automatic sync in background."""
        if interval:
            self._sync_interval = interval
        
        async def _auto_sync_loop():
            while True:
                await asyncio.sleep(self._sync_interval)
                try:
                    await self.sync()
                except Exception as e:
                    logger.error(f"Auto-sync error: {e}")
        
        self._auto_sync = True
        self._sync_task = asyncio.create_task(_auto_sync_loop())
        logger.info(f"Auto-sync started (interval: {self._sync_interval}s)")
    
    def stop_auto_sync(self):
        """Stop automatic sync."""
        self._auto_sync = False
        if self._sync_task:
            self._sync_task.cancel()
            self._sync_task = None
        logger.info("Auto-sync stopped")
    
    async def export_settings(self, path: Path) -> bool:
        """Export settings to a file."""
        try:
            with open(path, 'w') as f:
                json.dump(self._local_settings.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    async def import_settings(self, path: Path, apply: bool = True) -> SyncResult:
        """Import settings from a file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            imported = SyncSettings.from_dict(data)
            
            if apply:
                self._local_settings = imported
                self._save_local_settings()
                
                if self.on_settings_change:
                    self.on_settings_change(self._local_settings)
            
            return SyncResult(
                success=True,
                action='import',
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return SyncResult(
                success=False,
                action='import',
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )


# Global instance for easy access
_sync_service: Optional[CloudSyncService] = None


def get_sync_service(
    backend: str = 'local',
    **kwargs
) -> CloudSyncService:
    """Get or create the global sync service instance."""
    global _sync_service
    
    if _sync_service is None:
        _sync_service = CloudSyncService(backend=backend, **kwargs)
    
    return _sync_service


def configure_sync(
    backend: str = 'local',
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
    encryption_key: Optional[str] = None,
    auto_sync: bool = False,
) -> CloudSyncService:
    """
    Configure the cloud sync service.
    
    Call this once at application startup.
    
    Example:
        from enigma_engine.sync.cloud_sync import configure_sync
        
        # Configure with Firebase
        sync = configure_sync(
            backend='firebase',
            project_id='my-project',
            api_key=os.environ.get('FIREBASE_KEY'),
            auto_sync=True
        )
        
        # Or with REST API
        sync = configure_sync(
            backend='rest',
            api_url='https://api.example.com/sync',
            api_key=os.environ.get('SYNC_API_KEY')
        )
    """
    global _sync_service
    
    _sync_service = CloudSyncService(
        backend=backend,
        api_url=api_url,
        api_key=api_key,
        project_id=project_id,
        user_id=user_id,
        encryption_key=encryption_key,
        auto_sync=auto_sync,
    )
    
    return _sync_service
