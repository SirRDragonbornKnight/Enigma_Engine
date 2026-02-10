"""
Sync Module
===========

Cloud synchronization of settings and preferences across devices.

Classes:
    CloudSyncService: Main sync service
    SyncSettings: Settings data model
    SyncResult: Result of sync operations

Backends:
    RestApiBackend: Sync via REST API
    FirebaseBackend: Sync via Firebase
    LocalBackupBackend: Local file backup

Usage:
    from enigma_engine.sync import configure_sync, get_sync_service
    
    # Configure at startup
    sync = configure_sync(backend='firebase', project_id='my-project')
    
    # Sync settings
    await sync.sync()
    
    # Or get existing instance
    sync = get_sync_service()
    await sync.upload_settings()
"""

from .cloud_sync import (
    CloudSyncService,
    SyncSettings,
    SyncResult,
    CloudSyncBackend,
    RestApiBackend,
    FirebaseBackend,
    LocalBackupBackend,
    get_sync_service,
    configure_sync,
)

__all__ = [
    'CloudSyncService',
    'SyncSettings',
    'SyncResult',
    'CloudSyncBackend',
    'RestApiBackend',
    'FirebaseBackend',
    'LocalBackupBackend',
    'get_sync_service',
    'configure_sync',
]
