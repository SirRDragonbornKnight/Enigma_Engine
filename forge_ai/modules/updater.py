"""
ForgeAI Module Updater
====================

Module version checking and update management.

Features:
- Check for available updates
- Download and install updates
- Backup and rollback capability
- Changelog tracking
"""

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urljoin

from .manager import ModuleManager

logger = logging.getLogger(__name__)


@dataclass
class ModuleUpdate:
    """Information about an available update."""
    module_id: str
    current_version: str
    latest_version: str
    changelog: str
    download_url: Optional[str] = None
    is_breaking: bool = False


class ModuleUpdater:
    """Check for and apply module updates."""
    
    def __init__(
        self, 
        manager: ModuleManager, 
        registry_url: str = None
    ):
        """
        Initialize module updater.
        
        Args:
            manager: ModuleManager instance
            registry_url: URL of the update registry
                         (defaults to example URL - configure for production)
        """
        self.manager = manager
        self.registry_url = (
            registry_url or 
            "https://forge-modules.example.com/registry"
        )
        self.backup_dir = Path("backups/modules")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-update settings
        self._auto_update_enabled = False
        self._check_interval_hours = 24
    
    def check_updates(
        self, 
        module_id: Optional[str] = None
    ) -> List[ModuleUpdate]:
        """
        Check for available updates.
        
        Args:
            module_id: Specific module to check, or None to check all
            
        Returns:
            List of available updates
        """
        updates = []
        
        # Determine which modules to check
        if module_id:
            modules_to_check = [module_id]
        else:
            # Check all registered modules
            modules_to_check = list(self.manager.module_classes.keys())
        
        logger.info(f"Checking for updates for {len(modules_to_check)} module(s)")
        
        for mid in modules_to_check:
            if mid not in self.manager.module_classes:
                logger.warning(f"Module '{mid}' not found in registry")
                continue
            
            # Get current version
            current_info = self.manager.module_classes[mid].get_info()
            current_version = current_info.version
            
            # Check for updates from registry
            try:
                update_info = self._fetch_update_info(mid)
                
                if update_info:
                    latest_version = update_info.get('version')
                    
                    # Compare versions (simple string comparison)
                    # TODO: Use proper semantic versioning (e.g., packaging.version.parse)
                    # for production to correctly handle version ordering like:
                    # 1.0.0 < 2.0.0 < 10.0.0 (not 1.0.0 < 10.0.0 < 2.0.0)
                    if latest_version and latest_version != current_version:
                        update = ModuleUpdate(
                            module_id=mid,
                            current_version=current_version,
                            latest_version=latest_version,
                            changelog=update_info.get('changelog', 'No changelog available'),
                            download_url=update_info.get('download_url'),
                            is_breaking=update_info.get('is_breaking', False)
                        )
                        updates.append(update)
                        logger.info(
                            f"Update available for '{mid}': "
                            f"{current_version} -> {latest_version}"
                        )
            
            except Exception as e:
                logger.error(f"Error checking updates for '{mid}': {e}")
        
        return updates
    
    def _fetch_update_info(self, module_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch update information from registry.
        
        Args:
            module_id: Module to fetch info for
            
        Returns:
            Update information dict, or None if not available
            
        Note: This is a stub implementation. In production, this would
              make HTTP requests to the update registry.
        """
        # Stub implementation - would make HTTP request in production
        logger.debug(f"Fetching update info for '{module_id}' from {self.registry_url}")
        
        # For now, return None (no updates available)
        # In production, this would be:
        # import requests
        # response = requests.get(f"{self.registry_url}/modules/{module_id}")
        # return response.json()
        
        return None
    
    def get_changelog(
        self, 
        module_id: str, 
        from_version: str, 
        to_version: str
    ) -> str:
        """
        Get changelog between versions.
        
        Args:
            module_id: Module ID
            from_version: Starting version
            to_version: Target version
            
        Returns:
            Changelog text
        """
        try:
            # Stub implementation
            # In production, fetch from registry
            logger.debug(
                f"Fetching changelog for '{module_id}' "
                f"from {from_version} to {to_version}"
            )
            
            # Would make HTTP request in production
            changelog = (
                f"# Changelog for {module_id}\n\n"
                f"## {to_version}\n\n"
                f"Changes from {from_version} to {to_version}:\n"
                f"- Updates and improvements\n"
                f"- Bug fixes\n"
            )
            
            return changelog
            
        except Exception as e:
            logger.error(f"Error fetching changelog: {e}")
            return "Changelog not available"
    
    def update_module(
        self, 
        module_id: str, 
        version: str = "latest"
    ) -> bool:
        """
        Update a module to specified version.
        
        Handles:
        - Backup of current version
        - Download of new version
        - Verification
        - Installation
        - Rollback on failure
        
        Args:
            module_id: Module to update
            version: Target version ('latest' for newest)
            
        Returns:
            True if update successful
        """
        if module_id not in self.manager.module_classes:
            logger.error(f"Module '{module_id}' not found")
            return False
        
        logger.info(f"Updating module '{module_id}' to version {version}")
        
        # Get current info
        current_info = self.manager.module_classes[module_id].get_info()
        current_version = current_info.version
        
        # Check if module is currently loaded
        is_loaded = module_id in self.manager.modules
        was_active = False
        
        if is_loaded:
            module = self.manager.modules[module_id]
            was_active = module.state.value == 'active'
            
            # Unload module before updating
            logger.info(f"Unloading module '{module_id}' for update")
            if not self.manager.unload(module_id):
                logger.error(f"Failed to unload module '{module_id}'")
                return False
        
        try:
            # Step 1: Backup current version
            backup_path = self._backup_module(module_id, current_version)
            if not backup_path:
                logger.error("Backup failed")
                return False
            
            logger.info(f"Backed up to {backup_path}")
            
            # Step 2: Download new version
            # (Stub - would download from registry in production)
            logger.info(f"Downloading version {version}...")
            new_version_data = self._download_module(module_id, version)
            
            if not new_version_data:
                logger.error("Download failed")
                return False
            
            # Step 3: Verify download
            if not self._verify_module(new_version_data):
                logger.error("Verification failed")
                return False
            
            # Step 4: Install new version
            if not self._install_module(module_id, new_version_data):
                logger.error("Installation failed, rolling back")
                self._restore_backup(module_id, backup_path)
                return False
            
            logger.info(f"Successfully updated '{module_id}' to version {version}")
            
            # Step 5: Reload module if it was loaded
            if is_loaded:
                logger.info(f"Reloading module '{module_id}'")
                if not self.manager.load(module_id):
                    logger.error("Failed to reload module after update")
                    return False
                
                if was_active:
                    self.manager.activate(module_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error during update: {e}")
            
            # Attempt rollback
            logger.info("Attempting rollback...")
            self.rollback(module_id)
            
            return False
    
    def _backup_module(
        self, 
        module_id: str, 
        version: str
    ) -> Optional[Path]:
        """
        Backup current module version.
        
        Args:
            module_id: Module to backup
            version: Current version
            
        Returns:
            Path to backup, or None if failed
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{module_id}_v{version}_{timestamp}"
            backup_path = self.backup_dir / backup_name
            
            # In production, this would copy module files
            # For now, just create a marker file
            backup_path.mkdir(parents=True, exist_ok=True)
            
            metadata = {
                'module_id': module_id,
                'version': version,
                'timestamp': timestamp,
            }
            
            with open(backup_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return backup_path
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return None
    
    def _download_module(
        self, 
        module_id: str, 
        version: str
    ) -> Optional[Dict[str, Any]]:
        """
        Download module from registry.
        
        Args:
            module_id: Module to download
            version: Version to download
            
        Returns:
            Module data, or None if failed
            
        Note: Stub implementation
        """
        # Stub - would download from registry in production
        logger.debug(f"Downloading '{module_id}' version {version}")
        
        # Would make HTTP request and download files
        # For now, return dummy data
        return {
            'module_id': module_id,
            'version': version,
            'data': b'<module data>'
        }
    
    def _verify_module(self, module_data: Dict[str, Any]) -> bool:
        """
        Verify downloaded module integrity.
        
        Args:
            module_data: Downloaded module data
            
        Returns:
            True if valid
        """
        # Stub - would verify checksums, signatures in production
        logger.debug("Verifying module integrity")
        
        # Basic checks
        if not module_data.get('module_id'):
            return False
        if not module_data.get('version'):
            return False
        
        return True
    
    def _install_module(
        self, 
        module_id: str, 
        module_data: Dict[str, Any]
    ) -> bool:
        """
        Install downloaded module.
        
        Args:
            module_id: Module ID
            module_data: Module data to install
            
        Returns:
            True if successful
        """
        # Stub - would copy files and update registry in production
        logger.debug(f"Installing module '{module_id}'")
        
        # Would extract files to proper location
        # Update module registry
        # Re-register with manager
        
        return True
    
    def _restore_backup(self, module_id: str, backup_path: Path) -> bool:
        """
        Restore module from backup.
        
        Args:
            module_id: Module to restore
            backup_path: Path to backup
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Restoring '{module_id}' from {backup_path}")
            
            # Stub - would copy files back in production
            
            return True
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False
    
    def rollback(self, module_id: str) -> bool:
        """
        Rollback to previous version.
        
        Args:
            module_id: Module to rollback
            
        Returns:
            True if successful
        """
        try:
            # Find most recent backup
            backups = sorted(
                [d for d in self.backup_dir.iterdir() 
                 if d.is_dir() and d.name.startswith(module_id)],
                reverse=True
            )
            
            if not backups:
                logger.error(f"No backups found for '{module_id}'")
                return False
            
            latest_backup = backups[0]
            logger.info(f"Rolling back '{module_id}' from {latest_backup}")
            
            return self._restore_backup(module_id, latest_backup)
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def set_auto_update(
        self, 
        enabled: bool, 
        check_interval_hours: int = 24
    ):
        """
        Enable/disable automatic update checking.
        
        Args:
            enabled: Whether to enable auto-update
            check_interval_hours: How often to check (default: 24 hours)
        """
        self._auto_update_enabled = enabled
        self._check_interval_hours = check_interval_hours
        
        if enabled:
            logger.info(
                f"Auto-update enabled (check interval: {check_interval_hours}h)"
            )
            # In production, would start background thread
            # self._start_auto_update_thread()
        else:
            logger.info("Auto-update disabled")
            # In production, would stop background thread
            # self._stop_auto_update_thread()
    
    def get_update_status(self) -> Dict[str, Any]:
        """
        Get current update status and settings.
        
        Returns:
            Dictionary with update information
        """
        return {
            'auto_update_enabled': self._auto_update_enabled,
            'check_interval_hours': self._check_interval_hours,
            'registry_url': self.registry_url,
            'backup_dir': str(self.backup_dir),
            'available_backups': len(list(self.backup_dir.iterdir()))
        }
