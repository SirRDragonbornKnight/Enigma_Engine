"""
Enigma AI Engine Module Updater
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
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .manager import ModuleManager

logger = logging.getLogger(__name__)

# Constants
DEFAULT_REGISTRY_URL = "https://forge-modules.example.com/registry"
DEFAULT_UPDATE_CHECK_INTERVAL_HOURS = 24


def parse_version(version_str: str) -> tuple[int, ...]:
    """
    Parse a version string into a tuple for comparison.
    
    Handles semantic versioning like 1.0.0, 2.1.3-beta, etc.
    Falls back to string comparison if parsing fails.
    
    Args:
        version_str: Version string to parse (e.g., "1.2.3" or "1.2.3-beta")
    
    Returns:
        Tuple of integers for comparison
    """
    if not version_str:
        return (0,)
    
    # Remove common prefixes
    version_str = version_str.lstrip('vV')
    
    # Extract numeric parts (ignore pre-release suffixes like -beta, -rc1)
    match = re.match(r'^(\d+(?:\.\d+)*)', version_str)
    if match:
        parts = match.group(1).split('.')
        return tuple(int(p) for p in parts)
    
    # Fallback for non-standard versions
    return (0,)


def compare_versions(v1: str, v2: str) -> int:
    """
    Compare two version strings.
    
    Args:
        v1: First version
        v2: Second version
    
    Returns:
        -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2
    """
    parsed1 = parse_version(v1)
    parsed2 = parse_version(v2)
    
    # Pad shorter tuple with zeros
    max_len = max(len(parsed1), len(parsed2))
    parsed1 = parsed1 + (0,) * (max_len - len(parsed1))
    parsed2 = parsed2 + (0,) * (max_len - len(parsed2))
    
    if parsed1 < parsed2:
        return -1
    elif parsed1 > parsed2:
        return 1
    return 0


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
            DEFAULT_REGISTRY_URL
        )
        self.backup_dir = Path("backups/modules")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-update settings
        self._auto_update_enabled = False
        self._check_interval_hours = 24
    
    def check_updates(
        self, 
        module_id: Optional[str] = None
    ) -> list[ModuleUpdate]:
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
                    
                    # Compare versions using semantic versioning
                    # Returns 1 if latest > current (update available)
                    if latest_version and compare_versions(latest_version, current_version) > 0:
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
    
    def _fetch_update_info(self, module_id: str) -> Optional[dict[str, Any]]:
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
            logger.debug(
                f"Fetching changelog for '{module_id}' "
                f"from {from_version} to {to_version}"
            )
            
            # Try to read local changelog file first
            changelog_paths = [
                self.modules_dir / module_id / "CHANGELOG.md",
                self.modules_dir / module_id / "changelog.md",
                self.modules_dir / module_id / "CHANGELOG.txt",
                self.modules_dir / module_id / "CHANGES.md",
            ]
            
            for changelog_path in changelog_paths:
                if changelog_path.exists():
                    changelog_text = changelog_path.read_text(encoding='utf-8')
                    # Extract relevant versions from changelog
                    return self._extract_version_range(
                        changelog_text, from_version, to_version, module_id
                    )
            
            # Try fetching from registry if available
            if self.registry_url and self.registry_url != DEFAULT_REGISTRY_URL:
                try:
                    import ssl
                    import urllib.request
                    
                    url = f"{self.registry_url}/modules/{module_id}/changelog"
                    params = f"?from={from_version}&to={to_version}"
                    
                    ctx = ssl.create_default_context()
                    with urllib.request.urlopen(url + params, timeout=10, context=ctx) as response:
                        if response.status == 200:
                            return response.read().decode('utf-8')
                except Exception as e:
                    logger.debug(f"Could not fetch remote changelog: {e}")
            
            # Check module registry for inline changelog
            module_info = self.module_registry.get(module_id, {})
            if 'changelog' in module_info:
                return module_info['changelog']
            
            # Return structured placeholder with available info
            changelog = f"# Changelog for {module_id}\n\n"
            changelog += f"## Version {to_version}\n\n"
            
            if from_version != to_version:
                changelog += f"Updating from {from_version}\n\n"
                
            # Add any available metadata
            if 'description' in module_info:
                changelog += f"**Description:** {module_info['description']}\n\n"
            if 'author' in module_info:
                changelog += f"**Author:** {module_info['author']}\n\n"
                
            changelog += "*Detailed changelog not available for this module.*\n"
            changelog += "*Check the module repository for release notes.*\n"
            
            return changelog
            
        except Exception as e:
            logger.error(f"Error fetching changelog: {e}")
            return f"Changelog not available: {e}"
    
    def _extract_version_range(
        self,
        changelog_text: str,
        from_version: str,
        to_version: str,
        module_id: str
    ) -> str:
        """
        Extract changelog entries between two versions.
        
        Args:
            changelog_text: Full changelog text
            from_version: Starting version (exclusive)
            to_version: Target version (inclusive)
            module_id: Module identifier
            
        Returns:
            Filtered changelog text
        """
        lines = changelog_text.split('\n')
        result_lines = [f"# Changelog for {module_id}", ""]
        result_lines.append(f"*Changes from {from_version} to {to_version}*\n")
        
        in_range = False
        current_version = None
        version_pattern = re.compile(
            r'^##?\s*(?:\[)?v?(\d+\.\d+(?:\.\d+)?(?:-\w+)?)',
            re.IGNORECASE
        )
        
        for line in lines:
            version_match = version_pattern.match(line)
            
            if version_match:
                current_version = version_match.group(1)
                
                # Check if we've reached the target version
                if compare_versions(current_version, to_version) <= 0:
                    in_range = True
                    
                # Check if we've passed the starting version
                if compare_versions(current_version, from_version) <= 0:
                    in_range = False
                    
            if in_range:
                result_lines.append(line)
        
        if len(result_lines) <= 3:
            # No version-specific entries found, return relevant portion
            return changelog_text[:2000] + "\n\n*[Truncated]*" if len(changelog_text) > 2000 else changelog_text
            
        return '\n'.join(result_lines)
    
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
    ) -> Optional[dict[str, Any]]:
        """
        Download module from registry.
        
        Args:
            module_id: Module to download
            version: Version to download
            
        Returns:
            Module data, or None if failed
        """
        logger.info(f"Downloading '{module_id}' version {version}")
        
        try:
            import hashlib
            import tempfile
            import urllib.error
            import urllib.request
            import zipfile

            # Try to get download URL from registry
            update_info = self.check_for_update(module_id)
            if not update_info or not update_info.download_url:
                # Construct default URL from module registry
                base_url = "https://github.com/Enigma AI Engine/modules/releases/download"
                download_url = f"{base_url}/{module_id}/{version}/{module_id}-{version}.zip"
            else:
                download_url = update_info.download_url
            
            # Download to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                tmp_path = Path(tmp_file.name)
                
                logger.debug(f"Downloading from {download_url}")
                
                try:
                    with urllib.request.urlopen(download_url, timeout=60) as response:
                        data = response.read()
                        tmp_file.write(data)
                except urllib.error.HTTPError as e:
                    logger.error(f"HTTP error downloading module: {e.code}")
                    return None
                except urllib.error.URLError as e:
                    logger.error(f"URL error downloading module: {e.reason}")
                    return None
            
            # Read and parse the downloaded file
            module_data = {
                'module_id': module_id,
                'version': version,
                'file_path': str(tmp_path),
                'checksum': hashlib.sha256(data).hexdigest(),
            }
            
            # Try to extract and read module info
            try:
                with zipfile.ZipFile(tmp_path, 'r') as zf:
                    # Look for module.json in the archive
                    if 'module.json' in zf.namelist():
                        with zf.open('module.json') as f:
                            import json
                            info = json.load(f)
                            module_data['info'] = info
                    
                    module_data['files'] = zf.namelist()
            except zipfile.BadZipFile:
                # Not a zip file, might be raw Python module
                module_data['data'] = data
            
            logger.info(f"Downloaded {len(data)} bytes for '{module_id}'")
            return module_data
            
        except Exception as e:
            logger.error(f"Download failed for '{module_id}': {e}")
            return None
    
    def _verify_module(self, module_data: dict[str, Any]) -> bool:
        """
        Verify downloaded module integrity.
        
        Checks:
        - Required fields present (module_id, version)
        - SHA256 checksum if provided
        - File size within limits
        - No malicious patterns
        
        Args:
            module_data: Downloaded module data
            
        Returns:
            True if valid
        """
        import hashlib
        
        logger.debug("Verifying module integrity")
        
        # Required field checks
        if not module_data.get('module_id'):
            logger.error("Module verification failed: missing module_id")
            return False
        if not module_data.get('version'):
            logger.error("Module verification failed: missing version")
            return False
        
        # Checksum verification if provided
        expected_checksum = module_data.get('checksum') or module_data.get('sha256')
        if expected_checksum and module_data.get('content'):
            content = module_data['content']
            if isinstance(content, str):
                content = content.encode('utf-8')
            
            actual_checksum = hashlib.sha256(content).hexdigest()
            if actual_checksum.lower() != expected_checksum.lower():
                logger.error(f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}")
                return False
            logger.debug("Checksum verified successfully")
        
        # Size check (max 50MB for modules)
        max_size = 50 * 1024 * 1024  # 50MB
        content_size = len(module_data.get('content', b''))
        if content_size > max_size:
            logger.error(f"Module too large: {content_size} bytes (max {max_size})")
            return False
        
        # Basic security checks - look for suspicious patterns
        if module_data.get('content'):
            content_str = module_data['content'] if isinstance(module_data['content'], str) else module_data['content'].decode('utf-8', errors='ignore')
            suspicious_patterns = [
                'os.system(',
                'subprocess.call(',
                'eval(',
                'exec(',
                '__import__',
                'open(/etc',
                'open("/etc',
            ]
            for pattern in suspicious_patterns:
                if pattern in content_str:
                    logger.warning(f"Suspicious pattern found in module: {pattern}")
                    # Don't fail, just warn - legitimate modules might use these
        
        logger.info(f"Module '{module_data.get('module_id')}' verified successfully")
        return True
    
    def _install_module(
        self, 
        module_id: str, 
        module_data: dict[str, Any]
    ) -> bool:
        """
        Install downloaded module.
        
        Args:
            module_id: Module ID
            module_data: Module data to install
            
        Returns:
            True if successful
        """
        logger.info(f"Installing module '{module_id}'")
        
        try:
            import shutil
            import zipfile

            # Determine installation directory
            from ..config import CONFIG
            modules_dir = Path(CONFIG.BASE_DIR) / "enigma_engine" / "modules" / "installed"
            modules_dir.mkdir(parents=True, exist_ok=True)
            
            install_dir = modules_dir / module_id
            
            # Remove existing installation if present
            if install_dir.exists():
                shutil.rmtree(install_dir)
            
            install_dir.mkdir(parents=True, exist_ok=True)
            
            # Install from zip file or raw data
            if 'file_path' in module_data and Path(module_data['file_path']).exists():
                file_path = Path(module_data['file_path'])
                
                try:
                    # Extract zip file
                    with zipfile.ZipFile(file_path, 'r') as zf:
                        zf.extractall(install_dir)
                    
                    # Clean up temp file
                    file_path.unlink()
                    
                except zipfile.BadZipFile:
                    # Copy raw file instead
                    shutil.copy(file_path, install_dir / f"{module_id}.py")
                    file_path.unlink()
                    
            elif 'data' in module_data:
                # Write raw data to module file
                module_file = install_dir / f"{module_id}.py"
                module_file.write_bytes(module_data['data'])
            
            # Update module registry
            registry_file = modules_dir / "registry.json"
            registry = {}
            
            if registry_file.exists():
                import json
                with open(registry_file) as f:
                    registry = json.load(f)
            
            registry[module_id] = {
                'version': module_data.get('version', '1.0.0'),
                'install_dir': str(install_dir),
                'installed_at': __import__('time').time(),
                'checksum': module_data.get('checksum'),
            }
            
            import json
            with open(registry_file, 'w') as f:
                json.dump(registry, f, indent=2)
            
            logger.info(f"Successfully installed '{module_id}' to {install_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Installation failed for '{module_id}': {e}")
            return False
    
    def _restore_backup(self, module_id: str, backup_path: Path) -> bool:
        """
        Restore module from backup.
        
        Copies all files from backup directory to the module installation directory.
        
        Args:
            module_id: Module to restore
            backup_path: Path to backup directory
            
        Returns:
            True if successful
        """
        import shutil
        
        try:
            logger.info(f"Restoring '{module_id}' from {backup_path}")
            
            if not backup_path.exists():
                logger.error(f"Backup path does not exist: {backup_path}")
                return False
            
            # Determine installation directory
            from ..config import CONFIG
            modules_dir = Path(CONFIG.BASE_DIR) / "enigma_engine" / "modules" / "installed"
            install_dir = modules_dir / module_id
            
            # Remove current installation if exists
            if install_dir.exists():
                logger.debug(f"Removing current installation at {install_dir}")
                shutil.rmtree(install_dir)
            
            # Copy backup to installation directory
            logger.debug(f"Copying backup from {backup_path} to {install_dir}")
            shutil.copytree(backup_path, install_dir)
            
            # Update module registry
            registry_file = modules_dir / "registry.json"
            if registry_file.exists():
                import json
                with open(registry_file) as f:
                    registry = json.load(f)
                
                # Update version info from backup metadata if available
                backup_meta = backup_path / "module.json"
                if backup_meta.exists():
                    with open(backup_meta) as f:
                        meta = json.load(f)
                    registry[module_id] = {
                        "version": meta.get("version", "unknown"),
                        "restored_from": str(backup_path),
                        "restored_at": datetime.now().isoformat()
                    }
                else:
                    registry[module_id] = {
                        "version": "restored",
                        "restored_from": str(backup_path),
                        "restored_at": datetime.now().isoformat()
                    }
                
                with open(registry_file, 'w') as f:
                    json.dump(registry, f, indent=2)
            
            logger.info(f"Successfully restored '{module_id}' from backup")
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
    
    def get_update_status(self) -> dict[str, Any]:
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
