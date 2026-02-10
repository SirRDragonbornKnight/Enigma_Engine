"""
Auto Update System

Self-updating mechanism for Enigma AI Engine with rollback support.
Supports GitHub releases, custom update servers, and delta updates.

FILE: enigma_engine/utils/auto_update.py
TYPE: System/Updates
MAIN CLASSES: UpdateManager, UpdateChecker, UpdateInstaller
"""

import hashlib
import logging
import platform
import shutil
import subprocess
import tarfile
import tempfile
import threading
import zipfile
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class UpdateChannel(Enum):
    """Update channels."""
    STABLE = "stable"
    BETA = "beta"
    NIGHTLY = "nightly"
    DEV = "dev"


class UpdateState(Enum):
    """Update state."""
    IDLE = "idle"
    CHECKING = "checking"
    DOWNLOADING = "downloading"
    VERIFYING = "verifying"
    INSTALLING = "installing"
    COMPLETE = "complete"
    FAILED = "failed"
    ROLLBACK = "rollback"


@dataclass
class Version:
    """Semantic version."""
    major: int
    minor: int
    patch: int
    prerelease: str = ""
    
    @classmethod
    def parse(cls, version_str: str) -> "Version":
        """Parse version string."""
        version_str = version_str.lstrip("v")
        
        # Split prerelease
        if "-" in version_str:
            version_str, prerelease = version_str.split("-", 1)
        else:
            prerelease = ""
        
        parts = version_str.split(".")
        return cls(
            major=int(parts[0]) if len(parts) > 0 else 0,
            minor=int(parts[1]) if len(parts) > 1 else 0,
            patch=int(parts[2]) if len(parts) > 2 else 0,
            prerelease=prerelease
        )
    
    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        return version
    
    def __lt__(self, other: "Version") -> bool:
        if (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch):
            return True
        if (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch):
            # Prerelease versions are less than release versions
            if self.prerelease and not other.prerelease:
                return True
            if not self.prerelease and other.prerelease:
                return False
            return self.prerelease < other.prerelease
        return False


@dataclass
class UpdateInfo:
    """Information about an available update."""
    version: Version
    channel: UpdateChannel
    download_url: str
    release_notes: str
    checksum: str
    size_bytes: int
    release_date: str
    required: bool = False
    delta_url: Optional[str] = None
    delta_base_version: Optional[str] = None


@dataclass
class UpdateConfig:
    """Update configuration."""
    enabled: bool = True
    channel: UpdateChannel = UpdateChannel.STABLE
    check_interval_hours: int = 24
    auto_download: bool = False
    auto_install: bool = False
    show_release_notes: bool = True
    keep_backups: int = 3
    
    # Update sources
    github_repo: str = "Enigma AI Engine/enigma_engine"
    update_server_url: str = ""
    
    # Paths
    install_dir: Path = None
    backup_dir: Path = None
    temp_dir: Path = None
    
    def __post_init__(self):
        if self.install_dir is None:
            self.install_dir = Path(__file__).parent.parent.parent
        if self.backup_dir is None:
            self.backup_dir = self.install_dir / "backups"
        if self.temp_dir is None:
            self.temp_dir = Path(tempfile.gettempdir()) / "Enigma AI Engine_updates"


class UpdateChecker:
    """Check for available updates."""
    
    def __init__(self, config: UpdateConfig):
        self.config = config
        self._last_check: Optional[datetime] = None
        self._cached_update: Optional[UpdateInfo] = None
    
    def check(self, force: bool = False) -> Optional[UpdateInfo]:
        """
        Check for updates.
        
        Args:
            force: Force check even if cached
        
        Returns:
            UpdateInfo if update available, None otherwise
        """
        if not HAS_REQUESTS:
            logger.warning("requests library required for update checking")
            return None
        
        # Use cache if recent
        if not force and self._cached_update and self._last_check:
            elapsed = (datetime.now() - self._last_check).total_seconds()
            if elapsed < 3600:  # 1 hour
                return self._cached_update
        
        try:
            # Try GitHub first
            if self.config.github_repo:
                update = self._check_github()
                if update:
                    self._cached_update = update
                    self._last_check = datetime.now()
                    return update
            
            # Try custom update server
            if self.config.update_server_url:
                update = self._check_server()
                if update:
                    self._cached_update = update
                    self._last_check = datetime.now()
                    return update
        
        except Exception as e:
            logger.error(f"Update check failed: {e}")
        
        return None
    
    def _check_github(self) -> Optional[UpdateInfo]:
        """Check GitHub releases for updates."""
        url = f"https://api.github.com/repos/{self.config.github_repo}/releases/latest"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        version = Version.parse(data.get("tag_name", "0.0.0"))
        current = self._get_current_version()
        
        if not (version > current):
            return None
        
        # Find appropriate asset
        asset = self._find_asset(data.get("assets", []))
        if not asset:
            return None
        
        return UpdateInfo(
            version=version,
            channel=UpdateChannel.STABLE,
            download_url=asset.get("browser_download_url", ""),
            release_notes=data.get("body", ""),
            checksum="",  # GitHub doesn't provide checksums
            size_bytes=asset.get("size", 0),
            release_date=data.get("published_at", "")
        )
    
    def _check_server(self) -> Optional[UpdateInfo]:
        """Check custom update server."""
        url = f"{self.config.update_server_url}/check"
        
        response = requests.get(url, params={
            "current_version": str(self._get_current_version()),
            "channel": self.config.channel.value,
            "platform": platform.system().lower(),
            "arch": platform.machine()
        }, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get("update_available"):
            return None
        
        return UpdateInfo(
            version=Version.parse(data["version"]),
            channel=UpdateChannel(data.get("channel", "stable")),
            download_url=data["download_url"],
            release_notes=data.get("release_notes", ""),
            checksum=data.get("checksum", ""),
            size_bytes=data.get("size_bytes", 0),
            release_date=data.get("release_date", ""),
            required=data.get("required", False),
            delta_url=data.get("delta_url"),
            delta_base_version=data.get("delta_base_version")
        )
    
    def _find_asset(self, assets: list[dict]) -> Optional[dict]:
        """Find appropriate asset for current platform."""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        # Map platform to asset patterns
        patterns = []
        if system == "windows":
            patterns = ["windows", "win64", "win32", ".exe", ".msi"]
        elif system == "darwin":
            patterns = ["macos", "darwin", ".dmg"]
        elif system == "linux":
            if "arm" in machine or "aarch64" in machine:
                patterns = ["linux-arm", "aarch64", ".AppImage"]
            else:
                patterns = ["linux", "x86_64", ".AppImage"]
        
        for asset in assets:
            name = asset.get("name", "").lower()
            for pattern in patterns:
                if pattern in name:
                    return asset
        
        return None
    
    def _get_current_version(self) -> Version:
        """Get current installed version."""
        try:
            version_file = self.config.install_dir / "VERSION"
            if version_file.exists():
                return Version.parse(version_file.read_text().strip())
            
            # Try __init__.py
            init_file = self.config.install_dir / "enigma_engine" / "__init__.py"
            if init_file.exists():
                content = init_file.read_text()
                for line in content.split("\n"):
                    if "__version__" in line:
                        version_str = line.split("=")[1].strip().strip("\"'")
                        return Version.parse(version_str)
        except Exception as e:
            logger.warning(f"Could not determine current version: {e}")
        
        return Version(0, 0, 0)


class UpdateInstaller:
    """Download and install updates."""
    
    def __init__(self, config: UpdateConfig):
        self.config = config
        self.state = UpdateState.IDLE
        self.progress: float = 0.0
        self._progress_callback: Optional[Callable[[UpdateState, float], None]] = None
    
    def set_progress_callback(self, callback: Callable[[UpdateState, float], None]):
        """Set progress callback."""
        self._progress_callback = callback
    
    def _update_progress(self, state: UpdateState, progress: float):
        """Update progress state."""
        self.state = state
        self.progress = progress
        if self._progress_callback:
            self._progress_callback(state, progress)
    
    def download(self, update: UpdateInfo) -> Optional[Path]:
        """
        Download update package.
        
        Returns:
            Path to downloaded file
        """
        if not HAS_REQUESTS:
            return None
        
        self._update_progress(UpdateState.DOWNLOADING, 0.0)
        
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine filename
        url = update.delta_url or update.download_url
        filename = url.split("/")[-1]
        download_path = self.config.temp_dir / filename
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            
            with open(download_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        self._update_progress(
                            UpdateState.DOWNLOADING,
                            downloaded / total_size
                        )
            
            logger.info(f"Downloaded update to {download_path}")
            return download_path
        
        except Exception as e:
            logger.error(f"Download failed: {e}")
            self._update_progress(UpdateState.FAILED, 0.0)
            return None
    
    def verify(self, path: Path, expected_checksum: str) -> bool:
        """Verify downloaded update checksum."""
        if not expected_checksum:
            return True  # No checksum to verify
        
        self._update_progress(UpdateState.VERIFYING, 0.0)
        
        try:
            sha256 = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            
            actual = sha256.hexdigest()
            verified = actual == expected_checksum
            
            if not verified:
                logger.error(f"Checksum mismatch: {actual} != {expected_checksum}")
            
            self._update_progress(UpdateState.VERIFYING, 1.0)
            return verified
        
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False
    
    def install(self, path: Path) -> bool:
        """
        Install update.
        
        Args:
            path: Path to update package
        
        Returns:
            True if successful
        """
        self._update_progress(UpdateState.INSTALLING, 0.0)
        
        try:
            # Create backup
            backup_path = self._create_backup()
            if not backup_path:
                return False
            
            self._update_progress(UpdateState.INSTALLING, 0.2)
            
            # Extract update
            extract_dir = self.config.temp_dir / "extracted"
            extract_dir.mkdir(exist_ok=True)
            
            if path.suffix == ".zip":
                with zipfile.ZipFile(path, "r") as zf:
                    zf.extractall(extract_dir)
            elif path.suffix in (".tar", ".gz", ".tgz"):
                with tarfile.open(path, "r:*") as tf:
                    tf.extractall(extract_dir)
            else:
                # Direct copy (single file update)
                shutil.copy2(path, extract_dir / path.name)
            
            self._update_progress(UpdateState.INSTALLING, 0.5)
            
            # Apply update
            for item in extract_dir.iterdir():
                dest = self.config.install_dir / item.name
                
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
            
            self._update_progress(UpdateState.INSTALLING, 0.9)
            
            # Cleanup old backups
            self._cleanup_backups()
            
            # Cleanup temp
            shutil.rmtree(self.config.temp_dir, ignore_errors=True)
            
            self._update_progress(UpdateState.COMPLETE, 1.0)
            logger.info("Update installed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Installation failed: {e}")
            self._update_progress(UpdateState.FAILED, 0.0)
            return False
    
    def rollback(self) -> bool:
        """Rollback to previous version."""
        self._update_progress(UpdateState.ROLLBACK, 0.0)
        
        try:
            # Find latest backup
            backups = sorted(self.config.backup_dir.glob("backup_*"))
            if not backups:
                logger.error("No backups available for rollback")
                return False
            
            latest_backup = backups[-1]
            
            # Restore from backup
            with tarfile.open(latest_backup, "r:gz") as tf:
                tf.extractall(self.config.install_dir)
            
            logger.info(f"Rolled back to backup: {latest_backup}")
            self._update_progress(UpdateState.COMPLETE, 1.0)
            return True
        
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            self._update_progress(UpdateState.FAILED, 0.0)
            return False
    
    def _create_backup(self) -> Optional[Path]:
        """Create backup of current installation."""
        try:
            self.config.backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.config.backup_dir / f"backup_{timestamp}.tar.gz"
            
            with tarfile.open(backup_path, "w:gz") as tf:
                # Backup key directories
                for item in ["enigma_engine", "VERSION"]:
                    item_path = self.config.install_dir / item
                    if item_path.exists():
                        tf.add(item_path, arcname=item)
            
            logger.info(f"Created backup: {backup_path}")
            return backup_path
        
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return None
    
    def _cleanup_backups(self):
        """Remove old backups beyond retention limit."""
        backups = sorted(self.config.backup_dir.glob("backup_*.tar.gz"))
        
        while len(backups) > self.config.keep_backups:
            old_backup = backups.pop(0)
            old_backup.unlink()
            logger.info(f"Removed old backup: {old_backup}")


class UpdateManager:
    """
    Main update manager.
    
    Coordinates checking, downloading, and installing updates.
    """
    
    def __init__(self, config: UpdateConfig = None):
        self.config = config or UpdateConfig()
        self.checker = UpdateChecker(self.config)
        self.installer = UpdateInstaller(self.config)
        
        self._check_thread: Optional[threading.Thread] = None
        self._update_thread: Optional[threading.Thread] = None
        self._on_update_available: Optional[Callable[[UpdateInfo], None]] = None
    
    def set_update_callback(self, callback: Callable[[UpdateInfo], None]):
        """Set callback for when update is available."""
        self._on_update_available = callback
    
    def check_now(self) -> Optional[UpdateInfo]:
        """Check for updates synchronously."""
        return self.checker.check(force=True)
    
    def check_async(self):
        """Check for updates asynchronously."""
        def check():
            update = self.checker.check()
            if update and self._on_update_available:
                self._on_update_available(update)
        
        self._check_thread = threading.Thread(target=check, daemon=True)
        self._check_thread.start()
    
    def start_periodic_check(self):
        """Start periodic update checking."""
        def periodic():
            import time
            while True:
                self.check_async()
                time.sleep(self.config.check_interval_hours * 3600)
        
        thread = threading.Thread(target=periodic, daemon=True)
        thread.start()
    
    def download_and_install(
        self,
        update: UpdateInfo,
        progress_callback: Callable[[UpdateState, float], None] = None
    ) -> bool:
        """
        Download and install update.
        
        Args:
            update: Update to install
            progress_callback: Progress callback
        
        Returns:
            True if successful
        """
        if progress_callback:
            self.installer.set_progress_callback(progress_callback)
        
        # Download
        package_path = self.installer.download(update)
        if not package_path:
            return False
        
        # Verify
        if not self.installer.verify(package_path, update.checksum):
            return False
        
        # Install
        return self.installer.install(package_path)
    
    def download_and_install_async(
        self,
        update: UpdateInfo,
        progress_callback: Callable[[UpdateState, float], None] = None,
        complete_callback: Callable[[bool], None] = None
    ):
        """Download and install update asynchronously."""
        def install():
            result = self.download_and_install(update, progress_callback)
            if complete_callback:
                complete_callback(result)
        
        self._update_thread = threading.Thread(target=install, daemon=True)
        self._update_thread.start()
    
    def rollback(self) -> bool:
        """Rollback to previous version."""
        return self.installer.rollback()
    
    def restart_application(self):
        """Restart application after update."""
        import sys
        
        python = sys.executable
        script = sys.argv[0]
        args = sys.argv[1:]
        
        # Schedule restart
        if platform.system() == "Windows":
            subprocess.Popen([python, script] + args)
        else:
            import os
            os.execv(python, [python, script] + args)
    
    def get_state(self) -> dict[str, Any]:
        """Get current update state."""
        return {
            "state": self.installer.state.value,
            "progress": self.installer.progress,
            "channel": self.config.channel.value
        }


# Global instance
_manager: Optional[UpdateManager] = None


def get_update_manager(config: UpdateConfig = None) -> UpdateManager:
    """Get or create global update manager."""
    global _manager
    if _manager is None:
        _manager = UpdateManager(config)
    return _manager
