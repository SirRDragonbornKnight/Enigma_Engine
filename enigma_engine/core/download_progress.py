"""
================================================================================
Model Download Progress - Progress bars for downloading HuggingFace models.
================================================================================

Provides progress tracking for model downloads:
- CLI progress bars with tqdm/rich
- GUI progress callbacks for PyQt5
- Download speed and ETA estimation
- Resumable downloads
- Multi-file tracking

NOTE: This module integrates with enigma_engine.utils.progress for GUI callbacks.
      Use ProgressTracker for general progress, DownloadTracker for model downloads.

USAGE:
    from enigma_engine.core.download_progress import DownloadTracker, download_with_progress
    
    # Simple download with progress
    path = download_with_progress("microsoft/DialoGPT-small")
    
    # With custom callback
    def on_progress(progress):
        print(f"Downloading: {progress.percentage:.1f}%")
    
    tracker = DownloadTracker(callback=on_progress)
    path = tracker.download_model("gpt2")
    
    # GUI integration
    tracker = DownloadTracker(gui_mode=True)
    tracker.set_progress_signal(qt_signal)  # Emits progress to PyQt
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from threading import Lock
from typing import Any, Callable

# Import common progress utilities for GUI integration
from ..utils.progress import ProgressState

logger = logging.getLogger(__name__)


class DownloadState(Enum):
    """Download state."""
    
    PENDING = auto()
    DOWNLOADING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class DownloadProgress:
    """Progress information for a download."""
    
    file_name: str = ""
    file_size: int = 0
    downloaded: int = 0
    state: DownloadState = DownloadState.PENDING
    speed: float = 0.0           # bytes per second
    eta: float = 0.0             # estimated time remaining in seconds
    error: str | None = None
    
    # Multi-file tracking
    current_file: int = 0
    total_files: int = 0
    
    @property
    def percentage(self) -> float:
        """Download percentage (0-100)."""
        if self.file_size <= 0:
            return 0.0
        return min(100.0, (self.downloaded / self.file_size) * 100)
    
    @property
    def speed_str(self) -> str:
        """Human-readable download speed."""
        return format_size(self.speed) + "/s"
    
    @property
    def eta_str(self) -> str:
        """Human-readable ETA."""
        if self.eta <= 0:
            return "calculating..."
        elif self.eta < 60:
            return f"{int(self.eta)}s"
        elif self.eta < 3600:
            return f"{int(self.eta / 60)}m {int(self.eta % 60)}s"
        else:
            return f"{int(self.eta / 3600)}h {int((self.eta % 3600) / 60)}m"
    
    @property
    def size_str(self) -> str:
        """Human-readable size progress."""
        return f"{format_size(self.downloaded)} / {format_size(self.file_size)}"
    
    def to_progress_state(self) -> ProgressState:
        """
        Convert to common ProgressState for GUI integration.
        
        This allows download progress to be displayed using the same
        GUI components as other progress operations.
        """
        return ProgressState(
            task_name=f"Downloading {self.file_name}",
            total=self.file_size if self.file_size > 0 else None,
            current=self.downloaded,
            status=f"{self.speed_str} - {self.eta_str}" if self.speed > 0 else "",
            started_at=time.time() - (self.downloaded / self.speed if self.speed > 0 else 0),
            finished_at=time.time() if self.state == DownloadState.COMPLETED else None
        )


# Use format_bytes from utils - consolidated from duplicate implementations
from enigma_engine.utils import format_bytes


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string.
    
    Note: Uses format_bytes from utils (DRY - was duplicated).
    """
    return format_bytes(size_bytes)


class ProgressCallback:
    """
    Callback interface for HuggingFace Hub downloads.
    
    This hooks into huggingface_hub's download system to track progress.
    """
    
    def __init__(
        self,
        callback: Callable[[DownloadProgress], None] | None = None,
        show_cli: bool = True
    ):
        """
        Initialize progress callback.
        
        Args:
            callback: Optional function to call with progress updates
            show_cli: Show CLI progress bar
        """
        self.callback = callback
        self.show_cli = show_cli
        self._progress = DownloadProgress()
        self._lock = Lock()
        self._start_time = 0.0
        self._last_update = 0.0
        self._last_downloaded = 0
        self._pbar: Any | None = None
        self._tqdm_available = False
        self._rich_available = False
        
        # Check for progress libraries
        try:
            self._tqdm_available = True
        except ImportError:
            pass
        
        try:
            self._rich_available = True
        except ImportError:
            pass
    
    def __call__(
        self,
        chunk_size: int,
        total_size: int,
        current_size: int = 0
    ) -> None:
        """
        Called by huggingface_hub during download.
        
        Args:
            chunk_size: Size of chunk just downloaded
            total_size: Total file size
            current_size: Currently downloaded size
        """
        with self._lock:
            now = time.time()
            
            # Initialize on first call
            if self._start_time == 0:
                self._start_time = now
                self._last_update = now
                self._progress.file_size = total_size
                self._progress.state = DownloadState.DOWNLOADING
                
                if self.show_cli and self._tqdm_available:
                    import tqdm
                    self._pbar = tqdm.tqdm(
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        desc=self._progress.file_name or "Downloading"
                    )
            
            # Update downloaded amount
            self._progress.downloaded = current_size
            
            # Calculate speed (rolling average)
            time_delta = now - self._last_update
            if time_delta >= 0.5:  # Update every 500ms
                bytes_delta = current_size - self._last_downloaded
                self._progress.speed = bytes_delta / time_delta
                
                # Calculate ETA
                remaining = total_size - current_size
                if self._progress.speed > 0:
                    self._progress.eta = remaining / self._progress.speed
                
                self._last_update = now
                self._last_downloaded = current_size
            
            # Update progress bar
            if self._pbar is not None:
                self._pbar.update(chunk_size)
            
            # Call user callback
            if self.callback:
                self.callback(self._progress)
            
            # Check if complete
            if current_size >= total_size:
                self._progress.state = DownloadState.COMPLETED
                if self._pbar:
                    self._pbar.close()
    
    def set_file_name(self, name: str) -> None:
        """Set current file name being downloaded."""
        self._progress.file_name = name
    
    def set_file_count(self, current: int, total: int) -> None:
        """Set multi-file tracking."""
        self._progress.current_file = current
        self._progress.total_files = total
    
    def reset(self) -> None:
        """Reset for next file."""
        self._start_time = 0.0
        self._last_update = 0.0
        self._last_downloaded = 0
        self._progress = DownloadProgress()
        if self._pbar:
            self._pbar.close()
            self._pbar = None
    
    def get_progress(self) -> DownloadProgress:
        """Get current progress."""
        return self._progress


class DownloadTracker:
    """
    High-level download tracker for models.
    
    Integrates with HuggingFace Hub to track download progress.
    """
    
    def __init__(
        self,
        callback: Callable[[DownloadProgress], None] | None = None,
        show_cli: bool = True,
        gui_mode: bool = False,
        cache_dir: Path | None = None
    ):
        """
        Initialize download tracker.
        
        Args:
            callback: Function to call with progress updates
            show_cli: Show CLI progress bars
            gui_mode: Enable GUI-friendly mode (no CLI output)
            cache_dir: Custom cache directory for downloads
        """
        self.callback = callback
        self.show_cli = show_cli and not gui_mode
        self.gui_mode = gui_mode
        self.cache_dir = cache_dir
        
        self._progress_callback: ProgressCallback | None = None
        self._cancelled = False
        self._qt_signal: Any | None = None
    
    def set_progress_signal(self, signal: Any) -> None:
        """
        Set PyQt5 signal for GUI progress updates.
        
        Args:
            signal: pyqtSignal that accepts DownloadProgress
        """
        self._qt_signal = signal
    
    def _emit_progress(self, progress: DownloadProgress) -> None:
        """Emit progress to all listeners."""
        if self.callback:
            self.callback(progress)
        
        if self._qt_signal:
            self._qt_signal.emit(progress)
    
    def download_model(
        self,
        model_id: str,
        revision: str | None = None,
        token: str | None = None,
        resume: bool = True
    ) -> Path | None:
        """
        Download a HuggingFace model with progress tracking.
        
        Args:
            model_id: HuggingFace model ID (e.g., "gpt2")
            revision: Model revision/branch
            token: HuggingFace token for private models
            resume: Resume interrupted downloads
            
        Returns:
            Path to downloaded model, or None if failed
        """
        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.utils import (
                disable_progress_bars,
                enable_progress_bars,
            )

            # Disable default progress bars if we're handling it
            if not self.show_cli:
                disable_progress_bars()
            
            # Create progress callback
            self._progress_callback = ProgressCallback(
                callback=self._emit_progress,
                show_cli=self.show_cli
            )
            
            logger.info(f"Downloading model: {model_id}")
            
            # Set up download kwargs
            download_kwargs: dict[str, Any] = {
                "repo_id": model_id,
                "resume_download": resume,
            }
            
            if revision:
                download_kwargs["revision"] = revision
            if token:
                download_kwargs["token"] = token
            if self.cache_dir:
                download_kwargs["cache_dir"] = str(self.cache_dir)
            
            # Use snapshot_download for full model
            path = snapshot_download(**download_kwargs)
            
            # Re-enable progress bars
            if not self.show_cli:
                enable_progress_bars()
            
            logger.info(f"Model downloaded to: {path}")
            return Path(path)
            
        except ImportError:
            logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
            return None
        except Exception as e:
            logger.error(f"Download failed: {e}")
            if self._progress_callback:
                progress = self._progress_callback.get_progress()
                progress.state = DownloadState.FAILED
                progress.error = str(e)
                self._emit_progress(progress)
            return None
    
    def download_file(
        self,
        repo_id: str,
        filename: str,
        revision: str | None = None,
        token: str | None = None
    ) -> Path | None:
        """
        Download a specific file from HuggingFace.
        
        Args:
            repo_id: Repository ID
            filename: File path within repo
            revision: Revision/branch
            token: Auth token
            
        Returns:
            Path to downloaded file, or None if failed
        """
        try:
            from huggingface_hub import hf_hub_download
            
            self._progress_callback = ProgressCallback(
                callback=self._emit_progress,
                show_cli=self.show_cli
            )
            self._progress_callback.set_file_name(filename)
            
            download_kwargs: dict[str, Any] = {
                "repo_id": repo_id,
                "filename": filename,
            }
            
            if revision:
                download_kwargs["revision"] = revision
            if token:
                download_kwargs["token"] = token
            if self.cache_dir:
                download_kwargs["cache_dir"] = str(self.cache_dir)
            
            path = hf_hub_download(**download_kwargs)
            return Path(path)
            
        except Exception as e:
            logger.error(f"File download failed: {e}")
            return None
    
    def cancel(self) -> None:
        """Cancel current download."""
        self._cancelled = True
        if self._progress_callback:
            progress = self._progress_callback.get_progress()
            progress.state = DownloadState.CANCELLED
            self._emit_progress(progress)
    
    def is_model_cached(self, model_id: str ) -> bool:
        """Check if a model is already downloaded."""
        try:
            from huggingface_hub import scan_cache_dir

            # Check cache
            cache_info = scan_cache_dir(self.cache_dir)
            for repo in cache_info.repos:
                if repo.repo_id == model_id:
                    return True
            return False
            
        except ImportError:
            return False
        except Exception:
            return False
    
    def get_cache_size(self) -> int:
        """Get total size of model cache in bytes."""
        try:
            from huggingface_hub import scan_cache_dir
            
            cache_info = scan_cache_dir(self.cache_dir)
            return cache_info.size_on_disk
            
        except Exception:
            return 0
    
    def clear_cache(self, model_id: str | None = None) -> bool:
        """
        Clear download cache.
        
        Args:
            model_id: Specific model to clear, or None for all
            
        Returns:
            True if successful
        """
        try:
            from huggingface_hub import delete_revisions, scan_cache_dir
            
            cache_info = scan_cache_dir(self.cache_dir)
            
            for repo in cache_info.repos:
                if model_id is None or repo.repo_id == model_id:
                    for revision in repo.revisions:
                        delete_revisions([revision.commit_hash])
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False


def download_with_progress(
    model_id: str,
    show_progress: bool = True,
    callback: Callable[[DownloadProgress], None] | None = None
) -> Path | None:
    """
    Convenience function to download a model with progress.
    
    Args:
        model_id: HuggingFace model ID
        show_progress: Show CLI progress bar
        callback: Optional progress callback
        
    Returns:
        Path to downloaded model
    """
    tracker = DownloadTracker(
        callback=callback,
        show_cli=show_progress
    )
    return tracker.download_model(model_id)


def get_download_size(model_id: str) -> int:
    """
    Get the estimated download size for a model.
    
    Args:
        model_id: HuggingFace model ID
        
    Returns:
        Estimated size in bytes, or 0 if unknown
    """
    try:
        from huggingface_hub import HfApi
        
        api = HfApi()
        model_info = api.model_info(model_id)
        
        # Sum up all file sizes
        total_size = 0
        if hasattr(model_info, 'siblings'):
            for sibling in model_info.siblings:
                if hasattr(sibling, 'size') and sibling.size:
                    total_size += sibling.size
        
        return total_size
        
    except Exception as e:
        logger.debug(f"Could not get download size: {e}")
        return 0


def list_model_files(model_id: str) -> list[dict[str, Any]]:
    """
    List files in a model repository.
    
    Args:
        model_id: HuggingFace model ID
        
    Returns:
        List of file info dicts with 'name', 'size', 'type'
    """
    try:
        from huggingface_hub import HfApi
        
        api = HfApi()
        model_info = api.model_info(model_id)
        
        files = []
        if hasattr(model_info, 'siblings'):
            for sibling in model_info.siblings:
                files.append({
                    'name': sibling.rfilename,
                    'size': getattr(sibling, 'size', 0),
                    'type': 'file'
                })
        
        return files
        
    except Exception as e:
        logger.debug(f"Could not list model files: {e}")
        return []


# GUI Widget for download progress (optional PyQt5 integration)
def create_download_widget():
    """
    Create a PyQt5 widget for download progress.
    
    Returns:
        QWidget subclass or None if PyQt5 not available
    """
    try:
        from PyQt5.QtCore import pyqtSignal
        from PyQt5.QtWidgets import (
            QHBoxLayout,
            QLabel,
            QProgressBar,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )
        
        class DownloadWidget(QWidget):
            """Widget showing download progress."""
            
            progress_updated = pyqtSignal(object)  # DownloadProgress
            
            def __init__(self, parent=None):
                super().__init__(parent)
                self._setup_ui()
                self.progress_updated.connect(self._on_progress)
            
            def _setup_ui(self):
                layout = QVBoxLayout(self)
                
                # File name
                self.file_label = QLabel("Ready to download")
                layout.addWidget(self.file_label)
                
                # Progress bar
                self.progress_bar = QProgressBar()
                self.progress_bar.setMinimum(0)
                self.progress_bar.setMaximum(100)
                layout.addWidget(self.progress_bar)
                
                # Status line
                status_layout = QHBoxLayout()
                self.size_label = QLabel("")
                self.speed_label = QLabel("")
                self.eta_label = QLabel("")
                status_layout.addWidget(self.size_label)
                status_layout.addStretch()
                status_layout.addWidget(self.speed_label)
                status_layout.addWidget(self.eta_label)
                layout.addLayout(status_layout)
                
                # Cancel button
                self.cancel_btn = QPushButton("Cancel")
                self.cancel_btn.setEnabled(False)
                layout.addWidget(self.cancel_btn)
            
            def _on_progress(self, progress: DownloadProgress):
                self.file_label.setText(progress.file_name or "Downloading...")
                self.progress_bar.setValue(int(progress.percentage))
                self.size_label.setText(progress.size_str)
                self.speed_label.setText(progress.speed_str)
                self.eta_label.setText(f"ETA: {progress.eta_str}")
                
                if progress.state == DownloadState.DOWNLOADING:
                    self.cancel_btn.setEnabled(True)
                else:
                    self.cancel_btn.setEnabled(False)
            
            def get_tracker(self) -> DownloadTracker:
                """Get a tracker connected to this widget."""
                tracker = DownloadTracker(gui_mode=True)
                tracker.set_progress_signal(self.progress_updated)
                return tracker
        
        return DownloadWidget
        
    except ImportError:
        return None
