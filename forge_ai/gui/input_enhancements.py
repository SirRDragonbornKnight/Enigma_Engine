"""
Input Enhancement Utilities - Advanced input handling for the chat interface.

Features:
- Screen capture with region selection
- File drop zone with preview
- Input suggestions/autocomplete
- Multi-file upload handling
- Voice input waveform visualization

Part of the ForgeAI GUI enhancement suite.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Tuple, Union
from pathlib import Path
from enum import Enum
import logging
import base64
import mimetypes
import io

logger = logging.getLogger(__name__)


# =============================================================================
# SCREEN CAPTURE
# =============================================================================

@dataclass
class CaptureRegion:
    """A screen region for capture."""
    x: int
    y: int
    width: int
    height: int
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)
    
    @property
    def area(self) -> int:
        return self.width * self.height


class ScreenCapture:
    """
    Screen capture with region selection.
    
    Features:
    - Full screen capture
    - Region selection
    - Window capture
    - Clipboard integration
    """
    
    def __init__(self):
        """Initialize screen capture."""
        self._last_capture: Optional[bytes] = None
        self._last_region: Optional[CaptureRegion] = None
    
    def capture_full_screen(self, monitor: int = 0) -> Optional[bytes]:
        """
        Capture full screen.
        
        Args:
            monitor: Monitor index (0 = primary)
            
        Returns:
            PNG image bytes or None
        """
        try:
            # Try mss first (cross-platform)
            from mss import mss
            with mss() as sct:
                monitors = sct.monitors
                if monitor + 1 < len(monitors):
                    mon = monitors[monitor + 1]  # 0 is "all monitors"
                else:
                    mon = monitors[1]  # Primary
                
                screenshot = sct.grab(mon)
                # Convert to PNG
                from PIL import Image
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                self._last_capture = buffer.getvalue()
                return self._last_capture
                
        except ImportError:
            logger.warning("mss not installed, trying PIL")
            try:
                from PIL import ImageGrab
                screenshot = ImageGrab.grab()
                buffer = io.BytesIO()
                screenshot.save(buffer, format="PNG")
                self._last_capture = buffer.getvalue()
                return self._last_capture
            except Exception as e:
                logger.error(f"Screen capture failed: {e}")
                return None
    
    def capture_region(self, region: CaptureRegion) -> Optional[bytes]:
        """
        Capture a screen region.
        
        Args:
            region: Region to capture
            
        Returns:
            PNG image bytes or None
        """
        try:
            from mss import mss
            with mss() as sct:
                monitor = {
                    "left": region.x,
                    "top": region.y,
                    "width": region.width,
                    "height": region.height
                }
                screenshot = sct.grab(monitor)
                from PIL import Image
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                self._last_capture = buffer.getvalue()
                self._last_region = region
                return self._last_capture
                
        except ImportError:
            try:
                from PIL import ImageGrab
                bbox = (region.x, region.y, 
                       region.x + region.width, 
                       region.y + region.height)
                screenshot = ImageGrab.grab(bbox=bbox)
                buffer = io.BytesIO()
                screenshot.save(buffer, format="PNG")
                self._last_capture = buffer.getvalue()
                return self._last_capture
            except Exception as e:
                logger.error(f"Region capture failed: {e}")
                return None
    
    def capture_to_clipboard(self) -> bool:
        """
        Capture screen and copy to clipboard.
        
        Returns:
            True if successful
        """
        image_data = self.capture_full_screen()
        if not image_data:
            return False
        
        try:
            from PIL import Image
            import io
            
            img = Image.open(io.BytesIO(image_data))
            
            # Platform-specific clipboard
            try:
                import win32clipboard
                from io import BytesIO
                
                output = BytesIO()
                img.convert("RGB").save(output, "BMP")
                data = output.getvalue()[14:]  # Remove BMP header
                
                win32clipboard.OpenClipboard()
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
                win32clipboard.CloseClipboard()
                return True
            except ImportError:
                # Fallback: save to temp file
                logger.warning("win32clipboard not available")
                return False
                
        except Exception as e:
            logger.error(f"Clipboard copy failed: {e}")
            return False
    
    def get_last_capture(self) -> Optional[bytes]:
        """Get last captured image."""
        return self._last_capture
    
    def get_monitors(self) -> List[Dict[str, int]]:
        """Get available monitors."""
        try:
            from mss import mss
            with mss() as sct:
                return [
                    {"index": i, "width": m["width"], "height": m["height"],
                     "left": m["left"], "top": m["top"]}
                    for i, m in enumerate(sct.monitors[1:])  # Skip "all monitors"
                ]
        except ImportError:
            return [{"index": 0, "width": 1920, "height": 1080, "left": 0, "top": 0}]


# =============================================================================
# FILE DROP HANDLING
# =============================================================================

class FileType(Enum):
    """File type categories."""
    IMAGE = "image"
    DOCUMENT = "document"
    CODE = "code"
    AUDIO = "audio"
    VIDEO = "video"
    ARCHIVE = "archive"
    DATA = "data"
    OTHER = "other"


@dataclass
class DroppedFile:
    """A file dropped into the input area."""
    path: Path
    name: str
    size: int
    mime_type: str
    file_type: FileType
    preview: Optional[bytes] = None
    content: Optional[str] = None
    
    def size_formatted(self) -> str:
        """Get human-readable size."""
        if self.size < 1024:
            return f"{self.size} B"
        elif self.size < 1024 * 1024:
            return f"{self.size / 1024:.1f} KB"
        elif self.size < 1024 * 1024 * 1024:
            return f"{self.size / (1024 * 1024):.1f} MB"
        return f"{self.size / (1024 * 1024 * 1024):.1f} GB"
    
    def get_base64(self) -> Optional[str]:
        """Get file content as base64."""
        try:
            return base64.b64encode(self.path.read_bytes()).decode()
        except Exception:
            return None


class FileDropHandler:
    """
    Handle files dropped into the input area.
    
    Features:
    - File type detection
    - Preview generation
    - Content extraction
    - Size limits
    """
    
    # File type detection
    TYPE_MAPPING = {
        # Images
        ".jpg": FileType.IMAGE, ".jpeg": FileType.IMAGE, ".png": FileType.IMAGE,
        ".gif": FileType.IMAGE, ".webp": FileType.IMAGE, ".bmp": FileType.IMAGE,
        ".svg": FileType.IMAGE, ".ico": FileType.IMAGE,
        # Documents
        ".pdf": FileType.DOCUMENT, ".doc": FileType.DOCUMENT, ".docx": FileType.DOCUMENT,
        ".txt": FileType.DOCUMENT, ".rtf": FileType.DOCUMENT, ".odt": FileType.DOCUMENT,
        ".md": FileType.DOCUMENT, ".markdown": FileType.DOCUMENT,
        # Code
        ".py": FileType.CODE, ".js": FileType.CODE, ".ts": FileType.CODE,
        ".java": FileType.CODE, ".cpp": FileType.CODE, ".c": FileType.CODE,
        ".h": FileType.CODE, ".hpp": FileType.CODE, ".cs": FileType.CODE,
        ".rb": FileType.CODE, ".go": FileType.CODE, ".rs": FileType.CODE,
        ".php": FileType.CODE, ".html": FileType.CODE, ".css": FileType.CODE,
        ".json": FileType.CODE, ".xml": FileType.CODE, ".yaml": FileType.CODE,
        ".yml": FileType.CODE, ".sql": FileType.CODE, ".sh": FileType.CODE,
        # Audio
        ".mp3": FileType.AUDIO, ".wav": FileType.AUDIO, ".ogg": FileType.AUDIO,
        ".flac": FileType.AUDIO, ".m4a": FileType.AUDIO, ".aac": FileType.AUDIO,
        # Video
        ".mp4": FileType.VIDEO, ".avi": FileType.VIDEO, ".mkv": FileType.VIDEO,
        ".mov": FileType.VIDEO, ".webm": FileType.VIDEO, ".flv": FileType.VIDEO,
        # Archives
        ".zip": FileType.ARCHIVE, ".tar": FileType.ARCHIVE, ".gz": FileType.ARCHIVE,
        ".rar": FileType.ARCHIVE, ".7z": FileType.ARCHIVE,
        # Data
        ".csv": FileType.DATA, ".xlsx": FileType.DATA, ".xls": FileType.DATA,
        ".sqlite": FileType.DATA, ".db": FileType.DATA,
    }
    
    def __init__(
        self,
        max_file_size: int = 50 * 1024 * 1024,  # 50MB
        max_files: int = 10,
        allowed_types: Optional[List[FileType]] = None
    ):
        """
        Initialize file drop handler.
        
        Args:
            max_file_size: Maximum file size in bytes
            max_files: Maximum files per drop
            allowed_types: Allowed file types (None = all)
        """
        self.max_file_size = max_file_size
        self.max_files = max_files
        self.allowed_types = allowed_types
        self._files: List[DroppedFile] = []
        self._callbacks: List[Callable[[List[DroppedFile]], None]] = []
    
    def handle_drop(self, paths: List[str]) -> List[DroppedFile]:
        """
        Handle dropped files.
        
        Args:
            paths: List of file paths
            
        Returns:
            List of processed files
        """
        files = []
        
        for path_str in paths[:self.max_files]:
            path = Path(path_str)
            if not path.exists():
                continue
            
            # Check size
            size = path.stat().st_size
            if size > self.max_file_size:
                logger.warning(f"File too large: {path.name} ({size} bytes)")
                continue
            
            # Detect type
            file_type = self._detect_type(path)
            if self.allowed_types and file_type not in self.allowed_types:
                logger.warning(f"File type not allowed: {path.name}")
                continue
            
            # Get mime type
            mime_type, _ = mimetypes.guess_type(str(path))
            mime_type = mime_type or "application/octet-stream"
            
            file = DroppedFile(
                path=path,
                name=path.name,
                size=size,
                mime_type=mime_type,
                file_type=file_type
            )
            
            # Generate preview for images
            if file_type == FileType.IMAGE:
                file.preview = self._generate_preview(path)
            
            # Extract content for text files
            if file_type in [FileType.CODE, FileType.DOCUMENT]:
                file.content = self._extract_content(path)
            
            files.append(file)
        
        self._files.extend(files)
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(files)
            except Exception as e:
                logger.warning(f"Drop callback error: {e}")
        
        return files
    
    def get_files(self) -> List[DroppedFile]:
        """Get all dropped files."""
        return self._files.copy()
    
    def clear(self):
        """Clear dropped files."""
        self._files = []
    
    def remove(self, index: int):
        """Remove a file by index."""
        if 0 <= index < len(self._files):
            del self._files[index]
    
    def on_drop(self, callback: Callable[[List[DroppedFile]], None]):
        """Register drop callback."""
        self._callbacks.append(callback)
    
    def _detect_type(self, path: Path) -> FileType:
        """Detect file type from extension."""
        ext = path.suffix.lower()
        return self.TYPE_MAPPING.get(ext, FileType.OTHER)
    
    def _generate_preview(self, path: Path, max_size: int = 200) -> Optional[bytes]:
        """Generate image preview thumbnail."""
        try:
            from PIL import Image
            import io
            
            img = Image.open(path)
            img.thumbnail((max_size, max_size))
            
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return buffer.getvalue()
        except Exception:
            return None
    
    def _extract_content(self, path: Path, max_chars: int = 10000) -> Optional[str]:
        """Extract text content from file."""
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            return content[:max_chars]
        except Exception:
            return None


# =============================================================================
# INPUT SUGGESTIONS
# =============================================================================

@dataclass
class Suggestion:
    """An input suggestion."""
    text: str
    description: str = ""
    category: str = "general"
    score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class InputSuggestions:
    """
    Input autocomplete and suggestions.
    
    Features:
    - Command suggestions (slash commands)
    - History-based suggestions
    - Context-aware suggestions
    - Custom suggestion providers
    """
    
    # Built-in slash commands
    SLASH_COMMANDS = [
        Suggestion("/help", "Show available commands", "command"),
        Suggestion("/clear", "Clear conversation", "command"),
        Suggestion("/new", "Start new conversation", "command"),
        Suggestion("/image", "Generate an image", "command"),
        Suggestion("/code", "Generate code", "command"),
        Suggestion("/audio", "Generate audio/speech", "command"),
        Suggestion("/video", "Generate video", "command"),
        Suggestion("/3d", "Generate 3D model", "command"),
        Suggestion("/settings", "Open settings", "command"),
        Suggestion("/export", "Export conversation", "command"),
        Suggestion("/save", "Save conversation", "command"),
        Suggestion("/model", "Switch model", "command"),
        Suggestion("/system", "Set system prompt", "command"),
        Suggestion("/persona", "Switch persona", "command"),
    ]
    
    def __init__(
        self,
        max_suggestions: int = 10,
        history_weight: float = 0.5
    ):
        """
        Initialize input suggestions.
        
        Args:
            max_suggestions: Maximum suggestions to show
            history_weight: Weight for history-based scoring
        """
        self.max_suggestions = max_suggestions
        self.history_weight = history_weight
        
        self._history: List[str] = []
        self._custom_suggestions: List[Suggestion] = []
        self._providers: List[Callable[[str], List[Suggestion]]] = []
    
    def get_suggestions(self, input_text: str) -> List[Suggestion]:
        """
        Get suggestions for input text.
        
        Args:
            input_text: Current input text
            
        Returns:
            List of suggestions
        """
        if not input_text:
            return []
        
        suggestions = []
        input_lower = input_text.lower()
        
        # Slash command suggestions
        if input_text.startswith("/"):
            for cmd in self.SLASH_COMMANDS:
                if cmd.text.startswith(input_lower):
                    suggestions.append(cmd)
        
        # History-based suggestions
        for hist in self._history:
            if hist.lower().startswith(input_lower):
                suggestions.append(Suggestion(
                    text=hist,
                    description="From history",
                    category="history",
                    score=self.history_weight
                ))
        
        # Custom suggestions
        for sugg in self._custom_suggestions:
            if sugg.text.lower().startswith(input_lower):
                suggestions.append(sugg)
        
        # Provider suggestions
        for provider in self._providers:
            try:
                provider_suggestions = provider(input_text)
                suggestions.extend(provider_suggestions)
            except Exception as e:
                logger.warning(f"Suggestion provider error: {e}")
        
        # Sort by score and deduplicate
        suggestions.sort(key=lambda s: s.score, reverse=True)
        seen = set()
        unique = []
        for s in suggestions:
            if s.text not in seen:
                seen.add(s.text)
                unique.append(s)
        
        return unique[:self.max_suggestions]
    
    def add_to_history(self, text: str):
        """Add text to history."""
        # Remove if exists
        self._history = [h for h in self._history if h != text]
        # Add to front
        self._history.insert(0, text)
        # Limit size
        self._history = self._history[:100]
    
    def add_custom(self, suggestion: Suggestion):
        """Add custom suggestion."""
        self._custom_suggestions.append(suggestion)
    
    def add_provider(self, provider: Callable[[str], List[Suggestion]]):
        """Add suggestion provider function."""
        self._providers.append(provider)
    
    def clear_history(self):
        """Clear input history."""
        self._history = []


# =============================================================================
# MULTI-FILE UPLOAD
# =============================================================================

class UploadStatus(Enum):
    """Upload status."""
    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class UploadItem:
    """An item being uploaded."""
    id: str
    file: DroppedFile
    status: UploadStatus = UploadStatus.PENDING
    progress: float = 0.0
    error: Optional[str] = None
    result: Optional[Any] = None


class MultiFileUploader:
    """
    Handle multi-file uploads with progress.
    
    Features:
    - Concurrent uploads
    - Progress tracking
    - Error handling
    - Upload queue
    """
    
    def __init__(self, max_concurrent: int = 3):
        """
        Initialize uploader.
        
        Args:
            max_concurrent: Max concurrent uploads
        """
        self.max_concurrent = max_concurrent
        self._queue: List[UploadItem] = []
        self._active: List[UploadItem] = []
        self._completed: List[UploadItem] = []
        self._lock = threading.Lock()
        self._callbacks: Dict[str, List[Callable[[UploadItem], None]]] = {
            "progress": [],
            "complete": [],
            "error": []
        }
        self._id_counter = 0
    
    def add_files(self, files: List[DroppedFile]) -> List[str]:
        """
        Add files to upload queue.
        
        Args:
            files: Files to upload
            
        Returns:
            List of upload IDs
        """
        ids = []
        with self._lock:
            for file in files:
                self._id_counter += 1
                upload_id = f"upload_{self._id_counter}"
                item = UploadItem(id=upload_id, file=file)
                self._queue.append(item)
                ids.append(upload_id)
        return ids
    
    def start(self, processor: Callable[[DroppedFile], Any]):
        """
        Start processing uploads.
        
        Args:
            processor: Function to process each file
        """
        def process_queue():
            while self._queue or self._active:
                # Start new uploads
                with self._lock:
                    while len(self._active) < self.max_concurrent and self._queue:
                        item = self._queue.pop(0)
                        item.status = UploadStatus.UPLOADING
                        self._active.append(item)
                        threading.Thread(
                            target=self._process_item,
                            args=(item, processor)
                        ).start()
                
                time.sleep(0.1)
        
        threading.Thread(target=process_queue, daemon=True).start()
    
    def _process_item(self, item: UploadItem, processor: Callable):
        """Process a single upload item."""
        try:
            item.status = UploadStatus.PROCESSING
            self._notify("progress", item)
            
            # Simulate progress
            for i in range(10):
                item.progress = (i + 1) / 10
                self._notify("progress", item)
                time.sleep(0.05)
            
            # Process
            result = processor(item.file)
            
            item.status = UploadStatus.COMPLETE
            item.result = result
            item.progress = 1.0
            self._notify("complete", item)
            
        except Exception as e:
            item.status = UploadStatus.ERROR
            item.error = str(e)
            self._notify("error", item)
        
        finally:
            with self._lock:
                self._active.remove(item)
                self._completed.append(item)
    
    def get_status(self) -> Dict[str, List[UploadItem]]:
        """Get upload status."""
        return {
            "pending": self._queue.copy(),
            "active": self._active.copy(),
            "completed": self._completed.copy()
        }
    
    def on(self, event: str, callback: Callable[[UploadItem], None]):
        """Register event callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _notify(self, event: str, item: UploadItem):
        """Notify callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(item)
            except Exception as e:
                logger.warning(f"Upload callback error: {e}")


# =============================================================================
# VOICE INPUT VISUALIZATION
# =============================================================================

@dataclass 
class AudioLevel:
    """Audio level reading."""
    level: float  # 0-1 normalized
    timestamp: float
    is_speech: bool = False


class VoiceInputVisualizer:
    """
    Voice input waveform visualization.
    
    Features:
    - Real-time level monitoring
    - Voice activity indication
    - Waveform data for display
    """
    
    def __init__(self, buffer_size: int = 100):
        """
        Initialize visualizer.
        
        Args:
            buffer_size: Number of levels to keep
        """
        self.buffer_size = buffer_size
        self._levels: List[AudioLevel] = []
        self._is_recording = False
        self._callbacks: List[Callable[[AudioLevel], None]] = []
    
    def add_level(self, level: float, is_speech: bool = False):
        """Add audio level reading."""
        reading = AudioLevel(
            level=max(0.0, min(1.0, level)),
            timestamp=time.time(),
            is_speech=is_speech
        )
        
        self._levels.append(reading)
        self._levels = self._levels[-self.buffer_size:]
        
        for callback in self._callbacks:
            try:
                callback(reading)
            except Exception as e:
                logger.warning(f"Visualizer callback error: {e}")
    
    def get_levels(self) -> List[AudioLevel]:
        """Get recent audio levels."""
        return self._levels.copy()
    
    def get_waveform_data(self) -> List[float]:
        """Get waveform data for display."""
        return [l.level for l in self._levels]
    
    def clear(self):
        """Clear level buffer."""
        self._levels = []
    
    def set_recording(self, recording: bool):
        """Set recording state."""
        self._is_recording = recording
        if not recording:
            self.clear()
    
    def is_recording(self) -> bool:
        """Check if recording."""
        return self._is_recording
    
    def on_level(self, callback: Callable[[AudioLevel], None]):
        """Register level callback."""
        self._callbacks.append(callback)


# =============================================================================
# COMBINED INPUT ENHANCEMENTS
# =============================================================================

class InputEnhancements:
    """
    Combined input enhancement manager.
    
    Integrates:
    - Screen capture
    - File drop handling
    - Input suggestions
    - Multi-file upload
    - Voice visualization
    """
    
    def __init__(
        self,
        max_file_size: int = 50 * 1024 * 1024,
        max_suggestions: int = 10
    ):
        """
        Initialize input enhancements.
        
        Args:
            max_file_size: Max file size for drops
            max_suggestions: Max input suggestions
        """
        self.screen_capture = ScreenCapture()
        self.file_handler = FileDropHandler(max_file_size=max_file_size)
        self.suggestions = InputSuggestions(max_suggestions=max_suggestions)
        self.uploader = MultiFileUploader()
        self.voice_viz = VoiceInputVisualizer()


# Singleton
_input_enhancements: Optional[InputEnhancements] = None


def get_input_enhancements() -> InputEnhancements:
    """Get or create input enhancements."""
    global _input_enhancements
    if _input_enhancements is None:
        _input_enhancements = InputEnhancements()
    return _input_enhancements
