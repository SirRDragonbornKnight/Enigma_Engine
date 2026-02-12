"""
Image Paste Support

Allows pasting images directly into chat from clipboard.
Supports PNG, JPEG, and other common formats.

FILE: enigma_engine/gui/widgets/image_paste.py
TYPE: GUI Widget
MAIN CLASSES: ImagePasteHandler, PastedImage, ImagePreview
"""

import base64
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

try:
    from PyQt5.QtCore import QBuffer, Qt, pyqtSignal
    from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QImage, QPixmap
    from PyQt5.QtWidgets import (
        QFrame,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QScrollArea,
        QVBoxLayout,
        QWidget,
    )
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False

logger = logging.getLogger(__name__)


@dataclass
class PastedImage:
    """Represents a pasted or dropped image."""
    id: str
    data: bytes  # Raw image bytes
    format: str  # png, jpeg, etc.
    width: int
    height: int
    source: str  # "clipboard", "drop", "file"
    created_at: float = field(default_factory=time.time)
    filename: Optional[str] = None
    base64_data: Optional[str] = None
    
    def to_base64(self) -> str:
        """Convert to base64 string."""
        if self.base64_data is None:
            self.base64_data = base64.b64encode(self.data).decode('utf-8')
        return self.base64_data
    
    def to_data_uri(self) -> str:
        """Convert to data URI format."""
        mime = f"image/{self.format}"
        return f"data:{mime};base64,{self.to_base64()}"
    
    def save(self, path: Path) -> bool:
        """Save image to file."""
        try:
            with open(path, 'wb') as f:
                f.write(self.data)
            return True
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return False


class ImagePasteHandlerBase:
    """Base handler for image paste operations (non-Qt)."""
    
    SUPPORTED_FORMATS = {'png', 'jpeg', 'jpg', 'gif', 'webp', 'bmp'}
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    
    def __init__(self,
                 max_images: int = 10,
                 auto_resize: bool = True,
                 max_dimension: int = 2048):
        """
        Initialize handler.
        
        Args:
            max_images: Maximum images to hold
            auto_resize: Whether to auto-resize large images
            max_dimension: Maximum dimension for auto-resize
        """
        self._max_images = max_images
        self._auto_resize = auto_resize
        self._max_dimension = max_dimension
        self._images: dict[str, PastedImage] = {}
        self._image_counter = 0
        self._callbacks: list[Callable[[PastedImage], None]] = []
        
    def _generate_id(self) -> str:
        """Generate unique image ID."""
        self._image_counter += 1
        return f"img_{int(time.time())}_{self._image_counter}"
    
    def process_bytes(self, 
                      data: bytes, 
                      format: str = "png",
                      source: str = "bytes",
                      filename: str = None) -> Optional[PastedImage]:
        """
        Process raw image bytes.
        
        Args:
            data: Raw image bytes
            format: Image format
            source: Source of image
            filename: Optional filename
            
        Returns:
            PastedImage or None
        """
        if len(data) > self.MAX_IMAGE_SIZE:
            logger.warning(f"Image too large: {len(data)} bytes")
            return None
            
        # Get dimensions (basic)
        width, height = self._get_image_dimensions(data, format)
        
        image = PastedImage(
            id=self._generate_id(),
            data=data,
            format=format.lower().replace('jpg', 'jpeg'),
            width=width,
            height=height,
            source=source,
            filename=filename
        )
        
        # Store
        self._images[image.id] = image
        while len(self._images) > self._max_images:
            oldest = min(self._images.values(), key=lambda x: x.created_at)
            del self._images[oldest.id]
            
        # Notify
        for callback in self._callbacks:
            try:
                callback(image)
            except Exception as e:
                logger.error(f"Image callback error: {e}")
                
        return image
    
    def _get_image_dimensions(self, data: bytes, format: str) -> tuple[int, int]:
        """Get image dimensions from bytes."""
        # Simple dimension detection for common formats
        try:
            if format.lower() == 'png':
                # PNG: width at bytes 16-20, height at 20-24
                if len(data) > 24 and data[:8] == b'\x89PNG\r\n\x1a\n':
                    width = int.from_bytes(data[16:20], 'big')
                    height = int.from_bytes(data[20:24], 'big')
                    return width, height
            elif format.lower() in ('jpeg', 'jpg'):
                # JPEG: need to parse markers
                return self._parse_jpeg_dimensions(data)
        except (IndexError, ValueError, TypeError):
            pass  # Intentionally silent
        return 0, 0
    
    def _parse_jpeg_dimensions(self, data: bytes) -> tuple[int, int]:
        """Parse JPEG dimensions."""
        try:
            i = 0
            while i < len(data) - 9:
                if data[i] == 0xFF:
                    marker = data[i + 1]
                    if marker in (0xC0, 0xC1, 0xC2):  # SOF markers
                        height = int.from_bytes(data[i+5:i+7], 'big')
                        width = int.from_bytes(data[i+7:i+9], 'big')
                        return width, height
                    elif marker == 0xD9:  # EOI
                        break
                    elif marker not in (0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0x01, 0x00):
                        length = int.from_bytes(data[i+2:i+4], 'big')
                        i += length + 2
                        continue
                i += 1
        except (IndexError, ValueError):
            pass  # Intentionally silent
        return 0, 0
    
    def load_file(self, path: Path) -> Optional[PastedImage]:
        """Load image from file."""
        path = Path(path)
        if not path.exists():
            return None
            
        ext = path.suffix.lower().lstrip('.')
        if ext not in self.SUPPORTED_FORMATS:
            logger.warning(f"Unsupported format: {ext}")
            return None
            
        try:
            with open(path, 'rb') as f:
                data = f.read()
            return self.process_bytes(data, ext, "file", path.name)
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return None
    
    def get_image(self, image_id: str) -> Optional[PastedImage]:
        """Get image by ID."""
        return self._images.get(image_id)
    
    def get_all_images(self) -> list[PastedImage]:
        """Get all stored images."""
        return list(self._images.values())
    
    def remove_image(self, image_id: str) -> bool:
        """Remove an image."""
        if image_id in self._images:
            del self._images[image_id]
            return True
        return False
    
    def clear(self):
        """Clear all images."""
        self._images.clear()
        
    def on_image_added(self, callback: Callable[[PastedImage], None]):
        """Register callback for new images."""
        self._callbacks.append(callback)


if HAS_PYQT:
    class ImagePasteHandler(ImagePasteHandlerBase):
        """Qt-aware image paste handler."""
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            from PyQt5.QtWidgets import QApplication
            self._clipboard = QApplication.clipboard()
            
        def check_clipboard(self) -> Optional[PastedImage]:
            """Check clipboard for images."""
            mime = self._clipboard.mimeData()
            
            # Try image data directly
            if mime.hasImage():
                image = self._clipboard.image()
                if not image.isNull():
                    return self._process_qimage(image, "clipboard")
                    
            # Try URLs (files)
            if mime.hasUrls():
                for url in mime.urls():
                    if url.isLocalFile():
                        path = Path(url.toLocalFile())
                        if path.suffix.lower().lstrip('.') in self.SUPPORTED_FORMATS:
                            return self.load_file(path)
                            
            return None
        
        def _process_qimage(self, 
                           image: 'QImage', 
                           source: str) -> Optional[PastedImage]:
            """Process a QImage into PastedImage."""
            if image.isNull():
                return None
                
            # Convert to PNG bytes
            buffer = QBuffer()
            buffer.open(QBuffer.WriteOnly)
            image.save(buffer, "PNG")
            data = bytes(buffer.data())
            buffer.close()
            
            return self.process_bytes(
                data, 
                "png", 
                source,
                filename=None
            )


    class ImagePreviewWidget(QFrame):
        """Widget to preview a pasted image."""
        
        removed = pyqtSignal(str)  # Emits image_id when removed
        
        def __init__(self, image: PastedImage, max_preview: int = 150, parent=None):
            super().__init__(parent)
            self._image = image
            self._max_preview = max_preview
            self._setup_ui()
            
        def _setup_ui(self):
            self.setFrameStyle(QFrame.Box | QFrame.Raised)
            self.setLineWidth(1)
            
            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(2)
            
            # Image preview
            pixmap = QPixmap()
            pixmap.loadFromData(self._image.data)
            scaled = pixmap.scaled(
                self._max_preview, 
                self._max_preview,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            preview = QLabel()
            preview.setPixmap(scaled)
            preview.setAlignment(Qt.AlignCenter)
            layout.addWidget(preview)
            
            # Info label
            info = QLabel(f"{self._image.width}x{self._image.height}")
            info.setAlignment(Qt.AlignCenter)
            info.setStyleSheet("font-size: 10px; color: #888;")
            layout.addWidget(info)
            
            # Remove button
            remove_btn = QPushButton("X")
            remove_btn.setFixedSize(20, 20)
            remove_btn.setStyleSheet("""
                QPushButton {
                    background: #ff4444;
                    color: white;
                    border: none;
                    border-radius: 10px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background: #ff0000;
                }
            """)
            remove_btn.clicked.connect(lambda: self.removed.emit(self._image.id))
            
            # Position remove button
            remove_btn.setParent(self)
            remove_btn.move(self.width() - 24, 4)
            
        @property
        def image_id(self) -> str:
            return self._image.id
            

    class ImagePasteArea(QWidget):
        """Drop zone for pasting/dropping images."""
        
        images_changed = pyqtSignal(list)  # Emits list of PastedImage
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self._handler = ImagePasteHandler()
            self._previews: dict[str, ImagePreviewWidget] = {}
            self._setup_ui()
            
            self._handler.on_image_added(self._on_image_added)
            
        def _setup_ui(self):
            self.setAcceptDrops(True)
            
            layout = QVBoxLayout(self)
            layout.setContentsMargins(8, 8, 8, 8)
            
            # Scroll area for previews
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setMaximumHeight(180)
            
            # Preview container
            self._preview_container = QWidget()
            self._preview_layout = QHBoxLayout(self._preview_container)
            self._preview_layout.setContentsMargins(0, 0, 0, 0)
            self._preview_layout.setSpacing(8)
            self._preview_layout.addStretch()
            
            scroll.setWidget(self._preview_container)
            layout.addWidget(scroll)
            
            # Drop hint (shown when empty)
            self._drop_hint = QLabel("Paste or drop images here")
            self._drop_hint.setAlignment(Qt.AlignCenter)
            self._drop_hint.setStyleSheet("""
                QLabel {
                    color: #888;
                    padding: 20px;
                    border: 2px dashed #ccc;
                    border-radius: 8px;
                }
            """)
            layout.addWidget(self._drop_hint)
            
            self._update_visibility()
            
        def _update_visibility(self):
            """Update visibility of components."""
            has_images = len(self._previews) > 0
            self._drop_hint.setVisible(not has_images)
            self._preview_container.setVisible(has_images)
            
        def _on_image_added(self, image: PastedImage):
            """Handle new image."""
            preview = ImagePreviewWidget(image)
            preview.removed.connect(self._remove_image)
            
            # Insert before stretch
            self._preview_layout.insertWidget(
                self._preview_layout.count() - 1, 
                preview
            )
            self._previews[image.id] = preview
            
            self._update_visibility()
            self.images_changed.emit(self._handler.get_all_images())
            
        def _remove_image(self, image_id: str):
            """Remove an image."""
            if image_id in self._previews:
                self._previews[image_id].deleteLater()
                del self._previews[image_id]
                
            self._handler.remove_image(image_id)
            self._update_visibility()
            self.images_changed.emit(self._handler.get_all_images())
            
        def paste_from_clipboard(self) -> bool:
            """Paste image from clipboard."""
            image = self._handler.check_clipboard()
            return image is not None
            
        def dragEnterEvent(self, event: 'QDragEnterEvent'):
            """Handle drag enter."""
            if event.mimeData().hasImage() or event.mimeData().hasUrls():
                event.acceptProposedAction()
                self.setStyleSheet("background: rgba(0, 120, 200, 0.1);")
                
        def dragLeaveEvent(self, event):
            """Handle drag leave."""
            self.setStyleSheet("")
            
        def dropEvent(self, event: 'QDropEvent'):
            """Handle drop."""
            self.setStyleSheet("")
            mime = event.mimeData()
            
            if mime.hasImage():
                image = QImage(mime.imageData())
                self._handler._process_qimage(image, "drop")
            elif mime.hasUrls():
                for url in mime.urls():
                    if url.isLocalFile():
                        self._handler.load_file(Path(url.toLocalFile()))
                        
        def get_images(self) -> list[PastedImage]:
            """Get all pasted images."""
            return self._handler.get_all_images()
        
        def clear_images(self):
            """Clear all images."""
            for preview in list(self._previews.values()):
                preview.deleteLater()
            self._previews.clear()
            self._handler.clear()
            self._update_visibility()
            self.images_changed.emit([])

else:
    # Stub classes when PyQt5 is not available
    ImagePasteHandler = ImagePasteHandlerBase
    ImagePreviewWidget = None
    ImagePasteArea = None


# Utility functions
def images_to_message_format(images: list[PastedImage]) -> list[dict]:
    """
    Convert images to message attachment format.
    
    Args:
        images: List of PastedImage
        
    Returns:
        List of attachment dicts
    """
    return [{
        "type": "image",
        "id": img.id,
        "data_uri": img.to_data_uri(),
        "width": img.width,
        "height": img.height,
        "format": img.format
    } for img in images]


__all__ = [
    'PastedImage',
    'ImagePasteHandler',
    'ImagePasteHandlerBase',
    'ImagePreviewWidget',
    'ImagePasteArea',
    'images_to_message_format'
]
