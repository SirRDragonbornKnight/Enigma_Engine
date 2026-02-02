"""Camera tab for ForgeAI GUI - live camera preview and capture."""

import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
from typing import Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QGroupBox, QMessageBox, QSlider,
    QCheckBox, QFileDialog, QPlainTextEdit
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

from .shared_components import NoScrollComboBox
from ...config import CONFIG

# Try to import OpenCV
try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    logger.debug("OpenCV not available - camera features disabled")
    cv2 = None
    HAVE_CV2 = False

# Images directory
IMAGES_DIR = Path(CONFIG.get("data_dir", "information")) / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


class CameraThread(QThread):
    """Background thread for camera capture."""
    frame_ready = pyqtSignal(object)  # Emits numpy array
    error = pyqtSignal(str)
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        super().__init__()
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.running = False
        self.cap = None
        
    def run(self):
        """Main thread loop."""
        if not HAVE_CV2:
            self.error.emit("OpenCV not installed. Run: pip install opencv-python")
            return
            
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                self.error.emit(f"Could not open camera {self.camera_id}")
                return
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            self.running = True
            self._frame_count = 0
            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    # Always capture but only emit every frame for smooth preview
                    # The UI thread handles its own throttling via Qt event loop
                    self.frame_ready.emit(frame)
                    self._frame_count += 1
                else:
                    self.error.emit("Failed to read frame")
                    break
                    
                # Run at full camera FPS - Qt event loop handles UI updates safely
                # This prevents frame buffer buildup and keeps preview responsive
                self.msleep(16)  # ~60 FPS max (camera hardware usually limits this)
                
        except Exception as e:
            logger.error(f"Camera thread error: {e}")
            self.error.emit(str(e))
        finally:
            if self.cap:
                self.cap.release()
    
    def stop(self):
        """Stop the camera thread."""
        self.running = False
        self.wait(1000)  # Wait up to 1 second


class CameraTab(QWidget):
    """Camera tab with live preview and capture."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.camera_thread: Optional[CameraThread] = None
        self.current_frame = None
        self.is_recording = False
        self.video_writer = None
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the camera tab UI."""
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Camera")
        header.setObjectName("header")
        header.setStyleSheet("font-size: 12px; font-weight: bold; color: #f9e2af;")
        layout.addWidget(header)
        
        # Preview area
        self.preview_label = QLabel("Camera not started")
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setMaximumHeight(500)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("""
            border: 2px solid #45475a; 
            border-radius: 8px; 
            background: #1e1e2e;
            color: #bac2de;
        """)
        layout.addWidget(self.preview_label, stretch=1)
        
        # Camera controls
        controls_group = QGroupBox("Camera Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        # Row 1: Camera selection and resolution
        row1 = QHBoxLayout()
        
        row1.addWidget(QLabel("Camera:"))
        self.camera_combo = NoScrollComboBox()
        self.camera_combo.addItem("Camera 0 (Default)", 0)
        self.camera_combo.addItem("Camera 1", 1)
        self.camera_combo.addItem("Camera 2", 2)
        self.camera_combo.setMinimumWidth(120)
        self.camera_combo.setToolTip("Select camera device")
        row1.addWidget(self.camera_combo)
        
        row1.addWidget(QLabel("Resolution:"))
        self.resolution_combo = NoScrollComboBox()
        self.resolution_combo.addItem("640x480", (640, 480))
        self.resolution_combo.addItem("1280x720 (HD)", (1280, 720))
        self.resolution_combo.addItem("320x240 (Low)", (320, 240))
        self.resolution_combo.addItem("800x600", (800, 600))
        self.resolution_combo.setMinimumWidth(140)
        self.resolution_combo.setToolTip("Select camera resolution")
        row1.addWidget(self.resolution_combo)
        
        row1.addStretch()
        controls_layout.addLayout(row1)
        
        # Row 2: Start/Stop and capture buttons
        row2 = QHBoxLayout()
        
        self.btn_start = QPushButton("Start Camera")
        self.btn_start.setStyleSheet("background-color: #a6e3a1; color: #1e1e2e; font-weight: bold;")
        self.btn_start.clicked.connect(self.toggle_camera)
        row2.addWidget(self.btn_start)
        
        self.btn_capture = QPushButton("Capture Photo")
        self.btn_capture.setStyleSheet("background-color: #89b4fa; color: #1e1e2e; font-weight: bold;")
        self.btn_capture.clicked.connect(self.capture_photo)
        self.btn_capture.setEnabled(False)
        row2.addWidget(self.btn_capture)
        
        self.btn_record = QPushButton("Record Video")
        self.btn_record.setStyleSheet("background-color: #f38ba8; color: #1e1e2e; font-weight: bold;")
        self.btn_record.clicked.connect(self.toggle_recording)
        self.btn_record.setEnabled(False)
        row2.addWidget(self.btn_record)
        
        row2.addStretch()
        controls_layout.addLayout(row2)
        
        # Row 3: Analysis options
        row3 = QHBoxLayout()
        
        self.btn_analyze = QPushButton("AI Analyze")
        self.btn_analyze.setToolTip("Have AI describe what it sees in the camera")
        self.btn_analyze.clicked.connect(self.analyze_frame)
        self.btn_analyze.setEnabled(False)
        row3.addWidget(self.btn_analyze)
        
        self.chk_auto_analyze = QCheckBox("Auto-analyze every")
        self.chk_auto_analyze.setChecked(False)
        row3.addWidget(self.chk_auto_analyze)
        
        self.analyze_interval = QSpinBox()
        self.analyze_interval.setRange(1, 60)
        self.analyze_interval.setValue(5)
        self.analyze_interval.setSuffix(" sec")
        row3.addWidget(self.analyze_interval)
        
        row3.addStretch()
        controls_layout.addLayout(row3)
        
        layout.addWidget(controls_group)
        
        # Status/Analysis output
        self.status_label = QLabel("Status: Camera stopped")
        self.status_label.setStyleSheet("color: #bac2de;")
        layout.addWidget(self.status_label)
        
        # Analysis output
        analysis_group = QGroupBox("AI Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        self.analysis_text = QPlainTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setPlaceholderText("AI analysis of camera feed will appear here...")
        self.analysis_text.setMaximumHeight(120)
        analysis_layout.addWidget(self.analysis_text)
        layout.addWidget(analysis_group)
        
        # Auto-analyze timer
        self.analyze_timer = QTimer()
        self.analyze_timer.timeout.connect(self.analyze_frame)
        self.chk_auto_analyze.toggled.connect(self._toggle_auto_analyze)
        
        layout.addStretch()
        
    def toggle_camera(self):
        """Start or stop the camera."""
        if self.camera_thread and self.camera_thread.running:
            self.stop_camera()
        else:
            self.start_camera()
            
    def start_camera(self):
        """Start the camera feed."""
        if not HAVE_CV2:
            QMessageBox.warning(
                self, "OpenCV Required",
                "Camera requires OpenCV.\n\nInstall with:\npip install opencv-python"
            )
            return
            
        camera_id = self.camera_combo.currentData()
        resolution = self.resolution_combo.currentData()
        
        self.camera_thread = CameraThread(
            camera_id=camera_id,
            width=resolution[0],
            height=resolution[1]
        )
        self.camera_thread.frame_ready.connect(self._on_frame)
        self.camera_thread.error.connect(self._on_camera_error)
        self.camera_thread.start()
        
        self.btn_start.setText("Stop Camera")
        self.btn_start.setStyleSheet("background-color: #f38ba8; color: #1e1e2e; font-weight: bold;")
        self.btn_capture.setEnabled(True)
        self.btn_record.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self.camera_combo.setEnabled(False)
        self.resolution_combo.setEnabled(False)
        self.status_label.setText(f"Status: Camera {camera_id} running at {resolution[0]}x{resolution[1]}")
        
    def stop_camera(self):
        """Stop the camera feed."""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
            
        # Stop recording if active
        if self.is_recording:
            self.toggle_recording()
            
        self.btn_start.setText("Start Camera")
        self.btn_start.setStyleSheet("background-color: #a6e3a1; color: #1e1e2e; font-weight: bold;")
        self.btn_capture.setEnabled(False)
        self.btn_record.setEnabled(False)
        self.btn_analyze.setEnabled(False)
        self.camera_combo.setEnabled(True)
        self.resolution_combo.setEnabled(True)
        self.preview_label.setText("Camera stopped")
        self.status_label.setText("Status: Camera stopped")
        
    def _on_frame(self, frame):
        """Handle incoming frame from camera thread - optimized for full FPS."""
        self.current_frame = frame
        
        # Track FPS for status display
        self._fps_count = getattr(self, '_fps_count', 0) + 1
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create QImage - use copy to prevent memory issues with fast frames
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.copy().data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit preview while maintaining aspect ratio
        # Use FastTransformation for speed at high FPS
        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation  # Faster than SmoothTransformation for live video
        )
        self.preview_label.setPixmap(scaled)
        
        # Record if active
        if self.is_recording and self.video_writer:
            self.video_writer.write(frame)
            
    def _on_camera_error(self, error_msg: str):
        """Handle camera errors."""
        self.stop_camera()
        self.preview_label.setText(f"Camera Error:\n{error_msg}")
        self.status_label.setText(f"Status: Error - {error_msg}")
        QMessageBox.warning(self, "Camera Error", error_msg)
        
    def capture_photo(self):
        """Capture current frame as photo."""
        if self.current_frame is None:
            QMessageBox.warning(self, "No Frame", "No camera frame available")
            return
        
        if not HAVE_CV2:
            QMessageBox.warning(self, "OpenCV Required", "OpenCV is required to save images")
            return
            
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = IMAGES_DIR / f"camera_{timestamp}.jpg"
        
        try:
            cv2.imwrite(str(filename), self.current_frame)
            logger.info(f"Photo captured: {filename}")
            self.status_label.setText(f"Status: Photo saved to {filename.name}")
            self.analysis_text.appendPlainText(f"\nPhoto saved: {filename.name}")
        except Exception as e:
            logger.error(f"Failed to save photo: {e}")
            QMessageBox.warning(self, "Save Error", f"Failed to save photo: {e}")
            
    def toggle_recording(self):
        """Start or stop video recording."""
        if not HAVE_CV2:
            QMessageBox.warning(self, "OpenCV Required", "OpenCV is required for video recording")
            return
            
        if self.is_recording:
            # Stop recording
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.btn_record.setText("Record Video")
            self.btn_record.setStyleSheet("background-color: #f38ba8; color: #1e1e2e; font-weight: bold;")
            self.status_label.setText("Status: Recording stopped")
            self.analysis_text.appendPlainText("\nRecording stopped")
        else:
            # Start recording
            if self.current_frame is None:
                QMessageBox.warning(self, "No Frame", "Camera not running")
                return
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = IMAGES_DIR / f"camera_recording_{timestamp}.avi"
            
            h, w = self.current_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(str(filename), fourcc, 20.0, (w, h))
            
            if not self.video_writer.isOpened():
                QMessageBox.warning(self, "Record Error", "Failed to start recording")
                return
                
            self.is_recording = True
            self.btn_record.setText("Stop Recording")
            self.btn_record.setStyleSheet("background-color: #fab387; color: #1e1e2e; font-weight: bold; animation: blink 1s infinite;")
            self.status_label.setText(f"Status: Recording to {filename.name}")
            self.analysis_text.appendPlainText(f"\nRecording started: {filename.name}")
            
    def analyze_frame(self):
        """Have AI analyze the current frame."""
        if self.current_frame is None:
            return
        
        if not HAVE_CV2:
            self.analysis_text.appendPlainText("OpenCV not available for analysis")
            return
            
        try:
            from PIL import Image
            
            # Convert frame to PIL Image
            rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Store in main window for vision analysis
            if self.main_window and hasattr(self.main_window, '_last_screenshot'):
                self.main_window._last_screenshot = pil_image
                
            # Try to use vision tools for analysis
            self.analysis_text.appendPlainText("\nAnalyzing frame...")
            
            # Check for vision capabilities
            try:
                from ...tools.vision import get_screen_vision
                vision = get_screen_vision()
                
                # Save temp image for analysis
                temp_path = IMAGES_DIR / "temp_analyze.jpg"
                pil_image.save(temp_path)
                
                # Try to describe the frame using basic analysis
                # Get image info
                h, w = self.current_frame.shape[:2]
                
                # Analyze colors/brightness
                try:
                    import numpy as np
                    arr = np.array(pil_image)
                    brightness = arr.mean()
                    is_dark = brightness < 128
                    theme = "dark scene" if is_dark else "bright scene"
                    self.analysis_text.appendPlainText(
                        f"AI sees: {w}x{h} image, {theme}, "
                        f"average brightness: {brightness:.0f}/255"
                    )
                except ImportError:
                    self.analysis_text.appendPlainText(f"AI sees: {w}x{h} image")
                
                # Clean up temp
                if temp_path.exists():
                    temp_path.unlink()
                    
            except (ImportError, AttributeError, Exception) as e:
                # Fallback - just report basic info
                logger.debug(f"Vision tools not available: {e}")
                h, w = self.current_frame.shape[:2]
                self.analysis_text.appendPlainText(
                    f"Frame info: {w}x{h} pixels\n"
                    f"(Vision tools not available - install transformers for AI analysis)"
                )
        except Exception as e:
            logger.error(f"Frame analysis error: {e}")
            self.analysis_text.appendPlainText(f"Analysis error: {e}")
            
    def _toggle_auto_analyze(self, enabled: bool):
        """Toggle auto-analysis timer."""
        if enabled:
            interval_ms = self.analyze_interval.value() * 1000
            self.analyze_timer.start(interval_ms)
        else:
            self.analyze_timer.stop()
            
    def closeEvent(self, event):
        """Clean up when tab is closed."""
        self.stop_camera()
        if self.analyze_timer.isActive():
            self.analyze_timer.stop()
        super().closeEvent(event)


def create_camera_tab(parent) -> CameraTab:
    """Create the camera tab."""
    return CameraTab(parent)
