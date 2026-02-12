"""
Camera Module Support

Raspberry Pi Camera Module and USB webcam support.
Integrates with Enigma AI Engine vision for image analysis.

FILE: enigma_engine/edge/camera_module.py
TYPE: Edge
MAIN CLASSES: CameraModule, PiCamera2Wrapper, VideoStream
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for camera libraries
try:
    from picamera2 import Picamera2
    from picamera2.encoders import H264Encoder
    from picamera2.outputs import FileOutput
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False
    Picamera2 = None

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


class CameraType(Enum):
    """Camera types."""
    PI_CAMERA = auto()      # Raspberry Pi Camera Module
    PI_CAMERA_HQ = auto()   # Pi Camera HQ
    USB_WEBCAM = auto()     # USB webcam
    CSI_CAMERA = auto()     # Generic CSI camera
    AUTO = auto()           # Auto-detect


class ImageFormat(Enum):
    """Image formats."""
    JPEG = "jpeg"
    PNG = "png"
    RGB = "rgb"
    BGR = "bgr"
    YUV = "yuv"


@dataclass
class CameraConfig:
    """Camera configuration."""
    # Resolution
    width: int = 1280
    height: int = 720
    
    # Capture settings
    framerate: int = 30
    rotation: int = 0  # 0, 90, 180, 270
    hflip: bool = False
    vflip: bool = False
    
    # Image quality
    jpeg_quality: int = 85
    
    # Exposure/Color
    exposure_mode: str = "auto"
    awb_mode: str = "auto"      # Auto white balance
    iso: int = 0                # 0 = auto
    brightness: float = 0.5     # 0.0 - 1.0
    contrast: float = 1.0       # 0.0 - 2.0
    saturation: float = 1.0     # 0.0 - 2.0
    sharpness: float = 1.0      # 0.0 - 2.0
    
    # Focus (for autofocus cameras)
    autofocus: bool = True
    focus_distance: float = 0.0  # 0 = infinity


@dataclass
class CaptureResult:
    """Result from image capture."""
    success: bool
    image: Optional[Any] = None  # numpy array or bytes
    width: int = 0
    height: int = 0
    format: ImageFormat = ImageFormat.JPEG
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class CameraModule:
    """Main camera interface."""
    
    def __init__(self, config: CameraConfig = None, camera_type: CameraType = CameraType.AUTO):
        self.config = config or CameraConfig()
        self.camera_type = camera_type
        self._camera = None
        self._initialized = False
        self._streaming = False
        self._stream_thread: Optional[threading.Thread] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=10)
        self._frame_callbacks: list[Callable[[CaptureResult], None]] = []
        
        # Detect and initialize camera
        self._detect_camera()
    
    def _detect_camera(self):
        """Detect available camera."""
        if self.camera_type == CameraType.AUTO:
            # Try Pi Camera first
            if HAS_PICAMERA2:
                try:
                    test_cam = Picamera2()
                    test_cam.close()
                    self.camera_type = CameraType.PI_CAMERA
                    logger.info("Detected Pi Camera")
                except (RuntimeError, Exception):
                    pass  # Intentionally silent
            
            # Fall back to USB webcam
            if self.camera_type == CameraType.AUTO and HAS_OPENCV:
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    self.camera_type = CameraType.USB_WEBCAM
                    logger.info("Detected USB webcam")
                cap.release()
        
        if self.camera_type == CameraType.AUTO:
            logger.warning("No camera detected")
    
    def initialize(self) -> bool:
        """Initialize camera."""
        if self._initialized:
            return True
        
        try:
            if self.camera_type == CameraType.PI_CAMERA and HAS_PICAMERA2:
                self._camera = Picamera2()
                
                # Configure camera
                preview_config = self._camera.create_preview_configuration(
                    main={"size": (self.config.width, self.config.height), "format": "RGB888"},
                    lores={"size": (640, 480), "format": "YUV420"},
                    transform=self._get_transform()
                )
                self._camera.configure(preview_config)
                
                # Apply settings
                self._apply_camera_settings()
                
                self._camera.start()
                self._initialized = True
                logger.info("Pi Camera initialized")
                
            elif self.camera_type == CameraType.USB_WEBCAM and HAS_OPENCV:
                self._camera = cv2.VideoCapture(0)
                self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
                self._camera.set(cv2.CAP_PROP_FPS, self.config.framerate)
                
                if self._camera.isOpened():
                    self._initialized = True
                    logger.info("USB webcam initialized")
                else:
                    raise RuntimeError("Failed to open webcam")
            else:
                logger.error("No supported camera available")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    def _get_transform(self):
        """Get libcamera transform from config."""
        if not HAS_PICAMERA2:
            return None
        
        from libcamera import Transform
        
        transform = Transform()
        if self.config.hflip:
            transform = transform.hflip()
        if self.config.vflip:
            transform = transform.vflip()
        if self.config.rotation == 90:
            transform = transform.compose(Transform(rotation=90))
        elif self.config.rotation == 180:
            transform = transform.hflip().vflip()
        elif self.config.rotation == 270:
            transform = transform.compose(Transform(rotation=270))
        
        return transform
    
    def _apply_camera_settings(self):
        """Apply camera settings (Pi Camera only)."""
        if not HAS_PICAMERA2 or self._camera is None:
            return
        
        controls = {
            "Brightness": self.config.brightness,
            "Contrast": self.config.contrast,
            "Saturation": self.config.saturation,
            "Sharpness": self.config.sharpness,
        }
        
        # Exposure
        if self.config.exposure_mode == "auto":
            controls["AeEnable"] = True
        else:
            controls["AeEnable"] = False
        
        # ISO
        if self.config.iso > 0:
            controls["AnalogueGain"] = self.config.iso / 100
        
        # Autofocus
        if self.config.autofocus:
            controls["AfMode"] = 2  # Continuous
        else:
            controls["AfMode"] = 0  # Manual
            controls["LensPosition"] = self.config.focus_distance
        
        try:
            self._camera.set_controls(controls)
        except Exception as e:
            logger.warning(f"Failed to apply some camera controls: {e}")
    
    def capture(self, format: ImageFormat = ImageFormat.JPEG) -> CaptureResult:
        """
        Capture single image.
        
        Args:
            format: Output image format
            
        Returns:
            CaptureResult with image data
        """
        if not self._initialized:
            if not self.initialize():
                return CaptureResult(success=False, error="Camera not initialized")
        
        try:
            timestamp = time.time()
            
            if self.camera_type == CameraType.PI_CAMERA and HAS_PICAMERA2:
                # Capture with Pi Camera
                array = self._camera.capture_array("main")
                
                metadata = {}
                try:
                    metadata = self._camera.capture_metadata()
                except (RuntimeError, AttributeError):
                    pass  # Intentionally silent
                
                if format == ImageFormat.JPEG:
                    import io

                    from PIL import Image
                    img = Image.fromarray(array)
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG", quality=self.config.jpeg_quality)
                    image_data = buffer.getvalue()
                else:
                    image_data = array
                
                return CaptureResult(
                    success=True,
                    image=image_data,
                    width=self.config.width,
                    height=self.config.height,
                    format=format,
                    timestamp=timestamp,
                    metadata=metadata
                )
            
            elif self.camera_type == CameraType.USB_WEBCAM and HAS_OPENCV:
                # Capture with OpenCV
                ret, frame = self._camera.read()
                
                if not ret:
                    return CaptureResult(success=False, error="Failed to capture frame")
                
                if format == ImageFormat.JPEG:
                    _, encoded = cv2.imencode(".jpg", frame, 
                        [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality])
                    image_data = encoded.tobytes()
                elif format == ImageFormat.RGB:
                    image_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    image_data = frame
                
                return CaptureResult(
                    success=True,
                    image=image_data,
                    width=frame.shape[1],
                    height=frame.shape[0],
                    format=format,
                    timestamp=timestamp
                )
            
            return CaptureResult(success=False, error="No camera available")
            
        except Exception as e:
            logger.error(f"Capture failed: {e}")
            return CaptureResult(success=False, error=str(e))
    
    def capture_to_file(self, filepath: str, format: ImageFormat = ImageFormat.JPEG) -> bool:
        """
        Capture image directly to file.
        
        Args:
            filepath: Output file path
            format: Image format
            
        Returns:
            True if successful
        """
        result = self.capture(format)
        
        if not result.success:
            return False
        
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            if isinstance(result.image, bytes):
                path.write_bytes(result.image)
            elif HAS_OPENCV and isinstance(result.image, np.ndarray):
                cv2.imwrite(str(path), result.image)
            else:
                return False
            
            logger.info(f"Saved image to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return False
    
    def start_streaming(self, callback: Optional[Callable[[CaptureResult], None]] = None):
        """
        Start continuous video streaming.
        
        Args:
            callback: Optional callback for each frame
        """
        if self._streaming:
            return
        
        if callback:
            self._frame_callbacks.append(callback)
        
        self._streaming = True
        self._stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._stream_thread.start()
        logger.info("Camera streaming started")
    
    def stop_streaming(self):
        """Stop video streaming."""
        self._streaming = False
        if self._stream_thread:
            self._stream_thread.join(timeout=2)
        logger.info("Camera streaming stopped")
    
    def _stream_loop(self):
        """Internal streaming loop."""
        while self._streaming:
            result = self.capture(ImageFormat.RGB)
            
            if result.success:
                # Add to queue
                try:
                    self._frame_queue.put_nowait(result)
                except queue.Full:
                    # Remove old frame and add new
                    try:
                        self._frame_queue.get_nowait()
                        self._frame_queue.put_nowait(result)
                    except queue.Empty:
                        pass  # Intentionally silent
                
                # Call callbacks
                for callback in self._frame_callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error(f"Frame callback error: {e}")
            
            # Control frame rate
            time.sleep(1.0 / self.config.framerate)
    
    def get_frame(self, timeout: float = 1.0) -> Optional[CaptureResult]:
        """
        Get latest frame from stream.
        
        Args:
            timeout: Wait timeout in seconds
            
        Returns:
            Latest frame or None
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def add_frame_callback(self, callback: Callable[[CaptureResult], None]):
        """Add callback for streaming frames."""
        self._frame_callbacks.append(callback)
    
    def remove_frame_callback(self, callback: Callable[[CaptureResult], None]):
        """Remove frame callback."""
        if callback in self._frame_callbacks:
            self._frame_callbacks.remove(callback)
    
    def record_video(self, filepath: str, duration_seconds: float, 
                     codec: str = "h264", bitrate: int = 10000000) -> bool:
        """
        Record video to file.
        
        Args:
            filepath: Output file path
            duration_seconds: Recording duration
            codec: Video codec
            bitrate: Video bitrate
            
        Returns:
            True if successful
        """
        if not self._initialized:
            if not self.initialize():
                return False
        
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.camera_type == CameraType.PI_CAMERA and HAS_PICAMERA2:
                encoder = H264Encoder(bitrate=bitrate)
                output = FileOutput(str(path))
                
                self._camera.start_recording(encoder, output)
                time.sleep(duration_seconds)
                self._camera.stop_recording()
                
                logger.info(f"Recorded video to {filepath}")
                return True
            
            elif self.camera_type == CameraType.USB_WEBCAM and HAS_OPENCV:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(
                    str(path), fourcc, self.config.framerate,
                    (self.config.width, self.config.height)
                )
                
                start_time = time.time()
                while time.time() - start_time < duration_seconds:
                    ret, frame = self._camera.read()
                    if ret:
                        out.write(frame)
                
                out.release()
                logger.info(f"Recorded video to {filepath}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Video recording failed: {e}")
            return False
    
    def close(self):
        """Close camera."""
        self.stop_streaming()
        
        if self._camera:
            try:
                if self.camera_type == CameraType.PI_CAMERA and HAS_PICAMERA2:
                    self._camera.stop()
                    self._camera.close()
                elif self.camera_type == CameraType.USB_WEBCAM and HAS_OPENCV:
                    self._camera.release()
            except Exception:
                pass  # Cleanup should not raise
        
        self._camera = None
        self._initialized = False
        logger.info("Camera closed")
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, *args):
        self.close()


class MotionDetector:
    """Simple motion detection."""
    
    def __init__(self, camera: CameraModule, threshold: float = 25, min_area: int = 500):
        self.camera = camera
        self.threshold = threshold
        self.min_area = min_area
        self._previous_frame = None
        self._motion_callbacks: list[Callable[[bool, list[tuple[int, int, int, int]]], None]] = []
    
    def detect(self, frame: Any) -> tuple[bool, list[tuple[int, int, int, int]]]:
        """
        Detect motion in frame.
        
        Args:
            frame: Current frame (numpy array, BGR or RGB)
            
        Returns:
            Tuple of (motion_detected, list of bounding boxes)
        """
        if not HAS_OPENCV or not HAS_NUMPY:
            return False, []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self._previous_frame is None:
            self._previous_frame = gray
            return False, []
        
        # Compute difference
        frame_delta = cv2.absdiff(self._previous_frame, gray)
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate to fill gaps
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))
        
        self._previous_frame = gray
        motion_detected = len(boxes) > 0
        
        return motion_detected, boxes
    
    def add_motion_callback(self, callback: Callable[[bool, list[tuple[int, int, int, int]]], None]):
        """Add callback for motion events."""
        self._motion_callbacks.append(callback)
    
    def process_frame(self, result: CaptureResult):
        """Process frame for motion (use as streaming callback)."""
        if result.success and result.format == ImageFormat.RGB:
            frame = result.image
            if HAS_OPENCV:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            motion, boxes = self.detect(frame)
            
            for callback in self._motion_callbacks:
                try:
                    callback(motion, boxes)
                except Exception as e:
                    logger.error(f"Motion callback error: {e}")


# Integration with Enigma AI Engine vision
def analyze_image_with_forge(camera: CameraModule, prompt: str = "Describe this image") -> Optional[str]:
    """
    Capture image and analyze with Enigma AI Engine vision.
    
    Args:
        camera: Camera module instance
        prompt: Analysis prompt
        
    Returns:
        Analysis result or None
    """
    result = camera.capture(ImageFormat.JPEG)
    
    if not result.success:
        logger.error(f"Capture failed: {result.error}")
        return None
    
    try:
        from enigma_engine.tools.vision_tools import analyze_image
        return analyze_image(result.image, prompt)
    except ImportError:
        logger.warning("Vision tools not available")
        return None


# Global camera instance
_camera: Optional[CameraModule] = None


def get_camera(config: CameraConfig = None) -> CameraModule:
    """Get global camera instance."""
    global _camera
    if _camera is None:
        _camera = CameraModule(config)
    return _camera


def close_camera():
    """Close global camera."""
    global _camera
    if _camera:
        _camera.close()
        _camera = None
