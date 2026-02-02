"""
Camera Integration Example for ForgeAI

This example shows how to use cameras with ForgeAI.
Cameras enable vision, face tracking, object detection, etc.

SUPPORTED CAMERAS:
- USB webcams
- Raspberry Pi Camera
- IP cameras (RTSP/HTTP streams)
- Screen capture (virtual camera)

USAGE:
    python examples/camera_example.py
    
Or import in your own code:
    from examples.camera_example import create_camera
"""

import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List, Callable, Any


# =============================================================================
# CAMERA INTERFACES
# =============================================================================

class CameraState(Enum):
    """Camera states."""
    CLOSED = "closed"
    OPENING = "opening"
    OPEN = "open"
    STREAMING = "streaming"
    ERROR = "error"


@dataclass
class CameraInfo:
    """Camera information."""
    name: str
    resolution: Tuple[int, int]
    fps: int
    backend: str = "unknown"


class CameraInterface(ABC):
    """
    Abstract base class for cameras.
    Implement for your specific camera type.
    """
    
    def __init__(self, name: str = "camera"):
        self.name = name
        self.state = CameraState.CLOSED
        self.info: Optional[CameraInfo] = None
        self._frame = None
        self._callbacks: List[Callable] = []
    
    @abstractmethod
    def open(self) -> bool:
        """Open the camera. Return True if successful."""
    
    @abstractmethod
    def close(self) -> bool:
        """Close the camera."""
    
    @abstractmethod
    def read(self):
        """Read a single frame. Returns (success, frame)."""
    
    def on_frame(self, callback: Callable):
        """Register callback for new frames."""
        self._callbacks.append(callback)
    
    def _emit_frame(self, frame):
        """Emit frame to callbacks."""
        self._frame = frame
        for cb in self._callbacks:
            try:
                cb(frame)
            except Exception as e:
                print(f"[CAMERA] Callback error: {e}")


# =============================================================================
# EXAMPLE 1: USB Webcam (OpenCV)
# =============================================================================

class USBCamera(CameraInterface):
    """
    Standard USB webcam using OpenCV.
    
    Requirements:
        pip install opencv-python
    
    Args:
        device_id: Camera device (0 = first camera, 1 = second, etc.)
        resolution: (width, height) tuple
        fps: Frames per second
    """
    
    def __init__(
        self,
        device_id: int = 0,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
        name: str = "usb_camera"
    ):
        super().__init__(name)
        self.device_id = device_id
        self.resolution = resolution
        self.fps = fps
        self._cap = None
        self._stream_thread = None
        self._streaming = False
    
    def open(self) -> bool:
        try:
            import cv2
            
            self._cap = cv2.VideoCapture(self.device_id)
            
            if not self._cap.isOpened():
                print(f"[CAMERA] Failed to open device {self.device_id}")
                self.state = CameraState.ERROR
                return False
            
            # Set resolution and FPS
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Get actual settings
            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self._cap.get(cv2.CAP_PROP_FPS))
            
            self.info = CameraInfo(
                name=self.name,
                resolution=(actual_w, actual_h),
                fps=actual_fps,
                backend="opencv"
            )
            
            self.state = CameraState.OPEN
            print(f"[CAMERA] Opened: {actual_w}x{actual_h} @ {actual_fps}fps")
            return True
            
        except ImportError:
            print("[CAMERA] ERROR: Install OpenCV:")
            print("  pip install opencv-python")
            return False
        except Exception as e:
            print(f"[CAMERA] Open failed: {e}")
            self.state = CameraState.ERROR
            return False
    
    def close(self) -> bool:
        self._streaming = False
        if self._cap:
            self._cap.release()
            self._cap = None
        self.state = CameraState.CLOSED
        return True
    
    def read(self):
        """Read a single frame."""
        if not self._cap:
            return False, None
        
        ret, frame = self._cap.read()
        if ret:
            self._frame = frame
        return ret, frame
    
    def start_streaming(self):
        """Start continuous frame capture in background."""
        if self._streaming:
            return
        
        self._streaming = True
        self.state = CameraState.STREAMING
        
        def stream_loop():
            while self._streaming and self._cap:
                ret, frame = self._cap.read()
                if ret:
                    self._emit_frame(frame)
                time.sleep(1.0 / self.fps)
        
        self._stream_thread = threading.Thread(target=stream_loop)
        self._stream_thread.daemon = True
        self._stream_thread.start()
    
    def stop_streaming(self):
        """Stop continuous capture."""
        self._streaming = False
        self.state = CameraState.OPEN


# =============================================================================
# EXAMPLE 2: Raspberry Pi Camera
# =============================================================================

class PiCamera(CameraInterface):
    """
    Raspberry Pi Camera Module.
    
    Requirements:
        pip install picamera2  (Pi OS Bullseye+)
        # OR
        pip install picamera   (older Pi OS)
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
        name: str = "pi_camera"
    ):
        super().__init__(name)
        self.resolution = resolution
        self.fps = fps
        self._picam = None
    
    def open(self) -> bool:
        try:
            # Try picamera2 first (newer)
            try:
                from picamera2 import Picamera2
                
                self._picam = Picamera2()
                config = self._picam.create_preview_configuration(
                    main={"size": self.resolution}
                )
                self._picam.configure(config)
                self._picam.start()
                
                self.info = CameraInfo(
                    name=self.name,
                    resolution=self.resolution,
                    fps=self.fps,
                    backend="picamera2"
                )
                
            except ImportError:
                # Fall back to picamera (older)
                from picamera import PiCamera as PiCam
                from picamera.array import PiRGBArray
                
                self._picam = PiCam()
                self._picam.resolution = self.resolution
                self._picam.framerate = self.fps
                
                self.info = CameraInfo(
                    name=self.name,
                    resolution=self.resolution,
                    fps=self.fps,
                    backend="picamera"
                )
            
            self.state = CameraState.OPEN
            print(f"[CAMERA] Pi Camera opened: {self.resolution}")
            return True
            
        except ImportError:
            print("[CAMERA] ERROR: Install picamera2:")
            print("  pip install picamera2")
            return False
        except Exception as e:
            print(f"[CAMERA] Pi Camera failed: {e}")
            self.state = CameraState.ERROR
            return False
    
    def close(self) -> bool:
        if self._picam:
            try:
                self._picam.close()
            except AttributeError:
                self._picam.stop()
            self._picam = None
        self.state = CameraState.CLOSED
        return True
    
    def read(self):
        """Read a single frame."""
        if not self._picam:
            return False, None
        
        try:
            # picamera2
            if hasattr(self._picam, 'capture_array'):
                frame = self._picam.capture_array()
            else:
                # picamera (older)
                from picamera.array import PiRGBArray
                raw = PiRGBArray(self._picam)
                self._picam.capture(raw, format='rgb')
                frame = raw.array
            
            self._frame = frame
            return True, frame
        except Exception as e:
            print(f"[CAMERA] Capture failed: {e}")
            return False, None


# =============================================================================
# EXAMPLE 3: IP Camera (RTSP/HTTP Stream)
# =============================================================================

class IPCamera(CameraInterface):
    """
    IP camera via RTSP or HTTP stream.
    
    Works with:
        - Security cameras
        - RTSP streams
        - MJPEG HTTP streams
    
    Example URLs:
        rtsp://user:pass@192.168.1.100:554/stream
        http://192.168.1.100/video.mjpg
    """
    
    def __init__(self, url: str, name: str = "ip_camera"):
        super().__init__(name)
        self.url = url
        self._cap = None
    
    def open(self) -> bool:
        try:
            import cv2
            
            self._cap = cv2.VideoCapture(self.url)
            
            if not self._cap.isOpened():
                print(f"[CAMERA] Failed to connect to {self.url}")
                self.state = CameraState.ERROR
                return False
            
            # Get stream info
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self._cap.get(cv2.CAP_PROP_FPS)) or 30
            
            self.info = CameraInfo(
                name=self.name,
                resolution=(w, h),
                fps=fps,
                backend="opencv-stream"
            )
            
            self.state = CameraState.OPEN
            print(f"[CAMERA] IP Camera connected: {w}x{h}")
            return True
            
        except Exception as e:
            print(f"[CAMERA] IP Camera failed: {e}")
            self.state = CameraState.ERROR
            return False
    
    def close(self) -> bool:
        if self._cap:
            self._cap.release()
            self._cap = None
        self.state = CameraState.CLOSED
        return True
    
    def read(self):
        if not self._cap:
            return False, None
        
        ret, frame = self._cap.read()
        if ret:
            self._frame = frame
        return ret, frame


# =============================================================================
# EXAMPLE 4: Screen Capture (Virtual Camera)
# =============================================================================

class ScreenCamera(CameraInterface):
    """
    Capture screen as a camera source.
    
    Useful for:
        - Game AI (see what's on screen)
        - Desktop automation
        - Recording
    
    Requirements:
        pip install mss pillow
    """
    
    def __init__(
        self,
        region: Optional[Tuple[int, int, int, int]] = None,
        fps: int = 30,
        name: str = "screen_camera"
    ):
        super().__init__(name)
        self.region = region  # (x, y, width, height) or None for full screen
        self.fps = fps
        self._sct = None
    
    def open(self) -> bool:
        try:
            import mss
            
            self._sct = mss.mss()
            
            if self.region:
                w, h = self.region[2], self.region[3]
            else:
                monitor = self._sct.monitors[1]
                w, h = monitor["width"], monitor["height"]
            
            self.info = CameraInfo(
                name=self.name,
                resolution=(w, h),
                fps=self.fps,
                backend="mss"
            )
            
            self.state = CameraState.OPEN
            print(f"[CAMERA] Screen capture ready: {w}x{h}")
            return True
            
        except ImportError:
            print("[CAMERA] ERROR: Install mss:")
            print("  pip install mss")
            return False
        except Exception as e:
            print(f"[CAMERA] Screen capture failed: {e}")
            self.state = CameraState.ERROR
            return False
    
    def close(self) -> bool:
        if self._sct:
            self._sct.close()
            self._sct = None
        self.state = CameraState.CLOSED
        return True
    
    def read(self):
        if not self._sct:
            return False, None
        
        try:
            import numpy as np
            
            if self.region:
                monitor = {
                    "left": self.region[0],
                    "top": self.region[1],
                    "width": self.region[2],
                    "height": self.region[3],
                }
            else:
                monitor = self._sct.monitors[1]
            
            screenshot = self._sct.grab(monitor)
            
            # Convert to numpy array (BGR for OpenCV compatibility)
            frame = np.array(screenshot)[:, :, :3]  # Remove alpha
            frame = frame[:, :, ::-1]  # RGB to BGR
            
            self._frame = frame
            return True, frame
            
        except Exception as e:
            print(f"[CAMERA] Screen capture failed: {e}")
            return False, None


# =============================================================================
# CAMERA CONTROLLER
# =============================================================================

class CameraController:
    """
    Main camera controller for ForgeAI.
    Manages multiple cameras and provides unified interface.
    """
    
    def __init__(self):
        self._cameras: dict = {}
        self._active: Optional[str] = None
    
    def add(self, name: str, camera: CameraInterface):
        """Add a camera."""
        self._cameras[name] = camera
        if self._active is None:
            self._active = name
    
    def open(self, name: str = None) -> bool:
        """Open a camera."""
        cam = self._get(name)
        if cam:
            return cam.open()
        return False
    
    def close(self, name: str = None) -> bool:
        """Close a camera."""
        cam = self._get(name)
        if cam:
            return cam.close()
        return False
    
    def read(self, name: str = None):
        """Read frame from camera."""
        cam = self._get(name)
        if cam:
            return cam.read()
        return False, None
    
    def get_frame(self, name: str = None):
        """Get last frame (without new capture)."""
        cam = self._get(name)
        if cam:
            return cam._frame
        return None
    
    def _get(self, name: str = None) -> Optional[CameraInterface]:
        name = name or self._active
        return self._cameras.get(name)


# =============================================================================
# VISION HELPERS (for use with ForgeAI)
# =============================================================================

def detect_faces(frame, return_boxes: bool = True):
    """
    Detect faces in a frame.
    
    Args:
        frame: OpenCV frame (BGR numpy array)
        return_boxes: Return bounding boxes
    
    Returns:
        List of face locations [(x, y, w, h), ...]
    """
    try:
        import cv2
        
        # Load face cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return list(faces) if return_boxes else len(faces)
        
    except Exception as e:
        print(f"[VISION] Face detection failed: {e}")
        return []


def get_face_center(frame) -> Optional[Tuple[float, float]]:
    """
    Get center of first detected face, normalized to -1 to 1.
    
    Useful for face tracking (look_at).
    
    Returns:
        (x, y) where -1,-1 is top-left and 1,1 is bottom-right
        None if no face detected
    """
    faces = detect_faces(frame)
    if not faces:
        return None
    
    x, y, w, h = faces[0]
    frame_h, frame_w = frame.shape[:2]
    
    # Center of face
    cx = x + w / 2
    cy = y + h / 2
    
    # Normalize to -1 to 1
    nx = (cx / frame_w) * 2 - 1
    ny = (cy / frame_h) * 2 - 1
    
    return (nx, ny)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_usb_camera(device_id: int = 0) -> CameraController:
    """Create USB webcam."""
    ctrl = CameraController()
    ctrl.add("usb", USBCamera(device_id=device_id))
    return ctrl


def create_pi_camera() -> CameraController:
    """Create Raspberry Pi camera."""
    ctrl = CameraController()
    ctrl.add("pi", PiCamera())
    return ctrl


def create_ip_camera(url: str) -> CameraController:
    """Create IP camera."""
    ctrl = CameraController()
    ctrl.add("ip", IPCamera(url=url))
    return ctrl


def create_screen_camera(region: Optional[Tuple[int, int, int, int]] = None) -> CameraController:
    """Create screen capture camera."""
    ctrl = CameraController()
    ctrl.add("screen", ScreenCamera(region=region))
    return ctrl


# =============================================================================
# MAIN - Run this file directly to test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ForgeAI Camera Example")
    print("=" * 60)
    
    # Try to use USB webcam, fall back to screen capture
    print("\n[1] Creating camera...")
    
    try:
        import cv2
        controller = create_usb_camera(device_id=0)
        camera_type = "USB Webcam"
    except ImportError:
        controller = create_screen_camera()
        camera_type = "Screen Capture"
    
    print(f"Using: {camera_type}")
    
    print("\n[2] Opening camera...")
    if controller.open():
        print("Camera opened successfully!")
        
        print("\n[3] Capturing frames...")
        for i in range(5):
            ret, frame = controller.read()
            if ret:
                print(f"Frame {i+1}: {frame.shape}")
            time.sleep(0.5)
        
        print("\n[4] Testing face detection...")
        ret, frame = controller.read()
        if ret:
            faces = detect_faces(frame)
            print(f"Faces detected: {len(faces)}")
            
            center = get_face_center(frame)
            if center:
                print(f"Face center: x={center[0]:.2f}, y={center[1]:.2f}")
        
        print("\n[5] Closing camera...")
        controller.close()
    else:
        print("Failed to open camera!")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nAvailable camera types:")
    print("  - create_usb_camera(device_id=0)")
    print("  - create_pi_camera()")
    print("  - create_ip_camera('rtsp://...')")
    print("  - create_screen_camera()")
