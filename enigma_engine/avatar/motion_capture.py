"""
Real-time Motion Capture for Enigma AI Engine

Webcam-to-avatar tracking using MediaPipe for face mesh and pose.

Provides:
- Face landmark detection (468 points)
- Head pose estimation (pitch, yaw, roll)
- Eye and mouth tracking
- Optional body pose tracking

Usage:
    from enigma_engine.avatar.motion_capture import MotionCapture
    
    mocap = MotionCapture()
    mocap.start()  # Begin capturing
    
    # Get current face data
    face_data = mocap.get_face_data()
    print(face_data.head_rotation)  # (pitch, yaw, roll)
    print(face_data.mouth_open)     # 0.0 - 1.0
    print(face_data.eye_left_open)  # 0.0 - 1.0
    
    # Apply to avatar controller
    mocap.apply_to_live2d(live2d_controller)
    mocap.apply_to_3d(avatar_controller)
    
    mocap.stop()

Requirements:
    pip install mediapipe opencv-python
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
import math

logger = logging.getLogger(__name__)

# Try to import MediaPipe
try:
    import mediapipe as mp
    import cv2
    import numpy as np
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None
    cv2 = None
    np = None


class TrackingMode(Enum):
    """Motion capture tracking modes."""
    FACE_ONLY = auto()      # Face mesh only (fastest)
    FACE_HANDS = auto()     # Face + hands
    FULL_BODY = auto()      # Face + pose (full body)
    ALL = auto()            # Everything


@dataclass
class FaceLandmarks:
    """Key face landmark indices for MediaPipe Face Mesh."""
    # Eyes
    LEFT_EYE_TOP: int = 159
    LEFT_EYE_BOTTOM: int = 145
    RIGHT_EYE_TOP: int = 386
    RIGHT_EYE_BOTTOM: int = 374
    
    # Eye corners  
    LEFT_EYE_OUTER: int = 33
    LEFT_EYE_INNER: int = 133
    RIGHT_EYE_OUTER: int = 362
    RIGHT_EYE_INNER: int = 263
    
    # Mouth
    MOUTH_TOP: int = 13
    MOUTH_BOTTOM: int = 14
    MOUTH_LEFT: int = 61
    MOUTH_RIGHT: int = 291
    
    # Face outline for head pose
    NOSE_TIP: int = 1
    CHIN: int = 199
    LEFT_CHEEK: int = 234
    RIGHT_CHEEK: int = 454
    LEFT_EYE_LEFT: int = 33
    RIGHT_EYE_RIGHT: int = 263


@dataclass 
class FaceData:
    """Captured face tracking data."""
    # Head rotation (degrees)
    head_pitch: float = 0.0     # Up/down (-90 to 90)
    head_yaw: float = 0.0       # Left/right (-90 to 90)
    head_roll: float = 0.0      # Tilt (-90 to 90)
    
    @property
    def head_rotation(self) -> Tuple[float, float, float]:
        return (self.head_pitch, self.head_yaw, self.head_roll)
    
    # Eyes
    eye_left_open: float = 1.0      # 0=closed, 1=open
    eye_right_open: float = 1.0
    eye_left_x: float = 0.0         # Gaze direction (-1 to 1)
    eye_left_y: float = 0.0
    eye_right_x: float = 0.0
    eye_right_y: float = 0.0
    
    @property
    def eye_gaze(self) -> Tuple[float, float]:
        """Average gaze direction."""
        return (
            (self.eye_left_x + self.eye_right_x) / 2,
            (self.eye_left_y + self.eye_right_y) / 2
        )
    
    # Eyebrows
    eyebrow_left_y: float = 0.0    # Raise/lower (-1 to 1)
    eyebrow_right_y: float = 0.0
    
    # Mouth
    mouth_open: float = 0.0         # 0=closed, 1=fully open
    mouth_smile: float = 0.0        # -1=frown, 0=neutral, 1=smile
    mouth_pucker: float = 0.0       # 0=normal, 1=puckered
    
    # Face position in frame (normalized 0-1)
    face_x: float = 0.5
    face_y: float = 0.5
    
    # Confidence
    confidence: float = 0.0
    
    # Raw landmarks (468 points)
    landmarks: List[Tuple[float, float, float]] = field(default_factory=list)


@dataclass
class BodyData:
    """Captured body pose data."""
    # Positions are normalized (-1 to 1 from center)
    left_shoulder: Tuple[float, float, float] = (0, 0, 0)
    right_shoulder: Tuple[float, float, float] = (0, 0, 0)
    left_elbow: Tuple[float, float, float] = (0, 0, 0)
    right_elbow: Tuple[float, float, float] = (0, 0, 0)
    left_wrist: Tuple[float, float, float] = (0, 0, 0)
    right_wrist: Tuple[float, float, float] = (0, 0, 0)
    
    # Body lean
    torso_pitch: float = 0.0
    torso_roll: float = 0.0
    
    confidence: float = 0.0


class MotionCapture:
    """
    Real-time motion capture using webcam and MediaPipe.
    
    Tracks face mesh and optionally body pose, converting
    landmarks to avatar control parameters.
    """
    
    def __init__(
        self,
        camera_id: int = 0,
        mode: TrackingMode = TrackingMode.FACE_ONLY,
        mirror: bool = True
    ):
        """
        Initialize motion capture.
        
        Args:
            camera_id: Camera device ID (0 = default webcam)
            mode: What to track
            mirror: Mirror the input (for selfie-style)
        """
        self.camera_id = camera_id
        self.mode = mode
        self.mirror = mirror
        
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Current tracking data
        self._face_data = FaceData()
        self._body_data = BodyData()
        
        # MediaPipe instances
        self._face_mesh = None
        self._pose = None
        self._hands = None
        self._cap = None
        
        # Smoothing
        self._smooth_factor = 0.3  # Lower = smoother but more latency
        self._prev_face = FaceData()
        
        # Calibration
        self._neutral_pitch = 0.0
        self._neutral_yaw = 0.0
        self._neutral_roll = 0.0
        self._calibrated = False
        
        # Performance
        self._fps = 30
        self._last_frame_time = 0.0
        
        # Callbacks
        self._callbacks: List[Callable[[FaceData], None]] = []
        
        logger.info(f"MotionCapture initialized (camera={camera_id}, mode={mode.name})")
    
    def is_available(self) -> bool:
        """Check if MediaPipe is installed."""
        return MEDIAPIPE_AVAILABLE
    
    def start(self) -> bool:
        """
        Start motion capture.
        
        Returns:
            True if started successfully
        """
        if not MEDIAPIPE_AVAILABLE:
            logger.error("MediaPipe not installed. Run: pip install mediapipe opencv-python")
            return False
        
        if self._running:
            return True
        
        try:
            # Open camera
            self._cap = cv2.VideoCapture(self.camera_id)
            if not self._cap.isOpened():
                logger.error(f"Could not open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)
            
            # Initialize MediaPipe
            mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            if self.mode in [TrackingMode.FULL_BODY, TrackingMode.ALL]:
                mp_pose = mp.solutions.pose
                self._pose = mp_pose.Pose(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            
            self._running = True
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                daemon=True
            )
            self._capture_thread.start()
            
            logger.info("Motion capture started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start motion capture: {e}")
            self._cleanup()
            return False
    
    def stop(self):
        """Stop motion capture."""
        self._running = False
        
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None
        
        self._cleanup()
        logger.info("Motion capture stopped")
    
    def _cleanup(self):
        """Clean up resources."""
        if self._cap:
            self._cap.release()
            self._cap = None
        
        if self._face_mesh:
            self._face_mesh.close()
            self._face_mesh = None
        
        if self._pose:
            self._pose.close()
            self._pose = None
    
    def _capture_loop(self):
        """Main capture loop."""
        frame_time = 1.0 / self._fps
        
        while self._running:
            start = time.perf_counter()
            
            ret, frame = self._cap.read()
            if not ret:
                continue
            
            # Mirror if needed
            if self.mirror:
                frame = cv2.flip(frame, 1)
            
            # Convert to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process face
            face_results = self._face_mesh.process(rgb)
            if face_results.multi_face_landmarks:
                landmarks = face_results.multi_face_landmarks[0].landmark
                self._process_face_landmarks(landmarks)
            
            # Process body if enabled
            if self._pose:
                pose_results = self._pose.process(rgb)
                if pose_results.pose_landmarks:
                    self._process_pose_landmarks(pose_results.pose_landmarks.landmark)
            
            # Notify callbacks
            face_copy = self._get_face_data_copy()
            for callback in self._callbacks:
                try:
                    callback(face_copy)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            
            # Limit frame rate
            elapsed = time.perf_counter() - start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
    
    def _process_face_landmarks(self, landmarks: List):
        """Process face mesh landmarks into FaceData."""
        lm = FaceLandmarks()
        
        # Convert to list of tuples
        points = [(l.x, l.y, l.z) for l in landmarks]
        
        # Calculate head pose from key points
        nose = points[lm.NOSE_TIP]
        chin = points[lm.CHIN]
        left_eye = points[lm.LEFT_EYE_LEFT]
        right_eye = points[lm.RIGHT_EYE_RIGHT]
        left_cheek = points[lm.LEFT_CHEEK]
        right_cheek = points[lm.RIGHT_CHEEK]
        
        # Simple head pose estimation
        # Yaw from left-right asymmetry
        yaw = (right_cheek[0] - left_cheek[0] - 0.5) * 100
        
        # Pitch from nose-chin relationship  
        pitch = (nose[1] - 0.35) * 100  # Nose position relative to expected
        
        # Roll from eye tilt
        roll = (right_eye[1] - left_eye[1]) * 100
        
        # Apply calibration offset
        if self._calibrated:
            pitch -= self._neutral_pitch
            yaw -= self._neutral_yaw
            roll -= self._neutral_roll
        
        # Eye openness
        left_eye_top = points[lm.LEFT_EYE_TOP]
        left_eye_bottom = points[lm.LEFT_EYE_BOTTOM]
        right_eye_top = points[lm.RIGHT_EYE_TOP]
        right_eye_bottom = points[lm.RIGHT_EYE_BOTTOM]
        
        left_eye_open = min(1.0, abs(left_eye_top[1] - left_eye_bottom[1]) * 25)
        right_eye_open = min(1.0, abs(right_eye_top[1] - right_eye_bottom[1]) * 25)
        
        # Mouth openness
        mouth_top = points[lm.MOUTH_TOP]
        mouth_bottom = points[lm.MOUTH_BOTTOM]
        mouth_left = points[lm.MOUTH_LEFT]
        mouth_right = points[lm.MOUTH_RIGHT]
        
        mouth_open = min(1.0, abs(mouth_top[1] - mouth_bottom[1]) * 20)
        mouth_width = abs(mouth_left[0] - mouth_right[0])
        # Smile detection from mouth width ratio
        mouth_smile = max(-1, min(1, (mouth_width - 0.1) * 5 - 0.5))
        
        # Face position
        face_x = nose[0]
        face_y = nose[1]
        
        # Update face data with smoothing
        with self._lock:
            self._face_data.head_pitch = self._smooth(self._prev_face.head_pitch, pitch)
            self._face_data.head_yaw = self._smooth(self._prev_face.head_yaw, yaw)
            self._face_data.head_roll = self._smooth(self._prev_face.head_roll, roll)
            
            self._face_data.eye_left_open = self._smooth(self._prev_face.eye_left_open, left_eye_open)
            self._face_data.eye_right_open = self._smooth(self._prev_face.eye_right_open, right_eye_open)
            
            self._face_data.mouth_open = self._smooth(self._prev_face.mouth_open, mouth_open)
            self._face_data.mouth_smile = self._smooth(self._prev_face.mouth_smile, mouth_smile)
            
            self._face_data.face_x = face_x
            self._face_data.face_y = face_y
            
            self._face_data.landmarks = points
            self._face_data.confidence = 0.9
            
            # Store for next frame
            self._prev_face = FaceData(**self._face_data.__dict__)
    
    def _process_pose_landmarks(self, landmarks: List):
        """Process pose landmarks into BodyData."""
        # MediaPipe Pose indices
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        
        points = [(l.x - 0.5, l.y - 0.5, l.z) for l in landmarks]
        
        with self._lock:
            self._body_data.left_shoulder = points[LEFT_SHOULDER]
            self._body_data.right_shoulder = points[RIGHT_SHOULDER]
            self._body_data.left_elbow = points[LEFT_ELBOW]
            self._body_data.right_elbow = points[RIGHT_ELBOW]
            self._body_data.left_wrist = points[LEFT_WRIST]
            self._body_data.right_wrist = points[RIGHT_WRIST]
            
            # Calculate torso lean
            shoulder_mid = (
                (points[LEFT_SHOULDER][0] + points[RIGHT_SHOULDER][0]) / 2,
                (points[LEFT_SHOULDER][1] + points[RIGHT_SHOULDER][1]) / 2,
            )
            self._body_data.torso_roll = (points[RIGHT_SHOULDER][1] - points[LEFT_SHOULDER][1]) * 50
            self._body_data.torso_pitch = shoulder_mid[1] * 30
            
            self._body_data.confidence = 0.8
    
    def _smooth(self, prev: float, new: float) -> float:
        """Apply exponential smoothing."""
        return prev + (new - prev) * self._smooth_factor
    
    def calibrate(self):
        """
        Calibrate to current head position as neutral.
        
        Look straight ahead and call this to set the zero point.
        """
        with self._lock:
            self._neutral_pitch = self._face_data.head_pitch
            self._neutral_yaw = self._face_data.head_yaw
            self._neutral_roll = self._face_data.head_roll
            self._calibrated = True
        
        logger.info("Motion capture calibrated")
    
    def reset_calibration(self):
        """Reset calibration to defaults."""
        self._neutral_pitch = 0
        self._neutral_yaw = 0
        self._neutral_roll = 0
        self._calibrated = False
    
    def get_face_data(self) -> FaceData:
        """Get current face tracking data."""
        return self._get_face_data_copy()
    
    def _get_face_data_copy(self) -> FaceData:
        with self._lock:
            return FaceData(**self._face_data.__dict__)
    
    def get_body_data(self) -> BodyData:
        """Get current body tracking data."""
        with self._lock:
            return BodyData(**self._body_data.__dict__)
    
    def set_smoothing(self, factor: float):
        """
        Set smoothing factor.
        
        Args:
            factor: 0.0 (maximum smoothing) to 1.0 (no smoothing)
        """
        self._smooth_factor = max(0.05, min(1.0, factor))
    
    def add_callback(self, callback: Callable[[FaceData], None]):
        """Add callback for face data updates."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[FaceData], None]):
        """Remove callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    # === Avatar Integration ===
    
    def apply_to_live2d(self, controller: Any):
        """
        Apply face data to Live2D controller.
        
        Args:
            controller: Live2DController instance
        """
        data = self.get_face_data()
        
        # Head rotation
        controller.set_head_angle(
            x=data.head_yaw * 0.8,
            y=data.head_pitch * 0.8,
            z=data.head_roll * 0.8
        )
        
        # Eyes
        controller.set_eye_openness(
            left=data.eye_left_open,
            right=data.eye_right_open
        )
        
        # Mouth for lip sync
        controller.set_mouth_open(data.mouth_open)
    
    def apply_to_ai_controller(self, controller: Any):
        """
        Apply face data to AI avatar controller.
        
        Args:
            controller: AIAvatarController instance
        """
        data = self.get_face_data()
        
        # Override attention with tracked eye position
        gaze_x, gaze_y = data.eye_gaze
        controller.look_at_point(gaze_x, gaze_y)
        
        # Set speech intensity from mouth
        controller.set_speech_intensity(data.mouth_open)


# Convenience functions

def is_motion_capture_available() -> bool:
    """Check if motion capture is available."""
    return MEDIAPIPE_AVAILABLE


def get_motion_capture_requirements() -> str:
    """Get installation instructions."""
    return """
Motion Capture Requirements:

Install MediaPipe and OpenCV:
    pip install mediapipe opencv-python

Supported features:
- Face mesh (468 landmarks)
- Head pose estimation
- Eye tracking
- Mouth tracking  
- Body pose (optional)

Note: Requires a webcam.
"""
