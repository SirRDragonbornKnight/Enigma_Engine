"""
Full Body Tracking for Enigma AI Engine

Track body movements for avatar control.

Features:
- Webcam body detection
- Pose estimation
- Hand tracking
- Face landmarks
- Real-time retargeting

Usage:
    from enigma_engine.avatar.body_tracking import BodyTracker
    
    tracker = BodyTracker()
    
    # Start tracking
    tracker.start()
    
    # Get pose data
    pose = tracker.get_pose()
"""

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TrackingMode(Enum):
    """Tracking modes."""
    POSE_ONLY = "pose_only"  # Body pose only
    POSE_HANDS = "pose_hands"  # Body + hands
    FULL_BODY = "full_body"  # Body + hands + face
    HANDS_ONLY = "hands_only"  # Just hands
    FACE_ONLY = "face_only"  # Just face


class BodyPart(Enum):
    """Body parts."""
    # Core
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    
    # Upper body
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    
    # Lower body
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


@dataclass
class Landmark:
    """A single landmark point."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    visibility: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class PoseData:
    """Body pose data."""
    landmarks: Dict[BodyPart, Landmark] = field(default_factory=dict)
    timestamp: float = 0.0
    confidence: float = 0.0
    
    def get_landmark(self, part: BodyPart) -> Optional[Landmark]:
        return self.landmarks.get(part)
    
    def to_dict(self) -> Dict:
        return {
            "landmarks": {
                part.name: {
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                }
                for part, lm in self.landmarks.items()
            },
            "timestamp": self.timestamp,
            "confidence": self.confidence
        }


@dataclass
class HandData:
    """Hand tracking data."""
    landmarks: List[Landmark] = field(default_factory=list)  # 21 landmarks per hand
    handedness: str = "unknown"  # left/right
    confidence: float = 0.0
    
    def get_fingertip(self, finger_idx: int) -> Optional[Landmark]:
        """Get fingertip landmark (0=thumb, 4=pinky)."""
        tips = [4, 8, 12, 16, 20]  # MediaPipe fingertip indices
        if 0 <= finger_idx < 5 and len(self.landmarks) > tips[finger_idx]:
            return self.landmarks[tips[finger_idx]]
        return None


@dataclass
class FaceData:
    """Face tracking data."""
    landmarks: List[Landmark] = field(default_factory=list)  # 468 landmarks
    blend_shapes: Dict[str, float] = field(default_factory=dict)
    
    def get_eye_openness(self) -> Tuple[float, float]:
        """Get left/right eye openness (0-1)."""
        left = self.blend_shapes.get("eyeBlinkLeft", 0.0)
        right = self.blend_shapes.get("eyeBlinkRight", 0.0)
        return (1.0 - left, 1.0 - right)
    
    def get_mouth_openness(self) -> float:
        """Get mouth openness (0-1)."""
        return self.blend_shapes.get("jawOpen", 0.0)


@dataclass
class FullBodyData:
    """Complete body tracking data."""
    pose: Optional[PoseData] = None
    left_hand: Optional[HandData] = None
    right_hand: Optional[HandData] = None
    face: Optional[FaceData] = None
    timestamp: float = 0.0


class BodyTracker:
    """Full body tracking system."""
    
    def __init__(
        self,
        mode: TrackingMode = TrackingMode.POSE_ONLY,
        camera_index: int = 0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize body tracker.
        
        Args:
            mode: Tracking mode
            camera_index: Camera to use
            min_detection_confidence: Detection threshold
            min_tracking_confidence: Tracking threshold
        """
        self.mode = mode
        self.camera_index = camera_index
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Current data
        self._current_pose: Optional[PoseData] = None
        self._current_hands: Tuple[Optional[HandData], Optional[HandData]] = (None, None)
        self._current_face: Optional[FaceData] = None
        
        # MediaPipe (lazy loaded)
        self._mp_pose = None
        self._mp_hands = None
        self._mp_face_mesh = None
        self._cap = None
        
        # Callbacks
        self._pose_callbacks: List[Callable[[PoseData], None]] = []
        self._hand_callbacks: List[Callable[[HandData, str], None]] = []
        self._face_callbacks: List[Callable[[FaceData], None]] = []
        
        logger.info(f"BodyTracker initialized, mode: {mode.value}")
    
    def start(self):
        """Start tracking."""
        if self._running:
            return
        
        # Initialize MediaPipe
        if not self._init_mediapipe():
            logger.error("Failed to initialize MediaPipe")
            return
        
        # Open camera
        try:
            import cv2
            self._cap = cv2.VideoCapture(self.camera_index)
            if not self._cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                return
        except ImportError:
            logger.error("OpenCV not installed")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self._thread.start()
        
        logger.info("Body tracking started")
    
    def stop(self):
        """Stop tracking."""
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=2.0)
        
        if self._cap:
            self._cap.release()
        
        logger.info("Body tracking stopped")
    
    def _init_mediapipe(self) -> bool:
        """Initialize MediaPipe solutions."""
        try:
            import mediapipe as mp
            
            # Initialize based on mode
            if self.mode in [TrackingMode.POSE_ONLY, TrackingMode.POSE_HANDS, TrackingMode.FULL_BODY]:
                self._mp_pose = mp.solutions.pose.Pose(
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence
                )
            
            if self.mode in [TrackingMode.POSE_HANDS, TrackingMode.FULL_BODY, TrackingMode.HANDS_ONLY]:
                self._mp_hands = mp.solutions.hands.Hands(
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                    max_num_hands=2
                )
            
            if self.mode in [TrackingMode.FULL_BODY, TrackingMode.FACE_ONLY]:
                self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                    refine_landmarks=True
                )
            
            return True
            
        except ImportError:
            logger.error("MediaPipe not installed")
            return False
    
    def _tracking_loop(self):
        """Main tracking loop."""
        import cv2
        
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                continue
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process based on mode
            with self._lock:
                if self._mp_pose:
                    self._process_pose(frame_rgb)
                
                if self._mp_hands:
                    self._process_hands(frame_rgb)
                
                if self._mp_face_mesh:
                    self._process_face(frame_rgb)
            
            # Small delay
            time.sleep(0.01)
    
    def _process_pose(self, frame):
        """Process pose detection."""
        results = self._mp_pose.process(frame)
        
        if not results.pose_landmarks:
            return
        
        pose = PoseData(timestamp=time.time())
        
        for i, part in enumerate(BodyPart):
            if i < len(results.pose_landmarks.landmark):
                lm = results.pose_landmarks.landmark[i]
                pose.landmarks[part] = Landmark(
                    x=lm.x,
                    y=lm.y,
                    z=lm.z,
                    visibility=lm.visibility
                )
        
        pose.confidence = np.mean([
            lm.visibility for lm in pose.landmarks.values()
        ])
        
        self._current_pose = pose
        
        # Notify callbacks
        for callback in self._pose_callbacks:
            try:
                callback(pose)
            except Exception as e:
                logger.error(f"Pose callback error: {e}")
    
    def _process_hands(self, frame):
        """Process hand detection."""
        results = self._mp_hands.process(frame)
        
        left_hand = None
        right_hand = None
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                hand_data = HandData(
                    handedness=handedness.classification[0].label.lower(),
                    confidence=handedness.classification[0].score
                )
                
                for lm in hand_landmarks.landmark:
                    hand_data.landmarks.append(Landmark(
                        x=lm.x,
                        y=lm.y,
                        z=lm.z,
                        visibility=1.0
                    ))
                
                if hand_data.handedness == "left":
                    left_hand = hand_data
                else:
                    right_hand = hand_data
                
                # Notify callbacks
                for callback in self._hand_callbacks:
                    try:
                        callback(hand_data, hand_data.handedness)
                    except Exception as e:
                        logger.error(f"Hand callback error: {e}")
        
        self._current_hands = (left_hand, right_hand)
    
    def _process_face(self, frame):
        """Process face detection."""
        results = self._mp_face_mesh.process(frame)
        
        if not results.multi_face_landmarks:
            return
        
        # Take first face
        face_landmarks = results.multi_face_landmarks[0]
        
        face_data = FaceData()
        
        for lm in face_landmarks.landmark:
            face_data.landmarks.append(Landmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=1.0
            ))
        
        # Calculate blend shapes
        face_data.blend_shapes = self._calculate_blend_shapes(face_data.landmarks)
        
        self._current_face = face_data
        
        # Notify callbacks
        for callback in self._face_callbacks:
            try:
                callback(face_data)
            except Exception as e:
                logger.error(f"Face callback error: {e}")
    
    def _calculate_blend_shapes(self, landmarks: List[Landmark]) -> Dict[str, float]:
        """Calculate face blend shapes from landmarks."""
        # Simplified blend shape calculation
        blend_shapes = {}
        
        if len(landmarks) < 468:
            return blend_shapes
        
        # Eye blink (using upper/lower eyelid distance)
        # Left eye: landmarks 159, 145 (upper/lower)
        # Right eye: landmarks 386, 374
        try:
            left_eye_dist = abs(landmarks[159].y - landmarks[145].y)
            right_eye_dist = abs(landmarks[386].y - landmarks[374].y)
            
            # Normalize (typical open eye ~0.03)
            blend_shapes["eyeBlinkLeft"] = max(0, min(1, 1 - left_eye_dist / 0.03))
            blend_shapes["eyeBlinkRight"] = max(0, min(1, 1 - right_eye_dist / 0.03))
            
            # Mouth (landmarks 13, 14 for upper/lower lip)
            mouth_dist = abs(landmarks[13].y - landmarks[14].y)
            blend_shapes["jawOpen"] = max(0, min(1, mouth_dist / 0.05))
            
            # Smile (mouth corners relative to center)
            smile_left = landmarks[61].y - landmarks[13].y
            smile_right = landmarks[291].y - landmarks[13].y
            blend_shapes["mouthSmileLeft"] = max(0, min(1, -smile_left * 20))
            blend_shapes["mouthSmileRight"] = max(0, min(1, -smile_right * 20))
            
            # Eyebrows (landmarks 105, 334 for left/right brow)
            left_brow = 0.3 - landmarks[105].y
            right_brow = 0.3 - landmarks[334].y
            blend_shapes["browDownLeft"] = max(0, min(1, -left_brow * 10))
            blend_shapes["browDownRight"] = max(0, min(1, -right_brow * 10))
            
        except (IndexError, AttributeError) as e:
            logger.debug(f"Blend shape calculation error: {e}")
        
        return blend_shapes
    
    def get_pose(self) -> Optional[PoseData]:
        """Get current pose data."""
        with self._lock:
            return self._current_pose
    
    def get_hands(self) -> Tuple[Optional[HandData], Optional[HandData]]:
        """Get current hand data (left, right)."""
        with self._lock:
            return self._current_hands
    
    def get_face(self) -> Optional[FaceData]:
        """Get current face data."""
        with self._lock:
            return self._current_face
    
    def get_full_body(self) -> FullBodyData:
        """Get complete body tracking data."""
        with self._lock:
            left_hand, right_hand = self._current_hands
            return FullBodyData(
                pose=self._current_pose,
                left_hand=left_hand,
                right_hand=right_hand,
                face=self._current_face,
                timestamp=time.time()
            )
    
    def on_pose(self, callback: Callable[[PoseData], None]):
        """Register pose update callback."""
        self._pose_callbacks.append(callback)
    
    def on_hand(self, callback: Callable[[HandData, str], None]):
        """Register hand update callback."""
        self._hand_callbacks.append(callback)
    
    def on_face(self, callback: Callable[[FaceData], None]):
        """Register face update callback."""
        self._face_callbacks.append(callback)


class PoseRetargeter:
    """Retarget pose data to avatar bones."""
    
    # Standard joint mapping
    JOINT_MAP = {
        BodyPart.LEFT_SHOULDER: "LeftShoulder",
        BodyPart.RIGHT_SHOULDER: "RightShoulder",
        BodyPart.LEFT_ELBOW: "LeftElbow",
        BodyPart.RIGHT_ELBOW: "RightElbow",
        BodyPart.LEFT_WRIST: "LeftWrist",
        BodyPart.RIGHT_WRIST: "RightWrist",
        BodyPart.LEFT_HIP: "LeftHip",
        BodyPart.RIGHT_HIP: "RightHip",
        BodyPart.LEFT_KNEE: "LeftKnee",
        BodyPart.RIGHT_KNEE: "RightKnee",
        BodyPart.LEFT_ANKLE: "LeftAnkle",
        BodyPart.RIGHT_ANKLE: "RightAnkle",
    }
    
    def __init__(self, avatar_scale: float = 1.0):
        """
        Initialize retargeter.
        
        Args:
            avatar_scale: Scale factor for avatar
        """
        self.avatar_scale = avatar_scale
    
    def retarget(
        self,
        pose: PoseData,
        smoothing: float = 0.3
    ) -> Dict[str, Dict[str, float]]:
        """
        Retarget pose to avatar joints.
        
        Args:
            pose: Pose data from tracker
            smoothing: Smoothing factor (0-1)
            
        Returns:
            Dict of joint name -> rotation angles
        """
        joints = {}
        
        for body_part, joint_name in self.JOINT_MAP.items():
            lm = pose.get_landmark(body_part)
            if lm and lm.visibility > 0.5:
                # Calculate rotation based on limb direction
                rotation = self._calculate_rotation(pose, body_part)
                if rotation:
                    joints[joint_name] = rotation
        
        return joints
    
    def _calculate_rotation(
        self,
        pose: PoseData,
        part: BodyPart
    ) -> Optional[Dict[str, float]]:
        """Calculate rotation for a body part."""
        # Get parent-child landmarks
        connections = {
            BodyPart.LEFT_ELBOW: (BodyPart.LEFT_SHOULDER, BodyPart.LEFT_WRIST),
            BodyPart.RIGHT_ELBOW: (BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_WRIST),
            BodyPart.LEFT_KNEE: (BodyPart.LEFT_HIP, BodyPart.LEFT_ANKLE),
            BodyPart.RIGHT_KNEE: (BodyPart.RIGHT_HIP, BodyPart.RIGHT_ANKLE),
        }
        
        if part not in connections:
            return None
        
        parent_part, child_part = connections[part]
        
        parent = pose.get_landmark(parent_part)
        current = pose.get_landmark(part)
        child = pose.get_landmark(child_part)
        
        if not (parent and current and child):
            return None
        
        # Calculate vectors
        v1 = np.array([current.x - parent.x, current.y - parent.y, current.z - parent.z])
        v2 = np.array([child.x - current.x, child.y - current.y, child.z - current.z])
        
        # Normalize
        v1 = v1 / (np.linalg.norm(v1) + 1e-8)
        v2 = v2 / (np.linalg.norm(v2) + 1e-8)
        
        # Calculate angle
        angle = math.acos(np.clip(np.dot(v1, v2), -1, 1))
        
        # Calculate axis
        axis = np.cross(v1, v2)
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        
        return {
            "angle": math.degrees(angle),
            "axis_x": float(axis[0]),
            "axis_y": float(axis[1]),
            "axis_z": float(axis[2])
        }


# Global instance
_body_tracker: Optional[BodyTracker] = None


def get_body_tracker(mode: TrackingMode = TrackingMode.POSE_ONLY) -> BodyTracker:
    """Get or create global BodyTracker instance."""
    global _body_tracker
    if _body_tracker is None:
        _body_tracker = BodyTracker(mode=mode)
    return _body_tracker
