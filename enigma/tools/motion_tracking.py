"""
Motion Tracking System
======================

Real-time motion tracking for user gesture mimicry using MediaPipe.
Can track poses, hands, face, or full holistic tracking.

Usage:
    from enigma.tools.motion_tracking import MotionTracker
    
    tracker = MotionTracker(camera_id=0, tracking_mode='holistic')
    tracker.start()
    
    # Get latest pose data
    pose_data = tracker.get_pose()
    
    # Stop tracking
    tracker.stop()
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Check for dependencies
HAVE_MEDIAPIPE = False
HAVE_CV2 = False

try:
    import mediapipe as mp
    HAVE_MEDIAPIPE = True
except ImportError:
    logger.warning("MediaPipe not available - motion tracking disabled")

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    logger.warning("OpenCV not available - motion tracking disabled")


class TrackingMode(Enum):
    """Available tracking modes."""
    POSE = "pose"           # Body pose only
    HANDS = "hands"         # Hand tracking only
    FACE = "face"           # Face mesh only
    HOLISTIC = "holistic"   # Full body + hands + face


@dataclass
class PoseData:
    """Container for pose tracking data."""
    timestamp: float
    landmarks: Dict[str, List[Tuple[float, float, float]]]  # x, y, z normalized coords
    visibility: Dict[str, List[float]]  # Visibility scores for each landmark
    tracking_mode: str


class MotionTracker:
    """
    Real-time motion tracking using MediaPipe.
    
    Tracks user movements for avatar mimicry and gesture control.
    """
    
    def __init__(
        self,
        camera_id: int = 0,
        tracking_mode: str = "holistic",
        model_complexity: int = 1
    ):
        """
        Initialize motion tracker.
        
        Args:
            camera_id: Camera device ID
            tracking_mode: One of "pose", "hands", "face", "holistic"
            model_complexity: Model complexity (0=lite, 1=default, 2=heavy)
        """
        if not HAVE_MEDIAPIPE or not HAVE_CV2:
            raise RuntimeError(
                "Motion tracking requires mediapipe and opencv-python. "
                "Install with: pip install mediapipe opencv-python"
            )
        
        self.camera_id = camera_id
        self.tracking_mode = TrackingMode(tracking_mode)
        self.model_complexity = model_complexity
        
        # Initialize MediaPipe solutions
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create appropriate tracker
        if self.tracking_mode == TrackingMode.POSE:
            self.mp_pose = mp.solutions.pose
            self.tracker = self.mp_pose.Pose(
                model_complexity=model_complexity,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        elif self.tracking_mode == TrackingMode.HANDS:
            self.mp_hands = mp.solutions.hands
            self.tracker = self.mp_hands.Hands(
                model_complexity=model_complexity,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        elif self.tracking_mode == TrackingMode.FACE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.tracker = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:  # HOLISTIC
            self.mp_holistic = mp.solutions.holistic
            self.tracker = self.mp_holistic.Holistic(
                model_complexity=model_complexity,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        
        # State
        self.is_running = False
        self.capture = None
        self.thread = None
        self.latest_pose = None
        self.lock = threading.Lock()
        
        logger.info(f"Motion tracker initialized: {tracking_mode} mode")
    
    def start(self) -> bool:
        """
        Start motion tracking in background thread.
        
        Returns:
            True if started successfully
        """
        if self.is_running:
            logger.warning("Motion tracker already running")
            return False
        
        try:
            self.capture = cv2.VideoCapture(self.camera_id)
            if not self.capture.isOpened():
                raise RuntimeError(f"Cannot open camera {self.camera_id}")
            
            self.is_running = True
            self.thread = threading.Thread(target=self._tracking_loop, daemon=True)
            self.thread.start()
            
            logger.info("Motion tracking started")
            return True
        
        except Exception as e:
            logger.error(f"Failed to start motion tracking: {e}")
            return False
    
    def stop(self):
        """Stop motion tracking."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        
        if self.capture:
            self.capture.release()
            self.capture = None
        
        logger.info("Motion tracking stopped")
    
    def _tracking_loop(self):
        """Main tracking loop (runs in background thread)."""
        while self.is_running:
            try:
                success, image = self.capture.read()
                if not success:
                    logger.warning("Failed to read from camera")
                    time.sleep(0.1)
                    continue
                
                # Convert BGR to RGB for MediaPipe
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                
                # Process with MediaPipe
                results = self.tracker.process(image_rgb)
                
                # Extract landmarks based on tracking mode
                pose_data = self._extract_landmarks(results)
                
                # Update latest pose
                with self.lock:
                    self.latest_pose = pose_data
                
                # Small sleep to avoid hogging CPU
                time.sleep(0.01)
            
            except Exception as e:
                logger.error(f"Error in tracking loop: {e}")
                time.sleep(0.1)
    
    def _extract_landmarks(self, results) -> PoseData:
        """Extract landmarks from MediaPipe results."""
        landmarks = {}
        visibility = {}
        
        if self.tracking_mode == TrackingMode.POSE:
            if results.pose_landmarks:
                landmarks['pose'] = [
                    (lm.x, lm.y, lm.z) 
                    for lm in results.pose_landmarks.landmark
                ]
                visibility['pose'] = [
                    lm.visibility 
                    for lm in results.pose_landmarks.landmark
                ]
        
        elif self.tracking_mode == TrackingMode.HANDS:
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    landmarks[f'hand_{idx}'] = [
                        (lm.x, lm.y, lm.z) 
                        for lm in hand_landmarks.landmark
                    ]
                    visibility[f'hand_{idx}'] = [1.0] * len(hand_landmarks.landmark)
        
        elif self.tracking_mode == TrackingMode.FACE:
            if results.multi_face_landmarks:
                for idx, face_landmarks in enumerate(results.multi_face_landmarks):
                    landmarks[f'face_{idx}'] = [
                        (lm.x, lm.y, lm.z) 
                        for lm in face_landmarks.landmark
                    ]
                    visibility[f'face_{idx}'] = [1.0] * len(face_landmarks.landmark)
        
        else:  # HOLISTIC
            if results.pose_landmarks:
                landmarks['pose'] = [
                    (lm.x, lm.y, lm.z) 
                    for lm in results.pose_landmarks.landmark
                ]
                visibility['pose'] = [
                    lm.visibility 
                    for lm in results.pose_landmarks.landmark
                ]
            
            if results.left_hand_landmarks:
                landmarks['left_hand'] = [
                    (lm.x, lm.y, lm.z) 
                    for lm in results.left_hand_landmarks.landmark
                ]
                visibility['left_hand'] = [1.0] * len(results.left_hand_landmarks.landmark)
            
            if results.right_hand_landmarks:
                landmarks['right_hand'] = [
                    (lm.x, lm.y, lm.z) 
                    for lm in results.right_hand_landmarks.landmark
                ]
                visibility['right_hand'] = [1.0] * len(results.right_hand_landmarks.landmark)
            
            if results.face_landmarks:
                landmarks['face'] = [
                    (lm.x, lm.y, lm.z) 
                    for lm in results.face_landmarks.landmark
                ]
                visibility['face'] = [1.0] * len(results.face_landmarks.landmark)
        
        return PoseData(
            timestamp=time.time(),
            landmarks=landmarks,
            visibility=visibility,
            tracking_mode=self.tracking_mode.value
        )
    
    def get_pose(self) -> Optional[PoseData]:
        """
        Get the latest pose data.
        
        Returns:
            PoseData or None if no data available
        """
        with self.lock:
            return self.latest_pose
    
    def get_gesture(self) -> Optional[str]:
        """
        Recognize simple gestures from pose data.
        
        Returns:
            Gesture name or None
        """
        pose = self.get_pose()
        if not pose or not pose.landmarks:
            return None
        
        # Simple gesture recognition
        # This is a placeholder - can be extended with more sophisticated detection
        
        if 'left_hand' in pose.landmarks and 'right_hand' in pose.landmarks:
            left_hand = pose.landmarks['left_hand']
            right_hand = pose.landmarks['right_hand']
            
            # Check if hands are raised (y < 0.5 means upper half of frame)
            left_raised = left_hand[0][1] < 0.5 if left_hand else False
            right_raised = right_hand[0][1] < 0.5 if right_hand else False
            
            if left_raised and right_raised:
                return "hands_up"
            elif left_raised:
                return "left_hand_up"
            elif right_raised:
                return "right_hand_up"
        
        return None
    
    def is_active(self) -> bool:
        """Check if tracker is currently running."""
        return self.is_running
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop()


def test_motion_tracking():
    """Test function to verify motion tracking works."""
    print("Testing motion tracking...")
    
    if not HAVE_MEDIAPIPE or not HAVE_CV2:
        print("❌ MediaPipe or OpenCV not available")
        return False
    
    try:
        tracker = MotionTracker(camera_id=0, tracking_mode='pose')
        tracker.start()
        
        print("✓ Tracking started, reading poses for 5 seconds...")
        start_time = time.time()
        pose_count = 0
        
        while time.time() - start_time < 5.0:
            pose = tracker.get_pose()
            if pose:
                pose_count += 1
            time.sleep(0.1)
        
        tracker.stop()
        print(f"✓ Captured {pose_count} poses")
        return True
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    test_motion_tracking()
