"""
Sample Avatar Generator

Creates programmatically-generated sample avatars for users who don't have
their own images. These are simple but functional avatars.

Avatars Generated:
- Blob (cute circular character)
- Robot (geometric robot face)
- Cat (simple cat avatar)
- Ghost (friendly ghost)
- Cube (abstract geometric)
"""

from pathlib import Path
from typing import Optional

# Check for PyQt5 (needed for drawing)
try:
    from PyQt5.QtCore import QRect, Qt
    from PyQt5.QtGui import (
        QBrush,
        QColor,
        QLinearGradient,
        QPainter,
        QPainterPath,
        QPen,
        QPixmap,
        QRadialGradient,
    )
    from PyQt5.QtWidgets import QApplication
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False


class SampleAvatarGenerator:
    """
    Generates sample avatar images programmatically.
    
    Usage:
        generator = SampleAvatarGenerator()
        generator.generate_all("data/avatar/samples/")
        
        # Or generate specific avatars
        generator.generate_blob("my_blob.png", emotion="happy")
    """
    
    # Standard emotions to generate
    EMOTIONS = [
        "neutral", "happy", "sad", "surprised", 
        "thinking", "confused", "angry", "excited"
    ]
    
    def __init__(self, size: int = 256):
        self.size = size
        
        # Ensure QApplication exists for drawing
        if HAS_PYQT:
            self._app = QApplication.instance()
            if self._app is None:
                self._app = QApplication([])
    
    def generate_all(self, output_dir: str) -> dict[str, Path]:
        """
        Generate all sample avatars.
        
        Returns:
            Dict mapping avatar name to its directory path
        """
        if not HAS_PYQT:
            raise RuntimeError("PyQt5 required for avatar generation")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        avatars = {}
        
        # Generate each type
        generators = [
            ("blob", self.generate_blob_set),
            ("robot", self.generate_robot_set),
            ("cat", self.generate_cat_set),
            ("ghost", self.generate_ghost_set),
            ("cube", self.generate_cube_set),
        ]
        
        for name, generator in generators:
            avatar_dir = output_dir / name
            avatar_dir.mkdir(exist_ok=True)
            generator(str(avatar_dir))
            avatars[name] = avatar_dir
            
            # Create manifest
            self._create_manifest(avatar_dir, name)
        
        return avatars
    
    def _create_manifest(self, avatar_dir: Path, avatar_type: str):
        """Create a manifest.json for the avatar."""
        import json
        
        type_info = {
            "blob": {
                "name": "Friendly Blob",
                "description": "A cute, friendly blob character",
                "avatar_type": "ABSTRACT",
                "tags": ["cute", "simple", "friendly"]
            },
            "robot": {
                "name": "Robo Helper",
                "description": "A helpful robot assistant",
                "avatar_type": "ROBOT",
                "tags": ["robot", "tech", "helper"]
            },
            "cat": {
                "name": "Whiskers",
                "description": "An adorable cat companion",
                "avatar_type": "ANIMAL",
                "tags": ["cat", "cute", "pet"]
            },
            "ghost": {
                "name": "Boo",
                "description": "A friendly ghost friend",
                "avatar_type": "FANTASY",
                "tags": ["ghost", "cute", "spooky"]
            },
            "cube": {
                "name": "Cubey",
                "description": "An abstract geometric companion",
                "avatar_type": "ABSTRACT",
                "tags": ["abstract", "geometric", "minimal"]
            }
        }
        
        info = type_info.get(avatar_type, {})
        
        manifest = {
            "name": info.get("name", avatar_type.title()),
            "author": "Enigma AI Engine",
            "description": info.get("description", "A sample avatar"),
            "version": "1.0",
            "bundle_version": "1.0",
            "avatar_type": info.get("avatar_type", "ABSTRACT"),
            "default_mode": "PNG_BOUNCE",
            "base_image": "neutral.png",
            "emotion_images": {e: f"{e}.png" for e in self.EMOTIONS},
            "tags": info.get("tags", [])
        }
        
        manifest_path = avatar_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
    
    # =========================================================================
    # BLOB AVATAR
    # =========================================================================
    
    def generate_blob_set(self, output_dir: str):
        """Generate a full set of blob emotions."""
        for emotion in self.EMOTIONS:
            self.generate_blob(f"{output_dir}/{emotion}.png", emotion)
    
    def generate_blob(self, output_path: str, emotion: str = "neutral"):
        """Generate a blob avatar with specified emotion."""
        pixmap = QPixmap(self.size, self.size)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        cx, cy = self.size // 2, self.size // 2
        radius = self.size // 2 - 20
        
        # Body gradient
        gradient = QRadialGradient(cx - radius//3, cy - radius//3, radius * 1.5)
        gradient.setColorAt(0, QColor(150, 200, 255))
        gradient.setColorAt(1, QColor(80, 140, 220))
        
        # Adjust color by emotion
        if emotion == "happy":
            gradient.setColorAt(0, QColor(255, 220, 150))
            gradient.setColorAt(1, QColor(255, 180, 80))
        elif emotion == "sad":
            gradient.setColorAt(0, QColor(150, 180, 220))
            gradient.setColorAt(1, QColor(100, 130, 180))
        elif emotion == "angry":
            gradient.setColorAt(0, QColor(255, 150, 150))
            gradient.setColorAt(1, QColor(220, 80, 80))
        elif emotion == "excited":
            gradient.setColorAt(0, QColor(255, 200, 255))
            gradient.setColorAt(1, QColor(220, 100, 220))
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(QColor(60, 100, 160), 3))
        
        # Squash/stretch based on emotion
        squash = 1.0
        if emotion == "surprised":
            squash = 1.2
        elif emotion == "sad":
            squash = 0.85
        
        painter.drawEllipse(
            int(cx - radius), 
            int(cy - radius * squash), 
            int(radius * 2), 
            int(radius * 2 * squash)
        )
        
        # Eyes
        self._draw_blob_eyes(painter, cx, cy, radius, emotion)
        
        # Mouth
        self._draw_blob_mouth(painter, cx, cy, radius, emotion)
        
        painter.end()
        pixmap.save(output_path)
    
    def _draw_blob_eyes(self, painter, cx, cy, radius, emotion):
        """Draw blob eyes."""
        eye_y = cy - radius // 4
        eye_spacing = radius // 2
        eye_size = radius // 4
        
        # White of eyes
        painter.setBrush(QColor(255, 255, 255))
        painter.setPen(QPen(QColor(100, 100, 100), 2))
        
        if emotion == "surprised":
            eye_size = int(eye_size * 1.3)
        elif emotion in ["sad", "thinking"]:
            eye_size = int(eye_size * 0.9)
        
        # Left eye
        painter.drawEllipse(cx - eye_spacing - eye_size//2, eye_y - eye_size//2, eye_size, eye_size)
        # Right eye
        painter.drawEllipse(cx + eye_spacing - eye_size//2, eye_y - eye_size//2, eye_size, eye_size)
        
        # Pupils
        pupil_size = eye_size // 2
        painter.setBrush(QColor(30, 30, 30))
        painter.setPen(Qt.NoPen)
        
        # Pupil offset based on emotion
        pupil_offset_x = 0
        pupil_offset_y = 0
        if emotion == "thinking":
            pupil_offset_x = -3
            pupil_offset_y = -3
        elif emotion == "confused":
            pupil_offset_x = 3
        
        painter.drawEllipse(
            cx - eye_spacing - pupil_size//2 + pupil_offset_x,
            eye_y - pupil_size//2 + pupil_offset_y,
            pupil_size, pupil_size
        )
        painter.drawEllipse(
            cx + eye_spacing - pupil_size//2 + pupil_offset_x,
            eye_y - pupil_size//2 + pupil_offset_y,
            pupil_size, pupil_size
        )
        
        # Eyebrows for some emotions
        if emotion == "angry":
            painter.setPen(QPen(QColor(80, 80, 80), 4))
            painter.drawLine(cx - eye_spacing - eye_size, eye_y - eye_size, 
                           cx - eye_spacing + eye_size//2, eye_y - eye_size//2)
            painter.drawLine(cx + eye_spacing + eye_size, eye_y - eye_size,
                           cx + eye_spacing - eye_size//2, eye_y - eye_size//2)
        elif emotion == "sad":
            painter.setPen(QPen(QColor(80, 80, 80), 3))
            painter.drawLine(cx - eye_spacing - eye_size//2, eye_y - eye_size//2,
                           cx - eye_spacing + eye_size//2, eye_y - eye_size)
            painter.drawLine(cx + eye_spacing - eye_size//2, eye_y - eye_size,
                           cx + eye_spacing + eye_size//2, eye_y - eye_size//2)
    
    def _draw_blob_mouth(self, painter, cx, cy, radius, emotion):
        """Draw blob mouth."""
        mouth_y = cy + radius // 3
        mouth_width = radius // 2
        
        painter.setPen(QPen(QColor(80, 60, 60), 4))
        painter.setBrush(Qt.NoBrush)
        
        if emotion == "happy" or emotion == "excited":
            # Big smile
            path = QPainterPath()
            path.moveTo(cx - mouth_width, mouth_y)
            path.quadTo(cx, mouth_y + mouth_width//2, cx + mouth_width, mouth_y)
            painter.drawPath(path)
        elif emotion == "sad":
            # Frown
            path = QPainterPath()
            path.moveTo(cx - mouth_width//2, mouth_y + 10)
            path.quadTo(cx, mouth_y - mouth_width//3, cx + mouth_width//2, mouth_y + 10)
            painter.drawPath(path)
        elif emotion == "surprised":
            # O mouth
            painter.setBrush(QColor(80, 60, 60))
            painter.drawEllipse(cx - mouth_width//4, mouth_y - mouth_width//4, 
                              mouth_width//2, mouth_width//2)
        elif emotion == "thinking":
            # Small line
            painter.drawLine(cx - mouth_width//3, mouth_y, cx + mouth_width//3, mouth_y)
        elif emotion == "confused":
            # Wavy line
            path = QPainterPath()
            path.moveTo(cx - mouth_width//2, mouth_y)
            path.cubicTo(cx - mouth_width//4, mouth_y - 10, 
                        cx + mouth_width//4, mouth_y + 10,
                        cx + mouth_width//2, mouth_y)
            painter.drawPath(path)
        elif emotion == "angry":
            # Angry mouth
            path = QPainterPath()
            path.moveTo(cx - mouth_width//2, mouth_y - 5)
            path.lineTo(cx, mouth_y + 5)
            path.lineTo(cx + mouth_width//2, mouth_y - 5)
            painter.drawPath(path)
        else:
            # Neutral smile
            path = QPainterPath()
            path.moveTo(cx - mouth_width//2, mouth_y)
            path.quadTo(cx, mouth_y + mouth_width//4, cx + mouth_width//2, mouth_y)
            painter.drawPath(path)
    
    # =========================================================================
    # ROBOT AVATAR
    # =========================================================================
    
    def generate_robot_set(self, output_dir: str):
        """Generate a full set of robot emotions."""
        for emotion in self.EMOTIONS:
            self.generate_robot(f"{output_dir}/{emotion}.png", emotion)
    
    def generate_robot(self, output_path: str, emotion: str = "neutral"):
        """Generate a robot avatar."""
        pixmap = QPixmap(self.size, self.size)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        cx, cy = self.size // 2, self.size // 2
        head_size = self.size - 60
        
        # Head (rounded rectangle)
        gradient = QLinearGradient(0, 0, 0, self.size)
        gradient.setColorAt(0, QColor(180, 190, 200))
        gradient.setColorAt(1, QColor(120, 130, 140))
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(QColor(80, 90, 100), 3))
        
        head_rect = QRect(
            cx - head_size//2, cy - head_size//2,
            head_size, int(head_size * 0.9)
        )
        painter.drawRoundedRect(head_rect, 20, 20)
        
        # Antenna
        painter.setPen(QPen(QColor(100, 100, 100), 4))
        painter.drawLine(cx, cy - head_size//2, cx, cy - head_size//2 - 20)
        painter.setBrush(QColor(255, 100, 100) if emotion == "angry" else QColor(100, 255, 100))
        painter.drawEllipse(cx - 8, cy - head_size//2 - 35, 16, 16)
        
        # Eyes (LED screens)
        self._draw_robot_eyes(painter, cx, cy, head_size, emotion)
        
        # Mouth (LED bar)
        self._draw_robot_mouth(painter, cx, cy, head_size, emotion)
        
        # Bolts
        bolt_color = QColor(80, 80, 90)
        painter.setBrush(bolt_color)
        painter.setPen(Qt.NoPen)
        bolt_size = 10
        painter.drawEllipse(cx - head_size//2 + 15, cy - head_size//2 + 15, bolt_size, bolt_size)
        painter.drawEllipse(cx + head_size//2 - 25, cy - head_size//2 + 15, bolt_size, bolt_size)
        
        painter.end()
        pixmap.save(output_path)
    
    def _draw_robot_eyes(self, painter, cx, cy, head_size, emotion):
        """Draw robot LED eyes."""
        eye_y = cy - head_size // 6
        eye_spacing = head_size // 4
        eye_width = head_size // 4
        eye_height = head_size // 5
        
        # Eye socket
        painter.setBrush(QColor(30, 30, 40))
        painter.setPen(QPen(QColor(60, 60, 70), 2))
        painter.drawRoundedRect(cx - eye_spacing - eye_width//2, eye_y - eye_height//2,
                               eye_width, eye_height, 5, 5)
        painter.drawRoundedRect(cx + eye_spacing - eye_width//2, eye_y - eye_height//2,
                               eye_width, eye_height, 5, 5)
        
        # LED color based on emotion
        led_color = QColor(0, 200, 255)  # Cyan default
        if emotion == "happy" or emotion == "excited":
            led_color = QColor(0, 255, 100)
        elif emotion == "sad":
            led_color = QColor(100, 100, 255)
        elif emotion == "angry":
            led_color = QColor(255, 50, 50)
        elif emotion == "thinking":
            led_color = QColor(255, 255, 100)
        
        # LED pupils
        painter.setBrush(led_color)
        painter.setPen(Qt.NoPen)
        
        pupil_w = eye_width // 2
        pupil_h = eye_height // 2
        
        if emotion == "surprised":
            pupil_w = int(pupil_w * 1.3)
            pupil_h = int(pupil_h * 1.3)
        
        painter.drawEllipse(cx - eye_spacing - pupil_w//2, eye_y - pupil_h//2, pupil_w, pupil_h)
        painter.drawEllipse(cx + eye_spacing - pupil_w//2, eye_y - pupil_h//2, pupil_w, pupil_h)
    
    def _draw_robot_mouth(self, painter, cx, cy, head_size, emotion):
        """Draw robot LED mouth."""
        mouth_y = cy + head_size // 4
        mouth_width = head_size // 2
        mouth_height = head_size // 8
        
        # Mouth background
        painter.setBrush(QColor(30, 30, 40))
        painter.setPen(QPen(QColor(60, 60, 70), 2))
        painter.drawRoundedRect(cx - mouth_width//2, mouth_y - mouth_height//2,
                               mouth_width, mouth_height, 5, 5)
        
        # LED segments
        led_color = QColor(0, 200, 255)
        if emotion in ["happy", "excited"]:
            led_color = QColor(0, 255, 100)
        elif emotion == "angry":
            led_color = QColor(255, 50, 50)
        
        painter.setBrush(led_color)
        segment_width = mouth_width // 8
        gap = 3
        
        # Draw LED pattern based on emotion
        if emotion == "happy" or emotion == "excited":
            # Smile pattern (arc up)
            heights = [0.3, 0.6, 0.9, 1.0, 1.0, 0.9, 0.6, 0.3]
        elif emotion == "sad":
            # Frown pattern (arc down)
            heights = [0.9, 0.6, 0.4, 0.3, 0.3, 0.4, 0.6, 0.9]
        elif emotion == "surprised":
            # O pattern
            heights = [0.5, 0.8, 1.0, 1.0, 1.0, 1.0, 0.8, 0.5]
        else:
            # Flat
            heights = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        
        for i, h in enumerate(heights):
            seg_h = int((mouth_height - 6) * h)
            x = cx - mouth_width//2 + 3 + i * (segment_width + gap)
            y = mouth_y + mouth_height//2 - 3 - seg_h
            painter.drawRect(x, y, segment_width, seg_h)
    
    # =========================================================================
    # CAT AVATAR
    # =========================================================================
    
    def generate_cat_set(self, output_dir: str):
        """Generate a full set of cat emotions."""
        for emotion in self.EMOTIONS:
            self.generate_cat(f"{output_dir}/{emotion}.png", emotion)
    
    def generate_cat(self, output_path: str, emotion: str = "neutral"):
        """Generate a cat avatar."""
        pixmap = QPixmap(self.size, self.size)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        cx, cy = self.size // 2, self.size // 2 + 20
        head_radius = self.size // 2 - 40
        
        # Fur color
        fur_gradient = QRadialGradient(cx - head_radius//3, cy - head_radius//3, head_radius * 1.5)
        fur_gradient.setColorAt(0, QColor(255, 180, 100))  # Orange tabby
        fur_gradient.setColorAt(1, QColor(220, 140, 60))
        
        painter.setBrush(QBrush(fur_gradient))
        painter.setPen(QPen(QColor(180, 120, 60), 3))
        
        # Head (slightly squished circle)
        painter.drawEllipse(cx - head_radius, cy - int(head_radius * 0.9),
                           head_radius * 2, int(head_radius * 1.8))
        
        # Ears
        self._draw_cat_ears(painter, cx, cy, head_radius)
        
        # Eyes
        self._draw_cat_eyes(painter, cx, cy, head_radius, emotion)
        
        # Nose and mouth
        self._draw_cat_nose_mouth(painter, cx, cy, head_radius, emotion)
        
        # Whiskers
        self._draw_cat_whiskers(painter, cx, cy, head_radius)
        
        painter.end()
        pixmap.save(output_path)
    
    def _draw_cat_ears(self, painter, cx, cy, radius):
        """Draw cat ears."""
        ear_size = radius // 2
        ear_y = cy - radius + 10
        
        # Outer ear
        painter.setBrush(QColor(255, 180, 100))
        painter.setPen(QPen(QColor(180, 120, 60), 3))
        
        # Left ear
        left_ear = QPainterPath()
        left_ear.moveTo(cx - radius + 20, ear_y + ear_size)
        left_ear.lineTo(cx - radius - 10, ear_y - ear_size)
        left_ear.lineTo(cx - radius//2, ear_y + ear_size//2)
        left_ear.closeSubpath()
        painter.drawPath(left_ear)
        
        # Right ear
        right_ear = QPainterPath()
        right_ear.moveTo(cx + radius - 20, ear_y + ear_size)
        right_ear.lineTo(cx + radius + 10, ear_y - ear_size)
        right_ear.lineTo(cx + radius//2, ear_y + ear_size//2)
        right_ear.closeSubpath()
        painter.drawPath(right_ear)
        
        # Inner ear (pink)
        painter.setBrush(QColor(255, 180, 180))
        painter.setPen(Qt.NoPen)
        
        inner_left = QPainterPath()
        inner_left.moveTo(cx - radius + 25, ear_y + ear_size - 10)
        inner_left.lineTo(cx - radius - 2, ear_y - ear_size + 15)
        inner_left.lineTo(cx - radius//2 - 5, ear_y + ear_size//2)
        inner_left.closeSubpath()
        painter.drawPath(inner_left)
        
        inner_right = QPainterPath()
        inner_right.moveTo(cx + radius - 25, ear_y + ear_size - 10)
        inner_right.lineTo(cx + radius + 2, ear_y - ear_size + 15)
        inner_right.lineTo(cx + radius//2 + 5, ear_y + ear_size//2)
        inner_right.closeSubpath()
        painter.drawPath(inner_right)
    
    def _draw_cat_eyes(self, painter, cx, cy, radius, emotion):
        """Draw cat eyes."""
        eye_y = cy
        eye_spacing = radius // 2
        eye_size = radius // 3
        
        # Eye shape (almond)
        painter.setBrush(QColor(200, 220, 100))  # Yellow-green
        painter.setPen(QPen(QColor(50, 50, 50), 2))
        
        if emotion == "happy" or emotion == "excited":
            # Closed happy eyes
            painter.setPen(QPen(QColor(50, 50, 50), 4))
            painter.setBrush(Qt.NoBrush)
            path = QPainterPath()
            path.moveTo(cx - eye_spacing - eye_size//2, eye_y)
            path.quadTo(cx - eye_spacing, eye_y - eye_size//2, cx - eye_spacing + eye_size//2, eye_y)
            painter.drawPath(path)
            path = QPainterPath()
            path.moveTo(cx + eye_spacing - eye_size//2, eye_y)
            path.quadTo(cx + eye_spacing, eye_y - eye_size//2, cx + eye_spacing + eye_size//2, eye_y)
            painter.drawPath(path)
        else:
            # Open eyes
            painter.drawEllipse(cx - eye_spacing - eye_size//2, eye_y - eye_size//2, eye_size, eye_size)
            painter.drawEllipse(cx + eye_spacing - eye_size//2, eye_y - eye_size//2, eye_size, eye_size)
            
            # Pupils (vertical slits)
            painter.setBrush(QColor(20, 20, 20))
            pupil_w = eye_size // 6
            pupil_h = eye_size - 8
            if emotion == "surprised":
                pupil_w = eye_size // 3  # Dilated
            elif emotion == "angry":
                pupil_h = eye_size // 2  # Narrow
            
            painter.drawEllipse(cx - eye_spacing - pupil_w//2, eye_y - pupil_h//2, pupil_w, pupil_h)
            painter.drawEllipse(cx + eye_spacing - pupil_w//2, eye_y - pupil_h//2, pupil_w, pupil_h)
    
    def _draw_cat_nose_mouth(self, painter, cx, cy, radius, emotion):
        """Draw cat nose and mouth."""
        nose_y = cy + radius // 3
        
        # Nose (triangle)
        painter.setBrush(QColor(255, 150, 150))
        painter.setPen(QPen(QColor(180, 100, 100), 2))
        
        nose = QPainterPath()
        nose.moveTo(cx, nose_y - 10)
        nose.lineTo(cx - 12, nose_y + 8)
        nose.lineTo(cx + 12, nose_y + 8)
        nose.closeSubpath()
        painter.drawPath(nose)
        
        # Mouth
        painter.setPen(QPen(QColor(80, 60, 60), 3))
        painter.setBrush(Qt.NoBrush)
        
        mouth_y = nose_y + 15
        
        if emotion in ["happy", "excited"]:
            # W shape smile
            path = QPainterPath()
            path.moveTo(cx - 25, mouth_y)
            path.quadTo(cx - 12, mouth_y + 15, cx, mouth_y)
            path.quadTo(cx + 12, mouth_y + 15, cx + 25, mouth_y)
            painter.drawPath(path)
        elif emotion == "surprised":
            # O mouth
            painter.setBrush(QColor(80, 60, 60))
            painter.drawEllipse(cx - 8, mouth_y - 5, 16, 12)
        else:
            # Standard cat mouth
            painter.drawLine(cx, nose_y + 8, cx, mouth_y)
            path = QPainterPath()
            path.moveTo(cx - 20, mouth_y + 5)
            path.quadTo(cx - 10, mouth_y + 10, cx, mouth_y)
            painter.drawPath(path)
            path = QPainterPath()
            path.moveTo(cx + 20, mouth_y + 5)
            path.quadTo(cx + 10, mouth_y + 10, cx, mouth_y)
            painter.drawPath(path)
    
    def _draw_cat_whiskers(self, painter, cx, cy, radius):
        """Draw cat whiskers."""
        whisker_y = cy + radius // 3
        painter.setPen(QPen(QColor(80, 60, 40), 2))
        
        # Left whiskers
        painter.drawLine(cx - 20, whisker_y - 5, cx - radius - 10, whisker_y - 15)
        painter.drawLine(cx - 20, whisker_y + 5, cx - radius - 10, whisker_y + 5)
        painter.drawLine(cx - 20, whisker_y + 15, cx - radius - 10, whisker_y + 25)
        
        # Right whiskers
        painter.drawLine(cx + 20, whisker_y - 5, cx + radius + 10, whisker_y - 15)
        painter.drawLine(cx + 20, whisker_y + 5, cx + radius + 10, whisker_y + 5)
        painter.drawLine(cx + 20, whisker_y + 15, cx + radius + 10, whisker_y + 25)
    
    # =========================================================================
    # GHOST AVATAR
    # =========================================================================
    
    def generate_ghost_set(self, output_dir: str):
        """Generate a full set of ghost emotions."""
        for emotion in self.EMOTIONS:
            self.generate_ghost(f"{output_dir}/{emotion}.png", emotion)
    
    def generate_ghost(self, output_path: str, emotion: str = "neutral"):
        """Generate a ghost avatar."""
        pixmap = QPixmap(self.size, self.size)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        cx, cy = self.size // 2, self.size // 2
        body_width = self.size - 80
        body_height = self.size - 40
        
        # Ghost body gradient
        gradient = QRadialGradient(cx, cy - body_height//4, body_width)
        gradient.setColorAt(0, QColor(255, 255, 255, 240))
        gradient.setColorAt(0.7, QColor(220, 230, 255, 200))
        gradient.setColorAt(1, QColor(180, 200, 255, 150))
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(QColor(180, 190, 220), 3))
        
        # Body shape with wavy bottom
        path = QPainterPath()
        top = cy - body_height // 2 + 20
        bottom = cy + body_height // 2 - 10
        
        # Start at top left, go around
        path.moveTo(cx - body_width//2, cy)
        path.quadTo(cx - body_width//2, top - 20, cx, top - 30)
        path.quadTo(cx + body_width//2, top - 20, cx + body_width//2, cy)
        
        # Wavy bottom
        wave_count = 4
        wave_width = body_width // wave_count
        x = cx + body_width//2
        for i in range(wave_count):
            ctrl_y = bottom + 20 if i % 2 == 0 else bottom - 10
            x -= wave_width
            path.quadTo(x + wave_width//2, ctrl_y, x, bottom)
        
        path.closeSubpath()
        painter.drawPath(path)
        
        # Eyes
        self._draw_ghost_eyes(painter, cx, cy, body_width, emotion)
        
        # Mouth
        self._draw_ghost_mouth(painter, cx, cy, body_width, emotion)
        
        # Blush for some emotions
        if emotion in ["happy", "excited", "embarrassed"]:
            painter.setBrush(QColor(255, 180, 180, 100))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(cx - body_width//3, cy, 20, 12)
            painter.drawEllipse(cx + body_width//3 - 20, cy, 20, 12)
        
        painter.end()
        pixmap.save(output_path)
    
    def _draw_ghost_eyes(self, painter, cx, cy, body_width, emotion):
        """Draw ghost eyes."""
        eye_y = cy - 20
        eye_spacing = body_width // 5
        eye_size = body_width // 6
        
        painter.setBrush(QColor(30, 30, 50))
        painter.setPen(Qt.NoPen)
        
        if emotion == "happy":
            # Curved happy eyes
            painter.setPen(QPen(QColor(30, 30, 50), 4))
            painter.setBrush(Qt.NoBrush)
            path = QPainterPath()
            path.moveTo(cx - eye_spacing - eye_size//2, eye_y)
            path.quadTo(cx - eye_spacing, eye_y - eye_size//2, cx - eye_spacing + eye_size//2, eye_y)
            painter.drawPath(path)
            path = QPainterPath()
            path.moveTo(cx + eye_spacing - eye_size//2, eye_y)
            path.quadTo(cx + eye_spacing, eye_y - eye_size//2, cx + eye_spacing + eye_size//2, eye_y)
            painter.drawPath(path)
        elif emotion == "surprised":
            # Big round eyes
            painter.drawEllipse(cx - eye_spacing - eye_size//2, eye_y - eye_size//2,
                              int(eye_size * 1.2), int(eye_size * 1.2))
            painter.drawEllipse(cx + eye_spacing - eye_size//2, eye_y - eye_size//2,
                              int(eye_size * 1.2), int(eye_size * 1.2))
        else:
            # Normal oval eyes
            painter.drawEllipse(cx - eye_spacing - eye_size//2, eye_y - eye_size//3,
                              eye_size, int(eye_size * 0.7))
            painter.drawEllipse(cx + eye_spacing - eye_size//2, eye_y - eye_size//3,
                              eye_size, int(eye_size * 0.7))
    
    def _draw_ghost_mouth(self, painter, cx, cy, body_width, emotion):
        """Draw ghost mouth."""
        mouth_y = cy + 25
        mouth_width = body_width // 4
        
        painter.setPen(QPen(QColor(30, 30, 50), 3))
        painter.setBrush(Qt.NoBrush)
        
        if emotion in ["happy", "excited"]:
            path = QPainterPath()
            path.moveTo(cx - mouth_width//2, mouth_y)
            path.quadTo(cx, mouth_y + mouth_width//2, cx + mouth_width//2, mouth_y)
            painter.drawPath(path)
        elif emotion == "surprised":
            painter.setBrush(QColor(30, 30, 50))
            painter.drawEllipse(cx - 12, mouth_y - 8, 24, 20)
        elif emotion == "sad":
            path = QPainterPath()
            path.moveTo(cx - mouth_width//3, mouth_y + 10)
            path.quadTo(cx, mouth_y - 5, cx + mouth_width//3, mouth_y + 10)
            painter.drawPath(path)
        else:
            painter.drawLine(cx - mouth_width//3, mouth_y, cx + mouth_width//3, mouth_y)
    
    # =========================================================================
    # CUBE AVATAR
    # =========================================================================
    
    def generate_cube_set(self, output_dir: str):
        """Generate a full set of cube emotions."""
        for emotion in self.EMOTIONS:
            self.generate_cube(f"{output_dir}/{emotion}.png", emotion)
    
    def generate_cube(self, output_path: str, emotion: str = "neutral"):
        """Generate an abstract cube avatar."""
        pixmap = QPixmap(self.size, self.size)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        cx, cy = self.size // 2, self.size // 2
        cube_size = self.size // 2
        
        # Isometric cube
        # Colors based on emotion
        colors = {
            "neutral": (QColor(100, 150, 200), QColor(80, 130, 180), QColor(60, 110, 160)),
            "happy": (QColor(255, 220, 100), QColor(240, 200, 80), QColor(220, 180, 60)),
            "sad": (QColor(130, 150, 180), QColor(110, 130, 160), QColor(90, 110, 140)),
            "angry": (QColor(255, 120, 100), QColor(230, 100, 80), QColor(200, 80, 60)),
            "surprised": (QColor(200, 150, 255), QColor(180, 130, 235), QColor(160, 110, 215)),
            "thinking": (QColor(150, 200, 150), QColor(130, 180, 130), QColor(110, 160, 110)),
            "confused": (QColor(200, 180, 160), QColor(180, 160, 140), QColor(160, 140, 120)),
            "excited": (QColor(255, 180, 200), QColor(240, 160, 180), QColor(220, 140, 160)),
        }
        
        top_color, left_color, right_color = colors.get(emotion, colors["neutral"])
        
        # Calculate isometric points
        top = cy - cube_size // 2
        middle = cy + cube_size // 4
        bottom = cy + cube_size
        
        left = cx - cube_size
        right = cx + cube_size
        
        # Top face
        painter.setBrush(top_color)
        painter.setPen(QPen(QColor(50, 50, 50), 2))
        top_face = QPainterPath()
        top_face.moveTo(cx, top)
        top_face.lineTo(right, middle - cube_size//2)
        top_face.lineTo(cx, middle)
        top_face.lineTo(left, middle - cube_size//2)
        top_face.closeSubpath()
        painter.drawPath(top_face)
        
        # Left face
        painter.setBrush(left_color)
        left_face = QPainterPath()
        left_face.moveTo(cx, middle)
        left_face.lineTo(left, middle - cube_size//2)
        left_face.lineTo(left, bottom - cube_size//2)
        left_face.lineTo(cx, bottom)
        left_face.closeSubpath()
        painter.drawPath(left_face)
        
        # Right face
        painter.setBrush(right_color)
        right_face = QPainterPath()
        right_face.moveTo(cx, middle)
        right_face.lineTo(right, middle - cube_size//2)
        right_face.lineTo(right, bottom - cube_size//2)
        right_face.lineTo(cx, bottom)
        right_face.closeSubpath()
        painter.drawPath(right_face)
        
        # Face on front (eyes and mouth on right face)
        face_cx = cx + cube_size // 3
        face_cy = middle + cube_size // 4
        
        # Eyes
        painter.setBrush(QColor(255, 255, 255))
        painter.setPen(QPen(QColor(30, 30, 30), 2))
        eye_size = cube_size // 8
        
        if emotion == "happy":
            # Happy curve eyes
            painter.setPen(QPen(QColor(30, 30, 30), 3))
            painter.setBrush(Qt.NoBrush)
            painter.drawArc(face_cx - 25 - eye_size//2, face_cy - 25 - eye_size//2,
                          eye_size, eye_size, 0, 180 * 16)
            painter.drawArc(face_cx + 5 - eye_size//2, face_cy - 20 - eye_size//2,
                          eye_size, eye_size, 0, 180 * 16)
        else:
            painter.drawEllipse(face_cx - 25 - eye_size//2, face_cy - 25 - eye_size//2, eye_size, eye_size)
            painter.drawEllipse(face_cx + 5 - eye_size//2, face_cy - 20 - eye_size//2, eye_size, eye_size)
            
            # Pupils
            pupil_size = eye_size // 2
            painter.setBrush(QColor(30, 30, 30))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(face_cx - 25 - pupil_size//2, face_cy - 25 - pupil_size//2, pupil_size, pupil_size)
            painter.drawEllipse(face_cx + 5 - pupil_size//2, face_cy - 20 - pupil_size//2, pupil_size, pupil_size)
        
        # Mouth
        painter.setPen(QPen(QColor(30, 30, 30), 3))
        mouth_y = face_cy + 5
        
        if emotion in ["happy", "excited"]:
            path = QPainterPath()
            path.moveTo(face_cx - 20, mouth_y)
            path.quadTo(face_cx - 5, mouth_y + 15, face_cx + 10, mouth_y + 5)
            painter.drawPath(path)
        elif emotion == "sad":
            path = QPainterPath()
            path.moveTo(face_cx - 15, mouth_y + 10)
            path.quadTo(face_cx - 5, mouth_y - 5, face_cx + 10, mouth_y + 5)
            painter.drawPath(path)
        else:
            painter.drawLine(face_cx - 15, mouth_y, face_cx + 10, mouth_y + 5)
        
        painter.end()
        pixmap.save(output_path)


def generate_sample_avatars(output_dir: Optional[str] = None) -> dict[str, Path]:
    """
    Generate all sample avatars.
    
    Args:
        output_dir: Output directory (default: data/avatar/samples/)
        
    Returns:
        Dict mapping avatar name to directory path
    """
    if output_dir is None:
        from ..config import CONFIG
        output_dir = Path(CONFIG["data_dir"]) / "avatar" / "samples"
    
    generator = SampleAvatarGenerator()
    return generator.generate_all(str(output_dir))


if __name__ == "__main__":
    # Test generation
    import sys
    app = QApplication(sys.argv)
    
    generator = SampleAvatarGenerator()
    output = Path("data/avatar/samples")
    avatars = generator.generate_all(str(output))
    
    print(f"Generated {len(avatars)} avatar sets:")
    for name, path in avatars.items():
        print(f"  - {name}: {path}")
