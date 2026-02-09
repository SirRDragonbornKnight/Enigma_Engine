"""
Vision Integration Example for Enigma AI Engine

This example shows how to use Enigma AI Engine's vision capabilities.
The AI can "see" screens, images, and analyze visual content.

CAPABILITIES:
- Screen capture and analysis
- Image description and understanding
- Object/text detection
- Face detection and tracking
- Visual change monitoring

USAGE:
    python examples/vision_example.py
    
Or import in your own code:
    from examples.vision_example import capture_and_analyze
"""

import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple


# =============================================================================
# SCREEN CAPTURE
# =============================================================================

class ScreenVision:
    """
    Capture and analyze screen content.
    
    Works on:
        - Linux (X11, Wayland, Raspberry Pi)
        - Windows
        - macOS
    """
    
    def __init__(self):
        self._backend = None
        self._last_capture = None
        self._detect_backend()
    
    def _detect_backend(self):
        """Find best available screenshot method."""
        import platform
        import shutil
        
        system = platform.system()
        
        # Linux - prefer CLI tools (work better on Pi/Wayland)
        if system == "Linux":
            if shutil.which("scrot"):
                self._backend = "scrot"
                return
            elif shutil.which("gnome-screenshot"):
                self._backend = "gnome-screenshot"
                return
            elif shutil.which("import"):
                self._backend = "imagemagick"
                return
        
        # macOS
        if system == "Darwin":
            self._backend = "screencapture"
            return
        
        # Windows/Fallback - try Python libraries
        try:
            import mss
            self._backend = "mss"
            return
        except ImportError:
            pass
        
        try:
            from PIL import ImageGrab
            self._backend = "pillow"
            return
        except ImportError:
            pass
        
        print("[VISION] WARNING: No screenshot backend available!")
        print("  Install one of: scrot, mss, pillow")
        self._backend = None
    
    def capture(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[Any]:
        """
        Capture screenshot.
        
        Args:
            region: Optional (x, y, width, height) to capture specific area
        
        Returns:
            PIL Image or None if capture failed
        """
        if not self._backend:
            print("[VISION] No screenshot backend available")
            return None
        
        try:
            if self._backend == "scrot":
                return self._capture_scrot(region)
            elif self._backend == "mss":
                return self._capture_mss(region)
            elif self._backend == "pillow":
                return self._capture_pillow(region)
            elif self._backend == "screencapture":
                return self._capture_macos(region)
            else:
                return self._capture_mss(region)
        except Exception as e:
            print(f"[VISION] Capture failed: {e}")
            return None
    
    def _capture_scrot(self, region):
        """Capture using scrot (Linux)."""
        import subprocess
        import tempfile
        from PIL import Image
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            tmp_path = f.name
        
        try:
            if region:
                x, y, w, h = region
                subprocess.run(
                    ['scrot', '-a', f'{x},{y},{w},{h}', tmp_path],
                    check=True, capture_output=True, timeout=10
                )
            else:
                subprocess.run(['scrot', tmp_path], check=True, capture_output=True, timeout=10)
            
            img = Image.open(tmp_path)
            self._last_capture = img.copy()
            return self._last_capture
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def _capture_mss(self, region):
        """Capture using mss (cross-platform)."""
        import mss
        from PIL import Image
        
        with mss.mss() as sct:
            if region:
                monitor = {
                    "left": region[0],
                    "top": region[1],
                    "width": region[2],
                    "height": region[3]
                }
            else:
                monitor = sct.monitors[1]  # Primary monitor
            
            screenshot = sct.grab(monitor)
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            self._last_capture = img
            return img
    
    def _capture_pillow(self, region):
        """Capture using PIL (Windows/some Linux)."""
        from PIL import ImageGrab
        
        if region:
            x, y, w, h = region
            bbox = (x, y, x + w, y + h)
            img = ImageGrab.grab(bbox=bbox)
        else:
            img = ImageGrab.grab()
        
        self._last_capture = img
        return img
    
    def _capture_macos(self, region):
        """Capture using screencapture (macOS)."""
        import subprocess
        import tempfile
        from PIL import Image
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            tmp_path = f.name
        
        try:
            if region:
                x, y, w, h = region
                subprocess.run(
                    ['screencapture', '-x', '-R', f'{x},{y},{w},{h}', tmp_path],
                    check=True, timeout=10
                )
            else:
                subprocess.run(['screencapture', '-x', tmp_path], check=True, timeout=10)
            
            img = Image.open(tmp_path)
            self._last_capture = img.copy()
            return self._last_capture
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def save(self, path: str, image=None):
        """Save screenshot to file."""
        img = image or self._last_capture
        if img:
            img.save(path)
            print(f"[VISION] Saved to {path}")
            return True
        return False


# =============================================================================
# IMAGE ANALYSIS
# =============================================================================

class ImageAnalyzer:
    """
    Analyze images using various methods.
    
    Capabilities:
        - Text extraction (OCR)
        - Face detection
        - Object detection (if models available)
        - Color analysis
        - Image comparison
    """
    
    def __init__(self):
        self._ocr_available = self._check_ocr()
        self._face_cascade = None
    
    def _check_ocr(self) -> bool:
        """Check if OCR is available."""
        try:
            import pytesseract
            return True
        except ImportError:
            return False
    
    def extract_text(self, image) -> str:
        """
        Extract text from image using OCR.
        
        Args:
            image: PIL Image or path to image file
        
        Returns:
            Extracted text string
        """
        try:
            import pytesseract
            from PIL import Image
            
            if isinstance(image, (str, Path)):
                image = Image.open(image)
            
            text = pytesseract.image_to_string(image)
            return text.strip()
            
        except ImportError:
            print("[VISION] OCR requires pytesseract:")
            print("  pip install pytesseract")
            print("  Also install Tesseract: sudo apt install tesseract-ocr")
            return ""
        except Exception as e:
            print(f"[VISION] OCR failed: {e}")
            return ""
    
    def detect_faces(self, image) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image.
        
        Args:
            image: PIL Image, numpy array, or path
        
        Returns:
            List of (x, y, width, height) tuples for each face
        """
        try:
            import cv2
            import numpy as np
            from PIL import Image
            
            # Load face cascade
            if self._face_cascade is None:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self._face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Convert to numpy array if needed
            if isinstance(image, (str, Path)):
                image = Image.open(image)
            
            if hasattr(image, 'convert'):  # PIL Image
                img_array = np.array(image.convert('RGB'))
                img_array = img_array[:, :, ::-1]  # RGB to BGR
            else:
                img_array = image
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self._face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            return [tuple(f) for f in faces]
            
        except ImportError:
            print("[VISION] Face detection requires opencv:")
            print("  pip install opencv-python")
            return []
        except Exception as e:
            print(f"[VISION] Face detection failed: {e}")
            return []
    
    def get_face_center(self, image) -> Optional[Tuple[float, float]]:
        """
        Get normalized center of first detected face.
        
        Returns:
            (x, y) normalized to -1 to 1, or None if no face
        """
        faces = self.detect_faces(image)
        if not faces:
            return None
        
        from PIL import Image
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        img_w, img_h = image.size
        x, y, w, h = faces[0]
        
        # Face center
        cx = x + w / 2
        cy = y + h / 2
        
        # Normalize to -1 to 1
        nx = (cx / img_w) * 2 - 1
        ny = (cy / img_h) * 2 - 1
        
        return (nx, ny)
    
    def analyze_colors(self, image, num_colors: int = 5) -> List[Tuple[int, int, int]]:
        """
        Get dominant colors in image.
        
        Args:
            image: PIL Image or path
            num_colors: Number of colors to return
        
        Returns:
            List of (R, G, B) tuples
        """
        try:
            from PIL import Image
            from collections import Counter
            
            if isinstance(image, (str, Path)):
                image = Image.open(image)
            
            # Resize for speed
            small = image.copy()
            small.thumbnail((100, 100))
            
            # Get pixels
            pixels = list(small.getdata())
            
            # Quantize to reduce colors
            quantized = [(r // 32 * 32, g // 32 * 32, b // 32 * 32) 
                        for r, g, b in pixels if len((r, g, b)) == 3]
            
            # Count and return top colors
            counter = Counter(quantized)
            return [color for color, _ in counter.most_common(num_colors)]
            
        except Exception as e:
            print(f"[VISION] Color analysis failed: {e}")
            return []
    
    def compare_images(self, img1, img2) -> float:
        """
        Compare two images for similarity.
        
        Returns:
            Similarity score 0-1 (1 = identical)
        """
        try:
            from PIL import Image
            import hashlib
            
            # Load images
            if isinstance(img1, (str, Path)):
                img1 = Image.open(img1)
            if isinstance(img2, (str, Path)):
                img2 = Image.open(img2)
            
            # Resize to same size
            size = (64, 64)
            img1 = img1.resize(size).convert('L')
            img2 = img2.resize(size).convert('L')
            
            # Compare pixels
            pixels1 = list(img1.getdata())
            pixels2 = list(img2.getdata())
            
            diff = sum(abs(p1 - p2) for p1, p2 in zip(pixels1, pixels2))
            max_diff = 255 * len(pixels1)
            
            similarity = 1 - (diff / max_diff)
            return similarity
            
        except Exception as e:
            print(f"[VISION] Comparison failed: {e}")
            return 0.0


# =============================================================================
# CHANGE DETECTION
# =============================================================================

class ScreenMonitor:
    """
    Monitor screen for changes.
    
    Useful for:
        - Detecting when something happens on screen
        - Waiting for UI elements to appear
        - Game state detection
    """
    
    def __init__(self):
        self.vision = ScreenVision()
        self.analyzer = ImageAnalyzer()
        self._baseline = None
    
    def set_baseline(self, region=None):
        """Capture baseline image for comparison."""
        self._baseline = self.vision.capture(region)
        return self._baseline is not None
    
    def check_for_change(self, threshold: float = 0.95, region=None) -> bool:
        """
        Check if screen has changed from baseline.
        
        Args:
            threshold: Similarity threshold (below = changed)
            region: Screen region to check
        
        Returns:
            True if changed significantly
        """
        if self._baseline is None:
            print("[VISION] No baseline set - call set_baseline() first")
            return False
        
        current = self.vision.capture(region)
        if current is None:
            return False
        
        similarity = self.analyzer.compare_images(self._baseline, current)
        return similarity < threshold
    
    def wait_for_change(self, timeout: float = 30, check_interval: float = 0.5,
                        threshold: float = 0.95, region=None) -> bool:
        """
        Wait until screen changes.
        
        Args:
            timeout: Maximum seconds to wait
            check_interval: Seconds between checks
            threshold: Change threshold
            region: Screen region to monitor
        
        Returns:
            True if change detected, False if timeout
        """
        self.set_baseline(region)
        
        start = time.time()
        while time.time() - start < timeout:
            time.sleep(check_interval)
            if self.check_for_change(threshold, region):
                return True
        
        return False
    
    def wait_for_text(self, text: str, timeout: float = 30, 
                      check_interval: float = 1.0, region=None) -> bool:
        """
        Wait for specific text to appear on screen.
        
        Args:
            text: Text to look for
            timeout: Maximum seconds to wait
            check_interval: Seconds between checks
            region: Screen region to check
        
        Returns:
            True if text found, False if timeout
        """
        start = time.time()
        while time.time() - start < timeout:
            capture = self.vision.capture(region)
            if capture:
                screen_text = self.analyzer.extract_text(capture)
                if text.lower() in screen_text.lower():
                    return True
            time.sleep(check_interval)
        
        return False


# =============================================================================
# INTEGRATION WITH Enigma AI Engine
# =============================================================================

def capture_and_analyze(save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Capture screen and run full analysis.
    
    Args:
        save_path: Optional path to save screenshot
    
    Returns:
        Dictionary with analysis results
    """
    vision = ScreenVision()
    analyzer = ImageAnalyzer()
    
    # Capture
    image = vision.capture()
    if image is None:
        return {"success": False, "error": "Capture failed"}
    
    results = {
        "success": True,
        "size": image.size,
        "timestamp": time.time(),
    }
    
    # Save if requested
    if save_path:
        vision.save(save_path, image)
        results["saved_to"] = save_path
    
    # Extract text
    text = analyzer.extract_text(image)
    if text:
        results["text"] = text[:1000]  # Limit length
        results["text_length"] = len(text)
    
    # Detect faces
    faces = analyzer.detect_faces(image)
    results["faces_found"] = len(faces)
    if faces:
        results["face_locations"] = faces
    
    # Analyze colors
    colors = analyzer.analyze_colors(image)
    results["dominant_colors"] = colors
    
    return results


def track_face_for_robot() -> Optional[Tuple[float, float]]:
    """
    Get face position for robot/animatronic tracking.
    
    Returns:
        (x, y) normalized to -1 to 1, or None if no face
        
    Example with robot:
        from examples.robot_example import create_animatronic
        
        robot = create_animatronic()
        robot.connect()
        
        while True:
            pos = track_face_for_robot()
            if pos:
                robot.look_at(pos[0], pos[1])
            time.sleep(0.1)
    """
    vision = ScreenVision()
    analyzer = ImageAnalyzer()
    
    image = vision.capture()
    if image is None:
        return None
    
    return analyzer.get_face_center(image)


# =============================================================================
# MAIN - Run this file directly to test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Enigma AI Engine Vision Example")
    print("=" * 60)
    
    # Test screen capture
    print("\n[1] Testing screen capture...")
    vision = ScreenVision()
    print(f"Backend: {vision._backend}")
    
    image = vision.capture()
    if image:
        print(f"Captured: {image.size}")
        
        # Save screenshot
        output_dir = Path.home() / ".enigma_engine" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "vision_test.png"
        vision.save(str(save_path))
    else:
        print("Capture failed!")
    
    # Test analysis
    print("\n[2] Testing image analysis...")
    analyzer = ImageAnalyzer()
    
    if image:
        # Face detection
        faces = analyzer.detect_faces(image)
        print(f"Faces detected: {len(faces)}")
        
        # Color analysis
        colors = analyzer.analyze_colors(image)
        print(f"Dominant colors: {colors[:3]}")
        
        # OCR
        print("\n[3] Testing OCR (may take a moment)...")
        text = analyzer.extract_text(image)
        if text:
            preview = text[:200].replace('\n', ' ')
            print(f"Text found: {preview}...")
        else:
            print("No text extracted (OCR may not be installed)")
    
    # Test change detection
    print("\n[4] Testing change detection...")
    monitor = ScreenMonitor()
    monitor.set_baseline()
    print("Baseline set. Move something on screen...")
    
    changed = monitor.check_for_change()
    print(f"Changed: {changed}")
    
    # Full analysis
    print("\n[5] Running full analysis...")
    results = capture_and_analyze()
    print(f"Results: {list(results.keys())}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nInstall optional dependencies for full features:")
    print("  pip install pytesseract opencv-python mss pillow")
    print("  sudo apt install tesseract-ocr scrot  # Linux")
