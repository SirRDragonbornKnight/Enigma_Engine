"""
Screen Vision System

Allows the AI to "see" what's on screen:
  - Capture screenshots
  - Analyze screen content
  - Track changes
  - Identify UI elements

Works on: Desktop (Windows/Mac/Linux), Mobile (limited)
"""

from __future__ import annotations

import base64
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CAPTURE_MAX_SIZE = 1024
MAX_VISION_HISTORY = 100
COMPARISON_THUMBNAIL_SIZE = (200, 200)
MAX_SCREEN_CONTEXT_TEXT = 1000



class ScreenCapture:
    """
    Cross-platform screen capture.
    Prioritizes scrot on Linux (works on Wayland/Pi).
    """
    
    def __init__(self):
        self.last_capture = None
        self.last_capture_time = None
        self._backend = self._detect_backend()
    
    def _detect_backend(self) -> str:
        """Detect best available screenshot backend. Prioritize CLI tools on Linux."""
        import platform
        import shutil
        
        # On Linux, prefer scrot first (works on Wayland, X11, Pi)
        if platform.system() == "Linux":
            if shutil.which("scrot"):
                return "scrot"
            elif shutil.which("gnome-screenshot"):
                return "gnome-screenshot"
            elif shutil.which("import"):  # ImageMagick
                return "imagemagick"
        
        # macOS built-in
        if platform.system() == "Darwin":
            return "screencapture"
        
        # Fallback to Python libraries (may fail on Wayland)
        try:
            from PIL import ImageGrab
            ImageGrab.grab()
            return "pillow"
        except ImportError:
            logger.debug("PIL ImageGrab not available")
        except OSError as e:
            logger.debug(f"PIL ImageGrab failed (Wayland?): {e}")
        except Exception as e:
            logger.debug(f"PIL ImageGrab test failed: {e}")
        
        try:
            return "mss"
        except ImportError:
            pass
        
        try:
            return "pyautogui"
        except ImportError:
            pass
        
        try:
            return "pyscreenshot"
        except ImportError:
            pass
        
        return "none"
    
    def capture(self, region: Tuple[int, int, int, int] = None) -> Optional[Any]:
        """
        Capture the screen or a region.
        
        Args:
            region: Optional (x, y, width, height) to capture
            
        Returns:
            PIL Image or None
        """
        from PIL import Image
        
        img = None
        
        try:
            if self._backend == "pillow":
                from PIL import ImageGrab
                if region:
                    bbox = (region[0], region[1], region[0]+region[2], region[1]+region[3])
                    img = ImageGrab.grab(bbox=bbox)
                else:
                    img = ImageGrab.grab()
            
            elif self._backend == "mss":
                import mss
                with mss.mss() as sct:
                    if region:
                        monitor = {"left": region[0], "top": region[1], 
                                   "width": region[2], "height": region[3]}
                    else:
                        monitor = sct.monitors[1]  # Primary monitor
                    screenshot = sct.grab(monitor)
                    img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            
            elif self._backend == "pyautogui":
                import pyautogui
                if region:
                    img = pyautogui.screenshot(region=region)
                else:
                    img = pyautogui.screenshot()
            
            elif self._backend == "pyscreenshot":
                import pyscreenshot
                if region:
                    bbox = (region[0], region[1], region[0]+region[2], region[1]+region[3])
                    img = pyscreenshot.grab(bbox=bbox)
                else:
                    img = pyscreenshot.grab()
            
            elif self._backend in ["scrot", "gnome-screenshot", "imagemagick", "screencapture"]:
                img = self._capture_cli(region)
            
        except Exception as e:
            # Silently fail, return None
            return None
        
        if img:
            self.last_capture = img
            self.last_capture_time = datetime.now()
        
        return img
    
    def _capture_cli(self, region: Tuple = None) -> Optional[Any]:
        """Capture using command line tools."""
        import subprocess
        import tempfile
        from PIL import Image
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
        
        try:
            if self._backend == "scrot":
                cmd = ["scrot", "-o", temp_path]
            elif self._backend == "gnome-screenshot":
                cmd = ["gnome-screenshot", "-f", temp_path]
            elif self._backend == "imagemagick":
                cmd = ["import", "-window", "root", temp_path]
            elif self._backend == "screencapture":
                cmd = ["screencapture", "-x", temp_path]
            else:
                return None
            
            subprocess.run(cmd, capture_output=True, timeout=10)
            
            img = Image.open(temp_path)
            
            if region:
                bbox = (region[0], region[1], region[0]+region[2], region[1]+region[3])
                img = img.crop(bbox)
            
            return img
            
        except Exception as e:
            return None  # Silently fail
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def capture_to_base64(self, region: Tuple = None, max_size: int = 1024) -> Optional[str]:
        """
        Capture and return as base64 string (for API/AI).
        
        Args:
            region: Optional region to capture
            max_size: Max dimension (will resize if larger)
            
        Returns:
            Base64-encoded PNG string
        """
        img = self.capture(region)
        if img is None:
            return None
        
        # Resize if too large
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    def save(self, path: str, region: Tuple = None) -> bool:
        """Save screenshot to file."""
        img = self.capture(region)
        if img:
            img.save(path)
            return True
        return False


class ScreenVision:
    """
    AI screen understanding system.
    
    Provides:
      - Screenshot capture
      - Screen description (via vision model or OCR)
      - Change detection
      - Element identification
    """
    
    def __init__(self):
        self.capture = ScreenCapture()
        self.history: List[Dict] = []
        self._ocr = self._init_ocr()
        self._ocr_available = self._ocr is not None
    
    def _init_ocr(self):
        """Initialize OCR system (prefers built-in, falls back to tesseract)."""
        try:
            from .simple_ocr import AdvancedOCR
            return AdvancedOCR()
        except ImportError:
            pass
        
        # Fallback to tesseract check
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return "tesseract"
        except (ImportError, RuntimeError, OSError):
            return None
    
    def see(self, describe: bool = True, detect_text: bool = True) -> Dict[str, Any]:
        """
        Look at the screen and return what's visible.
        
        Args:
            describe: Include basic description
            detect_text: Run OCR to extract text
            
        Returns:
            Dict with screen information
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "image_base64": None,
            "text_content": None,
            "description": None,
            "size": None,
        }
        
        # Capture screen
        img = self.capture.capture()
        if img is None:
            result["error"] = "Failed to capture screen"
            return result
        
        result["success"] = True
        result["size"] = {"width": img.size[0], "height": img.size[1]}
        result["image_base64"] = self.capture.capture_to_base64(max_size=DEFAULT_CAPTURE_MAX_SIZE)
        
        # OCR for text
        if detect_text and self._ocr_available:
            try:
                if self._ocr == "tesseract":
                    import pytesseract
                    text = pytesseract.image_to_string(img)
                else:
                    text = self._ocr.extract_text(img)
                result["text_content"] = text.strip()
            except Exception as e:
                result["ocr_error"] = str(e)
        
        # Basic description
        if describe:
            result["description"] = self._describe_screen(img, result.get("text_content"))
        
        # Store in history
        self.history.append({
            "timestamp": result["timestamp"],
            "size": result["size"],
            "has_text": bool(result.get("text_content")),
        })
        
        # Keep history limited
        if len(self.history) > MAX_VISION_HISTORY:
            self.history = self.history[-MAX_VISION_HISTORY:]
        
        return result
    
    def _describe_screen(self, img, text_content: str = None) -> str:
        """Generate a basic screen description."""
        width, height = img.size
        
        # Analyze image colors/brightness
        try:
            import numpy as np
            arr = np.array(img)
            brightness = arr.mean()
            is_dark = brightness < 128
        except ImportError:
            is_dark = False
        
        desc_parts = [
            f"Screen resolution: {width}x{height}",
            f"Theme: {'dark' if is_dark else 'light'}",
        ]
        
        if text_content:
            # Summarize visible text
            lines = [l.strip() for l in text_content.split("\n") if l.strip()]
            if lines:
                desc_parts.append(f"Visible text elements: {len(lines)} lines")
                # Show first few lines
                preview = lines[:5]
                desc_parts.append(f"Text preview: {', '.join(preview[:3])}")
        
        return "; ".join(desc_parts)
    
    def detect_changes(self, threshold: float = 0.1) -> Dict[str, Any]:
        """
        Detect if screen has changed since last capture.
        
        Args:
            threshold: Difference threshold (0-1)
            
        Returns:
            Dict with change information
        """
        if self.capture.last_capture is None:
            # First capture
            self.see(describe=False, detect_text=False)
            return {"changed": True, "reason": "first_capture"}
        
        old_img = self.capture.last_capture
        new_img = self.capture.capture()
        
        if new_img is None:
            return {"changed": False, "error": "capture_failed"}
        
        # Compare images
        try:
            import numpy as np
            
            # Resize for faster comparison
            old_small = old_img.resize(COMPARISON_THUMBNAIL_SIZE)
            new_small = new_img.resize(COMPARISON_THUMBNAIL_SIZE)
            
            old_arr = np.array(old_small, dtype=float)
            new_arr = np.array(new_small, dtype=float)
            
            diff = np.abs(old_arr - new_arr).mean() / 255.0
            changed = diff > threshold
            
            return {
                "changed": changed,
                "difference": round(diff, 3),
                "threshold": threshold,
            }
        except (ImportError, ValueError, TypeError):
            return {"changed": True, "reason": "comparison_failed"}
    
    def find_text_on_screen(self, search_text: str) -> List[Dict]:
        """
        Find text on screen and return locations.
        
        Args:
            search_text: Text to find
            
        Returns:
            List of matches with locations
        """
        if not self._ocr_available:
            return [{"error": "OCR not available"}]
        
        img = self.capture.capture()
        if img is None:
            return []
        
        try:
            import pytesseract
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            
            matches = []
            search_lower = search_text.lower()
            
            for i, text in enumerate(data['text']):
                if search_lower in str(text).lower():
                    matches.append({
                        "text": text,
                        "x": data['left'][i],
                        "y": data['top'][i],
                        "width": data['width'][i],
                        "height": data['height'][i],
                        "confidence": data['conf'][i],
                    })
            
            return matches
        except Exception as e:
            return [{"error": str(e)}]
    
    def get_screen_context(self) -> str:
        """
        Get a text description of current screen for AI context.
        """
        vision_data = self.see(describe=True, detect_text=True)
        
        if not vision_data["success"]:
            return "Unable to see screen."
        
        parts = [
            "=== CURRENT SCREEN STATE ===",
            f"Resolution: {vision_data['size']['width']}x{vision_data['size']['height']}",
        ]
        
        if vision_data.get("description"):
            parts.append(f"Description: {vision_data['description']}")
        
        if vision_data.get("text_content"):
            # Truncate if too long
            text = vision_data["text_content"]
            if len(text) > MAX_SCREEN_CONTEXT_TEXT:
                text = text[:MAX_SCREEN_CONTEXT_TEXT] + "...[truncated]"
            parts.append(f"Visible text:\n{text}")
        
        return "\n".join(parts)


# Singleton instance
_screen_vision: Optional[ScreenVision] = None


def get_screen_vision() -> ScreenVision:
    """Get or create screen vision singleton."""
    global _screen_vision
    if _screen_vision is None:
        _screen_vision = ScreenVision()
    return _screen_vision


# Tool integration
from .tool_registry import Tool


class ScreenVisionTool(Tool):
    """Tool wrapper for AI to use screen vision."""
    
    name = "see_screen"
    description = "Look at the screen and describe what's visible"
    parameters = {
        "detect_text": "Whether to extract text via OCR (default: True)"
    }
    
    def execute(self, detect_text: bool = True, **kwargs) -> Dict[str, Any]:
        vision = get_screen_vision()
        return vision.see(describe=True, detect_text=detect_text)


class FindOnScreenTool(Tool):
    """Tool to find specific text on screen."""
    
    name = "find_on_screen"
    description = "Find text or element on the screen"
    parameters = {
        "text": "Text to search for on screen (required)"
    }
    
    def execute(self, text: str = "", **kwargs) -> Dict[str, Any]:
        vision = get_screen_vision()
        matches = vision.find_text_on_screen(text)
        return {
            "success": True,
            "query": text,
            "found": len(matches) > 0,
            "matches": matches
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def capture_screen(save_path: str = None) -> Dict[str, Any]:
    """
    Quick function to capture the screen.
    
    Args:
        save_path: Optional path to save the screenshot.
                   If None, saves to outputs/images with timestamp.
    
    Returns:
        Dict with success status and path.
    """
    import time
    from ..config import CONFIG
    
    try:
        vision = get_screen_vision()
        
        if save_path is None:
            output_dir = Path(CONFIG.get("outputs_dir", "outputs")) / "images"
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            save_path = str(output_dir / f"screenshot_{timestamp}.png")
        
        if vision.capture.save(save_path):
            return {
                "success": True,
                "path": save_path,
                "message": f"Screenshot saved to {save_path}"
            }
        else:
            return {
                "success": False,
                "error": "Failed to capture screen"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
