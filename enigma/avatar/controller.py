"""
Avatar System for Enigma

A controllable 3D avatar that can:
  - Display on screen with customizable appearance
  - Move around the desktop
  - Animate expressions and gestures
  - "Interact" with windows/files (visual effects)
  - Be turned on/off

DEFAULT STATE: OFF (requires explicit enable)

The avatar system is designed to be modular:
  - AvatarController: Main interface (on/off, commands)
  - AvatarRenderer: Handles display (can swap backends)
  - AvatarAnimator: Handles movements and expressions
  - ScreenInteractor: Visual effects for "touching" windows
"""

import ctypes
import json
import re
import subprocess
import time
import threading
from pathlib import Path
from typing import Dict, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum

from ..config import CONFIG


class AvatarState(Enum):
    """Avatar states."""
    OFF = "off"
    IDLE = "idle"
    SPEAKING = "speaking"
    THINKING = "thinking"
    MOVING = "moving"
    INTERACTING = "interacting"


@dataclass
class AvatarPosition:
    """Avatar position on screen."""
    x: int = 100
    y: int = 100
    width: int = 200
    height: int = 300
    screen: int = 0  # For multi-monitor


@dataclass
class AvatarConfig:
    """Avatar configuration."""
    enabled: bool = False  # DEFAULT OFF
    model_path: str = ""  # Path to 3D model
    scale: float = 1.0
    opacity: float = 1.0
    always_on_top: bool = True
    start_position: AvatarPosition = field(default_factory=AvatarPosition)
    idle_animation: bool = True
    follow_mouse: bool = False
    interaction_effects: bool = True


class AvatarController:
    """
    Main avatar controller.
    
    Manages avatar state, position, and interactions.
    
    Usage:
        avatar = AvatarController()
        avatar.enable()  # Turn on
        avatar.move_to(500, 300)  # Move
        avatar.speak("Hello!")  # Animate speaking
        avatar.interact_with_window("My File.txt")  # Visual effect
        avatar.disable()  # Turn off
    """
    
    def __init__(self, config: AvatarConfig = None):
        """
        Initialize avatar controller.
        
        Args:
            config: Avatar configuration (default: disabled)
        """
        self.config = config or AvatarConfig()
        self.state = AvatarState.OFF
        self.position = self.config.start_position
        
        # Components (lazy loaded)
        self._renderer: Optional['AvatarRenderer'] = None
        self._animator: Optional['AvatarAnimator'] = None
        self._interactor: Optional['ScreenInteractor'] = None
        
        # New components
        self._identity: Optional['AIAvatarIdentity'] = None
        self._emotion_sync: Optional['EmotionExpressionSync'] = None
        self._lip_sync: Optional['LipSync'] = None
        self._customizer: Optional['AvatarCustomizer'] = None
        
        # Event callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "state_change": [],
            "move": [],
            "speak": [],
            "interact": [],
        }
        
        # Animation queue
        self._animation_queue: List[Dict] = []
        self._animation_thread: Optional[threading.Thread] = None
        self._running = False
    
    @property
    def is_enabled(self) -> bool:
        """Check if avatar is enabled."""
        return self.state != AvatarState.OFF
    
    def enable(self) -> bool:
        """
        Enable the avatar (turn on).
        
        Returns:
            True if successfully enabled
        """
        if self.state != AvatarState.OFF:
            return True  # Already on
        
        try:
            # Initialize renderer
            self._init_renderer()
            
            # Start animation thread
            self._running = True
            self._animation_thread = threading.Thread(target=self._animation_loop, daemon=True)
            self._animation_thread.start()
            
            # Set to idle
            self._set_state(AvatarState.IDLE)
            self.config.enabled = True
            
            print("Avatar enabled")
            return True
            
        except Exception as e:
            print(f"Failed to enable avatar: {e}")
            return False
    
    def disable(self) -> None:
        """Disable the avatar (turn off)."""
        self._running = False
        
        if self._renderer:
            self._renderer.hide()
        
        # Stop emotion sync
        if self._emotion_sync:
            self._emotion_sync.stop_sync()
        
        self._set_state(AvatarState.OFF)
        self.config.enabled = False
        print("Avatar disabled")
    
    def toggle(self) -> bool:
        """Toggle avatar on/off. Returns new state (True=on)."""
        if self.is_enabled:
            self.disable()
            return False
        else:
            self.enable()
            return True
    
    def _set_state(self, state: AvatarState) -> None:
        """Set avatar state and notify callbacks."""
        old_state = self.state
        self.state = state
        
        for cb in self._callbacks["state_change"]:
            try:
                cb(old_state, state)
            except Exception as e:
                print(f"[Avatar] State callback error: {e}")
    
    def _init_renderer(self) -> None:
        """Initialize the renderer component."""
        if self._renderer is None:
            # Use SpriteRenderer by default (works everywhere)
            from .renderers import SpriteRenderer
            self._renderer = SpriteRenderer(self)
            
            # Set appearance if we have an identity
            if self._identity and self._identity.appearance:
                self._renderer.set_appearance(self._identity.appearance)
    
    def _animation_loop(self) -> None:
        """Background animation processing loop."""
        while self._running:
            if self._animation_queue:
                animation = self._animation_queue.pop(0)
                self._execute_animation(animation)
            elif self.state == AvatarState.IDLE and self.config.idle_animation:
                # Idle animation
                self._execute_animation({"type": "idle", "duration": 2.0})
            time.sleep(0.05)  # 20 FPS
    
    def _execute_animation(self, animation: Dict) -> None:
        """Execute a single animation."""
        anim_type = animation.get("type", "idle")
        duration = animation.get("duration", 1.0)
        on_complete = animation.get("on_complete")
        
        if self._animator:
            self._animator.play(anim_type, duration)
        
        # Wait for animation duration
        time.sleep(duration)
        
        # Call completion callback
        if on_complete and callable(on_complete):
            try:
                on_complete()
            except Exception as e:
                print(f"[Avatar] Animation callback error: {e}")
    
    # === Movement ===
    
    def move_to(self, x: int, y: int, animate: bool = True) -> None:
        """
        Move avatar to position.
        
        Args:
            x: Target X coordinate
            y: Target Y coordinate
            animate: Whether to animate the movement
        """
        if not self.is_enabled:
            return
        
        if animate:
            self._set_state(AvatarState.MOVING)
            self._animation_queue.append({
                "type": "move",
                "from": (self.position.x, self.position.y),
                "to": (x, y),
                "duration": 0.5,
                "on_complete": lambda: self._set_state(AvatarState.IDLE),
            })
        
        self.position.x = x
        self.position.y = y
        
        if self._renderer:
            self._renderer.set_position(x, y)
        
        for cb in self._callbacks["move"]:
            try:
                cb(x, y)
            except Exception as e:
                print(f"[Avatar] Move callback error: {e}")
    
    def move_relative(self, dx: int, dy: int) -> None:
        """Move avatar by offset."""
        self.move_to(self.position.x + dx, self.position.y + dy)
    
    def center_on_screen(self) -> None:
        """Center avatar on screen."""
        try:
            # Try to get screen dimensions
            import subprocess
            import platform
            
            width, height = 1920, 1080  # Default fallback
            
            if platform.system() == "Linux":
                result = subprocess.run(
                    ["xrandr", "--current"],
                    capture_output=True, text=True
                )
                import re
                match = re.search(r'current (\d+) x (\d+)', result.stdout)
                if match:
                    width, height = int(match.group(1)), int(match.group(2))
            elif platform.system() == "Windows":
                import ctypes
                user32 = ctypes.windll.user32
                width = user32.GetSystemMetrics(0)
                height = user32.GetSystemMetrics(1)
            
            x = (width - self.position.width) // 2
            y = (height - self.position.height) // 2
            self.move_to(x, y)
        except Exception as e:
            print(f"[Avatar] Could not detect screen size: {e}")
            self.move_to(400, 200)
    
    # === Speaking ===
    
    def speak(self, text: str, animate: bool = True) -> None:
        """
        Animate avatar speaking.
        
        Args:
            text: Text being spoken (affects animation duration)
            animate: Whether to animate mouth/expressions
        """
        if not self.is_enabled:
            return
        
        # Estimate speaking duration
        words = len(text.split())
        duration = max(1.0, words * 0.3)  # ~0.3 sec per word
        
        if animate:
            self._set_state(AvatarState.SPEAKING)
            self._animation_queue.append({
                "type": "speak",
                "text": text,
                "duration": duration,
            })
        
        for cb in self._callbacks["speak"]:
            try:
                cb(text)
            except Exception as e:
                print(f"[Avatar] Speak callback error: {e}")
    
    def think(self, duration: float = 2.0) -> None:
        """Show thinking animation."""
        if not self.is_enabled:
            return
        
        self._set_state(AvatarState.THINKING)
        self._animation_queue.append({
            "type": "think",
            "duration": duration,
        })
    
    # === Screen Interaction ===
    
    def interact_with_window(self, window_title: str, action: str = "touch") -> bool:
        """
        Visual effect of avatar interacting with a window.
        
        Args:
            window_title: Title of window to interact with
            action: Type of interaction ("touch", "drag", "point")
            
        Returns:
            True if window found and interaction animated
        """
        if not self.is_enabled:
            return False
        
        if self._interactor is None:
            self._interactor = ScreenInteractor()
        
        # Find window position
        window_pos = self._interactor.find_window(window_title)
        if window_pos is None:
            return False
        
        self._set_state(AvatarState.INTERACTING)
        
        # Move toward window
        target_x = window_pos["x"] - self.position.width
        target_y = window_pos["y"]
        self.move_to(target_x, target_y)
        
        # Animate interaction
        self._animation_queue.append({
            "type": f"interact_{action}",
            "target": window_pos,
            "duration": 1.0,
        })
        
        for cb in self._callbacks["interact"]:
            try:
                cb(window_title, action, window_pos)
            except Exception as e:
                print(f"[Avatar] Interact callback error: {e}")
        
        return True
    
    def interact_with_file(self, filepath: str, action: str = "touch") -> bool:
        """
        Visual effect of avatar interacting with a file.
        
        This looks up the file in a file manager window or on desktop.
        """
        # Try to find in file manager or desktop
        filename = Path(filepath).name
        return self.interact_with_window(filename, action)
    
    def point_at(self, x: int, y: int) -> None:
        """Make avatar point at screen location."""
        if not self.is_enabled:
            return
        
        self._animation_queue.append({
            "type": "point",
            "target": (x, y),
            "duration": 1.0,
        })
    
    # === Expressions ===
    
    def set_expression(self, expression: str) -> None:
        """
        Set avatar facial expression.
        
        Args:
            expression: One of: neutral, happy, sad, surprised, thinking, confused
        """
        if not self.is_enabled:
            return
        
        self._animation_queue.append({
            "type": "expression",
            "expression": expression,
            "duration": 0.5,
        })
    
    # === Customization ===
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a different 3D model for the avatar.
        
        Args:
            model_path: Path to 3D model file (.obj, .gltf, .fbx, etc.)
            
        Returns:
            True if model loaded successfully
        """
        path = Path(model_path)
        if not path.exists():
            print(f"Model not found: {model_path}")
            return False
        
        self.config.model_path = str(path)
        
        if self._renderer:
            self._renderer.load_model(model_path)
        
        print(f"Avatar model changed to: {path.name}")
        return True
    
    def set_scale(self, scale: float) -> None:
        """Set avatar size (1.0 = normal, 2.0 = double, 0.5 = half)."""
        self.config.scale = max(0.1, min(5.0, scale))
        if self._renderer:
            self._renderer.set_scale(self.config.scale)
    
    def set_opacity(self, opacity: float) -> None:
        """Set avatar transparency (0.0 = invisible, 1.0 = solid)."""
        self.config.opacity = max(0.0, min(1.0, opacity))
        if self._renderer:
            self._renderer.set_opacity(self.config.opacity)
    
    def set_color(self, color: str) -> None:
        """
        Set avatar color/tint.
        
        Args:
            color: Hex color like "#FF0000" or name like "blue"
        """
        if self._renderer:
            self._renderer.set_color(color)
    
    def list_available_models(self) -> List[str]:
        """List available avatar models in the avatars directory."""
        avatars_dir = Path(CONFIG.get("data_dir", "data")) / "avatars"
        if not avatars_dir.exists():
            avatars_dir.mkdir(parents=True, exist_ok=True)
            return []
        
        models = []
        for ext in [".obj", ".gltf", ".glb", ".fbx", ".dae"]:
            models.extend([f.name for f in avatars_dir.glob(f"*{ext}")])
        return models
    
    # === AI Identity & Personality Integration ===
    
    def link_personality(self, personality: 'AIPersonality') -> None:
        """
        Link avatar to AI personality for auto-sync.
        
        Args:
            personality: AIPersonality instance
        """
        from .avatar_identity import AIAvatarIdentity
        from .emotion_sync import EmotionExpressionSync
        
        self._identity = AIAvatarIdentity(personality)
        self._emotion_sync = EmotionExpressionSync(self, personality)
        self._emotion_sync.start_sync()
        
        print(f"[Avatar] Linked to personality: {personality.model_name}")
    
    def auto_design(self) -> Optional['AvatarAppearance']:
        """
        Let AI design its own appearance based on personality.
        
        Returns:
            AvatarAppearance designed by AI, or None if no personality linked
        """
        if self._identity:
            appearance = self._identity.design_from_personality()
            
            # Apply to renderer if enabled
            if self.is_enabled and self._renderer:
                self._renderer.set_appearance(appearance)
            
            print(f"[Avatar] AI designed appearance: {self._identity.reasoning}")
            return appearance
        else:
            print("[Avatar] No personality linked. Use link_personality() first.")
            return None
    
    def describe_desired_appearance(self, description: str) -> Optional['AvatarAppearance']:
        """
        AI describes desired appearance in natural language.
        
        Args:
            description: Natural language description
            
        Returns:
            AvatarAppearance based on description
        """
        from .avatar_identity import AIAvatarIdentity
        
        if not self._identity:
            self._identity = AIAvatarIdentity()
        
        appearance = self._identity.describe_desired_appearance(description)
        
        # Apply to renderer if enabled
        if self.is_enabled and self._renderer:
            self._renderer.set_appearance(appearance)
        
        print(f"[Avatar] Created appearance from: {description}")
        return appearance
    
    def get_customizer(self) -> 'AvatarCustomizer':
        """
        Get customizer for user modifications.
        
        Returns:
            AvatarCustomizer instance
        """
        if self._customizer is None:
            from .customizer import AvatarCustomizer
            self._customizer = AvatarCustomizer(self)
        return self._customizer
    
    def get_identity(self) -> Optional['AIAvatarIdentity']:
        """
        Get AI avatar identity.
        
        Returns:
            AIAvatarIdentity or None
        """
        return self._identity
    
    def set_appearance(self, appearance: 'AvatarAppearance') -> None:
        """
        Set avatar appearance directly.
        
        Args:
            appearance: AvatarAppearance to apply
        """
        from .avatar_identity import AIAvatarIdentity
        
        if not self._identity:
            self._identity = AIAvatarIdentity()
        
        self._identity.appearance = appearance
        
        # Apply to renderer if enabled
        if self.is_enabled and self._renderer:
            self._renderer.set_appearance(appearance)
        
        print("[Avatar] Appearance updated")
    
    def explain_appearance(self) -> str:
        """
        Get AI's explanation of its appearance choices.
        
        Returns:
            Explanation string
        """
        if self._identity:
            return self._identity.explain_appearance_choices()
        return "No appearance identity set."
    
    # === Event Registration ===
    
    def on(self, event: str, callback: Callable) -> None:
        """
        Register event callback.
        
        Events: state_change, move, speak, interact
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    # === Serialization ===
    
    def save_config(self, path: str = None) -> None:
        """Save avatar configuration."""
        path = path or str(Path(CONFIG.get("data_dir", "data")) / "avatar_config.json")
        
        config_dict = {
            "enabled": self.config.enabled,
            "model_path": self.config.model_path,
            "scale": self.config.scale,
            "opacity": self.config.opacity,
            "always_on_top": self.config.always_on_top,
            "idle_animation": self.config.idle_animation,
            "follow_mouse": self.config.follow_mouse,
            "interaction_effects": self.config.interaction_effects,
            "position": {
                "x": self.position.x,
                "y": self.position.y,
                "width": self.position.width,
                "height": self.position.height,
            }
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)
    
    def load_config(self, path: str = None) -> None:
        """Load avatar configuration."""
        path = path or str(Path(CONFIG.get("data_dir", "data")) / "avatar_config.json")
        
        if not Path(path).exists():
            return
        
        with open(path, "r") as f:
            config_dict = json.load(f)
        
        self.config.enabled = config_dict.get("enabled", False)
        self.config.model_path = config_dict.get("model_path", "")
        self.config.scale = config_dict.get("scale", 1.0)
        self.config.opacity = config_dict.get("opacity", 1.0)
        self.config.always_on_top = config_dict.get("always_on_top", True)
        self.config.idle_animation = config_dict.get("idle_animation", True)
        self.config.follow_mouse = config_dict.get("follow_mouse", False)
        self.config.interaction_effects = config_dict.get("interaction_effects", True)
        
        pos = config_dict.get("position", {})
        self.position.x = pos.get("x", 100)
        self.position.y = pos.get("y", 100)
        self.position.width = pos.get("width", 200)
        self.position.height = pos.get("height", 300)


class AvatarRenderer:
    """
    Renders the avatar on screen.
    
    This is a base/stub implementation. Can be replaced with:
      - PyQt5/PySide6 for 2D overlay
      - OpenGL/Pygame for 3D rendering
      - Web-based renderer
    """
    
    def __init__(self, controller: AvatarController):
        self.controller = controller
        self._window = None
        self._visible = False
    
    def show(self) -> None:
        """Show avatar window."""
        # Stub - implement with actual rendering
        self._visible = True
        print(f"[Avatar] Showing at ({self.controller.position.x}, {self.controller.position.y})")
    
    def hide(self) -> None:
        """Hide avatar window."""
        self._visible = False
        print("[Avatar] Hidden")
    
    def set_position(self, x: int, y: int) -> None:
        """Update avatar position."""
        if self._visible:
            print(f"[Avatar] Moving to ({x}, {y})")
    
    def render_frame(self, animation_data: Dict = None) -> None:
        """Render a single frame."""
        pass  # Implement with actual rendering
    
    def load_model(self, model_path: str) -> bool:
        """Load a 3D model file."""
        # Stub - implement with actual 3D rendering library
        print(f"[Avatar] Loading model: {model_path}")
        return True
    
    def set_scale(self, scale: float) -> None:
        """Set avatar scale."""
        print(f"[Avatar] Scale set to: {scale}")
    
    def set_opacity(self, opacity: float) -> None:
        """Set avatar opacity."""
        print(f"[Avatar] Opacity set to: {opacity}")
    
    def set_color(self, color: str) -> None:
        """Set avatar color/tint."""
        print(f"[Avatar] Color set to: {color}")


class AvatarAnimator:
    """
    Handles avatar animations.
    
    Animations include:
      - idle: Subtle breathing/movement
      - speak: Mouth movement synced to audio
      - think: Head tilt, looking up
      - move: Walking/floating animation
      - interact_*: Reaching, pointing, etc.
      - expression: Facial expressions
    """
    
    def __init__(self, controller: AvatarController):
        self.controller = controller
        self.current_animation = None
    
    def play(self, animation_type: str, duration: float = 1.0) -> None:
        """Play an animation."""
        self.current_animation = animation_type
        print(f"[Avatar] Animation: {animation_type} ({duration}s)")
    
    def stop(self) -> None:
        """Stop current animation."""
        self.current_animation = None


class ScreenInteractor:
    """
    Handles finding and interacting with screen elements.
    """
    
    def __init__(self):
        self._window_cache: Dict[str, Dict] = {}
    
    def find_window(self, title: str) -> Optional[Dict]:
        """
        Find a window by title.
        
        Returns:
            Dict with x, y, width, height or None
        """
        import platform
        
        try:
            if platform.system() == "Linux":
                return self._find_window_linux(title)
            elif platform.system() == "Windows":
                return self._find_window_windows(title)
            elif platform.system() == "Darwin":
                return self._find_window_macos(title)
        except Exception as e:
            print(f"Window search error: {e}")
        
        return None
    
    def _find_window_linux(self, title: str) -> Optional[Dict]:
        """Find window on Linux using wmctrl or xdotool."""
        import subprocess
        
        try:
            # Try wmctrl
            result = subprocess.run(
                ["wmctrl", "-l", "-G"],
                capture_output=True, text=True
            )
            
            for line in result.stdout.split("\n"):
                if title.lower() in line.lower():
                    parts = line.split()
                    if len(parts) >= 8:
                        return {
                            "x": int(parts[2]),
                            "y": int(parts[3]),
                            "width": int(parts[4]),
                            "height": int(parts[5]),
                        }
        except (subprocess.SubprocessError, ValueError, IndexError, OSError):
            pass
        
        try:
            # Try xdotool
            result = subprocess.run(
                ["xdotool", "search", "--name", title],
                capture_output=True, text=True
            )
            
            window_id = result.stdout.strip().split("\n")[0]
            if window_id:
                result = subprocess.run(
                    ["xdotool", "getwindowgeometry", window_id],
                    capture_output=True, text=True
                )
                
                import re
                pos_match = re.search(r'Position: (\d+),(\d+)', result.stdout)
                size_match = re.search(r'Geometry: (\d+)x(\d+)', result.stdout)
                
                if pos_match and size_match:
                    return {
                        "x": int(pos_match.group(1)),
                        "y": int(pos_match.group(2)),
                        "width": int(size_match.group(1)),
                        "height": int(size_match.group(2)),
                    }
        except (subprocess.SubprocessError, ValueError, IndexError, OSError):
            pass
        
        return None
    
    def _find_window_windows(self, title: str) -> Optional[Dict]:
        """Find window on Windows."""
        try:
            import ctypes
            from ctypes import wintypes
            
            user32 = ctypes.windll.user32
            
            # Find window
            hwnd = user32.FindWindowW(None, title)
            if not hwnd:
                # Search partial match
                def callback(hwnd, windows):
                    length = user32.GetWindowTextLengthW(hwnd)
                    if length:
                        buff = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(hwnd, buff, length + 1)
                        if title.lower() in buff.value.lower():
                            windows.append(hwnd)
                    return True
                
                WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.py_object)
                windows = []
                user32.EnumWindows(WNDENUMPROC(callback), windows)
                
                if windows:
                    hwnd = windows[0]
            
            if hwnd:
                rect = wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                return {
                    "x": rect.left,
                    "y": rect.top,
                    "width": rect.right - rect.left,
                    "height": rect.bottom - rect.top,
                }
        except (ImportError, OSError, ctypes.ArgumentError):
            pass
        
        return None
    
    def _find_window_macos(self, title: str) -> Optional[Dict]:
        """Find window on macOS."""
        try:
            import subprocess
            
            script = f'''
            tell application "System Events"
                set theWindows to every window of every process
                repeat with p in processes
                    repeat with w in windows of p
                        if name of w contains "{title}" then
                            return position of w & size of w
                        end if
                    end repeat
                end repeat
            end tell
            '''
            
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True
            )
            
            if result.stdout:
                parts = result.stdout.strip().split(", ")
                if len(parts) == 4:
                    return {
                        "x": int(parts[0]),
                        "y": int(parts[1]),
                        "width": int(parts[2]),
                        "height": int(parts[3]),
                    }
        except (subprocess.SubprocessError, ValueError, IndexError, OSError):
            pass
        
        return None


# Global avatar instance
_avatar: Optional[AvatarController] = None


def get_avatar() -> AvatarController:
    """Get or create global avatar controller."""
    global _avatar
    if _avatar is None:
        _avatar = AvatarController()
        _avatar.load_config()
    return _avatar


def enable_avatar() -> bool:
    """Enable the global avatar."""
    return get_avatar().enable()


def disable_avatar() -> None:
    """Disable the global avatar."""
    get_avatar().disable()


def toggle_avatar() -> bool:
    """Toggle avatar on/off."""
    return get_avatar().toggle()
