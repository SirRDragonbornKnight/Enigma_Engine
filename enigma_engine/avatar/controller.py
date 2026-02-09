"""
================================================================================
🤖 AVATAR CONTROLLER - YOUR AI PET COMPANION
================================================================================

A controllable visual avatar that lives on your desktop! Can move around,
express emotions, "interact" with windows, and speak with lip sync.

📍 FILE: enigma_engine/avatar/controller.py
🏷️ TYPE: Avatar Display & Control
🎯 MAIN CLASSES: AvatarController, AvatarState, AvatarPosition
🎯 MAIN FUNCTION: get_avatar()

┌─────────────────────────────────────────────────────────────────────────────┐
│  AVATAR CAPABILITIES:                                                       │
│                                                                             │
│  🖥️ Display on screen with customizable appearance                        │
│  🚶 Move around the desktop                                                 │
│  😊 Animate expressions and gestures                                        │
│  👆 "Interact" with windows/files (visual effects)                         │
│  🗣️ Speak with lip sync                                                    │
│  🔄 Turn on/off as needed                                                   │
└─────────────────────────────────────────────────────────────────────────────┘

⚠️ DEFAULT STATE: OFF (requires explicit enable())

🎭 AVATAR STATES (AvatarState enum):
    • OFF         - Not displayed
    • IDLE        - Standing/waiting
    • SPEAKING    - Talking with lip sync
    • THINKING    - Processing animation
    • MOVING      - Walking/transitioning
    • INTERACTING - "Touching" windows

🏗️ MODULAR ARCHITECTURE:
    • AvatarController - Main interface (on/off, commands)
    • AvatarRenderer   - Handles display (can swap backends)
    • AvatarAnimator   - Handles movements and expressions
    • ScreenInteractor - Visual effects for "touching" windows

🔗 CONNECTED FILES:
    → USES:      enigma_engine/avatar/animation_system.py (movement)
    → USES:      enigma_engine/avatar/lip_sync.py (speech sync)
    → USES:      enigma_engine/voice/voice_generator.py (speech)
    ← USED BY:   enigma_engine/avatar/autonomous.py (self-acting avatar)
    ← USED BY:   enigma_engine/gui/tabs/avatar_tab.py (GUI controls)

📖 USAGE:
    from enigma_engine.avatar import get_avatar
    
    avatar = get_avatar()
    avatar.enable()                    # Turn on
    avatar.move_to(500, 300)           # Move to position
    avatar.set_expression("happy")     # Change expression
    avatar.speak("Hello!")             # Speak with lip sync
    avatar.disable()                   # Turn off

📖 SEE ALSO:
    • enigma_engine/avatar/autonomous.py      - Make avatar act on its own
    • enigma_engine/avatar/desktop_pet.py     - Desktop overlay window
    • enigma_engine/avatar/customizer.py      - Customize appearance
    • docs/AVATAR_SYSTEM_GUIDE.md        - Full avatar documentation
"""

import ctypes
import json
import logging
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from threading import Lock
from typing import Callable, Dict, List, Optional

from ..config import CONFIG

logger = logging.getLogger(__name__)


class ControlPriority(IntEnum):
    """Priority levels for avatar control systems.
    
    Higher values = higher priority. When multiple systems want control,
    the highest priority wins. Bone animation is primary.
    """
    BONE_ANIMATION = 100    # Primary: Direct bone control for rigged models
    USER_MANUAL = 80        # User dragging/clicking avatar
    AI_TOOL_CALL = 70       # AI explicit commands via tools
    AUTONOMOUS = 50         # Autonomous behaviors
    IDLE_ANIMATION = 30     # Background idle animations
    FALLBACK = 10          # Last resort (for non-avatar-trained models)


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
    voice_profile: str = ""  # Name of voice profile for this avatar


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
        self._callbacks: dict[str, list[Callable]] = {
            "state_change": [],
            "move": [],
            "speak": [],
            "interact": [],
            "expression": [],  # Called when expression changes
        }
        
        # Current expression state
        self._current_expression: str = "neutral"
        
        # Animation queue (thread-safe)
        self._animation_queue: list[dict] = []
        self._animation_lock = Lock()
        self._animation_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Control priority system - bone animation is primary
        self._control_lock = Lock()
        self._current_controller: str = "none"
        self._current_priority: ControlPriority = ControlPriority.FALLBACK
        self._control_timeout: float = 0.0  # When current control expires
        self._priority_hold_time: float = 2.0  # Seconds to hold priority
    
    @property
    def is_enabled(self) -> bool:
        """Check if avatar is enabled."""
        return self.state != AvatarState.OFF
    
    @property
    def current_controller(self) -> str:
        """Get the name of the current controlling system."""
        with self._control_lock:
            # Check if control has expired
            if time.time() > self._control_timeout:
                self._current_controller = "none"
                self._current_priority = ControlPriority.FALLBACK
            return self._current_controller
    
    def request_control(self, requester: str, priority: ControlPriority, 
                       duration: float = None) -> bool:
        """Request control of the avatar.
        
        Args:
            requester: Name of the requesting system (e.g., 'bone_controller')
            priority: Priority level of the request
            duration: How long to hold control (seconds). None = use default
        
        Returns:
            True if control granted, False if denied
        
        """
        if not hasattr(self, '_control_lock'):
            # Priority system not initialized (old version compatibility)
            return True
            
        with self._control_lock:
            current_time = time.time()
            
            # If control expired, anyone can take it
            if current_time > self._control_timeout:
                self._current_controller = requester
                self._current_priority = priority
                self._control_timeout = current_time + (duration or self._priority_hold_time)
                logger.debug(f"Control granted to {requester} (priority {priority})")
                return True
            
            # If new priority is higher, override current controller
            if priority > self._current_priority:
                logger.debug(f"Control override: {self._current_controller} -> {requester} "
                           f"(priority {self._current_priority} -> {priority})")
                self._current_controller = requester
                self._current_priority = priority
                self._control_timeout = current_time + (duration or self._priority_hold_time)
                return True
            
            # If same priority and same requester, extend timeout
            if priority == self._current_priority and requester == self._current_controller:
                self._control_timeout = current_time + (duration or self._priority_hold_time)
                return True
            
            # Denied - lower priority or different requester
            return False
    
    def release_control(self, requester: str) -> None:
        """Release control of the avatar."""
        with self._control_lock:
            if self._current_controller == requester:
                self._current_controller = "none"
                self._current_priority = ControlPriority.FALLBACK
                self._control_timeout = 0.0
                logger.debug(f"Control released by {requester}")
    
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
            
            # Initialize animator (if not already created)
            if self._animator is None:
                self._animator = AvatarAnimator(self)
                logger.debug("Avatar animator initialized")
            
            # Start animation thread
            self._running = True
            self._animation_thread = threading.Thread(target=self._animation_loop, daemon=True)
            self._animation_thread.start()
            
            # Set to idle
            self._set_state(AvatarState.IDLE)
            self.config.enabled = True
            
            logger.info("Avatar enabled")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enable avatar: {e}")
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
        logger.info("Avatar disabled")
    
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
                logger.warning(f"State callback error: {e}")
    
    def _init_renderer(self) -> None:
        """Initialize the renderer component."""
        if self._renderer is None:
            # Try to use Qt renderer if PyQt5 is available (for visual rendering)
            # Fall back to SpriteRenderer (console-only) if not
            try:
                from .renderers import QtAvatarRenderer
                self._renderer = QtAvatarRenderer(self)
            except ImportError:
                from .renderers import SpriteRenderer
                self._renderer = SpriteRenderer(self)
            
            # Set appearance if we have an identity
            if self._identity and self._identity.appearance:
                self._renderer.set_appearance(self._identity.appearance)
    
    def _animation_loop(self) -> None:
        """Background animation processing loop."""
        while self._running:
            animation = None
            with self._animation_lock:
                if self._animation_queue:
                    animation = self._animation_queue.pop(0)
            
            if animation:
                self._execute_animation(animation)
            elif self.state == AvatarState.IDLE and self.config.idle_animation:
                # Idle animation
                self._execute_animation({"type": "idle", "duration": 2.0})
            time.sleep(0.05)  # 20 FPS
    
    def _execute_animation(self, animation: dict) -> None:
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
                logger.warning(f"Animation callback error: {e}")
    
    # === Movement ===
    
    def move_to(self, x: int, y: int, animate: bool = True, 
                requester: str = "manual", priority: ControlPriority = ControlPriority.USER_MANUAL) -> None:
        """
        Move avatar to position.
        
        Args:
            x: Target X coordinate
            y: Target Y coordinate
            animate: Whether to animate the movement
            requester: Name of the requesting system
            priority: Priority level of the request
        """
        if not self.is_enabled:
            return
        
        # Request control - if denied, don't move (backward compatible)
        if hasattr(self, 'request_control'):
            if not self.request_control(requester, priority, duration=1.0):
                logger.debug(f"Move denied for {requester} - controlled by {self.current_controller}")
                return
        
        if animate:
            self._set_state(AvatarState.MOVING)
            with self._animation_lock:
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
                logger.warning(f"Move callback error: {e}")
    
    def move_relative(self, dx: int, dy: int) -> None:
        """Move avatar by offset."""
        self.move_to(self.position.x + dx, self.position.y + dy)
    
    def center_on_screen(self) -> None:
        """Center avatar on screen."""
        try:
            # Try to get screen dimensions
            import platform
            import subprocess
            
            width, height = 1920, 1080  # Default fallback
            
            if platform.system() == "Linux":
                result = subprocess.run(
                    ["xrandr", "--current"],
                    capture_output=True, text=True, timeout=5
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
            logger.warning(f"Could not center on screen: {e}")
    
    # === Voice Profile ===
    
    def set_voice_profile(self, profile_name_or_profile) -> bool:
        """
        Set the avatar's voice profile.
        
        Args:
            profile_name_or_profile: Either a profile name (str) or VoiceProfile object
            
        Returns:
            True if successfully set
        """
        try:
            from ..voice.voice_profile import VoiceProfile
            
            if isinstance(profile_name_or_profile, str):
                # Try to load by name
                profile = VoiceProfile.load(profile_name_or_profile)
                self.config.voice_profile = profile_name_or_profile
                self.voice_profile = profile
            else:
                # It's a VoiceProfile object
                self.voice_profile = profile_name_or_profile
                self.config.voice_profile = profile_name_or_profile.name
            
            logger.info(f"Voice profile set to: {self.config.voice_profile}")
            return True
        except Exception as e:
            logger.error(f"Failed to set voice profile: {e}")
            return False
    
    def get_voice_profile_name(self) -> str:
        """Get the name of the current voice profile."""
        return self.config.voice_profile or "default"
    
    # === AI Control Interface ===
    
    def control(self, action: str, value: str = "") -> dict:
        """
        Control avatar from AI tool calls.
        
        Actions:
        - show: Show the avatar overlay on desktop
        - hide: Hide the avatar overlay
        - jump: Make the avatar jump
        - pin: Pin avatar in place (disable physics)
        - unpin: Unpin avatar (enable physics)
        - move: Move to position "x,y"
        - resize: Resize to "pixels" 
        - orientation: Set view angle "front", "back", "left", "right" or "x,y"
        
        Returns:
            Dict with success status and result/error message
        """
        # Write command to a file that the GUI watches
        import json
        from pathlib import Path
        
        command = {
            "action": action.lower().strip(),
            "value": value.strip() if value else "",
            "timestamp": time.time()
        }
        
        command_path = Path(__file__).parent.parent.parent / "data" / "avatar" / "ai_command.json"
        command_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(command_path, 'w') as f:
                json.dump(command, f)
            
            return {
                "success": True,
                "result": f"Avatar command '{action}' sent" + (f" with value '{value}'" if value else "")
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    # === Speaking ===
    
    def speak(self, text: str, animate: bool = True, use_tts: bool = True) -> None:
        """
        Animate avatar speaking and optionally use TTS.
        
        Args:
            text: Text being spoken (affects animation duration)
            animate: Whether to animate mouth/expressions
            use_tts: Whether to actually speak the text with TTS
        """
        if not self.is_enabled:
            return
        
        # Estimate speaking duration
        words = len(text.split())
        duration = max(1.0, words * 0.3)  # ~0.3 sec per word
        
        if animate:
            self._set_state(AvatarState.SPEAKING)
            with self._animation_lock:
                self._animation_queue.append({
                    "type": "speak",
                    "text": text,
                    "duration": duration,
                })
            
            # Also send command to desktop overlay via file bridge
            self.control("speak", text)
        
        # Use TTS with avatar's voice profile
        if use_tts:
            self._speak_with_voice_profile(text)
        
        for cb in self._callbacks["speak"]:
            try:
                cb(text)
            except Exception as e:
                logger.warning(f"Speak callback error: {e}")
    
    def animate_speak(self, text: str, duration: float = None) -> None:
        """
        Animate avatar speaking WITHOUT triggering TTS.
        
        This is used by SpeechSync when it controls TTS separately
        to keep avatar lip animation synchronized with audio.
        
        Args:
            text: Text being spoken (for animation timing)
            duration: Override duration in seconds (auto-calculated if None)
        """
        if not self.is_enabled:
            return
        
        # Calculate duration if not provided
        if duration is None:
            words = len(text.split())
            duration = max(1.0, words * 0.3)  # ~0.3 sec per word
        
        self._set_state(AvatarState.SPEAKING)
        with self._animation_lock:
            self._animation_queue.append({
                "type": "speak",
                "text": text,
                "duration": duration,
            })
        
        # Send command to desktop overlay via file bridge
        self.control("speak", text)
        
        for cb in self._callbacks["speak"]:
            try:
                cb(text)
            except Exception as e:
                logger.warning(f"Speak callback error: {e}")
    
    def _speak_with_voice_profile(self, text: str) -> None:
        """Speak text using avatar's configured voice profile."""
        try:
            from ..voice.voice_profile import VoiceProfile, get_engine
            
            engine = get_engine()
            
            # Try to use avatar's voice profile
            if self.config.voice_profile:
                try:
                    profile = VoiceProfile.load(self.config.voice_profile)
                    engine.set_profile(profile)
                except FileNotFoundError as e:
                    logger.debug(f"Voice profile not found, using default: {e}")
            
            # Also check if voice_profile is set as an attribute (from VoiceCloneTab)
            if hasattr(self, 'voice_profile') and self.voice_profile:
                engine.set_profile(self.voice_profile)
            
            engine.speak(text)
        except Exception as e:
            logger.error(f"TTS error: {e}")
    
    def think(self, duration: float = 2.0) -> None:
        """Show thinking animation."""
        if not self.is_enabled:
            return
        
        self._set_state(AvatarState.THINKING)
        with self._animation_lock:
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
        with self._animation_lock:
            self._animation_queue.append({
                "type": f"interact_{action}",
                "target": window_pos,
                "duration": 1.0,
            })
        
        for cb in self._callbacks["interact"]:
            try:
                cb(window_title, action, window_pos)
            except Exception as e:
                logger.warning(f"Interact callback error: {e}")
        
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
        
        with self._animation_lock:
            self._animation_queue.append({
                "type": "point",
                "target": (x, y),
                "duration": 1.0,
            })
    
    # === Expressions ===
    
    def set_expression(self, expression: str, requester: str = "manual", 
                      priority: ControlPriority = ControlPriority.USER_MANUAL) -> None:
        """
        Set avatar facial expression.
        
        Args:
            expression: One of: neutral, happy, sad, surprised, thinking, confused
            requester: Name of the requesting system
            priority: Priority level of the request
        """
        # Request control - if denied, don't change expression (backward compatible)
        if self.is_enabled and hasattr(self, 'request_control'):
            if not self.request_control(requester, priority, duration=0.5):
                logger.debug(f"Expression change denied for {requester} - controlled by {self.current_controller}")
                return
        
        # Update current expression (always track, even when disabled)
        old_expression = getattr(self, '_current_expression', 'neutral')
        self._current_expression = expression
        
        # Notify listeners immediately (for GUI updates) - works even when disabled
        for cb in self._callbacks.get("expression", []):
            try:
                cb(old_expression, expression)
            except Exception as e:
                logger.warning(f"Expression callback error: {e}")
        
        # Only queue animation if fully enabled
        if self.is_enabled:
            with self._animation_lock:
                self._animation_queue.append({
                    "type": "expression",
                    "expression": expression,
                    "duration": 0.5,
                })
            
            # Also send emotion to desktop overlay via file bridge
            self.control("emotion", expression)
    
    def execute_action(self, action: str, params: dict = None) -> dict:
        """
        Execute an avatar action (called by AI tool interface).
        
        Args:
            action: Action to perform: 'set_expression', 'speak', 'move', 'animate', 
                   'set_color', 'wave', 'nod', 'shake_head', 'look_at'
            params: Parameters for the action
            
        Returns:
            dict with 'success' and 'message' keys
        """
        params = params or {}
        
        try:
            if action == "set_expression":
                expression = params.get("expression", "neutral")
                self.set_expression(expression)
                return {"success": True, "message": f"Expression set to {expression}"}
            
            elif action == "speak":
                text = params.get("text", "")
                self.speak(text)
                return {"success": True, "message": f"Speaking: {text[:30]}..."}
            
            elif action == "move":
                x = params.get("x", self.position.x)
                y = params.get("y", self.position.y)
                self.move_to(x, y)
                return {"success": True, "message": f"Moving to ({x}, {y})"}
            
            elif action == "animate":
                animation = params.get("animation", "wave")
                with self._animation_lock:
                    self._animation_queue.append({
                        "type": animation,
                        "duration": params.get("duration", 1.0),
                    })
                return {"success": True, "message": f"Playing animation: {animation}"}
            
            elif action == "wave":
                with self._animation_lock:
                    self._animation_queue.append({"type": "wave", "duration": 1.5})
                return {"success": True, "message": "Waving!"}
            
            elif action == "nod":
                with self._animation_lock:
                    self._animation_queue.append({"type": "nod", "duration": 0.8})
                return {"success": True, "message": "Nodding"}
            
            elif action == "shake_head":
                with self._animation_lock:
                    self._animation_queue.append({"type": "shake_head", "duration": 1.0})
                return {"success": True, "message": "Shaking head"}
            
            elif action == "look_at":
                x = params.get("x", 0)
                y = params.get("y", 0)
                self.point_at(x, y)
                return {"success": True, "message": f"Looking at ({x}, {y})"}
            
            elif action == "set_color":
                color = params.get("color", "#6366f1")
                self.set_color(color)
                return {"success": True, "message": f"Color set to {color}"}
            
            elif action == "set_scale":
                scale = params.get("scale", 1.0)
                self.set_scale(scale)
                return {"success": True, "message": f"Scale set to {scale}"}
            
            elif action == "enable":
                self.enable()
                return {"success": True, "message": "Avatar enabled"}
            
            elif action == "disable":
                self.disable()
                return {"success": True, "message": "Avatar disabled"}
            
            else:
                return {"success": False, "message": f"Unknown action: {action}"}
                
        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}
    
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
            logger.warning(f"Model not found: {model_path}")
            return False
        
        self.config.model_path = str(path)
        
        if self._renderer:
            self._renderer.load_model(model_path)
        
        logger.info(f"Avatar model changed to: {path.name}")
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
    
    def list_available_models(self) -> list[str]:
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
        # Lazy imports to avoid circular dependencies
        from .avatar_identity import AIAvatarIdentity
        from .emotion_sync import EmotionExpressionSync
        
        self._identity = AIAvatarIdentity(personality)
        self._emotion_sync = EmotionExpressionSync(self, personality)
        self._emotion_sync.start_sync()
        
        logger.info(f"Linked to personality: {personality.model_name}")
    
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
            
            logger.info(f"AI designed appearance: {self._identity.reasoning}")
            return appearance
        else:
            logger.warning("No personality linked. Use link_personality() first.")
            return None
    
    def describe_desired_appearance(self, description: str) -> Optional['AvatarAppearance']:
        """
        AI describes desired appearance in natural language.
        
        Args:
            description: Natural language description
            
        Returns:
            AvatarAppearance based on description
        """
        # Lazy import to avoid circular dependencies
        from .avatar_identity import AIAvatarIdentity
        
        if not self._identity:
            self._identity = AIAvatarIdentity()
        
        appearance = self._identity.describe_desired_appearance(description)
        
        # Apply to renderer if enabled
        if self.is_enabled and self._renderer:
            self._renderer.set_appearance(appearance)
        
        logger.info(f"Created appearance from: {description}")
        return appearance
    
    def get_customizer(self) -> 'AvatarCustomizer':
        """
        Get customizer for user modifications.
        
        Returns:
            AvatarCustomizer instance
        """
        if self._customizer is None:
            # Lazy import to avoid circular dependencies
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
        # Lazy import to avoid circular dependencies
        from .avatar_identity import AIAvatarIdentity
        
        if not self._identity:
            self._identity = AIAvatarIdentity()
        
        self._identity.appearance = appearance
        
        # Apply to renderer if enabled
        if self.is_enabled and self._renderer:
            self._renderer.set_appearance(appearance)
        
        logger.debug("Appearance updated")
    
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
    
    def get_expression(self) -> str:
        """Get current expression."""
        return getattr(self, '_current_expression', 'neutral')
    
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
        
        with open(path) as f:
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
        logger.debug(f"Showing at ({self.controller.position.x}, {self.controller.position.y})")
    
    def hide(self) -> None:
        """Hide avatar window."""
        self._visible = False
        logger.debug("Hidden")
    
    def set_position(self, x: int, y: int) -> None:
        """Update avatar position."""
        if self._visible:
            logger.debug(f"Moving to ({x}, {y})")
    
    def render_frame(self, animation_data: dict = None) -> None:
        """Render a single frame."""
        pass  # Implement with actual rendering
    
    def load_model(self, model_path: str) -> bool:
        """Load a 3D model file."""
        # Stub - implement with actual 3D rendering library
        logger.debug(f"Loading model: {model_path}")
        return True
    
    def set_scale(self, scale: float) -> None:
        """Set avatar scale."""
        logger.debug(f"Scale set to: {scale}")
    
    def set_opacity(self, opacity: float) -> None:
        """Set avatar opacity."""
        logger.debug(f"Opacity set to: {opacity}")
    
    def set_color(self, color: str) -> None:
        """Set avatar color/tint."""
        logger.debug(f"Color set to: {color}")


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
        logger.debug(f"Animation: {animation_type} ({duration}s)")
    
    def stop(self) -> None:
        """Stop current animation."""
        self.current_animation = None


class ScreenInteractor:
    """
    Handles finding and interacting with screen elements.
    """
    
    def __init__(self):
        self._window_cache: dict[str, dict] = {}
    
    def find_window(self, title: str) -> Optional[dict]:
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
            logger.warning(f"Window search error: {e}")
        
        return None
    
    def _find_window_linux(self, title: str) -> Optional[dict]:
        """Find window on Linux using wmctrl or xdotool."""
        import subprocess
        
        try:
            # Try wmctrl
            result = subprocess.run(
                ["wmctrl", "-l", "-G"],
                capture_output=True, text=True, timeout=10
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
        except (subprocess.SubprocessError, ValueError, IndexError, OSError) as e:
            logger.debug(f"wmctrl window search failed: {e}")
        
        try:
            # Try xdotool
            result = subprocess.run(
                ["xdotool", "search", "--name", title],
                capture_output=True, text=True, timeout=10
            )
            
            window_id = result.stdout.strip().split("\n")[0]
            if window_id:
                result = subprocess.run(
                    ["xdotool", "getwindowgeometry", window_id],
                    capture_output=True, text=True, timeout=10
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
        except (subprocess.SubprocessError, ValueError, IndexError, OSError) as e:
            logger.debug(f"xdotool window search failed: {e}")
        
        return None
    
    def _find_window_windows(self, title: str) -> Optional[dict]:
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
        except (ImportError, OSError, ctypes.ArgumentError) as e:
            logger.debug(f"Windows window search failed: {e}")
        
        return None
    
    def _find_window_macos(self, title: str) -> Optional[dict]:
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
                capture_output=True, text=True, timeout=30
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
        except (subprocess.SubprocessError, ValueError, IndexError, OSError) as e:
            logger.debug(f"macOS window search failed: {e}")
        
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


def execute_action(action: str, params: dict = None) -> dict:
    """
    Execute an avatar action (module-level function for AI tool interface).
    
    Args:
        action: Action to perform: 'set_expression', 'speak', 'move', 'animate', etc.
        params: Parameters for the action
        
    Returns:
        dict with 'success' and 'message' keys
    """
    return get_avatar().execute_action(action, params)
