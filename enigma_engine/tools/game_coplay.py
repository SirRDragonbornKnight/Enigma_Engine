"""
Game Co-Play System - AI plays games WITH you, not FOR you.

This module allows the AI to:
1. See the game screen (via vision/screenshot)
2. Understand game state through conversation
3. Send inputs to the game (keyboard/mouse/controller)
4. Coordinate with you as a teammate or opponent

IMPORTANT: The AI plays as a PARTNER. It doesn't take over - it cooperates.

Supported input methods:
- Keyboard simulation (pynput)
- Mouse simulation (pynput)
- Controller simulation (vgamepad on Windows, uinput on Linux)
- Game-specific APIs (mods, WebSocket bridges)

Usage:
    from enigma_engine.tools.game_coplay import GameCoPlayer
    
    player = GameCoPlayer()
    player.set_role("teammate")  # or "opponent", "coach", "companion"
    player.connect_game("minecraft")
    
    # AI sees screen and decides action
    player.observe_and_act()
    
    # Or explicit coordination
    player.say("I'll cover the left side")
    player.press_key("a")  # Move left
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CoPlayRole(Enum):
    """Roles the AI can take when playing with you."""
    TEAMMATE = "teammate"       # Works with you toward shared goal
    OPPONENT = "opponent"       # Plays against you (friendly competition)
    COACH = "coach"             # Watches and gives advice, minimal input
    COMPANION = "companion"     # Casual co-op, follows your lead
    SUPPORT = "support"         # Heals, buffs, covers - support role
    EXPLORER = "explorer"       # Scouts ahead, finds items
    DEFENDER = "defender"       # Protects base/position


class InputMethod(Enum):
    """How the AI sends inputs to the game."""
    KEYBOARD = "keyboard"       # Keyboard simulation
    MOUSE = "mouse"             # Mouse simulation
    CONTROLLER = "controller"   # Virtual gamepad
    API = "api"                 # Game-specific API/mod
    HYBRID = "hybrid"           # Combination


@dataclass
class GameAction:
    """An action the AI wants to perform."""
    action_type: str            # "move", "attack", "use", "say", etc.
    parameters: dict[str, Any] = field(default_factory=dict)
    reason: str = ""            # Why the AI chose this action
    confidence: float = 1.0     # How confident (0-1)
    delay: float = 0.0          # Delay before executing
    
    def to_dict(self) -> dict:
        return {
            "type": self.action_type,
            "params": self.parameters,
            "reason": self.reason,
            "confidence": self.confidence,
        }


@dataclass
class CoPlayConfig:
    """Configuration for co-play behavior."""
    role: CoPlayRole = CoPlayRole.COMPANION
    input_method: InputMethod = InputMethod.KEYBOARD
    
    # Behavior settings
    reaction_delay: float = 0.1         # Human-like delay (seconds)
    max_actions_per_second: int = 5     # Don't spam inputs
    announce_actions: bool = True       # Tell player what AI is doing
    ask_before_major: bool = True       # Ask before big decisions
    
    # Safety
    pause_on_menu: bool = True          # Don't act in menus
    respect_cooldowns: bool = True      # Don't spam abilities
    emergency_stop_key: str = "escape"  # AI stops on this key
    
    # Coordination
    follow_player: bool = True          # Stay near player
    share_loot: bool = True             # Don't hoard items
    call_out_enemies: bool = True       # Warn about threats


class InputSimulator:
    """Simulates keyboard, mouse, and controller inputs."""
    
    def __init__(self):
        self._keyboard = None
        self._mouse = None
        self._controller = None
        self._available_methods: list[InputMethod] = []
        self._init_inputs()
    
    def _init_inputs(self):
        """Initialize available input methods."""
        # Try pynput for keyboard/mouse
        try:
            from pynput.keyboard import Controller as KeyboardController
            from pynput.keyboard import Key
            self._keyboard = KeyboardController()
            self._Key = Key
            self._available_methods.append(InputMethod.KEYBOARD)
            logger.info("Keyboard input available (pynput)")
        except ImportError:
            logger.warning("pynput not installed - keyboard input unavailable")
        
        try:
            from pynput.mouse import Button
            from pynput.mouse import Controller as MouseController
            self._mouse = MouseController()
            self._Button = Button
            self._available_methods.append(InputMethod.MOUSE)
            logger.info("Mouse input available (pynput)")
        except ImportError:
            logger.warning("pynput not installed - mouse input unavailable")
        
        # Try virtual gamepad
        try:
            import sys
            if sys.platform == 'win32':
                import vgamepad as vg
                self._controller = vg.VX360Gamepad()
                self._available_methods.append(InputMethod.CONTROLLER)
                logger.info("Controller input available (vgamepad)")
            else:
                # Linux - try uinput
                try:
                    self._controller = "uinput"
                    self._available_methods.append(InputMethod.CONTROLLER)
                    logger.info("Controller input available (uinput)")
                except ImportError:
                    pass
        except ImportError:
            logger.debug("Virtual gamepad not available")
    
    @property
    def available_methods(self) -> list[InputMethod]:
        return self._available_methods
    
    def press_key(self, key: str, duration: float = 0.05):
        """Press and release a keyboard key."""
        if not self._keyboard:
            logger.warning("Keyboard not available")
            return False
        
        try:
            # Handle special keys
            if hasattr(self._Key, key.lower()):
                k = getattr(self._Key, key.lower())
            else:
                k = key
            
            self._keyboard.press(k)
            time.sleep(duration)
            self._keyboard.release(k)
            return True
        except Exception as e:
            logger.error(f"Key press failed: {e}")
            return False
    
    def hold_key(self, key: str):
        """Hold a key down."""
        if not self._keyboard:
            return False
        
        try:
            if hasattr(self._Key, key.lower()):
                k = getattr(self._Key, key.lower())
            else:
                k = key
            self._keyboard.press(k)
            return True
        except Exception as e:
            logger.error(f"Key hold failed: {e}")
            return False
    
    def release_key(self, key: str):
        """Release a held key."""
        if not self._keyboard:
            return False
        
        try:
            if hasattr(self._Key, key.lower()):
                k = getattr(self._Key, key.lower())
            else:
                k = key
            self._keyboard.release(k)
            return True
        except Exception as e:
            logger.error(f"Key release failed: {e}")
            return False
    
    def type_text(self, text: str, delay: float = 0.02):
        """Type a string of text."""
        if not self._keyboard:
            return False
        
        try:
            for char in text:
                self._keyboard.press(char)
                self._keyboard.release(char)
                time.sleep(delay)
            return True
        except Exception as e:
            logger.error(f"Type text failed: {e}")
            return False
    
    def click_mouse(self, button: str = "left", x: Optional[int] = None, y: Optional[int] = None):
        """Click mouse button, optionally at position."""
        if not self._mouse:
            return False
        
        try:
            if x is not None and y is not None:
                self._mouse.position = (x, y)
            
            btn = self._Button.left if button == "left" else self._Button.right
            self._mouse.click(btn)
            return True
        except Exception as e:
            logger.error(f"Mouse click failed: {e}")
            return False
    
    def move_mouse(self, x: int, y: int, relative: bool = False):
        """Move mouse to position or by offset."""
        if not self._mouse:
            return False
        
        try:
            if relative:
                current = self._mouse.position
                self._mouse.position = (current[0] + x, current[1] + y)
            else:
                self._mouse.position = (x, y)
            return True
        except Exception as e:
            logger.error(f"Mouse move failed: {e}")
            return False


class GameCoPlayer:
    """
    AI Co-Player - plays games WITH you.
    
    The AI observes the game, understands context, and performs actions
    as your teammate, opponent, or companion.
    """
    
    def __init__(self, config: Optional[CoPlayConfig] = None):
        self.config = config or CoPlayConfig()
        self._input = InputSimulator()
        self._active = False
        self._paused = False
        self._game_id: Optional[str] = None
        self._game_state: dict[str, Any] = {}
        
        # Action queue and timing
        self._action_queue: list[GameAction] = []
        self._last_action_time = 0.0
        self._actions_this_second = 0
        
        # Communication
        self._on_action: list[Callable] = []  # Callbacks when AI acts
        self._on_speak: list[Callable] = []   # Callbacks when AI speaks
        
        # Key bindings per game (loaded from config)
        self._key_bindings: dict[str, dict[str, str]] = {}
        self._load_default_bindings()
        
        # Thread for action execution
        self._action_thread: Optional[threading.Thread] = None
    
    def _load_default_bindings(self):
        """Load default key bindings for common games."""
        self._key_bindings = {
            "default": {
                "move_forward": "w",
                "move_backward": "s",
                "move_left": "a",
                "move_right": "d",
                "jump": "space",
                "crouch": "ctrl",
                "sprint": "shift",
                "interact": "e",
                "attack": "mouse_left",
                "alt_attack": "mouse_right",
                "inventory": "i",
                "map": "m",
                "chat": "t",
            },
            "minecraft": {
                "move_forward": "w",
                "move_backward": "s",
                "move_left": "a",
                "move_right": "d",
                "jump": "space",
                "crouch": "shift",
                "sprint": "ctrl",
                "interact": "e",
                "attack": "mouse_left",
                "place": "mouse_right",
                "inventory": "e",
                "drop": "q",
                "chat": "t",
                "hotbar_1": "1",
                "hotbar_2": "2",
                "hotbar_3": "3",
            },
            "terraria": {
                "move_left": "a",
                "move_right": "d",
                "jump": "space",
                "grapple": "e",
                "attack": "mouse_left",
                "use": "mouse_right",
                "inventory": "escape",
                "quick_heal": "h",
                "quick_mana": "j",
            },
        }
    
    def set_role(self, role: str):
        """Set the AI's role in the game."""
        try:
            self.config.role = CoPlayRole(role.lower())
            logger.info(f"AI role set to: {self.config.role.value}")
        except ValueError:
            logger.warning(f"Unknown role: {role}, keeping {self.config.role.value}")
    
    def connect_game(self, game_id: str):
        """Connect to a specific game."""
        self._game_id = game_id
        self._game_state = {"connected": True, "game": game_id}
        
        # Load game-specific bindings if available
        if game_id not in self._key_bindings:
            self._key_bindings[game_id] = self._key_bindings["default"].copy()
        
        logger.info(f"Connected to game: {game_id}")
        return True
    
    def disconnect(self):
        """Disconnect from game."""
        self._active = False
        self._game_id = None
        self._game_state = {}
        logger.info("Disconnected from game")
    
    def start(self):
        """Start the co-player (begins processing actions)."""
        if self._active:
            return
        
        self._active = True
        self._paused = False
        
        # Start action processing thread
        self._action_thread = threading.Thread(target=self._action_loop, daemon=True)
        self._action_thread.start()
        
        logger.info("Co-player started")
    
    def stop(self):
        """Stop the co-player."""
        self._active = False
        if self._action_thread:
            self._action_thread.join(timeout=1.0)
        logger.info("Co-player stopped")
    
    def pause(self):
        """Pause action execution."""
        self._paused = True
        logger.info("Co-player paused")
    
    def resume(self):
        """Resume action execution."""
        self._paused = False
        logger.info("Co-player resumed")
    
    def _action_loop(self):
        """Process queued actions."""
        while self._active:
            if self._paused or not self._action_queue:
                time.sleep(0.05)
                continue
            
            # Rate limiting
            current_time = time.time()
            if current_time - self._last_action_time < 1.0:
                if self._actions_this_second >= self.config.max_actions_per_second:
                    time.sleep(0.1)
                    continue
            else:
                self._actions_this_second = 0
            
            # Get next action
            action = self._action_queue.pop(0)
            
            # Execute with delay
            if action.delay > 0:
                time.sleep(action.delay)
            
            # Human-like reaction delay
            time.sleep(self.config.reaction_delay)
            
            # Execute the action
            self._execute_action(action)
            
            # Update timing
            self._last_action_time = time.time()
            self._actions_this_second += 1
            
            # Notify callbacks
            for cb in self._on_action:
                try:
                    cb(action)
                except Exception as e:
                    logger.debug(f"Action callback error: {e}")
    
    def _execute_action(self, action: GameAction):
        """Execute a single game action."""
        action_type = action.action_type
        params = action.parameters
        
        # Get key bindings for current game
        bindings = self._key_bindings.get(self._game_id, self._key_bindings["default"])
        
        if action_type == "move":
            direction = params.get("direction", "forward")
            duration = params.get("duration", 0.1)
            key = bindings.get(f"move_{direction}", "w")
            if key.startswith("mouse_"):
                # Mouse movement
                pass
            else:
                self._input.hold_key(key)
                time.sleep(duration)
                self._input.release_key(key)
        
        elif action_type == "press":
            key = params.get("key", "")
            if key in bindings:
                key = bindings[key]
            self._input.press_key(key)
        
        elif action_type == "hold":
            key = params.get("key", "")
            duration = params.get("duration", 0.5)
            if key in bindings:
                key = bindings[key]
            self._input.hold_key(key)
            time.sleep(duration)
            self._input.release_key(key)
        
        elif action_type == "click":
            button = params.get("button", "left")
            x = params.get("x")
            y = params.get("y")
            self._input.click_mouse(button, x, y)
        
        elif action_type == "type":
            text = params.get("text", "")
            self._input.type_text(text)
        
        elif action_type == "chat":
            # Open chat, type message, send
            chat_key = bindings.get("chat", "t")
            message = params.get("message", "")
            self._input.press_key(chat_key)
            time.sleep(0.1)
            self._input.type_text(message)
            time.sleep(0.05)
            self._input.press_key("enter")
        
        elif action_type == "combo":
            # Execute a sequence of keys
            keys = params.get("keys", [])
            delay = params.get("delay", 0.05)
            for key in keys:
                if key in bindings:
                    key = bindings[key]
                self._input.press_key(key)
                time.sleep(delay)
        
        logger.debug(f"Executed action: {action_type} - {action.reason}")
    
    # === High-level actions for AI to call ===
    
    def move(self, direction: str, duration: float = 0.2, reason: str = ""):
        """Queue a movement action."""
        action = GameAction(
            action_type="move",
            parameters={"direction": direction, "duration": duration},
            reason=reason or f"Moving {direction}",
        )
        self._action_queue.append(action)
    
    def press_key(self, key: str, reason: str = ""):
        """Queue a key press."""
        action = GameAction(
            action_type="press",
            parameters={"key": key},
            reason=reason or f"Pressing {key}",
        )
        self._action_queue.append(action)
    
    def hold_key(self, key: str, duration: float = 0.5, reason: str = ""):
        """Queue a key hold."""
        action = GameAction(
            action_type="hold",
            parameters={"key": key, "duration": duration},
            reason=reason or f"Holding {key}",
        )
        self._action_queue.append(action)
    
    def click(self, button: str = "left", x: int = None, y: int = None, reason: str = ""):
        """Queue a mouse click."""
        action = GameAction(
            action_type="click",
            parameters={"button": button, "x": x, "y": y},
            reason=reason or f"Clicking {button}",
        )
        self._action_queue.append(action)
    
    def say_in_game(self, message: str, reason: str = ""):
        """Queue a chat message in-game."""
        action = GameAction(
            action_type="chat",
            parameters={"message": message},
            reason=reason or "Communicating with player",
        )
        self._action_queue.append(action)
        
        # Also notify speak callbacks
        for cb in self._on_speak:
            try:
                cb(message)
            except Exception:
                pass
    
    def combo(self, keys: list[str], delay: float = 0.05, reason: str = ""):
        """Queue a combo of key presses."""
        action = GameAction(
            action_type="combo",
            parameters={"keys": keys, "delay": delay},
            reason=reason or f"Combo: {' -> '.join(keys)}",
        )
        self._action_queue.append(action)
    
    def announce(self, message: str):
        """Announce what the AI is doing (voice or overlay)."""
        if self.config.announce_actions:
            for cb in self._on_speak:
                try:
                    cb(message)
                except Exception:
                    pass
    
    # === Observation and decision making ===
    
    def observe(self) -> dict[str, Any]:
        """
        Observe the current game state by capturing the screen.
        
        Uses mss for fast screenshot capture to understand
        what's happening in the game.
        
        Returns:
            Dict with screen capture and analysis
        """
        try:
            # Fast screen capture with mss
            try:
                import mss
                with mss.mss() as sct:
                    monitor = sct.monitors[1]  # Primary monitor
                    screenshot = sct.grab(monitor)
                    # Convert to bytes for analysis
                    self._game_state["screen_width"] = screenshot.width
                    self._game_state["screen_height"] = screenshot.height
                    self._game_state["last_capture_time"] = time.time()
                    self._game_state["has_screen"] = True
                    
                    # Store raw pixels for vision processing
                    from PIL import Image
                    img = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')
                    self._last_screen_capture = img
                    
            except ImportError:
                # Fallback to PIL
                from PIL import ImageGrab
                img = ImageGrab.grab()
                self._game_state["screen_width"] = img.width
                self._game_state["screen_height"] = img.height
                self._game_state["last_capture_time"] = time.time()
                self._game_state["has_screen"] = True
                self._last_screen_capture = img
                
        except Exception as e:
            logger.warning(f"Screen capture failed: {e}")
            self._game_state["has_screen"] = False
        
        return self._game_state
    
    def get_screen_image(self) -> Optional[Any]:
        """Get the last captured screen image for AI analysis."""
        return getattr(self, '_last_screen_capture', None)
    
    def analyze_screen(self) -> str:
        """
        Capture screen and get AI analysis of what's happening.
        
        Returns:
            Text description of game state
        """
        self.observe()
        
        img = self.get_screen_image()
        if not img:
            return "Could not capture screen"
        
        # Try to get AI vision analysis
        try:
            from enigma_engine.tools.vision_tools import describe_image
            return describe_image(img)
        except ImportError:
            return f"Screen captured: {img.width}x{img.height} - vision tools not available"
    
    def start_continuous_observation(self, interval: float = 1.0):
        """
        Start continuously observing the game screen.
        
        Args:
            interval: Seconds between observations
        """
        if hasattr(self, '_observe_thread') and self._observe_thread:
            return  # Already running
        
        self._observing = True
        
        def observe_loop():
            while self._observing and self._active:
                self.observe()
                
                # Let AI analyze if we have a good capture
                if self._game_state.get("has_screen"):
                    # AI could react based on what it sees
                    pass
                
                time.sleep(interval)
        
        self._observe_thread = threading.Thread(target=observe_loop, daemon=True)
        self._observe_thread.start()
        logger.info(f"Started continuous game observation (every {interval}s)")
    
    def stop_continuous_observation(self):
        """Stop continuous screen observation."""
        self._observing = False
        if hasattr(self, '_observe_thread'):
            self._observe_thread = None
        logger.info("Stopped continuous game observation")

    def decide_action(self, context: str) -> Optional[GameAction]:
        """
        Let the AI decide what action to take based on context.
        
        Args:
            context: Description of current situation (from vision or player)
            
        Returns:
            The action the AI wants to take, or None if no action needed
        """
        # This would call the main AI model to decide
        # For now, return None - the AI calls specific actions
        return None
    
    def on_action(self, callback: Callable):
        """Register callback for when AI performs an action."""
        self._on_action.append(callback)
    
    def on_speak(self, callback: Callable):
        """Register callback for when AI speaks/announces."""
        self._on_speak.append(callback)
    
    # === Coordination methods ===
    
    def follow_player(self):
        """Follow the player's position."""
        self.announce("Following you!")
        # Would use vision to track player
    
    def cover_direction(self, direction: str):
        """Cover a specific direction for the player."""
        self.announce(f"Covering {direction}!")
        # Would turn to face that direction
    
    def call_out(self, what: str, where: str = ""):
        """Call out something to the player."""
        message = f"{what}"
        if where:
            message += f" at {where}"
        self.say_in_game(message)
    
    def request_help(self, situation: str):
        """Ask the player for help."""
        self.say_in_game(f"Need help! {situation}")
    
    def celebrate(self):
        """Celebrate a victory/achievement."""
        self.say_in_game("Nice!")
        self.press_key("jump", reason="Celebrating")


# Singleton instance
_coplay_instance: Optional[GameCoPlayer] = None


def get_coplayer() -> GameCoPlayer:
    """Get or create the global co-player instance."""
    global _coplay_instance
    if _coplay_instance is None:
        _coplay_instance = GameCoPlayer()
    return _coplay_instance
