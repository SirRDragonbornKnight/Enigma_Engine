"""
Voice Commands for Gaming in Enigma AI Engine

Voice-activated game control and commands.

Features:
- Wake word detection
- Command recognition
- Game action mapping
- Macro execution
- Context-aware commands

Usage:
    from enigma_engine.tools.voice_gaming import VoiceGameController, get_voice_controller
    
    controller = get_voice_controller()
    
    # Register game commands
    controller.register_command("reload", action="key:r")
    controller.register_command("jump", action="key:space")
    
    # Start listening
    controller.start()
"""

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of game actions."""
    KEY_PRESS = "key_press"
    KEY_HOLD = "key_hold"
    MOUSE_CLICK = "mouse_click"
    MOUSE_MOVE = "mouse_move"
    MACRO = "macro"
    CUSTOM = "custom"


class CommandPriority(Enum):
    """Priority levels for commands."""
    CRITICAL = 100  # Emergency stop, etc.
    HIGH = 75
    NORMAL = 50
    LOW = 25


@dataclass
class GameAction:
    """A game action to execute."""
    action_type: ActionType
    
    # Key/mouse
    key: Optional[str] = None
    button: Optional[str] = None
    
    # Movement
    x: Optional[int] = None
    y: Optional[int] = None
    
    # Timing
    duration: float = 0.0  # For hold actions
    delay: float = 0.0  # Delay before action
    
    # Macro
    sequence: List['GameAction'] = field(default_factory=list)
    
    # Custom
    callback: Optional[Callable] = None


@dataclass
class VoiceCommand:
    """A voice command definition."""
    name: str
    triggers: List[str]  # Phrases that trigger this command
    action: GameAction
    
    # Settings
    priority: CommandPriority = CommandPriority.NORMAL
    cooldown: float = 0.0
    enabled: bool = True
    
    # Context
    requires_context: Optional[str] = None  # Game/menu context
    
    # Tracking
    last_used: float = 0.0
    use_count: int = 0


@dataclass
class RecognitionResult:
    """Result from voice recognition."""
    text: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    
    # Matched command
    command: Optional[VoiceCommand] = None
    match_score: float = 0.0


class CommandMatcher:
    """Match spoken text to commands."""
    
    def __init__(
        self,
        fuzzy_threshold: float = 0.7
    ):
        self._fuzzy_threshold = fuzzy_threshold
        self._commands: Dict[str, VoiceCommand] = {}
    
    def register(self, command: VoiceCommand):
        """Register a command."""
        self._commands[command.name] = command
    
    def unregister(self, name: str):
        """Unregister a command."""
        if name in self._commands:
            del self._commands[name]
    
    def match(
        self,
        text: str,
        context: Optional[str] = None
    ) -> Optional[Tuple[VoiceCommand, float]]:
        """
        Match text to a command.
        
        Returns:
            (command, score) or None
        """
        text_lower = text.lower().strip()
        best_match: Optional[Tuple[VoiceCommand, float]] = None
        
        for command in self._commands.values():
            if not command.enabled:
                continue
            
            # Check context
            if command.requires_context and command.requires_context != context:
                continue
            
            # Check cooldown
            if command.cooldown > 0:
                if time.time() - command.last_used < command.cooldown:
                    continue
            
            # Check triggers
            for trigger in command.triggers:
                score = self._calculate_match_score(text_lower, trigger.lower())
                
                if score >= self._fuzzy_threshold:
                    if best_match is None or score > best_match[1]:
                        best_match = (command, score)
                        
                        # Perfect match
                        if score == 1.0:
                            return best_match
        
        return best_match
    
    def _calculate_match_score(
        self,
        text: str,
        trigger: str
    ) -> float:
        """Calculate match score between text and trigger."""
        # Exact match
        if text == trigger:
            return 1.0
        
        # Contains match
        if trigger in text:
            return 0.9
        
        if text in trigger:
            return 0.85
        
        # Word overlap
        text_words = set(text.split())
        trigger_words = set(trigger.split())
        
        if not trigger_words:
            return 0.0
        
        overlap = len(text_words & trigger_words)
        score = overlap / len(trigger_words)
        
        return score * 0.8  # Cap at 0.8 for fuzzy matches


class ActionExecutor:
    """Execute game actions."""
    
    def __init__(self):
        self._pyautogui = None
        self._pynput_keyboard = None
        self._pynput_mouse = None
        
        self._load_libraries()
    
    def _load_libraries(self):
        """Load input libraries."""
        try:
            import pyautogui
            pyautogui.FAILSAFE = True
            self._pyautogui = pyautogui
        except ImportError:
            logger.warning("pyautogui not available")
        
        try:
            from pynput import keyboard, mouse
            self._pynput_keyboard = keyboard
            self._pynput_mouse = mouse
        except ImportError:
            logger.warning("pynput not available")
    
    def execute(self, action: GameAction):
        """Execute a game action."""
        if action.delay > 0:
            time.sleep(action.delay)
        
        if action.action_type == ActionType.KEY_PRESS:
            self._key_press(action.key)
        
        elif action.action_type == ActionType.KEY_HOLD:
            self._key_hold(action.key, action.duration)
        
        elif action.action_type == ActionType.MOUSE_CLICK:
            self._mouse_click(action.button, action.x, action.y)
        
        elif action.action_type == ActionType.MOUSE_MOVE:
            self._mouse_move(action.x, action.y)
        
        elif action.action_type == ActionType.MACRO:
            self._execute_macro(action.sequence)
        
        elif action.action_type == ActionType.CUSTOM:
            if action.callback:
                action.callback()
    
    def _key_press(self, key: Optional[str]):
        """Press a key."""
        if not key:
            return
        
        if self._pyautogui:
            self._pyautogui.press(key)
        elif self._pynput_keyboard:
            keyboard = self._pynput_keyboard.Controller()
            keyboard.press(key)
            keyboard.release(key)
    
    def _key_hold(self, key: Optional[str], duration: float):
        """Hold a key for duration."""
        if not key:
            return
        
        if self._pyautogui:
            self._pyautogui.keyDown(key)
            time.sleep(duration)
            self._pyautogui.keyUp(key)
        elif self._pynput_keyboard:
            keyboard = self._pynput_keyboard.Controller()
            keyboard.press(key)
            time.sleep(duration)
            keyboard.release(key)
    
    def _mouse_click(
        self,
        button: Optional[str],
        x: Optional[int],
        y: Optional[int]
    ):
        """Click mouse."""
        button = button or "left"
        
        if self._pyautogui:
            if x is not None and y is not None:
                self._pyautogui.click(x, y, button=button)
            else:
                self._pyautogui.click(button=button)
    
    def _mouse_move(
        self,
        x: Optional[int],
        y: Optional[int]
    ):
        """Move mouse."""
        if x is None or y is None:
            return
        
        if self._pyautogui:
            self._pyautogui.moveTo(x, y)
    
    def _execute_macro(self, sequence: List[GameAction]):
        """Execute a sequence of actions."""
        for action in sequence:
            self.execute(action)


class VoiceRecognizer:
    """Voice recognition interface."""
    
    def __init__(
        self,
        wake_word: Optional[str] = None
    ):
        self._wake_word = wake_word
        self._recognizer = None
        self._microphone = None
        
        self._callback: Optional[Callable[[str, float], None]] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        self._load_speech_recognition()
    
    def _load_speech_recognition(self):
        """Load speech recognition library."""
        try:
            import speech_recognition as sr
            self._recognizer = sr.Recognizer()
            self._microphone = sr.Microphone()
            
            # Calibrate for ambient noise
            with self._microphone as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=1)
                
        except ImportError:
            logger.warning("speech_recognition not available")
        except Exception as e:
            logger.error(f"Speech recognition setup failed: {e}")
    
    def on_speech(
        self,
        callback: Callable[[str, float], None]
    ):
        """Set callback for recognized speech."""
        self._callback = callback
    
    def start(self):
        """Start listening."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop listening."""
        self._running = False
    
    def _listen_loop(self):
        """Main listening loop."""
        if not self._recognizer or not self._microphone:
            logger.error("Speech recognition not available")
            return
        
        import speech_recognition as sr
        
        logger.info("Voice recognition started")
        
        while self._running:
            try:
                with self._microphone as source:
                    audio = self._recognizer.listen(
                        source,
                        timeout=1,
                        phrase_time_limit=5
                    )
                
                # Recognize
                text = self._recognizer.recognize_google(audio)
                confidence = 0.8  # Google doesn't return confidence
                
                # Check wake word
                if self._wake_word:
                    if self._wake_word.lower() not in text.lower():
                        continue
                    # Remove wake word from text
                    text = re.sub(
                        rf'\b{re.escape(self._wake_word)}\b',
                        '',
                        text,
                        flags=re.IGNORECASE
                    ).strip()
                
                if text and self._callback:
                    self._callback(text, confidence)
                    
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                logger.error(f"Speech recognition API error: {e}")
                time.sleep(1)
            except Exception as e:
                logger.error(f"Recognition error: {e}")
                time.sleep(0.5)


class VoiceGameController:
    """Main voice command controller for gaming."""
    
    def __init__(
        self,
        wake_word: Optional[str] = None
    ):
        """
        Initialize controller.
        
        Args:
            wake_word: Optional wake word to require
        """
        self._matcher = CommandMatcher()
        self._executor = ActionExecutor()
        self._recognizer = VoiceRecognizer(wake_word)
        
        self._context: Optional[str] = None
        self._enabled = True
        
        # Callbacks
        self._on_command: Optional[Callable[[VoiceCommand], None]] = None
        self._on_speech: Optional[Callable[[str], None]] = None
        
        # Setup recognition callback
        self._recognizer.on_speech(self._handle_speech)
    
    def register_command(
        self,
        name: str,
        triggers: Optional[List[str]] = None,
        action: Optional[str] = None,
        priority: CommandPriority = CommandPriority.NORMAL,
        cooldown: float = 0.0,
        context: Optional[str] = None
    ):
        """
        Register a voice command.
        
        Args:
            name: Command name
            triggers: Phrases that trigger (default: [name])
            action: Action string (e.g., "key:r", "macro:reload_weapon")
            priority: Command priority
            cooldown: Cooldown in seconds
            context: Required context
        """
        triggers = triggers or [name]
        
        # Parse action string
        game_action = self._parse_action(action) if action else GameAction(
            action_type=ActionType.CUSTOM
        )
        
        command = VoiceCommand(
            name=name,
            triggers=triggers,
            action=game_action,
            priority=priority,
            cooldown=cooldown,
            requires_context=context
        )
        
        self._matcher.register(command)
        logger.info(f"Registered voice command: {name}")
    
    def _parse_action(self, action_str: str) -> GameAction:
        """Parse action string into GameAction."""
        parts = action_str.split(":")
        action_type = parts[0].lower()
        
        if action_type == "key":
            return GameAction(
                action_type=ActionType.KEY_PRESS,
                key=parts[1] if len(parts) > 1 else None
            )
        
        elif action_type == "hold":
            return GameAction(
                action_type=ActionType.KEY_HOLD,
                key=parts[1] if len(parts) > 1 else None,
                duration=float(parts[2]) if len(parts) > 2 else 0.5
            )
        
        elif action_type == "click":
            return GameAction(
                action_type=ActionType.MOUSE_CLICK,
                button=parts[1] if len(parts) > 1 else "left"
            )
        
        elif action_type == "move":
            return GameAction(
                action_type=ActionType.MOUSE_MOVE,
                x=int(parts[1]) if len(parts) > 1 else None,
                y=int(parts[2]) if len(parts) > 2 else None
            )
        
        else:
            return GameAction(action_type=ActionType.CUSTOM)
    
    def register_macro(
        self,
        name: str,
        triggers: List[str],
        actions: List[str]
    ):
        """
        Register a macro command.
        
        Args:
            name: Macro name
            triggers: Trigger phrases
            actions: List of action strings
        """
        sequence = [self._parse_action(a) for a in actions]
        
        command = VoiceCommand(
            name=name,
            triggers=triggers,
            action=GameAction(
                action_type=ActionType.MACRO,
                sequence=sequence
            )
        )
        
        self._matcher.register(command)
    
    def unregister_command(self, name: str):
        """Unregister a command."""
        self._matcher.unregister(name)
    
    def set_context(self, context: Optional[str]):
        """Set current game context."""
        self._context = context
    
    def enable(self):
        """Enable voice commands."""
        self._enabled = True
    
    def disable(self):
        """Disable voice commands."""
        self._enabled = False
    
    def start(self):
        """Start listening for voice commands."""
        self._recognizer.start()
    
    def stop(self):
        """Stop listening."""
        self._recognizer.stop()
    
    def on_command(
        self,
        callback: Callable[[VoiceCommand], None]
    ):
        """Set callback for when commands are executed."""
        self._on_command = callback
    
    def on_speech(
        self,
        callback: Callable[[str], None]
    ):
        """Set callback for all recognized speech."""
        self._on_speech = callback
    
    def _handle_speech(self, text: str, confidence: float):
        """Handle recognized speech."""
        if not self._enabled:
            return
        
        if self._on_speech:
            self._on_speech(text)
        
        # Match to command
        result = self._matcher.match(text, self._context)
        
        if result:
            command, score = result
            
            # Update command stats
            command.last_used = time.time()
            command.use_count += 1
            
            # Execute
            self._executor.execute(command.action)
            
            if self._on_command:
                self._on_command(command)
            
            logger.debug(f"Executed command: {command.name} (score: {score:.2f})")
    
    def process_text(self, text: str):
        """Process text as if it was spoken."""
        self._handle_speech(text, 1.0)
    
    def get_commands(self) -> List[str]:
        """Get list of registered commands."""
        return list(self._matcher._commands.keys())


# Common game command presets
PRESET_FPS = {
    "reload": "key:r",
    "jump": "key:space",
    "crouch": "key:c",
    "prone": "key:z",
    "sprint": "hold:shift:2.0",
    "use": "key:e",
    "melee": "key:v",
    "grenade": "key:g",
    "primary weapon": "key:1",
    "secondary weapon": "key:2",
    "switch weapon": "key:q"
}

PRESET_RTS = {
    "select all": "key:ctrl+a",
    "group 1": "key:1",
    "group 2": "key:2",
    "attack": "key:a",
    "stop": "key:s",
    "hold position": "key:h",
    "patrol": "key:p"
}


# Global instance
_controller: Optional[VoiceGameController] = None


def get_voice_controller(
    wake_word: Optional[str] = None
) -> VoiceGameController:
    """Get or create global voice controller."""
    global _controller
    if _controller is None:
        _controller = VoiceGameController(wake_word)
    return _controller
