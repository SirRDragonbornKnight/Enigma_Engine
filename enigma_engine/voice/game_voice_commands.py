"""
Voice Game Commands for Enigma AI Engine

Control games using voice commands like "go left", "jump", "use item".

Features:
- Natural language command recognition
- Game-specific command mappings
- Continuous listening mode
- Hotword activation
- Combo command support

Usage:
    from enigma_engine.voice.game_voice_commands import VoiceGameController
    
    controller = VoiceGameController(game="minecraft")
    controller.start_listening()
    
    # Say "go forward" -> presses W
    # Say "jump" -> presses Space
    # Say "use item" -> presses E
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try imports
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    sr = None

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    pyautogui = None

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    keyboard = None

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


class CommandType(Enum):
    """Types of game commands."""
    MOVEMENT = auto()
    ACTION = auto()
    COMBAT = auto()
    INVENTORY = auto()
    COMMUNICATION = auto()
    CAMERA = auto()
    MENU = auto()
    CUSTOM = auto()


class InputMethod(Enum):
    """Input method for commands."""
    KEYBOARD = auto()
    MOUSE = auto()
    MOUSE_MOVE = auto()
    COMBO = auto()


@dataclass
class VoiceCommand:
    """A voice-activated game command."""
    # Trigger phrases (any of these will activate)
    phrases: List[str]
    
    # Action to take
    action: str  # Key to press or action name
    
    # Command metadata
    command_type: CommandType = CommandType.ACTION
    input_method: InputMethod = InputMethod.KEYBOARD
    
    # Timing
    hold_duration: float = 0.0  # 0 = tap, >0 = hold for seconds
    repeat_count: int = 1
    
    # For combo commands
    combo_keys: List[str] = field(default_factory=list)
    
    # For mouse commands
    mouse_button: str = "left"  # left, right, middle
    mouse_coords: Optional[Tuple[int, int]] = None
    
    # Feedback
    confirmation_phrase: str = ""  # Optional voice confirmation
    
    # Internal
    cooldown: float = 0.1  # Seconds between repeats
    last_used: float = 0.0


@dataclass
class GameProfile:
    """Game-specific command configuration."""
    name: str
    commands: List[VoiceCommand] = field(default_factory=list)
    
    # Custom key mappings
    key_map: Dict[str, str] = field(default_factory=dict)
    
    # Settings
    hold_modifier: str = ""  # Key to hold while speaking
    hotword: str = "enigma"
    require_hotword: bool = False


# Default movement commands (universal)
DEFAULT_MOVEMENT_COMMANDS = [
    VoiceCommand(
        phrases=["go forward", "forward", "move forward", "walk forward"],
        action="w",
        command_type=CommandType.MOVEMENT,
        hold_duration=0.5
    ),
    VoiceCommand(
        phrases=["go back", "backward", "move back", "walk back", "retreat"],
        action="s",
        command_type=CommandType.MOVEMENT,
        hold_duration=0.5
    ),
    VoiceCommand(
        phrases=["go left", "left", "strafe left", "move left"],
        action="a",
        command_type=CommandType.MOVEMENT,
        hold_duration=0.3
    ),
    VoiceCommand(
        phrases=["go right", "right", "strafe right", "move right"],
        action="d",
        command_type=CommandType.MOVEMENT,
        hold_duration=0.3
    ),
    VoiceCommand(
        phrases=["jump", "hop", "leap"],
        action="space",
        command_type=CommandType.MOVEMENT
    ),
    VoiceCommand(
        phrases=["crouch", "duck", "sneak"],
        action="ctrl",
        command_type=CommandType.MOVEMENT,
        hold_duration=0.5
    ),
    VoiceCommand(
        phrases=["sprint", "run", "dash"],
        action="shift",
        command_type=CommandType.MOVEMENT,
        hold_duration=1.0
    ),
    VoiceCommand(
        phrases=["stop", "halt", "freeze"],
        action="",  # Release all movement
        command_type=CommandType.MOVEMENT
    ),
]

# Default action commands
DEFAULT_ACTION_COMMANDS = [
    VoiceCommand(
        phrases=["attack", "hit", "strike", "punch"],
        action="left_click",
        command_type=CommandType.COMBAT,
        input_method=InputMethod.MOUSE
    ),
    VoiceCommand(
        phrases=["block", "shield", "defend"],
        action="right_click",
        command_type=CommandType.COMBAT,
        input_method=InputMethod.MOUSE,
        hold_duration=0.5
    ),
    VoiceCommand(
        phrases=["use", "interact", "activate", "pick up"],
        action="e",
        command_type=CommandType.ACTION
    ),
    VoiceCommand(
        phrases=["inventory", "bag", "items"],
        action="i",
        command_type=CommandType.INVENTORY
    ),
    VoiceCommand(
        phrases=["reload", "recharge"],
        action="r",
        command_type=CommandType.COMBAT
    ),
    VoiceCommand(
        phrases=["map", "show map"],
        action="m",
        command_type=CommandType.MENU
    ),
    VoiceCommand(
        phrases=["pause", "menu", "escape"],
        action="escape",
        command_type=CommandType.MENU
    ),
]

# Game-specific profiles
GAME_PROFILES: Dict[str, GameProfile] = {
    "minecraft": GameProfile(
        name="Minecraft",
        commands=[
            VoiceCommand(
                phrases=["mine", "dig", "break"],
                action="left_click",
                input_method=InputMethod.MOUSE,
                hold_duration=1.0
            ),
            VoiceCommand(
                phrases=["place", "build", "put"],
                action="right_click",
                input_method=InputMethod.MOUSE
            ),
            VoiceCommand(
                phrases=["drop", "throw"],
                action="q",
                command_type=CommandType.INVENTORY
            ),
            VoiceCommand(
                phrases=["hotbar 1", "slot 1", "first slot"],
                action="1",
                command_type=CommandType.INVENTORY
            ),
            VoiceCommand(
                phrases=["hotbar 2", "slot 2", "second slot"],
                action="2",
                command_type=CommandType.INVENTORY
            ),
            VoiceCommand(
                phrases=["hotbar 3", "slot 3", "third slot"],
                action="3",
                command_type=CommandType.INVENTORY
            ),
            VoiceCommand(
                phrases=["craft", "crafting", "crafting table"],
                action="c",
                command_type=CommandType.INVENTORY
            ),
            VoiceCommand(
                phrases=["fly", "toggle fly"],
                action="space",
                command_type=CommandType.MOVEMENT,
                repeat_count=2,
                cooldown=0.1
            ),
        ]
    ),
    
    "fps": GameProfile(
        name="FPS Generic",
        commands=[
            VoiceCommand(
                phrases=["aim", "scope", "zoom"],
                action="right_click",
                input_method=InputMethod.MOUSE,
                hold_duration=0.5
            ),
            VoiceCommand(
                phrases=["fire", "shoot"],
                action="left_click",
                input_method=InputMethod.MOUSE
            ),
            VoiceCommand(
                phrases=["grenade", "throw grenade", "frag"],
                action="g",
                command_type=CommandType.COMBAT
            ),
            VoiceCommand(
                phrases=["knife", "melee"],
                action="v",
                command_type=CommandType.COMBAT
            ),
            VoiceCommand(
                phrases=["prone", "lay down"],
                action="z",
                command_type=CommandType.MOVEMENT
            ),
            VoiceCommand(
                phrases=["lean left", "peek left"],
                action="q",
                command_type=CommandType.MOVEMENT,
                hold_duration=0.5
            ),
            VoiceCommand(
                phrases=["lean right", "peek right"],
                action="e",
                command_type=CommandType.MOVEMENT,
                hold_duration=0.5
            ),
            VoiceCommand(
                phrases=["weapon 1", "primary", "primary weapon"],
                action="1",
                command_type=CommandType.INVENTORY
            ),
            VoiceCommand(
                phrases=["weapon 2", "secondary", "pistol"],
                action="2",
                command_type=CommandType.INVENTORY
            ),
        ]
    ),
    
    "rpg": GameProfile(
        name="RPG Generic",
        commands=[
            VoiceCommand(
                phrases=["spell 1", "first spell", "ability 1"],
                action="1",
                command_type=CommandType.COMBAT
            ),
            VoiceCommand(
                phrases=["spell 2", "second spell", "ability 2"],
                action="2",
                command_type=CommandType.COMBAT
            ),
            VoiceCommand(
                phrases=["spell 3", "third spell", "ability 3"],
                action="3",
                command_type=CommandType.COMBAT
            ),
            VoiceCommand(
                phrases=["ultimate", "ult", "special"],
                action="q",
                command_type=CommandType.COMBAT
            ),
            VoiceCommand(
                phrases=["heal", "health potion", "use potion"],
                action="h",
                command_type=CommandType.ACTION
            ),
            VoiceCommand(
                phrases=["mount", "horse", "summon mount"],
                action="o",
                command_type=CommandType.ACTION
            ),
            VoiceCommand(
                phrases=["quest log", "quests", "journal"],
                action="j",
                command_type=CommandType.MENU
            ),
            VoiceCommand(
                phrases=["character", "stats", "character sheet"],
                action="c",
                command_type=CommandType.MENU
            ),
        ]
    ),
    
    "racing": GameProfile(
        name="Racing Generic",
        commands=[
            VoiceCommand(
                phrases=["gas", "accelerate", "go", "faster"],
                action="w",
                command_type=CommandType.MOVEMENT,
                hold_duration=1.0
            ),
            VoiceCommand(
                phrases=["brake", "slow", "stop"],
                action="s",
                command_type=CommandType.MOVEMENT,
                hold_duration=0.5
            ),
            VoiceCommand(
                phrases=["nitro", "boost", "turbo"],
                action="shift",
                command_type=CommandType.ACTION
            ),
            VoiceCommand(
                phrases=["handbrake", "drift", "e-brake"],
                action="space",
                command_type=CommandType.MOVEMENT
            ),
            VoiceCommand(
                phrases=["look back", "rear view", "behind"],
                action="b",
                command_type=CommandType.CAMERA
            ),
            VoiceCommand(
                phrases=["reset", "respawn", "recover"],
                action="r",
                command_type=CommandType.ACTION
            ),
        ]
    ),
}


class VoiceGameController:
    """
    Voice-controlled game input.
    """
    
    def __init__(
        self,
        game: Optional[str] = None,
        custom_commands: Optional[List[VoiceCommand]] = None,
        use_hotword: bool = False,
        hotword: str = "enigma",
        voice_feedback: bool = True
    ):
        """
        Initialize voice game controller.
        
        Args:
            game: Game profile name (minecraft, fps, rpg, racing)
            custom_commands: Additional custom commands
            use_hotword: Require hotword before command
            hotword: Activation word
            voice_feedback: Speak confirmations
        """
        # Check dependencies
        if not SR_AVAILABLE:
            logger.warning("speech_recognition not available - limited functionality")
        
        # Settings
        self._game = game
        self._use_hotword = use_hotword
        self._hotword = hotword.lower()
        self._voice_feedback = voice_feedback
        
        # Build command list
        self._commands: List[VoiceCommand] = []
        self._build_commands(game, custom_commands)
        
        # State
        self._listening = False
        self._listen_thread: Optional[threading.Thread] = None
        self._paused = False
        
        # Speech recognition
        if SR_AVAILABLE:
            self._recognizer = sr.Recognizer()
            self._microphone = sr.Microphone()
        else:
            self._recognizer = None
            self._microphone = None
        
        # TTS engine
        self._tts_engine = None
        if TTS_AVAILABLE and voice_feedback:
            try:
                self._tts_engine = pyttsx3.init()
                self._tts_engine.setProperty('rate', 180)
            except Exception:
                pass
        
        # Callbacks
        self._on_command: Optional[Callable[[str, VoiceCommand], None]] = None
        self._on_speech: Optional[Callable[[str], None]] = None
    
    def _build_commands(
        self,
        game: Optional[str],
        custom: Optional[List[VoiceCommand]]
    ):
        """Build the command list."""
        # Start with default commands
        self._commands.extend(DEFAULT_MOVEMENT_COMMANDS)
        self._commands.extend(DEFAULT_ACTION_COMMANDS)
        
        # Add game-specific
        if game and game.lower() in GAME_PROFILES:
            profile = GAME_PROFILES[game.lower()]
            self._commands.extend(profile.commands)
            logger.info(f"Loaded profile: {profile.name}")
        
        # Add custom commands
        if custom:
            self._commands.extend(custom)
        
        # Build phrase index for fast lookup
        self._phrase_index: Dict[str, VoiceCommand] = {}
        for cmd in self._commands:
            for phrase in cmd.phrases:
                self._phrase_index[phrase.lower()] = cmd
    
    def add_command(self, command: VoiceCommand):
        """Add a new voice command."""
        self._commands.append(command)
        for phrase in command.phrases:
            self._phrase_index[phrase.lower()] = command
    
    def remove_command(self, phrase: str):
        """Remove a command by phrase."""
        phrase = phrase.lower()
        if phrase in self._phrase_index:
            cmd = self._phrase_index[phrase]
            # Remove all phrases for this command
            for p in cmd.phrases:
                self._phrase_index.pop(p.lower(), None)
            # Remove command
            if cmd in self._commands:
                self._commands.remove(cmd)
    
    def start_listening(self, blocking: bool = False):
        """
        Start listening for voice commands.
        
        Args:
            blocking: If True, run in current thread
        """
        if not SR_AVAILABLE:
            logger.error("speech_recognition not installed")
            return
        
        self._listening = True
        
        if blocking:
            self._listen_loop()
        else:
            self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self._listen_thread.start()
            logger.info("Voice command listener started")
    
    def stop_listening(self):
        """Stop the voice command listener."""
        self._listening = False
        if self._listen_thread:
            self._listen_thread.join(timeout=2)
    
    def pause(self):
        """Pause command processing."""
        self._paused = True
    
    def resume(self):
        """Resume command processing."""
        self._paused = False
    
    def _listen_loop(self):
        """Main listening loop."""
        with self._microphone as source:
            # Adjust for ambient noise
            logger.info("Adjusting for ambient noise...")
            self._recognizer.adjust_for_ambient_noise(source, duration=1)
            logger.info("Ready for voice commands")
            
            while self._listening:
                try:
                    # Listen for speech
                    audio = self._recognizer.listen(
                        source,
                        timeout=5,
                        phrase_time_limit=5
                    )
                    
                    # Recognize
                    text = self._recognize(audio)
                    
                    if text:
                        self._process_speech(text)
                        
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    logger.debug(f"Listen error: {e}")
                    continue
    
    def _recognize(self, audio) -> Optional[str]:
        """Recognize speech from audio."""
        try:
            # Try Google Speech Recognition (free, no API key)
            text = self._recognizer.recognize_google(audio)
            return text.lower()
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            logger.error(f"Recognition service error: {e}")
            return None
    
    def _process_speech(self, text: str):
        """Process recognized speech."""
        logger.debug(f"Heard: {text}")
        
        # Callback
        if self._on_speech:
            self._on_speech(text)
        
        if self._paused:
            return
        
        # Check for hotword if required
        if self._use_hotword:
            if self._hotword not in text:
                return
            # Remove hotword from text
            text = text.replace(self._hotword, "").strip()
        
        # Find matching command
        command = self._find_command(text)
        
        if command:
            self._execute_command(command, text)
    
    def _find_command(self, text: str) -> Optional[VoiceCommand]:
        """Find a matching command for the text."""
        text = text.lower().strip()
        
        # Exact match first
        if text in self._phrase_index:
            return self._phrase_index[text]
        
        # Fuzzy match - check if any phrase is in the text
        for phrase, command in self._phrase_index.items():
            if phrase in text or text in phrase:
                return command
        
        # Word overlap match
        text_words = set(text.split())
        best_match = None
        best_score = 0
        
        for phrase, command in self._phrase_index.items():
            phrase_words = set(phrase.split())
            overlap = len(text_words & phrase_words)
            if overlap > best_score and overlap >= len(phrase_words) * 0.6:
                best_score = overlap
                best_match = command
        
        return best_match
    
    def _execute_command(self, command: VoiceCommand, original_text: str):
        """Execute a voice command."""
        # Check cooldown
        now = time.time()
        if now - command.last_used < command.cooldown:
            return
        command.last_used = now
        
        logger.info(f"Executing: {command.action} (from '{original_text}')")
        
        # Callback
        if self._on_command:
            self._on_command(original_text, command)
        
        # Execute based on input method
        if command.input_method == InputMethod.KEYBOARD:
            self._do_keyboard(command)
        elif command.input_method == InputMethod.MOUSE:
            self._do_mouse(command)
        elif command.input_method == InputMethod.MOUSE_MOVE:
            self._do_mouse_move(command)
        elif command.input_method == InputMethod.COMBO:
            self._do_combo(command)
        
        # Voice feedback
        if self._voice_feedback and command.confirmation_phrase and self._tts_engine:
            try:
                self._tts_engine.say(command.confirmation_phrase)
                self._tts_engine.runAndWait()
            except Exception:
                pass
    
    def _do_keyboard(self, command: VoiceCommand):
        """Execute keyboard command."""
        if not command.action:
            # Stop command - release movement keys
            if KEYBOARD_AVAILABLE:
                for key in ['w', 'a', 's', 'd', 'space', 'shift', 'ctrl']:
                    try:
                        keyboard.release(key)
                    except Exception:
                        pass
            return
        
        key = command.action
        
        for _ in range(command.repeat_count):
            if KEYBOARD_AVAILABLE:
                if command.hold_duration > 0:
                    keyboard.press(key)
                    time.sleep(command.hold_duration)
                    keyboard.release(key)
                else:
                    keyboard.press_and_release(key)
            elif PYAUTOGUI_AVAILABLE:
                if command.hold_duration > 0:
                    pyautogui.keyDown(key)
                    time.sleep(command.hold_duration)
                    pyautogui.keyUp(key)
                else:
                    pyautogui.press(key)
            
            if command.repeat_count > 1:
                time.sleep(command.cooldown)
    
    def _do_mouse(self, command: VoiceCommand):
        """Execute mouse command."""
        if not PYAUTOGUI_AVAILABLE and not KEYBOARD_AVAILABLE:
            return
        
        button = command.action.replace("_click", "") if "_click" in command.action else command.mouse_button
        
        if PYAUTOGUI_AVAILABLE:
            if command.mouse_coords:
                pyautogui.moveTo(command.mouse_coords[0], command.mouse_coords[1])
            
            if command.hold_duration > 0:
                pyautogui.mouseDown(button=button)
                time.sleep(command.hold_duration)
                pyautogui.mouseUp(button=button)
            else:
                pyautogui.click(button=button)
    
    def _do_mouse_move(self, command: VoiceCommand):
        """Execute mouse move command."""
        if PYAUTOGUI_AVAILABLE and command.mouse_coords:
            pyautogui.moveTo(command.mouse_coords[0], command.mouse_coords[1])
    
    def _do_combo(self, command: VoiceCommand):
        """Execute combo key command."""
        if not command.combo_keys:
            return
        
        if KEYBOARD_AVAILABLE:
            # Press all keys
            for key in command.combo_keys:
                keyboard.press(key)
            
            time.sleep(0.1)
            
            # Release in reverse order
            for key in reversed(command.combo_keys):
                keyboard.release(key)
        elif PYAUTOGUI_AVAILABLE:
            pyautogui.hotkey(*command.combo_keys)
    
    def on_command(self, callback: Callable[[str, VoiceCommand], None]):
        """Set callback for when a command is executed."""
        self._on_command = callback
    
    def on_speech(self, callback: Callable[[str], None]):
        """Set callback for when speech is recognized."""
        self._on_speech = callback
    
    def process_text(self, text: str):
        """
        Process text as if it was spoken.
        Useful for testing without microphone.
        """
        self._process_speech(text.lower())
    
    def get_commands(self, category: Optional[CommandType] = None) -> List[VoiceCommand]:
        """Get list of available commands."""
        if category:
            return [c for c in self._commands if c.command_type == category]
        return self._commands.copy()
    
    def get_available_games(self) -> List[str]:
        """Get list of available game profiles."""
        return list(GAME_PROFILES.keys())


# Convenience functions

_controller: Optional[VoiceGameController] = None

def get_voice_controller(game: Optional[str] = None) -> VoiceGameController:
    """Get or create the global voice controller."""
    global _controller
    if _controller is None or game:
        _controller = VoiceGameController(game=game)
    return _controller

def start_voice_gaming(game: str = "fps"):
    """Quick start voice gaming for a game type."""
    controller = get_voice_controller(game)
    controller.start_listening()
    return controller

def add_voice_command(
    phrases: List[str],
    action: str,
    hold_duration: float = 0.0,
    **kwargs
):
    """Quick function to add a voice command."""
    controller = get_voice_controller()
    command = VoiceCommand(
        phrases=phrases,
        action=action,
        hold_duration=hold_duration,
        **kwargs
    )
    controller.add_command(command)
