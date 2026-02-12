"""
Companion Mode - Always-on AI assistant that watches, comments, and helps.

This is the heart of Enigma AI Engine's "lifelike" behavior - an AI that:
- Watches the screen and comments on what's happening
- Looks up information proactively
- Talks naturally (voice output)
- Controls its avatar expressively
- Helps with files and system maintenance
- Initiates conversation, not just responds

The AI becomes a desktop companion, not just a chatbot.

Usage:
    from enigma_engine.companion import CompanionMode
    
    companion = CompanionMode()
    companion.start()
    
    # Companion will now:
    # - Watch screen every few seconds
    # - Comment on interesting things
    # - React through avatar
    # - Offer help proactively
"""

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


class CompanionState(Enum):
    """Current state of the companion."""
    IDLE = "idle"              # Watching, waiting
    OBSERVING = "observing"    # Actively looking at screen
    THINKING = "thinking"      # Processing what it sees
    COMMENTING = "commenting"  # Speaking/typing a comment
    HELPING = "helping"        # Actively assisting
    RESEARCHING = "researching"  # Looking something up
    RESTING = "resting"        # Low activity mode (late night)


@dataclass
class CompanionConfig:
    """Configuration for companion behavior."""
    enabled: bool = True
    
    # How often to observe screen (seconds)
    observe_interval: float = 10.0
    
    # How often to potentially comment (seconds)  
    comment_interval: float = 30.0
    
    # Chance to comment on something interesting (0-1)
    comment_chance: float = 0.3
    
    # Be more quiet during these hours
    quiet_hours: list[int] = field(default_factory=lambda: [23, 0, 1, 2, 3, 4, 5, 6])
    
    # Things that trigger immediate attention
    attention_triggers: list[str] = field(default_factory=lambda: [
        "error", "warning", "failed", "crash", "exception",
        "update available", "download complete", "install"
    ])
    
    # Topics the companion is interested in (will comment more)
    interests: list[str] = field(default_factory=lambda: [
        "programming", "code", "python", "AI", "game",
        "music", "video", "news", "weather"
    ])
    
    # Voice settings
    speak_comments: bool = True
    voice_volume: float = 0.8
    
    # Avatar reactivity
    avatar_reactions: bool = True
    avatar_follows_focus: bool = True
    
    # Multi-monitor settings
    multi_monitor_aware: bool = True  # Know which monitor window is on
    preferred_monitor: int = -1  # -1 = follow active, 0+ = specific monitor index
        # Proactive help
    offer_help: bool = True
    help_cooldown: float = 300.0  # Don't offer help too often


class CompanionMode:
    """
    The always-on AI companion.
    
    Watches the screen, comments on things, helps proactively,
    and behaves like a lifelike assistant.
    """
    
    def __init__(self, config: Optional[CompanionConfig] = None):
        self.config = config or CompanionConfig()
        
        # State
        self._state = CompanionState.IDLE
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Timing
        self._last_observe = 0.0
        self._last_comment = 0.0
        self._last_help_offer = 0.0
        
        # Context
        self._current_screen_context = ""
        self._recent_observations: list[dict] = []
        self._conversation_history: list[dict] = []
        
        # Connections to other systems
        self._chat_callback: Optional[Callable] = None
        self._avatar_callback: Optional[Callable] = None
        self._voice_callback: Optional[Callable] = None
        self._tool_executor: Optional[Any] = None
        
        # Memory
        self._seen_windows: set = set()
        self._commented_on: set = set()  # Don't repeat comments
        self._help_offered_for: set = set()
        
        # Storage
        self._storage_path = Path(__file__).parent.parent.parent / "data" / "companion"
        self._storage_path.mkdir(parents=True, exist_ok=True)
        
        self._load_state()
    
    @property
    def state(self) -> CompanionState:
        return self._state
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def connect_chat(self, callback: Callable[[str], None]):
        """Connect to chat system - callback sends message to user."""
        self._chat_callback = callback
    
    def connect_avatar(self, callback: Callable[[str, Any], None]):
        """Connect to avatar - callback controls avatar (action, value)."""
        self._avatar_callback = callback
    
    def connect_voice(self, callback: Callable[[str], None]):
        """Connect to voice system - callback speaks text."""
        self._voice_callback = callback
    
    def connect_tools(self, executor):
        """Connect to tool executor for proactive tool use."""
        self._tool_executor = executor
    
    def start(self):
        """Start companion mode."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._main_loop, daemon=True)
        self._thread.start()
        
        self._set_state(CompanionState.IDLE)
        self._say("I'm here! Let me know if you need anything.", quiet=True)
    
    def stop(self):
        """Stop companion mode."""
        self._running = False
        self._save_state()
    
    def notify_user_message(self, message: str):
        """Called when user sends a message - companion pays attention."""
        self._last_help_offer = time.time()  # Reset help cooldown
        self._set_state(CompanionState.HELPING)
        
        # Avatar reacts
        if self.config.avatar_reactions and self._avatar_callback:
            self._avatar_callback("emotion", "curious")
    
    def notify_ai_response(self, response: str):
        """Called when AI responds - companion done helping."""
        self._set_state(CompanionState.IDLE)
        
        # Avatar reacts to its own response
        if self.config.avatar_reactions and self._avatar_callback:
            # Detect emotion from response
            emotion = self._detect_emotion(response)
            self._avatar_callback("emotion", emotion)
    
    def _main_loop(self):
        """Main companion loop - observe, think, react."""
        while self._running:
            try:
                now = time.time()
                hour = datetime.now().hour
                
                # Quiet hours - less activity
                is_quiet = hour in self.config.quiet_hours
                
                if is_quiet:
                    self._set_state(CompanionState.RESTING)
                    time.sleep(60)  # Check less often at night
                    continue
                
                # Observe screen periodically
                if now - self._last_observe >= self.config.observe_interval:
                    self._observe_screen()
                    self._last_observe = now
                
                # Comment when interval reached (AI decides relevance in _maybe_comment)
                if now - self._last_comment >= self.config.comment_interval:
                    self._maybe_comment()
                    self._last_comment = now
                
                # Offer help if user seems stuck
                if self.config.offer_help:
                    if now - self._last_help_offer >= self.config.help_cooldown:
                        self._maybe_offer_help()
                
                time.sleep(1)
                
            except Exception as e:
                print(f"[Companion] Error: {e}")
                time.sleep(5)
    
    def _observe_screen(self):
        """Look at the screen and understand context."""
        self._set_state(CompanionState.OBSERVING)
        
        try:
            # Get active window info
            window_info = self._get_active_window()
            
            if window_info:
                self._current_screen_context = window_info.get("title", "")
                
                # Record observation
                observation = {
                    "time": datetime.now().isoformat(),
                    "window": window_info.get("title", ""),
                    "app": window_info.get("app", ""),
                }
                self._recent_observations.append(observation)
                
                # Keep only recent observations
                if len(self._recent_observations) > 100:
                    self._recent_observations = self._recent_observations[-50:]
                
                # Avatar looks at screen
                if self.config.avatar_follows_focus and self._avatar_callback:
                    # Look toward center of screen
                    self._avatar_callback("look_at", "500,400")
                
                # Check for attention triggers
                for trigger in self.config.attention_triggers:
                    if trigger.lower() in self._current_screen_context.lower():
                        self._handle_attention_trigger(trigger)
                        break
                
                # New window?
                if window_info.get("title") and window_info["title"] not in self._seen_windows:
                    self._seen_windows.add(window_info["title"])
                    self._on_new_window(window_info)
        
        except Exception as e:
            print(f"[Companion] Observe error: {e}")
        
        self._set_state(CompanionState.IDLE)
    
    def _maybe_comment(self):
        """Maybe make a comment about what's on screen."""
        if not self._current_screen_context:
            return
        
        # Don't repeat comments
        context_key = self._current_screen_context[:50]
        if context_key in self._commented_on:
            return
        
        self._set_state(CompanionState.THINKING)
        
        # Check if context matches interests
        is_interesting = any(
            interest.lower() in self._current_screen_context.lower()
            for interest in self.config.interests
        )
        
        if is_interesting:  # Only comment when genuinely relevant to interests
            comment = self._generate_comment()
            if comment:
                self._set_state(CompanionState.COMMENTING)
                self._say(comment)
                self._commented_on.add(context_key)
                
                # Avatar reacts
                if self._avatar_callback:
                    self._avatar_callback("gesture", "nod")
        
        self._set_state(CompanionState.IDLE)
    
    def _maybe_offer_help(self):
        """Offer help if user seems to need it."""
        # Look for signs user might need help
        help_signs = [
            ("error" in self._current_screen_context.lower(), "I see an error - need help with that?"),
            ("stackoverflow" in self._current_screen_context.lower(), "Looking something up? I can help research that."),
            ("google" in self._current_screen_context.lower(), "Need me to look something up for you?"),
        ]
        
        for condition, offer in help_signs:
            if condition:
                offer_key = offer[:20]
                if offer_key not in self._help_offered_for:
                    self._say(offer, quiet=True)
                    self._help_offered_for.add(offer_key)
                    self._last_help_offer = time.time()
                    
                    if self._avatar_callback:
                        self._avatar_callback("gesture", "wave")
                    break
    
    def _handle_attention_trigger(self, trigger: str):
        """React to something that needs attention."""
        self._set_state(CompanionState.COMMENTING)
        
        reactions = {
            "error": ("I noticed an error!", "surprised"),
            "warning": ("Heads up - there's a warning.", "thinking"),
            "failed": ("Something failed - want me to help?", "concerned"),
            "crash": ("Oh no, a crash! Need help recovering?", "surprised"),
            "exception": ("Exception detected - I can help debug if you want.", "thinking"),
            "update available": ("There's an update available!", "happy"),
            "download complete": ("Download finished!", "happy"),
            "install": ("Installing something? Let me know if you need help.", "curious"),
        }
        
        message, emotion = reactions.get(trigger, ("Something caught my attention.", "curious"))
        
        self._say(message)
        if self._avatar_callback:
            self._avatar_callback("emotion", emotion)
        
        self._set_state(CompanionState.IDLE)
    
    def _on_new_window(self, window_info: dict):
        """React to a new window appearing."""
        if self.config.avatar_reactions and self._avatar_callback:
            self._avatar_callback("emotion", "curious")
            self._avatar_callback("gesture", "look_at")
    
    def _generate_comment(self) -> Optional[str]:
        """Generate a contextual comment about what's on screen using AI."""
        context = self._current_screen_context
        
        if not context:
            return None
        
        # Try to use AI for natural comments
        try:
            from ..core.inference import EnigmaEngine
            engine = EnigmaEngine.get_instance()
            
            if engine and engine.model:
                prompt = f"""You're a friendly AI companion watching the user's screen. 
The active window is: "{context}"

Generate a SHORT, natural comment (1 sentence max) that feels like a friend noticing what you're doing.
Be casual and helpful, not robotic. Only comment if you have something genuinely useful or friendly to say.
If there's nothing interesting to comment on, respond with just "..." to stay quiet.

Your comment:"""
                
                response = engine.generate(prompt, max_gen=50, temperature=0.8)
                if response and response.strip() and response.strip() != "...":
                    return response.strip()
                return None  # AI chose to stay quiet
        except Exception:
            pass  # Intentionally silent
        
        # Fallback: Only comment on specific situations, don't force generic comments
        context_lower = context.lower()
        
        # Only speak up for genuinely notable situations
        if "error" in context_lower:
            return "I noticed an error - need help with that?"
        elif "stackoverflow" in context_lower or "github.com" in context_lower:
            return "Researching something? I can help look things up."
        
        # For everything else, stay quiet rather than say something generic
        return None
    
    def _detect_emotion(self, text: str) -> str:
        """Detect emotion from text for avatar using AI when available."""
        if not text:
            return "neutral"
        
        # Try AI-based emotion detection first
        try:
            from ..core.inference import EnigmaEngine
            engine = EnigmaEngine.get_instance()
            
            if engine and engine.model:
                prompt = f"""What emotion does this text express? Choose ONE: happy, sad, thinking, excited, curious, neutral

Text: "{text[:200]}"

Emotion:"""
                response = engine.generate(prompt, max_length=15, temperature=0.3)
                emotion = response.strip().lower().split()[0] if response else "neutral"
                
                # Validate
                valid_emotions = {"happy", "sad", "thinking", "excited", "curious", "neutral"}
                if emotion in valid_emotions:
                    return emotion
        except Exception:
            pass  # Intentionally silent
        
        # Fallback: Simple keyword detection
        text_lower = text.lower()
        
        if any(w in text_lower for w in ["happy", "great", "awesome", "excellent", "wonderful", "love"]):
            return "happy"
        elif any(w in text_lower for w in ["sorry", "unfortunately", "can't", "error", "failed"]):
            return "sad"
        elif any(w in text_lower for w in ["think", "perhaps", "maybe", "consider", "hmm"]):
            return "thinking"
        elif any(w in text_lower for w in ["!", "wow", "amazing", "incredible"]):
            return "excited"
        elif any(w in text_lower for w in ["?", "curious", "wonder", "what", "how", "why"]):
            return "curious"
        else:
            return "neutral"
    
    def _say(self, message: str, quiet: bool = False):
        """Say something to the user (chat and/or voice)."""
        # Send to chat
        if self._chat_callback:
            self._chat_callback(message)
        
        # Speak if enabled and not quiet mode
        if self.config.speak_comments and not quiet and self._voice_callback:
            self._voice_callback(message)
    
    def _set_state(self, state: CompanionState):
        """Update companion state."""
        if state != self._state:
            self._state = state
    
    def _get_active_window(self) -> Optional[dict]:
        """Get info about the currently active window."""
        import sys
        
        if sys.platform == 'win32':
            try:
                import ctypes
                from ctypes import wintypes
                
                user32 = ctypes.windll.user32
                hwnd = user32.GetForegroundWindow()
                
                length = user32.GetWindowTextLengthW(hwnd)
                title = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, title, length + 1)
                
                # Get process name
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                
                # Get window position to determine which monitor
                rect = wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                
                # Get monitor info for this window
                monitor = user32.MonitorFromWindow(hwnd, 2)  # MONITOR_DEFAULTTONEAREST
                monitor_info = self._get_monitor_info(monitor)
                
                return {
                    "title": title.value,
                    "hwnd": hwnd,
                    "pid": pid.value,
                    "rect": {
                        "left": rect.left,
                        "top": rect.top,
                        "right": rect.right,
                        "bottom": rect.bottom,
                    },
                    "monitor": monitor_info,
                }
            except Exception:
                pass  # Intentionally silent
        
        return None
    
    def _get_monitor_info(self, monitor_handle) -> Optional[dict]:
        """Get info about a specific monitor."""
        try:
            import ctypes
            from ctypes import wintypes
            
            class MONITORINFO(ctypes.Structure):
                _fields_ = [
                    ("cbSize", wintypes.DWORD),
                    ("rcMonitor", wintypes.RECT),
                    ("rcWork", wintypes.RECT),
                    ("dwFlags", wintypes.DWORD),
                ]
            
            user32 = ctypes.windll.user32
            mi = MONITORINFO()
            mi.cbSize = ctypes.sizeof(MONITORINFO)
            
            if user32.GetMonitorInfoW(monitor_handle, ctypes.byref(mi)):
                return {
                    "is_primary": bool(mi.dwFlags & 1),
                    "work_area": {
                        "left": mi.rcWork.left,
                        "top": mi.rcWork.top,
                        "right": mi.rcWork.right,
                        "bottom": mi.rcWork.bottom,
                    },
                    "full_area": {
                        "left": mi.rcMonitor.left,
                        "top": mi.rcMonitor.top,
                        "right": mi.rcMonitor.right,
                        "bottom": mi.rcMonitor.bottom,
                    },
                }
        except Exception:
            pass  # Intentionally silent
        return None
    
    def _get_all_monitors(self) -> list[dict]:
        """Get info about all connected monitors."""
        monitors = []
        
        try:
            import ctypes
            from ctypes import wintypes
            
            user32 = ctypes.windll.user32
            
            # Callback to enumerate monitors
            MONITORENUMPROC = ctypes.WINFUNCTYPE(
                ctypes.c_bool,
                ctypes.POINTER(ctypes.c_ulong),
                ctypes.POINTER(ctypes.c_ulong),
                ctypes.POINTER(wintypes.RECT),
                ctypes.c_double
            )
            
            def callback(hMonitor, hdcMonitor, lprcMonitor, dwData):
                info = self._get_monitor_info(hMonitor)
                if info:
                    monitors.append(info)
                return True
            
            user32.EnumDisplayMonitors(None, None, MONITORENUMPROC(callback), 0)
            
        except Exception:
            pass  # Intentionally silent
        
        return monitors
    
    def _load_state(self):
        """Load companion state from disk."""
        state_file = self._storage_path / "companion_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    data = json.load(f)
                    self._seen_windows = set(data.get("seen_windows", []))
                    self._commented_on = set(data.get("commented_on", []))
            except Exception:
                pass  # Intentionally silent
    
    def _save_state(self):
        """Save companion state to disk."""
        state_file = self._storage_path / "companion_state.json"
        try:
            with open(state_file, 'w') as f:
                json.dump({
                    "seen_windows": list(self._seen_windows)[-100:],
                    "commented_on": list(self._commented_on)[-100:],
                }, f)
        except Exception:
            pass  # Intentionally silent


# Singleton instance
_companion: Optional[CompanionMode] = None


def get_companion() -> CompanionMode:
    """Get the global companion instance."""
    global _companion
    if _companion is None:
        _companion = CompanionMode()
    return _companion


def start_companion():
    """Start companion mode."""
    companion = get_companion()
    companion.start()
    return companion


def stop_companion():
    """Stop companion mode."""
    if _companion:
        _companion.stop()
