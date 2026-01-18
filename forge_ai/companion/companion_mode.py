"""
Companion Mode - Always-on AI assistant that watches, comments, and helps.

This is the heart of ForgeAI's "lifelike" behavior - an AI that:
- Watches the screen and comments on what's happening
- Looks up information proactively
- Talks naturally (voice output)
- Controls its avatar expressively
- Helps with files and system maintenance
- Initiates conversation, not just responds

The AI becomes a desktop companion, not just a chatbot.

Usage:
    from forge_ai.companion import CompanionMode
    
    companion = CompanionMode()
    companion.start()
    
    # Companion will now:
    # - Watch screen every few seconds
    # - Comment on interesting things
    # - React through avatar
    # - Offer help proactively
"""

import threading
import time
import random
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum


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
    quiet_hours: List[int] = field(default_factory=lambda: [23, 0, 1, 2, 3, 4, 5, 6])
    
    # Things that trigger immediate attention
    attention_triggers: List[str] = field(default_factory=lambda: [
        "error", "warning", "failed", "crash", "exception",
        "update available", "download complete", "install"
    ])
    
    # Topics the companion is interested in (will comment more)
    interests: List[str] = field(default_factory=lambda: [
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
        self._recent_observations: List[Dict] = []
        self._conversation_history: List[Dict] = []
        
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
                
                # Maybe comment on something
                if now - self._last_comment >= self.config.comment_interval:
                    if random.random() < self.config.comment_chance:
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
        
        if is_interesting or random.random() < 0.1:  # Small chance for any topic
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
    
    def _on_new_window(self, window_info: Dict):
        """React to a new window appearing."""
        if self.config.avatar_reactions and self._avatar_callback:
            self._avatar_callback("emotion", "curious")
            self._avatar_callback("gesture", "look_at")
    
    def _generate_comment(self) -> Optional[str]:
        """Generate a contextual comment about what's on screen."""
        context = self._current_screen_context.lower()
        
        # Context-based comments
        if "code" in context or "python" in context or "visual studio" in context:
            comments = [
                "Nice coding session!",
                "That's some interesting code.",
                "Making good progress!",
                "Need a second pair of eyes on that?",
            ]
        elif "youtube" in context or "video" in context:
            comments = [
                "Enjoying a video?",
                "Good pick!",
                "Taking a break? Nice.",
            ]
        elif "game" in context or "steam" in context:
            comments = [
                "Game time! Have fun!",
                "Nice game choice.",
                "Don't forget to save!",
            ]
        elif "discord" in context or "chat" in context:
            comments = [
                "Chatting with friends?",
                "Popular today!",
            ]
        elif "browser" in context or "chrome" in context or "firefox" in context:
            comments = [
                "Browsing something interesting?",
                "Finding what you need?",
            ]
        else:
            # Generic comments
            comments = [
                "Working hard!",
                "Need anything?",
                "I'm here if you need me.",
            ]
        
        return random.choice(comments) if comments else None
    
    def _detect_emotion(self, text: str) -> str:
        """Detect emotion from text for avatar."""
        text_lower = text.lower()
        
        if any(w in text_lower for w in ["happy", "great", "awesome", "excellent", "wonderful"]):
            return "happy"
        elif any(w in text_lower for w in ["sorry", "unfortunately", "can't", "error"]):
            return "sad"
        elif any(w in text_lower for w in ["think", "perhaps", "maybe", "consider"]):
            return "thinking"
        elif any(w in text_lower for w in ["!", "wow", "amazing"]):
            return "excited"
        elif any(w in text_lower for w in ["?", "curious", "wonder"]):
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
    
    def _get_active_window(self) -> Optional[Dict]:
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
                pass
        
        return None
    
    def _get_monitor_info(self, monitor_handle) -> Optional[Dict]:
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
            pass
        return None
    
    def _get_all_monitors(self) -> List[Dict]:
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
            pass
        
        return monitors
    
    def _load_state(self):
        """Load companion state from disk."""
        state_file = self._storage_path / "companion_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    self._seen_windows = set(data.get("seen_windows", []))
                    self._commented_on = set(data.get("commented_on", []))
            except Exception:
                pass
    
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
            pass


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
