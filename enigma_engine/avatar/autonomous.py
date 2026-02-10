"""
================================================================================
🤖 AUTONOMOUS AVATAR - SELF-ACTING COMPANION
================================================================================

Makes the avatar react to the screen, do things on its own, express curiosity,
and behave naturally WITHOUT explicit commands! Your AI pet comes to life!

FALLBACK SYSTEM: Only takes control when bone controller is not active.
Priority: AUTONOMOUS (50) - lower than BONE_ANIMATION (100)

📍 FILE: enigma_engine/avatar/autonomous.py
🏷️ TYPE: Autonomous Behavior System (Fallback)
🎯 MAIN CLASSES: AutonomousAvatar, AutonomousConfig, AvatarMood

┌─────────────────────────────────────────────────────────────────────────────┐
│  AUTONOMOUS BEHAVIORS:                                                      │
│                                                                             │
│  🖥️ Screen Watching  - React to what's on screen                          │
│  💤 Idle Animations  - Random movements when bored                         │
│  👁️ Curiosity        - "Look at" interesting things                         │
│  😊 Mood System      - Happy, bored, excited, sleepy...                     │
│  🧠 Memory           - Remember what it's seen                              │
│  ⏰ Time Awareness   - Sleepy at night, energetic during day               │
└─────────────────────────────────────────────────────────────────────────────┘

🎭 MOOD STATES (AvatarMood enum):
    • NEUTRAL, HAPPY, CURIOUS, BORED
    • EXCITED, SLEEPY, FOCUSED, PLAYFUL, THOUGHTFUL

⏱️ TIMING CONFIG (AutonomousConfig):
    • action_interval: 3-15 seconds between actions
    • screen_scan_interval: 5 seconds
    • get_bored_after: 300 seconds of no activity
    • sleepy_hours: [22, 23, 0, 1, 2, 3, 4, 5]
    • energetic_hours: [9, 10, 11, 14, 15, 16]

🔗 CONNECTED FILES:
    → USES:      enigma_engine/avatar/controller.py (AvatarController)
    → USES:      enigma_engine/tools/vision.py (screen capture)
    ← USED BY:   enigma_engine/gui/tabs/avatar_tab.py (GUI controls)

📖 USAGE:
    from enigma_engine.avatar import get_avatar
    from enigma_engine.avatar.autonomous import AutonomousAvatar
    
    avatar = get_avatar()
    avatar.enable()
    
    autonomous = AutonomousAvatar(avatar)
    autonomous.start()  # Avatar starts doing things on its own!
    autonomous.stop()   # Back to manual control

📖 SEE ALSO:
    • enigma_engine/avatar/controller.py      - Manual avatar control
    • enigma_engine/avatar/desktop_pet.py     - Desktop overlay window
    • enigma_engine/avatar/animation_system.py - Movement animations
    • enigma_engine/avatar/lip_sync.py        - Sync mouth to speech
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from .controller import AvatarController

# Type alias for optional AvatarController
_AvatarController = Optional['AvatarController']

logger = logging.getLogger(__name__)


class AvatarMood(Enum):
    """Avatar mood states - affects behavior."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    CURIOUS = "curious"
    BORED = "bored"
    EXCITED = "excited"
    SLEEPY = "sleepy"
    FOCUSED = "focused"
    PLAYFUL = "playful"
    THOUGHTFUL = "thoughtful"


@dataclass
class ScreenRegion:
    """A region of interest on screen."""
    x: int
    y: int
    width: int
    height: int
    content_type: str = "unknown"  # window, text, image, video, etc.
    title: str = ""
    interest_score: float = 0.5  # 0-1, how interesting


@dataclass
class AutonomousConfig:
    """Configuration for autonomous behavior."""
    enabled: bool = False
    
    # Timing
    action_interval_min: float = 3.0  # Minimum seconds between actions
    action_interval_max: float = 15.0  # Maximum seconds between actions
    screen_scan_interval: float = 5.0  # How often to look at screen
    
    # Behavior weights (higher = more likely)
    idle_animation_weight: float = 0.3
    look_around_weight: float = 0.25
    react_to_screen_weight: float = 0.2
    express_mood_weight: float = 0.15
    random_gesture_weight: float = 0.1
    
    # Screen reaction
    react_to_new_windows: bool = True
    react_to_videos: bool = True
    react_to_text: bool = True
    follow_mouse_sometimes: bool = True
    
    # Mood
    mood_change_rate: float = 0.1  # How fast mood changes
    get_bored_after: float = 300.0  # Seconds of no activity before bored
    
    # Time-based behavior
    enable_time_awareness: bool = True
    sleepy_hours: list[int] = field(default_factory=lambda: [22, 23, 0, 1, 2, 3, 4, 5])  # Night hours
    energetic_hours: list[int] = field(default_factory=lambda: [9, 10, 11, 14, 15, 16])  # Day hours
    
    # Personality influence
    personality_influence: float = 0.5  # How much personality affects behavior


class AutonomousAvatar:
    """
    Makes avatar behave autonomously - react to screen, do idle things.
    
    The avatar will:
    - Watch the screen and react to changes
    - Do idle animations and expressions
    - "Look at" interesting things
    - Change mood over time
    - Occasionally do random gestures
    - React to time of day
    - Be influenced by AI personality
    """
    
    def __init__(self, avatar: 'AvatarController', config: Optional[AutonomousConfig] = None):
        self.avatar = avatar
        self.config = config or AutonomousConfig()
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._state_lock = Lock()  # Thread safety for state changes
        
        # State
        self._mood = AvatarMood.NEUTRAL
        self._last_action_time = 0.0
        self._last_screen_scan = 0.0
        self._last_activity_time = time.time()
        self._screen_regions: list[ScreenRegion] = []
        self._current_focus: Optional[ScreenRegion] = None
        
        # Memory - what has avatar seen/done
        self._seen_windows: list[str] = []
        self._recent_actions: list[str] = []
        self._interaction_count: int = 0
        
        # Callbacks for external integration
        self._on_mood_change: list[Callable] = []
        self._on_action: list[Callable] = []
        
        # Personality link
        self._personality = None
    
    @property
    def mood(self) -> AvatarMood:
        return self._mood
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def link_personality(self, personality) -> None:
        """Link AI personality to influence avatar behavior."""
        self._personality = personality
    
    def start(self):
        """Start autonomous behavior."""
        if self._running:
            return
        
        if not self.avatar.is_enabled:
            logger.warning("Avatar must be enabled first")
            return
        
        self._running = True
        self.config.enabled = True
        self._thread = threading.Thread(target=self._behavior_loop, daemon=True)
        self._thread.start()
        logger.info("Avatar autonomous mode started")
    
    def stop(self):
        """Stop autonomous behavior."""
        self._running = False
        self.config.enabled = False
        logger.info("Avatar autonomous mode stopped")
    
    def set_mood(self, mood: AvatarMood):
        """Set avatar mood."""
        with self._state_lock:
            if mood != self._mood:
                old_mood = self._mood
                self._mood = mood
                
                # Update expression to match mood
                mood_expressions = {
                    AvatarMood.HAPPY: "happy",
                    AvatarMood.CURIOUS: "thinking",
                    AvatarMood.BORED: "neutral",
                    AvatarMood.EXCITED: "excited",
                    AvatarMood.SLEEPY: "sleeping",
                    AvatarMood.FOCUSED: "thinking",
                    AvatarMood.NEUTRAL: "neutral",
                    AvatarMood.PLAYFUL: "winking",
                    AvatarMood.THOUGHTFUL: "thinking",
                }
                # Use AUTONOMOUS priority (fallback to bone control)
                try:
                    from .controller import ControlPriority
                    self.avatar.set_expression(
                        mood_expressions.get(mood, "neutral"),
                        requester="autonomous",
                        priority=ControlPriority.AUTONOMOUS
                    )
                except TypeError:
                    # Fallback for old signature
                    self.avatar.set_expression(mood_expressions.get(mood, "neutral"))
                
                # Notify callbacks
                for cb in self._on_mood_change:
                    try:
                        cb(old_mood, mood)
                    except Exception as e:
                        logger.debug(f"Mood change callback failed: {e}")
    
    def on_mood_change(self, callback: Callable):
        """Register callback for mood changes."""
        self._on_mood_change.append(callback)
    
    def on_action(self, callback: Callable):
        """Register callback when avatar does something."""
        self._on_action.append(callback)
    
    def notify_activity(self):
        """Call this when user does something - resets boredom timer."""
        self._last_activity_time = time.time()
        self._interaction_count += 1
        if self._mood == AvatarMood.BORED:
            self.set_mood(AvatarMood.NEUTRAL)
    
    def notify_chat_message(self, is_from_user: bool = True):
        """Call when a chat message is received/sent."""
        self.notify_activity()
        if is_from_user:
            # User engaged - alternate between curious and happy based on interaction count
            if self._interaction_count % 2 == 0:
                self.set_mood(AvatarMood.CURIOUS)
            else:
                self.set_mood(AvatarMood.HAPPY)
    
    def _get_time_based_mood_modifier(self) -> Optional[AvatarMood]:
        """Get mood based on time of day."""
        if not self.config.enable_time_awareness:
            return None
        
        hour = datetime.now().hour
        
        if hour in self.config.sleepy_hours:
            return AvatarMood.SLEEPY
        elif hour in self.config.energetic_hours:
            # Be playful during energetic hours based on minute (varies naturally)
            minute = datetime.now().minute
            if minute % 4 == 0:  # Every 4th minute during energetic hours
                return AvatarMood.PLAYFUL
        
        return None
    
    def _behavior_loop(self):
        """Main autonomous behavior loop."""
        while self._running:
            try:
                current_time = time.time()
                
                # Use consistent interval (average of min/max) instead of random
                interval = (self.config.action_interval_min + self.config.action_interval_max) / 2
                
                if current_time - self._last_action_time >= interval:
                    self._do_autonomous_action()
                    self._last_action_time = current_time
                
                # Periodic screen scan
                if current_time - self._last_screen_scan >= self.config.screen_scan_interval:
                    self._scan_screen()
                    self._last_screen_scan = current_time
                
                # Check for boredom
                if current_time - self._last_activity_time > self.config.get_bored_after:
                    if self._mood != AvatarMood.BORED and self._mood != AvatarMood.SLEEPY:
                        self.set_mood(AvatarMood.BORED)
                
                # Time-based mood check - apply consistently when conditions met
                time_mood = self._get_time_based_mood_modifier()
                if time_mood and self._mood != time_mood:
                    # Only change mood once per 5 minutes to avoid constant switching
                    last_mood_change = getattr(self, '_last_mood_change', 0)
                    if current_time - last_mood_change > 300:  # 5 minutes
                        self.set_mood(time_mood)
                        self._last_mood_change = current_time
                
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error in behavior loop: {e}")
                time.sleep(1)
    
    def _do_autonomous_action(self):
        """Choose and perform an autonomous action."""
        # Adjust weights based on mood
        weights = self._get_adjusted_weights()
        
        actions = [
            (weights['idle'], self._do_idle_animation),
            (weights['look'], self._look_around),
            (weights['react'], self._react_to_screen),
            (weights['express'], self._express_mood),
            (weights['gesture'], self._random_gesture),
        ]
        
        # Sort actions by weight (highest priority first) and cycle through
        actions.sort(key=lambda x: x[0], reverse=True)
        
        # Track action index for cycling
        if not hasattr(self, '_action_cycle_index'):
            self._action_cycle_index = 0
        
        # Execute current action in cycle
        if actions:
            _, action = actions[self._action_cycle_index % len(actions)]
            action()
            self._action_cycle_index = (self._action_cycle_index + 1) % len(actions)
    
    def _get_adjusted_weights(self) -> dict[str, float]:
        """Get behavior weights adjusted for current mood."""
        base = {
            'idle': self.config.idle_animation_weight,
            'look': self.config.look_around_weight,
            'react': self.config.react_to_screen_weight,
            'express': self.config.express_mood_weight,
            'gesture': self.config.random_gesture_weight,
        }
        
        # Adjust based on mood
        if self._mood == AvatarMood.BORED:
            base['idle'] *= 1.5
            base['look'] *= 0.5
        elif self._mood == AvatarMood.CURIOUS:
            base['look'] *= 2.0
            base['react'] *= 1.5
        elif self._mood == AvatarMood.PLAYFUL:
            base['gesture'] *= 2.0
            base['express'] *= 1.5
        elif self._mood == AvatarMood.SLEEPY:
            base['idle'] *= 2.0
            base['gesture'] *= 0.3
            base['look'] *= 0.5
        elif self._mood == AvatarMood.EXCITED:
            base['gesture'] *= 1.5
            base['express'] *= 1.5
        
        # Adjust based on personality if linked
        if self._personality and hasattr(self._personality, 'get_effective_trait'):
            try:
                playfulness = self._personality.get_effective_trait('playfulness')
                if playfulness > 0.6:
                    base['gesture'] *= 1.3
                    base['express'] *= 1.2
            except Exception:
                pass
        
        return base
    
    def _do_idle_animation(self):
        """Small idle movement/animation - cycle through rather than random."""
        animations = [
            "breathe",
            "blink",
            "subtle_move",
            "look_around",
        ]
        
        # Cycle through animations in order
        self._idle_anim_index = getattr(self, '_idle_anim_index', 0)
        anim = animations[self._idle_anim_index % len(animations)]
        self._idle_anim_index += 1
        
        if hasattr(self.avatar, '_animator') and self.avatar._animator:
            self.avatar._animator.play(anim, duration=1.0)
        
        self._record_action(f"idle:{anim}")
    
    def _look_around(self):
        """Look at interesting parts of the screen, not random positions."""
        try:
            from PyQt5.QtWidgets import QApplication
            screen = QApplication.primaryScreen()
            if not screen:
                return
            
            geo = screen.geometry()
            
            # If we have screen regions detected, look at the most interesting one
            if self._screen_regions:
                sorted_regions = sorted(
                    self._screen_regions,
                    key=lambda r: r.interest_score,
                    reverse=True
                )
                region = sorted_regions[0]
                x = region.x + region.width // 2
                y = region.y + region.height // 2
            else:
                # No regions - look at center or common UI areas
                # Cycle through meaningful positions
                positions = [
                    (geo.width() // 2, geo.height() // 3),      # Top center (title bars)
                    (geo.width() // 4, geo.height() // 2),      # Left side
                    (3 * geo.width() // 4, geo.height() // 2),  # Right side
                    (geo.width() // 2, 2 * geo.height() // 3),  # Bottom center
                ]
                self._look_pos_index = getattr(self, '_look_pos_index', 0)
                x, y = positions[self._look_pos_index % len(positions)]
                self._look_pos_index += 1
            
            self.avatar.point_at(x, y)
            self._record_action(f"look_at:{x},{y}")
        except Exception as e:
            logger.debug(f"Look around failed: {e}")
    
    def _react_to_screen(self):
        """React to something on screen."""
        if not self._screen_regions:
            return
        
        # Find most interesting region
        if self._screen_regions:
            # Sort by interest score
            sorted_regions = sorted(
                self._screen_regions,
                key=lambda r: r.interest_score,
                reverse=True
            )
            region = sorted_regions[0]
            
            # Look at it
            center_x = region.x + region.width // 2
            center_y = region.y + region.height // 2
            self.avatar.point_at(center_x, center_y)
            
            # Maybe change mood based on content
            if region.content_type == "video":
                if self._mood != AvatarMood.FOCUSED:
                    self.set_mood(AvatarMood.FOCUSED)
            elif region.content_type == "new_window":
                self.set_mood(AvatarMood.CURIOUS)
            
            self._record_action(f"react:{region.content_type}")
    
    def _express_mood(self):
        """Express current mood through animation/expression."""
        mood_expressions = {
            AvatarMood.HAPPY: ["happy", "friendly", "winking"],
            AvatarMood.CURIOUS: ["thinking", "confused"],
            AvatarMood.BORED: ["neutral", "sleeping"],
            AvatarMood.EXCITED: ["excited", "surprised", "happy"],
            AvatarMood.SLEEPY: ["sleeping", "neutral"],
            AvatarMood.FOCUSED: ["thinking", "neutral"],
            AvatarMood.NEUTRAL: ["neutral", "idle"],
            AvatarMood.PLAYFUL: ["winking", "happy", "excited"],
            AvatarMood.THOUGHTFUL: ["thinking", "neutral"],
        }
        
        expressions = mood_expressions.get(self._mood, ["neutral"])
        # Cycle through expressions for this mood
        self._expr_index = getattr(self, '_expr_index', 0)
        expression = expressions[self._expr_index % len(expressions)]
        self._expr_index += 1
        
        # Use AUTONOMOUS priority (fallback to bone control)
        try:
            from .controller import ControlPriority
            self.avatar.set_expression(
                expression,
                requester="autonomous",
                priority=ControlPriority.AUTONOMOUS
            )
        except TypeError:
            # Fallback for old signature
            self.avatar.set_expression(expression)
        
        self._record_action(f"express:{expression}")
    
    def _random_gesture(self):
        """Do a contextual gesture (cycles through options)."""
        gestures = [
            "wave",
            "nod",
            "shake_head",
            "thumbs_up",
            "shrug",
            "point",
            "stretch",
        ]
        # Cycle through gestures in order
        self._gesture_index = getattr(self, '_gesture_index', 0)
        gesture = gestures[self._gesture_index % len(gestures)]
        self._gesture_index += 1
        
        if hasattr(self.avatar, '_animator') and self.avatar._animator:
            self.avatar._animator.play(gesture, duration=1.5)
        
        self._record_action(f"gesture:{gesture}")
    
    def _scan_screen(self):
        """Scan screen for interesting regions."""
        self._screen_regions.clear()
        
        try:
            import sys
            
            if sys.platform == 'win32':
                # Windows - get visible windows using ctypes (internal)
                import ctypes
                from ctypes import wintypes
                
                user32 = ctypes.windll.user32
                
                def callback(hwnd, _):
                    if user32.IsWindowVisible(hwnd):
                        length = user32.GetWindowTextLengthW(hwnd)
                        if length > 0:
                            title = ctypes.create_unicode_buffer(length + 1)
                            user32.GetWindowTextW(hwnd, title, length + 1)
                            
                            # Get window rect
                            rect = wintypes.RECT()
                            user32.GetWindowRect(hwnd, ctypes.byref(rect))
                            
                            if rect.right - rect.left > 100 and rect.bottom - rect.top > 100:
                                region = ScreenRegion(
                                    x=rect.left,
                                    y=rect.top,
                                    width=rect.right - rect.left,
                                    height=rect.bottom - rect.top,
                                    title=title.value,
                                    content_type=self._classify_window(title.value),
                                    interest_score=self._calc_interest(title.value)
                                )
                                self._screen_regions.append(region)
                                
                                # Check if new window
                                if title.value not in self._seen_windows:
                                    self._seen_windows.append(title.value)
                                    if len(self._seen_windows) > 100:
                                        self._seen_windows.pop(0)
                                    region.interest_score = 0.9  # New windows are interesting!
                                    region.content_type = "new_window"
                    return True
                
                WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
                user32.EnumWindows(WNDENUMPROC(callback), 0)
            
            else:
                # Linux/macOS - use Xlib or fallback to process-based detection (internal)
                try:
                    # Try python-xlib for X11 (internal library)
                    from Xlib import X, display
                    
                    d = display.Display()
                    root = d.screen().root
                    
                    # Get window list from X11
                    window_ids = root.get_full_property(
                        d.intern_atom('_NET_CLIENT_LIST'),
                        X.AnyPropertyType
                    )
                    
                    if window_ids:
                        for win_id in window_ids.value:
                            try:
                                window = d.create_resource_object('window', win_id)
                                
                                # Get window name
                                name_prop = window.get_full_property(
                                    d.intern_atom('_NET_WM_NAME'),
                                    X.AnyPropertyType
                                )
                                if not name_prop:
                                    name_prop = window.get_full_property(
                                        d.intern_atom('WM_NAME'),
                                        X.AnyPropertyType
                                    )
                                
                                title = name_prop.value.decode() if name_prop else "Unknown"
                                
                                # Get geometry
                                geom = window.get_geometry()
                                
                                if geom.width > 100 and geom.height > 100:
                                    region = ScreenRegion(
                                        x=geom.x, y=geom.y,
                                        width=geom.width, height=geom.height,
                                        title=title,
                                        content_type=self._classify_window(title),
                                        interest_score=self._calc_interest(title)
                                    )
                                    self._screen_regions.append(region)
                                    
                                    if title not in self._seen_windows:
                                        self._seen_windows.append(title)
                                        if len(self._seen_windows) > 100:
                                            self._seen_windows.pop(0)
                                        region.interest_score = 0.9
                                        region.content_type = "new_window"
                            except Exception as e:
                                logger.debug(f"Failed to get X11 window info: {e}")
                    
                    d.close()
                    
                except ImportError:
                    # Xlib not available - use psutil for process-based detection (internal)
                    try:
                        import psutil
                        for proc in psutil.process_iter(['name', 'cmdline']):
                            try:
                                name = proc.info['name'] or ''
                                # Create pseudo-regions from running processes
                                region = ScreenRegion(
                                    x=0, y=0, width=800, height=600,
                                    title=name,
                                    content_type=self._classify_window(name),
                                    interest_score=self._calc_interest(name)
                                )
                                if name not in self._seen_windows:
                                    self._seen_windows.append(name)
                                    if len(self._seen_windows) > 100:
                                        self._seen_windows.pop(0)
                                    region.interest_score = 0.5
                                self._screen_regions.append(region)
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                    except ImportError:
                        pass
                
        except Exception as e:
            # Screen scanning failed, continue without
            logger.debug(f"Screen scanning failed: {e}")
    
    def _classify_window(self, title: str) -> str:
        """Classify window by title."""
        title_lower = title.lower()
        
        if any(w in title_lower for w in ['youtube', 'netflix', 'video', 'vlc', 'player']):
            return "video"
        elif any(w in title_lower for w in ['code', 'visual studio', 'pycharm', 'sublime']):
            return "code"
        elif any(w in title_lower for w in ['chrome', 'firefox', 'edge', 'browser']):
            return "browser"
        elif any(w in title_lower for w in ['game', 'steam', 'minecraft', 'fortnite']):
            return "game"
        elif any(w in title_lower for w in ['chat', 'discord', 'slack', 'teams']):
            return "chat"
        else:
            return "window"
    
    def _calc_interest(self, title: str) -> float:
        """Calculate how interesting a window is."""
        title_lower = title.lower()
        
        # High interest
        if any(w in title_lower for w in ['video', 'youtube', 'game', 'chat']):
            return 0.8
        # Medium interest
        elif any(w in title_lower for w in ['browser', 'code', 'music']):
            return 0.6
        # Low interest
        else:
            return 0.3
    
    def _record_action(self, action: str):
        """Record an action in memory."""
        self._recent_actions.append(action)
        if len(self._recent_actions) > 50:
            self._recent_actions.pop(0)
        
        # Notify callbacks
        for cb in self._on_action:
            try:
                cb(action)
            except Exception as e:
                logger.debug(f"Action callback failed: {e}")


# Global instance
_autonomous_avatar: Optional[AutonomousAvatar] = None


def get_autonomous_avatar(avatar: Optional['AvatarController'] = None) -> AutonomousAvatar:
    """Get or create autonomous avatar controller."""
    global _autonomous_avatar
    
    if _autonomous_avatar is None:
        if avatar is None:
            from . import get_avatar
            avatar = get_avatar()
        _autonomous_avatar = AutonomousAvatar(avatar)
    
    return _autonomous_avatar
