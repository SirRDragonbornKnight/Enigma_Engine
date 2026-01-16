"""
Autonomous Avatar Behavior

Makes the avatar react to the screen, do things on its own,
express curiosity, and behave naturally without explicit commands.

Features:
- Screen watching: React to what's on screen
- Idle behaviors: Random movements, expressions, gestures
- Curiosity: "Look at" interesting things on screen
- Mood system: Gradual mood changes based on what it sees
- Memory: Remember what it's seen/done
- Context awareness: React to user activity
- Time-based behaviors: Different behaviors at different times

Usage:
    from forge_ai.avatar import get_avatar
    from forge_ai.avatar.autonomous import AutonomousAvatar
    
    avatar = get_avatar()
    avatar.enable()
    
    autonomous = AutonomousAvatar(avatar)
    autonomous.start()  # Avatar starts doing things on its own
    autonomous.stop()   # Back to manual control
"""

import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from .controller import AvatarController

# Type alias for optional AvatarController
_AvatarController = Optional['AvatarController']


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
    sleepy_hours: List[int] = field(default_factory=lambda: [22, 23, 0, 1, 2, 3, 4, 5])  # Night hours
    energetic_hours: List[int] = field(default_factory=lambda: [9, 10, 11, 14, 15, 16])  # Day hours
    
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
        
        # State
        self._mood = AvatarMood.NEUTRAL
        self._last_action_time = 0.0
        self._last_screen_scan = 0.0
        self._last_activity_time = time.time()
        self._screen_regions: List[ScreenRegion] = []
        self._current_focus: Optional[ScreenRegion] = None
        
        # Memory - what has avatar seen/done
        self._seen_windows: List[str] = []
        self._recent_actions: List[str] = []
        self._interaction_count: int = 0
        
        # Callbacks for external integration
        self._on_mood_change: List[Callable] = []
        self._on_action: List[Callable] = []
        
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
            print("[Autonomous] Avatar must be enabled first")
            return
        
        self._running = True
        self.config.enabled = True
        self._thread = threading.Thread(target=self._behavior_loop, daemon=True)
        self._thread.start()
        print("[Autonomous] Avatar autonomous mode started")
    
    def stop(self):
        """Stop autonomous behavior."""
        self._running = False
        self.config.enabled = False
        print("[Autonomous] Avatar autonomous mode stopped")
    
    def set_mood(self, mood: AvatarMood):
        """Set avatar mood."""
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
            self.avatar.set_expression(mood_expressions.get(mood, "neutral"))
            
            # Notify callbacks
            for cb in self._on_mood_change:
                try:
                    cb(old_mood, mood)
                except Exception:
                    pass
    
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
            # User engaged, avatar gets curious/happy
            if random.random() > 0.5:
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
            if random.random() > 0.7:
                return AvatarMood.PLAYFUL
        
        return None
    
    def _behavior_loop(self):
        """Main autonomous behavior loop."""
        while self._running:
            try:
                current_time = time.time()
                
                # Calculate next action timing
                interval = random.uniform(
                    self.config.action_interval_min,
                    self.config.action_interval_max
                )
                
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
                
                # Time-based mood check
                time_mood = self._get_time_based_mood_modifier()
                if time_mood and random.random() > 0.95:  # Rarely override
                    self.set_mood(time_mood)
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"[Autonomous] Error in behavior loop: {e}")
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
        
        # Normalize weights
        total_weight = sum(w for w, _ in actions)
        rand = random.random() * total_weight
        
        cumulative = 0
        for weight, action in actions:
            cumulative += weight
            if rand <= cumulative:
                action()
                break
    
    def _get_adjusted_weights(self) -> Dict[str, float]:
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
        """Small idle movement/animation."""
        animations = [
            "breathe",
            "blink",
            "subtle_move",
            "look_around",
        ]
        anim = random.choice(animations)
        
        if hasattr(self.avatar, '_animator') and self.avatar._animator:
            self.avatar._animator.play(anim, duration=1.0)
        
        self._record_action(f"idle:{anim}")
    
    def _look_around(self):
        """Look at different parts of the screen."""
        try:
            from PyQt5.QtWidgets import QApplication
            screen = QApplication.primaryScreen()
            if screen:
                geo = screen.geometry()
                # Pick random point on screen
                x = random.randint(0, geo.width())
                y = random.randint(0, geo.height())
                self.avatar.point_at(x, y)
                self._record_action(f"look_at:{x},{y}")
        except Exception:
            pass
    
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
        expression = random.choice(expressions)
        self.avatar.set_expression(expression)
        self._record_action(f"express:{expression}")
    
    def _random_gesture(self):
        """Do a random gesture."""
        gestures = [
            "wave",
            "nod",
            "shake_head",
            "thumbs_up",
            "shrug",
            "point",
            "stretch",
        ]
        gesture = random.choice(gestures)
        
        if hasattr(self.avatar, '_animator') and self.avatar._animator:
            self.avatar._animator.play(gesture, duration=1.5)
        
        self._record_action(f"gesture:{gesture}")
    
    def _scan_screen(self):
        """Scan screen for interesting regions."""
        self._screen_regions.clear()
        
        try:
            # Try to get window list
            import subprocess
            import sys
            
            if sys.platform == 'win32':
                # Windows - get visible windows
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
                                    region.interest_score = 0.9  # New windows are interesting!
                                    region.content_type = "new_window"
                    return True
                
                WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
                user32.EnumWindows(WNDENUMPROC(callback), 0)
                
        except Exception as e:
            # Screen scanning failed, continue without
            pass
    
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
            except Exception:
                pass


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
