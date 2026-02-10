"""
Game Profile Auto-Detection for Enigma AI Engine

Automatically recognize games and load appropriate profiles.

Provides:
- Process name detection
- Window title matching
- Game-specific AI configurations
- Overlay positioning
- Knowledge base selection

Usage:
    from enigma_engine.tools.game_detector import GameDetector, GameProfile
    
    detector = GameDetector()
    detector.start()  # Begin monitoring
    
    # Get current game
    game = detector.current_game  # Returns GameProfile or None
    
    # Add callback for game changes
    detector.on_game_change(lambda game: print(f"Now playing: {game.name}"))
    
    # Add custom game profile
    detector.add_profile(GameProfile(
        name="My Game",
        process_names=["mygame.exe"],
        window_titles=["My Game*"],
        knowledge_topic="my_game_wiki"
    ))

Built-in profiles for popular games are included.
"""

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set
import json

logger = logging.getLogger(__name__)

# Try imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# Windows-specific for window titles
try:
    import ctypes
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False


@dataclass
class GameProfile:
    """Configuration profile for a specific game."""
    # Identification
    name: str
    process_names: List[str] = field(default_factory=list)  # e.g., ["game.exe"]
    window_titles: List[str] = field(default_factory=list)  # Glob patterns
    
    # Categorization
    genre: str = ""
    subgenre: str = ""
    
    # AI Configuration
    knowledge_topic: str = ""           # Wiki/knowledge base to use
    personality_override: str = ""       # AI personality for this game
    response_style: str = "casual"       # casual, tactical, supportive
    
    # Overlay settings
    overlay_position: str = "bottom-right"  # top-left, top-right, etc.
    overlay_opacity: float = 0.8
    auto_minimize: bool = True              # Minimize when game closes
    
    # Voice settings
    voice_enabled: bool = True
    voice_activation: str = "push-to-talk"  # always, push-to-talk, keyword
    
    # Game-specific features
    track_achievements: bool = True
    track_stats: bool = True
    provide_tips: bool = True
    
    # Commands
    custom_commands: Dict[str, str] = field(default_factory=dict)  # !cmd -> action
    
    # Metadata
    icon_path: str = ""
    color_theme: str = ""                   # Overlay color scheme


# Built-in profiles for popular games
BUILTIN_PROFILES = [
    # FPS Games
    GameProfile(
        name="Counter-Strike 2",
        process_names=["cs2.exe", "csgo.exe"],
        window_titles=["Counter-Strike*"],
        genre="FPS",
        subgenre="Competitive",
        knowledge_topic="counter_strike",
        response_style="tactical",
        overlay_position="top-right",
        custom_commands={
            "!callout": "Provide callout for current location",
            "!eco": "Suggest eco round buy",
            "!strat": "Suggest strategy",
        }
    ),
    GameProfile(
        name="Valorant",
        process_names=["valorant.exe", "valorant-win64-shipping.exe"],
        window_titles=["VALORANT*"],
        genre="FPS",
        subgenre="Hero Shooter",
        knowledge_topic="valorant",
        response_style="tactical",
        overlay_position="top-right",
    ),
    GameProfile(
        name="Overwatch 2",
        process_names=["overwatch.exe"],
        window_titles=["Overwatch*"],
        genre="FPS",
        subgenre="Hero Shooter",
        knowledge_topic="overwatch",
        response_style="supportive",
    ),
    
    # MOBAs
    GameProfile(
        name="League of Legends",
        process_names=["league of legends.exe", "leagueclient.exe"],
        window_titles=["League of Legends*"],
        genre="MOBA",
        knowledge_topic="league_of_legends",
        response_style="tactical",
        overlay_position="bottom-left",
        custom_commands={
            "!build": "Suggest item build",
            "!matchup": "Analyze lane matchup",
            "!objectives": "Prioritize objectives",
        }
    ),
    GameProfile(
        name="Dota 2",
        process_names=["dota2.exe"],
        window_titles=["Dota 2*"],
        genre="MOBA",
        knowledge_topic="dota2",
        response_style="tactical",
    ),
    
    # Battle Royale
    GameProfile(
        name="Apex Legends",
        process_names=["r5apex.exe"],
        window_titles=["Apex Legends*"],
        genre="Battle Royale",
        knowledge_topic="apex_legends",
        response_style="tactical",
        custom_commands={
            "!loot": "Suggest loot priorities",
            "!legend": "Legend tips and abilities",
        }
    ),
    GameProfile(
        name="Fortnite",
        process_names=["fortnite*.exe", "fortniteclient*.exe"],
        window_titles=["Fortnite*"],
        genre="Battle Royale",
        knowledge_topic="fortnite",
        response_style="casual",
    ),
    
    # RPGs
    GameProfile(
        name="Elden Ring",
        process_names=["eldenring.exe"],
        window_titles=["ELDEN RING*"],
        genre="RPG",
        subgenre="Souls-like",
        knowledge_topic="elden_ring",
        response_style="supportive",
        provide_tips=True,
        custom_commands={
            "!boss": "Boss fight tips",
            "!build": "Build suggestions",
            "!location": "Where to go next",
        }
    ),
    GameProfile(
        name="Baldur's Gate 3",
        process_names=["bg3.exe", "bg3_dx11.exe"],
        window_titles=["Baldur's Gate 3*"],
        genre="RPG",
        subgenre="CRPG",
        knowledge_topic="baldurs_gate_3",
        response_style="supportive",
    ),
    
    # Survival/Crafting
    GameProfile(
        name="Minecraft",
        process_names=["javaw.exe", "minecraft.exe"],
        window_titles=["Minecraft*"],
        genre="Sandbox",
        knowledge_topic="minecraft",
        response_style="casual",
        custom_commands={
            "!craft": "Crafting recipe",
            "!redstone": "Redstone help",
            "!enchant": "Enchanting guide",
        }
    ),
    GameProfile(
        name="Terraria",
        process_names=["terraria.exe"],
        window_titles=["Terraria*"],
        genre="Sandbox",
        knowledge_topic="terraria",
        response_style="casual",
    ),
    
    # Strategy
    GameProfile(
        name="Civilization VI",
        process_names=["civilizationvi.exe", "civ6.exe"],
        window_titles=["Sid Meier's Civilization*"],
        genre="Strategy",
        subgenre="4X",
        knowledge_topic="civilization",
        response_style="tactical",
    ),
    
    # Racing
    GameProfile(
        name="Forza Horizon 5",
        process_names=["forzahorizon5.exe"],
        window_titles=["Forza Horizon 5*"],
        genre="Racing",
        knowledge_topic="forza",
        response_style="casual",
    ),
]


class GameDetector:
    """
    Automatically detect running games and load profiles.
    
    Monitors running processes and window titles to identify games.
    """
    
    def __init__(self, custom_profiles_path: Optional[str] = None):
        """
        Initialize game detector.
        
        Args:
            custom_profiles_path: Path to custom profiles JSON
        """
        self._profiles: Dict[str, GameProfile] = {}
        self._current_game: Optional[GameProfile] = None
        self._callbacks: List[Callable[[Optional[GameProfile]], None]] = []
        
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._check_interval = 5.0  # Seconds between checks
        
        self._lock = threading.Lock()
        
        # Load built-in profiles
        for profile in BUILTIN_PROFILES:
            self._profiles[profile.name.lower()] = profile
        
        # Load custom profiles
        if custom_profiles_path:
            self._load_custom_profiles(custom_profiles_path)
        
        logger.info(f"GameDetector initialized with {len(self._profiles)} profiles")
    
    def _load_custom_profiles(self, path: str):
        """Load profiles from JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for profile_data in data.get("profiles", []):
                profile = GameProfile(**profile_data)
                self._profiles[profile.name.lower()] = profile
            
            logger.info(f"Loaded {len(data.get('profiles', []))} custom profiles")
            
        except Exception as e:
            logger.warning(f"Failed to load custom profiles: {e}")
    
    def add_profile(self, profile: GameProfile):
        """Add a game profile."""
        self._profiles[profile.name.lower()] = profile
        logger.debug(f"Added profile: {profile.name}")
    
    def remove_profile(self, name: str):
        """Remove a game profile."""
        key = name.lower()
        if key in self._profiles:
            del self._profiles[key]
    
    def get_profile(self, name: str) -> Optional[GameProfile]:
        """Get a profile by name."""
        return self._profiles.get(name.lower())
    
    def list_profiles(self) -> List[GameProfile]:
        """Get all available profiles."""
        return list(self._profiles.values())
    
    @property
    def current_game(self) -> Optional[GameProfile]:
        """Get currently detected game."""
        with self._lock:
            return self._current_game
    
    def start(self):
        """Start monitoring for games."""
        if self._running:
            return
        
        if not PSUTIL_AVAILABLE:
            logger.error("psutil required for game detection. Run: pip install psutil")
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Game detection started")
    
    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
        logger.info("Game detection stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                detected = self._detect_game()
                
                with self._lock:
                    if detected != self._current_game:
                        old_game = self._current_game
                        self._current_game = detected
                        
                        if detected:
                            logger.info(f"Game detected: {detected.name}")
                        elif old_game:
                            logger.info(f"Game closed: {old_game.name}")
                        
                        # Notify callbacks
                        for callback in self._callbacks:
                            try:
                                callback(detected)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")
                
            except Exception as e:
                logger.error(f"Detection error: {e}")
            
            time.sleep(self._check_interval)
    
    def _detect_game(self) -> Optional[GameProfile]:
        """Detect currently running game."""
        running_processes = self._get_running_processes()
        window_titles = self._get_window_titles()
        
        # Check each profile
        for profile in self._profiles.values():
            # Check process names
            for proc_name in profile.process_names:
                proc_lower = proc_name.lower()
                for running in running_processes:
                    if self._match_pattern(running, proc_lower):
                        return profile
            
            # Check window titles
            for title_pattern in profile.window_titles:
                for window_title in window_titles:
                    if self._match_pattern(window_title, title_pattern):
                        return profile
        
        return None
    
    def _get_running_processes(self) -> Set[str]:
        """Get names of running processes."""
        processes = set()
        
        try:
            for proc in psutil.process_iter(['name']):
                try:
                    name = proc.info['name']
                    if name:
                        processes.add(name.lower())
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.debug(f"Error getting processes: {e}")
        
        return processes
    
    def _get_window_titles(self) -> List[str]:
        """Get titles of open windows."""
        titles = []
        
        if WIN32_AVAILABLE:
            try:
                titles = self._get_win32_window_titles()
            except Exception:
                pass
        
        return titles
    
    def _get_win32_window_titles(self) -> List[str]:
        """Get window titles on Windows."""
        titles = []
        
        EnumWindows = ctypes.windll.user32.EnumWindows
        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
        GetWindowText = ctypes.windll.user32.GetWindowTextW
        GetWindowTextLength = ctypes.windll.user32.GetWindowTextLengthW
        IsWindowVisible = ctypes.windll.user32.IsWindowVisible
        
        def callback(hwnd, lParam):
            if IsWindowVisible(hwnd):
                length = GetWindowTextLength(hwnd)
                if length > 0:
                    buff = ctypes.create_unicode_buffer(length + 1)
                    GetWindowText(hwnd, buff, length + 1)
                    titles.append(buff.value)
            return True
        
        EnumWindows(EnumWindowsProc(callback), 0)
        return titles
    
    def _match_pattern(self, text: str, pattern: str) -> bool:
        """Match text against glob pattern."""
        # Convert glob to regex
        pattern = pattern.lower()
        text = text.lower()
        
        # Simple glob: * matches anything
        regex = pattern.replace(".", r"\.").replace("*", ".*")
        
        try:
            return bool(re.match(regex, text))
        except Exception:
            return pattern in text
    
    def on_game_change(self, callback: Callable[[Optional[GameProfile]], None]):
        """Register callback for game detection changes."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def detect_once(self) -> Optional[GameProfile]:
        """Run detection once without starting monitor."""
        if not PSUTIL_AVAILABLE:
            return None
        return self._detect_game()
    
    def save_profiles(self, path: str):
        """Save current profiles to JSON."""
        profiles_data = []
        
        for profile in self._profiles.values():
            profiles_data.append({
                "name": profile.name,
                "process_names": profile.process_names,
                "window_titles": profile.window_titles,
                "genre": profile.genre,
                "subgenre": profile.subgenre,
                "knowledge_topic": profile.knowledge_topic,
                "personality_override": profile.personality_override,
                "response_style": profile.response_style,
                "overlay_position": profile.overlay_position,
                "overlay_opacity": profile.overlay_opacity,
                "custom_commands": profile.custom_commands,
            })
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({"profiles": profiles_data}, f, indent=2)
        
        logger.info(f"Saved {len(profiles_data)} profiles to {path}")


# Global instance
_detector: Optional[GameDetector] = None


def get_detector() -> GameDetector:
    """Get or create global game detector."""
    global _detector
    if _detector is None:
        _detector = GameDetector()
    return _detector


def detect_current_game() -> Optional[GameProfile]:
    """Quick detection of current game."""
    return get_detector().detect_once()


def is_game_detection_available() -> bool:
    """Check if game detection is available."""
    return PSUTIL_AVAILABLE
