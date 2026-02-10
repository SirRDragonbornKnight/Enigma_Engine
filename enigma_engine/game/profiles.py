"""
Per-Game Profiles

Game-specific configuration profiles for Enigma AI Engine.
Store overlay position, behavior, prompts, and hotkeys per game.

FILE: enigma_engine/game/profiles.py
TYPE: Game
MAIN CLASSES: GameProfile, GameProfileManager, GameDetector
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GameGenre(Enum):
    """Game genre categories."""
    FPS = "fps"
    RPG = "rpg"
    RTS = "rts"
    MOBA = "moba"
    MMO = "mmo"
    SPORTS = "sports"
    RACING = "racing"
    PUZZLE = "puzzle"
    SURVIVAL = "survival"
    SANDBOX = "sandbox"
    FIGHTING = "fighting"
    PLATFORMER = "platformer"
    ADVENTURE = "adventure"
    SIMULATION = "simulation"
    OTHER = "other"


@dataclass
class OverlaySettings:
    """Overlay settings for a game."""
    position_x: int = 10
    position_y: int = 10
    width: int = 400
    height: int = 300
    opacity: float = 0.85
    always_on_top: bool = True
    auto_minimize: bool = True
    show_on_hotkey: bool = True
    toggle_hotkey: str = "Ctrl+Shift+O"


@dataclass
class AIBehavior:
    """AI behavior settings for a game."""
    system_prompt: str = ""
    response_style: str = "concise"  # concise, detailed, casual
    max_response_length: int = 200
    context_window: int = 10
    temperature: float = 0.7
    include_game_context: bool = True
    proactive_tips: bool = False
    tip_interval_seconds: int = 300


@dataclass
class GameProfile:
    """Complete profile for a specific game."""
    # Identification
    game_id: str
    game_name: str
    executable_names: list[str] = field(default_factory=list)
    window_titles: list[str] = field(default_factory=list)
    
    # Classification
    genre: GameGenre = GameGenre.OTHER
    tags: list[str] = field(default_factory=list)
    
    # Settings
    overlay: OverlaySettings = field(default_factory=OverlaySettings)
    ai_behavior: AIBehavior = field(default_factory=AIBehavior)
    
    # State
    enabled: bool = True
    last_used: Optional[str] = None
    total_playtime_minutes: int = 0
    
    # Custom data
    custom_hotkeys: dict[str, str] = field(default_factory=dict)
    custom_commands: dict[str, str] = field(default_factory=dict)
    notes: str = ""


# Built-in game profiles
BUILTIN_PROFILES = {
    "minecraft": GameProfile(
        game_id="minecraft",
        game_name="Minecraft",
        executable_names=["javaw.exe", "minecraft.exe", "java.exe"],
        window_titles=["Minecraft"],
        genre=GameGenre.SANDBOX,
        tags=["building", "survival", "crafting"],
        ai_behavior=AIBehavior(
            system_prompt="You are a Minecraft assistant. Help with crafting recipes, building tips, redstone, and game mechanics. Be concise.",
            response_style="concise",
            proactive_tips=True,
            tip_interval_seconds=600
        )
    ),
    "league_of_legends": GameProfile(
        game_id="league_of_legends",
        game_name="League of Legends",
        executable_names=["League of Legends.exe", "LeagueClient.exe"],
        window_titles=["League of Legends"],
        genre=GameGenre.MOBA,
        tags=["moba", "competitive", "team"],
        ai_behavior=AIBehavior(
            system_prompt="You are a League of Legends coach. Provide champion builds, counter picks, and macro strategy advice. Keep responses brief for in-game use.",
            response_style="concise",
            max_response_length=150
        )
    ),
    "valorant": GameProfile(
        game_id="valorant",
        game_name="Valorant",
        executable_names=["VALORANT-Win64-Shipping.exe", "VALORANT.exe"],
        window_titles=["VALORANT"],
        genre=GameGenre.FPS,
        tags=["fps", "tactical", "competitive"],
        ai_behavior=AIBehavior(
            system_prompt="You are a Valorant coach. Help with agent abilities, map callouts, strategies, and aim tips. Be brief.",
            response_style="concise",
            max_response_length=150
        )
    ),
    "elden_ring": GameProfile(
        game_id="elden_ring",
        game_name="Elden Ring",
        executable_names=["eldenring.exe"],
        window_titles=["ELDEN RING"],
        genre=GameGenre.RPG,
        tags=["rpg", "souls", "open world"],
        ai_behavior=AIBehavior(
            system_prompt="You are an Elden Ring guide. Help with boss strategies, build advice, item locations, and lore. Avoid major spoilers unless asked.",
            response_style="detailed",
            include_game_context=True
        )
    ),
    "counter_strike": GameProfile(
        game_id="counter_strike",
        game_name="Counter-Strike",
        executable_names=["cs2.exe", "csgo.exe"],
        window_titles=["Counter-Strike 2", "Counter-Strike: Global Offensive"],
        genre=GameGenre.FPS,
        tags=["fps", "tactical", "competitive"],
        ai_behavior=AIBehavior(
            system_prompt="You are a CS coach. Help with smokes, flashes, callouts, economy management, and strategies. Keep it brief.",
            response_style="concise",
            max_response_length=150
        )
    ),
    "fortnite": GameProfile(
        game_id="fortnite",
        game_name="Fortnite",
        executable_names=["FortniteClient-Win64-Shipping.exe"],
        window_titles=["Fortnite"],
        genre=GameGenre.FPS,
        tags=["battle royale", "building"],
        ai_behavior=AIBehavior(
            system_prompt="You are a Fortnite coach. Help with building techniques, rotations, loadouts, and strategies.",
            response_style="concise"
        )
    ),
    "stardew_valley": GameProfile(
        game_id="stardew_valley",
        game_name="Stardew Valley",
        executable_names=["Stardew Valley.exe"],
        window_titles=["Stardew Valley"],
        genre=GameGenre.SIMULATION,
        tags=["farming", "simulation", "relaxing"],
        ai_behavior=AIBehavior(
            system_prompt="You are a Stardew Valley helper. Assist with crop planning, villager gifts, fishing, and seasonal strategies.",
            response_style="detailed",
            proactive_tips=True,
            tip_interval_seconds=900
        )
    )
}


class GameDetector:
    """
    Detect running games by process or window title.
    """
    
    def __init__(self):
        self._cached_processes: dict[str, str] = {}
        self._last_scan = 0
        self._scan_interval = 5.0  # seconds
    
    def get_running_games(self, profiles: dict[str, GameProfile]) -> list[str]:
        """
        Get list of running games that have profiles.
        
        Args:
            profiles: Dictionary of game profiles
        
        Returns:
            List of game_ids for running games
        """
        if not HAS_PSUTIL:
            return []
        
        running = []
        
        # Get all process names
        processes = set()
        try:
            for proc in psutil.process_iter(['name']):
                try:
                    name = proc.info['name']
                    if name:
                        processes.add(name.lower())
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            logger.debug(f"Error scanning processes: {e}")
            return []
        
        # Match against profiles
        for game_id, profile in profiles.items():
            for exe in profile.executable_names:
                if exe.lower() in processes:
                    running.append(game_id)
                    break
        
        return running
    
    def detect_active_game(self, profiles: dict[str, GameProfile]) -> Optional[str]:
        """
        Detect the currently focused game.
        
        Args:
            profiles: Dictionary of game profiles
        
        Returns:
            game_id of focused game, or None
        """
        # Try to get foreground window title
        try:
            if os.name == 'nt':
                import ctypes
                user32 = ctypes.windll.user32
                hwnd = user32.GetForegroundWindow()
                length = user32.GetWindowTextLengthW(hwnd)
                buff = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buff, length + 1)
                window_title = buff.value.lower()
                
                for game_id, profile in profiles.items():
                    for title in profile.window_titles:
                        if title.lower() in window_title:
                            return game_id
        except Exception as e:
            logger.debug(f"Error detecting active window: {e}")
        
        return None


class GameProfileManager:
    """
    Manage game profiles - create, load, save, and switch.
    """
    
    def __init__(self, profiles_dir: str = None):
        self.profiles_dir = Path(profiles_dir or "data/game_profiles")
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        
        self._profiles: dict[str, GameProfile] = {}
        self._active_profile: Optional[str] = None
        self._detector = GameDetector()
        self._callbacks: list[Callable[[str], None]] = []
        
        # Load built-in profiles
        self._profiles.update(BUILTIN_PROFILES)
        
        # Load custom profiles
        self._load_custom_profiles()
    
    def _load_custom_profiles(self):
        """Load profiles from disk."""
        for file in self.profiles_dir.glob("*.json"):
            try:
                with open(file) as f:
                    data = json.load(f)
                
                profile = self._dict_to_profile(data)
                self._profiles[profile.game_id] = profile
                logger.info(f"Loaded profile: {profile.game_name}")
            except Exception as e:
                logger.warning(f"Error loading profile {file}: {e}")
    
    def _dict_to_profile(self, data: dict[str, Any]) -> GameProfile:
        """Convert dictionary to GameProfile."""
        overlay_data = data.get('overlay', {})
        ai_data = data.get('ai_behavior', {})
        
        overlay = OverlaySettings(**{
            k: v for k, v in overlay_data.items()
            if hasattr(OverlaySettings, k)
        })
        
        ai_behavior = AIBehavior(**{
            k: v for k, v in ai_data.items()
            if hasattr(AIBehavior, k)
        })
        
        genre = GameGenre(data.get('genre', 'other'))
        
        return GameProfile(
            game_id=data['game_id'],
            game_name=data['game_name'],
            executable_names=data.get('executable_names', []),
            window_titles=data.get('window_titles', []),
            genre=genre,
            tags=data.get('tags', []),
            overlay=overlay,
            ai_behavior=ai_behavior,
            enabled=data.get('enabled', True),
            last_used=data.get('last_used'),
            total_playtime_minutes=data.get('total_playtime_minutes', 0),
            custom_hotkeys=data.get('custom_hotkeys', {}),
            custom_commands=data.get('custom_commands', {}),
            notes=data.get('notes', '')
        )
    
    def _profile_to_dict(self, profile: GameProfile) -> dict[str, Any]:
        """Convert GameProfile to dictionary."""
        return {
            "game_id": profile.game_id,
            "game_name": profile.game_name,
            "executable_names": profile.executable_names,
            "window_titles": profile.window_titles,
            "genre": profile.genre.value,
            "tags": profile.tags,
            "overlay": {
                "position_x": profile.overlay.position_x,
                "position_y": profile.overlay.position_y,
                "width": profile.overlay.width,
                "height": profile.overlay.height,
                "opacity": profile.overlay.opacity,
                "always_on_top": profile.overlay.always_on_top,
                "auto_minimize": profile.overlay.auto_minimize,
                "toggle_hotkey": profile.overlay.toggle_hotkey
            },
            "ai_behavior": {
                "system_prompt": profile.ai_behavior.system_prompt,
                "response_style": profile.ai_behavior.response_style,
                "max_response_length": profile.ai_behavior.max_response_length,
                "context_window": profile.ai_behavior.context_window,
                "temperature": profile.ai_behavior.temperature,
                "include_game_context": profile.ai_behavior.include_game_context,
                "proactive_tips": profile.ai_behavior.proactive_tips,
                "tip_interval_seconds": profile.ai_behavior.tip_interval_seconds
            },
            "enabled": profile.enabled,
            "last_used": profile.last_used,
            "total_playtime_minutes": profile.total_playtime_minutes,
            "custom_hotkeys": profile.custom_hotkeys,
            "custom_commands": profile.custom_commands,
            "notes": profile.notes
        }
    
    def get_profile(self, game_id: str) -> Optional[GameProfile]:
        """Get a profile by game ID."""
        return self._profiles.get(game_id)
    
    def get_all_profiles(self) -> dict[str, GameProfile]:
        """Get all profiles."""
        return self._profiles.copy()
    
    def get_active_profile(self) -> Optional[GameProfile]:
        """Get currently active profile."""
        if self._active_profile:
            return self._profiles.get(self._active_profile)
        return None
    
    def set_active_profile(self, game_id: str) -> bool:
        """
        Set the active profile.
        
        Args:
            game_id: Game ID to activate
        
        Returns:
            True if successful
        """
        if game_id not in self._profiles:
            return False
        
        self._active_profile = game_id
        profile = self._profiles[game_id]
        profile.last_used = datetime.now().isoformat()
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(game_id)
            except Exception as e:
                logger.error(f"Error in profile callback: {e}")
        
        return True
    
    def create_profile(
        self,
        game_id: str,
        game_name: str,
        **kwargs
    ) -> GameProfile:
        """
        Create a new game profile.
        
        Args:
            game_id: Unique identifier
            game_name: Display name
            **kwargs: Additional profile attributes
        
        Returns:
            Created GameProfile
        """
        profile = GameProfile(
            game_id=game_id,
            game_name=game_name,
            **kwargs
        )
        
        self._profiles[game_id] = profile
        self.save_profile(game_id)
        
        return profile
    
    def update_profile(self, game_id: str, updates: dict[str, Any]) -> bool:
        """
        Update a profile.
        
        Args:
            game_id: Profile to update
            updates: Dictionary of updates
        
        Returns:
            True if successful
        """
        profile = self._profiles.get(game_id)
        if not profile:
            return False
        
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        
        self.save_profile(game_id)
        return True
    
    def delete_profile(self, game_id: str) -> bool:
        """Delete a profile."""
        if game_id not in self._profiles:
            return False
        
        # Don't delete built-in profiles from memory, just file
        profile_file = self.profiles_dir / f"{game_id}.json"
        if profile_file.exists():
            profile_file.unlink()
        
        if game_id not in BUILTIN_PROFILES:
            del self._profiles[game_id]
        
        return True
    
    def save_profile(self, game_id: str):
        """Save a profile to disk."""
        profile = self._profiles.get(game_id)
        if not profile:
            return
        
        data = self._profile_to_dict(profile)
        
        file_path = self.profiles_dir / f"{game_id}.json"
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_all_profiles(self):
        """Save all profiles to disk."""
        for game_id in self._profiles:
            self.save_profile(game_id)
    
    def auto_detect_profile(self) -> Optional[str]:
        """
        Auto-detect running game and activate profile.
        
        Returns:
            game_id of detected game, or None
        """
        detected = self._detector.detect_active_game(self._profiles)
        
        if detected:
            self.set_active_profile(detected)
            return detected
        
        # Check running games
        running = self._detector.get_running_games(self._profiles)
        if running:
            self.set_active_profile(running[0])
            return running[0]
        
        return None
    
    def add_profile_callback(self, callback: Callable[[str], None]):
        """Add callback for profile changes."""
        self._callbacks.append(callback)
    
    def search_profiles(self, query: str) -> list[GameProfile]:
        """
        Search profiles by name or tags.
        
        Args:
            query: Search string
        
        Returns:
            Matching profiles
        """
        query_lower = query.lower()
        results = []
        
        for profile in self._profiles.values():
            if (query_lower in profile.game_name.lower() or
                query_lower in profile.game_id.lower() or
                any(query_lower in tag for tag in profile.tags)):
                results.append(profile)
        
        return results
    
    def get_profiles_by_genre(self, genre: GameGenre) -> list[GameProfile]:
        """Get all profiles of a specific genre."""
        return [p for p in self._profiles.values() if p.genre == genre]
    
    def get_recent_profiles(self, limit: int = 5) -> list[GameProfile]:
        """Get recently used profiles."""
        profiles_with_dates = [
            p for p in self._profiles.values()
            if p.last_used is not None
        ]
        
        profiles_with_dates.sort(key=lambda p: p.last_used, reverse=True)
        return profiles_with_dates[:limit]
    
    def export_profile(self, game_id: str, path: str):
        """Export a profile to a file."""
        profile = self._profiles.get(game_id)
        if not profile:
            raise ValueError(f"Profile not found: {game_id}")
        
        data = self._profile_to_dict(profile)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def import_profile(self, path: str) -> GameProfile:
        """Import a profile from a file."""
        with open(path) as f:
            data = json.load(f)
        
        profile = self._dict_to_profile(data)
        self._profiles[profile.game_id] = profile
        self.save_profile(profile.game_id)
        
        return profile


def get_profile_manager(profiles_dir: str = None) -> GameProfileManager:
    """
    Get or create the profile manager singleton.
    
    Args:
        profiles_dir: Directory for profiles
    
    Returns:
        GameProfileManager instance
    """
    if not hasattr(get_profile_manager, '_instance'):
        get_profile_manager._instance = GameProfileManager(profiles_dir)
    return get_profile_manager._instance
