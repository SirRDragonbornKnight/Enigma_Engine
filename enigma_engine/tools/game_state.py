"""
Game State Awareness for Enigma AI Engine

Read screen/memory for game context.

Features:
- Screen capture analysis
- Memory reading (with caution)
- State detection
- HUD parsing
- Event detection

Usage:
    from enigma_engine.tools.game_state import GameStateReader
    
    reader = GameStateReader()
    
    # Configure for a game
    reader.configure_game("minecraft", {
        "health_region": (10, 10, 100, 30),
        "inventory_key": "e"
    })
    
    # Read state
    state = reader.get_state()
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ScreenRegion:
    """A region of the screen."""
    x: int
    y: int
    width: int
    height: int
    name: str = ""


@dataclass
class GameState:
    """Current game state."""
    game_name: str
    timestamp: float = field(default_factory=time.time)
    
    # Basic state
    is_playing: bool = False
    is_paused: bool = False
    is_menu: bool = False
    is_loading: bool = False
    
    # Player state
    health: float = 100.0
    max_health: float = 100.0
    mana: float = 100.0
    stamina: float = 100.0
    
    # Position
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Context
    current_area: str = ""
    current_quest: str = ""
    
    # Inventory
    inventory_items: List[str] = field(default_factory=list)
    equipped_weapon: str = ""
    
    # Custom data
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GameProfile:
    """Configuration profile for a game."""
    game_name: str
    process_name: str = ""
    
    # Screen regions for HUD elements
    health_region: Optional[ScreenRegion] = None
    mana_region: Optional[ScreenRegion] = None
    minimap_region: Optional[ScreenRegion] = None
    inventory_region: Optional[ScreenRegion] = None
    
    # Colors for detection
    health_color: Tuple[int, int, int] = (255, 0, 0)  # Red
    mana_color: Tuple[int, int, int] = (0, 0, 255)  # Blue
    
    # State detection patterns
    menu_patterns: List[str] = field(default_factory=list)
    loading_patterns: List[str] = field(default_factory=list)
    
    # Custom extractors
    extractors: Dict[str, Callable] = field(default_factory=dict)


class ScreenAnalyzer:
    """Analyze game screen captures."""
    
    def __init__(self):
        """Initialize analyzer."""
        self._last_capture = None
        self._ocr_available = False
        
        # Try to import OCR
        try:
            self._ocr_available = True
        except ImportError:
            logger.debug("pytesseract not available, OCR disabled")
    
    def capture_screen(self, region: Optional[ScreenRegion] = None) -> Optional[Any]:
        """
        Capture screen or region.
        
        Args:
            region: Optional region to capture
            
        Returns:
            Image (PIL Image if available)
        """
        try:
            from PIL import ImageGrab
            
            if region:
                bbox = (region.x, region.y, 
                       region.x + region.width, 
                       region.y + region.height)
                img = ImageGrab.grab(bbox=bbox)
            else:
                img = ImageGrab.grab()
            
            self._last_capture = img
            return img
            
        except ImportError:
            logger.warning("PIL not available for screen capture")
            return None
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            return None
    
    def read_text(self, region: ScreenRegion) -> str:
        """
        Read text from screen region using OCR.
        
        Args:
            region: Region to read
            
        Returns:
            Extracted text
        """
        if not self._ocr_available:
            return ""
        
        img = self.capture_screen(region)
        if img is None:
            return ""
        
        try:
            import pytesseract
            text = pytesseract.image_to_string(img)
            return text.strip()
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""
    
    def get_dominant_color(self, region: ScreenRegion) -> Tuple[int, int, int]:
        """
        Get dominant color in a region.
        
        Args:
            region: Screen region
            
        Returns:
            RGB tuple
        """
        img = self.capture_screen(region)
        if img is None:
            return (0, 0, 0)
        
        try:
            # Resize for faster processing
            small = img.resize((10, 10))
            pixels = list(small.getdata())
            
            # Average color
            r = sum(p[0] for p in pixels) // len(pixels)
            g = sum(p[1] for p in pixels) // len(pixels)
            b = sum(p[2] for p in pixels) // len(pixels)
            
            return (r, g, b)
            
        except Exception as e:
            logger.error(f"Color analysis failed: {e}")
            return (0, 0, 0)
    
    def estimate_bar_fill(
        self,
        region: ScreenRegion,
        bar_color: Tuple[int, int, int],
        tolerance: int = 50
    ) -> float:
        """
        Estimate fill percentage of a health/mana bar.
        
        Args:
            region: Bar region
            bar_color: Expected bar color
            tolerance: Color tolerance
            
        Returns:
            Fill percentage (0-1)
        """
        img = self.capture_screen(region)
        if img is None:
            return 0.0
        
        try:
            pixels = list(img.getdata())
            width = img.width
            
            filled_pixels = 0
            for x in range(width):
                # Check first row of pixels
                pixel = pixels[x]
                if self._color_match(pixel, bar_color, tolerance):
                    filled_pixels += 1
            
            return filled_pixels / width
            
        except Exception as e:
            logger.error(f"Bar estimation failed: {e}")
            return 0.0
    
    def _color_match(
        self,
        color1: Tuple[int, ...],
        color2: Tuple[int, int, int],
        tolerance: int
    ) -> bool:
        """Check if colors match within tolerance."""
        return all(
            abs(c1 - c2) <= tolerance
            for c1, c2 in zip(color1[:3], color2)
        )


class GameStateReader:
    """Read game state from screen and memory."""
    
    def __init__(self):
        """Initialize reader."""
        self._profiles: Dict[str, GameProfile] = {}
        self._current_profile: Optional[GameProfile] = None
        self._analyzer = ScreenAnalyzer()
        
        # State history
        self._history: List[GameState] = []
        self._max_history = 100
        
        # Event listeners
        self._listeners: List[Callable[[GameState, Optional[GameState]], None]] = []
        
        # Load built-in profiles
        self._load_builtin_profiles()
    
    def _load_builtin_profiles(self):
        """Load built-in game profiles."""
        # Minecraft
        self._profiles["minecraft"] = GameProfile(
            game_name="Minecraft",
            process_name="javaw.exe",
            health_region=ScreenRegion(10, 10, 200, 20, "health"),
            menu_patterns=["Singleplayer", "Multiplayer", "Options"]
        )
        
        # Generic profile
        self._profiles["generic"] = GameProfile(
            game_name="Generic",
            process_name=""
        )
    
    def configure_game(self, game_name: str, config: Dict[str, Any]):
        """
        Configure a game profile.
        
        Args:
            game_name: Game identifier
            config: Configuration dict
        """
        profile = self._profiles.get(game_name) or GameProfile(game_name=game_name)
        
        # Apply config
        if "health_region" in config:
            r = config["health_region"]
            profile.health_region = ScreenRegion(r[0], r[1], r[2], r[3], "health")
        
        if "mana_region" in config:
            r = config["mana_region"]
            profile.mana_region = ScreenRegion(r[0], r[1], r[2], r[3], "mana")
        
        if "process_name" in config:
            profile.process_name = config["process_name"]
        
        self._profiles[game_name] = profile
    
    def set_active_game(self, game_name: str):
        """Set the active game profile."""
        self._current_profile = self._profiles.get(game_name)
        if not self._current_profile:
            logger.warning(f"Unknown game: {game_name}, using generic")
            self._current_profile = self._profiles["generic"]
    
    def get_state(self) -> GameState:
        """
        Get current game state.
        
        Returns:
            Current game state
        """
        if not self._current_profile:
            self._current_profile = self._profiles["generic"]
        
        state = GameState(game_name=self._current_profile.game_name)
        
        # Read health
        if self._current_profile.health_region:
            fill = self._analyzer.estimate_bar_fill(
                self._current_profile.health_region,
                self._current_profile.health_color
            )
            state.health = fill * state.max_health
        
        # Read mana
        if self._current_profile.mana_region:
            fill = self._analyzer.estimate_bar_fill(
                self._current_profile.mana_region,
                self._current_profile.mana_color
            )
            state.mana = fill * 100
        
        # Detect menu/loading
        state.is_menu = self._detect_menu()
        state.is_loading = self._detect_loading()
        state.is_playing = not state.is_menu and not state.is_loading
        
        # Run custom extractors
        for name, extractor in self._current_profile.extractors.items():
            try:
                state.custom[name] = extractor(self._analyzer)
            except Exception as e:
                logger.error(f"Extractor {name} failed: {e}")
        
        # Check for changes and notify
        prev_state = self._history[-1] if self._history else None
        self._notify_listeners(state, prev_state)
        
        # Store in history
        self._history.append(state)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        
        return state
    
    def _detect_menu(self) -> bool:
        """Detect if game is in menu."""
        if not self._current_profile or not self._current_profile.menu_patterns:
            return False
        
        # Try to read screen for menu text
        text = self._analyzer.read_text(ScreenRegion(0, 0, 400, 300))
        
        for pattern in self._current_profile.menu_patterns:
            if pattern.lower() in text.lower():
                return True
        
        return False
    
    def _detect_loading(self) -> bool:
        """Detect if game is loading."""
        if not self._current_profile or not self._current_profile.loading_patterns:
            return False
        
        text = self._analyzer.read_text(ScreenRegion(0, 0, 400, 300))
        
        for pattern in self._current_profile.loading_patterns:
            if pattern.lower() in text.lower():
                return True
        
        return False
    
    def add_listener(self, callback: Callable[[GameState, Optional[GameState]], None]):
        """Add state change listener."""
        self._listeners.append(callback)
    
    def _notify_listeners(self, state: GameState, prev_state: Optional[GameState]):
        """Notify listeners of state change."""
        for listener in self._listeners:
            try:
                listener(state, prev_state)
            except Exception as e:
                logger.error(f"Listener error: {e}")
    
    def get_history(self) -> List[GameState]:
        """Get state history."""
        return list(self._history)
    
    def add_extractor(self, name: str, extractor: Callable):
        """
        Add custom state extractor.
        
        Args:
            name: Extractor name
            extractor: Function that takes ScreenAnalyzer and returns value
        """
        if self._current_profile:
            self._current_profile.extractors[name] = extractor


# Event detection helpers
def detect_low_health(state: GameState, threshold: float = 0.2) -> bool:
    """Detect low health condition."""
    return state.health < state.max_health * threshold


def detect_death(current: GameState, previous: Optional[GameState]) -> bool:
    """Detect player death."""
    if previous is None:
        return False
    return previous.health > 0 and current.health <= 0


def detect_combat(current: GameState, previous: Optional[GameState]) -> bool:
    """Detect combat (health decrease)."""
    if previous is None:
        return False
    return current.health < previous.health


# Convenience function
def create_game_reader() -> GameStateReader:
    """Create a new game state reader."""
    return GameStateReader()
