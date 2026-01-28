"""
Game Mode - Zero Lag Gaming with AI Companion

Automatically detects games and reduces AI resource usage to prevent frame drops.
"""

import logging
import threading
import time
from typing import Optional, Callable, List, Dict, Any
from pathlib import Path
import json

from .process_monitor import ProcessMonitor, get_process_monitor
from .resource_limiter import ResourceLimits, ResourceLimiter

logger = logging.getLogger(__name__)


class GameMode:
    """
    Gaming-optimized AI mode.
    
    When active:
    - Reduces CPU usage to <5%
    - Reduces GPU usage to 0% (offload to CPU or pause)
    - Disables background tasks (autonomous mode pauses)
    - Enables overlay mode
    - Enables hotkey activation
    - Auto-detects fullscreen games
    """
    
    def __init__(self):
        """Initialize game mode."""
        self._enabled = False
        self._aggressive = False
        self._active = False  # True when game is detected
        
        # Process monitor
        self._process_monitor = get_process_monitor()
        
        # Resource limiter
        self._normal_limits = ResourceLimits(
            max_cpu_percent=100.0,
            max_memory_mb=8192,
            gpu_allowed=True,
            background_tasks=True,
            inference_allowed=True,
            max_response_tokens=512,
            batch_processing=True,
        )
        
        self._game_limits_balanced = ResourceLimits(
            max_cpu_percent=10.0,
            max_memory_mb=500,
            gpu_allowed=False,
            background_tasks=False,
            inference_allowed=True,
            max_response_tokens=100,
            batch_processing=False,
        )
        
        self._game_limits_aggressive = ResourceLimits(
            max_cpu_percent=5.0,
            max_memory_mb=300,
            gpu_allowed=False,
            background_tasks=False,
            inference_allowed=True,
            max_response_tokens=50,
            batch_processing=False,
        )
        
        self._current_limits = self._normal_limits
        self._resource_limiter = ResourceLimiter(self._current_limits)
        
        # Watcher thread
        self._watcher: Optional['GameModeWatcher'] = None
        
        # Callbacks
        self._on_game_detected: List[Callable[[str], None]] = []
        self._on_game_ended: List[Callable[[], None]] = []
        self._on_limits_changed: List[Callable[[ResourceLimits], None]] = []
        
        # Load config
        self._load_config()
    
    def enable(self, aggressive: bool = False):
        """
        Enable game mode.
        
        Args:
            aggressive: If True, maximum performance mode
                       If False, balanced mode
        """
        self._enabled = True
        self._aggressive = aggressive
        
        # Start watcher if not running
        if self._watcher is None:
            self._watcher = GameModeWatcher(self)
            self._watcher.start()
        
        # Start resource monitoring
        self._resource_limiter.start_monitoring()
        
        # Save config
        self._save_config()
        
        logger.info(f"Game mode enabled (aggressive={aggressive})")
    
    def disable(self):
        """Disable game mode and return to normal operation."""
        self._enabled = False
        
        # Stop watcher
        if self._watcher:
            self._watcher.stop()
            self._watcher = None
        
        # Stop resource monitoring
        self._resource_limiter.stop_monitoring()
        
        # Restore normal limits
        if self._active:
            self._deactivate_game_mode()
        
        # Save config
        self._save_config()
        
        logger.info("Game mode disabled")
    
    def auto_detect_game(self) -> bool:
        """
        Detect if a game is running.
        
        Checks:
        - Fullscreen applications
        - Known game processes
        - User-defined game list
        
        Returns:
            True if game detected
        """
        # Check for known games
        if self._process_monitor.is_game_running():
            return True
        
        # Check for fullscreen app
        fullscreen_app = self._process_monitor.get_fullscreen_app()
        if fullscreen_app:
            logger.debug(f"Fullscreen app detected: {fullscreen_app}")
            return True
        
        return False
    
    def get_resource_limits(self) -> ResourceLimits:
        """Get current resource limits."""
        return self._current_limits
    
    def _activate_game_mode(self, game_name: str):
        """Activate game mode restrictions."""
        if self._active:
            return
        
        self._active = True
        
        # Apply appropriate limits
        if self._aggressive:
            self._current_limits = self._game_limits_aggressive
        else:
            self._current_limits = self._game_limits_balanced
        
        self._resource_limiter.update_limits(self._current_limits)
        
        # Notify callbacks
        for callback in self._on_game_detected:
            try:
                callback(game_name)
            except Exception as e:
                logger.error(f"Game detected callback error: {e}")
        
        for callback in self._on_limits_changed:
            try:
                callback(self._current_limits)
            except Exception as e:
                logger.error(f"Limits changed callback error: {e}")
        
        logger.info(f"Game mode activated for: {game_name}")
    
    def _deactivate_game_mode(self):
        """Deactivate game mode and restore normal limits."""
        if not self._active:
            return
        
        self._active = False
        self._current_limits = self._normal_limits
        self._resource_limiter.update_limits(self._current_limits)
        
        # Notify callbacks
        for callback in self._on_game_ended:
            try:
                callback()
            except Exception as e:
                logger.error(f"Game ended callback error: {e}")
        
        for callback in self._on_limits_changed:
            try:
                callback(self._current_limits)
            except Exception as e:
                logger.error(f"Limits changed callback error: {e}")
        
        logger.info("Game mode deactivated")
    
    def on_game_detected(self, callback: Callable[[str], None]):
        """Register callback for when game is detected."""
        self._on_game_detected.append(callback)
    
    def on_game_ended(self, callback: Callable[[], None]):
        """Register callback for when game ends."""
        self._on_game_ended.append(callback)
    
    def on_limits_changed(self, callback: Callable[[ResourceLimits], None]):
        """Register callback for when resource limits change."""
        self._on_limits_changed.append(callback)
    
    def is_enabled(self) -> bool:
        """Check if game mode is enabled."""
        return self._enabled
    
    def is_active(self) -> bool:
        """Check if game mode is currently active (game detected)."""
        return self._active
    
    def get_status(self) -> Dict[str, Any]:
        """Get current game mode status."""
        return {
            "enabled": self._enabled,
            "aggressive": self._aggressive,
            "active": self._active,
            "limits": {
                "max_cpu_percent": self._current_limits.max_cpu_percent,
                "max_memory_mb": self._current_limits.max_memory_mb,
                "gpu_allowed": self._current_limits.gpu_allowed,
                "background_tasks": self._current_limits.background_tasks,
                "inference_allowed": self._current_limits.inference_allowed,
            },
        }
    
    def _load_config(self):
        """Load game mode configuration from file."""
        try:
            from ..config import CONFIG
            config_dir = Path(CONFIG.get("data_dir", "data"))
            config_file = config_dir / "game_mode_config.json"
            
            if config_file.exists():
                with open(config_file) as f:
                    data = json.load(f)
                
                self._enabled = data.get("enabled", False)
                self._aggressive = data.get("aggressive", False)
                
                # Load custom games
                custom_games = data.get("custom_games", [])
                for game in custom_games:
                    self._process_monitor.add_custom_game(game)
                
                logger.info("Loaded game mode config")
        
        except Exception as e:
            logger.debug(f"Could not load game mode config: {e}")
    
    def _save_config(self):
        """Save game mode configuration to file."""
        try:
            from ..config import CONFIG
            config_dir = Path(CONFIG.get("data_dir", "data"))
            config_dir.mkdir(parents=True, exist_ok=True)
            config_file = config_dir / "game_mode_config.json"
            
            data = {
                "enabled": self._enabled,
                "aggressive": self._aggressive,
                "custom_games": self._process_monitor.custom_games,
            }
            
            with open(config_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Could not save game mode config: {e}")


class GameModeWatcher:
    """
    Background thread that auto-enables game mode when games are detected.
    """
    
    def __init__(self, game_mode: GameMode):
        """
        Initialize watcher.
        
        Args:
            game_mode: GameMode instance to manage
        """
        self.game_mode = game_mode
        self.check_interval = 5.0  # seconds
        self.enabled = True
        
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_game: Optional[str] = None
    
    def start(self):
        """Start watching for games."""
        if self._thread and self._thread.is_alive():
            return
        
        self.enabled = True
        self._stop_event.clear()
        
        self._thread = threading.Thread(
            target=self._watch_loop,
            daemon=True,
            name="GameModeWatcher"
        )
        self._thread.start()
        
        logger.info("Game mode watcher started")
    
    def stop(self):
        """Stop watching."""
        self.enabled = False
        self._stop_event.set()
        
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        
        logger.info("Game mode watcher stopped")
    
    def _watch_loop(self):
        """Main watch loop."""
        while not self._stop_event.is_set():
            try:
                self._check_for_games()
            except Exception as e:
                logger.error(f"Game watcher error: {e}")
            
            self._stop_event.wait(self.check_interval)
    
    def _check_for_games(self):
        """Check if games are running and update game mode."""
        if not self.game_mode.is_enabled():
            return
        
        # Get running games
        running_games = self.game_mode._process_monitor.get_running_games()
        fullscreen = self.game_mode._process_monitor.get_fullscreen_app()
        
        # Determine current game
        current_game = None
        if running_games:
            current_game = list(running_games)[0]
        elif fullscreen:
            current_game = fullscreen
        
        # Check if game state changed
        if current_game and not self.game_mode.is_active():
            # Game started
            self.on_game_detected(current_game)
            self._last_game = current_game
        
        elif not current_game and self.game_mode.is_active():
            # Game ended
            self.on_game_closed()
            self._last_game = None
    
    def on_game_detected(self, game_name: str):
        """Called when game starts - activate game mode."""
        logger.info(f"Game detected: {game_name}")
        self.game_mode._activate_game_mode(game_name)
    
    def on_game_closed(self):
        """Called when game ends - deactivate game mode."""
        logger.info("Game closed")
        
        # Wait a bit before resuming (user might restart)
        time.sleep(5)
        
        # Check again if still no game
        if not self.game_mode.auto_detect_game():
            self.game_mode._deactivate_game_mode()


# Global instance
_game_mode: Optional[GameMode] = None


def get_game_mode() -> GameMode:
    """Get or create global GameMode instance."""
    global _game_mode
    if _game_mode is None:
        _game_mode = GameMode()
    return _game_mode


__all__ = ['GameMode', 'GameModeWatcher', 'get_game_mode']
