"""
================================================================================
Gaming Mode - Run Enigma AI Engine alongside games without performance impact.
================================================================================

This module provides intelligent resource management that allows Enigma AI Engine to
run in the background while the user plays games. It automatically:

1. Detects when games are running (fullscreen apps, known game processes)
2. Reduces AI resource usage to not impact frame rates
3. Queues non-urgent tasks for when the game ends
4. Uses CPU inference instead of GPU when game needs VRAM
5. Throttles background processing based on frame rate impact

USAGE:
    from enigma_engine.core.gaming_mode import GamingMode, get_gaming_mode
    
    # Auto-detect and apply gaming optimizations
    gaming = get_gaming_mode()
    gaming.enable()
    
    # Or with specific settings
    gaming = GamingMode(
        target_fps_headroom=5,    # Stay 5 FPS below limit
        max_vram_percent=10,      # Use max 10% VRAM when gaming
        cpu_inference=True,       # Force CPU when game detected
    )

GAMING PRIORITY LEVELS:
    BACKGROUND (1): Completely hidden, minimal resources
    LOW (2): Basic features, no generation
    MEDIUM (3): Text generation ok, no image/video
    HIGH (4): All features, but throttled
    FULL (5): No restrictions (no game detected)
"""

import json
import logging
import platform
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class GamingPriority(Enum):
    """How much resources AI can use while gaming."""
    BACKGROUND = 1   # Minimal - only essential background tasks
    LOW = 2          # Basic features, queue heavy tasks
    MEDIUM = 3       # Text generation allowed
    HIGH = 4         # Most features, but throttled
    FULL = 5         # No game - full resources available


@dataclass
class GamingProfile:
    """Profile for a specific game or app."""
    name: str
    process_names: list[str] = field(default_factory=list)
    priority: GamingPriority = GamingPriority.MEDIUM
    cpu_inference: bool = True      # Force CPU inference
    max_vram_mb: int = 512          # Max VRAM AI can use
    max_ram_mb: int = 2048          # Max RAM AI can use
    batch_size: int = 1             # Force batch size 1
    defer_heavy_tasks: bool = True  # Queue image/video gen
    voice_enabled: bool = True      # Allow voice I/O
    avatar_enabled: bool = True     # Show avatar overlay
    notes: str = ""


# Default gaming profiles for popular games
DEFAULT_GAMING_PROFILES: dict[str, GamingProfile] = {
    "competitive_fps": GamingProfile(
        name="Competitive FPS",
        process_names=[
            "csgo.exe", "cs2.exe", "valorant.exe", "valorant-win64-shipping.exe",
            "overwatch.exe", "r5apex.exe", "pubg.exe", "fortnite.exe",
            "cod.exe", "modernwarfare.exe",
        ],
        priority=GamingPriority.BACKGROUND,
        cpu_inference=True,
        max_vram_mb=256,
        max_ram_mb=1024,
        defer_heavy_tasks=True,
        voice_enabled=True,  # Voice still useful for callouts
        avatar_enabled=False,  # No overlay in competitive
    ),
    "singleplayer_rpg": GamingProfile(
        name="Singleplayer RPG",
        process_names=[
            "witcher3.exe", "cyberpunk2077.exe", "eldenring.exe",
            "baldursgate3.exe", "bg3.exe", "starfield.exe",
            "skyrim.exe", "fallout4.exe",
        ],
        priority=GamingPriority.MEDIUM,
        cpu_inference=True,
        max_vram_mb=512,
        max_ram_mb=2048,
        defer_heavy_tasks=True,
        voice_enabled=True,
        avatar_enabled=True,  # Companion avatar OK
    ),
    "strategy": GamingProfile(
        name="Strategy Games",
        process_names=[
            "stellaris.exe", "ck3.exe", "eu4.exe", "hoi4.exe",
            "civ6.exe", "totalwar.exe", "aoe4.exe",
        ],
        priority=GamingPriority.HIGH,
        cpu_inference=False,  # Usually GPU headroom
        max_vram_mb=1024,
        max_ram_mb=4096,
        defer_heavy_tasks=False,
        voice_enabled=True,
        avatar_enabled=True,
    ),
    "vr": GamingProfile(
        name="VR Games",
        process_names=[
            "vrchat.exe", "boneworks.exe", "alyx.exe",
            "beatsaber.exe", "pavlov.exe",
        ],
        priority=GamingPriority.BACKGROUND,
        cpu_inference=True,
        max_vram_mb=128,  # VR needs all VRAM
        max_ram_mb=1024,
        defer_heavy_tasks=True,
        voice_enabled=True,  # Voice is great for VR
        avatar_enabled=False,
    ),
    "creative": GamingProfile(
        name="Creative Software",
        process_names=[
            "photoshop.exe", "premiere.exe", "davinci.exe",
            "blender.exe", "unity.exe", "unreal.exe",
            "obs64.exe", "obs.exe",
        ],
        priority=GamingPriority.LOW,
        cpu_inference=True,
        max_vram_mb=256,
        max_ram_mb=2048,
        defer_heavy_tasks=True,
        voice_enabled=True,
        avatar_enabled=False,
    ),
}


@dataclass
class ResourceLimits:
    """Current resource limits for AI."""
    max_vram_mb: int = 0          # 0 = unlimited
    max_ram_mb: int = 0
    batch_size: int = 0           # 0 = default
    cpu_only: bool = False
    max_threads: int = 0          # 0 = default
    generation_allowed: bool = True
    heavy_tasks_allowed: bool = True
    priority: GamingPriority = GamingPriority.FULL


class GamingMode:
    """
    Manages AI resource usage while gaming.
    
    Automatically detects games and adjusts resources to minimize impact.
    """
    
    def __init__(
        self,
        target_fps_headroom: int = 5,
        max_vram_percent: float = 0.1,
        cpu_inference: bool = False,
        custom_profiles: dict[str, GamingProfile] = None,
        check_interval: float = 5.0,
    ):
        """
        Initialize gaming mode.
        
        Args:
            target_fps_headroom: Try to stay this many FPS below monitor refresh
            max_vram_percent: Max VRAM to use as fraction (0.1 = 10%)
            cpu_inference: Force CPU-only inference when gaming
            custom_profiles: Additional game profiles
            check_interval: How often to check for games (seconds)
        """
        self.target_fps_headroom = target_fps_headroom
        self.max_vram_percent = max_vram_percent
        self.default_cpu_inference = cpu_inference
        self.check_interval = check_interval
        
        # Profiles
        self.profiles = {**DEFAULT_GAMING_PROFILES}
        if custom_profiles:
            self.profiles.update(custom_profiles)
        
        # Build process name to profile lookup
        self._process_to_profile: dict[str, GamingProfile] = {}
        for profile in self.profiles.values():
            for proc in profile.process_names:
                self._process_to_profile[proc.lower()] = profile
        
        # Current state
        self._enabled = False
        self._active_game: Optional[str] = None
        self._active_profile: Optional[GamingProfile] = None
        self._current_limits = ResourceLimits()
        
        # Monitoring
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Deferred task queue
        self._deferred_tasks: list[dict[str, Any]] = []
        
        # Callbacks
        self._on_game_start: list[Callable[[str, GamingProfile], None]] = []
        self._on_game_end: list[Callable[[str], None]] = []
        self._on_limits_change: list[Callable[[ResourceLimits], None]] = []
    
    @property
    def enabled(self) -> bool:
        """Whether gaming mode is enabled."""
        return self._enabled
    
    @property
    def active_game(self) -> Optional[str]:
        """Currently detected game process."""
        return self._active_game
    
    @property
    def limits(self) -> ResourceLimits:
        """Current resource limits."""
        return self._current_limits
    
    def enable(self):
        """Enable gaming mode monitoring."""
        if self._enabled:
            return
        
        self._enabled = True
        self._stop_event.clear()
        
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="GamingModeMonitor",
        )
        self._monitor_thread.start()
        
        logger.info("Gaming mode enabled")
    
    def disable(self):
        """Disable gaming mode monitoring."""
        if not self._enabled:
            return
        
        self._enabled = False
        self._stop_event.set()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
        
        # Restore full limits
        self._current_limits = ResourceLimits()
        self._notify_limits_change()
        
        # Process deferred tasks
        self._process_deferred_tasks()
        
        logger.info("Gaming mode disabled")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            try:
                self._check_games()
            except Exception as e:
                logger.error(f"Gaming mode monitor error: {e}")
            
            self._stop_event.wait(self.check_interval)
    
    def _check_games(self):
        """Check for running games and update limits."""
        running_processes = self._get_running_processes()
        
        # Find highest-priority matching game
        best_match: Optional[str] = None
        best_profile: Optional[GamingProfile] = None
        best_priority = GamingPriority.FULL
        
        for proc in running_processes:
            proc_lower = proc.lower()
            if proc_lower in self._process_to_profile:
                profile = self._process_to_profile[proc_lower]
                if profile.priority.value < best_priority.value:
                    best_match = proc
                    best_profile = profile
                    best_priority = profile.priority
        
        # Also check for fullscreen apps (generic game detection)
        if not best_match and self._is_fullscreen_app_active():
            best_match = "unknown_fullscreen"
            best_profile = GamingProfile(
                name="Unknown Game",
                priority=GamingPriority.MEDIUM,
                cpu_inference=self.default_cpu_inference,
            )
            best_priority = GamingPriority.MEDIUM
        
        # Update state if changed
        old_game = self._active_game
        
        if best_match != self._active_game:
            self._active_game = best_match
            self._active_profile = best_profile
            
            if best_match:
                logger.info(f"Game detected: {best_match} (profile: {best_profile.name})")
                self._apply_profile(best_profile)
                self._notify_game_start(best_match, best_profile)
            elif old_game:
                logger.info(f"Game ended: {old_game}")
                self._restore_full_limits()
                self._notify_game_end(old_game)
                self._process_deferred_tasks()
    
    def _get_running_processes(self) -> set[str]:
        """Get set of running process names."""
        processes = set()
        
        try:
            if platform.system() == "Windows":
                # Use tasklist on Windows
                output = subprocess.check_output(
                    ["tasklist", "/fo", "csv", "/nh"],
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                for line in output.strip().split('\n'):
                    if line:
                        parts = line.split('","')
                        if parts:
                            proc_name = parts[0].strip('"')
                            processes.add(proc_name)
            else:
                # Use ps on Unix
                output = subprocess.check_output(
                    ["ps", "-A", "-o", "comm="],
                    text=True,
                )
                for line in output.strip().split('\n'):
                    if line:
                        processes.add(line.strip())
                        
        except Exception as e:
            logger.debug(f"Could not get process list: {e}")
        
        return processes
    
    def _is_fullscreen_app_active(self) -> bool:
        """Check if a fullscreen application is active."""
        if platform.system() != "Windows":
            return False
        
        try:
            import ctypes
            from ctypes import wintypes
            
            user32 = ctypes.windll.user32
            
            # Get foreground window
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return False
            
            # Get window rect
            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            
            # Get screen size
            screen_width = user32.GetSystemMetrics(0)
            screen_height = user32.GetSystemMetrics(1)
            
            # Check if window covers entire screen
            window_width = rect.right - rect.left
            window_height = rect.bottom - rect.top
            
            return (
                window_width >= screen_width and 
                window_height >= screen_height
            )
            
        except Exception:
            return False
    
    def _apply_profile(self, profile: GamingProfile):
        """Apply a gaming profile's limits."""
        self._current_limits = ResourceLimits(
            max_vram_mb=profile.max_vram_mb,
            max_ram_mb=profile.max_ram_mb,
            batch_size=profile.batch_size,
            cpu_only=profile.cpu_inference,
            generation_allowed=profile.priority.value >= GamingPriority.MEDIUM.value,
            heavy_tasks_allowed=not profile.defer_heavy_tasks,
            priority=profile.priority,
        )
        
        self._notify_limits_change()
    
    def _restore_full_limits(self):
        """Restore full resource limits."""
        self._current_limits = ResourceLimits()
        self._active_profile = None
        self._notify_limits_change()
    
    def _notify_limits_change(self):
        """Notify callbacks of limit changes."""
        for callback in self._on_limits_change:
            try:
                callback(self._current_limits)
            except Exception as e:
                logger.error(f"Limits change callback error: {e}")
    
    def _notify_game_start(self, game: str, profile: GamingProfile):
        """Notify callbacks of game start."""
        for callback in self._on_game_start:
            try:
                callback(game, profile)
            except Exception as e:
                logger.error(f"Game start callback error: {e}")
    
    def _notify_game_end(self, game: str):
        """Notify callbacks of game end."""
        for callback in self._on_game_end:
            try:
                callback(game)
            except Exception as e:
                logger.error(f"Game end callback error: {e}")
    
    def defer_task(self, task_type: str, task_data: dict[str, Any]):
        """Queue a task to run after gaming ends."""
        self._deferred_tasks.append({
            "type": task_type,
            "data": task_data,
            "queued_at": time.time(),
        })
        logger.debug(f"Deferred task: {task_type}")
    
    def _process_deferred_tasks(self):
        """Process queued deferred tasks."""
        if not self._deferred_tasks:
            return
        
        logger.info(f"Processing {len(self._deferred_tasks)} deferred tasks")
        
        # Process in background thread
        tasks = self._deferred_tasks.copy()
        self._deferred_tasks.clear()
        
        def process():
            for task in tasks:
                task_type = task.get('type', 'unknown')
                task_data = task.get('data', {})
                
                try:
                    # Dispatch to appropriate handler based on task type
                    if task_type == 'text_generation':
                        # Deferred text generation
                        from .inference import EnigmaEngine
                        engine = EnigmaEngine()
                        if 'prompt' in task_data:
                            result = engine.generate(task_data['prompt'], max_tokens=task_data.get('max_tokens', 100))
                            logger.debug(f"Completed deferred text generation: {len(result)} chars")
                    
                    elif task_type == 'embedding':
                        # Deferred embedding computation
                        from ..memory.vector_db import get_embedding_function
                        embed_fn = get_embedding_function()
                        if 'texts' in task_data:
                            embeddings = embed_fn(task_data['texts'])
                            logger.debug(f"Computed {len(embeddings)} deferred embeddings")
                    
                    elif task_type == 'memory_save':
                        # Deferred memory operations
                        from ..memory.manager import ConversationManager
                        if 'conversation_id' in task_data and 'messages' in task_data:
                            manager = ConversationManager()
                            for msg in task_data['messages']:
                                manager.add_message(task_data['conversation_id'], msg['role'], msg['content'])
                            logger.debug(f"Saved {len(task_data['messages'])} deferred messages")
                    
                    elif task_type == 'training_example':
                        # Deferred training data collection
                        from ..learning.training_scheduler import get_training_scheduler
                        scheduler = get_training_scheduler()
                        if 'example' in task_data:
                            scheduler.add_example(task_data['example'])
                            logger.debug("Added deferred training example")
                    
                    elif task_type == 'notification':
                        # Deferred notification
                        try:
                            from ..utils.notifications import send_notification
                            send_notification(
                                task_data.get('title', 'Enigma AI Engine'),
                                task_data.get('message', '')
                            )
                        except ImportError:
                            logger.debug(f"Deferred notification: {task_data.get('message', '')}")
                    
                    else:
                        logger.warning(f"Unknown deferred task type: {task_type}")
                        
                except Exception as e:
                    logger.error(f"Error processing deferred task {task_type}: {e}")
        
        threading.Thread(target=process, daemon=True).start()
    
    def can_generate(self, task_type: str = "text") -> bool:
        """Check if generation is allowed right now."""
        if not self._enabled or not self._active_game:
            return True
        
        if task_type in ("image", "video", "audio"):
            return self._current_limits.heavy_tasks_allowed
        
        return self._current_limits.generation_allowed
    
    def should_use_cpu(self) -> bool:
        """Check if CPU inference should be used."""
        return self._current_limits.cpu_only
    
    def get_max_batch_size(self) -> int:
        """Get max batch size for current mode."""
        if self._current_limits.batch_size > 0:
            return self._current_limits.batch_size
        return 0  # 0 = default
    
    def on_game_start(self, callback: Callable[[str, GamingProfile], None]):
        """Register callback for when a game starts."""
        self._on_game_start.append(callback)
    
    def on_game_end(self, callback: Callable[[str], None]):
        """Register callback for when a game ends."""
        self._on_game_end.append(callback)
    
    def on_limits_change(self, callback: Callable[[ResourceLimits], None]):
        """Register callback for when limits change."""
        self._on_limits_change.append(callback)
    
    def add_game_profile(self, profile: GamingProfile):
        """Add or update a game profile."""
        key = profile.name.lower().replace(" ", "_")
        self.profiles[key] = profile
        
        for proc in profile.process_names:
            self._process_to_profile[proc.lower()] = profile
    
    def save_profiles(self, path: Path = None):
        """Save custom profiles to file."""
        if path is None:
            from ..config import CONFIG
            path = Path(CONFIG.get("data_dir", "data")) / "gaming_profiles.json"
        
        data = {}
        for key, profile in self.profiles.items():
            if key not in DEFAULT_GAMING_PROFILES:
                data[key] = {
                    "name": profile.name,
                    "process_names": profile.process_names,
                    "priority": profile.priority.name,
                    "cpu_inference": profile.cpu_inference,
                    "max_vram_mb": profile.max_vram_mb,
                    "max_ram_mb": profile.max_ram_mb,
                    "batch_size": profile.batch_size,
                    "defer_heavy_tasks": profile.defer_heavy_tasks,
                    "voice_enabled": profile.voice_enabled,
                    "avatar_enabled": profile.avatar_enabled,
                    "notes": profile.notes,
                }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_profiles(self, path: Path = None):
        """Load custom profiles from file."""
        if path is None:
            from ..config import CONFIG
            path = Path(CONFIG.get("data_dir", "data")) / "gaming_profiles.json"
        
        if not path.exists():
            return
        
        try:
            with open(path) as f:
                data = json.load(f)
            
            for key, pdata in data.items():
                profile = GamingProfile(
                    name=pdata["name"],
                    process_names=pdata.get("process_names", []),
                    priority=GamingPriority[pdata.get("priority", "MEDIUM")],
                    cpu_inference=pdata.get("cpu_inference", True),
                    max_vram_mb=pdata.get("max_vram_mb", 512),
                    max_ram_mb=pdata.get("max_ram_mb", 2048),
                    batch_size=pdata.get("batch_size", 1),
                    defer_heavy_tasks=pdata.get("defer_heavy_tasks", True),
                    voice_enabled=pdata.get("voice_enabled", True),
                    avatar_enabled=pdata.get("avatar_enabled", True),
                    notes=pdata.get("notes", ""),
                )
                self.add_game_profile(profile)
                
        except Exception as e:
            logger.error(f"Failed to load gaming profiles: {e}")
    
    def get_status(self) -> dict[str, Any]:
        """Get current gaming mode status."""
        return {
            "enabled": self._enabled,
            "active_game": self._active_game,
            "active_profile": self._active_profile.name if self._active_profile else None,
            "priority": self._current_limits.priority.name,
            "cpu_only": self._current_limits.cpu_only,
            "generation_allowed": self._current_limits.generation_allowed,
            "heavy_tasks_allowed": self._current_limits.heavy_tasks_allowed,
            "deferred_tasks": len(self._deferred_tasks),
        }


# Global instance
_gaming_mode: Optional[GamingMode] = None


def get_gaming_mode(**kwargs) -> GamingMode:
    """Get or create the global gaming mode instance."""
    global _gaming_mode
    if _gaming_mode is None:
        _gaming_mode = GamingMode(**kwargs)
    return _gaming_mode


__all__ = [
    'GamingMode',
    'GamingPriority',
    'GamingProfile',
    'ResourceLimits',
    'get_gaming_mode',
]
