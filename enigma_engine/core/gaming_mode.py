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
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

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


@dataclass
class FPSStats:
    """FPS monitoring statistics."""
    current_fps: float = 0.0
    average_fps: float = 0.0
    min_fps: float = 0.0
    max_fps: float = 0.0
    target_fps: float = 60.0
    samples: int = 0
    fps_drop_detected: bool = False
    last_update: float = 0.0


class FPSMonitor:
    """
    Monitor game frame rates to dynamically adjust AI resource usage.
    
    Uses multiple methods to estimate FPS:
    1. GPU utilization polling (NVIDIA/AMD APIs)
    2. Present rate from D3D11 (Windows)
    3. Process CPU/GPU activity correlation
    """
    
    def __init__(self, target_fps: float = 60.0, headroom: float = 5.0):
        self.target_fps = target_fps
        self.headroom = headroom  # How many FPS below target to maintain
        self.stats = FPSStats(target_fps=target_fps)
        
        self._fps_history: list[float] = []
        self._max_history = 60  # Keep last 60 samples
        self._lock = threading.Lock()
        
        # Detection method availability
        self._nvidia_available = False
        self._amd_available = False
        self._wmi_available = False
        
        self._detect_available_methods()
    
    def _detect_available_methods(self):
        """Detect which FPS monitoring methods are available."""
        # Check for pynvml (NVIDIA)
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvidia_available = True
            pynvml.nvmlShutdown()
        except Exception:
            self._nvidia_available = False
        
        # Check for WMI (Windows performance counters)
        if platform.system() == "Windows":
            try:
                self._wmi_available = True
            except Exception:
                self._wmi_available = False
    
    def get_gpu_activity(self) -> tuple[float, float]:
        """
        Get GPU utilization and memory usage.
        
        Returns:
            Tuple of (utilization %, memory_used_mb)
        """
        if self._nvidia_available:
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                pynvml.nvmlShutdown()
                return util.gpu, mem.used / (1024 * 1024)
            except Exception:
                pass
        
        return 0.0, 0.0
    
    def estimate_fps_from_gpu(self) -> float:
        """
        Estimate FPS based on GPU utilization patterns.
        
        High GPU utilization with consistent patterns indicates
        a game running at its frame rate cap.
        """
        gpu_util, _ = self.get_gpu_activity()
        
        if gpu_util < 10:
            # GPU mostly idle - no game or game paused
            return 0.0
        elif gpu_util > 95:
            # GPU fully loaded - game likely at its cap
            return self.target_fps
        else:
            # Estimate based on utilization
            # This is very rough - real implementation would
            # use present timing from D3D/Vulkan
            return self.target_fps * (gpu_util / 100.0)
    
    def update(self) -> FPSStats:
        """
        Update FPS statistics.
        
        Returns:
            Current FPS stats
        """
        fps = self.estimate_fps_from_gpu()
        
        with self._lock:
            self._fps_history.append(fps)
            if len(self._fps_history) > self._max_history:
                self._fps_history.pop(0)
            
            if self._fps_history:
                self.stats.current_fps = fps
                self.stats.average_fps = sum(self._fps_history) / len(self._fps_history)
                self.stats.min_fps = min(self._fps_history)
                self.stats.max_fps = max(self._fps_history)
                self.stats.samples = len(self._fps_history)
            
            # Detect FPS drops
            self.stats.fps_drop_detected = (
                self.stats.average_fps > 0 and
                self.stats.current_fps < (self.stats.average_fps - self.headroom)
            )
            self.stats.last_update = time.time()
        
        return self.stats
    
    def should_reduce_load(self) -> bool:
        """Check if AI should reduce its resource usage."""
        return (
            self.stats.fps_drop_detected or
            self.stats.current_fps < (self.target_fps - self.headroom * 2)
        )
    
    def get_recommended_scale(self) -> float:
        """
        Get recommended scaling factor for AI resources.
        
        Returns:
            Float between 0.0 (stop everything) and 1.0 (full resources)
        """
        if self.stats.average_fps <= 0:
            return 1.0  # No data - don't restrict
        
        # Calculate how far we are from target
        fps_ratio = self.stats.current_fps / self.target_fps
        
        if fps_ratio >= 1.0:
            return 1.0  # At or above target - full resources
        elif fps_ratio >= 0.9:
            return 0.8  # Slightly below - minor reduction
        elif fps_ratio >= 0.75:
            return 0.5  # Noticeably below - significant reduction
        elif fps_ratio >= 0.5:
            return 0.2  # Major drop - minimal resources
        else:
            return 0.0  # Severe drop - pause AI tasks
    
    def reset(self):
        """Reset FPS statistics."""
        with self._lock:
            self._fps_history.clear()
            self.stats = FPSStats(target_fps=self.target_fps)


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
        
        # Auto-load saved custom profiles
        self.load_profiles()
        
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
        self._on_fps_update: list[Callable[[FPSStats], None]] = []
        
        # FPS monitoring
        self._fps_monitor = FPSMonitor(target_fps=60.0, headroom=target_fps_headroom)
        self._fps_adaptive_enabled = True  # Enable adaptive resource scaling
        self._last_fps_scale = 1.0
    
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
        
        # Also enable fullscreen controller for visibility management
        try:
            from .fullscreen_mode import get_fullscreen_controller
            controller = get_fullscreen_controller()
            controller.enable()
        except Exception:
            pass
        
        # Hook up screen effects integration
        try:
            from ..avatar.screen_effects import setup_gaming_mode_integration
            setup_gaming_mode_integration()
        except Exception:
            pass
        
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
                
                # Update FPS monitoring if a game is active
                if self._active_game and self._fps_adaptive_enabled:
                    self._update_fps_monitoring()
                    
            except Exception as e:
                logger.error(f"Gaming mode monitor error: {e}")
            
            self._stop_event.wait(self.check_interval)
    
    def _update_fps_monitoring(self):
        """Update FPS stats and adjust resources if needed."""
        stats = self._fps_monitor.update()
        
        # Notify callbacks
        for callback in self._on_fps_update:
            try:
                callback(stats)
            except Exception as e:
                logger.error(f"FPS update callback error: {e}")
        
        # Get recommended scaling
        new_scale = self._fps_monitor.get_recommended_scale()
        
        # Only adjust if scale changed significantly
        if abs(new_scale - self._last_fps_scale) > 0.1:
            self._last_fps_scale = new_scale
            self._adjust_limits_for_fps(new_scale)
    
    def _adjust_limits_for_fps(self, scale: float):
        """Adjust resource limits based on FPS scaling factor."""
        if not self._active_profile:
            return
        
        profile = self._active_profile
        
        # Scale the limits down based on FPS pressure
        self._current_limits.max_vram_mb = int(profile.max_vram_mb * scale)
        self._current_limits.max_ram_mb = int(profile.max_ram_mb * scale)
        self._current_limits.batch_size = max(1, int(profile.batch_size * scale))
        
        # At very low scale, disable generation
        if scale < 0.3:
            self._current_limits.generation_allowed = False
            self._current_limits.heavy_tasks_allowed = False
        elif scale < 0.6:
            self._current_limits.generation_allowed = True
            self._current_limits.heavy_tasks_allowed = False
        else:
            self._current_limits.generation_allowed = profile.priority.value >= GamingPriority.MEDIUM.value
            self._current_limits.heavy_tasks_allowed = not profile.defer_heavy_tasks
        
        logger.debug(f"Adjusted limits for FPS (scale={scale:.2f})")
        self._notify_limits_change()
    
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
                    timeout=10
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
                    timeout=10
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
        
        # Also control avatar visibility via fullscreen controller
        try:
            from .fullscreen_mode import get_fullscreen_controller
            controller = get_fullscreen_controller()
            controller.set_category_visible('avatar', profile.avatar_enabled)
            controller.set_category_visible('spawned_objects', profile.avatar_enabled)
            controller.set_category_visible('effects', profile.avatar_enabled)
        except Exception:
            pass
        
        self._notify_limits_change()
    
    def _restore_full_limits(self):
        """Restore full resource limits."""
        self._current_limits = ResourceLimits()
        self._active_profile = None
        
        # Restore avatar visibility
        try:
            from .fullscreen_mode import get_fullscreen_controller
            controller = get_fullscreen_controller()
            controller.set_category_visible('avatar', True)
            controller.set_category_visible('spawned_objects', True)
            controller.set_category_visible('effects', True)
        except Exception:
            pass
        
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
    
    def on_fps_update(self, callback: Callable[[FPSStats], None]):
        """Register callback for FPS updates."""
        self._on_fps_update.append(callback)
    
    def set_fps_adaptive(self, enabled: bool):
        """Enable or disable FPS-adaptive resource scaling."""
        self._fps_adaptive_enabled = enabled
        if not enabled:
            self._last_fps_scale = 1.0
        logger.debug(f"FPS adaptive scaling: {enabled}")
    
    def set_target_fps(self, fps: float):
        """Set the target FPS for monitoring."""
        self._fps_monitor.target_fps = fps
        self._fps_monitor.stats.target_fps = fps
        logger.debug(f"Target FPS set to: {fps}")
    
    def get_fps_stats(self) -> FPSStats:
        """Get current FPS statistics."""
        return self._fps_monitor.stats
    
    def get_fps_scale(self) -> float:
        """Get current FPS-based resource scaling factor."""
        return self._last_fps_scale
    
    def add_game_profile(self, profile: GamingProfile, auto_save: bool = True):
        """Add or update a game profile."""
        key = profile.name.lower().replace(" ", "_")
        self.profiles[key] = profile
        
        for proc in profile.process_names:
            self._process_to_profile[proc.lower()] = profile
        
        if auto_save:
            self.save_profiles()
    
    def remove_game_profile(self, name: str, auto_save: bool = True) -> bool:
        """Remove a custom game profile by name."""
        key = name.lower().replace(" ", "_")
        
        # Don't remove default profiles
        if key in DEFAULT_GAMING_PROFILES:
            logger.warning(f"Cannot remove default profile: {name}")
            return False
        
        if key not in self.profiles:
            return False
        
        profile = self.profiles[key]
        
        # Remove from process lookup
        for proc in profile.process_names:
            proc_key = proc.lower()
            if proc_key in self._process_to_profile:
                del self._process_to_profile[proc_key]
        
        # Remove profile
        del self.profiles[key]
        
        if auto_save:
            self.save_profiles()
        
        return True
    
    def get_custom_profiles(self) -> dict[str, GamingProfile]:
        """Get only user-defined custom profiles (excluding defaults)."""
        return {
            k: v for k, v in self.profiles.items()
            if k not in DEFAULT_GAMING_PROFILES
        }
    
    def get_all_profile_names(self) -> list[str]:
        """Get list of all profile names."""
        return list(self.profiles.keys())
    
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
    'FPSStats',
    'FPSMonitor',
    'get_gaming_mode',
]
