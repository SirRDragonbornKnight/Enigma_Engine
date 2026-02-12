"""
================================================================================
Audio Ducking - Lower other audio when AI speaks.
================================================================================

Automatically reduces system/application audio volume when TTS is playing,
then restores it when speech ends. Prevents AI speech from competing with
other audio sources.

Multi-platform support:
1. PulseAudio (Linux) - via pulsectl
2. ALSA (Linux fallback) - via amixer
3. Windows - via pycaw
4. macOS - via osascript

USAGE:
    from enigma_engine.voice.audio_ducking import AudioDucker
    
    ducker = AudioDucker()
    
    # Duck audio during speech
    with ducker.duck():
        play_tts_audio()
    
    # Or manually control
    ducker.duck_start()
    play_tts_audio()
    ducker.duck_end()
"""

from __future__ import annotations

import logging
import platform
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto

logger = logging.getLogger(__name__)


class AudioBackend(Enum):
    """Available audio control backends."""
    PULSEAUDIO = auto()   # Linux with PulseAudio
    ALSA = auto()         # Linux with ALSA
    WINDOWS = auto()      # Windows via pycaw
    MACOS = auto()        # macOS via osascript
    NONE = auto()         # No audio control available


@dataclass
class AudioDuckingConfig:
    """Configuration for audio ducking."""
    
    # Ducking level (0.0 = mute, 1.0 = no change)
    duck_level: float = 0.3
    
    # Fade timing
    fade_in_ms: float = 100.0    # Time to duck down
    fade_out_ms: float = 300.0   # Time to restore
    
    # Which streams to duck
    duck_music: bool = True
    duck_video: bool = True
    duck_games: bool = False     # Often want game audio during AI
    duck_system: bool = False    # System sounds
    
    # Exclusions (app names to never duck)
    excluded_apps: list = None
    
    def __post_init__(self):
        if self.excluded_apps is None:
            self.excluded_apps = []


class AudioDucker:
    """
    Cross-platform audio ducking for AI speech.
    
    Automatically lowers other audio when the AI is speaking,
    making the voice more audible without user intervention.
    """
    
    def __init__(self, config: AudioDuckingConfig = None):
        self.config = config or AudioDuckingConfig()
        
        # State
        self._is_ducked = False
        self._original_volumes: dict = {}
        self._lock = threading.Lock()
        
        # Detect platform and backend
        self._backend = self._detect_backend()
        
        # Backend-specific state
        self._pulse = None
        self._windows_sessions = None
        
        logger.info(f"AudioDucker initialized with {self._backend.name} backend")
    
    def _detect_backend(self) -> AudioBackend:
        """Detect the best available audio backend."""
        system = platform.system()
        
        if system == "Linux":
            # Try PulseAudio first
            try:
                import pulsectl  # type: ignore
                return AudioBackend.PULSEAUDIO
            except ImportError:
                pass  # Intentionally silent
            
            # Check for ALSA
            try:
                result = subprocess.run(
                    ["amixer", "--version"],
                    capture_output=True,
                    timeout=2
                )
                if result.returncode == 0:
                    return AudioBackend.ALSA
            except (subprocess.SubprocessError, FileNotFoundError):
                pass  # Intentionally silent
        
        elif system == "Windows":
            try:
                from pycaw.pycaw import AudioUtilities  # type: ignore
                return AudioBackend.WINDOWS
            except ImportError:
                pass  # Intentionally silent
        
        elif system == "Darwin":  # macOS
            return AudioBackend.MACOS
        
        return AudioBackend.NONE
    
    @contextmanager
    def duck(self):
        """
        Context manager for audio ducking.
        
        Usage:
            with ducker.duck():
                play_speech()
        """
        self.duck_start()
        try:
            yield
        finally:
            self.duck_end()
    
    def duck_start(self):
        """Start ducking - lower other audio."""
        with self._lock:
            if self._is_ducked:
                return
            
            self._is_ducked = True
            
            if self._backend == AudioBackend.PULSEAUDIO:
                self._duck_pulseaudio()
            elif self._backend == AudioBackend.ALSA:
                self._duck_alsa()
            elif self._backend == AudioBackend.WINDOWS:
                self._duck_windows()
            elif self._backend == AudioBackend.MACOS:
                self._duck_macos()
            else:
                logger.debug("No audio backend available for ducking")
    
    def duck_end(self):
        """End ducking - restore audio levels."""
        with self._lock:
            if not self._is_ducked:
                return
            
            self._is_ducked = False
            
            if self._backend == AudioBackend.PULSEAUDIO:
                self._restore_pulseaudio()
            elif self._backend == AudioBackend.ALSA:
                self._restore_alsa()
            elif self._backend == AudioBackend.WINDOWS:
                self._restore_windows()
            elif self._backend == AudioBackend.MACOS:
                self._restore_macos()
            
            self._original_volumes.clear()
    
    def _should_duck_app(self, app_name: str) -> bool:
        """Check if an application should be ducked."""
        app_lower = app_name.lower()
        
        # Check exclusions
        for excluded in self.config.excluded_apps:
            if excluded.lower() in app_lower:
                return False
        
        # Categorize by app name patterns
        music_apps = ['spotify', 'music', 'rhythmbox', 'vlc', 'audacious', 'clementine', 'itunes']
        video_apps = ['youtube', 'netflix', 'vlc', 'mpv', 'totem', 'video']
        game_apps = ['steam', 'game', 'minecraft', 'wine']
        
        if any(m in app_lower for m in music_apps):
            return self.config.duck_music
        if any(v in app_lower for v in video_apps):
            return self.config.duck_video
        if any(g in app_lower for g in game_apps):
            return self.config.duck_games
        
        # Default: duck it
        return True
    
    # =========================================================================
    # PulseAudio Backend
    # =========================================================================
    
    def _duck_pulseaudio(self):
        """Duck audio using PulseAudio."""
        try:
            import pulsectl
            
            with pulsectl.Pulse('forge-ai-ducker') as pulse:
                # Get all sink inputs (playing streams)
                for sink_input in pulse.sink_input_list():
                    app_name = sink_input.proplist.get('application.name', 'unknown')
                    
                    if not self._should_duck_app(app_name):
                        continue
                    
                    # Store original volume
                    original = sink_input.volume.value_flat
                    self._original_volumes[sink_input.index] = original
                    
                    # Calculate ducked volume
                    ducked = original * self.config.duck_level
                    
                    # Apply ducking with fade
                    self._fade_volume_pulse(pulse, sink_input.index, original, ducked)
                    
                    logger.debug(f"Ducked {app_name}: {original:.2f} -> {ducked:.2f}")
                    
        except Exception as e:
            logger.error(f"PulseAudio ducking failed: {e}")
    
    def _restore_pulseaudio(self):
        """Restore audio using PulseAudio."""
        try:
            import pulsectl
            
            with pulsectl.Pulse('forge-ai-ducker') as pulse:
                for sink_input in pulse.sink_input_list():
                    if sink_input.index in self._original_volumes:
                        original = self._original_volumes[sink_input.index]
                        current = sink_input.volume.value_flat
                        
                        self._fade_volume_pulse(pulse, sink_input.index, current, original)
                        
        except Exception as e:
            logger.error(f"PulseAudio restore failed: {e}")
    
    def _fade_volume_pulse(self, pulse, index: int, from_vol: float, to_vol: float):
        """Fade volume over time using PulseAudio."""
        import pulsectl
        
        steps = 10
        duration = self.config.fade_in_ms / 1000 if to_vol < from_vol else self.config.fade_out_ms / 1000
        step_time = duration / steps
        
        for i in range(steps + 1):
            t = i / steps
            vol = from_vol + (to_vol - from_vol) * t
            
            try:
                # Get current sink input
                sink_input = pulse.sink_input_info(index)
                # Set volume
                pulse.volume_set_all_chans(sink_input, vol)
            except pulsectl.PulseIndexError:
                break  # Stream ended
            
            if i < steps:
                time.sleep(step_time)
    
    # =========================================================================
    # ALSA Backend
    # =========================================================================
    
    def _duck_alsa(self):
        """Duck audio using ALSA amixer."""
        try:
            # Get current master volume
            result = subprocess.run(
                ["amixer", "get", "Master"],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            # Parse volume (e.g., "[75%]")
            import re
            match = re.search(r'\[(\d+)%\]', result.stdout)
            if match:
                original = int(match.group(1))
                self._original_volumes['master'] = original
                
                ducked = int(original * self.config.duck_level)
                
                subprocess.run(
                    ["amixer", "set", "Master", f"{ducked}%"],
                    capture_output=True,
                    timeout=2
                )
                
                logger.debug(f"ALSA ducked: {original}% -> {ducked}%")
                
        except Exception as e:
            logger.error(f"ALSA ducking failed: {e}")
    
    def _restore_alsa(self):
        """Restore audio using ALSA amixer."""
        try:
            if 'master' in self._original_volumes:
                original = self._original_volumes['master']
                
                subprocess.run(
                    ["amixer", "set", "Master", f"{original}%"],
                    capture_output=True,
                    timeout=2
                )
                
        except Exception as e:
            logger.error(f"ALSA restore failed: {e}")
    
    # =========================================================================
    # Windows Backend
    # =========================================================================
    
    def _duck_windows(self):
        """Duck audio using Windows pycaw."""
        try:
            from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
            
            sessions = AudioUtilities.GetAllSessions()
            
            for session in sessions:
                if session.Process:
                    app_name = session.Process.name()
                    
                    if not self._should_duck_app(app_name):
                        continue
                    
                    volume = session._ctl.QueryInterface(ISimpleAudioVolume)
                    original = volume.GetMasterVolume()
                    
                    self._original_volumes[app_name] = original
                    
                    ducked = original * self.config.duck_level
                    volume.SetMasterVolume(ducked, None)
                    
                    logger.debug(f"Ducked {app_name}: {original:.2f} -> {ducked:.2f}")
                    
        except Exception as e:
            logger.error(f"Windows ducking failed: {e}")
    
    def _restore_windows(self):
        """Restore audio using Windows pycaw."""
        try:
            from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
            
            sessions = AudioUtilities.GetAllSessions()
            
            for session in sessions:
                if session.Process:
                    app_name = session.Process.name()
                    
                    if app_name in self._original_volumes:
                        volume = session._ctl.QueryInterface(ISimpleAudioVolume)
                        volume.SetMasterVolume(self._original_volumes[app_name], None)
                        
        except Exception as e:
            logger.error(f"Windows restore failed: {e}")
    
    # =========================================================================
    # macOS Backend
    # =========================================================================
    
    def _duck_macos(self):
        """Duck audio using macOS osascript."""
        try:
            # Get current volume
            result = subprocess.run(
                ["osascript", "-e", "output volume of (get volume settings)"],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            original = int(result.stdout.strip())
            self._original_volumes['system'] = original
            
            ducked = int(original * self.config.duck_level)
            
            subprocess.run(
                ["osascript", "-e", f"set volume output volume {ducked}"],
                capture_output=True,
                timeout=2
            )
            
            logger.debug(f"macOS ducked: {original} -> {ducked}")
            
        except Exception as e:
            logger.error(f"macOS ducking failed: {e}")
    
    def _restore_macos(self):
        """Restore audio using macOS osascript."""
        try:
            if 'system' in self._original_volumes:
                original = self._original_volumes['system']
                
                subprocess.run(
                    ["osascript", "-e", f"set volume output volume {original}"],
                    capture_output=True,
                    timeout=2
                )
                
        except Exception as e:
            logger.error(f"macOS restore failed: {e}")
    
    @property
    def is_ducked(self) -> bool:
        """Check if audio is currently ducked."""
        return self._is_ducked
    
    @property
    def backend_name(self) -> str:
        """Get the name of the active backend."""
        return self._backend.name


# Singleton instance for easy use
_ducker: AudioDucker | None = None


def get_ducker(config: AudioDuckingConfig = None) -> AudioDucker:
    """Get or create the global audio ducker instance."""
    global _ducker
    if _ducker is None:
        _ducker = AudioDucker(config)
    return _ducker


@contextmanager
def duck_audio(config: AudioDuckingConfig = None):
    """
    Convenience context manager for audio ducking.
    
    Usage:
        with duck_audio():
            play_speech()
    """
    ducker = get_ducker(config)
    with ducker.duck():
        yield
