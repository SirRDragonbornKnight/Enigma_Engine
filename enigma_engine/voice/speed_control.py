"""
================================================================================
Voice Speed Control - Adjustable TTS playback speed.
================================================================================

Provides comprehensive speech rate control:
- Runtime speed adjustment (0.5x to 3.0x)
- Named presets (slow, normal, fast, etc.)
- Time-stretching for pre-recorded audio
- Per-engine rate mapping
- Pitch preservation option

USAGE:
    from enigma_engine.voice.speed_control import SpeedController, SpeedPreset
    
    controller = SpeedController()
    
    # Set speed by preset
    controller.set_preset(SpeedPreset.SLOW)
    
    # Set exact multiplier
    controller.set_speed(1.5)  # 1.5x faster
    
    # Apply to TTS engine
    controller.apply_to_engine(tts_engine)
    
    # Time-stretch audio
    stretched = controller.stretch_audio(audio, target_speed=0.75)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class SpeedPreset(Enum):
    """Named speed presets."""
    
    VERY_SLOW = auto()    # 0.5x - For careful listening
    SLOW = auto()         # 0.75x - Relaxed pace
    NORMAL = auto()       # 1.0x - Default
    FAST = auto()         # 1.25x - Quick but clear
    VERY_FAST = auto()    # 1.5x - Rapid delivery
    MAXIMUM = auto()      # 2.0x - Maximum speed
    
    # Context-specific presets
    AUDIOBOOK = auto()    # 0.9x - Comfortable for long listening
    LECTURE = auto()      # 0.85x - Educational content
    NEWS = auto()         # 1.1x - Professional news pace
    CASUAL = auto()       # 1.0x - Conversational
    URGENT = auto()       # 1.3x - Time-sensitive
    ACCESSIBILITY = auto() # 0.8x - For hearing impaired


# Preset to speed multiplier mapping
PRESET_SPEEDS: dict[SpeedPreset, float] = {
    SpeedPreset.VERY_SLOW: 0.5,
    SpeedPreset.SLOW: 0.75,
    SpeedPreset.NORMAL: 1.0,
    SpeedPreset.FAST: 1.25,
    SpeedPreset.VERY_FAST: 1.5,
    SpeedPreset.MAXIMUM: 2.0,
    SpeedPreset.AUDIOBOOK: 0.9,
    SpeedPreset.LECTURE: 0.85,
    SpeedPreset.NEWS: 1.1,
    SpeedPreset.CASUAL: 1.0,
    SpeedPreset.URGENT: 1.3,
    SpeedPreset.ACCESSIBILITY: 0.8,
}


@dataclass
class SpeedConfig:
    """Configuration for speed control."""
    
    # Speed limits
    min_speed: float = 0.25       # Minimum speed multiplier
    max_speed: float = 3.0        # Maximum speed multiplier
    
    # Default settings
    default_speed: float = 1.0    # Default speed multiplier
    default_preset: SpeedPreset = SpeedPreset.NORMAL
    
    # Pitch preservation
    preserve_pitch: bool = True   # Keep pitch when changing speed
    
    # Engine-specific rate bases (words per minute)
    pyttsx3_base_rate: int = 150
    espeak_base_rate: int = 175
    
    # Smooth transitions
    transition_steps: int = 10    # Steps for gradual speed change
    transition_time_ms: float = 200.0  # Time for transition


class SpeedController:
    """
    Control TTS playback speed with presets and fine-tuning.
    
    Works with multiple TTS engines and provides audio time-stretching.
    """
    
    def __init__(self, config: SpeedConfig = None):
        self.config = config or SpeedConfig()
        
        # Current speed state
        self._speed: float = self.config.default_speed
        self._preset: SpeedPreset = self.config.default_preset
        
        # Callbacks for speed changes
        self._on_change_callbacks: list = []
        
        # Cached engine reference
        self._engine = None
        self._engine_type: str = "unknown"
    
    @property
    def speed(self) -> float:
        """Get current speed multiplier."""
        return self._speed
    
    @property
    def preset(self) -> SpeedPreset:
        """Get current preset (may not match if speed was set directly)."""
        return self._preset
    
    def set_speed(self, multiplier: float, apply: bool = True) -> float:
        """
        Set speed by multiplier.
        
        Args:
            multiplier: Speed multiplier (1.0 = normal)
            apply: Whether to apply to engine immediately
        
        Returns:
            Actual speed set (clamped to limits)
        """
        # Clamp to limits
        multiplier = max(self.config.min_speed, min(self.config.max_speed, multiplier))
        
        old_speed = self._speed
        self._speed = multiplier
        
        # Find matching preset (or closest)
        self._preset = self._find_preset(multiplier)
        
        logger.debug(f"Speed set: {old_speed:.2f}x -> {multiplier:.2f}x")
        
        # Apply to engine
        if apply and self._engine:
            self._apply_to_engine_internal()
        
        # Notify callbacks
        for callback in self._on_change_callbacks:
            try:
                callback(multiplier, old_speed)
            except Exception as e:
                logger.error(f"Speed change callback error: {e}")
        
        return multiplier
    
    def set_preset(self, preset: SpeedPreset, apply: bool = True) -> float:
        """
        Set speed by preset.
        
        Args:
            preset: Speed preset
            apply: Whether to apply to engine immediately
        
        Returns:
            Speed multiplier for the preset
        """
        self._preset = preset
        speed = PRESET_SPEEDS.get(preset, 1.0)
        return self.set_speed(speed, apply)
    
    def adjust(self, delta: float, apply: bool = True) -> float:
        """
        Adjust speed relative to current.
        
        Args:
            delta: Change in speed (e.g., 0.1 for 10% faster)
            apply: Whether to apply to engine immediately
        
        Returns:
            New speed multiplier
        """
        return self.set_speed(self._speed + delta, apply)
    
    def faster(self, amount: float = 0.25) -> float:
        """Increase speed."""
        return self.adjust(amount)
    
    def slower(self, amount: float = 0.25) -> float:
        """Decrease speed."""
        return self.adjust(-amount)
    
    def reset(self, apply: bool = True) -> float:
        """Reset to default speed."""
        return self.set_speed(self.config.default_speed, apply)
    
    def _find_preset(self, speed: float) -> SpeedPreset:
        """Find the closest preset for a speed."""
        closest = SpeedPreset.NORMAL
        min_diff = float('inf')
        
        for preset, preset_speed in PRESET_SPEEDS.items():
            diff = abs(preset_speed - speed)
            if diff < min_diff:
                min_diff = diff
                closest = preset
        
        return closest
    
    def apply_to_engine(self, engine: Any) -> bool:
        """
        Apply current speed settings to a TTS engine.
        
        Args:
            engine: TTS engine instance
        
        Returns:
            True if successfully applied
        """
        self._engine = engine
        self._engine_type = self._detect_engine_type(engine)
        return self._apply_to_engine_internal()
    
    def _detect_engine_type(self, engine: Any) -> str:
        """Detect the type of TTS engine."""
        if engine is None:
            return "none"
        
        engine_str = str(type(engine)).lower()
        
        if "pyttsx3" in engine_str:
            return "pyttsx3"
        elif "espeak" in engine_str:
            return "espeak"
        elif "coqui" in engine_str or "tts" in engine_str:
            return "coqui"
        
        # Check for common methods
        if hasattr(engine, 'setProperty'):
            return "pyttsx3"
        
        return "generic"
    
    def _apply_to_engine_internal(self) -> bool:
        """Apply speed to the cached engine."""
        if not self._engine:
            return False
        
        try:
            if self._engine_type == "pyttsx3":
                return self._apply_pyttsx3()
            elif self._engine_type == "espeak":
                return self._apply_espeak()
            else:
                return self._apply_generic()
                
        except Exception as e:
            logger.error(f"Failed to apply speed to engine: {e}")
            return False
    
    def _apply_pyttsx3(self) -> bool:
        """Apply speed to pyttsx3 engine."""
        # pyttsx3 uses words per minute
        base_rate = self.config.pyttsx3_base_rate
        new_rate = int(base_rate * self._speed)
        
        self._engine.setProperty('rate', new_rate)
        logger.debug(f"pyttsx3 rate set to {new_rate} WPM")
        return True
    
    def _apply_espeak(self) -> bool:
        """Apply speed to espeak engine."""
        # espeak uses words per minute
        base_rate = self.config.espeak_base_rate
        new_rate = int(base_rate * self._speed)
        
        if hasattr(self._engine, 'setProperty'):
            self._engine.setProperty('rate', new_rate)
        return True
    
    def _apply_generic(self) -> bool:
        """Apply speed to generic engine."""
        if hasattr(self._engine, 'setProperty'):
            # Assume WPM-based rate
            new_rate = int(150 * self._speed)
            self._engine.setProperty('rate', new_rate)
            return True
        
        if hasattr(self._engine, 'rate'):
            self._engine.rate = self._speed
            return True
        
        return False
    
    def stretch_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        target_speed: float = None
    ) -> np.ndarray:
        """
        Time-stretch audio to match target speed.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate in Hz
            target_speed: Target speed (uses current speed if None)
        
        Returns:
            Time-stretched audio
        """
        speed = target_speed if target_speed is not None else self._speed
        
        if abs(speed - 1.0) < 0.01:
            return audio  # No change needed
        
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        if self.config.preserve_pitch:
            return self._stretch_preserve_pitch(audio, sample_rate, speed)
        else:
            return self._stretch_simple(audio, speed)
    
    def _stretch_simple(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """
        Simple time-stretching by resampling (changes pitch).
        
        This is fast but pitch will change proportionally to speed.
        """
        original_len = len(audio)
        new_len = int(original_len / speed)
        
        # Linear interpolation
        indices = np.linspace(0, original_len - 1, new_len)
        stretched = np.interp(indices, np.arange(original_len), audio)
        
        return stretched.astype(np.float32)
    
    def _stretch_preserve_pitch(
        self,
        audio: np.ndarray,
        sample_rate: int,
        speed: float
    ) -> np.ndarray:
        """
        Time-stretch with pitch preservation using phase vocoder.
        
        More CPU-intensive but maintains natural pitch.
        """
        try:
            # Try librosa first (best quality)
            import librosa
            
            stretched = librosa.effects.time_stretch(audio, rate=speed)
            return stretched.astype(np.float32)
            
        except ImportError:
            pass
        
        try:
            # Try scipy-based WSOLA
            return self._wsola_stretch(audio, sample_rate, speed)
        except Exception:
            pass
        
        # Fallback to simple stretch
        logger.debug("Pitch preservation unavailable, using simple stretch")
        return self._stretch_simple(audio, speed)
    
    def _wsola_stretch(
        self,
        audio: np.ndarray,
        sample_rate: int,
        speed: float
    ) -> np.ndarray:
        """
        WSOLA (Waveform Similarity Overlap-Add) time-stretching.
        
        A simpler pitch-preserving algorithm that works without librosa.
        """
        # Frame parameters
        frame_len = int(sample_rate * 0.025)  # 25ms frames
        hop_in = int(frame_len * 0.5)         # 50% overlap
        hop_out = int(hop_in / speed)
        
        # Hanning window
        window = np.hanning(frame_len)
        
        # Output length
        n_frames = int((len(audio) - frame_len) / hop_in) + 1
        out_len = int((n_frames - 1) * hop_out + frame_len)
        output = np.zeros(out_len, dtype=np.float32)
        
        # Overlap-add
        for i in range(n_frames):
            in_start = i * hop_in
            out_start = i * hop_out
            
            if in_start + frame_len > len(audio):
                break
            if out_start + frame_len > out_len:
                break
            
            frame = audio[in_start:in_start + frame_len] * window
            output[out_start:out_start + frame_len] += frame
        
        # Normalize
        max_val = np.abs(output).max()
        if max_val > 0:
            output = output / max_val * np.abs(audio).max()
        
        return output
    
    def on_change(self, callback: Callable[[float, float], None]):
        """
        Register callback for speed changes.
        
        Callback receives (new_speed, old_speed).
        """
        self._on_change_callbacks.append(callback)
    
    def get_rate_for_engine(self, engine_type: str = None) -> int:
        """
        Get the rate value for a specific engine type.
        
        Args:
            engine_type: Engine type (pyttsx3, espeak, etc.)
        
        Returns:
            Rate value (usually words per minute)
        """
        engine_type = engine_type or self._engine_type
        
        if engine_type == "pyttsx3":
            return int(self.config.pyttsx3_base_rate * self._speed)
        elif engine_type == "espeak":
            return int(self.config.espeak_base_rate * self._speed)
        else:
            return int(150 * self._speed)


# Global speed controller instance
_controller: SpeedController | None = None


def get_speed_controller(config: SpeedConfig = None) -> SpeedController:
    """Get or create the global speed controller."""
    global _controller
    if _controller is None:
        _controller = SpeedController(config)
    return _controller


def set_speed(multiplier: float) -> float:
    """
    Convenience function to set global speech speed.
    
    Args:
        multiplier: Speed multiplier (1.0 = normal)
    
    Returns:
        Actual speed set
    """
    return get_speed_controller().set_speed(multiplier)


def set_speed_preset(preset: SpeedPreset) -> float:
    """
    Convenience function to set speed by preset.
    
    Args:
        preset: Speed preset
    
    Returns:
        Speed multiplier for the preset
    """
    return get_speed_controller().set_preset(preset)


def get_speed() -> float:
    """Get current speech speed multiplier."""
    return get_speed_controller().speed


def faster(amount: float = 0.25) -> float:
    """Increase speech speed."""
    return get_speed_controller().faster(amount)


def slower(amount: float = 0.25) -> float:
    """Decrease speech speed."""
    return get_speed_controller().slower(amount)
