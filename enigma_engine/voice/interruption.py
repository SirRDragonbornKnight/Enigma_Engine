"""
================================================================================
Interruption Handling - Stop TTS when user starts speaking.
================================================================================

Provides barge-in detection to allow natural conversation flow:
- Monitors audio input during TTS playback
- Detects when user starts speaking
- Immediately stops TTS output
- Captures user's speech for processing

USAGE:
    from enigma_engine.voice.interruption import InterruptionHandler
    
    handler = InterruptionHandler()
    handler.on_interrupt(lambda: print("User interrupted!"))
    
    # Start monitoring during TTS
    handler.start_monitoring()
    
    # ... TTS plays ...
    
    # Check if interrupted
    if handler.was_interrupted():
        captured_audio = handler.get_captured_audio()
    
    handler.stop_monitoring()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


class InterruptionMode(Enum):
    """How to handle interruptions."""
    
    IMMEDIATE = auto()      # Stop TTS immediately on any voice
    CONFIRMED = auto()      # Wait for sustained speech before stopping
    WORD_BOUNDARY = auto()  # Stop at next word boundary
    SENTENCE_END = auto()   # Stop at next sentence end
    DISABLED = auto()       # Ignore interruptions


class InterruptionSensitivity(Enum):
    """Sensitivity levels for detecting interruptions."""
    
    LOW = auto()       # Only loud, sustained speech
    MEDIUM = auto()    # Normal conversation level
    HIGH = auto()       # Detect quiet speech too
    VERY_HIGH = auto()  # Very sensitive (may false trigger)


@dataclass
class InterruptionConfig:
    """Configuration for interruption handling."""
    
    # Detection mode
    mode: InterruptionMode = InterruptionMode.CONFIRMED
    sensitivity: InterruptionSensitivity = InterruptionSensitivity.MEDIUM
    
    # Timing thresholds
    min_speech_duration_ms: float = 150.0    # Min speech duration to trigger
    confirmation_window_ms: float = 300.0    # Time to confirm it's speech
    debounce_ms: float = 500.0               # Cooldown after interrupt
    
    # Energy thresholds (auto-calibrated if 0)
    energy_threshold: float = 0.0            # Min RMS energy (0 = auto)
    energy_ratio: float = 2.0                # Ratio above ambient noise
    
    # Audio capture
    capture_audio: bool = True               # Save interrupted audio
    max_capture_seconds: float = 5.0         # Max audio to capture
    
    # Sample rate
    sample_rate: int = 16000


# Sensitivity to threshold multipliers
SENSITIVITY_THRESHOLDS = {
    InterruptionSensitivity.LOW: 3.0,
    InterruptionSensitivity.MEDIUM: 2.0,
    InterruptionSensitivity.HIGH: 1.5,
    InterruptionSensitivity.VERY_HIGH: 1.2,
}


@dataclass
class InterruptionEvent:
    """Details of an interruption event."""
    
    timestamp: float                          # When interruption was detected
    audio: np.ndarray | None = None        # Captured audio (if enabled)
    duration_ms: float = 0.0                  # Duration of speech so far
    energy: float = 0.0                       # RMS energy of interruption
    confirmed: bool = False                   # Whether speech was confirmed


class InterruptionHandler:
    """
    Detects and handles user interruptions during TTS playback.
    
    Uses voice activity detection to identify when the user starts
    speaking and triggers callbacks to stop TTS output.
    """
    
    def __init__(self, config: InterruptionConfig = None):
        self.config = config or InterruptionConfig()
        
        # State
        self._monitoring = False
        self._interrupted = False
        self._last_interrupt_time = 0.0
        
        # Audio capture
        self._captured_audio: list[np.ndarray] = []
        self._capture_lock = threading.Lock()
        
        # Ambient noise level (for auto-threshold)
        self._ambient_energy = 0.0
        self._energy_samples: list[float] = []
        
        # Speech detection state
        self._speech_start_time = 0.0
        self._speech_energy_sum = 0.0
        self._speech_sample_count = 0
        
        # Callbacks
        self._interrupt_callbacks: list[Callable[[], None]] = []
        self._speech_start_callbacks: list[Callable[[], None]] = []
        
        # TTS control reference (set externally)
        self._tts_stop_func: Callable[[], None] | None = None
        
        # Thread safety
        self._lock = threading.Lock()
        
        # VAD integration (optional)
        self._vad = None
        self._init_vad()
    
    def _init_vad(self):
        """Initialize VAD if available."""
        try:
            from .vad import get_vad
            self._vad = get_vad()
            logger.debug("Using VAD for interruption detection")
        except ImportError:
            logger.debug("VAD not available, using energy-based detection")
    
    def start_monitoring(self):
        """Start monitoring for interruptions."""
        with self._lock:
            self._monitoring = True
            self._interrupted = False
            self._captured_audio.clear()
            self._speech_start_time = 0.0
            self._speech_energy_sum = 0.0
            self._speech_sample_count = 0
        
        logger.debug("Interruption monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring for interruptions."""
        with self._lock:
            self._monitoring = False
        
        logger.debug("Interruption monitoring stopped")
    
    def is_monitoring(self) -> bool:
        """Check if currently monitoring."""
        with self._lock:
            return self._monitoring
    
    def was_interrupted(self) -> bool:
        """Check if an interruption occurred."""
        with self._lock:
            return self._interrupted
    
    def reset(self):
        """Reset interruption state."""
        with self._lock:
            self._interrupted = False
            self._captured_audio.clear()
    
    def set_tts_stop_function(self, func: Callable[[], None]):
        """Set the function to call to stop TTS."""
        self._tts_stop_func = func
    
    def on_interrupt(self, callback: Callable[[], None]):
        """Register callback for when user interrupts."""
        self._interrupt_callbacks.append(callback)
    
    def on_speech_start(self, callback: Callable[[], None]):
        """Register callback for when user speech is first detected."""
        self._speech_start_callbacks.append(callback)
    
    def process_audio(self, audio: np.ndarray) -> bool:
        """
        Process audio chunk to detect interruptions.
        
        Call this with incoming mic audio while TTS is playing.
        
        Args:
            audio: Audio samples (float32, mono)
        
        Returns:
            True if interruption detected
        """
        with self._lock:
            if not self._monitoring:
                return False
            
            if self.config.mode == InterruptionMode.DISABLED:
                return False
            
            # Check debounce
            now = time.time()
            if now - self._last_interrupt_time < self.config.debounce_ms / 1000.0:
                return False
        
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Calculate energy
        energy = self._calculate_energy(audio)
        
        # Update ambient noise estimate
        self._update_ambient_estimate(energy)
        
        # Check for voice activity
        is_speech = self._detect_speech(audio, energy)
        
        if is_speech:
            return self._handle_speech_detected(audio, energy)
        else:
            # Reset speech detection state if no speech
            with self._lock:
                self._speech_start_time = 0.0
                self._speech_energy_sum = 0.0
                self._speech_sample_count = 0
            return False
    
    def _calculate_energy(self, audio: np.ndarray) -> float:
        """Calculate RMS energy of audio."""
        return float(np.sqrt(np.mean(audio ** 2)))
    
    def _update_ambient_estimate(self, energy: float):
        """Update ambient noise level estimate."""
        # Keep recent energy samples
        self._energy_samples.append(energy)
        if len(self._energy_samples) > 50:
            self._energy_samples.pop(0)
        
        # Use lower percentile as ambient estimate
        if len(self._energy_samples) >= 10:
            sorted_energies = sorted(self._energy_samples)
            # 20th percentile
            idx = len(sorted_energies) // 5
            self._ambient_energy = sorted_energies[idx]
    
    def _detect_speech(self, audio: np.ndarray, energy: float) -> bool:
        """Detect if audio contains speech."""
        # Try VAD first
        if self._vad:
            try:
                return self._vad.is_speech(audio, self.config.sample_rate)
            except Exception:
                pass
        
        # Fall back to energy-based detection
        threshold = self._get_energy_threshold()
        return energy > threshold
    
    def _get_energy_threshold(self) -> float:
        """Get the energy threshold for speech detection."""
        # Use configured threshold if set
        if self.config.energy_threshold > 0:
            return self.config.energy_threshold
        
        # Auto-threshold based on ambient + sensitivity
        sensitivity_mult = SENSITIVITY_THRESHOLDS.get(
            self.config.sensitivity,
            2.0
        )
        
        # Minimum threshold to avoid noise triggering
        min_threshold = 0.01
        
        ambient_threshold = self._ambient_energy * self.config.energy_ratio * sensitivity_mult
        return max(min_threshold, ambient_threshold)
    
    def _handle_speech_detected(self, audio: np.ndarray, energy: float) -> bool:
        """Handle when speech is detected."""
        now = time.time()
        
        with self._lock:
            # First detection - mark start time
            if self._speech_start_time == 0.0:
                self._speech_start_time = now
                self._speech_energy_sum = energy
                self._speech_sample_count = 1
                
                # Notify speech start
                for callback in self._speech_start_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"Speech start callback error: {e}")
                
                # Capture audio if enabled
                if self.config.capture_audio:
                    self._captured_audio.append(audio.copy())
                
                # Immediate mode - interrupt right away
                if self.config.mode == InterruptionMode.IMMEDIATE:
                    return self._trigger_interrupt(energy)
                
                return False
            
            # Accumulate speech stats
            self._speech_energy_sum += energy
            self._speech_sample_count += 1
            
            # Capture more audio
            if self.config.capture_audio:
                total_samples = sum(len(a) for a in self._captured_audio)
                max_samples = int(self.config.max_capture_seconds * self.config.sample_rate)
                if total_samples < max_samples:
                    self._captured_audio.append(audio.copy())
            
            # Check if we've had enough sustained speech
            speech_duration_ms = (now - self._speech_start_time) * 1000.0
            
            if self.config.mode == InterruptionMode.CONFIRMED:
                if speech_duration_ms >= self.config.confirmation_window_ms:
                    return self._trigger_interrupt(energy)
            
            elif self.config.mode == InterruptionMode.WORD_BOUNDARY:
                # For word boundary, we need to detect pauses
                # For now, treat like confirmed
                if speech_duration_ms >= self.config.confirmation_window_ms:
                    return self._trigger_interrupt(energy)
            
            elif self.config.mode == InterruptionMode.SENTENCE_END:
                # For sentence end, wait longer
                if speech_duration_ms >= self.config.confirmation_window_ms * 2:
                    return self._trigger_interrupt(energy)
        
        return False
    
    def _trigger_interrupt(self, energy: float) -> bool:
        """Trigger an interruption."""
        with self._lock:
            if self._interrupted:
                return True  # Already interrupted
            
            self._interrupted = True
            self._last_interrupt_time = time.time()
            
            duration_ms = (time.time() - self._speech_start_time) * 1000.0
            
            logger.info(f"Interruption triggered (energy={energy:.4f}, duration={duration_ms:.0f}ms)")
        
        # Stop TTS
        if self._tts_stop_func:
            try:
                self._tts_stop_func()
            except Exception as e:
                logger.error(f"Failed to stop TTS: {e}")
        
        # Notify callbacks
        for callback in self._interrupt_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Interrupt callback error: {e}")
        
        return True
    
    def get_captured_audio(self) -> np.ndarray | None:
        """Get audio captured during interruption."""
        with self._capture_lock:
            if not self._captured_audio:
                return None
            return np.concatenate(self._captured_audio)
    
    def get_last_event(self) -> InterruptionEvent | None:
        """Get details of the last interruption."""
        with self._lock:
            if not self._interrupted:
                return None
            
            audio = self.get_captured_audio()
            avg_energy = (
                self._speech_energy_sum / self._speech_sample_count
                if self._speech_sample_count > 0 else 0.0
            )
            duration_ms = (
                (self._last_interrupt_time - self._speech_start_time) * 1000.0
                if self._speech_start_time > 0 else 0.0
            )
            
            return InterruptionEvent(
                timestamp=self._last_interrupt_time,
                audio=audio,
                duration_ms=duration_ms,
                energy=avg_energy,
                confirmed=True
            )
    
    def calibrate(self, audio: np.ndarray, duration_seconds: float = 2.0):
        """
        Calibrate ambient noise level from audio sample.
        
        Args:
            audio: Audio samples of ambient noise
            duration_seconds: Expected duration of sample
        """
        if len(audio) == 0:
            return
        
        # Calculate energy of the sample
        energy = self._calculate_energy(audio)
        self._ambient_energy = energy
        
        logger.info(f"Calibrated ambient energy: {energy:.4f}")


# Global handler instance
_handler: InterruptionHandler | None = None


def get_interruption_handler(config: InterruptionConfig = None) -> InterruptionHandler:
    """Get or create the global interruption handler."""
    global _handler
    if _handler is None:
        _handler = InterruptionHandler(config)
    return _handler


def start_barge_in_detection(tts_stop_func: Callable[[], None] = None) -> InterruptionHandler:
    """
    Start listening for user barge-in (interruption).
    
    Args:
        tts_stop_func: Function to call to stop TTS playback
    
    Returns:
        The interruption handler
    """
    handler = get_interruption_handler()
    if tts_stop_func:
        handler.set_tts_stop_function(tts_stop_func)
    handler.start_monitoring()
    return handler


def stop_barge_in_detection():
    """Stop listening for user barge-in."""
    if _handler:
        _handler.stop_monitoring()


def was_interrupted() -> bool:
    """Check if user interrupted."""
    if _handler:
        return _handler.was_interrupted()
    return False


def get_interrupted_audio() -> np.ndarray | None:
    """Get audio captured during interruption."""
    if _handler:
        return _handler.get_captured_audio()
    return None
