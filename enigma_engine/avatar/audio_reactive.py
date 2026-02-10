"""
Audio-Reactive Avatar for Enigma AI Engine

Makes avatars respond to audio input (music, voice, etc.).

Features:
- Beat detection for music
- Energy level tracking
- Frequency band analysis
- Emotion-driven movement
- Lip sync from voice
- Customizable reactions

Usage:
    from enigma_engine.avatar.audio_reactive import AudioReactiveAvatar
    
    reactive = AudioReactiveAvatar()
    
    # Start listening to audio
    reactive.start_listening()
    
    # Get current animation state
    state = reactive.get_avatar_state()
    # state.body_bounce, state.head_bob, state.mouth_open, etc.
    
    # Or connect to avatar controller
    reactive.connect_to_avatar(avatar_controller)
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    sd = None


class AudioMode(Enum):
    """Audio analysis modes."""
    MUSIC = auto()      # Beat detection, energy
    VOICE = auto()      # Lip sync, emotion
    AMBIENT = auto()    # Background awareness
    HYBRID = auto()     # Both music and voice


class FrequencyBand(Enum):
    """Frequency bands for analysis."""
    SUB_BASS = (20, 60)       # Deep bass
    BASS = (60, 250)          # Bass
    LOW_MID = (250, 500)      # Low mids
    MID = (500, 2000)         # Mids (voice fundamentals)
    HIGH_MID = (2000, 4000)   # High mids (consonants)
    HIGH = (4000, 8000)       # Highs
    PRESENCE = (8000, 16000)  # Air/Presence


@dataclass
class AudioState:
    """Current state of audio analysis."""
    # Overall energy (0-1)
    energy: float = 0.0
    
    # Beat detected this frame
    beat: bool = False
    
    # Time since last beat
    beat_interval: float = 0.0
    
    # Estimated BPM
    bpm: float = 0.0
    
    # Frequency band energies
    band_energies: Dict[str, float] = field(default_factory=dict)
    
    # Voice-specific
    is_speaking: bool = False
    mouth_open: float = 0.0  # 0-1
    pitch: float = 0.0       # Normalized pitch
    
    # Timestamp
    timestamp: float = 0.0


@dataclass
class AvatarReactionState:
    """State for avatar reactions to audio."""
    # Body movement
    body_bounce: float = 0.0    # Vertical bounce (0-1)
    body_sway: float = 0.0      # Side-to-side sway (-1 to 1)
    
    # Head movement
    head_bob: float = 0.0       # Head vertical movement
    head_tilt: float = 0.0      # Head side tilt
    
    # Face
    mouth_open: float = 0.0     # Lip sync
    eye_energy: float = 0.0     # Eye expressiveness
    blink_rate: float = 1.0     # Blink multiplier
    
    # Arms (if available)
    arm_raise: float = 0.0      # Arms up for energy
    
    # Overall intensity
    intensity: float = 0.0      # How active the avatar is


@dataclass
class ReactionConfig:
    """Configuration for audio reactions."""
    # Enable/disable different reactions
    enable_body: bool = True
    enable_head: bool = True
    enable_face: bool = True
    enable_arms: bool = True
    
    # Sensitivity
    energy_sensitivity: float = 1.0
    beat_sensitivity: float = 1.0
    voice_sensitivity: float = 1.0
    
    # Smoothing (0 = no smoothing, 1 = max smoothing)
    smoothing: float = 0.3
    
    # Mapping
    bass_to_bounce: float = 0.8    # How much bass affects bounce
    mids_to_sway: float = 0.5      # How much mids affect sway
    energy_to_arms: float = 0.6    # Energy to arm raise
    
    # Limits
    max_bounce: float = 1.0
    max_sway: float = 0.5


class AudioReactiveAvatar:
    """
    Makes avatars react to audio input in real-time.
    
    Analyzes incoming audio for beats, energy, voice, and
    generates avatar movement states.
    """
    
    def __init__(
        self,
        mode: AudioMode = AudioMode.HYBRID,
        sample_rate: int = 44100,
        buffer_size: int = 1024,
        config: Optional[ReactionConfig] = None
    ):
        """
        Initialize audio-reactive avatar.
        
        Args:
            mode: Audio analysis mode
            sample_rate: Audio sample rate
            buffer_size: FFT buffer size
            config: Reaction configuration
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy required for AudioReactiveAvatar")
        
        self._mode = mode
        self._sample_rate = sample_rate
        self._buffer_size = buffer_size
        self._config = config or ReactionConfig()
        
        # Audio state
        self._audio_state = AudioState()
        self._avatar_state = AvatarReactionState()
        
        # Analysis buffers
        self._audio_buffer = deque(maxlen=buffer_size * 4)
        self._energy_history = deque(maxlen=50)  # For beat detection
        self._beat_times = deque(maxlen=10)      # Recent beat timestamps
        
        # Smoothed values
        self._smooth_energy = 0.0
        self._smooth_bounce = 0.0
        self._smooth_sway = 0.0
        self._smooth_mouth = 0.0
        
        # Stream
        self._stream = None
        self._running = False
        self._callbacks: List[Callable[[AvatarReactionState], None]] = []
        
        # Thread
        self._analysis_thread = None
        self._lock = threading.Lock()
        
        logger.info(f"AudioReactiveAvatar initialized (mode={mode.name})")
    
    def start_listening(self, device: Optional[int] = None):
        """
        Start listening to audio input.
        
        Args:
            device: Audio input device index (None for default)
        """
        if not SOUNDDEVICE_AVAILABLE:
            logger.warning("sounddevice not available, using dummy mode")
            return
        
        if self._running:
            return
        
        self._running = True
        
        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.debug(f"Audio status: {status}")
            
            # Add to buffer
            self._audio_buffer.extend(indata[:, 0].tolist())
        
        try:
            self._stream = sd.InputStream(
                device=device,
                channels=1,
                samplerate=self._sample_rate,
                blocksize=self._buffer_size,
                callback=audio_callback
            )
            self._stream.start()
            
            # Start analysis thread
            self._analysis_thread = threading.Thread(
                target=self._analysis_loop,
                daemon=True
            )
            self._analysis_thread.start()
            
            logger.info("Audio listening started")
            
        except Exception as e:
            logger.error(f"Failed to start audio: {e}")
            self._running = False
    
    def stop_listening(self):
        """Stop listening to audio."""
        self._running = False
        
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        if self._analysis_thread:
            self._analysis_thread.join(timeout=1.0)
            self._analysis_thread = None
        
        logger.info("Audio listening stopped")
    
    def _analysis_loop(self):
        """Main analysis loop."""
        while self._running:
            try:
                if len(self._audio_buffer) >= self._buffer_size:
                    # Get buffer
                    samples = np.array(list(self._audio_buffer)[-self._buffer_size:])
                    
                    # Analyze
                    self._analyze_audio(samples)
                    
                    # Update avatar state
                    self._update_avatar_state()
                    
                    # Notify callbacks
                    for callback in self._callbacks:
                        try:
                            callback(self._avatar_state)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")
                
                time.sleep(1 / 60)  # ~60 FPS analysis
                
            except Exception as e:
                logger.error(f"Analysis error: {e}")
                time.sleep(0.1)
    
    def _analyze_audio(self, samples: np.ndarray):
        """Analyze audio samples."""
        with self._lock:
            current_time = time.time()
            
            # Calculate energy
            energy = np.sqrt(np.mean(samples ** 2))
            self._audio_state.energy = min(1.0, energy * 10)  # Normalize
            
            self._energy_history.append(energy)
            
            # Beat detection
            if len(self._energy_history) > 10:
                avg_energy = np.mean(list(self._energy_history)[:-1])
                threshold = avg_energy * (1.5 / self._config.beat_sensitivity)
                
                if energy > threshold and energy > 0.02:
                    # Check cooldown
                    if not self._beat_times or (current_time - self._beat_times[-1]) > 0.2:
                        self._audio_state.beat = True
                        self._beat_times.append(current_time)
                        
                        # Calculate BPM
                        if len(self._beat_times) >= 4:
                            intervals = np.diff(list(self._beat_times)[-4:])
                            avg_interval = np.mean(intervals)
                            if avg_interval > 0:
                                self._audio_state.bpm = 60 / avg_interval
                    else:
                        self._audio_state.beat = False
                else:
                    self._audio_state.beat = False
            
            # FFT for frequency analysis
            fft = np.abs(np.fft.rfft(samples))
            freqs = np.fft.rfftfreq(len(samples), 1 / self._sample_rate)
            
            # Calculate band energies
            for band in FrequencyBand:
                low, high = band.value
                mask = (freqs >= low) & (freqs < high)
                if np.any(mask):
                    band_energy = np.mean(fft[mask])
                    self._audio_state.band_energies[band.name] = min(1.0, band_energy * 20)
            
            # Voice detection
            if self._mode in [AudioMode.VOICE, AudioMode.HYBRID]:
                # Check for voice frequencies (80-400 Hz fundamental + harmonics)
                voice_mask = (freqs >= 80) & (freqs <= 4000)
                voice_energy = np.mean(fft[voice_mask]) if np.any(voice_mask) else 0
                
                self._audio_state.is_speaking = voice_energy > 0.01
                self._audio_state.mouth_open = min(1.0, voice_energy * 50)
                
                # Estimate pitch
                if self._audio_state.is_speaking:
                    peak_idx = np.argmax(fft[voice_mask])
                    peak_freq = freqs[voice_mask][peak_idx]
                    # Normalize to 0-1 (assuming 80-400Hz range for fundamental)
                    self._audio_state.pitch = (peak_freq - 80) / 320
            
            self._audio_state.timestamp = current_time
    
    def _update_avatar_state(self):
        """Update avatar reaction state from audio state."""
        config = self._config
        audio = self._audio_state
        avatar = self._avatar_state
        
        # Smoothing factor
        smooth = config.smoothing
        
        # Energy smoothing
        self._smooth_energy = self._smooth_energy * smooth + audio.energy * (1 - smooth)
        
        # Body bounce (triggered by bass/beat)
        if config.enable_body:
            bass_energy = audio.band_energies.get('BASS', 0) + audio.band_energies.get('SUB_BASS', 0)
            
            target_bounce = bass_energy * config.bass_to_bounce * config.energy_sensitivity
            if audio.beat:
                target_bounce = min(config.max_bounce, target_bounce + 0.5)
            
            self._smooth_bounce = self._smooth_bounce * smooth + target_bounce * (1 - smooth)
            avatar.body_bounce = self._smooth_bounce
            
            # Sway from mids
            mid_energy = audio.band_energies.get('MID', 0) + audio.band_energies.get('LOW_MID', 0)
            sway_dir = np.sin(time.time() * 2) * mid_energy * config.mids_to_sway
            self._smooth_sway = self._smooth_sway * smooth + sway_dir * (1 - smooth)
            avatar.body_sway = np.clip(self._smooth_sway, -config.max_sway, config.max_sway)
        
        # Head movement
        if config.enable_head:
            avatar.head_bob = avatar.body_bounce * 0.5
            avatar.head_tilt = avatar.body_sway * 0.3
        
        # Face
        if config.enable_face:
            self._smooth_mouth = self._smooth_mouth * smooth + audio.mouth_open * (1 - smooth)
            avatar.mouth_open = self._smooth_mouth * config.voice_sensitivity
            
            avatar.eye_energy = self._smooth_energy
            avatar.blink_rate = 1.0 + self._smooth_energy * 0.5
        
        # Arms
        if config.enable_arms:
            avatar.arm_raise = self._smooth_energy * config.energy_to_arms
        
        # Overall intensity
        avatar.intensity = self._smooth_energy
    
    def get_audio_state(self) -> AudioState:
        """Get current audio analysis state."""
        with self._lock:
            return AudioState(
                energy=self._audio_state.energy,
                beat=self._audio_state.beat,
                beat_interval=self._audio_state.beat_interval,
                bpm=self._audio_state.bpm,
                band_energies=dict(self._audio_state.band_energies),
                is_speaking=self._audio_state.is_speaking,
                mouth_open=self._audio_state.mouth_open,
                pitch=self._audio_state.pitch,
                timestamp=self._audio_state.timestamp
            )
    
    def get_avatar_state(self) -> AvatarReactionState:
        """Get current avatar reaction state."""
        with self._lock:
            return AvatarReactionState(
                body_bounce=self._avatar_state.body_bounce,
                body_sway=self._avatar_state.body_sway,
                head_bob=self._avatar_state.head_bob,
                head_tilt=self._avatar_state.head_tilt,
                mouth_open=self._avatar_state.mouth_open,
                eye_energy=self._avatar_state.eye_energy,
                blink_rate=self._avatar_state.blink_rate,
                arm_raise=self._avatar_state.arm_raise,
                intensity=self._avatar_state.intensity
            )
    
    def add_callback(self, callback: Callable[[AvatarReactionState], None]):
        """Add callback for avatar state updates."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[AvatarReactionState], None]):
        """Remove callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def connect_to_avatar(self, avatar_controller):
        """
        Connect to an avatar controller.
        
        Args:
            avatar_controller: AvatarController or AIAvatarController instance
        """
        def apply_state(state: AvatarReactionState):
            try:
                # Apply to avatar controller
                if hasattr(avatar_controller, 'set_mouth_open'):
                    avatar_controller.set_mouth_open(state.mouth_open)
                
                if hasattr(avatar_controller, 'set_body_position'):
                    # Apply bounce and sway
                    avatar_controller.set_body_position(
                        y_offset=state.body_bounce * 10,  # Scale to pixels/units
                        x_offset=state.body_sway * 20
                    )
                
                if hasattr(avatar_controller, 'set_energy_level'):
                    avatar_controller.set_energy_level(state.intensity)
                
            except Exception as e:
                logger.error(f"Failed to apply state to avatar: {e}")
        
        self.add_callback(apply_state)
    
    def feed_audio(self, samples: np.ndarray):
        """
        Feed audio samples directly (for non-microphone sources).
        
        Args:
            samples: Audio samples (mono, float32)
        """
        self._audio_buffer.extend(samples.tolist())
    
    def set_mode(self, mode: AudioMode):
        """Change the analysis mode."""
        self._mode = mode
    
    def set_sensitivity(
        self,
        energy: Optional[float] = None,
        beat: Optional[float] = None,
        voice: Optional[float] = None
    ):
        """Adjust sensitivity settings."""
        if energy is not None:
            self._config.energy_sensitivity = energy
        if beat is not None:
            self._config.beat_sensitivity = beat
        if voice is not None:
            self._config.voice_sensitivity = voice


# Global instance
_instance: Optional[AudioReactiveAvatar] = None


def get_audio_reactive_avatar() -> AudioReactiveAvatar:
    """Get or create global AudioReactiveAvatar instance."""
    global _instance
    if _instance is None:
        _instance = AudioReactiveAvatar()
    return _instance
