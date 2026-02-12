"""
Real-time Voice Chat System

Full duplex voice conversation with interruption support.
Handles audio streaming, VAD, and turn-taking.

FILE: enigma_engine/voice/voice_chat.py
TYPE: Voice
MAIN CLASSES: VoiceChat, AudioStream, VAD, TurnManager
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Optional imports
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logger.warning("pyaudio not available - audio streaming disabled")

try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    WEBRTC_VAD_AVAILABLE = False
    logger.warning("webrtcvad not available - using energy-based VAD")


class ChatState(Enum):
    """Voice chat state."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


class AudioFormat(Enum):
    """Audio format options."""
    PCM_16BIT = "pcm_16bit"
    FLOAT_32 = "float_32"


@dataclass
class AudioConfig:
    """Audio configuration."""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 480  # 30ms at 16kHz
    format: AudioFormat = AudioFormat.PCM_16BIT
    
    # VAD settings
    vad_aggressiveness: int = 2  # 0-3, higher = more aggressive
    speech_pad_ms: int = 300  # Padding around speech
    min_speech_ms: int = 250  # Minimum speech duration
    silence_threshold_ms: int = 800  # Silence to end utterance


@dataclass
class AudioChunk:
    """A chunk of audio data."""
    data: bytes
    timestamp: float
    is_speech: bool = False
    energy: float = 0.0


class EnergyVAD:
    """Simple energy-based Voice Activity Detection."""
    
    def __init__(
        self,
        threshold: float = 0.01,
        history_size: int = 10
    ):
        self.threshold = threshold
        self.history_size = history_size
        self._energy_history: list[float] = []
        self._adaptive_threshold = threshold
    
    def is_speech(self, audio_data: bytes, sample_width: int = 2) -> bool:
        """Check if audio chunk contains speech."""
        # Calculate RMS energy
        import struct
        
        if sample_width == 2:
            fmt = f'<{len(audio_data)//2}h'
            samples = struct.unpack(fmt, audio_data)
        else:
            samples = list(audio_data)
        
        if not samples:
            return False
        
        rms = (sum(s**2 for s in samples) / len(samples)) ** 0.5
        normalized = rms / 32768.0  # Normalize for 16-bit
        
        # Update adaptive threshold
        self._energy_history.append(normalized)
        if len(self._energy_history) > self.history_size:
            self._energy_history.pop(0)
        
        noise_floor = sum(self._energy_history) / len(self._energy_history)
        self._adaptive_threshold = max(self.threshold, noise_floor * 1.5)
        
        return normalized > self._adaptive_threshold
    
    @property
    def current_threshold(self) -> float:
        return self._adaptive_threshold


class WebRTCVAD:
    """WebRTC-based Voice Activity Detection."""
    
    def __init__(self, aggressiveness: int = 2, sample_rate: int = 16000):
        if not WEBRTC_VAD_AVAILABLE:
            raise ImportError("webrtcvad not installed")
        
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
    
    def is_speech(self, audio_data: bytes, sample_width: int = 2) -> bool:
        """Check if audio chunk contains speech."""
        try:
            return self.vad.is_speech(audio_data, self.sample_rate)
        except Exception:
            return False


class VAD:
    """Voice Activity Detection wrapper."""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        
        # Try WebRTC VAD first
        if WEBRTC_VAD_AVAILABLE:
            self._vad = WebRTCVAD(
                config.vad_aggressiveness,
                config.sample_rate
            )
            self._type = "webrtc"
        else:
            self._vad = EnergyVAD()
            self._type = "energy"
        
        logger.info(f"Using {self._type} VAD")
    
    def is_speech(self, audio_data: bytes) -> bool:
        """Check if audio contains speech."""
        return self._vad.is_speech(audio_data)


class AudioStream:
    """Audio input/output streaming."""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self._pyaudio: Any = None
        self._input_stream: Any = None
        self._output_stream: Any = None
        self._is_recording = False
        self._is_playing = False
    
    def start_recording(self) -> bool:
        """Start audio recording."""
        if not PYAUDIO_AVAILABLE:
            logger.error("pyaudio not available")
            return False
        
        try:
            self._pyaudio = pyaudio.PyAudio()
            
            fmt = pyaudio.paInt16 if self.config.format == AudioFormat.PCM_16BIT else pyaudio.paFloat32
            
            self._input_stream = self._pyaudio.open(
                format=fmt,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size
            )
            
            self._is_recording = True
            logger.info("Audio recording started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return False
    
    def stop_recording(self):
        """Stop audio recording."""
        self._is_recording = False
        
        if self._input_stream:
            self._input_stream.stop_stream()
            self._input_stream.close()
            self._input_stream = None
    
    def read_chunk(self) -> Optional[bytes]:
        """Read a chunk of audio data."""
        if not self._is_recording or not self._input_stream:
            return None
        
        try:
            return self._input_stream.read(
                self.config.chunk_size,
                exception_on_overflow=False
            )
        except Exception as e:
            logger.debug(f"Audio read error: {e}")
            return None
    
    def start_playback(self) -> bool:
        """Start audio playback."""
        if not PYAUDIO_AVAILABLE:
            logger.error("pyaudio not available")
            return False
        
        try:
            if not self._pyaudio:
                self._pyaudio = pyaudio.PyAudio()
            
            fmt = pyaudio.paInt16 if self.config.format == AudioFormat.PCM_16BIT else pyaudio.paFloat32
            
            self._output_stream = self._pyaudio.open(
                format=fmt,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                output=True,
                frames_per_buffer=self.config.chunk_size
            )
            
            self._is_playing = True
            logger.info("Audio playback started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start playback: {e}")
            return False
    
    def stop_playback(self):
        """Stop audio playback."""
        self._is_playing = False
        
        if self._output_stream:
            self._output_stream.stop_stream()
            self._output_stream.close()
            self._output_stream = None
    
    def write_chunk(self, data: bytes):
        """Write audio chunk for playback."""
        if self._is_playing and self._output_stream:
            try:
                self._output_stream.write(data)
            except Exception as e:
                logger.debug(f"Audio write error: {e}")
    
    def close(self):
        """Clean up resources."""
        self.stop_recording()
        self.stop_playback()
        
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None


class TurnManager:
    """Manages conversation turn-taking."""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        
        self._speech_frames: list[AudioChunk] = []
        self._silence_frames = 0
        self._is_speaking = False
        self._utterance_start: float = 0
    
    def process_chunk(self, chunk: AudioChunk) -> Optional[bytes]:
        """
        Process audio chunk and return complete utterance if ready.
        
        Returns:
            Complete utterance audio bytes, or None if still collecting
        """
        frames_per_ms = self.config.sample_rate / 1000
        silence_frames_needed = int(
            self.config.silence_threshold_ms * frames_per_ms / self.config.chunk_size
        )
        min_speech_frames = int(
            self.config.min_speech_ms * frames_per_ms / self.config.chunk_size
        )
        
        if chunk.is_speech:
            if not self._is_speaking:
                # Speech started
                self._is_speaking = True
                self._utterance_start = chunk.timestamp
                logger.debug("Speech started")
            
            self._speech_frames.append(chunk)
            self._silence_frames = 0
            
        else:
            if self._is_speaking:
                self._silence_frames += 1
                self._speech_frames.append(chunk)  # Include silence padding
                
                if self._silence_frames >= silence_frames_needed:
                    # Utterance complete
                    if len(self._speech_frames) >= min_speech_frames:
                        # Enough speech - return utterance
                        audio = b''.join(c.data for c in self._speech_frames)
                        self._reset()
                        return audio
                    else:
                        # Too short - discard
                        logger.debug("Discarding short utterance")
                        self._reset()
        
        return None
    
    def _reset(self):
        """Reset turn state."""
        self._speech_frames = []
        self._silence_frames = 0
        self._is_speaking = False
    
    def force_end_turn(self) -> Optional[bytes]:
        """Force end current turn and return audio."""
        if self._speech_frames:
            audio = b''.join(c.data for c in self._speech_frames)
            self._reset()
            return audio
        return None
    
    @property
    def is_user_speaking(self) -> bool:
        return self._is_speaking


class VoiceChat:
    """
    Real-time voice chat system.
    
    Supports full duplex conversation with interruption handling.
    """
    
    def __init__(
        self,
        config: AudioConfig = None,
        on_user_speech: Callable[[bytes], None] = None,
        on_state_change: Callable[[ChatState], None] = None
    ):
        """
        Initialize voice chat.
        
        Args:
            config: Audio configuration
            on_user_speech: Callback when user finishes speaking (receives audio bytes)
            on_state_change: Callback when chat state changes
        """
        self.config = config or AudioConfig()
        self._on_user_speech = on_user_speech
        self._on_state_change = on_state_change
        
        self._audio_stream = AudioStream(self.config)
        self._vad = VAD(self.config)
        self._turn_manager = TurnManager(self.config)
        
        # Thread safety lock for shared state
        self._lock = threading.Lock()
        
        self._state = ChatState.IDLE
        self._running = False
        self._listen_thread: Optional[threading.Thread] = None
        
        # Playback queue
        self._playback_queue: queue.Queue = queue.Queue()
        self._playback_thread: Optional[threading.Thread] = None
        self._is_playing_response = False
        
        # Interruption handling
        self._allow_interruption = True
        self._interrupt_threshold = 3  # Speech frames to trigger interrupt
        self._consecutive_speech = 0
    
    @property
    def state(self) -> ChatState:
        with self._lock:
            return self._state
    
    def _set_state(self, new_state: ChatState):
        """Set state and notify callback."""
        with self._lock:
            if new_state != self._state:
                old_state = self._state
                self._state = new_state
                logger.debug(f"State change: {old_state.value} -> {new_state.value}")
                
                if self._on_state_change:
                    self._on_state_change(new_state)
    
    def start(self) -> bool:
        """Start voice chat session."""
        with self._lock:
            if self._running:
                return True
            self._running = True
        
        # Start audio
        if not self._audio_stream.start_recording():
            with self._lock:
                self._running = False
            return False
        
        if not self._audio_stream.start_playback():
            self._audio_stream.stop_recording()
            with self._lock:
                self._running = False
            return False
        
        # Start listening thread
        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()
        
        # Start playback thread
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._playback_thread.start()
        
        self._set_state(ChatState.LISTENING)
        logger.info("Voice chat started")
        return True
    
    def stop(self):
        """Stop voice chat session."""
        with self._lock:
            self._running = False
        self._set_state(ChatState.IDLE)
        
        # Signal playback thread to exit
        self._playback_queue.put(None)
        
        # Clear playback queue
        while not self._playback_queue.empty():
            try:
                self._playback_queue.get_nowait()
            except queue.Empty:
                break
        
        self._audio_stream.close()
        
        if self._listen_thread:
            self._listen_thread.join(timeout=1.0)
        if self._playback_thread:
            self._playback_thread.join(timeout=1.0)
        
        logger.info("Voice chat stopped")
    
    def __del__(self):
        """Ensure proper cleanup when the object is garbage collected."""
        try:
            self._running = False
            
            # Close audio stream
            if hasattr(self, '_audio_stream') and self._audio_stream:
                try:
                    self._audio_stream.close()
                except Exception:
                    pass  # Intentionally silent
            
            # Signal playback queue to stop (don't wait - daemon threads will die with process)
            if hasattr(self, '_playback_queue') and self._playback_queue:
                try:
                    self._playback_queue.put_nowait(None)
                except Exception:
                    pass  # Intentionally silent
        except Exception:
            # Ignore errors during garbage collection
            pass
    
    def _listen_loop(self):
        """Main listening loop."""
        while self._running:
            # Read audio chunk
            data = self._audio_stream.read_chunk()
            if data is None:
                time.sleep(0.01)
                continue
            
            # Run VAD
            is_speech = self._vad.is_speech(data)
            
            chunk = AudioChunk(
                data=data,
                timestamp=time.time(),
                is_speech=is_speech
            )
            
            # Check for interruption
            if self._is_playing_response and self._allow_interruption:
                if is_speech:
                    self._consecutive_speech += 1
                    if self._consecutive_speech >= self._interrupt_threshold:
                        self._handle_interruption()
                else:
                    self._consecutive_speech = 0
            
            # Process turn
            if self._state in (ChatState.LISTENING, ChatState.INTERRUPTED):
                utterance = self._turn_manager.process_chunk(chunk)
                
                if self._turn_manager.is_user_speaking:
                    self._set_state(ChatState.LISTENING)
                
                if utterance:
                    # User finished speaking
                    self._set_state(ChatState.PROCESSING)
                    if self._on_user_speech:
                        self._on_user_speech(utterance)
    
    def _playback_loop(self):
        """Audio playback loop."""
        while self._running:
            try:
                data = self._playback_queue.get(timeout=0.1)
                if data is None:
                    # End of response marker
                    self._is_playing_response = False
                    if self._state == ChatState.SPEAKING:
                        self._set_state(ChatState.LISTENING)
                else:
                    self._audio_stream.write_chunk(data)
            except queue.Empty:
                continue
    
    def _handle_interruption(self):
        """Handle user interruption."""
        logger.info("User interrupted")
        self._set_state(ChatState.INTERRUPTED)
        
        # Clear playback queue
        while not self._playback_queue.empty():
            try:
                self._playback_queue.get_nowait()
            except queue.Empty:
                break
        
        self._is_playing_response = False
        self._consecutive_speech = 0
    
    def play_response(self, audio_data: bytes):
        """
        Play AI response audio.
        
        Args:
            audio_data: Raw audio bytes to play
        """
        if not self._running:
            return
        
        self._set_state(ChatState.SPEAKING)
        self._is_playing_response = True
        self._consecutive_speech = 0
        
        # Chunk the audio for streaming playback
        chunk_size = self.config.chunk_size * 2  # 2 bytes per sample
        
        for i in range(0, len(audio_data), chunk_size):
            if not self._is_playing_response:
                break  # Interrupted
            self._playback_queue.put(audio_data[i:i + chunk_size])
        
        # End marker
        self._playback_queue.put(None)
    
    def play_response_stream(self, audio_generator):
        """
        Play streamed AI response.
        
        Args:
            audio_generator: Generator yielding audio chunks
        """
        if not self._running:
            return
        
        self._set_state(ChatState.SPEAKING)
        self._is_playing_response = True
        self._consecutive_speech = 0
        
        for chunk in audio_generator:
            if not self._is_playing_response:
                break  # Interrupted
            self._playback_queue.put(chunk)
        
        # End marker
        self._playback_queue.put(None)
    
    @property
    def is_user_speaking(self) -> bool:
        """Check if user is currently speaking."""
        return self._turn_manager.is_user_speaking
    
    @property
    def allow_interruption(self) -> bool:
        return self._allow_interruption
    
    @allow_interruption.setter
    def allow_interruption(self, value: bool):
        self._allow_interruption = value


def create_voice_chat(
    on_user_speech: Callable[[bytes], None] = None,
    on_state_change: Callable[[ChatState], None] = None,
    **config_kwargs
) -> VoiceChat:
    """
    Create a voice chat instance.
    
    Args:
        on_user_speech: Callback for completed user speech
        on_state_change: Callback for state changes
        **config_kwargs: Audio configuration options
    
    Returns:
        Configured VoiceChat instance
    """
    config = AudioConfig(**config_kwargs)
    return VoiceChat(
        config=config,
        on_user_speech=on_user_speech,
        on_state_change=on_state_change
    )


__all__ = [
    'VoiceChat',
    'AudioStream',
    'VAD',
    'TurnManager',
    'ChatState',
    'AudioConfig',
    'AudioChunk',
    'create_voice_chat'
]
