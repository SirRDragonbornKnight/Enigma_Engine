"""
Background Music Mixing for Enigma AI Engine

Mix background music and voice output.

Features:
- Auto-ducking when speaking
- Volume normalization
- Multiple audio channels
- Crossfading
- Mood-based playlists

Usage:
    from enigma_engine.voice.audio_mixer import AudioMixer
    
    mixer = AudioMixer()
    
    # Set background music
    mixer.set_background("ambient.mp3", volume=0.3)
    
    # Play speech (auto-ducks music)
    mixer.play_speech(audio_data)
    
    # Change music with crossfade
    mixer.crossfade_to("action.mp3", duration=2.0)
"""

import logging
import os
import queue
import threading
import time
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class AudioChannel(Enum):
    """Audio channel types."""
    MASTER = "master"
    MUSIC = "music"
    SPEECH = "speech"
    EFFECTS = "effects"
    AMBIENT = "ambient"


class MixerState(Enum):
    """Mixer states."""
    IDLE = "idle"
    PLAYING = "playing"
    DUCKED = "ducked"  # Music lowered for speech
    CROSSFADING = "crossfading"


@dataclass
class AudioSource:
    """Audio source configuration."""
    path: str = ""
    volume: float = 1.0
    loop: bool = False
    channel: AudioChannel = AudioChannel.MUSIC
    
    # Ducking settings
    duck_on_speech: bool = True
    duck_volume: float = 0.2  # Volume when ducked
    duck_attack: float = 0.1  # Seconds to duck down
    duck_release: float = 0.3  # Seconds to duck up


@dataclass  
class MixerConfig:
    """Audio mixer configuration."""
    sample_rate: int = 44100
    channels: int = 2
    buffer_size: int = 1024
    
    # Volume levels
    master_volume: float = 1.0
    music_volume: float = 0.5
    speech_volume: float = 1.0
    effects_volume: float = 0.7
    
    # Ducking
    auto_duck: bool = True
    duck_threshold: float = 0.1  # Speech level to trigger duck


class AudioMixer:
    """Multi-channel audio mixer with ducking."""
    
    def __init__(self, config: Optional[MixerConfig] = None):
        """
        Initialize audio mixer.
        
        Args:
            config: Mixer configuration
        """
        self.config = config or MixerConfig()
        
        # State
        self.state = MixerState.IDLE
        self._running = False
        
        # Audio buffers per channel
        self._buffers: Dict[AudioChannel, List[np.ndarray]] = {
            ch: [] for ch in AudioChannel
        }
        
        # Volume levels per channel
        self._volumes: Dict[AudioChannel, float] = {
            AudioChannel.MASTER: self.config.master_volume,
            AudioChannel.MUSIC: self.config.music_volume,
            AudioChannel.SPEECH: self.config.speech_volume,
            AudioChannel.EFFECTS: self.config.effects_volume,
            AudioChannel.AMBIENT: 0.3,
        }
        
        # Ducking state
        self._current_duck = 1.0  # 1.0 = full volume
        self._target_duck = 1.0
        
        # Current playing
        self._music_source: Optional[AudioSource] = None
        self._music_position = 0
        self._music_data: Optional[np.ndarray] = None
        
        # Crossfade state
        self._crossfade_from: Optional[np.ndarray] = None
        self._crossfade_to: Optional[np.ndarray] = None
        self._crossfade_progress = 0.0
        self._crossfade_duration = 0.0
        
        # Speech queue
        self._speech_queue: queue.Queue = queue.Queue()
        
        # Threads
        self._play_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        logger.info("AudioMixer initialized")
    
    def start(self):
        """Start the mixer."""
        if self._running:
            return
        
        self._running = True
        self._play_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._play_thread.start()
        
        self.state = MixerState.PLAYING
        logger.info("AudioMixer started")
    
    def stop(self):
        """Stop the mixer."""
        self._running = False
        
        if self._play_thread:
            self._play_thread.join(timeout=2.0)
        
        self.state = MixerState.IDLE
        logger.info("AudioMixer stopped")
    
    def set_background(
        self,
        audio_path: str,
        volume: float = 0.5,
        loop: bool = True,
        fade_in: float = 1.0
    ):
        """
        Set background music.
        
        Args:
            audio_path: Path to audio file
            volume: Playback volume (0-1)
            loop: Loop the audio
            fade_in: Fade in duration in seconds
        """
        source = AudioSource(
            path=audio_path,
            volume=volume,
            loop=loop,
            channel=AudioChannel.MUSIC
        )
        
        # Load audio
        audio_data = self._load_audio(audio_path)
        
        if audio_data is None:
            logger.error(f"Failed to load audio: {audio_path}")
            return
        
        with self._lock:
            self._music_source = source
            self._music_data = audio_data
            self._music_position = 0
            self._volumes[AudioChannel.MUSIC] = volume
        
        logger.info(f"Set background music: {audio_path}")
    
    def stop_background(self, fade_out: float = 1.0):
        """Stop background music."""
        with self._lock:
            self._music_source = None
            self._music_data = None
            self._music_position = 0
    
    def play_speech(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 22050,
        wait: bool = False
    ):
        """
        Play speech audio.
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate
            wait: Wait for playback to complete
        """
        # Resample if needed
        if sample_rate != self.config.sample_rate:
            audio_data = self._resample(audio_data, sample_rate, self.config.sample_rate)
        
        # Add to queue
        self._speech_queue.put(audio_data)
        
        # Trigger ducking
        if self.config.auto_duck:
            self._duck()
        
        if wait:
            # Wait for queue to empty
            while not self._speech_queue.empty():
                time.sleep(0.1)
    
    def play_effect(
        self,
        audio_path: str,
        volume: float = 0.7
    ):
        """Play a sound effect."""
        audio_data = self._load_audio(audio_path)
        
        if audio_data is not None:
            with self._lock:
                # Scale volume
                audio_data = audio_data * volume * self._volumes[AudioChannel.EFFECTS]
                self._buffers[AudioChannel.EFFECTS].append(audio_data)
    
    def crossfade_to(
        self,
        audio_path: str,
        duration: float = 2.0,
        volume: float = 0.5
    ):
        """
        Crossfade to new background music.
        
        Args:
            audio_path: New audio file
            duration: Crossfade duration in seconds
            volume: New track volume
        """
        new_audio = self._load_audio(audio_path)
        
        if new_audio is None:
            return
        
        with self._lock:
            self._crossfade_from = self._music_data
            self._crossfade_to = new_audio
            self._crossfade_progress = 0.0
            self._crossfade_duration = duration
            
            self._music_data = new_audio
            self._music_position = 0
            self._volumes[AudioChannel.MUSIC] = volume
        
        self.state = MixerState.CROSSFADING
    
    def set_volume(self, channel: AudioChannel, volume: float):
        """Set channel volume."""
        self._volumes[channel] = max(0.0, min(1.0, volume))
    
    def get_volume(self, channel: AudioChannel) -> float:
        """Get channel volume."""
        return self._volumes.get(channel, 1.0)
    
    def _duck(self):
        """Start ducking music."""
        self._target_duck = self._music_source.duck_volume if self._music_source else 0.2
        self.state = MixerState.DUCKED
    
    def _unduck(self):
        """Stop ducking music."""
        self._target_duck = 1.0
        self.state = MixerState.PLAYING
    
    def _playback_loop(self):
        """Main playback loop."""
        try:
            import sounddevice as sd
            has_sounddevice = True
        except ImportError:
            has_sounddevice = False
            logger.warning("sounddevice not installed, audio output disabled")
        
        buffer_samples = self.config.buffer_size
        
        while self._running:
            # Mix audio
            mixed = self._mix_buffer(buffer_samples)
            
            # Apply master volume
            mixed = mixed * self._volumes[AudioChannel.MASTER]
            
            # Output
            if has_sounddevice and mixed is not None and len(mixed) > 0:
                try:
                    import sounddevice as sd
                    sd.play(mixed, self.config.sample_rate, blocking=False)
                except Exception as e:
                    logger.debug(f"Playback error: {e}")
            
            time.sleep(buffer_samples / self.config.sample_rate)
    
    def _mix_buffer(self, num_samples: int) -> np.ndarray:
        """Mix all channels into output buffer."""
        output = np.zeros(num_samples)
        
        with self._lock:
            # Music channel
            if self._music_data is not None and self._music_source:
                music_samples = self._get_music_samples(num_samples)
                
                # Apply ducking
                self._update_duck()
                music_samples = music_samples * self._current_duck
                
                output += music_samples * self._volumes[AudioChannel.MUSIC]
            
            # Speech channel
            if not self._speech_queue.empty():
                try:
                    speech_data = self._speech_queue.get_nowait()
                    # Only take what we need
                    speech_samples = speech_data[:num_samples]
                    if len(speech_samples) < num_samples:
                        speech_samples = np.pad(speech_samples, (0, num_samples - len(speech_samples)))
                    output += speech_samples * self._volumes[AudioChannel.SPEECH]
                    
                    # Put back remainder
                    if len(speech_data) > num_samples:
                        self._speech_queue.put(speech_data[num_samples:])
                    else:
                        # Speech done, unduck
                        if self._speech_queue.empty():
                            self._unduck()
                except queue.Empty:
                    pass
            
            # Effects channel
            for buffer in list(self._buffers[AudioChannel.EFFECTS]):
                samples = buffer[:num_samples]
                if len(samples) < num_samples:
                    samples = np.pad(samples, (0, num_samples - len(samples)))
                output += samples
                
                # Remove finished buffers
                if len(buffer) <= num_samples:
                    self._buffers[AudioChannel.EFFECTS].remove(buffer)
                else:
                    idx = self._buffers[AudioChannel.EFFECTS].index(buffer)
                    self._buffers[AudioChannel.EFFECTS][idx] = buffer[num_samples:]
        
        # Clip to prevent distortion
        output = np.clip(output, -1.0, 1.0)
        
        return output
    
    def _get_music_samples(self, num_samples: int) -> np.ndarray:
        """Get samples from current music."""
        if self._music_data is None:
            return np.zeros(num_samples)
        
        end_pos = self._music_position + num_samples
        
        if end_pos <= len(self._music_data):
            samples = self._music_data[self._music_position:end_pos]
            self._music_position = end_pos
        else:
            # Wrap around if looping
            samples = self._music_data[self._music_position:]
            
            if self._music_source and self._music_source.loop:
                remaining = num_samples - len(samples)
                samples = np.concatenate([
                    samples,
                    self._music_data[:remaining]
                ])
                self._music_position = remaining
            else:
                samples = np.pad(samples, (0, num_samples - len(samples)))
                self._music_source = None
        
        return samples
    
    def _update_duck(self):
        """Update ducking level."""
        if self._current_duck != self._target_duck:
            # Smooth transition
            rate = 0.1 if self._current_duck > self._target_duck else 0.05
            diff = self._target_duck - self._current_duck
            self._current_duck += diff * rate
            
            # Snap when close
            if abs(diff) < 0.01:
                self._current_duck = self._target_duck
    
    def _load_audio(self, path: str) -> Optional[np.ndarray]:
        """Load audio file to numpy array."""
        path = Path(path)
        
        if not path.exists():
            return None
        
        try:
            if path.suffix.lower() == '.wav':
                return self._load_wav(str(path))
            else:
                # Try using pydub for other formats
                try:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_file(str(path))
                    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
                    samples /= 32768.0  # Normalize
                    
                    # Resample if needed
                    if audio.frame_rate != self.config.sample_rate:
                        samples = self._resample(samples, audio.frame_rate, self.config.sample_rate)
                    
                    return samples
                except ImportError:
                    logger.warning("pydub not installed, only WAV supported")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to load audio {path}: {e}")
            return None
    
    def _load_wav(self, path: str) -> np.ndarray:
        """Load WAV file."""
        with wave.open(path, 'r') as wav:
            sample_rate = wav.getframerate()
            n_frames = wav.getnframes()
            n_channels = wav.getnchannels()
            audio_bytes = wav.readframes(n_frames)
            
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            audio /= 32768.0
            
            # Convert stereo to mono
            if n_channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)
            
            # Resample
            if sample_rate != self.config.sample_rate:
                audio = self._resample(audio, sample_rate, self.config.sample_rate)
            
            return audio
    
    def _resample(self, audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """Resample audio."""
        if src_rate == dst_rate:
            return audio
        
        ratio = dst_rate / src_rate
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio)


@dataclass
class Playlist:
    """Music playlist."""
    name: str
    tracks: List[str] = field(default_factory=list)
    mood: str = "neutral"  # neutral, happy, sad, intense, calm
    shuffle: bool = False
    current_index: int = 0
    
    def next(self) -> Optional[str]:
        """Get next track."""
        if not self.tracks:
            return None
        
        if self.shuffle:
            import random
            return random.choice(self.tracks)
        
        track = self.tracks[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.tracks)
        return track


class MoodMusicManager:
    """Manage music based on mood/context."""
    
    def __init__(self, mixer: AudioMixer):
        self.mixer = mixer
        self._playlists: Dict[str, Playlist] = {}
        self._current_mood = "neutral"
    
    def add_playlist(self, playlist: Playlist):
        """Add a playlist."""
        self._playlists[playlist.mood] = playlist
    
    def set_mood(self, mood: str, crossfade: float = 2.0):
        """Change current mood and music."""
        if mood == self._current_mood:
            return
        
        playlist = self._playlists.get(mood)
        if not playlist:
            logger.warning(f"No playlist for mood: {mood}")
            return
        
        track = playlist.next()
        if track:
            self.mixer.crossfade_to(track, duration=crossfade)
        
        self._current_mood = mood


# Global instance
_audio_mixer: Optional[AudioMixer] = None


def get_audio_mixer() -> AudioMixer:
    """Get or create global AudioMixer instance."""
    global _audio_mixer
    if _audio_mixer is None:
        _audio_mixer = AudioMixer()
    return _audio_mixer
