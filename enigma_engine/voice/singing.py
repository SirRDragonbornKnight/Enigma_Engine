"""
Singing Synthesis for Enigma AI Engine

Text-to-speech synthesis with melody for songs.

Features:
- Lyrics to singing conversion
- Pitch/note control
- Tempo adjustment
- Multiple voice styles
- MIDI synchronization

Usage:
    from enigma_engine.voice.singing import SingingVoice
    
    singer = SingingVoice()
    
    # Simple singing
    audio = singer.sing("Hello world", notes=["C4", "D4", "E4"])
    
    # With MIDI
    audio = singer.sing_with_midi("lyrics.txt", "melody.mid")
"""

import logging
import math
import struct
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Musical notes to frequencies
NOTE_FREQUENCIES = {
    "C": 261.63, "C#": 277.18, "Db": 277.18,
    "D": 293.66, "D#": 311.13, "Eb": 311.13,
    "E": 329.63, "F": 349.23, "F#": 369.99,
    "Gb": 369.99, "G": 392.00, "G#": 415.30,
    "Ab": 415.30, "A": 440.00, "A#": 466.16,
    "Bb": 466.16, "B": 493.88
}


def note_to_frequency(note: str) -> float:
    """
    Convert a note name to frequency.
    
    Args:
        note: Note name like "A4", "C#5", etc.
        
    Returns:
        Frequency in Hz
    """
    # Parse note name and octave
    import re
    match = re.match(r'^([A-G][#b]?)(\d+)$', note)
    
    if not match:
        return 440.0  # Default to A4
    
    note_name = match.group(1)
    octave = int(match.group(2))
    
    base_freq = NOTE_FREQUENCIES.get(note_name, 440.0)
    
    # Adjust for octave (A4 = 440 Hz)
    octave_diff = octave - 4
    return base_freq * (2 ** octave_diff)


class VoiceStyle(Enum):
    """Singing voice styles."""
    NATURAL = "natural"
    SOFT = "soft"
    POWERFUL = "powerful"
    WHISPER = "whisper"
    ROBOTIC = "robotic"
    CHOIR = "choir"


@dataclass
class Note:
    """A musical note."""
    pitch: str  # e.g., "C4", "A#3"
    duration: float  # in seconds
    text: str = ""  # syllable to sing
    velocity: float = 1.0  # volume 0-1
    
    @property
    def frequency(self) -> float:
        return note_to_frequency(self.pitch)


@dataclass
class SingingConfig:
    """Configuration for singing synthesis."""
    sample_rate: int = 44100
    tempo: int = 120  # BPM
    voice_style: VoiceStyle = VoiceStyle.NATURAL
    vibrato_depth: float = 0.02  # pitch variation
    vibrato_rate: float = 5.0  # Hz
    breathiness: float = 0.1
    portamento: float = 0.05  # pitch slide time


class Phoneme:
    """Phoneme representation for singing."""
    
    # Basic phoneme durations (relative)
    DURATIONS = {
        'a': 1.0, 'e': 0.9, 'i': 0.8, 'o': 1.0, 'u': 0.9,
        'p': 0.1, 'b': 0.1, 't': 0.1, 'd': 0.1, 'k': 0.1,
        'm': 0.3, 'n': 0.3, 'l': 0.3, 'r': 0.3
    }
    
    @classmethod
    def text_to_phonemes(cls, text: str) -> List[str]:
        """Simple text to phoneme conversion."""
        # Very simplified - just use characters
        phonemes = []
        for char in text.lower():
            if char.isalpha():
                phonemes.append(char)
        return phonemes


class SingingVoice:
    """Text-to-speech with singing capabilities."""
    
    def __init__(self, config: Optional[SingingConfig] = None):
        """
        Initialize singing voice.
        
        Args:
            config: Singing configuration
        """
        self.config = config or SingingConfig()
        
        # Try to load TTS engine
        self._tts_engine = self._init_tts()
    
    def _init_tts(self) -> Optional[Any]:
        """Initialize TTS engine."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            return engine
        except Exception as e:
            logger.warning(f"TTS not available: {e}")
            return None
    
    def sing(
        self,
        lyrics: str,
        notes: List[str],
        durations: Optional[List[float]] = None
    ) -> bytes:
        """
        Sing lyrics with specified notes.
        
        Args:
            lyrics: Text to sing
            notes: Note names for each syllable
            durations: Duration for each note (seconds)
            
        Returns:
            Audio data as bytes
        """
        # Parse lyrics into syllables
        syllables = self._syllabify(lyrics)
        
        # Match notes to syllables
        if len(notes) < len(syllables):
            # Repeat last note
            notes = notes + [notes[-1]] * (len(syllables) - len(notes))
        
        # Default durations
        if durations is None:
            beat_duration = 60.0 / self.config.tempo
            durations = [beat_duration] * len(syllables)
        
        # Create note objects
        note_objects = []
        for i, syllable in enumerate(syllables):
            note_objects.append(Note(
                pitch=notes[i] if i < len(notes) else "A4",
                duration=durations[i] if i < len(durations) else 0.5,
                text=syllable
            ))
        
        # Synthesize
        return self._synthesize(note_objects)
    
    def sing_with_midi(
        self,
        lyrics_file: Path,
        midi_file: Path
    ) -> bytes:
        """
        Sing lyrics synchronized to MIDI file.
        
        Args:
            lyrics_file: Path to lyrics file
            midi_file: Path to MIDI file
            
        Returns:
            Audio data as bytes
        """
        # Load lyrics
        lyrics = Path(lyrics_file).read_text()
        
        # Load MIDI
        notes = self._load_midi(midi_file)
        
        # Match lyrics to notes
        syllables = self._syllabify(lyrics)
        
        note_objects = []
        for i, syllable in enumerate(syllables):
            if i < len(notes):
                note_objects.append(Note(
                    pitch=notes[i]["pitch"],
                    duration=notes[i]["duration"],
                    text=syllable,
                    velocity=notes[i].get("velocity", 1.0)
                ))
        
        return self._synthesize(note_objects)
    
    def _syllabify(self, text: str) -> List[str]:
        """Split text into syllables."""
        # Simple syllabification based on vowels
        import re
        
        # Split on whitespace first
        words = text.split()
        syllables = []
        
        for word in words:
            # Simple: split on vowel boundaries
            word_syllables = re.findall(r'[^aeiou]*[aeiou]+[^aeiou]*', word.lower())
            
            if word_syllables:
                syllables.extend(word_syllables)
            else:
                syllables.append(word)
        
        return syllables
    
    def _load_midi(self, midi_file: Path) -> List[Dict]:
        """Load notes from MIDI file."""
        try:
            import mido
            
            midi = mido.MidiFile(midi_file)
            notes = []
            
            # Convert MIDI notes to our format
            tempo = 500000  # microseconds per beat (default 120 BPM)
            ticks_per_beat = midi.ticks_per_beat
            
            for msg in midi:
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                elif msg.type == 'note_on' and msg.velocity > 0:
                    # Convert MIDI note number to note name
                    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                    note_num = msg.note
                    octave = (note_num // 12) - 1
                    note_name = note_names[note_num % 12]
                    
                    notes.append({
                        "pitch": f"{note_name}{octave}",
                        "duration": 0.5,  # Will be updated
                        "velocity": msg.velocity / 127
                    })
            
            return notes
            
        except ImportError:
            logger.warning("mido not installed, MIDI support unavailable")
            return []
        except Exception as e:
            logger.error(f"Failed to load MIDI: {e}")
            return []
    
    def _synthesize(self, notes: List[Note]) -> bytes:
        """
        Synthesize singing audio.
        
        Args:
            notes: List of notes to sing
            
        Returns:
            Audio data
        """
        sample_rate = self.config.sample_rate
        samples = []
        
        prev_freq = None
        
        for note in notes:
            # Generate samples for this note
            num_samples = int(note.duration * sample_rate)
            freq = note.frequency
            
            for i in range(num_samples):
                t = i / sample_rate
                
                # Base frequency with vibrato
                vibrato = 1.0 + self.config.vibrato_depth * math.sin(
                    2 * math.pi * self.config.vibrato_rate * t
                )
                
                # Portamento (pitch slide from previous note)
                if prev_freq and i < sample_rate * self.config.portamento:
                    blend = i / (sample_rate * self.config.portamento)
                    current_freq = prev_freq + (freq - prev_freq) * blend
                else:
                    current_freq = freq
                
                current_freq *= vibrato
                
                # Generate waveform based on voice style
                sample = self._generate_voice_sample(
                    t, current_freq, note.velocity
                )
                
                # Apply envelope
                envelope = self._get_envelope(i, num_samples)
                sample *= envelope
                
                samples.append(sample)
            
            prev_freq = freq
        
        # Convert to bytes
        return self._samples_to_bytes(samples)
    
    def _generate_voice_sample(
        self,
        t: float,
        freq: float,
        velocity: float
    ) -> float:
        """Generate a single voice sample."""
        style = self.config.voice_style
        
        if style == VoiceStyle.NATURAL:
            # Mix of harmonics
            sample = (
                0.5 * math.sin(2 * math.pi * freq * t) +
                0.25 * math.sin(4 * math.pi * freq * t) +
                0.125 * math.sin(6 * math.pi * freq * t) +
                0.0625 * math.sin(8 * math.pi * freq * t)
            )
        
        elif style == VoiceStyle.SOFT:
            # Mostly fundamental
            sample = (
                0.8 * math.sin(2 * math.pi * freq * t) +
                0.15 * math.sin(4 * math.pi * freq * t)
            )
        
        elif style == VoiceStyle.POWERFUL:
            # Strong harmonics
            sample = (
                0.4 * math.sin(2 * math.pi * freq * t) +
                0.3 * math.sin(4 * math.pi * freq * t) +
                0.2 * math.sin(6 * math.pi * freq * t) +
                0.1 * math.sin(8 * math.pi * freq * t)
            )
        
        elif style == VoiceStyle.WHISPER:
            # Add noise
            import random
            noise = random.uniform(-0.3, 0.3)
            sample = 0.5 * math.sin(2 * math.pi * freq * t) + noise * 0.5
        
        elif style == VoiceStyle.ROBOTIC:
            # Square-ish wave
            val = math.sin(2 * math.pi * freq * t)
            sample = 1.0 if val > 0 else -1.0
            sample *= 0.5
        
        elif style == VoiceStyle.CHOIR:
            # Multiple detuned voices
            detuning = [0.98, 1.0, 1.02, 1.005, 0.995]
            sample = sum(
                0.2 * math.sin(2 * math.pi * freq * d * t)
                for d in detuning
            )
        
        else:
            sample = math.sin(2 * math.pi * freq * t)
        
        # Add breathiness
        if self.config.breathiness > 0:
            import random
            sample += random.uniform(-1, 1) * self.config.breathiness
        
        return sample * velocity
    
    def _get_envelope(self, sample: int, total: int) -> float:
        """Get ADSR envelope value."""
        attack = int(total * 0.05)
        decay = int(total * 0.1)
        sustain_level = 0.7
        release = int(total * 0.15)
        
        if sample < attack:
            # Attack
            return sample / attack
        elif sample < attack + decay:
            # Decay
            decay_progress = (sample - attack) / decay
            return 1.0 - (1.0 - sustain_level) * decay_progress
        elif sample < total - release:
            # Sustain
            return sustain_level
        else:
            # Release
            release_progress = (sample - (total - release)) / release
            return sustain_level * (1 - release_progress)
    
    def _samples_to_bytes(self, samples: List[float]) -> bytes:
        """Convert float samples to WAV bytes."""
        # Normalize
        max_val = max(abs(s) for s in samples) or 1.0
        normalized = [s / max_val for s in samples]
        
        # Convert to 16-bit integers
        int_samples = [int(s * 32767) for s in normalized]
        
        # Pack as bytes
        return struct.pack(f'{len(int_samples)}h', *int_samples)
    
    def save_wav(
        self,
        audio_data: bytes,
        output_path: Path
    ):
        """
        Save audio data as WAV file.
        
        Args:
            audio_data: Audio bytes
            output_path: Output file path
        """
        with wave.open(str(output_path), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(self.config.sample_rate)
            wav.writeframes(audio_data)
    
    def play(self, audio_data: bytes):
        """Play audio data."""
        try:
            import simpleaudio as sa
            
            play_obj = sa.play_buffer(
                audio_data,
                1,  # channels
                2,  # bytes per sample
                self.config.sample_rate
            )
            play_obj.wait_done()
            
        except ImportError:
            logger.warning("simpleaudio not installed, saving to temp file")
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                self.save_wav(audio_data, Path(f.name))
                logger.info(f"Saved to {f.name}")


# Convenience function
def sing(
    lyrics: str,
    notes: List[str],
    output_path: Optional[Path] = None
) -> bytes:
    """
    Quick singing synthesis.
    
    Args:
        lyrics: Text to sing
        notes: Note names
        output_path: Optional output file
        
    Returns:
        Audio data
    """
    singer = SingingVoice()
    audio = singer.sing(lyrics, notes)
    
    if output_path:
        singer.save_wav(audio, output_path)
    
    return audio
