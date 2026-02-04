"""
================================================================================
Streaming TTS - Start playing audio before full generation completes.
================================================================================

Provides low-latency text-to-speech by streaming audio chunks:
- Sentence-by-sentence streaming for faster first response
- Chunk-based audio playback
- Buffer management for smooth playback
- Multiple backend support (pyttsx3, edge-tts, coqui)

USAGE:
    from forge_ai.voice.streaming_tts import StreamingTTS
    
    tts = StreamingTTS()
    
    # Stream with callback
    def on_chunk(audio):
        play_audio(audio)
    
    tts.stream("This is a long text. It will play sentence by sentence.", on_chunk)
    
    # Or use async iterator
    async for chunk in tts.stream_async("Hello world"):
        await play_audio_async(chunk)
"""

from __future__ import annotations

import io
import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Generator, Iterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class StreamingBackend(Enum):
    """Available streaming TTS backends."""
    
    PYTTSX3 = auto()      # Local pyttsx3 (sentence chunking)
    EDGE_TTS = auto()     # Microsoft Edge TTS (true streaming)
    COQUI = auto()        # Coqui TTS (chunk generation)
    ESPEAK = auto()       # eSpeak (fast local)
    CALLBACK = auto()     # Custom callback-based


@dataclass
class StreamingConfig:
    """Configuration for streaming TTS."""
    
    # Chunking settings
    chunk_by: str = "sentence"        # sentence, clause, word, fixed
    fixed_chunk_size: int = 50        # Characters per chunk (if chunk_by=fixed)
    min_chunk_length: int = 3         # Minimum characters to speak
    
    # Buffer settings
    buffer_ahead: int = 2             # Chunks to buffer ahead
    buffer_timeout_ms: float = 5000   # Max wait for buffer
    
    # Playback settings
    sample_rate: int = 22050          # Output sample rate
    overlap_ms: float = 50            # Crossfade overlap between chunks
    
    # Backend settings
    preferred_backend: StreamingBackend = StreamingBackend.PYTTSX3
    voice: str = "default"
    rate: int = 150                   # Words per minute
    
    # Latency optimization
    prefetch_first: bool = True       # Start generating before full text
    parallel_generation: bool = False # Generate next chunk while playing


@dataclass
class AudioChunk:
    """A chunk of generated audio."""
    
    audio: np.ndarray                 # Audio samples (float32)
    sample_rate: int                  # Sample rate
    text: str                         # Source text
    index: int                        # Chunk index
    is_first: bool = False            # First chunk in stream
    is_last: bool = False             # Last chunk in stream
    duration_ms: float = 0.0          # Duration in milliseconds


class TextChunker:
    """Split text into chunks for streaming."""
    
    # Sentence-ending patterns
    SENTENCE_END = re.compile(r'(?<=[.!?])\s+')
    CLAUSE_END = re.compile(r'(?<=[.!?,;:])\s+')
    
    def __init__(self, config: StreamingConfig):
        self.config = config
    
    def chunk(self, text: str) -> List[str]:
        """Split text into chunks."""
        if self.config.chunk_by == "sentence":
            return self._chunk_sentences(text)
        elif self.config.chunk_by == "clause":
            return self._chunk_clauses(text)
        elif self.config.chunk_by == "word":
            return self._chunk_words(text)
        elif self.config.chunk_by == "fixed":
            return self._chunk_fixed(text)
        else:
            return [text]  # No chunking
    
    def _chunk_sentences(self, text: str) -> List[str]:
        """Split by sentence boundaries."""
        sentences = self.SENTENCE_END.split(text)
        return [s.strip() for s in sentences if len(s.strip()) >= self.config.min_chunk_length]
    
    def _chunk_clauses(self, text: str) -> List[str]:
        """Split by clause boundaries (more granular)."""
        clauses = self.CLAUSE_END.split(text)
        return [c.strip() for c in clauses if len(c.strip()) >= self.config.min_chunk_length]
    
    def _chunk_words(self, text: str) -> List[str]:
        """Split into individual words."""
        words = text.split()
        return [w for w in words if len(w) >= self.config.min_chunk_length]
    
    def _chunk_fixed(self, text: str) -> List[str]:
        """Split into fixed-size chunks."""
        chunks = []
        size = self.config.fixed_chunk_size
        
        for i in range(0, len(text), size):
            chunk = text[i:i + size]
            # Try to break at word boundary
            if i + size < len(text):
                last_space = chunk.rfind(' ')
                if last_space > size // 2:
                    chunk = chunk[:last_space]
            chunks.append(chunk.strip())
        
        return [c for c in chunks if len(c) >= self.config.min_chunk_length]


class StreamingTTS:
    """
    Streaming text-to-speech with low latency.
    
    Generates and plays audio chunks as text is processed,
    reducing time-to-first-audio significantly.
    """
    
    def __init__(self, config: StreamingConfig = None):
        self.config = config or StreamingConfig()
        
        # Text chunker
        self._chunker = TextChunker(self.config)
        
        # Backend engine (lazy loaded)
        self._engine = None
        self._backend: StreamingBackend = self.config.preferred_backend
        
        # Streaming state
        self._streaming = False
        self._stop_requested = False
        
        # Audio buffer
        self._audio_queue: queue.Queue = queue.Queue()
        
        # Playback thread
        self._playback_thread: Optional[threading.Thread] = None
        
        # Stats
        self._first_chunk_time: float = 0.0
        self._total_chunks: int = 0
    
    def _init_engine(self):
        """Initialize TTS engine based on backend."""
        if self._engine is not None:
            return
        
        backend = self.config.preferred_backend
        
        if backend == StreamingBackend.PYTTSX3:
            self._init_pyttsx3()
        elif backend == StreamingBackend.EDGE_TTS:
            self._init_edge_tts()
        elif backend == StreamingBackend.ESPEAK:
            self._init_espeak()
        elif backend == StreamingBackend.COQUI:
            self._init_coqui()
        else:
            # Fallback to pyttsx3
            self._init_pyttsx3()
    
    def _init_pyttsx3(self):
        """Initialize pyttsx3 backend."""
        try:
            import pyttsx3
            
            engine = pyttsx3.init()
            engine.setProperty('rate', self.config.rate)
            
            # Set voice if specified
            if self.config.voice != "default":
                voices = engine.getProperty('voices')
                for v in voices:
                    if self.config.voice.lower() in v.name.lower():
                        engine.setProperty('voice', v.id)
                        break
            
            self._engine = {"type": "pyttsx3", "engine": engine}
            self._backend = StreamingBackend.PYTTSX3
            logger.debug("Initialized pyttsx3 streaming backend")
            
        except Exception as e:
            logger.warning(f"Failed to init pyttsx3: {e}")
            self._init_espeak()
    
    def _init_espeak(self):
        """Initialize espeak backend."""
        try:
            import subprocess
            
            # Check if espeak is available
            result = subprocess.run(
                ["espeak", "--version"],
                capture_output=True,
                timeout=5
            )
            
            if result.returncode == 0:
                self._engine = {"type": "espeak"}
                self._backend = StreamingBackend.ESPEAK
                logger.debug("Initialized espeak streaming backend")
            else:
                raise RuntimeError("espeak not available")
                
        except Exception as e:
            logger.warning(f"Failed to init espeak: {e}")
            self._engine = {"type": "dummy"}
    
    def _init_edge_tts(self):
        """Initialize Edge TTS backend (async streaming)."""
        try:
            import edge_tts
            
            self._engine = {"type": "edge_tts", "module": edge_tts}
            self._backend = StreamingBackend.EDGE_TTS
            logger.debug("Initialized edge-tts streaming backend")
            
        except ImportError:
            logger.warning("edge-tts not installed, falling back to pyttsx3")
            self._init_pyttsx3()
    
    def _init_coqui(self):
        """Initialize Coqui TTS backend."""
        try:
            from TTS.api import TTS
            
            tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
            self._engine = {"type": "coqui", "tts": tts}
            self._backend = StreamingBackend.COQUI
            logger.debug("Initialized Coqui TTS streaming backend")
            
        except Exception as e:
            logger.warning(f"Failed to init Coqui TTS: {e}")
            self._init_pyttsx3()
    
    def stream(
        self,
        text: str,
        on_chunk: Callable[[AudioChunk], None] = None,
        on_complete: Callable[[], None] = None
    ) -> Generator[AudioChunk, None, None]:
        """
        Stream TTS audio chunk by chunk.
        
        Args:
            text: Text to speak
            on_chunk: Callback for each audio chunk
            on_complete: Callback when streaming completes
        
        Yields:
            AudioChunk objects with audio data
        """
        self._init_engine()
        self._streaming = True
        self._stop_requested = False
        self._first_chunk_time = 0.0
        self._total_chunks = 0
        
        start_time = time.time()
        
        # Split text into chunks
        chunks = self._chunker.chunk(text)
        
        if not chunks:
            self._streaming = False
            return
        
        logger.debug(f"Streaming {len(chunks)} chunks")
        
        try:
            for i, chunk_text in enumerate(chunks):
                if self._stop_requested:
                    break
                
                # Generate audio for this chunk
                audio = self._generate_chunk(chunk_text)
                
                if audio is None or len(audio) == 0:
                    continue
                
                # Track first chunk latency
                if self._first_chunk_time == 0.0:
                    self._first_chunk_time = time.time() - start_time
                    logger.debug(f"First chunk latency: {self._first_chunk_time*1000:.0f}ms")
                
                # Create chunk object
                chunk = AudioChunk(
                    audio=audio,
                    sample_rate=self.config.sample_rate,
                    text=chunk_text,
                    index=i,
                    is_first=(i == 0),
                    is_last=(i == len(chunks) - 1),
                    duration_ms=len(audio) / self.config.sample_rate * 1000
                )
                
                self._total_chunks += 1
                
                # Callback
                if on_chunk:
                    on_chunk(chunk)
                
                yield chunk
                
        finally:
            self._streaming = False
            if on_complete:
                on_complete()
    
    def stream_and_play(
        self,
        text: str,
        blocking: bool = True,
        on_complete: Callable[[], None] = None
    ):
        """
        Stream TTS and play audio automatically.
        
        Args:
            text: Text to speak
            blocking: Whether to wait for playback to complete
            on_complete: Callback when done
        """
        def play_chunks():
            try:
                import sounddevice as sd
            except ImportError:
                logger.warning("sounddevice not available for playback")
                return
            
            for chunk in self.stream(text):
                if self._stop_requested:
                    break
                
                try:
                    sd.play(chunk.audio, chunk.sample_rate)
                    sd.wait()
                except Exception as e:
                    logger.error(f"Playback error: {e}")
            
            if on_complete:
                on_complete()
        
        if blocking:
            play_chunks()
        else:
            self._playback_thread = threading.Thread(target=play_chunks, daemon=True)
            self._playback_thread.start()
    
    def _generate_chunk(self, text: str) -> Optional[np.ndarray]:
        """Generate audio for a text chunk."""
        if not self._engine:
            return None
        
        engine_type = self._engine.get("type")
        
        if engine_type == "pyttsx3":
            return self._generate_pyttsx3(text)
        elif engine_type == "espeak":
            return self._generate_espeak(text)
        elif engine_type == "edge_tts":
            return self._generate_edge_tts(text)
        elif engine_type == "coqui":
            return self._generate_coqui(text)
        else:
            return None
    
    def _generate_pyttsx3(self, text: str) -> Optional[np.ndarray]:
        """Generate audio using pyttsx3."""
        try:
            import tempfile
            import wave
            
            engine = self._engine["engine"]
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            
            engine.save_to_file(text, temp_path)
            engine.runAndWait()
            
            # Read the audio
            with wave.open(temp_path, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                
                # Convert to float32
                if sample_width == 2:
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                elif sample_width == 1:
                    audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
                else:
                    audio = np.frombuffer(frames, dtype=np.float32)
            
            # Resample if needed
            if sample_rate != self.config.sample_rate:
                audio = self._resample(audio, sample_rate, self.config.sample_rate)
            
            # Cleanup
            import os
            try:
                os.unlink(temp_path)
            except Exception:
                pass
            
            return audio
            
        except Exception as e:
            logger.error(f"pyttsx3 generation error: {e}")
            return None
    
    def _generate_espeak(self, text: str) -> Optional[np.ndarray]:
        """Generate audio using espeak."""
        try:
            import subprocess
            import tempfile
            import wave
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            
            # Run espeak
            subprocess.run(
                ["espeak", "-w", temp_path, "-s", str(self.config.rate), text],
                capture_output=True,
                timeout=30
            )
            
            # Read the audio
            with wave.open(temp_path, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                sample_rate = wf.getframerate()
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Resample if needed
            if sample_rate != self.config.sample_rate:
                audio = self._resample(audio, sample_rate, self.config.sample_rate)
            
            # Cleanup
            import os
            try:
                os.unlink(temp_path)
            except Exception:
                pass
            
            return audio
            
        except Exception as e:
            logger.error(f"espeak generation error: {e}")
            return None
    
    def _generate_edge_tts(self, text: str) -> Optional[np.ndarray]:
        """Generate audio using edge-tts (sync wrapper)."""
        try:
            import asyncio
            import tempfile
            import wave
            
            edge_tts = self._engine["module"]
            
            async def generate():
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    temp_path = f.name
                
                communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
                await communicate.save(temp_path)
                return temp_path
            
            # Run async in sync context
            loop = asyncio.new_event_loop()
            try:
                temp_path = loop.run_until_complete(generate())
            finally:
                loop.close()
            
            # Convert MP3 to numpy array
            audio = self._load_audio_file(temp_path)
            
            # Cleanup
            import os
            try:
                os.unlink(temp_path)
            except Exception:
                pass
            
            return audio
            
        except Exception as e:
            logger.error(f"edge-tts generation error: {e}")
            return None
    
    def _generate_coqui(self, text: str) -> Optional[np.ndarray]:
        """Generate audio using Coqui TTS."""
        try:
            tts = self._engine["tts"]
            
            # Generate to numpy array
            audio = tts.tts(text)
            
            if isinstance(audio, list):
                audio = np.array(audio, dtype=np.float32)
            
            return audio
            
        except Exception as e:
            logger.error(f"Coqui generation error: {e}")
            return None
    
    def _load_audio_file(self, path: str) -> Optional[np.ndarray]:
        """Load audio file to numpy array."""
        try:
            # Try soundfile first
            import soundfile as sf
            audio, sr = sf.read(path)
            if sr != self.config.sample_rate:
                audio = self._resample(audio, sr, self.config.sample_rate)
            return audio.astype(np.float32)
        except ImportError:
            pass
        
        try:
            # Try pydub
            from pydub import AudioSegment
            
            audio_seg = AudioSegment.from_file(path)
            audio_seg = audio_seg.set_frame_rate(self.config.sample_rate)
            samples = np.array(audio_seg.get_array_of_samples())
            return samples.astype(np.float32) / 32768.0
        except ImportError:
            pass
        
        return None
    
    def _resample(self, audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if src_rate == dst_rate:
            return audio
        
        try:
            import scipy.signal
            num_samples = int(len(audio) * dst_rate / src_rate)
            return scipy.signal.resample(audio, num_samples).astype(np.float32)
        except ImportError:
            # Simple linear interpolation fallback
            ratio = dst_rate / src_rate
            new_len = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_len)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
    
    def stop(self):
        """Stop streaming."""
        self._stop_requested = True
        
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=1.0)
    
    def is_streaming(self) -> bool:
        """Check if currently streaming."""
        return self._streaming
    
    def get_latency_ms(self) -> float:
        """Get first-chunk latency from last stream."""
        return self._first_chunk_time * 1000
    
    @property
    def backend_name(self) -> str:
        """Get current backend name."""
        return self._backend.name if self._backend else "NONE"


# Global streaming TTS instance
_streaming_tts: Optional[StreamingTTS] = None


def get_streaming_tts(config: StreamingConfig = None) -> StreamingTTS:
    """Get or create global streaming TTS instance."""
    global _streaming_tts
    if _streaming_tts is None:
        _streaming_tts = StreamingTTS(config)
    return _streaming_tts


def stream_speak(text: str, blocking: bool = True):
    """
    Convenience function for streaming TTS.
    
    Args:
        text: Text to speak
        blocking: Whether to wait for completion
    """
    tts = get_streaming_tts()
    tts.stream_and_play(text, blocking=blocking)


def stream_chunks(text: str) -> Generator[AudioChunk, None, None]:
    """
    Convenience generator for streaming audio chunks.
    
    Args:
        text: Text to speak
    
    Yields:
        AudioChunk objects
    """
    tts = get_streaming_tts()
    yield from tts.stream(text)
