"""
Natural Text-to-Speech using Coqui TTS or Bark.

Much more natural sounding than pyttsx3.
"""

import logging

logger = logging.getLogger(__name__)


class NaturalTTS:
    """
    High-quality TTS using Coqui or Bark.
    
    Usage:
        tts = NaturalTTS(engine="coqui")  # or "bark"
        tts.speak("Hello, how are you?")
        tts.save("output.wav", "Hello world")
    """
    
    def __init__(self, engine: str = "coqui"):
        self.engine = engine
        self._tts = None
        
    def load(self) -> bool:
        """Load TTS engine."""
        if self.engine == "coqui":
            return self._load_coqui()
        elif self.engine == "bark":
            return self._load_bark()
        return False
    
    def _load_coqui(self) -> bool:
        try:
            from TTS.api import TTS
            self._tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
            return True
        except ImportError:
            logger.warning("Coqui TTS not installed. Install with: pip install TTS")
            return False
    
    def _load_bark(self) -> bool:
        try:
            from bark import SAMPLE_RATE, generate_audio, preload_models
            preload_models()
            self._bark_generate = generate_audio
            self._bark_sr = SAMPLE_RATE
            return True
        except ImportError:
            logger.warning("Bark not installed. Install with: pip install git+https://github.com/suno-ai/bark.git")
            return False
    
    def speak(self, text: str):
        """Speak text through speakers."""
        import atexit
        import os
        import subprocess
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            
        self.save(temp_path, text)
        # Play audio using subprocess (safer than os.system)
        if os.name == 'nt':
            # Windows: Popen doesn't wait, file may still be in use
            # Schedule cleanup for when process exits
            atexit.register(lambda p=temp_path: os.unlink(p) if os.path.exists(p) else None)
            subprocess.Popen(['cmd', '/c', 'start', '', temp_path], shell=False)
        elif os.name == 'posix':
            # Try aplay first, then afplay (macOS)
            try:
                subprocess.run(['aplay', temp_path], stderr=subprocess.DEVNULL, check=False, timeout=60)
            except FileNotFoundError:
                subprocess.run(['afplay', temp_path], stderr=subprocess.DEVNULL, check=False, timeout=60)
            # Cleanup temp file after playback
            try:
                os.unlink(temp_path)
            except Exception:
                pass  # Intentionally silent
    
    def save(self, path: str, text: str):
        """Save speech to file."""
        if not self._tts and self.engine == "coqui":
            self.load()
        
        if self.engine == "coqui" and self._tts:
            self._tts.tts_to_file(text=text, file_path=path)
        elif self.engine == "bark":
            import scipy.io.wavfile as wav
            audio = self._bark_generate(text)
            wav.write(path, self._bark_sr, audio)
