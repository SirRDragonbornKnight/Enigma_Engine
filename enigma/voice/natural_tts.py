"""
Natural Text-to-Speech using Coqui TTS or Bark.

Much more natural sounding than pyttsx3.
"""

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
            print("Install Coqui TTS: pip install TTS")
            return False
    
    def _load_bark(self) -> bool:
        try:
            from bark import SAMPLE_RATE, generate_audio, preload_models
            preload_models()
            self._bark_generate = generate_audio
            self._bark_sr = SAMPLE_RATE
            return True
        except ImportError:
            print("Install Bark: pip install git+https://github.com/suno-ai/bark.git")
            return False
    
    def speak(self, text: str):
        """Speak text through speakers."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            self.save(f.name, text)
            # Play audio
            if os.name == 'nt':
                os.system(f'start {f.name}')
            elif os.name == 'posix':
                os.system(f'aplay {f.name} 2>/dev/null || afplay {f.name}')
    
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
