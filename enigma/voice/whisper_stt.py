"""
Whisper Speech-to-Text - Much more accurate than Vosk/SpeechRecognition.

Uses OpenAI's Whisper model locally (no API needed).
"""

class WhisperSTT:
    """
    Local Whisper speech recognition.
    
    Usage:
        stt = WhisperSTT(model_size="base")  # tiny, base, small, medium, large
        text = stt.transcribe("audio.wav")
        # Or from microphone
        text = stt.listen()
    """
    
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.model = None
        
    def load(self):
        """Load Whisper model (downloads on first use)."""
        try:
            import whisper
            self.model = whisper.load_model(self.model_size)
            return True
        except ImportError:
            print("Install whisper: pip install openai-whisper")
            return False
    
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text."""
        if not self.model:
            self.load()
        result = self.model.transcribe(audio_path)
        return result["text"]
    
    def listen(self, duration: int = 5) -> str:
        """Listen from microphone and transcribe."""
        import tempfile
        import sounddevice as sd
        import scipy.io.wavfile as wav
        
        # Record audio
        sample_rate = 16000
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav.write(f.name, sample_rate, audio)
            return self.transcribe(f.name)
