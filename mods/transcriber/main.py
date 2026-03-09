"""
Transcriber Mod — Audio/video transcription using speech recognition.

Supports:
- File transcription (WAV, MP3, FLAC, OGG, MP4, etc.)
- Live microphone transcription
- Configurable energy threshold and language
"""

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict

from mod_base import ModClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranscriberMod(ModClient):
    """Audio/video transcription mod."""

    def __init__(self):
        super().__init__()
        self._listening = False
        self._listen_thread: threading.Thread | None = None
        self._recognizer = None
        self._microphone = None

    def _ensure_imports(self):
        """Lazy-import speech_recognition."""
        if self._recognizer is not None:
            return True
        try:
            import speech_recognition as sr
            self._recognizer = sr.Recognizer()
            energy = self.config.get("settings", {}).get("energy_threshold", 300)
            self._recognizer.energy_threshold = energy
            self._microphone = sr.Microphone
            return True
        except ImportError:
            logger.error("speech_recognition not installed. pip install SpeechRecognition")
            return False

    # =========================================================================
    # COMMANDS
    # =========================================================================

    def cmd_transcribe(self, args: Dict[str, Any]) -> str:
        """Transcribe an audio or video file to text."""
        if not self._ensure_imports():
            return "Error: speech_recognition package not installed"

        import speech_recognition as sr

        file_path = args.get("file_path", "").strip()
        if not file_path:
            return "Error: No file path provided"

        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found: {file_path}"

        try:
            audio_file = sr.AudioFile(str(path))
            with audio_file as source:
                audio = self._recognizer.record(source)

            language = self.config.get("settings", {}).get("language", "en-US")
            text = self._recognizer.recognize_google(audio, language=language)
            return f"Transcription:\n{text}"
        except sr.UnknownValueError:
            return "Could not understand the audio"
        except sr.RequestError as e:
            return f"Transcription service error: {e}"
        except Exception as e:
            return f"Error transcribing file: {e}"

    def cmd_listen(self, args: Dict[str, Any]) -> str:
        """Start live microphone transcription."""
        if not self._ensure_imports():
            return "Error: speech_recognition package not installed"

        if self._listening:
            return "Already listening"

        duration = args.get("duration", 0)
        self._listening = True
        self._listen_thread = threading.Thread(
            target=self._listen_loop,
            args=(duration,),
            daemon=True,
        )
        self._listen_thread.start()
        return "Live transcription started"

    def cmd_stop_listen(self, args: Dict[str, Any]) -> str:
        """Stop live transcription."""
        if not self._listening:
            return "Not currently listening"
        self._listening = False
        return "Live transcription stopped"

    def cmd_status(self, args: Dict[str, Any]) -> str:
        """Get transcriber status."""
        status = "listening" if self._listening else "idle"
        return f"Transcriber status: {status}"

    # =========================================================================
    # INTERNAL
    # =========================================================================

    def _listen_loop(self, duration: float) -> None:
        """Background loop for live mic transcription."""
        import speech_recognition as sr

        language = self.config.get("settings", {}).get("language", "en-US")
        start = time.monotonic()

        with self._microphone() as source:
            self._recognizer.adjust_for_ambient_noise(source, duration=0.5)

            while self._listening:
                if duration > 0 and (time.monotonic() - start) > duration:
                    break

                try:
                    audio = self._recognizer.listen(source, timeout=5, phrase_time_limit=15)
                    text = self._recognizer.recognize_google(audio, language=language)
                    if text:
                        self.send_message({
                            "type": "output",
                            "text": f"[Transcribed] {text}",
                        })
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    continue
                except Exception as e:
                    logger.debug(f"Listen error: {e}")
                    continue

        self._listening = False


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    mod = TranscriberMod()
    if mod.connect():
        mod.register()
        mod.run()
