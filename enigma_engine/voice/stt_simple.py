"""
Pluggable STT adapter.
Tries to use VOSK (offline) if installed. If not found, uses SpeechRecognition with Google's API (online).
The API surface:
  - transcribe_from_mic(timeout=10) -> str
  - transcribe_from_file(path) -> str
"""
import logging
import os

logger = logging.getLogger(__name__)

# Try VOSK first (offline)
try:
    import sounddevice as sd
    import vosk
    HAVE_VOSK = True
except ImportError:
    HAVE_VOSK = False
    logger.debug("VOSK not available for offline STT")

# Try SpeechRecognition (fallback)
try:
    import speech_recognition as sr
    HAVE_SR = True
except ImportError:
    HAVE_SR = False
    logger.debug("SpeechRecognition not available")

def transcribe_from_mic(timeout=8):
    if HAVE_VOSK:
        return _transcribe_vosk_from_mic(timeout)
    elif HAVE_SR:
        return _transcribe_sr_from_mic(timeout)
    else:
        # no STT available; return empty string
        return ""

def transcribe_from_file(path):
    if HAVE_VOSK:
        return _transcribe_vosk_file(path)
    elif HAVE_SR:
        return _transcribe_sr_file(path)
    else:
        return ""

# VOSK helpers (simple)
def _transcribe_vosk_from_mic(timeout):
    try:
        model_path = "model"  # user should place a vosk model under ./model or change path
        if not os.path.exists(model_path):
            return ""  # no model
        wf = vosk.Model(model_path)
        rec = vosk.KaldiRecognizer(wf, 16000)
        # record for `timeout` seconds using sounddevice
        duration = timeout
        data = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='int16')
        sd.wait()
        rec.AcceptWaveform(data.tobytes())
        res = rec.Result()
        import json
        r = json.loads(res)
        return r.get("text", "")
    except Exception as e:
        logger.debug(f"VOSK microphone transcription failed: {e}")
        return ""

VOSK_FRAME_SIZE = 4000  # Number of audio frames to read per iteration
VOSK_MODEL_PATH = "model"  # Default path for VOSK model


def _transcribe_vosk_file(path):
    try:
        # simple file transcription using wav file path and vosk
        import json
        import wave
        with wave.open(path, "rb") as wf:
            model = vosk.Model(VOSK_MODEL_PATH)
            rec = vosk.KaldiRecognizer(model, wf.getframerate())
            results = []
            while True:
                data = wf.readframes(VOSK_FRAME_SIZE)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    results.append(rec.Result())
            results.append(rec.FinalResult())
        texts = []
        for r in results:
            try:
                j = json.loads(r)
                if j.get("text"):
                    texts.append(j["text"])
            except json.JSONDecodeError:
                logger.debug(f"Failed to parse VOSK result: {r}")
        return " ".join(texts)
    except Exception as e:
        logger.warning(f"VOSK file transcription failed: {e}")
        return ""

# SpeechRecognition fallback (online by default)
def _transcribe_sr_from_mic(timeout):
    try:
        import os
        import sys
        
        r = sr.Recognizer()
        
        # Suppress PyAudio stderr spam when opening microphone
        old_stderr = sys.stderr
        devnull = open(os.devnull, 'w')
        try:
            sys.stderr = devnull
            mic = sr.Microphone()
        except Exception:
            raise
        finally:
            sys.stderr = old_stderr
            devnull.close()
        
        with mic as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source, timeout=timeout)
        return r.recognize_google(audio)  # uses Google Web Speech API (online)
    except sr.WaitTimeoutError:
        logger.debug("Microphone listen timed out")
        return ""
    except sr.UnknownValueError:
        logger.debug("Speech not recognized")
        return ""
    except sr.RequestError as e:
        logger.warning(f"Google Speech API request failed: {e}")
        return ""
    except Exception as e:
        logger.error(f"Unexpected STT error: {e}")
        return ""

def _transcribe_sr_file(path):
    try:
        r = sr.Recognizer()
        with sr.AudioFile(path) as source:
            audio = r.record(source)
        return r.recognize_google(audio)
    except FileNotFoundError:
        logger.error(f"Audio file not found: {path}")
        return ""
    except sr.UnknownValueError:
        logger.debug(f"Speech not recognized in file: {path}")
        return ""
    except sr.RequestError as e:
        logger.warning(f"Google Speech API request failed: {e}")
        return ""
    except Exception as e:
        logger.error(f"Failed to transcribe file {path}: {e}")
        return ""
