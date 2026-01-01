"""
Pluggable STT adapter.
Tries to use VOSK (offline) if installed. If not found, uses SpeechRecognition with Google's API (online).
The API surface:
  - transcribe_from_mic(timeout=10) -> str
  - transcribe_from_file(path) -> str
"""
import os

# Try VOSK first (offline)
try:
    import vosk
    import sounddevice as sd
    HAVE_VOSK = True
except Exception:
    HAVE_VOSK = False

# Try SpeechRecognition (fallback)
try:
    import speech_recognition as sr
    HAVE_SR = True
except Exception:
    HAVE_SR = False

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
    except Exception:
        return ""

def _transcribe_vosk_file(path):
    try:
        # simple file transcription using wav file path and vosk
        import wave, json
        wf = wave.open(path, "rb")
        model = vosk.Model("model")
        rec = vosk.KaldiRecognizer(model, wf.getframerate())
        results = []
        while True:
            data = wf.readframes(4000)
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
            except Exception:
                pass
        return " ".join(texts)
    except Exception:
        return ""

# SpeechRecognition fallback (online by default)
def _transcribe_sr_from_mic(timeout):
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source, timeout=timeout)
        return r.recognize_google(audio)  # uses Google Web Speech API (online)
    except Exception:
        return ""

def _transcribe_sr_file(path):
    try:
        r = sr.Recognizer()
        with sr.AudioFile(path) as source:
            audio = r.record(source)
        return r.recognize_google(audio)
    except Exception:
        return ""
