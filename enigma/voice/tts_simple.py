"""
Pluggable TTS adapter. Prefer offline `pyttsx3` when available; fallback to platform speakers.
API:
  - speak(text)
  - HAVE_PYTTSX3, HAVE_ESPEAK: Check available backends
"""
import platform
import shutil

# Check available backends
HAVE_PYTTSX3 = False
try:
    import pyttsx3
    HAVE_PYTTSX3 = True
except Exception:
    pass

HAVE_ESPEAK = shutil.which("espeak") is not None


def speak(text: str):
    if not text:
        return
    if HAVE_PYTTSX3:
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            return
        except Exception:
            pass
    # fallback to earlier tts_simple implementation
    try:
        if platform.system() == "Darwin":
            import os
            os.system(f'say "{text}"')
        elif platform.system() == "Windows":
            import subprocess
            subprocess.call(['powershell', '-c', f'Add-Type -AssemblyName System.speech; $s=new-object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak("{text}")'])
        else:
            import os
            os.system(f'echo "{text}" | espeak')
    except Exception:
        print("TTS failed")
