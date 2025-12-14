"""
Pluggable TTS adapter. Prefer offline `pyttsx3` when available; fallback to platform speakers.
API:
  - speak(text)
"""
import platform
try:
    import pyttsx3
    HAVE_PYTT = True
except Exception:
    HAVE_PYTT = False

def speak(text: str):
    if not text:
        return
    if HAVE_PYTT:
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
