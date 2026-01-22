"""
Pluggable TTS adapter. Prefer offline `pyttsx3` when available; fallback to platform speakers.
API:
  - speak(text)
  - HAVE_PYTTSX3, HAVE_ESPEAK, HAVE_FESTIVAL: Check available backends
"""
import platform
import shutil
import subprocess

# Check available backends
HAVE_PYTTSX3 = False
try:
    import pyttsx3
    HAVE_PYTTSX3 = True
except Exception:
    pass

HAVE_ESPEAK = shutil.which("espeak") is not None or shutil.which("espeak-ng") is not None
HAVE_FESTIVAL = shutil.which("festival") is not None
HAVE_PICO = shutil.which("pico2wave") is not None


def _sanitize_text(text: str) -> str:
    """Sanitize text to prevent shell injection."""
    # Remove potentially dangerous characters for shell commands
    return text.replace('"', "'").replace('`', "'").replace('$', '').replace('\\', '')


def speak(text: str):
    if not text:
        return
    
    # Sanitize for shell commands
    safe_text = _sanitize_text(text)
    
    if HAVE_PYTTSX3:
        try:
            engine = pyttsx3.init()
            engine.say(text)  # pyttsx3 handles its own text safely
            engine.runAndWait()
            return
        except Exception:
            pass
    
    # Platform-specific fallbacks
    try:
        if platform.system() == "Darwin":
            # macOS: use 'say' command
            subprocess.run(['say', safe_text], check=False)
        elif platform.system() == "Windows":
            # Windows: use SAPI via PowerShell
            subprocess.run(
                ['powershell', '-c', 
                 f'Add-Type -AssemblyName System.speech; $s=new-object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak("{safe_text}")'],
                check=False
            )
        else:
            # Linux: try multiple TTS options
            if shutil.which("espeak-ng"):
                subprocess.run(['espeak-ng', safe_text], check=False)
            elif shutil.which("espeak"):
                subprocess.run(['espeak', safe_text], check=False)
            elif shutil.which("festival"):
                # Festival requires text via stdin
                process = subprocess.Popen(['festival', '--tts'], stdin=subprocess.PIPE, text=True)
                process.communicate(input=safe_text)
            elif shutil.which("pico2wave"):
                # Pico TTS (high quality, needs temp file)
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    temp_path = f.name
                subprocess.run(['pico2wave', '-w', temp_path, safe_text], check=False)
                if shutil.which("aplay"):
                    subprocess.run(['aplay', temp_path], check=False)
                elif shutil.which("paplay"):
                    subprocess.run(['paplay', temp_path], check=False)
                os.unlink(temp_path)
            elif shutil.which("spd-say"):
                # Speech-dispatcher
                subprocess.run(['spd-say', safe_text], check=False)
            else:
                print(f"TTS: {text}")  # Fallback: just print
    except Exception as e:
        print(f"TTS failed: {e}")
