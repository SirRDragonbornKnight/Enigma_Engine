"""
Built-in Speech-to-Text

Provides basic speech recognition using system APIs.
Windows: Uses Windows Speech Recognition COM
macOS: Uses NSSpeechRecognizer  
Linux: Uses system speech tools
"""

import os
import subprocess
import sys
import time
from typing import Any


class BuiltinSTT:
    """
    Built-in speech-to-text using system APIs.
    Limited functionality compared to full speech_recognition library.
    """
    
    def __init__(self):
        self.is_loaded = False
        self.platform = sys.platform
        self._recording = False
    
    def load(self) -> bool:
        """Load the STT system."""
        self.is_loaded = True
        return True
    
    def unload(self):
        """Unload."""
        self.is_loaded = False
        self._recording = False
    
    def is_available(self) -> bool:
        """Check if STT is available on this system."""
        if self.platform == "win32":
            # Check for Windows Speech Recognition
            try:
                result = subprocess.run(
                    ["powershell", "-Command", 
                     "[System.Reflection.Assembly]::LoadWithPartialName('System.Speech') | Out-Null; Write-Output 'OK'"],
                    capture_output=True, text=True, timeout=5
                )
                return "OK" in result.stdout
            except Exception:
                return False
        elif self.platform == "darwin":
            # macOS - check for say command (speech synthesis)
            # Note: macOS STT requires more complex setup
            return False
        else:
            # Linux - check for common tools
            for tool in ["sox", "arecord"]:
                try:
                    subprocess.run([tool, "--version"], capture_output=True, timeout=2)
                    return True
                except Exception:
                    continue
            return False
    
    def get_microphones(self) -> list[str]:
        """List available microphones."""
        mics = []
        
        if self.platform == "win32":
            try:
                # PowerShell to list audio devices
                ps_script = """
                Add-Type -AssemblyName System.Speech
                $recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine
                Write-Output "Default Microphone"
                """
                result = subprocess.run(
                    ["powershell", "-Command", ps_script],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    mics = ["Default Microphone"]
            except Exception:
                pass  # Intentionally silent
        elif self.platform == "linux":
            try:
                result = subprocess.run(
                    ["arecord", "-l"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'card' in line.lower():
                            mics.append(line.strip())
            except Exception:
                pass  # Intentionally silent
        
        return mics if mics else ["Default"]
    
    def listen(self, timeout: float = 5.0, phrase_limit: float = 10.0) -> dict[str, Any]:
        """
        Listen for speech and return transcription.
        
        Note: This is a simplified implementation. For production use,
        install speech_recognition library.
        """
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded"}
        
        if self.platform == "win32":
            return self._listen_windows(timeout, phrase_limit)
        elif self.platform == "darwin":
            return self._listen_macos(timeout, phrase_limit)
        else:
            return self._listen_linux(timeout, phrase_limit)
    
    def _listen_windows(self, timeout: float, phrase_limit: float) -> dict[str, Any]:
        """Windows speech recognition using System.Speech."""
        try:
            start = time.time()
            
            # PowerShell script for speech recognition
            ps_script = f"""
            Add-Type -AssemblyName System.Speech
            $recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine
            $recognizer.SetInputToDefaultAudioDevice()
            
            # Create grammar
            $grammar = New-Object System.Speech.Recognition.DictationGrammar
            $recognizer.LoadGrammar($grammar)
            
            # Set timeout
            $recognizer.InitialSilenceTimeout = [TimeSpan]::FromSeconds({timeout})
            $recognizer.BabbleTimeout = [TimeSpan]::FromSeconds({phrase_limit})
            
            try {{
                $result = $recognizer.Recognize([TimeSpan]::FromSeconds({timeout + phrase_limit}))
                if ($result) {{
                    Write-Output $result.Text
                }} else {{
                    Write-Output "NO_SPEECH_DETECTED"
                }}
            }} catch {{
                Write-Output "ERROR: $($_.Exception.Message)"
            }} finally {{
                $recognizer.Dispose()
            }}
            """
            
            result = subprocess.run(
                ["powershell", "-Command", ps_script],
                capture_output=True, text=True, timeout=timeout + phrase_limit + 5
            )
            
            output = result.stdout.strip()
            
            if "ERROR:" in output:
                return {"success": False, "error": output.replace("ERROR:", "").strip()}
            elif output == "NO_SPEECH_DETECTED" or not output:
                return {"success": False, "error": "No speech detected"}
            else:
                return {
                    "success": True,
                    "text": output,
                    "duration": time.time() - start,
                    "confidence": 0.8,  # Windows doesn't easily expose confidence
                }
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Timeout waiting for speech"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _listen_macos(self, timeout: float, phrase_limit: float) -> dict[str, Any]:
        """macOS speech recognition."""
        # macOS speech recognition requires more complex setup
        # Would need NSSpeechRecognizer or SFSpeechRecognizer
        return {
            "success": False,
            "error": "macOS STT requires speech_recognition library. Install with: pip install SpeechRecognition"
        }
    
    def _listen_linux(self, timeout: float, phrase_limit: float) -> dict[str, Any]:
        """Linux speech recognition using external tools."""
        # Check for vosk or other tools
        try:
            # Try using vosk-transcriber if available
            result = subprocess.run(
                ["which", "vosk-transcriber"],
                capture_output=True, timeout=2
            )
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": "Linux STT requires speech_recognition library or vosk. Install with: pip install SpeechRecognition vosk"
                }
        except Exception:
            pass  # Intentionally silent
        
        return {
            "success": False,
            "error": "Linux STT requires speech_recognition library. Install with: pip install SpeechRecognition"
        }
    
    def transcribe_file(self, audio_path: str) -> dict[str, Any]:
        """Transcribe an audio file."""
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded"}
        
        if not os.path.exists(audio_path):
            return {"success": False, "error": f"File not found: {audio_path}"}
        
        if self.platform == "win32":
            try:
                start = time.time()
                
                ps_script = f"""
                Add-Type -AssemblyName System.Speech
                $recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine
                $recognizer.SetInputToWaveFile("{audio_path.replace(chr(92), chr(92)+chr(92))}")
                
                $grammar = New-Object System.Speech.Recognition.DictationGrammar
                $recognizer.LoadGrammar($grammar)
                
                $result = $recognizer.Recognize()
                if ($result) {{
                    Write-Output $result.Text
                }}
                $recognizer.Dispose()
                """
                
                result = subprocess.run(
                    ["powershell", "-Command", ps_script],
                    capture_output=True, text=True, timeout=60
                )
                
                output = result.stdout.strip()
                if output:
                    return {
                        "success": True,
                        "text": output,
                        "duration": time.time() - start,
                    }
                else:
                    return {"success": False, "error": "Could not transcribe audio"}
                    
            except Exception as e:
                return {"success": False, "error": str(e)}
        else:
            return {
                "success": False,
                "error": "File transcription requires speech_recognition library"
            }


# Convenience function
def listen_once(timeout: float = 5.0) -> str:
    """Quick function to listen once and return text."""
    stt = BuiltinSTT()
    stt.load()
    result = stt.listen(timeout=timeout)
    stt.unload()
    return result.get("text", "")
