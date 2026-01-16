"""
Built-in Text-to-Speech (TTS)

Works without any external dependencies using:
1. Windows SAPI (via PowerShell)
2. macOS say command
3. Linux espeak
4. Fallback: Save text to file
"""

import os
import sys
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional


class BuiltinTTS:
    """
    Built-in TTS that works without pyttsx3 or other libraries.
    Uses system speech APIs via subprocess.
    """
    
    def __init__(self):
        self.is_loaded = False
        self.platform = sys.platform
        self.rate = 150  # Words per minute (approximate)
        self.volume = 1.0
        self.voice_index = 0
        self._voices: List[str] = []
        self._output_dir = Path(tempfile.gettempdir()) / "forge_tts"
        
    def load(self) -> bool:
        """Initialize TTS - always succeeds since we use system commands."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._detect_voices()
        self.is_loaded = True
        return True
    
    def unload(self):
        """Cleanup."""
        self.is_loaded = False
    
    def _detect_voices(self):
        """Detect available system voices."""
        self._voices = []
        
        if self.platform == 'win32':
            # Windows - query SAPI voices
            try:
                ps_script = '''
                Add-Type -AssemblyName System.Speech
                $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
                $synth.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }
                '''
                result = subprocess.run(
                    ['powershell', '-Command', ps_script],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    self._voices = [v.strip() for v in result.stdout.strip().split('\n') if v.strip()]
            except Exception:
                pass
            
            if not self._voices:
                self._voices = ['Microsoft David', 'Microsoft Zira', 'Default']
                
        elif self.platform == 'darwin':
            # macOS - use 'say -v ?' to list voices
            try:
                result = subprocess.run(['say', '-v', '?'], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if line.strip():
                            voice = line.split()[0]
                            self._voices.append(voice)
            except Exception:
                self._voices = ['Alex', 'Samantha', 'Victoria']
                
        else:
            # Linux - check for espeak voices
            try:
                result = subprocess.run(['espeak', '--voices'], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    for line in result.stdout.split('\n')[1:]:
                        parts = line.split()
                        if len(parts) >= 4:
                            self._voices.append(parts[3])
            except Exception:
                self._voices = ['default', 'en', 'en-us']
        
        if not self._voices:
            self._voices = ['Default']
    
    def get_voices(self) -> List[str]:
        """Return available voices."""
        return self._voices
    
    def set_voice(self, index: int):
        """Set voice by index."""
        if 0 <= index < len(self._voices):
            self.voice_index = index
    
    def set_rate(self, rate: int):
        """Set speech rate (words per minute)."""
        self.rate = max(50, min(400, rate))
    
    def set_volume(self, volume: float):
        """Set volume (0.0 to 1.0)."""
        self.volume = max(0.0, min(1.0, volume))
    
    def speak(self, text: str) -> Dict[str, Any]:
        """Speak text directly without saving to file."""
        if not self.is_loaded:
            return {"success": False, "error": "TTS not loaded"}
        
        if not text.strip():
            return {"success": False, "error": "Empty text"}
        
        try:
            # Sanitize text for command line
            safe_text = text.replace('"', '\\"').replace("'", "\\'")
            
            if self.platform == 'win32':
                # Windows - use PowerShell with SAPI
                voice = self._voices[self.voice_index] if self._voices else 'Default'
                # Rate: SAPI uses -10 to 10, we convert from WPM
                sapi_rate = int((self.rate - 150) / 15)  # Map 50-400 WPM to -7 to +17
                sapi_rate = max(-10, min(10, sapi_rate))
                
                ps_script = f'''
                Add-Type -AssemblyName System.Speech
                $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
                $synth.Rate = {sapi_rate}
                $synth.Volume = {int(self.volume * 100)}
                $synth.Speak("{safe_text}")
                '''
                subprocess.run(['powershell', '-Command', ps_script], timeout=60)
                
            elif self.platform == 'darwin':
                # macOS - use 'say' command
                voice = self._voices[self.voice_index] if self._voices else 'Alex'
                # Rate is in words per minute
                subprocess.run(['say', '-v', voice, '-r', str(self.rate), safe_text], timeout=60)
                
            else:
                # Linux - use espeak
                voice = self._voices[self.voice_index] if self._voices else 'en'
                # espeak uses speed in words per minute
                amplitude = int(self.volume * 200)  # espeak amplitude 0-200
                subprocess.run([
                    'espeak', '-v', voice, '-s', str(self.rate), 
                    '-a', str(amplitude), safe_text
                ], timeout=60)
            
            return {"success": True}
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Speech timed out"}
        except FileNotFoundError as e:
            return {"success": False, "error": f"TTS command not found: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def generate(self, text: str, **kwargs) -> Dict[str, Any]:
        """Generate audio file from text."""
        if not self.is_loaded:
            return {"success": False, "error": "TTS not loaded"}
        
        if not text.strip():
            return {"success": False, "error": "Empty text"}
        
        try:
            start = time.time()
            timestamp = int(time.time())
            
            safe_text = text.replace('"', '\\"').replace("'", "\\'")
            
            if self.platform == 'win32':
                # Windows - save to WAV using SAPI
                filename = f"tts_{timestamp}.wav"
                filepath = self._output_dir / filename
                
                voice = self._voices[self.voice_index] if self._voices else 'Default'
                sapi_rate = int((self.rate - 150) / 15)
                sapi_rate = max(-10, min(10, sapi_rate))
                
                ps_script = f'''
                Add-Type -AssemblyName System.Speech
                $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
                $synth.Rate = {sapi_rate}
                $synth.Volume = {int(self.volume * 100)}
                $synth.SetOutputToWaveFile("{filepath}")
                $synth.Speak("{safe_text}")
                $synth.SetOutputToDefaultAudioDevice()
                '''
                result = subprocess.run(['powershell', '-Command', ps_script], 
                                        capture_output=True, timeout=60)
                
                if filepath.exists():
                    return {
                        "success": True,
                        "path": str(filepath),
                        "duration": time.time() - start
                    }
                else:
                    return {"success": False, "error": "Failed to generate audio file"}
                
            elif self.platform == 'darwin':
                # macOS - use 'say' to generate AIFF, then convert if needed
                filename = f"tts_{timestamp}.aiff"
                filepath = self._output_dir / filename
                
                voice = self._voices[self.voice_index] if self._voices else 'Alex'
                subprocess.run([
                    'say', '-v', voice, '-r', str(self.rate),
                    '-o', str(filepath), safe_text
                ], timeout=60)
                
                if filepath.exists():
                    return {
                        "success": True,
                        "path": str(filepath),
                        "duration": time.time() - start
                    }
                else:
                    return {"success": False, "error": "Failed to generate audio file"}
                
            else:
                # Linux - use espeak to generate WAV
                filename = f"tts_{timestamp}.wav"
                filepath = self._output_dir / filename
                
                voice = self._voices[self.voice_index] if self._voices else 'en'
                amplitude = int(self.volume * 200)
                subprocess.run([
                    'espeak', '-v', voice, '-s', str(self.rate),
                    '-a', str(amplitude), '-w', str(filepath), safe_text
                ], timeout=60)
                
                if filepath.exists():
                    return {
                        "success": True,
                        "path": str(filepath),
                        "duration": time.time() - start
                    }
                else:
                    return {"success": False, "error": "Failed to generate audio file"}
                    
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Audio generation timed out"}
        except FileNotFoundError as e:
            return {"success": False, "error": f"TTS command not found: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
