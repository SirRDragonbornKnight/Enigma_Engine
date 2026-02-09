"""
Audio Generation Example for Enigma AI Engine

This example shows how to use Enigma AI Engine's audio/TTS capabilities.
Generate speech from text using local or cloud providers.

SUPPORTED PROVIDERS:
- Local: pyttsx3 (offline, no API key)
- ElevenLabs: High-quality AI voices (API key required)
- Edge TTS: Microsoft voices (free, online)

USAGE:
    python examples/audio_example.py
    
Or import in your own code:
    from examples.audio_example import speak, generate_audio
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

AUDIO_OUTPUT_DIR = Path.home() / ".enigma_engine" / "outputs" / "audio"
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# TTS INTERFACES
# =============================================================================

class TTSProvider(ABC):
    """Base class for TTS providers."""
    
    @abstractmethod
    def speak(self, text: str) -> bool:
        """Speak text directly (blocking)."""
    
    @abstractmethod
    def generate(self, text: str, output_path: str) -> bool:
        """Generate audio file."""
    
    @abstractmethod
    def get_voices(self) -> List[Dict[str, str]]:
        """Get available voices."""


# =============================================================================
# LOCAL TTS (pyttsx3)
# =============================================================================

class LocalTTS(TTSProvider):
    """
    Local text-to-speech using pyttsx3.
    
    Works offline, no API key needed!
    Quality is basic but reliable.
    
    Requirements:
        pip install pyttsx3
        
        # Linux may need:
        sudo apt install espeak
    """
    
    def __init__(self):
        self._engine = None
        self._init_engine()
    
    def _init_engine(self):
        """Initialize TTS engine."""
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            print("[TTS] Local engine initialized")
        except ImportError:
            print("[TTS] pyttsx3 not installed:")
            print("  pip install pyttsx3")
        except Exception as e:
            print(f"[TTS] Engine init failed: {e}")
    
    def speak(self, text: str) -> bool:
        """Speak text directly."""
        if not self._engine:
            return False
        
        try:
            self._engine.say(text)
            self._engine.runAndWait()
            return True
        except Exception as e:
            print(f"[TTS] Speak failed: {e}")
            return False
    
    def generate(self, text: str, output_path: str) -> bool:
        """Generate audio file."""
        if not self._engine:
            return False
        
        try:
            self._engine.save_to_file(text, output_path)
            self._engine.runAndWait()
            return Path(output_path).exists()
        except Exception as e:
            print(f"[TTS] Generate failed: {e}")
            return False
    
    def get_voices(self) -> List[Dict[str, str]]:
        """Get available voices."""
        if not self._engine:
            return []
        
        voices = []
        for voice in self._engine.getProperty('voices'):
            voices.append({
                "id": voice.id,
                "name": voice.name,
                "language": getattr(voice, 'languages', ['unknown'])[0] if hasattr(voice, 'languages') else 'unknown',
            })
        return voices
    
    def set_voice(self, voice_id: str):
        """Set voice by ID."""
        if self._engine:
            self._engine.setProperty('voice', voice_id)
    
    def set_rate(self, rate: int = 150):
        """Set speech rate (words per minute)."""
        if self._engine:
            self._engine.setProperty('rate', rate)
    
    def set_volume(self, volume: float = 1.0):
        """Set volume (0.0 to 1.0)."""
        if self._engine:
            self._engine.setProperty('volume', volume)


# =============================================================================
# EDGE TTS (Microsoft)
# =============================================================================

class EdgeTTS(TTSProvider):
    """
    Microsoft Edge TTS - free, high quality!
    
    Requirements:
        pip install edge-tts
    """
    
    def __init__(self, voice: str = "en-US-AriaNeural"):
        self.voice = voice
        self._available = self._check_available()
    
    def _check_available(self) -> bool:
        try:
            import edge_tts
            return True
        except ImportError:
            print("[TTS] edge-tts not installed:")
            print("  pip install edge-tts")
            return False
    
    def speak(self, text: str) -> bool:
        """Speak text (generates temp file and plays)."""
        if not self._available:
            return False
        
        # Generate to temp file
        temp_path = AUDIO_OUTPUT_DIR / "temp_speech.mp3"
        if self.generate(text, str(temp_path)):
            # Play the file
            return self._play_audio(str(temp_path))
        return False
    
    def generate(self, text: str, output_path: str) -> bool:
        """Generate audio file."""
        if not self._available:
            return False
        
        try:
            import asyncio
            import edge_tts
            
            async def _generate():
                communicate = edge_tts.Communicate(text, self.voice)
                await communicate.save(output_path)
            
            asyncio.run(_generate())
            return Path(output_path).exists()
            
        except Exception as e:
            print(f"[TTS] Edge generate failed: {e}")
            return False
    
    def get_voices(self) -> List[Dict[str, str]]:
        """Get available voices."""
        if not self._available:
            return []
        
        try:
            import asyncio
            import edge_tts
            
            async def _get_voices():
                return await edge_tts.list_voices()
            
            voices = asyncio.run(_get_voices())
            return [{"id": v["ShortName"], "name": v["FriendlyName"], 
                     "language": v["Locale"]} for v in voices]
        except:
            return []
    
    def _play_audio(self, path: str) -> bool:
        """Play audio file."""
        try:
            import platform
            import subprocess
            
            system = platform.system()
            
            if system == "Linux":
                subprocess.run(["aplay", path], check=True, capture_output=True, timeout=60)
            elif system == "Darwin":
                subprocess.run(["afplay", path], check=True, timeout=60)
            elif system == "Windows":
                os.startfile(path)
            
            return True
        except:
            return False


# =============================================================================
# ELEVENLABS TTS
# =============================================================================

class ElevenLabsTTS(TTSProvider):
    """
    ElevenLabs - Premium AI voice synthesis.
    
    Features:
        - Ultra-realistic voices
        - Voice cloning
        - Emotion control
    
    Requirements:
        pip install elevenlabs
        
    Get API key at: https://elevenlabs.io
    """
    
    CONFIG_FILE = Path.home() / ".enigma_engine" / "elevenlabs.json"
    
    def __init__(self, api_key: str = None, voice_id: str = None):
        self.api_key = api_key or self._load_api_key()
        self.voice_id = voice_id or "21m00Tcm4TlvDq8ikWAM"  # Default: Rachel
        self._available = self._check_available()
    
    def _load_api_key(self) -> Optional[str]:
        """Load API key from config or environment."""
        # Try environment
        key = os.environ.get("ELEVENLABS_API_KEY")
        if key:
            return key
        
        # Try config file
        if self.CONFIG_FILE.exists():
            import json
            with open(self.CONFIG_FILE) as f:
                config = json.load(f)
                return config.get("api_key")
        
        return None
    
    def save_api_key(self, api_key: str):
        """Save API key for later use."""
        import json
        self.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(self.CONFIG_FILE, 'w') as f:
            json.dump({"api_key": api_key}, f)
        self.api_key = api_key
        print(f"[TTS] API key saved to {self.CONFIG_FILE}")
    
    def _check_available(self) -> bool:
        if not self.api_key:
            print("[TTS] ElevenLabs needs API key:")
            print("  tts = ElevenLabsTTS()")
            print("  tts.save_api_key('your-api-key')")
            return False
        
        try:
            import elevenlabs
            return True
        except ImportError:
            print("[TTS] elevenlabs not installed:")
            print("  pip install elevenlabs")
            return False
    
    def speak(self, text: str) -> bool:
        """Speak text."""
        if not self._available:
            return False
        
        try:
            from elevenlabs import generate, play, set_api_key
            
            set_api_key(self.api_key)
            audio = generate(text=text, voice=self.voice_id)
            play(audio)
            return True
            
        except Exception as e:
            print(f"[TTS] ElevenLabs speak failed: {e}")
            return False
    
    def generate(self, text: str, output_path: str) -> bool:
        """Generate audio file."""
        if not self._available:
            return False
        
        try:
            from elevenlabs import generate, save, set_api_key
            
            set_api_key(self.api_key)
            audio = generate(text=text, voice=self.voice_id)
            save(audio, output_path)
            return True
            
        except Exception as e:
            print(f"[TTS] ElevenLabs generate failed: {e}")
            return False
    
    def get_voices(self) -> List[Dict[str, str]]:
        """Get available voices."""
        if not self._available:
            return []
        
        try:
            from elevenlabs import voices, set_api_key
            
            set_api_key(self.api_key)
            voice_list = voices()
            
            return [{"id": v.voice_id, "name": v.name, 
                     "category": v.category} for v in voice_list]
        except:
            return []


# =============================================================================
# AUDIO CONTROLLER
# =============================================================================

class AudioController:
    """
    Unified audio generation interface.
    Automatically selects best available provider.
    """
    
    def __init__(self, preferred_provider: str = "auto"):
        """
        Args:
            preferred_provider: "local", "edge", "elevenlabs", or "auto"
        """
        self.providers: Dict[str, TTSProvider] = {}
        self._active_provider = None
        
        self._init_providers(preferred_provider)
    
    def _init_providers(self, preferred: str):
        """Initialize available providers."""
        # Try to init all providers
        try:
            self.providers["local"] = LocalTTS()
        except:
            pass
        
        try:
            edge = EdgeTTS()
            if edge._available:
                self.providers["edge"] = edge
        except:
            pass
        
        try:
            eleven = ElevenLabsTTS()
            if eleven._available:
                self.providers["elevenlabs"] = eleven
        except:
            pass
        
        # Select active provider
        if preferred != "auto" and preferred in self.providers:
            self._active_provider = preferred
        elif "edge" in self.providers:
            self._active_provider = "edge"  # Best free option
        elif "elevenlabs" in self.providers:
            self._active_provider = "elevenlabs"
        elif "local" in self.providers:
            self._active_provider = "local"
        
        if self._active_provider:
            print(f"[AUDIO] Using provider: {self._active_provider}")
    
    def speak(self, text: str, provider: str = None) -> bool:
        """Speak text using selected provider."""
        provider = provider or self._active_provider
        if provider and provider in self.providers:
            return self.providers[provider].speak(text)
        return False
    
    def generate(self, text: str, output_path: str = None, 
                 provider: str = None) -> Optional[str]:
        """
        Generate audio file.
        
        Args:
            text: Text to synthesize
            output_path: Output file path (auto-generated if None)
            provider: Provider to use (None for default)
        
        Returns:
            Path to generated file, or None if failed
        """
        provider = provider or self._active_provider
        if not provider or provider not in self.providers:
            return None
        
        # Auto-generate filename if needed
        if output_path is None:
            timestamp = int(time.time())
            output_path = str(AUDIO_OUTPUT_DIR / f"speech_{timestamp}.mp3")
        
        if self.providers[provider].generate(text, output_path):
            return output_path
        return None
    
    def get_voices(self, provider: str = None) -> List[Dict[str, str]]:
        """Get available voices."""
        provider = provider or self._active_provider
        if provider and provider in self.providers:
            return self.providers[provider].get_voices()
        return []
    
    def list_providers(self) -> List[str]:
        """List available providers."""
        return list(self.providers.keys())


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def speak(text: str) -> bool:
    """
    Quick speak function.
    Uses best available TTS provider.
    """
    controller = AudioController()
    return controller.speak(text)


def generate_audio(text: str, output_path: str = None) -> Optional[str]:
    """
    Quick audio generation.
    
    Args:
        text: Text to synthesize
        output_path: Output path (auto-generated if None)
    
    Returns:
        Path to generated file
    """
    controller = AudioController()
    return controller.generate(text, output_path)


def list_voices() -> List[Dict[str, str]]:
    """List all available voices."""
    controller = AudioController()
    return controller.get_voices()


# =============================================================================
# MAIN - Run this file directly to test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Enigma AI Engine Audio Example")
    print("=" * 60)
    
    # Initialize controller
    print("\n[1] Initializing audio system...")
    controller = AudioController()
    
    print(f"Available providers: {controller.list_providers()}")
    print(f"Active provider: {controller._active_provider}")
    
    # List voices
    print("\n[2] Available voices (first 5):")
    voices = controller.get_voices()
    for v in voices[:5]:
        print(f"  - {v.get('name', v.get('id'))}")
    
    # Test speech
    print("\n[3] Testing speech...")
    test_text = "Hello! This is Enigma AI Engine speaking."
    
    if controller._active_provider:
        print(f"Speaking: '{test_text}'")
        success = controller.speak(test_text)
        print(f"Success: {success}")
    else:
        print("No TTS provider available!")
    
    # Test file generation
    print("\n[4] Testing file generation...")
    output = controller.generate("This is a test audio file from Enigma AI Engine.")
    
    if output:
        print(f"Generated: {output}")
        print(f"File size: {Path(output).stat().st_size} bytes")
    else:
        print("Generation failed (no provider available)")
    
    # Test local TTS specifically
    print("\n[5] Testing local TTS...")
    local = LocalTTS()
    if local._engine:
        print("Local TTS voices:")
        for v in local.get_voices()[:3]:
            print(f"  - {v['name']}")
        
        local.set_rate(150)  # Normal speed
        local.speak("Local text to speech is working.")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nInstall providers:")
    print("  pip install pyttsx3     # Local (offline)")
    print("  pip install edge-tts    # Microsoft (free, online)")
    print("  pip install elevenlabs  # Premium (API key needed)")
