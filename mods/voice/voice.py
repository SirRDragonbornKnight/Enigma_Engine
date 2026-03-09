#!/usr/bin/env python3
"""
Voice - Standalone Voice Input/Output Service

Speech-to-text and text-to-speech using:
- Whisper STT (local)
- Speech Recognition (Google/Sphinx)
- pyttsx3 TTS (local)
- System speech fallback

Usage:
    python voice.py                    # Start service
    python voice.py --port 9907       # Custom port
    python voice.py --listen          # Listen for speech
    python voice.py --speak "Hello"  # Speak text
"""

import argparse
import json
import logging
import os
import socket
import struct
import threading
import time
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/voice")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Protocol
# =============================================================================

class MessageType(Enum):
    REGISTER = "register"
    COMMAND = "command"
    RESPONSE = "response"
    EVENT = "event"
    HEARTBEAT = "heartbeat"
    SHUTDOWN = "shutdown"

@dataclass
class Message:
    type: MessageType
    payload: Dict[str, Any] = field(default_factory=dict)
    id: str = ""
    
    def to_bytes(self) -> bytes:
        data = json.dumps({
            "type": self.type.value,
            "payload": self.payload,
            "id": self.id
        }).encode('utf-8')
        return struct.pack('>I', len(data)) + data
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'Message':
        obj = json.loads(data.decode('utf-8'))
        return cls(
            type=MessageType(obj["type"]),
            payload=obj.get("payload", {}),
            id=obj.get("id", "")
        )


# =============================================================================
# Speech-to-Text Providers
# =============================================================================

class WhisperSTT:
    """Local speech-to-text using Whisper."""
    
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.model = None
        self.is_loaded = False
    
    def load(self) -> bool:
        try:
            import whisper
            logger.info(f"Loading Whisper {self.model_size}...")
            self.model = whisper.load_model(self.model_size)
            self.is_loaded = True
            logger.info("Whisper loaded")
            return True
        except ImportError:
            logger.warning("Install: pip install openai-whisper")
        except Exception as e:
            logger.warning(f"Whisper failed: {e}")
        return False
    
    def unload(self):
        if self.model:
            del self.model
            self.model = None
        self.is_loaded = False
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded"}
        
        try:
            start = time.time()
            options = {"language": language} if language else {}
            result = self.model.transcribe(audio_path, **options)
            
            return {
                "success": True,
                "text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "segments": [{"start": s["start"], "end": s["end"], "text": s["text"]}
                           for s in result.get("segments", [])],
                "duration": time.time() - start
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class SpeechRecognitionSTT:
    """Speech recognition using Google/Sphinx."""
    
    def __init__(self, engine: str = "google"):
        self.engine = engine
        self.recognizer = None
        self.is_loaded = False
    
    def load(self) -> bool:
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.is_loaded = True
            logger.info(f"Speech recognition loaded ({self.engine})")
            return True
        except ImportError:
            logger.warning("Install: pip install SpeechRecognition")
        return False
    
    def unload(self):
        self.recognizer = None
        self.is_loaded = False
    
    def transcribe(self, audio_path: str, language: str = "en-US") -> Dict[str, Any]:
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded"}
        
        try:
            import speech_recognition as sr
            
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
            
            start = time.time()
            
            if self.engine == "google":
                text = self.recognizer.recognize_google(audio, language=language)
            elif self.engine == "sphinx":
                text = self.recognizer.recognize_sphinx(audio)
            else:
                return {"success": False, "error": f"Unknown engine: {self.engine}"}
            
            return {
                "success": True,
                "text": text,
                "duration": time.time() - start
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def listen_microphone(self, timeout: float = 5.0) -> Dict[str, Any]:
        """Listen from microphone."""
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded"}
        
        try:
            import speech_recognition as sr
            
            with sr.Microphone() as source:
                logger.info("Listening...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout)
            
            if self.engine == "google":
                text = self.recognizer.recognize_google(audio)
            else:
                text = self.recognizer.recognize_sphinx(audio)
            
            return {"success": True, "text": text}
        except Exception as e:
            return {"success": False, "error": str(e)}


# =============================================================================
# Text-to-Speech Providers
# =============================================================================

class Pyttsx3TTS:
    """Local TTS using pyttsx3."""
    
    def __init__(self):
        self.engine = None
        self.is_loaded = False
        self.voices = []
        self.rate = 150
        self.volume = 1.0
    
    def load(self) -> bool:
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.voices = self.engine.getProperty('voices')
            self.is_loaded = True
            logger.info("pyttsx3 TTS loaded")
            return True
        except ImportError:
            logger.warning("Install: pip install pyttsx3")
        except Exception as e:
            logger.warning(f"pyttsx3 failed: {e}")
        return False
    
    def unload(self):
        if self.engine:
            try:
                self.engine.stop()
            except Exception:
                pass
            self.engine = None
        self.is_loaded = False
    
    def get_voices(self) -> List[str]:
        return [v.name for v in self.voices] if self.is_loaded else []
    
    def set_voice(self, index: int):
        if self.is_loaded and 0 <= index < len(self.voices):
            self.engine.setProperty('voice', self.voices[index].id)
    
    def set_rate(self, rate: int):
        if self.is_loaded:
            self.rate = rate
            self.engine.setProperty('rate', rate)
    
    def set_volume(self, volume: float):
        if self.is_loaded:
            self.volume = max(0.0, min(1.0, volume))
            self.engine.setProperty('volume', self.volume)
    
    def speak(self, text: str) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded"}
        try:
            self.engine.say(text)
            self.engine.runAndWait()
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def generate(self, text: str, **kwargs) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded"}
        
        try:
            timestamp = int(time.time())
            filepath = OUTPUT_DIR / f"tts_{timestamp}.wav"
            
            self.engine.save_to_file(text, str(filepath))
            self.engine.runAndWait()
            
            return {"success": True, "path": str(filepath)}
        except Exception as e:
            return {"success": False, "error": str(e)}


class ElevenLabsTTS:
    """ElevenLabs cloud TTS."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        self.client = None
        self.is_loaded = False
        self.voices = []
        self.current_voice = None

    def load(self) -> bool:
        try:
            from elevenlabs import ElevenLabs as EL
            self.client = EL(api_key=self.api_key)

            if not self.api_key:
                logger.warning("ELEVENLABS_API_KEY not set")
                return False

            response = self.client.voices.get_all()
            self.voices = [
                {"id": v.voice_id, "name": v.name}
                for v in response.voices
            ]
            if self.voices:
                self.current_voice = self.voices[0]["id"]

            self.is_loaded = True
            return True
        except ImportError:
            logger.warning("Install: pip install elevenlabs")
        except Exception as e:
            logger.warning(f"ElevenLabs failed: {e}")
        return False

    def unload(self):
        self.client = None
        self.is_loaded = False

    def get_voices(self) -> List[str]:
        return [v["name"] for v in self.voices]

    def set_voice(self, index: int):
        if 0 <= index < len(self.voices):
            self.current_voice = self.voices[index]["id"]

    def speak(self, text: str) -> Dict[str, Any]:
        result = self.generate(text)
        if not result.get("success"):
            return result
        try:
            import platform
            import subprocess
            path = result.get("path", "")
            if not path:
                return result
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                subprocess.run(["afplay", path])
            else:
                subprocess.run(["aplay", path])
        except Exception:
            pass
        return result

    def generate(self, text: str, **kwargs) -> Dict[str, Any]:
        if not self.is_loaded or not self.client:
            return {"success": False, "error": "Not loaded"}

        try:
            start = time.time()
            audio = self.client.generate(
                text=text,
                voice=self.current_voice,
                model="eleven_monolingual_v1"
            )

            timestamp = int(time.time())
            filepath = OUTPUT_DIR / f"eleven_{timestamp}.mp3"
            with open(filepath, "wb") as f:
                for chunk in audio:
                    f.write(chunk)

            return {
                "success": True,
                "path": str(filepath),
                "duration": time.time() - start,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class SystemTTS:
    """System speech using OS features."""
    
    def __init__(self):
        self.is_loaded = False
        self._platform = None
    
    def load(self) -> bool:
        import platform
        self._platform = platform.system()
        self.is_loaded = True
        return True
    
    def unload(self):
        self.is_loaded = False
    
    def speak(self, text: str) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded"}
        
        try:
            import subprocess
            
            if self._platform == "Windows":
                script = f'Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak("{text}")'
                subprocess.run(["powershell", "-Command", script], capture_output=True)
            elif self._platform == "Darwin":
                subprocess.run(["say", text])
            else:
                subprocess.run(["espeak", text])
            
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}


# =============================================================================
# Voice Pipeline
# =============================================================================

class VoicePipeline:
    """Complete voice pipeline for conversations."""
    
    def __init__(self, stt_provider: str = "speech", tts_provider: str = "pyttsx3"):
        self.stt_provider = stt_provider
        self.tts_provider = tts_provider
        self.stt = None
        self.tts = None
        self._tts_instances: Dict[str, Any] = {}
        self.is_loaded = False
        self._listening = False
        self._callback = None
    
    def load(self) -> bool:
        # Load STT
        if self.stt_provider == "whisper":
            self.stt = WhisperSTT()
        else:
            self.stt = SpeechRecognitionSTT()
        
        stt_ok = self.stt.load()
        tts_ok = self.load_tts_provider(self.tts_provider)
        
        self.is_loaded = stt_ok or tts_ok
        return self.is_loaded

    def _make_tts(self, provider: str):
        if provider == "pyttsx3":
            return Pyttsx3TTS()
        if provider == "system":
            return SystemTTS()
        if provider == "elevenlabs":
            return ElevenLabsTTS()
        return None

    def load_tts_provider(self, provider: str) -> bool:
        tts = self._tts_instances.get(provider)
        if tts is None:
            tts = self._make_tts(provider)
            if tts is None:
                return False
            self._tts_instances[provider] = tts

        if not tts.is_loaded and not tts.load():
            return False

        self.tts_provider = provider
        self.tts = tts
        return True

    def unload_tts_provider(self, provider: str):
        tts = self._tts_instances.get(provider)
        if tts and tts.is_loaded:
            tts.unload()
        if self.tts_provider == provider:
            self.tts = None
    
    def unload(self):
        self._listening = False
        if self.stt:
            self.stt.unload()
        for tts in self._tts_instances.values():
            if getattr(tts, "is_loaded", False):
                tts.unload()
        self._tts_instances.clear()
        self.tts = None
        self.is_loaded = False
    
    def listen(self, timeout: float = 5.0) -> Dict[str, Any]:
        """Listen for speech from microphone."""
        if not self.stt or not self.stt.is_loaded:
            return {"success": False, "error": "STT not loaded"}
        
        if hasattr(self.stt, 'listen_microphone'):
            return self.stt.listen_microphone(timeout)
        
        return {"success": False, "error": "Microphone not supported"}
    
    def speak(self, text: str, provider: Optional[str] = None) -> Dict[str, Any]:
        """Speak text."""
        if provider and provider != self.tts_provider:
            if not self.load_tts_provider(provider):
                return {"success": False, "error": f"TTS provider not available: {provider}"}
        if not self.tts or not self.tts.is_loaded:
            return {"success": False, "error": "TTS not loaded"}
        return self.tts.speak(text)
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio file."""
        if not self.stt or not self.stt.is_loaded:
            return {"success": False, "error": "STT not loaded"}
        return self.stt.transcribe(audio_path)
    
    def generate_audio(self, text: str, provider: Optional[str] = None) -> Dict[str, Any]:
        """Generate audio file from text."""
        if provider and provider != self.tts_provider:
            if not self.load_tts_provider(provider):
                return {"success": False, "error": f"TTS provider not available: {provider}"}
        if not self.tts or not self.tts.is_loaded:
            return {"success": False, "error": "TTS not loaded"}
        if hasattr(self.tts, 'generate'):
            return self.tts.generate(text)
        return {"success": False, "error": "Audio generation not supported"}

    def list_tts_providers(self) -> Dict[str, Dict[str, bool]]:
        providers = {"pyttsx3": {"loaded": False},
                     "system": {"loaded": False},
                     "elevenlabs": {"loaded": False}}
        for name in providers:
            inst = self._tts_instances.get(name)
            providers[name]["loaded"] = bool(inst and inst.is_loaded)
        return providers


# =============================================================================
# Voice Service
# =============================================================================

class Voice:
    """Voice Service - STT and TTS unified."""
    
    def __init__(self, stt_provider: str = "speech", tts_provider: str = "pyttsx3"):
        self.pipeline = VoicePipeline(stt_provider, tts_provider)
        self._running = False
        self._socket: Optional[socket.socket] = None
        self._continuous_listen = False
        self._listen_thread = None
        
        self.commands = {
            "listen": self._cmd_listen,
            "speak": self._cmd_speak,
            "transcribe": self._cmd_transcribe,
            "generate_audio": self._cmd_generate_audio,
            "generate": self._cmd_generate_audio,
            "start_continuous": self._cmd_start_continuous,
            "stop_continuous": self._cmd_stop_continuous,
            "set_voice": self._cmd_set_voice,
            "set_rate": self._cmd_set_rate,
            "set_volume": self._cmd_set_volume,
            "list_voices": self._cmd_list_voices,
            "load_provider": self._cmd_load_provider,
            "unload_provider": self._cmd_unload_provider,
            "list_providers": self._cmd_list_providers,
            "set_default": self._cmd_set_default,
            "load": self._cmd_load,
            "status": self._cmd_status,
        }
    
    def _cmd_listen(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.pipeline.is_loaded:
            self.pipeline.load()
        return self.pipeline.listen(params.get("timeout", 5.0))
    
    def _cmd_speak(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.pipeline.is_loaded:
            self.pipeline.load()
        text = params.get("text", "")
        provider = params.get("provider")
        return self.pipeline.speak(text, provider=provider)
    
    def _cmd_transcribe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.pipeline.is_loaded:
            self.pipeline.load()
        audio_path = params.get("audio_path") or params.get("path", "")
        return self.pipeline.transcribe(audio_path)
    
    def _cmd_generate_audio(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.pipeline.is_loaded:
            self.pipeline.load()
        text = params.get("text", "")
        provider = params.get("provider")
        return self.pipeline.generate_audio(text, provider=provider)
    
    def _cmd_start_continuous(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self._continuous_listen:
            return {"success": True, "message": "Already listening"}
        
        self._continuous_listen = True
        self._listen_thread = threading.Thread(
            target=self._continuous_listen_loop,
            daemon=True
        )
        self._listen_thread.start()
        return {"success": True, "message": "Continuous listening started"}
    
    def _cmd_stop_continuous(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._continuous_listen = False
        return {"success": True, "message": "Continuous listening stopped"}
    
    def _continuous_listen_loop(self):
        while self._continuous_listen and self._running:
            result = self._cmd_listen({"timeout": 3.0})
            if result.get("success") and result.get("text"):
                logger.info(f"Heard: {result['text']}")
                # Could emit event here if connected to router
            time.sleep(0.5)
    
    def _cmd_set_voice(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.pipeline.is_loaded:
            self.pipeline.load()
        provider = params.get("provider")
        if provider and provider != self.pipeline.tts_provider:
            if not self.pipeline.load_tts_provider(provider):
                return {"success": False, "error": f"Unknown provider: {provider}"}
        index = params.get("index", 0)
        if self.pipeline.tts and hasattr(self.pipeline.tts, 'set_voice'):
            self.pipeline.tts.set_voice(index)
        return {"success": True}

    def _cmd_set_rate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.pipeline.is_loaded:
            self.pipeline.load()
        rate = params.get("rate", 150)
        if self.pipeline.tts and hasattr(self.pipeline.tts, "set_rate"):
            self.pipeline.tts.set_rate(rate)
        return {"success": True}

    def _cmd_set_volume(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.pipeline.is_loaded:
            self.pipeline.load()
        volume = params.get("volume", 1.0)
        if self.pipeline.tts and hasattr(self.pipeline.tts, "set_volume"):
            self.pipeline.tts.set_volume(volume)
        return {"success": True}
    
    def _cmd_list_voices(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.pipeline.is_loaded:
            self.pipeline.load()
        provider = params.get("provider")
        if provider and provider != self.pipeline.tts_provider:
            if not self.pipeline.load_tts_provider(provider):
                return {"success": False, "error": f"Unknown provider: {provider}"}
        voices = []
        if self.pipeline.tts and hasattr(self.pipeline.tts, 'get_voices'):
            voices = self.pipeline.tts.get_voices()
        return {"success": True, "voices": voices}

    def _cmd_load_provider(self, params: Dict[str, Any]) -> Dict[str, Any]:
        provider = params.get("provider", self.pipeline.tts_provider)
        success = self.pipeline.load_tts_provider(provider)
        if not success:
            return {"success": False, "error": f"Unknown provider: {provider}"}
        return {"success": True, "provider": provider}

    def _cmd_unload_provider(self, params: Dict[str, Any]) -> Dict[str, Any]:
        provider = params.get("provider", self.pipeline.tts_provider)
        self.pipeline.unload_tts_provider(provider)
        return {"success": True, "provider": provider}

    def _cmd_list_providers(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "providers": self.pipeline.list_tts_providers(),
            "default": self.pipeline.tts_provider,
        }

    def _cmd_set_default(self, params: Dict[str, Any]) -> Dict[str, Any]:
        provider = params.get("provider", "")
        if provider not in ("pyttsx3", "system", "elevenlabs"):
            return {"success": False, "error": f"Unknown provider: {provider}"}
        self.pipeline.tts_provider = provider
        return {"success": True, "default": provider}
    
    def _cmd_load(self, params: Dict[str, Any]) -> Dict[str, Any]:
        success = self.pipeline.load()
        return {"success": success}
    
    def _cmd_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "service": "voice",
            "loaded": self.pipeline.is_loaded,
            "stt_loaded": self.pipeline.stt.is_loaded if self.pipeline.stt else False,
            "tts_loaded": self.pipeline.tts.is_loaded if self.pipeline.tts else False,
            "default_provider": self.pipeline.tts_provider,
            "providers": self.pipeline.list_tts_providers(),
            "continuous_listen": self._continuous_listen
        }
    
    def handle_command(self, cmd: str, params: Dict[str, Any]) -> Dict[str, Any]:
        handler = self.commands.get(cmd)
        if not handler:
            return {"success": False, "error": f"Unknown command: {cmd}"}
        try:
            return handler(params)
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def connect_to_router(self, host: str = "localhost", port: int = 9900):
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((host, port))
            
            reg_msg = Message(
                type=MessageType.REGISTER,
                payload={
                    "name": "voice",
                    "capabilities": ["stt", "tts", "listen", "speak", "transcribe", "audio", "generate_audio"],
                    "commands": list(self.commands.keys())
                }
            )
            self._socket.sendall(reg_msg.to_bytes())
            self._running = True
            logger.info("Connected to router")
            
            while self._running:
                try:
                    len_data = self._socket.recv(4)
                    if not len_data:
                        break
                    msg_len = struct.unpack('>I', len_data)[0]
                    msg_data = b''
                    while len(msg_data) < msg_len:
                        chunk = self._socket.recv(min(4096, msg_len - len(msg_data)))
                        if not chunk:
                            break
                        msg_data += chunk
                    
                    msg = Message.from_bytes(msg_data)
                    if msg.type == MessageType.COMMAND:
                        result = self.handle_command(
                            msg.payload.get("command", ""),
                            msg.payload.get("params", {})
                        )
                        resp = Message(type=MessageType.RESPONSE, payload=result, id=msg.id)
                        self._socket.sendall(resp.to_bytes())
                    elif msg.type == MessageType.SHUTDOWN:
                        self._running = False
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Error: {e}")
                    break
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            if self._socket:
                self._socket.close()
    
    def run_standalone(self, port: int = 9907):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("localhost", port))
        server.listen(5)
        logger.info(f"Voice server listening on port {port}")
        self._running = True
        
        while self._running:
            try:
                server.settimeout(1.0)
                try:
                    client, _ = server.accept()
                except socket.timeout:
                    continue
                threading.Thread(target=self._handle_client, args=(client,), daemon=True).start()
            except Exception as e:
                logger.error(f"Server error: {e}")
        server.close()
    
    def _handle_client(self, client: socket.socket):
        try:
            while self._running:
                len_data = client.recv(4)
                if not len_data:
                    break
                msg_len = struct.unpack('>I', len_data)[0]
                msg_data = b''
                while len(msg_data) < msg_len:
                    chunk = client.recv(min(4096, msg_len - len(msg_data)))
                    if not chunk:
                        break
                    msg_data += chunk
                
                msg = Message.from_bytes(msg_data)
                if msg.type == MessageType.COMMAND:
                    result = self.handle_command(
                        msg.payload.get("command", ""),
                        msg.payload.get("params", {})
                    )
                    resp = Message(type=MessageType.RESPONSE, payload=result, id=msg.id)
                    client.sendall(resp.to_bytes())
        except Exception as e:
            logger.error(f"Client error: {e}")
        finally:
            client.close()
    
    def shutdown(self):
        self._running = False
        self._continuous_listen = False
        self.pipeline.unload()


def main():
    parser = argparse.ArgumentParser(description="Voice Service")
    parser.add_argument("--port", type=int, default=9907)
    parser.add_argument("--router", type=str)
    parser.add_argument("--listen", action="store_true", help="Listen for speech")
    parser.add_argument("--speak", type=str, help="Speak text")
    parser.add_argument("--transcribe", type=str, help="Transcribe audio file")
    parser.add_argument("--stt", type=str, default="speech", choices=["whisper", "speech"])
    parser.add_argument("--tts", type=str, default="pyttsx3", choices=["pyttsx3", "system", "elevenlabs"])
    
    args = parser.parse_args()
    service = Voice(stt_provider=args.stt, tts_provider=args.tts)
    
    if args.listen:
        result = service.handle_command("listen", {})
        print(json.dumps(result, indent=2))
        return
    
    if args.speak:
        result = service.handle_command("speak", {"text": args.speak})
        print(json.dumps(result, indent=2))
        return
    
    if args.transcribe:
        result = service.handle_command("transcribe", {"audio_path": args.transcribe})
        print(json.dumps(result, indent=2))
        return
    
    try:
        if args.router:
            host, port = args.router.split(":")
            service.connect_to_router(host, int(port))
        else:
            service.run_standalone(args.port)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        service.shutdown()


if __name__ == "__main__":
    main()
