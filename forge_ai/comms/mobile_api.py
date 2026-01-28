"""
Mobile App API for Forge

Provides a REST API specifically designed for mobile apps:
  - Lightweight responses (optimized for mobile bandwidth)
  - WebSocket support for real-time chat
  - Voice input/output endpoints
  - Screen-free interaction mode

Works with: iOS, Android (native or React Native/Flutter)
"""

import base64
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

try:
    from flask import Flask, request, jsonify, Response
    from flask_cors import CORS
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False



class MobileAPI:
    """
    Mobile-optimized API server for Forge.
    
    Features:
      - Compact JSON responses
      - Voice input (base64 audio)
      - Voice output (TTS)
      - Conversation context
      - Lightweight status checks
    """
    
    def __init__(self, port: int = 5000, model_name: str = None):
        if not HAS_FLASK:
            raise ImportError("Flask required. Install with: pip install flask flask-cors")
        
        self.port = port
        self.model_name = model_name
        self.app = Flask("forge_mobile")
        CORS(self.app)
        
        # Engine (lazy loaded)
        self._engine = None
        self._voice = None
        
        # Conversation context per device
        self._contexts: Dict[str, list] = {}
        
        # Setup routes
        self._setup_routes()
    
    @property
    def engine(self):
        """Lazy load inference engine."""
        if self._engine is None:
            if self.model_name:
                from ..core.model_registry import ModelRegistry
                from ..core.inference import ForgeEngine
                
                registry = ModelRegistry()
                model, config = registry.load_model(self.model_name)
                
                self._engine = ForgeEngine.__new__(ForgeEngine)
                self._engine.device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
                self._engine.model = model
                self._engine.model.to(self._engine.device)
                self._engine.model.eval()
                from ..core.tokenizer import load_tokenizer
                self._engine.tokenizer = load_tokenizer()
            else:
                from ..core.inference import ForgeEngine
                self._engine = ForgeEngine()
        return self._engine
    
    @property
    def voice(self):
        """Lazy load voice system."""
        if self._voice is None:
            try:
                from ..voice import speak
                # Create simple wrapper object
                class SimpleVoice:
                    def speak(self, text):
                        speak(text)
                    
                    def save_to_file(self, text, filepath):
                        """Save TTS output to file with multiple engine fallbacks."""
                        import tempfile
                        import os
                        
                        # Try pyttsx3 first (most common)
                        try:
                            import pyttsx3
                            engine = pyttsx3.init()
                            engine.save_to_file(text, filepath)
                            engine.runAndWait()
                            return True
                        except Exception:
                            pass
                        
                        # Try gTTS (Google TTS) - outputs MP3
                        try:
                            from gtts import gTTS
                            tts = gTTS(text=text, lang='en')
                            # gTTS outputs MP3, convert if needed
                            if filepath.endswith('.wav'):
                                mp3_path = filepath.replace('.wav', '.mp3')
                                tts.save(mp3_path)
                                # Try to convert with pydub
                                try:
                                    from pydub import AudioSegment
                                    audio = AudioSegment.from_mp3(mp3_path)
                                    audio.export(filepath, format='wav')
                                    os.remove(mp3_path)
                                except ImportError:
                                    # pydub not available, keep as MP3
                                    os.rename(mp3_path, filepath)
                            else:
                                tts.save(filepath)
                            return True
                        except Exception:
                            pass
                        
                        # Try edge-tts (Microsoft Edge TTS)
                        try:
                            import asyncio
                            import edge_tts
                            
                            async def save_edge_tts():
                                communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
                                await communicate.save(filepath)
                            
                            asyncio.run(save_edge_tts())
                            return True
                        except Exception:
                            pass
                        
                        # No TTS engine available
                        raise RuntimeError(
                            "No TTS engine available for save_to_file. "
                            "Install one of: pyttsx3, gtts, edge-tts"
                        )
                
                self._voice = SimpleVoice()
            except ImportError as e:
                print(f"[MobileAPI] Voice not available: {e}")
                self._voice = None
        return self._voice
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.route("/")
        def index():
            return jsonify({
                "name": "Forge Mobile API",
                "version": "1.0",
                "endpoints": [
                    "GET /status - Server status",
                    "POST /chat - Send message, get response",
                    "POST /voice/input - Send voice audio",
                    "GET /voice/output/<text> - Get TTS audio",
                    "POST /context/clear - Clear conversation context",
                ]
            })
        
        @self.app.route("/status")
        def status():
            """Lightweight status check with device-aware info."""
            # Use device profiles for better hardware detection
            try:
                from ..core.device_profiles import get_device_profiler
                profiler = get_device_profiler()
                device_class = profiler.classify()
                caps = profiler.detect()
                profile = profiler.get_profile()
                
                return jsonify({
                    "ok": True,
                    "model": self.model_name or "default",
                    "device_class": device_class.name,
                    "device": profiler.get_torch_device(),
                    "recommended_size": profile.recommended_model_size,
                    "hardware": {
                        "cpu_cores": caps.cpu_cores,
                        "ram_mb": caps.ram_total_mb,
                        "has_gpu": caps.has_cuda or caps.has_mps,
                        "gpu_name": caps.gpu_name if caps.has_cuda else None,
                        "vram_mb": caps.vram_total_mb if caps.has_cuda else None,
                    },
                    "capabilities": {
                        "can_serve_inference": profile.can_serve_inference,
                        "can_serve_training": profile.can_serve_training,
                        "use_quantization": profile.use_quantization,
                        "max_batch_size": profile.max_batch_size,
                        "max_sequence_length": profile.max_sequence_length,
                    }
                })
            except ImportError:
                # Fallback to old hardware module
                try:
                    from ..core.hardware import get_hardware
                    hw = get_hardware()
                    return jsonify({
                        "ok": True,
                        "model": self.model_name or "default",
                        "device": hw.get_device(),
                        "recommended_size": hw.profile.get("recommended_model_size", "tiny"),
                    })
                except ImportError:
                    return jsonify({
                        "ok": True,
                        "model": self.model_name or "default",
                        "device": "cpu",
                        "recommended_size": "tiny",
                    })
        
        @self.app.route("/chat", methods=["POST"])
        def chat():
            """
            Main chat endpoint.
            
            Input JSON:
                {
                    "message": "Hello",
                    "device_id": "phone_123",  // Optional for context
                    "max_length": 50,  // Optional
                    "include_context": true  // Include conversation history
                }
            
            Output JSON:
                {
                    "response": "Hi there!",
                    "tokens": 3,
                    "time_ms": 150
                }
            """
            data = request.json or {}
            message = data.get("message", "")
            device_id = data.get("device_id", "default")
            max_length = int(data.get("max_length", 50))
            include_context = data.get("include_context", True)
            
            if not message:
                return jsonify({"error": "No message provided"}), 400
            
            # Build prompt with context
            if include_context and device_id in self._contexts:
                context = self._contexts[device_id][-5:]  # Last 5 exchanges
                prompt = "\n".join([
                    f"User: {c['user']}\nAssistant: {c['assistant']}"
                    for c in context
                ]) + f"\nUser: {message}\nAssistant:"
            else:
                prompt = message
            
            # Generate
            start = time.time()
            response = self.engine.generate(prompt, max_gen=max_length)
            elapsed = int((time.time() - start) * 1000)
            
            # Clean response
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            # Store context
            if device_id not in self._contexts:
                self._contexts[device_id] = []
            self._contexts[device_id].append({
                "user": message,
                "assistant": response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep context limited
            if len(self._contexts[device_id]) > 20:
                self._contexts[device_id] = self._contexts[device_id][-20:]
            
            return jsonify({
                "response": response,
                "tokens": len(response.split()),
                "time_ms": elapsed,
            })
        
        @self.app.route("/voice/input", methods=["POST"])
        def voice_input():
            """
            Voice input endpoint.
            
            Accepts base64-encoded audio, returns transcription + response.
            
            Input JSON:
                {
                    "audio": "base64_encoded_audio",
                    "format": "wav",  // or "mp3", "m4a"
                    "device_id": "phone_123"
                }
            """
            data = request.json or {}
            audio_b64 = data.get("audio", "")
            audio_format = data.get("format", "wav")
            device_id = data.get("device_id", "default")
            
            if not audio_b64:
                return jsonify({"error": "No audio provided"}), 400
            
            # Decode audio
            try:
                audio_data = base64.b64decode(audio_b64)
            except Exception as e:
                return jsonify({"error": f"Invalid base64 audio: {e}"}), 400
            
            # Save temp file and transcribe
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as f:
                f.write(audio_data)
                temp_path = f.name
            
            try:
                # Try speech recognition
                transcription = self._transcribe(temp_path)
                
                if transcription:
                    # Generate response
                    response = self.engine.generate(transcription, max_gen=50)
                    
                    return jsonify({
                        "transcription": transcription,
                        "response": response,
                    })
                else:
                    return jsonify({"error": "Could not transcribe audio"}), 400
            finally:
                Path(temp_path).unlink(missing_ok=True)
        
        @self.app.route("/voice/output")
        def voice_output():
            """
            TTS endpoint - returns audio for text.
            
            Query params:
                text: Text to speak
                format: Output format (wav, mp3)
            """
            text = request.args.get("text", "")
            out_format = request.args.get("format", "wav")
            
            if not text:
                return jsonify({"error": "No text provided"}), 400
            
            if self.voice is None:
                return jsonify({"error": "Voice system not available"}), 500
            
            # Generate audio
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=f".{out_format}", delete=False) as f:
                temp_path = f.name
            
            try:
                self.voice.save_to_file(text, temp_path)
                
                with open(temp_path, "rb") as f:
                    audio_data = f.read()
                
                return Response(
                    audio_data,
                    mimetype=f"audio/{out_format}",
                    headers={
                        "Content-Disposition": f"attachment; filename=speech.{out_format}"
                    }
                )
            finally:
                Path(temp_path).unlink(missing_ok=True)
        
        @self.app.route("/context/clear", methods=["POST"])
        def clear_context():
            """Clear conversation context for a device."""
            data = request.json or {}
            device_id = data.get("device_id", "default")
            
            if device_id in self._contexts:
                del self._contexts[device_id]
            
            return jsonify({"cleared": True})
        
        @self.app.route("/context/get")
        def get_context():
            """Get conversation context."""
            device_id = request.args.get("device_id", "default")
            context = self._contexts.get(device_id, [])
            return jsonify({"device_id": device_id, "context": context})
    
    def _transcribe(self, audio_path: str) -> Optional[str]:
        """Transcribe audio file to text."""
        # Try speech_recognition
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            
            with sr.AudioFile(audio_path) as source:
                audio = recognizer.record(source)
            
            # Try Google (requires internet)
            try:
                return recognizer.recognize_google(audio)
            except Exception:
                pass
            
            # Try offline (if available)
            try:
                return recognizer.recognize_sphinx(audio)
            except Exception:
                pass
        except ImportError:
            pass
        
        # Try whisper (if available)
        try:
            import whisper
            model = whisper.load_model("tiny")
            result = model.transcribe(audio_path)
            return result["text"]
        except Exception:
            pass
        
        return None
    
    def run(self, host: str = "0.0.0.0", debug: bool = False):
        """Start the API server."""
        print(f"Starting Mobile API on {host}:{self.port}")
        print("Endpoints:")
        print("  /status - Server status")
        print("  /chat - Chat with AI")
        print("  /voice/input - Voice input")
        print("  /voice/output - TTS output")
        self.app.run(host=host, port=self.port, debug=debug)


def create_mobile_api(port: int = 5000, model_name: str = None) -> MobileAPI:
    """Create and return a mobile API instance."""
    return MobileAPI(port=port, model_name=model_name)


# Simple Flutter/React Native code templates
MOBILE_CLIENT_TEMPLATES = {
    "flutter": r'''
// Flutter client for Forge Mobile API

import 'dart:convert';
import 'package:http/http.dart' as http;

class ForgeClient {
  final String baseUrl;
  final String deviceId;
  
  ForgeClient({required this.baseUrl, String? deviceId})
      : deviceId = deviceId ?? 'flutter_app';
  
  Future<String> chat(String message) async {
    final response = await http.post(
      Uri.parse('$baseUrl/chat'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'message': message,
        'device_id': deviceId,
      }),
    );
    
    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['response'];
    }
    throw Exception('Failed to get response');
  }
  
  Future<Map<String, dynamic>> status() async {
    final response = await http.get(Uri.parse('$baseUrl/status'));
    return jsonDecode(response.body);
  }
}
''',
    
    "react_native": '''
// React Native client for Forge Mobile API

const FORGE_URL = 'http://YOUR_SERVER:5000';
const DEVICE_ID = 'react_native_app';

export async function chat(message) {
  const response = await fetch(`${FORGE_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      device_id: DEVICE_ID,
    }),
  });
  
  const data = await response.json();
  return data.response;
}

export async function getStatus() {
  const response = await fetch(`${FORGE_URL}/status`);
  return response.json();
}

export async function clearContext() {
  await fetch(`${FORGE_URL}/context/clear`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ device_id: DEVICE_ID }),
  });
}
''',
}


def print_mobile_client_template(framework: str = "flutter"):
    """Print client code template for mobile frameworks."""
    if framework in MOBILE_CLIENT_TEMPLATES:
        print(MOBILE_CLIENT_TEMPLATES[framework])
    else:
        print(f"Unknown framework: {framework}")
        print(f"Available: {list(MOBILE_CLIENT_TEMPLATES.keys())}")
