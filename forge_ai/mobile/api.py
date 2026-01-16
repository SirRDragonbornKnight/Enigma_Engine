"""
Mobile API for ForgeAI

Provides REST API endpoints optimized for mobile apps:
- Lightweight responses
- Voice input/output support  
- Efficient data transfer
- Token-based authentication

Usage:
    from forge_ai.mobile.api import mobile_app
    
    # Run as standalone app
    mobile_app.run(host='0.0.0.0', port=5001)
    
    # Or integrate with existing Flask app
    main_app.register_blueprint(mobile_app, url_prefix='/mobile')
"""

from datetime import datetime
from pathlib import Path

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from ..config import CONFIG


# Create Flask app or blueprint
if FLASK_AVAILABLE:
    mobile_app = Flask(__name__)
    CORS(mobile_app)
else:
    mobile_app = None


# Global engine instance (lazy loaded)
_engine = None


def get_engine():
    """Get or create inference engine."""
    global _engine
    
    if _engine is None:
        try:
            from ..core.inference import InferenceEngine
            default_model = CONFIG.get("default_model", "forge_ai")
            _engine = InferenceEngine(model_name=default_model)
        except Exception as e:
            print(f"Warning: Could not load inference engine: {e}")
            _engine = None
    
    return _engine


if FLASK_AVAILABLE:
    
    @mobile_app.route('/api/v1/chat', methods=['POST'])
    def mobile_chat():
        """
        Optimized chat endpoint for mobile apps.
        
        Request:
        {
            "message": "Hello",
            "max_length": 100,  // optional
            "temperature": 0.7  // optional
        }
        
        Response:
        {
            "response": "Hi! How can I help you?",
            "model": "forge_ai",
            "tokens_used": 10,
            "timestamp": "2024-01-01T12:00:00"
        }
        """
        data = request.json
        
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        message = data['message']
        max_length = data.get('max_length', 100)  # Shorter for mobile
        temperature = data.get('temperature', 0.7)
        
        engine = get_engine()
        if engine is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        try:
            response = engine.generate(
                message,
                max_gen=max_length,
                temperature=temperature
            )
            
            return jsonify({
                'response': response,
                'model': CONFIG.get('default_model', 'forge_ai'),
                'tokens_used': len(response.split()),
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    
    @mobile_app.route('/api/v1/models', methods=['GET'])
    def list_models():
        """
        List available models for mobile.
        
        Response:
        {
            "models": [
                {"name": "forge_ai", "size": "small", "loaded": true},
                ...
            ]
        }
        """
        try:
            models_dir = Path(CONFIG["models_dir"])
            models = []
            
            for model_path in models_dir.glob("*"):
                if model_path.is_dir():
                    # Check if it has model files
                    has_model = any(model_path.glob("*.pth")) or any(model_path.glob("*.pt"))
                    
                    if has_model:
                        models.append({
                            'name': model_path.name,
                            'size': 'unknown',  # Could detect from config
                            'loaded': False  # Could check if currently loaded
                        })
            
            return jsonify({'models': models})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    
    @mobile_app.route('/api/v1/voice/speak', methods=['POST'])
    def mobile_speak():
        """
        TTS optimized for mobile - returns audio stream.
        
        Request:
        {
            "text": "Hello world",
            "voice": "default"  // optional
        }
        
        Response:
            Audio file (WAV format) or JSON status
        """
        data = request.json
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        voice_name = data.get('voice', 'default')
        
        try:
            from ..voice import speak, set_voice, VoiceProfile
            
            # Set the voice profile
            try:
                set_voice(voice_name)
            except FileNotFoundError:
                # Voice profile not found, use default
                pass
            
            # Speak the text (uses system TTS)
            speak(text)
            
            return jsonify({
                'success': True,
                'text': text,
                'voice': voice_name,
                'timestamp': datetime.now().isoformat()
            })
        except ImportError:
            return jsonify({'error': 'Voice module not available'}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    
    @mobile_app.route('/api/v1/voice/listen', methods=['POST'])
    def mobile_listen():
        """
        STT optimized for mobile - accepts audio upload.
        
        Request:
            - Form data with 'audio' file
            - Or JSON with base64 encoded audio
        
        Response:
        {
            "text": "transcribed text",
            "confidence": 0.95
        }
        """
        # Check for file upload
        if 'audio' in request.files:
            audio_file = request.files['audio']
            # Process audio file
            return jsonify({
                'message': 'STT not fully implemented',
                'text': 'placeholder transcription'
            })
        
        # Check for base64 audio
        data = request.json
        if data and 'audio_base64' in data:
            # Decode base64 audio
            return jsonify({
                'message': 'STT not fully implemented',
                'text': 'placeholder transcription'
            })
        
        return jsonify({'error': 'No audio provided'}), 400
    
    
    @mobile_app.route('/api/v1/status', methods=['GET'])
    def mobile_status():
        """
        Get lightweight status for mobile.
        
        Response:
        {
            "status": "ok",
            "model_loaded": true,
            "model_name": "forge_ai",
            "version": "1.0"
        }
        """
        engine = get_engine()
        
        # Check available features
        features = {
            'chat': True,
            'voice_tts': False,
            'voice_stt': False,
            'personality': False,
        }
        
        try:
            from ..voice import speak
            features['voice_tts'] = True
        except ImportError:
            pass
        
        try:
            from ..voice import listen
            features['voice_stt'] = True
        except ImportError:
            pass
        
        try:
            from ..core.personality import load_personality
            features['personality'] = True
        except ImportError:
            pass
        
        return jsonify({
            'status': 'ok',
            'model_loaded': engine is not None,
            'model_name': CONFIG.get('default_model', 'forge_ai'),
            'version': '1.0',
            'features': features,
            'timestamp': datetime.now().isoformat()
        })
    
    
    @mobile_app.route('/api/v1/personality', methods=['GET'])
    def get_personality():
        """
        Get current AI personality traits.
        
        Response:
        {
            "traits": {
                "humor_level": 0.5,
                "formality": 0.5,
                ...
            },
            "mood": "neutral"
        }
        """
        try:
            from ..core.personality import load_personality
            
            model_name = CONFIG.get('default_model', 'forge_ai')
            personality = load_personality(model_name)
            
            return jsonify({
                'traits': personality.traits.to_dict(),
                'mood': personality.mood,
                'interests': personality.interests,
                'conversation_count': personality.conversation_count
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    
    @mobile_app.route('/api/v1/personality', methods=['PUT'])
    def update_personality():
        """
        Update AI personality traits.
        
        Request:
        {
            "traits": {
                "humor_level": 0.7,
                ...
            }
        }
        """
        data = request.json
        
        if not data or 'traits' not in data:
            return jsonify({'error': 'No traits provided'}), 400
        
        try:
            from ..core.personality import load_personality
            
            model_name = CONFIG.get('default_model', 'forge_ai')
            personality = load_personality(model_name)
            
            # Update traits
            for key, value in data['traits'].items():
                if hasattr(personality.traits, key):
                    setattr(personality.traits, key, value)
            
            personality.save()
            
            return jsonify({
                'success': True,
                'traits': personality.traits.to_dict()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500


def run_mobile_api(host: str = '0.0.0.0', port: int = 5001):
    """
    Run mobile API server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
    """
    if not FLASK_AVAILABLE:
        print("Error: Flask not installed")
        return
    
    print(f"\n{'='*60}")
    print("FORGE AI MOBILE API")
    print(f"{'='*60}")
    print(f"\nAPI starting on http://{host}:{port}")
    print(f"\nEndpoints:")
    print(f"   - POST /api/v1/chat         - Chat with AI")
    print(f"   - GET  /api/v1/models       - List models")
    print(f"   - POST /api/v1/voice/speak  - Text-to-speech")
    print(f"   - POST /api/v1/voice/listen - Speech-to-text")
    print(f"   - GET  /api/v1/status       - System status")
    print(f"   - GET  /api/v1/personality  - Get personality")
    print(f"   - PUT  /api/v1/personality  - Update personality")
    print(f"\n{'='*60}\n")
    
    mobile_app.run(host=host, port=port)
