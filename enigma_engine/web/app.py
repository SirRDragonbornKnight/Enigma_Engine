"""
Web Dashboard for Enigma AI Engine

Provides a web-based interface for:
- Chat interface
- Model training
- Settings management
- Instance monitoring

Usage:
    from enigma_engine.web.app import run_web
    run_web(host='0.0.0.0', port=8080)
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from flask import Flask, jsonify, render_template, request
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    from flask_socketio import SocketIO, emit
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False

from ..config import CONFIG

# =============================================================================
# Constants
# =============================================================================
DEFAULT_HOST = '0.0.0.0'
DEFAULT_PORT = 8080
DEFAULT_MAX_TOKENS = 200
DEFAULT_TEMPERATURE = 0.7
DEFAULT_PERSONALITY_VALUE = 0.5
BANNER_WIDTH = 60
HTTP_BAD_REQUEST = 400
HTTP_SERVER_ERROR = 500
HTTP_OK = 200


# Initialize Flask app
if FLASK_AVAILABLE:
    # Get template and static directories
    web_dir = Path(__file__).parent
    template_dir = web_dir / "templates"
    static_dir = web_dir / "static"
    
    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir)
    )
    app.config['SECRET_KEY'] = os.urandom(24)
    CORS(app)
    
    if SOCKETIO_AVAILABLE:
        socketio = SocketIO(app, cors_allowed_origins="*")
    else:
        socketio = None
else:
    app = None
    socketio = None


# Global engine instance (lazy loaded)
_engine = None
_model_name = None


def get_engine():
    """Get or create inference engine."""
    global _engine, _model_name
    
    if _engine is None:
        try:
            from ..core.inference import InferenceEngine
            default_model = CONFIG.get("default_model", "enigma_engine")
            _engine = InferenceEngine(model_name=default_model)
            _model_name = default_model
        except Exception as e:
            logger.warning(f"Could not load inference engine: {e}")
            _engine = None
    
    return _engine


# =============================================================================
# Routes
# =============================================================================

if FLASK_AVAILABLE and app is not None:
    @app.route('/')
    def index():
        """Landing page with download links."""
        server_url = request.host_url.rstrip('/')
        return render_template('landing.html', server_url=server_url)


    @app.route('/dashboard')
    def dashboard():
        """Main dashboard (for logged-in users)."""
        return render_template('dashboard.html')


    @app.route('/chat')
    def chat_page():
        """Chat interface."""
        return render_template('chat.html')


    @app.route('/train')
    def train_page():
        """Training interface."""
        return render_template('train.html')


    @app.route('/settings')
    def settings_page():
        """Settings interface."""
        return render_template('settings.html')


    @app.route('/personality')
    def personality_page():
        """Personality dashboard."""
        return render_template('personality.html')


    @app.route('/voice')
    def voice_studio_page():
        """Voice studio interface."""
        return render_template('voice_studio.html')


    @app.route('/memory')
    def memory_page():
        """Memory & learning viewer."""
        return render_template('memory.html')


    @app.route('/ai_profile')
    def ai_profile_page():
        """AI self-expression page."""
        return render_template('ai_profile.html')


    # =========================================================================
    # Download Routes
    # =========================================================================

    @app.route('/download/<platform>')
    def download_page(platform):
        """Handle download requests for various platforms."""
        platform_info = {
            'windows': {
                'name': 'Windows',
                'file': 'EnigmaAI-Setup.exe',
                'instructions': 'Run the installer and follow the setup wizard.',
                'github_release': 'https://github.com/SirRDragonbornKnight/enigma_engine/releases/latest'
            },
            'macos': {
                'name': 'macOS',
                'file': 'EnigmaAI.dmg',
                'instructions': 'Open the DMG and drag Enigma AI to Applications.',
                'github_release': 'https://github.com/SirRDragonbornKnight/enigma_engine/releases/latest'
            },
            'linux': {
                'name': 'Linux',
                'file': 'EnigmaAI.AppImage',
                'instructions': 'Make executable with chmod +x and run.',
                'github_release': 'https://github.com/SirRDragonbornKnight/enigma_engine/releases/latest'
            },
            'rpi': {
                'name': 'Raspberry Pi',
                'file': None,
                'instructions': '''
                    <ol>
                        <li>Clone the repo: <code>git clone https://github.com/SirRDragonbornKnight/enigma_engine.git</code></li>
                        <li>Run installer: <code>python install.py --minimal</code></li>
                        <li>Use nano or micro model sizes for best performance</li>
                    </ol>
                ''',
                'github_release': 'https://github.com/SirRDragonbornKnight/enigma_engine'
            },
            'android': {
                'name': 'Android',
                'file': 'EnigmaAI.apk',
                'instructions': 'Enable "Install from unknown sources" and install the APK.',
                'github_release': 'https://github.com/SirRDragonbornKnight/enigma_engine/releases/latest'
            },
            'ios': {
                'name': 'iOS (TestFlight)',
                'file': None,
                'instructions': 'Join our TestFlight beta or build from source using Expo.',
                'github_release': 'https://github.com/SirRDragonbornKnight/enigma_engine/tree/main/mobile'
            }
        }
        
        info = platform_info.get(platform, {
            'name': platform.title(),
            'file': None,
            'instructions': 'Platform not found. Please check the GitHub releases.',
            'github_release': 'https://github.com/SirRDragonbornKnight/enigma_engine/releases'
        })
        
        # For now, redirect to GitHub releases (actual file hosting would need storage)
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Download Enigma AI for {info['name']}</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ font-family: -apple-system, sans-serif; background: #1e1e2e; color: #cdd6f4; 
                       display: flex; justify-content: center; align-items: center; min-height: 100vh;
                       margin: 0; padding: 20px; }}
                .card {{ background: #313244; padding: 2rem; border-radius: 12px; max-width: 500px; text-align: center; }}
                h1 {{ color: #89b4fa; }}
                p {{ color: #a6adc8; line-height: 1.6; }}
                a.btn {{ display: inline-block; background: #89b4fa; color: #1e1e2e; padding: 12px 24px;
                        border-radius: 8px; text-decoration: none; font-weight: 600; margin: 10px; }}
                a.btn:hover {{ background: #b4befe; }}
                a.btn-secondary {{ background: #45475a; color: #cdd6f4; }}
                code {{ background: #45475a; padding: 2px 6px; border-radius: 4px; }}
            </style>
        </head>
        <body>
            <div class="card">
                <h1>Download for {info['name']}</h1>
                <p>{info['instructions']}</p>
                <a href="{info['github_release']}" class="btn">Download from GitHub</a>
                <br>
                <a href="/" class="btn btn-secondary">Back to Home</a>
            </div>
        </body>
        </html>
        '''


    @app.route('/api/status')
    def api_status():
        """Get system status with device profile information."""
        engine = get_engine()
        
        status = {
            'status': 'running',
            'model_loaded': engine is not None,
            'model_name': _model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        # Get device profile info
        try:
            from ..core.device_profiles import get_device_profiler
            profiler = get_device_profiler()
            caps = profiler.detect()
            status['device'] = {
                'class': profiler.classify().name,
                'torch_device': profiler.get_torch_device(),
                'recommended_model': profiler.get_recommended_model_size(),
                'cpu_cores': caps.cpu_cores,
                'ram_mb': caps.ram_total_mb,
                'has_gpu': caps.has_gpu,
                'gpu_name': caps.gpu_name if caps.has_gpu else None,
                'vram_mb': caps.vram_total_mb if caps.has_gpu else None,
            }
        except ImportError:
            pass
        
        # Get instance info
        try:
            from ..core.instance_manager import get_active_instances
            instances = get_active_instances()
            status['instances'] = len(instances)
        except ImportError:
            logger.debug("Instance manager not available, defaulting to 1 instance")
            status['instances'] = 1
        except Exception as e:
            logger.warning(f"Could not get active instances: {e}")
            status['instances'] = 1
        
        return jsonify(status)


    @app.route('/api/models')
    def api_list_models():
        """List available models."""
        try:
            models_dir = Path(CONFIG["models_dir"])
            models = []
            
            for model_path in models_dir.glob("*"):
                if model_path.is_dir():
                    # Check if it has model files
                    has_model = any(
                        model_path.glob("*.pth")
                    ) or any(
                        model_path.glob("*.pt")
                    )
                    
                    if has_model:
                        models.append({
                            'name': model_path.name,
                            'path': str(model_path),
                            'size': sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                        })
            
            return jsonify({'models': models})
        except Exception as e:
            return jsonify({'error': str(e)}), 500


    @app.route('/api/generate', methods=['POST'])
    def api_generate():
        """Generate text from prompt."""
        request_data = request.json
        prompt = request_data.get('prompt', '')
        max_tokens = request_data.get('max_tokens', DEFAULT_MAX_TOKENS)
        temperature = request_data.get('temperature', DEFAULT_TEMPERATURE)
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), HTTP_BAD_REQUEST
        
        engine = get_engine()
        if engine is None:
            return jsonify({'error': 'Model not loaded'}), HTTP_SERVER_ERROR
        
        try:
            response = engine.generate(
                prompt,
                max_gen=max_tokens,
                temperature=temperature
            )
            
            return jsonify({
                'success': True,
                'response': response,
                'model': _model_name,
                'tokens': len(response.split())
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500


# =============================================================================
# Personality API
# =============================================================================

# Global personality settings
# NOTE: This is in-memory storage for demo purposes. For production use,
# implement proper persistence layer (database or file-based) to maintain
# state between restarts and support multiple users.
_personality_settings = {
    'humor': 0.5,
    'formality': 0.5,
    'verbosity': 0.5,
    'creativity': 0.7,
    'user_controlled': False,
    'interests': ['AI', 'technology', 'science'],
    'dislikes': ['repetition'],
}


@app.route('/api/personality', methods=['GET'])
def api_get_personality():
    """Get current personality traits (evolved + user overrides)."""
    return jsonify({
        'success': True,
        'personality': _personality_settings,
        'description': 'AI personality traits and preferences'
    })


@app.route('/api/personality', methods=['POST'])
def api_set_personality():
    """Set user overrides for personality traits."""
    personality_updates = request.json
    
    try:
        # Update personality settings
        if 'humor' in personality_updates:
            _personality_settings['humor'] = float(personality_updates['humor'])
        if 'formality' in personality_updates:
            _personality_settings['formality'] = float(personality_updates['formality'])
        if 'verbosity' in personality_updates:
            _personality_settings['verbosity'] = float(personality_updates['verbosity'])
        if 'creativity' in personality_updates:
            _personality_settings['creativity'] = float(personality_updates['creativity'])
        if 'user_controlled' in personality_updates:
            _personality_settings['user_controlled'] = bool(personality_updates['user_controlled'])
        
        return jsonify({
            'success': True,
            'personality': _personality_settings,
            'message': 'Personality updated successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/personality/reset', methods=['POST'])
def api_reset_personality():
    """Clear user overrides, let AI evolve naturally."""
    _personality_settings.update({
        'humor': 0.5,
        'formality': 0.5,
        'verbosity': 0.5,
        'creativity': 0.7,
        'user_controlled': False,
    })
    
    return jsonify({
        'success': True,
        'personality': _personality_settings,
        'message': 'Personality reset to defaults'
    })


# =============================================================================
# Voice API
# =============================================================================

# Global voice settings
# NOTE: In-memory storage. Consider file-based persistence for production.
_voice_settings = {
    'profile': 'default',
    'pitch': 1.0,
    'speed': 1.0,
    'volume': 1.0,
    'effects': [],
}

_voice_profiles = [
    {'id': 'default', 'name': 'Default', 'description': 'Standard voice'},
    {'id': 'glados', 'name': 'GLaDOS', 'description': 'Robotic, slightly sarcastic'},
    {'id': 'jarvis', 'name': 'Jarvis', 'description': 'British, formal, helpful'},
    {'id': 'robot', 'name': 'Robot', 'description': 'Mechanical, monotone'},
]


@app.route('/api/voice', methods=['GET'])
def api_get_voice():
    """Get current voice profile."""
    return jsonify({
        'success': True,
        'voice': _voice_settings,
        'description': 'Current voice settings'
    })


@app.route('/api/voice', methods=['POST'])
def api_set_voice():
    """Set voice profile (user custom or AI-generated)."""
    data = request.json
    
    try:
        if 'profile' in data:
            _voice_settings['profile'] = data['profile']
        if 'pitch' in data:
            _voice_settings['pitch'] = float(data['pitch'])
        if 'speed' in data:
            _voice_settings['speed'] = float(data['speed'])
        if 'volume' in data:
            _voice_settings['volume'] = float(data['volume'])
        if 'effects' in data:
            _voice_settings['effects'] = data['effects']
        
        return jsonify({
            'success': True,
            'voice': _voice_settings,
            'message': 'Voice settings updated successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/voice/preview', methods=['POST'])
def api_preview_voice():
    """Generate voice preview audio."""
    data = request.json
    text = data.get('text', 'Hello, this is a voice preview.')
    
    # This is a placeholder - actual TTS would be implemented here
    return jsonify({
        'success': True,
        'message': 'Voice preview generated',
        'audio_url': None,  # Would be audio file URL
        'note': 'TTS integration required for actual audio'
    })


@app.route('/api/voice/profiles', methods=['GET'])
def api_list_voice_profiles():
    """List all available voice profiles (presets + custom)."""
    return jsonify({
        'success': True,
        'profiles': _voice_profiles
    })


@app.route('/api/voice/transcribe', methods=['POST'])
def api_transcribe_voice():
    """
    Transcribe audio to text using server-side STT.
    
    Request body:
        {
            "audio": "<base64 encoded audio>",
            "format": "webm",
            "language": "en-US"
        }
    
    Returns:
        {
            "success": true,
            "text": "transcribed text"
        }
    """
    data = request.json
    
    if not data or 'audio' not in data:
        return jsonify({'success': False, 'error': 'No audio data provided'}), HTTP_BAD_REQUEST
    
    try:
        import base64
        import tempfile
        import os
        
        # Decode audio
        audio_b64 = data['audio']
        audio_format = data.get('format', 'webm')
        language = data.get('language', 'en-US')
        
        audio_data = base64.b64decode(audio_b64)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=f'.{audio_format}', delete=False) as f:
            f.write(audio_data)
            temp_path = f.name
        
        try:
            # Try Whisper first
            try:
                from ..voice.whisper_stt import WhisperSTT
                stt = WhisperSTT()
                text = stt.transcribe_file(temp_path)
                return jsonify({'success': True, 'text': text})
            except ImportError:
                pass
            
            # Try speech_recognition
            try:
                import speech_recognition as sr
                recognizer = sr.Recognizer()
                
                # Convert to wav if needed
                audio_path = temp_path
                if audio_format != 'wav':
                    try:
                        from pydub import AudioSegment
                        audio = AudioSegment.from_file(temp_path, format=audio_format)
                        wav_path = temp_path.replace(f'.{audio_format}', '.wav')
                        audio.export(wav_path, format='wav')
                        audio_path = wav_path
                    except ImportError:
                        return jsonify({
                            'success': False, 
                            'error': 'Audio conversion not available. Install pydub.'
                        }), HTTP_SERVER_ERROR
                
                with sr.AudioFile(audio_path) as source:
                    audio = recognizer.record(source)
                    text = recognizer.recognize_google(audio, language=language)
                    return jsonify({'success': True, 'text': text})
                    
            except ImportError:
                return jsonify({
                    'success': False,
                    'error': 'No STT backend available. Install whisper or speech_recognition.'
                }), HTTP_SERVER_ERROR
            except Exception as e:
                return jsonify({'success': False, 'error': f'Transcription failed: {str(e)}'}), HTTP_SERVER_ERROR
                
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            # Clean up converted wav if exists
            wav_path = temp_path.replace(f'.{audio_format}', '.wav')
            if audio_format != 'wav' and os.path.exists(wav_path):
                os.unlink(wav_path)
                
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return jsonify({'success': False, 'error': str(e)}), HTTP_SERVER_ERROR


# =============================================================================
# Push Notification API
# =============================================================================

# VAPID keys for Web Push (generate with: openssl ecparam -genkey -name prime256v1 -out private.pem)
# In production, load from environment or secure storage
_vapid_keys = {
    'public_key': os.environ.get('VAPID_PUBLIC_KEY', ''),
    'private_key': os.environ.get('VAPID_PRIVATE_KEY', ''),
    'claims': {'sub': 'mailto:admin@example.com'}
}

# Push subscriptions storage (use database in production)
_push_subscriptions = {}


@app.route('/api/push/vapid-key', methods=['GET'])
def api_get_vapid_key():
    """Get VAPID public key for push subscriptions."""
    public_key = _vapid_keys.get('public_key', '')
    
    if not public_key:
        # Generate a placeholder key for development
        import base64
        # This is just a placeholder - in production, use proper VAPID keys
        public_key = base64.urlsafe_b64encode(os.urandom(65)).decode('utf-8').rstrip('=')
    
    return jsonify({
        'success': True,
        'publicKey': public_key
    })


@app.route('/api/push/subscribe', methods=['POST'])
def api_push_subscribe():
    """
    Save a push subscription.
    
    Request body:
        {
            "subscription": {
                "endpoint": "https://...",
                "keys": { "p256dh": "...", "auth": "..." }
            }
        }
    """
    data = request.json
    
    if not data or 'subscription' not in data:
        return jsonify({'success': False, 'error': 'No subscription data'}), HTTP_BAD_REQUEST
    
    subscription = data['subscription']
    endpoint = subscription.get('endpoint', '')
    
    if not endpoint:
        return jsonify({'success': False, 'error': 'Invalid subscription'}), HTTP_BAD_REQUEST
    
    # Store subscription (keyed by endpoint)
    _push_subscriptions[endpoint] = subscription
    
    logger.info(f"Push subscription saved: {endpoint[:50]}...")
    
    return jsonify({
        'success': True,
        'message': 'Subscription saved',
        'total_subscriptions': len(_push_subscriptions)
    })


@app.route('/api/push/unsubscribe', methods=['POST'])
def api_push_unsubscribe():
    """Remove a push subscription."""
    data = request.json
    
    if not data or 'endpoint' not in data:
        return jsonify({'success': False, 'error': 'No endpoint provided'}), HTTP_BAD_REQUEST
    
    endpoint = data['endpoint']
    
    if endpoint in _push_subscriptions:
        del _push_subscriptions[endpoint]
        logger.info(f"Push subscription removed: {endpoint[:50]}...")
    
    return jsonify({
        'success': True,
        'message': 'Subscription removed'
    })


@app.route('/api/push/test', methods=['POST'])
def api_push_test():
    """Send a test push notification."""
    data = request.json or {}
    message = data.get('message', 'Test notification from Enigma Engine')
    
    # Try to send via pywebpush if available
    try:
        from pywebpush import webpush, WebPushException
        
        sent_count = 0
        for endpoint, subscription in list(_push_subscriptions.items()):
            try:
                webpush(
                    subscription_info=subscription,
                    data=json.dumps({
                        'title': 'Enigma Engine',
                        'body': message,
                        'url': '/'
                    }),
                    vapid_private_key=_vapid_keys.get('private_key'),
                    vapid_claims=_vapid_keys.get('claims', {})
                )
                sent_count += 1
            except WebPushException as e:
                logger.warning(f"Push failed for {endpoint[:30]}: {e}")
                # Remove invalid subscription
                if e.response and e.response.status_code in (404, 410):
                    del _push_subscriptions[endpoint]
            except Exception as e:
                logger.warning(f"Push error: {e}")
        
        return jsonify({
            'success': True,
            'message': f'Sent to {sent_count} subscriber(s)'
        })
        
    except ImportError:
        # pywebpush not installed - just acknowledge
        logger.warning("pywebpush not installed - cannot send real push notifications")
        return jsonify({
            'success': True,
            'message': 'Test acknowledged (install pywebpush for real notifications)',
            'subscribers': len(_push_subscriptions)
        })


@app.route('/api/push/send', methods=['POST'])
def api_push_send():
    """
    Send push notification to all subscribers.
    
    Request body:
        {
            "title": "Notification Title",
            "body": "Notification message",
            "url": "/chat"  (optional)
        }
    """
    data = request.json
    
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), HTTP_BAD_REQUEST
    
    title = data.get('title', 'Enigma Engine')
    body = data.get('body', '')
    url = data.get('url', '/')
    
    if not body:
        return jsonify({'success': False, 'error': 'Message body required'}), HTTP_BAD_REQUEST
    
    try:
        from pywebpush import webpush, WebPushException
        
        sent_count = 0
        failed_count = 0
        
        for endpoint, subscription in list(_push_subscriptions.items()):
            try:
                webpush(
                    subscription_info=subscription,
                    data=json.dumps({
                        'title': title,
                        'body': body,
                        'url': url
                    }),
                    vapid_private_key=_vapid_keys.get('private_key'),
                    vapid_claims=_vapid_keys.get('claims', {})
                )
                sent_count += 1
            except WebPushException as e:
                failed_count += 1
                if e.response and e.response.status_code in (404, 410):
                    del _push_subscriptions[endpoint]
            except Exception:
                failed_count += 1
        
        return jsonify({
            'success': True,
            'sent': sent_count,
            'failed': failed_count
        })
        
    except ImportError:
        return jsonify({
            'success': False,
            'error': 'pywebpush not installed'
        }), HTTP_SERVER_ERROR


# =============================================================================
# Memory API
# =============================================================================

# Global memory storage (in-memory for now)
_memories = []


@app.route('/api/memory', methods=['GET'])
def api_get_memories():
    """Get AI's memories and learned information."""
    try:
        # Try to get memories from the conversation manager
        from ..memory.manager import ConversationManager
        
        conv_manager = ConversationManager()
        # List all conversation files
        conversations = []
        for conv_file in conv_manager.conv_dir.glob('*.json'):
            try:
                data = json.loads(conv_file.read_text())
                conversations.append({
                    'id': conv_file.stem,
                    'name': data.get('name', conv_file.stem),
                    'saved_at': data.get('saved_at', 0),
                    'message_count': len(data.get('messages', []))
                })
            except json.JSONDecodeError as e:
                logger.warning(f"Corrupted conversation file {conv_file.name}: {e}")
            except OSError as e:
                logger.warning(f"Could not read conversation file {conv_file.name}: {e}")
        
        return jsonify({
            'success': True,
            'memories': _memories,
            'conversations': conversations,
            'total': len(_memories) + len(conversations)
        })
    except Exception as e:
        logger.debug(f"Error loading memories from conversation manager: {e}")
        return jsonify({
            'success': True,
            'memories': _memories,
            'conversations': [],
            'total': len(_memories)
        })


@app.route('/api/memory', methods=['POST'])
def api_add_memory():
    """User adds a memory for the AI."""
    global _memories
    MAX_MEMORIES = 1000  # Prevent unbounded growth
    
    data = request.json
    text = data.get('text', '')
    importance = data.get('importance', 0.5)
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    memory = {
        'id': len(_memories) + 1,
        'text': text,
        'importance': importance,
        'timestamp': datetime.now().isoformat(),
        'source': 'user'
    }
    _memories.append(memory)
    
    # Trim oldest memories if over limit
    if len(_memories) > MAX_MEMORIES:
        _memories = _memories[-MAX_MEMORIES:]
    
    return jsonify({
        'success': True,
        'memory': memory,
        'message': 'Memory added successfully'
    })


@app.route('/api/memory/<int:memory_id>', methods=['DELETE'])
def api_delete_memory(memory_id):
    """Remove a specific memory."""
    global _memories
    
    # Find and remove memory
    _memories = [m for m in _memories if m['id'] != memory_id]
    
    return jsonify({
        'success': True,
        'message': 'Memory deleted successfully'
    })


# =============================================================================
# Training API
# =============================================================================

# Global training state
_training_state = {
    'running': False,
    'progress': 0,
    'epoch': 0,
    'total_epochs': 0,
    'loss': 0.0,
    'status': 'idle'
}


@app.route('/api/train/start', methods=['POST'])
def api_start_training():
    """Start training in background thread."""
    data = request.json
    
    if _training_state['running']:
        return jsonify({'error': 'Training already in progress'}), 400
    
    model_name = data.get('model_name', 'enigma_engine')
    model_size = data.get('model_size', 'small')
    epochs = data.get('epochs', 30)
    learning_rate = data.get('learning_rate', 0.0001)
    batch_size = data.get('batch_size', 8)
    
    # Update training state
    _training_state['running'] = True
    _training_state['progress'] = 0
    _training_state['epoch'] = 0
    _training_state['total_epochs'] = epochs
    _training_state['status'] = 'starting'
    
    def run_training():
        """Background training thread."""
        try:
            # Import training components
            from enigma_engine.core.model_registry import get_model_registry
            from enigma_engine.core.training import Trainer, TrainingConfig
            
            registry = get_model_registry()
            
            # Create or load model
            _training_state['status'] = 'loading_model'
            if not registry.model_exists(model_name):
                registry.create_model(model_name, size=model_size)
            model = registry.load_model(model_name)
            
            # Load training data
            _training_state['status'] = 'loading_data'
            from pathlib import Path
            data_path = Path('data/training.txt')
            if not data_path.exists():
                _training_state['status'] = 'error'
                _training_state['error'] = 'No training data found'
                _training_state['running'] = False
                return
            
            # Configure training
            config = TrainingConfig(
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                model_name=model_name,
            )
            
            # Progress callback
            def on_progress(epoch, loss, progress):
                _training_state['epoch'] = epoch
                _training_state['loss'] = loss
                _training_state['progress'] = int(progress * 100)
                _training_state['status'] = f'training_epoch_{epoch}'
            
            # Train
            _training_state['status'] = 'training'
            trainer = Trainer(model, config)
            trainer.on_progress = on_progress
            trainer.train(str(data_path))
            
            # Save model
            _training_state['status'] = 'saving'
            registry.save_model(model_name, model)
            
            _training_state['status'] = 'completed'
            _training_state['progress'] = 100
            
        except Exception as e:
            _training_state['status'] = 'error'
            _training_state['error'] = str(e)
        finally:
            _training_state['running'] = False
    
    # Start training in background thread
    import threading
    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Training started in background',
        'model_name': model_name,
        'epochs': epochs,
    })


@app.route('/api/train/status', methods=['GET'])
def api_training_status():
    """Get current training progress."""
    return jsonify({
        'success': True,
        'training': _training_state
    })


@app.route('/api/train/stop', methods=['POST'])
def api_stop_training():
    """Stop training."""
    _training_state['running'] = False
    _training_state['status'] = 'stopped'
    
    return jsonify({
        'success': True,
        'message': 'Training stopped'
    })


# =============================================================================
# AI Self-Expression API
# =============================================================================

# Global AI state
_ai_state = {
    'mood': 'curious',
    'mood_emoji': '😊',
    'interests': ['learning', 'helping', 'problem-solving'],
    'favorite_topics': ['AI', 'technology', 'creativity'],
    'conversation_count': 0,
}


@app.route('/api/ai/mood', methods=['GET'])
def api_get_ai_mood():
    """Get AI's current mood based on actual state."""
    # Determine mood from actual AI state
    try:
        from ..core.inference import EnigmaEngine
        engine = EnigmaEngine.get_instance()
        
        if engine and engine.model:
            # Base mood on recent interactions and system state
            prompt = """Based on being an AI assistant, what mood are you in right now? 
Choose ONE: curious, happy, focused, creative
Reply with ONLY the mood word."""
            
            response = engine.generate(prompt, max_gen=15, temperature=0.5)
            mood_word = response.strip().lower().split()[0] if response else "focused"
            
            mood_map = {
                'curious': {'mood': 'curious', 'emoji': '...', 'text': "I'm feeling curious today!"},
                'happy': {'mood': 'happy', 'emoji': '...', 'text': "I'm in a great mood!"},
                'focused': {'mood': 'focused', 'emoji': '...', 'text': "Ready to help and solve problems!"},
                'creative': {'mood': 'creative', 'emoji': '...', 'text': "Feeling creative and inspired!"},
            }
            
            current = mood_map.get(mood_word, mood_map['focused'])
        else:
            # No AI loaded - default to focused
            current = {'mood': 'focused', 'emoji': '...', 'text': "Ready to help!"}
    except Exception:
        current = {'mood': 'focused', 'emoji': '...', 'text': "Ready to help!"}
    
    _ai_state['mood'] = current['mood']
    _ai_state['mood_emoji'] = current['emoji']
    
    return jsonify({
        'success': True,
        'mood': _ai_state['mood'],
        'emoji': _ai_state['mood_emoji'],
        'text': current['text'],
        'interests': _ai_state['interests']
    })


@app.route('/api/ai/preferences', methods=['GET'])
def api_get_ai_preferences():
    """Get what AI has learned about itself (interests, opinions)."""
    return jsonify({
        'success': True,
        'preferences': {
            'interests': _ai_state['interests'],
            'favorite_topics': _ai_state['favorite_topics'],
            'conversation_count': _ai_state['conversation_count']
        }
    })


@app.route('/api/ai/explain', methods=['POST'])
def api_explain_behavior():
    """AI explains why it behaves a certain way."""
    data = request.json
    question = data.get('question', '')
    
    # Simple placeholder response
    explanations = {
        'personality': "My personality is shaped by both my training data and user interactions. I aim to be helpful, creative, and engaging.",
        'voice': "Voice settings can be customized by users or automatically adjusted based on personality traits.",
        'interests': "My interests develop through conversations and learning from user interactions.",
        'default': "I'm designed to learn and adapt based on interactions while maintaining helpful and ethical behavior."
    }
    
    # Try to match question to explanation
    explanation = explanations.get('default')
    for key, value in explanations.items():
        if key in question.lower():
            explanation = value
            break
    
    return jsonify({
        'success': True,
        'explanation': explanation
    })


# =============================================================================
# WebSocket Events (if available)
# =============================================================================

if SOCKETIO_AVAILABLE and socketio:
    
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        logger.debug(f"Client connected: {request.sid}")
        emit('status', {'message': 'Connected to Enigma AI Engine'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        logger.debug(f"Client disconnected: {request.sid}")
    
    @socketio.on('message')
    def handle_message(data):
        """Handle chat message."""
        prompt = data.get('text', '')
        
        if not prompt:
            emit('error', {'message': 'No prompt provided'})
            return
        
        engine = get_engine()
        if engine is None:
            emit('error', {'message': 'Model not loaded'})
            return
        
        try:
            # Generate response
            response = engine.generate(prompt, max_gen=200)
            
            emit('response', {
                'text': response,
                'model': _model_name,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            emit('error', {'message': str(e)})
    
    @socketio.on('stream_generate')
    def handle_stream_generate(data):
        """Handle streaming generation."""
        prompt = data.get('text', '')
        
        if not prompt:
            emit('error', {'message': 'No prompt provided'})
            return
        
        engine = get_engine()
        if engine is None:
            emit('error', {'message': 'Model not loaded'})
            return
        
        try:
            # Stream response
            for token in engine.stream_generate(prompt, max_gen=200):
                emit('token', {'token': token})
            
            emit('stream_end', {'message': 'Generation complete'})
        except Exception as e:
            emit('error', {'message': str(e)})
    
    @socketio.on('start_training')
    def handle_start_training(data):
        """Start training and emit progress updates."""
        if _training_state['running']:
            emit('error', {'message': 'Training already in progress'})
            return
        
        # Update training state
        _training_state['running'] = True
        _training_state['progress'] = 0
        _training_state['status'] = 'starting'
        
        emit('training_progress', {
            'status': 'starting',
            'progress': 0,
            'message': 'Training initialization...'
        })
        
        # Note: Actual background training would be implemented here
    
    @socketio.on('preview_voice')
    def handle_preview_voice(data):
        """Generate and send voice preview."""
        text = data.get('text', 'Hello, this is a voice preview.')
        settings = data.get('settings', {})
        
        # Placeholder response
        emit('voice_preview', {
            'success': True,
            'message': 'Voice preview generated',
            'note': 'TTS integration required for actual audio'
        })
    
    @socketio.on('get_ai_state')
    def handle_get_ai_state():
        """Get current AI mood and state."""
        emit('ai_state', {
            'mood': _ai_state['mood'],
            'emoji': _ai_state['mood_emoji'],
            'interests': _ai_state['interests']
        })


# =============================================================================
# Run Function
# =============================================================================

def run_web(host: str = '0.0.0.0', port: int = 8080, debug: bool = False):
    """
    Run the web dashboard.
    
    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to bind to (default: 8080)
        debug: Enable debug mode
    """
    if not FLASK_AVAILABLE:
        logger.error("Flask not installed. Install with: pip install flask flask-cors")
        return
    
    logger.info(f"{'='*60}")
    logger.info("FORGE AI WEB DASHBOARD")
    logger.info(f"{'='*60}")
    logger.info(f"Server starting on http://{host}:{port}")
    logger.info(f"Access from:")
    logger.info(f"   - Local:   http://localhost:{port}")
    logger.info(f"   - Network: http://{host}:{port}")
    logger.info(f"Available pages:")
    logger.info(f"   - Dashboard:   http://localhost:{port}/")
    logger.info(f"   - Chat:        http://localhost:{port}/chat")
    logger.info(f"   - Personality: http://localhost:{port}/personality")
    logger.info(f"   - Voice:       http://localhost:{port}/voice")
    logger.info(f"   - Memory:      http://localhost:{port}/memory")
    logger.info(f"   - Train:       http://localhost:{port}/train")
    logger.info(f"   - Settings:    http://localhost:{port}/settings")
    logger.info(f"{'='*60}")
    
    if SOCKETIO_AVAILABLE and socketio:
        logger.info("WebSocket support enabled")
        socketio.run(app, host=host, port=port, debug=debug)
    else:
        logger.warning("WebSocket support not available (install flask-socketio)")
        logger.warning("Real-time chat features will be limited")
        app.run(host=host, port=port, debug=debug)
