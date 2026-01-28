"""
Web Dashboard for ForgeAI

Provides a web-based interface for:
- Chat interface
- Model training
- Settings management
- Instance monitoring

Usage:
    from forge_ai.web.app import run_web
    run_web(host='0.0.0.0', port=8080)
"""

import os
import json
import random
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from flask import Flask, render_template, jsonify, request
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
            default_model = CONFIG.get("default_model", "forge_ai")
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
        """Main dashboard."""
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
            except IOError as e:
                logger.warning(f"Could not read conversation file {conv_file.name}: {e}")
        
        return jsonify({
            'success': True,
            'memories': _memories,
            'conversations': conversations,
            'total': len(_memories) + len(conversations)
        })
    except Exception as e:
        return jsonify({
            'success': True,
            'memories': _memories,
            'conversations': [],
            'total': len(_memories)
        })


@app.route('/api/memory', methods=['POST'])
def api_add_memory():
    """User adds a memory for the AI."""
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
    
    model_name = data.get('model_name', 'forge_ai')
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
            from forge_ai.core.training import Trainer, TrainingConfig
            from forge_ai.core.model_registry import get_model_registry
            
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
    """Get AI's current mood and state."""
    moods = [
        {'mood': 'curious', 'emoji': '🤔', 'text': "I'm feeling curious today!"},
        {'mood': 'happy', 'emoji': '😊', 'text': "I'm in a great mood!"},
        {'mood': 'focused', 'emoji': '🎯', 'text': "Ready to help and solve problems!"},
        {'mood': 'creative', 'emoji': '🎨', 'text': "Feeling creative and inspired!"},
    ]
    
    current = random.choice(moods)
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
        emit('status', {'message': 'Connected to ForgeAI'})
    
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
        print("Error: Flask not installed. Install with: pip install flask flask-cors")
        return
    
    print(f"\n{'='*60}")
    print("FORGE AI WEB DASHBOARD")
    print(f"{'='*60}")
    print(f"\nServer starting on http://{host}:{port}")
    print(f"\nAccess from:")
    print(f"   - Local:   http://localhost:{port}")
    print(f"   - Network: http://{host}:{port}")
    print(f"\nAvailable pages:")
    print(f"   - Dashboard:   http://localhost:{port}/")
    print(f"   - Chat:        http://localhost:{port}/chat")
    print(f"   - Personality: http://localhost:{port}/personality")
    print(f"   - Voice:       http://localhost:{port}/voice")
    print(f"   - Memory:      http://localhost:{port}/memory")
    print(f"   - Train:       http://localhost:{port}/train")
    print(f"   - Settings:    http://localhost:{port}/settings")
    print(f"\n{'='*60}\n")
    
    if SOCKETIO_AVAILABLE and socketio:
        print("[OK] WebSocket support enabled")
        socketio.run(app, host=host, port=port, debug=debug)
    else:
        print("[!] WebSocket support not available (install flask-socketio)")
        print("  Real-time chat features will be limited")
        app.run(host=host, port=port, debug=debug)
