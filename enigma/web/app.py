"""
Web Dashboard for Enigma Engine

Provides a web-based interface for:
- Chat interface
- Model training
- Settings management
- Instance monitoring

Usage:
    from enigma.web.app import run_web
    run_web(host='0.0.0.0', port=8080)
"""

import os
import json
import random
from pathlib import Path
from datetime import datetime

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
            default_model = CONFIG.get("default_model", "enigma")
            _engine = InferenceEngine(model_name=default_model)
            _model_name = default_model
        except Exception as e:
            print(f"Warning: Could not load inference engine: {e}")
            _engine = None
    
    return _engine


# =============================================================================
# Routes
# =============================================================================

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
    """Get system status."""
    engine = get_engine()
    
    status = {
        'status': 'running',
        'model_loaded': engine is not None,
        'model_name': _model_name,
        'timestamp': datetime.now().isoformat()
    }
    
    # Get instance info
    try:
        from ..core.instance_manager import get_active_instances
        instances = get_active_instances()
        status['instances'] = len(instances)
    except Exception:
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
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 200)
    temperature = data.get('temperature', 0.7)
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    engine = get_engine()
    if engine is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
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
    data = request.json
    
    try:
        # Update personality settings
        if 'humor' in data:
            _personality_settings['humor'] = float(data['humor'])
        if 'formality' in data:
            _personality_settings['formality'] = float(data['formality'])
        if 'verbosity' in data:
            _personality_settings['verbosity'] = float(data['verbosity'])
        if 'creativity' in data:
            _personality_settings['creativity'] = float(data['creativity'])
        if 'user_controlled' in data:
            _personality_settings['user_controlled'] = bool(data['user_controlled'])
        
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
            except Exception:
                pass
        
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
    
    model_name = data.get('model_name', 'enigma')
    model_size = data.get('model_size', 'small')
    epochs = data.get('epochs', 30)
    
    # Update training state
    _training_state['running'] = True
    _training_state['progress'] = 0
    _training_state['epoch'] = 0
    _training_state['total_epochs'] = epochs
    _training_state['status'] = 'starting'
    
    # Note: Actual training would be done in a background thread
    # For now, this is a placeholder
    
    return jsonify({
        'success': True,
        'message': 'Training started',
        'note': 'Background training not yet implemented'
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
        print(f"Client connected: {request.sid}")
        emit('status', {'message': 'Connected to Enigma Engine'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        print(f"Client disconnected: {request.sid}")
    
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
    print("ENIGMA WEB DASHBOARD")
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
