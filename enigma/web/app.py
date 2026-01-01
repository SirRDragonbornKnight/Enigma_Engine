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
from pathlib import Path
from typing import Optional
from datetime import datetime

try:
    from flask import Flask, render_template, jsonify, request, send_from_directory
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
    print(f"\n🌐 Server starting on http://{host}:{port}")
    print(f"\n📱 Access from:")
    print(f"   • Local:   http://localhost:{port}")
    print(f"   • Network: http://{host}:{port}")
    print(f"\n💡 Available pages:")
    print(f"   • Dashboard: http://localhost:{port}/")
    print(f"   • Chat:      http://localhost:{port}/chat")
    print(f"   • Train:     http://localhost:{port}/train")
    print(f"   • Settings:  http://localhost:{port}/settings")
    print(f"\n{'='*60}\n")
    
    if SOCKETIO_AVAILABLE and socketio:
        print("✓ WebSocket support enabled")
        socketio.run(app, host=host, port=port, debug=debug)
    else:
        print("⚠ WebSocket support not available (install flask-socketio)")
        print("  Real-time chat features will be limited")
        app.run(host=host, port=port, debug=debug)
