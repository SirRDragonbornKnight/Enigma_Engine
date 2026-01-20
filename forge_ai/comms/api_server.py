"""
================================================================================
🌐 REST API SERVER - NETWORK GATEWAY
================================================================================

Minimal Flask API for programmatic access to Forge. Use this for scripts,
automation, and integrations with external systems.

📍 FILE: forge_ai/comms/api_server.py
🏷️ TYPE: REST API Server (Flask)
🎯 MAIN FUNCTIONS: create_app(), create_api_server()

┌─────────────────────────────────────────────────────────────────────────────┐
│  API ENDPOINTS:                                                             │
│                                                                             │
│  GET  /health    → {"ok": true}         (Health check)                     │
│  GET  /info      → Server information                                      │
│  POST /generate  → Generate AI response                                    │
│       Body: {"prompt": "...", "max_gen": 50, "temperature": 1.0}           │
└─────────────────────────────────────────────────────────────────────────────┘

🚀 HOW TO START:
    python run.py --serve
    # OR
    python -m forge_ai.comms.api_server
    
    Then access: http://localhost:5000

📝 EXAMPLE REQUEST:
    curl -X POST http://localhost:5000/generate \
         -H "Content-Type: application/json" \
         -d '{"prompt": "Hello, AI!", "max_gen": 50}'

🔗 CONNECTED FILES:
    → USES:      forge_ai/core/inference.py (ForgeEngine for generation)
    ← USED BY:   run.py --serve (entry point)
    📄 ALTERNATIVE: forge_ai/comms/web_server.py (full web UI + WebSocket)

📖 SEE ALSO:
    • forge_ai/comms/network.py       - Multi-device networking
    • forge_ai/comms/remote_client.py - Connect to remote servers
    • forge_ai/comms/mobile_api.py    - Mobile app API
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
import secrets
from ..core.inference import ForgeEngine
from ..config import CONFIG


def generate_api_key() -> str:
    """Generate a secure random API key."""
    return f"sk-forge-{secrets.token_urlsafe(32)}"


def require_api_key(f):
    """Decorator to require API key authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Skip auth if disabled in config
        if not CONFIG.get("require_api_key", True):
            return f(*args, **kwargs)
        
        # Check for API key
        api_key = CONFIG.get("forgeai_api_key")
        if not api_key:
            # No API key configured - allow access (development mode)
            return f(*args, **kwargs)
        
        # Get key from request (multiple headers supported)
        request_key = (
            request.headers.get('X-API-Key') or
            request.headers.get('Authorization', '').replace('Bearer ', '') or
            request.args.get('api_key')
        )
        
        if not request_key or request_key != api_key:
            return jsonify({
                'error': 'Unauthorized',
                'message': 'Invalid or missing API key. Set FORGEAI_API_KEY environment variable.'
            }), 401
        
        return f(*args, **kwargs)
    return decorated_function


def create_app():
    """Create Flask app without starting it."""
    app = Flask("forge_api")
    CORS(app)
    engine = ForgeEngine()

    @app.route("/health")
    def health():
        return jsonify({"ok": True})
    
    @app.route("/info")
    def info():
        api_key_configured = CONFIG.get("forgeai_api_key") is not None
        auth_required = CONFIG.get("require_api_key", True)
        return jsonify({
            "name": "Forge API",
            "version": "1.0",
            "endpoints": ["/health", "/info", "/generate", "/generate_key"],
            "authentication": {
                "required": auth_required and api_key_configured,
                "methods": ["X-API-Key header", "Authorization: Bearer", "api_key query param"]
            }
        })
    
    @app.route("/generate_key", methods=["POST"])
    def generate_key():
        """Generate a new API key (for initial setup only)."""
        # Only allow if no key is set yet
        if CONFIG.get("forgeai_api_key"):
            return jsonify({
                "error": "API key already configured",
                "message": "Unset FORGEAI_API_KEY to generate a new one"
            }), 403
        
        new_key = generate_api_key()
        return jsonify({
            "api_key": new_key,
            "message": "Save this key! Set it as FORGEAI_API_KEY environment variable.",
            "example": f"export FORGEAI_API_KEY={new_key}"
        })

    @app.route("/generate", methods=["POST"])
    @require_api_key
    def generate():
        data = request.json or {}
        prompt = data.get("prompt", "")
        max_gen = int(data.get("max_gen", 50))
        temp = float(data.get("temperature", 1.0))
        text = engine.generate(prompt, max_gen=max_gen, temperature=temp)
        return jsonify({"text": text})

    return app


def create_api_server(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    """
    Create and start the API server.
    
    Args:
        host: Host to bind to
        port: Port to listen on
        debug: Enable debug mode
        
    Returns:
        The Flask app (after starting)
    """
    app = create_app()
    print(f"Starting Forge API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
    return app
