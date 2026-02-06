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
import logging
import secrets
from functools import wraps

from flask import Flask, jsonify, request
from flask_cors import CORS

from ..config import CONFIG
from ..core.inference import ForgeEngine

logger = logging.getLogger(__name__)


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
    
    # Use engine pool for efficient reuse
    _engine_cache = {"engine": None}
    
    def get_cached_engine():
        """Get or create engine (lazy loading, singleton per app)."""
        if _engine_cache["engine"] is None:
            try:
                from ..core.engine_pool import get_engine
                _engine_cache["engine"] = get_engine()
            except ImportError:
                _engine_cache["engine"] = ForgeEngine()
        return _engine_cache["engine"]

    @app.route("/health")
    def health():
        """
        Health check endpoint.
        
        Returns:
            JSON: {"ok": true} if server is running
            
        Example:
            curl http://localhost:5000/health
        """
        return jsonify({"ok": True})
    
    @app.route("/info")
    def info():
        """
        Get server information and available endpoints.
        
        Returns:
            JSON containing:
            - name: Server name
            - version: API version
            - endpoints: List of available endpoints
            - authentication: Auth configuration
            
        Example:
            curl http://localhost:5000/info
        """
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
        """
        Generate a new API key for authentication.
        
        Only works if no API key is currently configured.
        
        Returns:
            JSON containing:
            - api_key: The generated key
            - message: Instructions for use
            - example: Shell command to set the key
            
        Errors:
            403: API key already configured
            
        Example:
            curl -X POST http://localhost:5000/generate_key
        """
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
        """
        Generate text from a prompt using the AI model.
        
        Request Body (JSON):
            - prompt (str): The input text to generate from
            - max_gen (int, optional): Maximum tokens to generate (1-2048, default: 50)
            - temperature (float, optional): Creativity level (0.0-2.0, default: 1.0)
            
        Returns:
            JSON containing:
            - text: The generated text
            
        Errors:
            400: Invalid parameter values
            401: Unauthorized (missing or invalid API key)
            503: AI engine not available
            500: Generation failed
            
        Example:
            curl -X POST http://localhost:5000/generate \\
                 -H "Content-Type: application/json" \\
                 -H "X-API-Key: YOUR_API_KEY" \\
                 -d '{"prompt": "Hello, AI!", "max_gen": 100}'
        """
        data = request.json or {}
        prompt = data.get("prompt", "")
        
        # Validate and bound numeric parameters to prevent abuse
        try:
            max_gen = int(data.get("max_gen", 50))
            max_gen = max(1, min(2048, max_gen))  # Bound: 1-2048 tokens
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid max_gen value - must be integer"}), 400
        
        try:
            temp = float(data.get("temperature", 1.0))
            temp = max(0.0, min(2.0, temp))  # Bound: 0.0-2.0
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid temperature value - must be number"}), 400
        
        # Use pooled engine
        engine = get_cached_engine()
        if engine is None:
            return jsonify({"error": "AI engine not available - check model is loaded"}), 503
        
        try:
            text = engine.generate(prompt, max_gen=max_gen, temperature=temp)
            return jsonify({"text": text})
        except Exception as e:
            return jsonify({"error": f"Generation failed: {e}"}), 500

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
    logger.info(f"Starting Forge API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
    return app
