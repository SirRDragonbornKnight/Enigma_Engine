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
from ..core.inference import ForgeEngine


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
        return jsonify({
            "name": "Forge API",
            "version": "1.0",
            "endpoints": ["/health", "/info", "/generate"]
        })

    @app.route("/generate", methods=["POST"])
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
