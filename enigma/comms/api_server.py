"""
Simple REST API Server

Minimal Flask API for programmatic access to Enigma.
Use this for scripts, automation, and lightweight integrations.

For full web interface with WebSocket support, use web_server.py instead.

Endpoints:
  GET  /health    - Health check
  GET  /info      - Server info
  POST /generate  - Generate response (prompt, max_gen, temperature)

Usage:
  python -m enigma.comms.api_server
  # or
  python run.py --serve
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from ..core.inference import EnigmaEngine


def create_app():
    """Create Flask app without starting it."""
    app = Flask("enigma_api")
    CORS(app)
    engine = EnigmaEngine()

    @app.route("/health")
    def health():
        return jsonify({"ok": True})
    
    @app.route("/info")
    def info():
        return jsonify({
            "name": "Enigma API",
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
    print(f"Starting Enigma API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
    return app
