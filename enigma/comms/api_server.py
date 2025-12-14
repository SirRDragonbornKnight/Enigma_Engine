"""
Very small Flask API exposing a single /generate endpoint.
Use for local integrations and testing.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from ..core.inference import EnigmaEngine

def create_app():
    app = Flask("enigma_api")
    CORS(app)
    engine = EnigmaEngine()

    @app.route("/health")
    def health():
        return jsonify({"ok": True})

    @app.route("/generate", methods=["POST"])
    def generate():
        data = request.json or {}
        prompt = data.get("prompt", "")
        max_gen = int(data.get("max_gen", 50))
        temp = float(data.get("temperature", 1.0))
        text = engine.generate(prompt, max_gen=max_gen, temperature=temp)
        return jsonify({"text": text})

    return app
