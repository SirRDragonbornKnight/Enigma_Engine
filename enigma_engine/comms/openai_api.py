"""
================================================================================
OpenAI-Compatible API Server
================================================================================

Drop-in replacement for OpenAI API. Any tool that works with OpenAI
(LangChain, LlamaIndex, AutoGPT, etc.) works with enigma_engine.

📍 FILE: enigma_engine/comms/openai_api.py
🏷️ TYPE: REST API Server (OpenAI-compatible)

ENDPOINTS:
    POST /v1/chat/completions      - Chat completions (GPT-4 style)
    POST /v1/completions           - Text completions (legacy)
    POST /v1/embeddings            - Text embeddings
    GET  /v1/models                - List available models
    GET  /v1/models/{model}        - Get model info

USAGE:
    # Start server
    python -m enigma_engine.comms.openai_api
    
    # Use with OpenAI SDK
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    response = client.chat.completions.create(
        model="forge",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    # Use with LangChain
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
"""

import json
import logging
import time
import uuid
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS

logger = logging.getLogger(__name__)

# =============================================================================
# OpenAI API Types
# =============================================================================

@dataclass
class ChatMessage:
    """A chat message."""
    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None
    
@dataclass
class Choice:
    """A completion choice."""
    index: int
    message: Optional[ChatMessage] = None
    text: Optional[str] = None
    finish_reason: str = "stop"
    
@dataclass
class Usage:
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass
class ChatCompletionResponse:
    """OpenAI-compatible chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = "forge"
    choices: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": self.choices,
            "usage": self.usage
        }

@dataclass
class CompletionResponse:
    """OpenAI-compatible completion response."""
    id: str
    object: str = "text_completion"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = "forge"
    choices: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": self.choices,
            "usage": self.usage
        }


# =============================================================================
# OpenAI-Compatible API Server
# =============================================================================

class OpenAICompatibleServer:
    """
    OpenAI-compatible API server for Forge models.
    
    Implements the OpenAI API specification so any OpenAI-compatible
    tool or library can use enigma_engine as a drop-in replacement.
    """
    
    def __init__(self, engine=None, host: str = "0.0.0.0", port: int = 8000):
        """
        Initialize the server.
        
        Args:
            engine: EnigmaEngine instance (lazy-loaded if None)
            host: Host to bind to
            port: Port to listen on
        """
        self.host = host
        self.port = port
        self._engine = engine
        self._embedding_model = None
        
        self.app = Flask("forge_openai_api")
        CORS(self.app)
        
        self._setup_routes()
    
    @property
    def engine(self):
        """Lazy-load the inference engine."""
        if self._engine is None:
            try:
                from ..core.engine_pool import get_engine
                self._engine = get_engine()
            except ImportError:
                from ..core.inference import EnigmaEngine
                self._engine = EnigmaEngine()
        return self._engine
    
    def _setup_routes(self):
        """Set up API routes."""
        
        @self.app.route("/v1/models", methods=["GET"])
        def list_models():
            """List available models."""
            models = self._get_available_models()
            return jsonify({
                "object": "list",
                "data": models
            })
        
        @self.app.route("/v1/models/<model_id>", methods=["GET"])
        def get_model(model_id: str):
            """Get model details."""
            models = self._get_available_models()
            for model in models:
                if model["id"] == model_id:
                    return jsonify(model)
            return jsonify({"error": {"message": f"Model {model_id} not found"}}), 404
        
        @self.app.route("/v1/chat/completions", methods=["POST"])
        def chat_completions():
            """
            Create a chat completion (GPT-4 style).
            
            Request body:
                model: str - Model to use
                messages: List[{role, content}] - Conversation history
                temperature: float - Sampling temperature (0-2)
                max_tokens: int - Maximum tokens to generate
                stream: bool - Stream response
                top_p: float - Nucleus sampling
                n: int - Number of completions (always 1)
                stop: List[str] - Stop sequences
            """
            data = request.json or {}
            
            # Parse request
            messages = data.get("messages", [])
            model = data.get("model", "forge")
            temperature = min(2.0, max(0.0, float(data.get("temperature", 1.0))))
            max_tokens = min(4096, max(1, int(data.get("max_tokens", 512))))
            stream = data.get("stream", False)
            top_p = float(data.get("top_p", 1.0))
            stop = data.get("stop", None)
            
            # Convert messages to prompt
            prompt = self._messages_to_prompt(messages)
            
            if stream:
                return self._stream_chat_response(prompt, model, temperature, max_tokens, top_p, stop)
            else:
                return self._generate_chat_response(prompt, model, temperature, max_tokens, top_p, stop)
        
        @self.app.route("/v1/completions", methods=["POST"])
        def completions():
            """
            Create a text completion (legacy API).
            
            Request body:
                model: str - Model to use
                prompt: str - Text prompt
                temperature: float - Sampling temperature
                max_tokens: int - Maximum tokens to generate
                stream: bool - Stream response
            """
            data = request.json or {}
            
            prompt = data.get("prompt", "")
            model = data.get("model", "forge")
            temperature = min(2.0, max(0.0, float(data.get("temperature", 1.0))))
            max_tokens = min(4096, max(1, int(data.get("max_tokens", 512))))
            stream = data.get("stream", False)
            
            if stream:
                return self._stream_completion_response(prompt, model, temperature, max_tokens)
            else:
                return self._generate_completion_response(prompt, model, temperature, max_tokens)
        
        @self.app.route("/v1/embeddings", methods=["POST"])
        def embeddings():
            """
            Create embeddings for text.
            
            Request body:
                model: str - Model to use
                input: str | List[str] - Text to embed
            """
            data = request.json or {}
            
            input_text = data.get("input", "")
            model = data.get("model", "text-embedding-forge")
            
            # Handle string or list input
            if isinstance(input_text, str):
                texts = [input_text]
            else:
                texts = input_text
            
            embeddings = self._generate_embeddings(texts, model)
            
            return jsonify({
                "object": "list",
                "data": embeddings,
                "model": model,
                "usage": {
                    "prompt_tokens": sum(len(t.split()) for t in texts),
                    "total_tokens": sum(len(t.split()) for t in texts)
                }
            })
        
        @self.app.route("/health", methods=["GET"])
        def health():
            """Health check endpoint."""
            return jsonify({"status": "ok", "type": "openai_compatible"})
        
        @self.app.route("/", methods=["GET"])
        def root():
            """Root endpoint with API info."""
            return jsonify({
                "name": "enigma_engine OpenAI-Compatible API",
                "version": "1.0.0",
                "endpoints": {
                    "chat": "/v1/chat/completions",
                    "completions": "/v1/completions",
                    "embeddings": "/v1/embeddings",
                    "models": "/v1/models"
                },
                "compatible_with": [
                    "OpenAI Python SDK",
                    "LangChain",
                    "LlamaIndex",
                    "AutoGPT",
                    "Any OpenAI-compatible tool"
                ]
            })
    
    def _get_available_models(self) -> list[dict]:
        """Get list of available models."""
        from pathlib import Path

        from ..config import CONFIG
        
        models = []
        models_dir = Path(CONFIG.get("models_dir", "models"))
        
        # Add main Forge model
        models.append({
            "id": "forge",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "enigma_engine",
            "permission": [],
            "root": "forge",
            "parent": None
        })
        
        # Add embedding model
        models.append({
            "id": "text-embedding-forge",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "enigma_engine",
            "permission": [],
            "root": "text-embedding-forge",
            "parent": None
        })
        
        # Scan for additional models
        if models_dir.exists():
            for model_file in models_dir.glob("*.pth"):
                model_name = model_file.stem
                if model_name not in ["forge", "text-embedding-forge"]:
                    models.append({
                        "id": model_name,
                        "object": "model",
                        "created": int(model_file.stat().st_mtime),
                        "owned_by": "enigma_engine",
                        "permission": [],
                        "root": model_name,
                        "parent": None
                    })
            
            # Also check for GGUF models
            for gguf_file in models_dir.glob("*.gguf"):
                model_name = gguf_file.stem
                models.append({
                    "id": model_name,
                    "object": "model",
                    "created": int(gguf_file.stat().st_mtime),
                    "owned_by": "enigma_engine",
                    "permission": [],
                    "root": model_name,
                    "parent": None
                })
        
        return models
    
    def _messages_to_prompt(self, messages: list[dict]) -> str:
        """
        Convert OpenAI-style messages to a prompt string.
        
        Supports common chat formats.
        """
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        # Add final assistant prompt
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def _generate_chat_response(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop: Optional[list[str]]
    ):
        """Generate a non-streaming chat response."""
        try:
            # Generate text
            text = self.engine.generate(
                prompt,
                max_gen=max_tokens,
                temperature=temperature,
                top_p=top_p if top_p < 1.0 else None,
                stop_sequences=stop
            )
            
            # Count tokens (approximate)
            prompt_tokens = len(prompt.split())
            completion_tokens = len(text.split())
            
            response = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text.strip()
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return jsonify({
                "error": {
                    "message": str(e),
                    "type": "api_error",
                    "code": "generation_failed"
                }
            }), 500
    
    def _stream_chat_response(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop: Optional[list[str]]
    ):
        """Generate a streaming chat response (SSE)."""
        def generate():
            response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            
            try:
                # Stream tokens
                for token in self.engine.stream_generate(
                    prompt,
                    max_gen=max_tokens,
                    temperature=temperature
                ):
                    chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                # Send final chunk
                final_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                error_chunk = {
                    "error": {
                        "message": str(e),
                        "type": "api_error"
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
        
        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    def _generate_completion_response(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int
    ):
        """Generate a non-streaming completion response."""
        try:
            text = self.engine.generate(
                prompt,
                max_gen=max_tokens,
                temperature=temperature
            )
            
            prompt_tokens = len(prompt.split())
            completion_tokens = len(text.split())
            
            response = {
                "id": f"cmpl-{uuid.uuid4().hex[:12]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "text": text,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({
                "error": {
                    "message": str(e),
                    "type": "api_error"
                }
            }), 500
    
    def _stream_completion_response(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int
    ):
        """Generate a streaming completion response."""
        def generate():
            response_id = f"cmpl-{uuid.uuid4().hex[:12]}"
            
            try:
                for token in self.engine.stream_generate(
                    prompt,
                    max_gen=max_tokens,
                    temperature=temperature
                ):
                    chunk = {
                        "id": response_id,
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "text": token,
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                final_chunk = {
                    "id": response_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "text": "",
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream"
        )
    
    def _generate_embeddings(self, texts: list[str], model: str) -> list[dict]:
        """Generate embeddings for texts."""
        embeddings = []
        
        try:
            # Try to use the embedding module
            from ..gui.tabs.embeddings_tab import LocalEmbedding
            
            if self._embedding_model is None:
                self._embedding_model = LocalEmbedding()
            
            for i, text in enumerate(texts):
                emb = self._embedding_model.embed(text)
                embeddings.append({
                    "object": "embedding",
                    "index": i,
                    "embedding": emb.tolist() if hasattr(emb, 'tolist') else list(emb)
                })
                
        except Exception as e:
            logger.warning(f"Embedding error, using fallback: {e}")
            # Fallback: simple hash-based embedding
            import hashlib
            
            for i, text in enumerate(texts):
                # Create deterministic pseudo-embedding
                hash_bytes = hashlib.sha256(text.encode()).digest()
                embedding = [((b - 128) / 128.0) for b in hash_bytes[:256]]
                # Pad to standard size if needed
                while len(embedding) < 384:
                    embedding.extend(embedding[:384 - len(embedding)])
                embedding = embedding[:384]
                
                embeddings.append({
                    "object": "embedding",
                    "index": i,
                    "embedding": embedding
                })
        
        return embeddings
    
    def run(self, debug: bool = False):
        """Start the server."""
        print(f"\n{'='*60}")
        print("  enigma_engine OpenAI-Compatible API Server")
        print(f"{'='*60}")
        print(f"\n  Base URL: http://{self.host}:{self.port}/v1")
        print(f"\n  Endpoints:")
        print(f"    POST /v1/chat/completions  - Chat (GPT-4 style)")
        print(f"    POST /v1/completions       - Text completion")
        print(f"    POST /v1/embeddings        - Embeddings")
        print(f"    GET  /v1/models            - List models")
        print(f"\n  Usage with OpenAI SDK:")
        print(f'    client = OpenAI(base_url="http://localhost:{self.port}/v1", api_key="not-needed")')
        print(f"\n{'='*60}\n")
        
        self.app.run(host=self.host, port=self.port, debug=debug, threaded=True)


def create_openai_server(host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
    """Create and start the OpenAI-compatible server."""
    server = OpenAICompatibleServer(host=host, port=port)
    server.run(debug=debug)
    return server


# Allow running as module
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="enigma_engine OpenAI-Compatible API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    create_openai_server(args.host, args.port, args.debug)
