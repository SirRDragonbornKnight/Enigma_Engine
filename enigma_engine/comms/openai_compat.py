"""
OpenAI API Compatibility Layer

Full OpenAI API v1 compatibility for easy integration
with existing OpenAI-compatible tools and libraries.

FILE: enigma_engine/comms/openai_compat.py
TYPE: API/Server
MAIN CLASSES: OpenAIServer, ChatCompletion, Embedding
"""

import json
import logging
import time
import uuid
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any, Optional, Union

try:
    from flask import Flask, Response, jsonify, request, stream_with_context
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OpenAIConfig:
    """OpenAI API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str = ""  # Empty = no auth required
    organization: str = "forge-ai"
    max_tokens_default: int = 2048
    default_model: str = "forge-ai"


@dataclass
class ModelInfo:
    """Model information for /v1/models endpoint."""
    id: str
    created: int
    owned_by: str = "forge-ai"
    object: str = "model"
    permission: list[dict] = field(default_factory=list)
    root: str = ""
    parent: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "owned_by": self.owned_by,
            "permission": self.permission or [{
                "id": f"modelperm-{self.id}",
                "object": "model_permission",
                "created": self.created,
                "allow_create_engine": False,
                "allow_sampling": True,
                "allow_logprobs": True,
                "allow_search_indices": False,
                "allow_view": True,
                "allow_fine_tuning": False,
                "organization": "*",
                "group": None,
                "is_blocking": False
            }],
            "root": self.root or self.id,
            "parent": self.parent
        }


@dataclass
class Usage:
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }


@dataclass
class ChatMessage:
    """Chat message format."""
    role: str  # system, user, assistant, function
    content: str
    name: Optional[str] = None
    function_call: Optional[dict] = None
    
    def to_dict(self) -> dict[str, Any]:
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.function_call:
            d["function_call"] = self.function_call
        return d


@dataclass
class Choice:
    """Chat/completion choice."""
    index: int
    message: Optional[ChatMessage] = None
    text: Optional[str] = None
    finish_reason: str = "stop"
    logprobs: Optional[dict] = None
    
    def to_dict(self, is_chat: bool = True) -> dict[str, Any]:
        if is_chat:
            return {
                "index": self.index,
                "message": self.message.to_dict() if self.message else None,
                "finish_reason": self.finish_reason
            }
        else:
            return {
                "index": self.index,
                "text": self.text,
                "finish_reason": self.finish_reason,
                "logprobs": self.logprobs
            }


@dataclass
class ChatCompletionResponse:
    """OpenAI chat completion response format."""
    id: str
    choices: list[Choice]
    created: int
    model: str
    usage: Usage
    object: str = "chat.completion"
    system_fingerprint: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [c.to_dict(is_chat=True) for c in self.choices],
            "usage": self.usage.to_dict(),
            "system_fingerprint": self.system_fingerprint or f"fp_{uuid.uuid4().hex[:8]}"
        }


@dataclass
class CompletionResponse:
    """OpenAI completion response format."""
    id: str
    choices: list[Choice]
    created: int
    model: str
    usage: Usage
    object: str = "text_completion"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [c.to_dict(is_chat=False) for c in self.choices],
            "usage": self.usage.to_dict()
        }


@dataclass
class EmbeddingData:
    """Single embedding result."""
    embedding: list[float]
    index: int
    object: str = "embedding"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "object": self.object,
            "embedding": self.embedding,
            "index": self.index
        }


@dataclass
class EmbeddingResponse:
    """OpenAI embedding response format."""
    data: list[EmbeddingData]
    model: str
    usage: Usage
    object: str = "list"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "object": self.object,
            "data": [d.to_dict() for d in self.data],
            "model": self.model,
            "usage": self.usage.to_dict()
        }


class OpenAIServer:
    """
    OpenAI API-compatible server.
    
    Provides full compatibility with OpenAI API v1 endpoints:
    - /v1/models
    - /v1/chat/completions
    - /v1/completions
    - /v1/embeddings
    """
    
    def __init__(
        self,
        inference_fn: callable = None,
        embedding_fn: callable = None,
        tokenizer: Any = None,
        config: OpenAIConfig = None
    ):
        """
        Initialize OpenAI-compatible server.
        
        Args:
            inference_fn: Function to generate text (prompt) -> response
            embedding_fn: Function to generate embeddings (text) -> List[float]
            tokenizer: Tokenizer for counting tokens
            config: Server configuration
        """
        self.inference_fn = inference_fn
        self.embedding_fn = embedding_fn
        self.tokenizer = tokenizer
        self.config = config or OpenAIConfig()
        
        self._models: dict[str, ModelInfo] = {}
        self._register_default_models()
        
        if HAS_FLASK:
            self.app = Flask(__name__)
            self._setup_routes()
        else:
            self.app = None
            logger.warning("Flask not available - server cannot start")
    
    def _register_default_models(self):
        """Register default models."""
        now = int(time.time())
        
        self._models["forge-ai"] = ModelInfo(
            id="forge-ai",
            created=now,
            owned_by="forge-ai"
        )
        
        # Alias compatibility
        self._models["gpt-3.5-turbo"] = ModelInfo(
            id="gpt-3.5-turbo",
            created=now,
            owned_by="forge-ai",
            root="forge-ai"
        )
        
        self._models["gpt-4"] = ModelInfo(
            id="gpt-4",
            created=now,
            owned_by="forge-ai",
            root="forge-ai"
        )
        
        self._models["text-embedding-ada-002"] = ModelInfo(
            id="text-embedding-ada-002",
            created=now,
            owned_by="forge-ai"
        )
    
    def register_model(self, model_id: str, model_info: ModelInfo = None):
        """Register a new model."""
        if model_info is None:
            model_info = ModelInfo(
                id=model_id,
                created=int(time.time())
            )
        self._models[model_id] = model_info
    
    def _setup_routes(self):
        """Set up Flask routes."""
        
        @self.app.before_request
        def check_auth():
            """Check API key authentication."""
            if self.config.api_key:
                auth_header = request.headers.get("Authorization", "")
                if not auth_header.startswith("Bearer "):
                    return jsonify({"error": {"message": "Missing API key", "type": "invalid_request_error"}}), 401
                
                provided_key = auth_header[7:]
                if provided_key != self.config.api_key:
                    return jsonify({"error": {"message": "Invalid API key", "type": "invalid_request_error"}}), 401
        
        @self.app.route("/v1/models", methods=["GET"])
        def list_models():
            """List available models."""
            return jsonify({
                "object": "list",
                "data": [m.to_dict() for m in self._models.values()]
            })
        
        @self.app.route("/v1/models/<model_id>", methods=["GET"])
        def get_model(model_id):
            """Get model details."""
            if model_id not in self._models:
                return jsonify({
                    "error": {"message": f"Model '{model_id}' not found", "type": "invalid_request_error"}
                }), 404
            return jsonify(self._models[model_id].to_dict())
        
        @self.app.route("/v1/chat/completions", methods=["POST"])
        def chat_completions():
            """Chat completions endpoint."""
            data = request.json
            
            # Extract parameters
            messages = data.get("messages", [])
            model = data.get("model", self.config.default_model)
            temperature = data.get("temperature", 1.0)
            max_tokens = data.get("max_tokens", self.config.max_tokens_default)
            stream = data.get("stream", False)
            n = data.get("n", 1)
            stop = data.get("stop", None)
            
            # Build prompt from messages
            prompt = self._messages_to_prompt(messages)
            
            if stream:
                return Response(
                    stream_with_context(self._stream_response(prompt, model, max_tokens, temperature)),
                    mimetype="text/event-stream"
                )
            
            # Generate response
            response_text = self._generate(prompt, max_tokens, temperature, stop)
            
            # Count tokens
            prompt_tokens = self._count_tokens(prompt)
            completion_tokens = self._count_tokens(response_text)
            
            response = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                choices=[Choice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop"
                )],
                created=int(time.time()),
                model=model,
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens
                )
            )
            
            return jsonify(response.to_dict())
        
        @self.app.route("/v1/completions", methods=["POST"])
        def completions():
            """Text completions endpoint."""
            data = request.json
            
            prompt = data.get("prompt", "")
            model = data.get("model", self.config.default_model)
            max_tokens = data.get("max_tokens", self.config.max_tokens_default)
            temperature = data.get("temperature", 1.0)
            stream = data.get("stream", False)
            n = data.get("n", 1)
            stop = data.get("stop", None)
            echo = data.get("echo", False)
            
            if stream:
                return Response(
                    stream_with_context(self._stream_completion(prompt, model, max_tokens, temperature)),
                    mimetype="text/event-stream"
                )
            
            response_text = self._generate(prompt, max_tokens, temperature, stop)
            
            if echo:
                response_text = prompt + response_text
            
            prompt_tokens = self._count_tokens(prompt)
            completion_tokens = self._count_tokens(response_text)
            
            response = CompletionResponse(
                id=f"cmpl-{uuid.uuid4().hex}",
                choices=[Choice(
                    index=0,
                    text=response_text,
                    finish_reason="stop"
                )],
                created=int(time.time()),
                model=model,
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens
                )
            )
            
            return jsonify(response.to_dict())
        
        @self.app.route("/v1/embeddings", methods=["POST"])
        def embeddings():
            """Embeddings endpoint."""
            data = request.json
            
            input_data = data.get("input", "")
            model = data.get("model", "text-embedding-ada-002")
            
            # Handle single string or list
            if isinstance(input_data, str):
                inputs = [input_data]
            else:
                inputs = input_data
            
            embeddings_list = []
            total_tokens = 0
            
            for i, text in enumerate(inputs):
                embedding = self._get_embedding(text)
                embeddings_list.append(EmbeddingData(
                    embedding=embedding,
                    index=i
                ))
                total_tokens += self._count_tokens(text)
            
            response = EmbeddingResponse(
                data=embeddings_list,
                model=model,
                usage=Usage(
                    prompt_tokens=total_tokens,
                    completion_tokens=0,
                    total_tokens=total_tokens
                )
            )
            
            return jsonify(response.to_dict())
        
        @self.app.route("/health", methods=["GET"])
        @self.app.route("/v1/health", methods=["GET"])
        def health():
            """Health check endpoint."""
            return jsonify({"status": "healthy", "timestamp": int(time.time())})
    
    def _messages_to_prompt(self, messages: list[dict]) -> str:
        """Convert chat messages to a prompt string."""
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
            elif role == "function":
                name = msg.get("name", "function")
                prompt_parts.append(f"Function ({name}): {content}")
        
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)
    
    def _generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stop: Union[str, list[str]] = None
    ) -> str:
        """Generate text using inference function."""
        if self.inference_fn is None:
            return "[No inference function configured]"
        
        try:
            response = self.inference_fn(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Apply stop sequences
            if stop:
                if isinstance(stop, str):
                    stop = [stop]
                for s in stop:
                    if s in response:
                        response = response.split(s)[0]
            
            return response
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"[Error: {str(e)}]"
    
    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text."""
        if self.embedding_fn is None:
            # Return dummy embedding
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            # Generate pseudo-random embedding from hash
            embedding = []
            for i in range(0, min(len(h), 1536), 4):
                val = int.from_bytes(h[i:i+4], 'big') / (2**32) - 0.5
                embedding.append(val)
            # Pad to 1536 dimensions
            while len(embedding) < 1536:
                embedding.append(0.0)
            return embedding[:1536]
        
        try:
            return self.embedding_fn(text)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return [0.0] * 1536
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer is not None:
            try:
                if hasattr(self.tokenizer, 'encode'):
                    return len(self.tokenizer.encode(text))
            except (ValueError, RuntimeError):
                pass
        # Rough estimate: ~4 chars per token
        return len(text) // 4
    
    def _stream_response(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float
    ) -> Generator[str, None, None]:
        """Stream chat completion response."""
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        
        # Generate full response
        response_text = self._generate(prompt, max_tokens, temperature, None)
        
        # Stream token by token
        for i, char in enumerate(response_text):
            chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": char} if i > 0 else {"role": "assistant", "content": char},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        
        # Final chunk
        final = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"
    
    def _stream_completion(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float
    ) -> Generator[str, None, None]:
        """Stream text completion response."""
        response_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        
        response_text = self._generate(prompt, max_tokens, temperature, None)
        
        for char in response_text:
            chunk = {
                "id": response_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "text": char,
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        
        final = {
            "id": response_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "text": "",
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"
    
    def run(self, host: str = None, port: int = None, debug: bool = False):
        """Run the server."""
        if self.app is None:
            raise RuntimeError("Flask not available")
        
        self.app.run(
            host=host or self.config.host,
            port=port or self.config.port,
            debug=debug,
            threaded=True
        )


def create_openai_server(
    inference_fn: callable = None,
    embedding_fn: callable = None,
    tokenizer: Any = None,
    api_key: str = "",
    port: int = 8000
) -> OpenAIServer:
    """
    Create an OpenAI-compatible API server.
    
    Args:
        inference_fn: Text generation function
        embedding_fn: Embedding generation function
        tokenizer: Tokenizer for token counting
        api_key: API key (empty = no auth)
        port: Server port
    
    Returns:
        OpenAIServer instance
    """
    config = OpenAIConfig(port=port, api_key=api_key)
    return OpenAIServer(
        inference_fn=inference_fn,
        embedding_fn=embedding_fn,
        tokenizer=tokenizer,
        config=config
    )
