"""
Streaming Inference API for Enigma AI Engine

Token-by-token streaming responses for real-time output.

Features:
- Server-sent events (SSE) for streaming
- WebSocket support
- Async generation
- Callback-based streaming
- Progress tracking
- Cancellation support

Usage:
    # Server-side
    from enigma_engine.comms.streaming_api import StreamingInference, create_streaming_server
    
    server = create_streaming_server(port=8080)
    server.run()
    
    # Client-side
    from enigma_engine.comms.streaming_api import StreamingClient
    
    client = StreamingClient("http://localhost:8080")
    
    # Callback-based streaming
    for token in client.generate_stream("Tell me a story"):
        print(token, end='', flush=True)
"""

import asyncio
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, AsyncGenerator, Callable, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# Try imports
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    from flask import Flask, Response, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False


class StreamEvent(Enum):
    """Types of streaming events."""
    TOKEN = auto()          # New token generated
    START = auto()          # Generation started
    END = auto()            # Generation complete
    ERROR = auto()          # Error occurred
    PROGRESS = auto()       # Progress update
    METADATA = auto()       # Additional metadata


@dataclass
class StreamChunk:
    """A single chunk in the stream."""
    event: StreamEvent
    data: str = ""
    token_id: Optional[int] = None
    index: int = 0
    total_tokens: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_sse(self) -> str:
        """Convert to Server-Sent Event format."""
        return f"event: {self.event.name.lower()}\ndata: {json.dumps(self.to_dict())}\n\n"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'event': self.event.name,
            'data': self.data,
            'token_id': self.token_id,
            'index': self.index,
            'total_tokens': self.total_tokens,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'StreamChunk':
        """Create from dictionary."""
        return cls(
            event=StreamEvent[d.get('event', 'TOKEN')],
            data=d.get('data', ''),
            token_id=d.get('token_id'),
            index=d.get('index', 0),
            total_tokens=d.get('total_tokens', 0),
            timestamp=d.get('timestamp', time.time()),
            metadata=d.get('metadata', {})
        )


class StreamingInference:
    """
    Streaming inference wrapper.
    
    Wraps the inference engine to provide token-by-token streaming.
    """
    
    def __init__(self, engine=None):
        """
        Initialize streaming inference.
        
        Args:
            engine: EnigmaEngine instance (will try to get default if None)
        """
        self._engine = engine
        self._active_streams: Dict[str, bool] = {}  # stream_id -> running
        self._lock = threading.Lock()
        
        # Try to get engine
        if self._engine is None:
            try:
                from enigma_engine.core.inference import get_engine
                self._engine = get_engine()
            except ImportError:
                logger.warning("Inference engine not available")
    
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: Optional[List[str]] = None,
        callback: Optional[Callable[[StreamChunk], None]] = None,
        stream_id: Optional[str] = None
    ) -> Generator[StreamChunk, None, None]:
        """
        Generate tokens in a streaming fashion.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop_sequences: Stop generation on these sequences
            callback: Optional callback for each chunk
            stream_id: Optional ID for cancellation
            
        Yields:
            StreamChunk objects
        """
        # Generate stream ID
        if stream_id is None:
            stream_id = f"stream_{int(time.time() * 1000)}"
        
        with self._lock:
            self._active_streams[stream_id] = True
        
        try:
            # Start event
            start_chunk = StreamChunk(
                event=StreamEvent.START,
                metadata={'prompt_length': len(prompt), 'max_tokens': max_tokens}
            )
            if callback:
                callback(start_chunk)
            yield start_chunk
            
            # Generate tokens
            generated_text = ""
            token_count = 0
            
            if self._engine is not None:
                # Use actual engine if available
                for token_data in self._stream_from_engine(
                    prompt, max_tokens, temperature, top_p, stop_sequences
                ):
                    # Check if cancelled
                    with self._lock:
                        if not self._active_streams.get(stream_id, False):
                            break
                    
                    token = token_data.get('token', '')
                    token_id = token_data.get('token_id')
                    
                    generated_text += token
                    token_count += 1
                    
                    chunk = StreamChunk(
                        event=StreamEvent.TOKEN,
                        data=token,
                        token_id=token_id,
                        index=token_count,
                        total_tokens=token_count,
                        metadata={'cumulative_text': generated_text}
                    )
                    
                    if callback:
                        callback(chunk)
                    yield chunk
            else:
                # Demo mode - simulate streaming
                demo_response = f"[Demo response for: {prompt[:50]}...]"
                for i, char in enumerate(demo_response):
                    # Check cancellation
                    with self._lock:
                        if not self._active_streams.get(stream_id, False):
                            break
                    
                    generated_text += char
                    token_count += 1
                    
                    chunk = StreamChunk(
                        event=StreamEvent.TOKEN,
                        data=char,
                        index=token_count,
                        total_tokens=len(demo_response)
                    )
                    
                    if callback:
                        callback(chunk)
                    yield chunk
                    
                    time.sleep(0.02)  # Simulate generation time
            
            # End event
            end_chunk = StreamChunk(
                event=StreamEvent.END,
                data=generated_text,
                total_tokens=token_count,
                metadata={'complete': True}
            )
            if callback:
                callback(end_chunk)
            yield end_chunk
            
        except Exception as e:
            error_chunk = StreamChunk(
                event=StreamEvent.ERROR,
                data=str(e),
                metadata={'error_type': type(e).__name__}
            )
            if callback:
                callback(error_chunk)
            yield error_chunk
            
        finally:
            with self._lock:
                if stream_id in self._active_streams:
                    del self._active_streams[stream_id]
    
    def _stream_from_engine(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: Optional[List[str]]
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream from the actual inference engine."""
        if hasattr(self._engine, 'generate_stream'):
            # Engine supports streaming
            yield from self._engine.generate_stream(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop_sequences
            )
        else:
            # Fallback: generate all at once, then stream
            result = self._engine.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            # Simulate streaming
            for i, char in enumerate(result):
                yield {'token': char, 'token_id': None}
                time.sleep(0.01)
    
    def cancel_stream(self, stream_id: str) -> bool:
        """
        Cancel an active stream.
        
        Args:
            stream_id: ID of stream to cancel
            
        Returns:
            True if cancelled, False if not found
        """
        with self._lock:
            if stream_id in self._active_streams:
                self._active_streams[stream_id] = False
                return True
        return False
    
    def get_active_streams(self) -> List[str]:
        """Get list of active stream IDs."""
        with self._lock:
            return [k for k, v in self._active_streams.items() if v]


class StreamingClient:
    """
    Client for consuming streaming inference API.
    """
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        """
        Initialize streaming client.
        
        Args:
            base_url: API server URL
        """
        self._base_url = base_url.rstrip('/')
    
    def generate_stream(
        self,
        prompt: str,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Generate tokens from remote API.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Yields:
            Token strings
        """
        import urllib.request
        import urllib.parse
        
        url = f"{self._base_url}/v1/completions/stream"
        
        data = json.dumps({
            'prompt': prompt,
            **kwargs
        }).encode('utf-8')
        
        req = urllib.request.Request(
            url,
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                for line in response:
                    line = line.decode('utf-8').strip()
                    
                    if line.startswith('data:'):
                        json_str = line[5:].strip()
                        if json_str:
                            chunk = json.loads(json_str)
                            if chunk.get('event') == 'TOKEN':
                                yield chunk.get('data', '')
                            elif chunk.get('event') == 'END':
                                break
                            elif chunk.get('event') == 'ERROR':
                                raise Exception(chunk.get('data', 'Unknown error'))
                                
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate complete response (non-streaming).
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            Complete response text
        """
        tokens = list(self.generate_stream(prompt, **kwargs))
        return ''.join(tokens)


def create_streaming_server(
    port: int = 8080,
    host: str = "0.0.0.0",
    engine=None
) -> 'StreamingServer':
    """
    Create a streaming inference server.
    
    Args:
        port: Port to listen on
        host: Host to bind to
        engine: Optional inference engine
        
    Returns:
        StreamingServer instance
    """
    return StreamingServer(port=port, host=host, engine=engine)


class StreamingServer:
    """
    HTTP server for streaming inference.
    
    Provides SSE endpoints for streaming token generation.
    """
    
    def __init__(
        self,
        port: int = 8080,
        host: str = "0.0.0.0",
        engine=None
    ):
        """
        Initialize streaming server.
        
        Args:
            port: Port to listen on
            host: Host to bind to
            engine: Inference engine
        """
        self._port = port
        self._host = host
        self._streaming = StreamingInference(engine)
        self._app = None
        
        if FLASK_AVAILABLE:
            self._setup_flask()
    
    def _setup_flask(self):
        """Set up Flask routes."""
        self._app = Flask(__name__)
        
        @self._app.route('/v1/completions/stream', methods=['POST'])
        def stream_completion():
            data = request.get_json() or {}
            prompt = data.get('prompt', '')
            
            def generate():
                for chunk in self._streaming.generate_stream(
                    prompt=prompt,
                    max_tokens=data.get('max_tokens', 256),
                    temperature=data.get('temperature', 0.7),
                    top_p=data.get('top_p', 0.9),
                    stop_sequences=data.get('stop', None)
                ):
                    yield chunk.to_sse()
            
            return Response(
                generate(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                }
            )
        
        @self._app.route('/v1/completions', methods=['POST'])
        def completion():
            data = request.get_json() or {}
            prompt = data.get('prompt', '')
            
            tokens = []
            for chunk in self._streaming.generate_stream(
                prompt=prompt,
                max_tokens=data.get('max_tokens', 256),
                temperature=data.get('temperature', 0.7),
            ):
                if chunk.event == StreamEvent.TOKEN:
                    tokens.append(chunk.data)
            
            return jsonify({
                'text': ''.join(tokens),
                'usage': {
                    'prompt_tokens': len(prompt.split()),
                    'completion_tokens': len(tokens)
                }
            })
        
        @self._app.route('/v1/streams', methods=['GET'])
        def list_streams():
            return jsonify({
                'active': self._streaming.get_active_streams()
            })
        
        @self._app.route('/v1/streams/<stream_id>', methods=['DELETE'])
        def cancel_stream(stream_id):
            cancelled = self._streaming.cancel_stream(stream_id)
            return jsonify({'cancelled': cancelled})
        
        @self._app.route('/health', methods=['GET'])
        def health():
            return jsonify({'status': 'ok'})
    
    def run(self, debug: bool = False):
        """
        Run the server.
        
        Args:
            debug: Enable debug mode
        """
        if self._app:
            logger.info(f"Starting streaming server on {self._host}:{self._port}")
            self._app.run(host=self._host, port=self._port, debug=debug, threaded=True)
        else:
            logger.error("Flask not available, cannot run server")
    
    def run_async(self):
        """Run server in background thread."""
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        return thread


# Async support
async def async_generate_stream(
    prompt: str,
    streaming: Optional[StreamingInference] = None,
    **kwargs
) -> AsyncGenerator[StreamChunk, None]:
    """
    Async generator for streaming inference.
    
    Args:
        prompt: Input prompt
        streaming: StreamingInference instance
        **kwargs: Generation parameters
        
    Yields:
        StreamChunk objects
    """
    if streaming is None:
        streaming = StreamingInference()
    
    # Run sync generator in thread pool
    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()
    
    def producer():
        for chunk in streaming.generate_stream(prompt, **kwargs):
            loop.call_soon_threadsafe(q.put_nowait, chunk)
        loop.call_soon_threadsafe(q.put_nowait, None)  # End marker
    
    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    
    while True:
        chunk = await q.get()
        if chunk is None:
            break
        yield chunk


# Global instance
_streaming_instance: Optional[StreamingInference] = None


def get_streaming_inference() -> StreamingInference:
    """Get or create global StreamingInference instance."""
    global _streaming_instance
    if _streaming_instance is None:
        _streaming_instance = StreamingInference()
    return _streaming_instance


def stream(prompt: str, **kwargs) -> Generator[str, None, None]:
    """Quick function to stream tokens."""
    for chunk in get_streaming_inference().generate_stream(prompt, **kwargs):
        if chunk.event == StreamEvent.TOKEN:
            yield chunk.data
