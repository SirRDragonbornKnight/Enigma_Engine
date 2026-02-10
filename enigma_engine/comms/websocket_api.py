"""
WebSocket API for enigma_engine

Real-time streaming inference via WebSocket connections.

Features:
- True streaming (token-by-token)
- Bidirectional communication
- Connection multiplexing
- Automatic reconnection support
- Compatible with popular WebSocket clients

Usage:
    # Server
    from enigma_engine.comms.websocket_api import WebSocketServer
    server = WebSocketServer(model, tokenizer)
    server.start(port=8765)
    
    # Client (JavaScript)
    const ws = new WebSocket('ws://localhost:8765');
    ws.send(JSON.stringify({type: 'generate', prompt: 'Hello'}));
    ws.onmessage = (e) => console.log(JSON.parse(e.data).token);
"""

import asyncio
import json
import logging
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Try to import websockets, provide fallback
try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketServerProtocol = None  # Type stub for annotations
    logger.warning("websockets not installed. Install with: pip install websockets")


@dataclass
class WebSocketConfig:
    """Configuration for WebSocket server."""
    host: str = "0.0.0.0"
    port: int = 8765
    max_connections: int = 100
    ping_interval: float = 30.0
    ping_timeout: float = 10.0
    max_message_size: int = 1024 * 1024  # 1MB
    require_auth: bool = False
    auth_tokens: set[str] = field(default_factory=set)


@dataclass
class Message:
    """WebSocket message structure."""
    type: str  # 'generate', 'stop', 'status', 'config'
    id: str = ""  # Request ID for tracking
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, data: str) -> 'Message':
        parsed = json.loads(data)
        return cls(**parsed)


@dataclass
class StreamToken:
    """Streaming token response."""
    request_id: str
    token: str
    token_id: int
    is_finished: bool = False
    finish_reason: Optional[str] = None
    
    def to_json(self) -> str:
        return json.dumps({
            'type': 'token',
            'request_id': self.request_id,
            'token': self.token,
            'token_id': self.token_id,
            'is_finished': self.is_finished,
            'finish_reason': self.finish_reason
        })


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self, max_connections: int = 100):
        self.max_connections = max_connections
        self._connections: dict[str, Any] = {}  # connection_id -> websocket
        self._active_requests: dict[str, str] = {}  # request_id -> connection_id
        self._lock = threading.Lock()
    
    def add_connection(self, connection_id: str, websocket: Any) -> bool:
        """Add a new connection."""
        with self._lock:
            if len(self._connections) >= self.max_connections:
                return False
            self._connections[connection_id] = websocket
            return True
    
    def remove_connection(self, connection_id: str):
        """Remove a connection."""
        with self._lock:
            self._connections.pop(connection_id, None)
            # Remove any active requests for this connection
            to_remove = [
                rid for rid, cid in self._active_requests.items()
                if cid == connection_id
            ]
            for rid in to_remove:
                del self._active_requests[rid]
    
    def get_connection(self, connection_id: str) -> Optional[Any]:
        """Get a connection by ID."""
        return self._connections.get(connection_id)
    
    def register_request(self, request_id: str, connection_id: str):
        """Register an active request."""
        with self._lock:
            self._active_requests[request_id] = connection_id
    
    def unregister_request(self, request_id: str):
        """Unregister a request."""
        with self._lock:
            self._active_requests.pop(request_id, None)
    
    def get_connection_for_request(self, request_id: str) -> Optional[Any]:
        """Get the connection for a request."""
        connection_id = self._active_requests.get(request_id)
        if connection_id:
            return self._connections.get(connection_id)
        return None
    
    @property
    def num_connections(self) -> int:
        return len(self._connections)


class WebSocketServer:
    """
    WebSocket server for streaming inference.
    
    Protocol:
    
    Client -> Server:
        {
            "type": "generate",
            "id": "request-123",
            "data": {
                "prompt": "Hello, how are you?",
                "max_tokens": 100,
                "temperature": 0.7,
                "stream": true
            }
        }
    
    Server -> Client (streaming):
        {"type": "token", "request_id": "request-123", "token": "I", "token_id": 42}
        {"type": "token", "request_id": "request-123", "token": "'m", "token_id": 55}
        ...
        {"type": "token", "request_id": "request-123", "token": "", "is_finished": true}
    
    Client -> Server (stop):
        {"type": "stop", "data": {"request_id": "request-123"}}
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[WebSocketConfig] = None
    ):
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets not installed. Run: pip install websockets")
        
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or WebSocketConfig()
        
        self.connection_manager = ConnectionManager(self.config.max_connections)
        self._stop_requests: set[str] = set()
        self._server = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle a WebSocket connection."""
        connection_id = str(uuid.uuid4())
        
        # Check auth if required
        if self.config.require_auth:
            try:
                auth_msg = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                auth_data = json.loads(auth_msg)
                if auth_data.get('token') not in self.config.auth_tokens:
                    await websocket.close(4001, "Unauthorized")
                    return
            except Exception as e:
                await websocket.close(4001, "Auth failed")
                return
        
        # Add connection
        if not self.connection_manager.add_connection(connection_id, websocket):
            await websocket.close(4002, "Too many connections")
            return
        
        logger.info(f"New connection: {connection_id}")
        
        try:
            async for message in websocket:
                await self._handle_message(connection_id, websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connection_manager.remove_connection(connection_id)
            logger.info(f"Connection closed: {connection_id}")
    
    async def _handle_message(
        self,
        connection_id: str,
        websocket: WebSocketServerProtocol,
        raw_message: str
    ):
        """Handle an incoming message."""
        try:
            msg = Message.from_json(raw_message)
        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                'type': 'error',
                'error': 'Invalid JSON'
            }))
            return
        
        if msg.type == 'generate':
            await self._handle_generate(connection_id, websocket, msg)
        elif msg.type == 'stop':
            self._handle_stop(msg)
        elif msg.type == 'status':
            await self._handle_status(websocket)
        elif msg.type == 'ping':
            await websocket.send(json.dumps({'type': 'pong', 'id': msg.id}))
        else:
            await websocket.send(json.dumps({
                'type': 'error',
                'error': f'Unknown message type: {msg.type}'
            }))
    
    async def _handle_generate(
        self,
        connection_id: str,
        websocket: WebSocketServerProtocol,
        msg: Message
    ):
        """Handle a generation request."""
        request_id = msg.id
        data = msg.data
        
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 100)
        temperature = data.get('temperature', 1.0)
        stream = data.get('stream', True)
        
        self.connection_manager.register_request(request_id, connection_id)
        
        try:
            if stream:
                await self._stream_generate(websocket, request_id, prompt, max_tokens, temperature)
            else:
                await self._batch_generate(websocket, request_id, prompt, max_tokens, temperature)
        finally:
            self.connection_manager.unregister_request(request_id)
            self._stop_requests.discard(request_id)
    
    async def _stream_generate(
        self,
        websocket: WebSocketServerProtocol,
        request_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float
    ):
        """Stream tokens one at a time."""
        import torch
        
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(device)
        generated = input_ids.clone()
        
        self.model.eval()
        
        with torch.inference_mode():
            for i in range(max_tokens):
                # Check for stop request
                if request_id in self._stop_requests:
                    token_msg = StreamToken(
                        request_id=request_id,
                        token="",
                        token_id=-1,
                        is_finished=True,
                        finish_reason="stop_requested"
                    )
                    await websocket.send(token_msg.to_json())
                    return
                
                # Generate next token
                outputs = self.model(generated)
                logits = outputs[:, -1, :] / temperature
                
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                token_id = next_token.item()
                token_text = self.tokenizer.decode([token_id])
                
                # Send token
                token_msg = StreamToken(
                    request_id=request_id,
                    token=token_text,
                    token_id=token_id
                )
                await websocket.send(token_msg.to_json())
                
                # Small yield to allow other tasks
                await asyncio.sleep(0)
        
        # Send completion
        done_msg = StreamToken(
            request_id=request_id,
            token="",
            token_id=-1,
            is_finished=True,
            finish_reason="length"
        )
        await websocket.send(done_msg.to_json())
    
    async def _batch_generate(
        self,
        websocket: WebSocketServerProtocol,
        request_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float
    ):
        """Generate all tokens and send as single response."""
        import torch
        
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(device)
        generated = input_ids.clone()
        
        self.model.eval()
        
        with torch.inference_mode():
            for _ in range(max_tokens):
                if request_id in self._stop_requests:
                    break
                
                outputs = self.model(generated)
                logits = outputs[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
        
        # Decode full response
        output_ids = generated[0].tolist()
        text = self.tokenizer.decode(output_ids)
        
        await websocket.send(json.dumps({
            'type': 'complete',
            'request_id': request_id,
            'text': text,
            'tokens_generated': len(output_ids) - len(input_ids[0])
        }))
    
    def _handle_stop(self, msg: Message):
        """Handle a stop request."""
        request_id = msg.data.get('request_id')
        if request_id:
            self._stop_requests.add(request_id)
            logger.info(f"Stop requested for {request_id}")
    
    async def _handle_status(self, websocket: WebSocketServerProtocol):
        """Send server status."""
        status = {
            'type': 'status',
            'connections': self.connection_manager.num_connections,
            'max_connections': self.config.max_connections,
        }
        await websocket.send(json.dumps(status))
    
    def start(self, blocking: bool = True):
        """Start the WebSocket server."""
        async def run_server():
            self._server = await websockets.serve(
                self.handle_connection,
                self.config.host,
                self.config.port,
                ping_interval=self.config.ping_interval,
                ping_timeout=self.config.ping_timeout,
                max_size=self.config.max_message_size
            )
            logger.info(f"WebSocket server started on ws://{self.config.host}:{self.config.port}")
            await self._server.wait_closed()
        
        if blocking:
            asyncio.run(run_server())
        else:
            self._loop = asyncio.new_event_loop()
            thread = threading.Thread(target=self._loop.run_until_complete, args=(run_server(),))
            thread.daemon = True
            thread.start()
    
    def stop(self):
        """Stop the WebSocket server."""
        if self._server:
            self._server.close()


class WebSocketClient:
    """
    WebSocket client for connecting to enigma_engine servers.
    
    Usage:
        client = WebSocketClient("ws://localhost:8765")
        
        async def main():
            await client.connect()
            
            async for token in client.generate("Hello!", max_tokens=50):
                print(token, end="", flush=True)
            
            await client.disconnect()
        
        asyncio.run(main())
    """
    
    def __init__(self, uri: str):
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets not installed")
        
        self.uri = uri
        self._websocket = None
    
    async def connect(self, auth_token: Optional[str] = None):
        """Connect to server."""
        self._websocket = await websockets.connect(self.uri)
        
        if auth_token:
            await self._websocket.send(json.dumps({'token': auth_token}))
    
    async def disconnect(self):
        """Disconnect from server."""
        if self._websocket:
            await self._websocket.close()
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        stream: bool = True
    ):
        """
        Generate text from prompt.
        
        Yields tokens if streaming, otherwise returns full text.
        """
        request_id = str(uuid.uuid4())
        
        msg = Message(
            type='generate',
            id=request_id,
            data={
                'prompt': prompt,
                'max_tokens': max_tokens,
                'temperature': temperature,
                'stream': stream
            }
        )
        
        await self._websocket.send(msg.to_json())
        
        if stream:
            async for response in self._websocket:
                data = json.loads(response)
                
                if data.get('type') == 'token':
                    if data.get('is_finished'):
                        return
                    yield data.get('token', '')
                elif data.get('type') == 'error':
                    raise RuntimeError(data.get('error', 'Unknown WebSocket error'))
        else:
            response = await self._websocket.recv()
            data = json.loads(response)
            yield data.get('text', '')
    
    async def stop(self, request_id: str):
        """Stop a generation request."""
        msg = Message(type='stop', data={'request_id': request_id})
        await self._websocket.send(msg.to_json())


def create_websocket_server(
    model: Any,
    tokenizer: Any,
    port: int = 8765,
    require_auth: bool = False
) -> WebSocketServer:
    """
    Create a WebSocket server.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        port: Port to listen on
        require_auth: Whether to require authentication
    
    Returns:
        WebSocketServer instance
    """
    config = WebSocketConfig(port=port, require_auth=require_auth)
    return WebSocketServer(model, tokenizer, config)
