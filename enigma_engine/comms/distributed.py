"""
Distributed Forge Protocol - Connect Pi, Phone, PC, and Datacenter.

This provides a unified protocol for connecting any Enigma AI Engine instance to any other,
with automatic role negotiation based on device capabilities.

Architecture:
    [Raspberry Pi] -----> [Desktop PC] <----> [Datacenter]
         ^                     |
         |                     v
    [Mobile Phone] <----- [Avatar Display]

Roles:
    - INFERENCE_CLIENT: Sends prompts, receives responses (Pi, Phone)
    - INFERENCE_SERVER: Processes prompts, generates responses (PC, Server)
    - AVATAR_DISPLAY: Shows avatar, minimal processing (Phone, PC)
    - TRAINING_NODE: Participates in distributed training (PC, Server)
    - RELAY_NODE: Routes messages between devices (PC)

Usage:
    # On PC (as server)
    from enigma_engine.comms.distributed import DistributedNode, NodeRole
    
    node = DistributedNode("pc_server", role=NodeRole.INFERENCE_SERVER)
    node.start()
    
    # On Pi (as client)
    node = DistributedNode("pi_client", role=NodeRole.INFERENCE_CLIENT)
    node.connect("192.168.1.100:5000")
    response = node.generate("Hello world")  # Processed on PC
"""

import hashlib
import hmac
import json
import logging
import queue
import socket
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class NodeRole(Enum):
    """Device role in the distributed network."""
    AUTO = auto()              # Auto-detect based on hardware
    INFERENCE_CLIENT = auto()  # Sends prompts to servers
    INFERENCE_SERVER = auto()  # Processes prompts locally
    AVATAR_DISPLAY = auto()    # Displays avatar UI
    TRAINING_NODE = auto()     # Participates in training
    RELAY_NODE = auto()        # Routes between networks
    HYBRID = auto()            # Multiple roles


class MessageType(Enum):
    """Types of messages in the protocol."""
    # Discovery
    ANNOUNCE = "announce"           # Node announcing presence
    DISCOVER = "discover"           # Looking for nodes
    HEARTBEAT = "heartbeat"         # Keep-alive
    
    # Inference
    GENERATE_REQUEST = "gen_req"    # Request generation
    GENERATE_RESPONSE = "gen_resp"  # Generation result
    GENERATE_STREAM = "gen_stream"  # Streaming token
    
    # Avatar
    AVATAR_UPDATE = "avatar_upd"    # Avatar state change
    AVATAR_SYNC = "avatar_sync"     # Sync avatar state
    
    # Memory
    MEMORY_SYNC = "mem_sync"        # Sync conversation memory
    
    # Training
    TRAIN_REQUEST = "train_req"     # Request training participation
    GRADIENT_SHARE = "grad_share"   # Share gradients
    MODEL_SYNC = "model_sync"       # Sync model weights
    
    # Control
    PING = "ping"
    PONG = "pong"
    ERROR = "error"


@dataclass
class NodeInfo:
    """Information about a network node."""
    node_id: str
    name: str
    role: NodeRole
    address: str
    port: int
    device_class: str = "unknown"
    capabilities: list[str] = field(default_factory=list)
    last_seen: str = ""
    latency_ms: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d['role'] = self.role.name
        return d
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'NodeInfo':
        d['role'] = NodeRole[d['role']] if isinstance(d['role'], str) else d['role']
        return cls(**d)


@dataclass
class ProtocolMessage:
    """A message in the distributed protocol."""
    msg_type: MessageType
    payload: dict[str, Any]
    sender_id: str
    target_id: str = "*"  # "*" = broadcast
    timestamp: str = ""
    msg_id: str = ""
    signature: str = ""  # For authenticated messages
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
        if not self.msg_id:
            self.msg_id = f"{self.sender_id}_{int(time.time()*1000)}"
    
    def to_json(self) -> str:
        d = {
            'type': self.msg_type.value,
            'payload': self.payload,
            'sender': self.sender_id,
            'target': self.target_id,
            'ts': self.timestamp,
            'id': self.msg_id,
        }
        if self.signature:
            d['sig'] = self.signature
        return json.dumps(d)
    
    @classmethod
    def from_json(cls, data: str) -> 'ProtocolMessage':
        d = json.loads(data)
        return cls(
            msg_type=MessageType(d['type']),
            payload=d.get('payload', {}),
            sender_id=d['sender'],
            target_id=d.get('target', '*'),
            timestamp=d.get('ts', ''),
            msg_id=d.get('id', ''),
            signature=d.get('sig', ''),
        )
    
    def sign(self, secret: str):
        """Sign the message with a shared secret."""
        content = f"{self.msg_type.value}:{self.sender_id}:{self.timestamp}:{json.dumps(self.payload)}"
        self.signature = hmac.new(
            secret.encode(), content.encode(), hashlib.sha256
        ).hexdigest()[:16]
    
    def verify(self, secret: str) -> bool:
        """Verify message signature."""
        content = f"{self.msg_type.value}:{self.sender_id}:{self.timestamp}:{json.dumps(self.payload)}"
        expected = hmac.new(
            secret.encode(), content.encode(), hashlib.sha256
        ).hexdigest()[:16]
        return hmac.compare_digest(self.signature, expected)


class DistributedNode:
    """
    A node in the Enigma AI Engine distributed network.
    
    Can act as client, server, or both depending on role and hardware.
    Automatically negotiates capabilities with other nodes.
    """
    
    def __init__(
        self,
        name: str,
        role: NodeRole = NodeRole.AUTO,
        port: int = 5000,
        secret: str = "",
    ):
        self.name = name
        self.node_id = f"{name}_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        self._requested_role = role
        self.role = self._detect_role() if role == NodeRole.AUTO else role
        self.port = port
        self.secret = secret or "forge_default_secret"
        
        # Network state
        self.peers: dict[str, NodeInfo] = {}
        self.address = self._get_local_ip()
        
        # Message handling
        self.incoming = queue.Queue()
        self.outgoing = queue.Queue()
        self._handlers: dict[MessageType, Callable] = {}
        
        # Server state
        self._server_thread: Optional[threading.Thread] = None
        self._running = False
        self._app = None
        
        # Inference engine (lazy loaded)
        self._engine = None
        
        # Capabilities
        self.capabilities = self._detect_capabilities()
        
        # Register default handlers
        self._register_default_handlers()
    
    def _detect_role(self) -> NodeRole:
        """Auto-detect appropriate role based on hardware."""
        try:
            from ..core.device_profiles import DeviceClass, get_device_profiler
            profiler = get_device_profiler()
            device_class = profiler.classify()
            
            role_map = {
                DeviceClass.EMBEDDED: NodeRole.INFERENCE_CLIENT,
                DeviceClass.MOBILE: NodeRole.INFERENCE_CLIENT,
                DeviceClass.LAPTOP_LOW: NodeRole.INFERENCE_CLIENT,
                DeviceClass.LAPTOP_MID: NodeRole.HYBRID,
                DeviceClass.DESKTOP_CPU: NodeRole.HYBRID,
                DeviceClass.DESKTOP_GPU: NodeRole.INFERENCE_SERVER,
                DeviceClass.WORKSTATION: NodeRole.INFERENCE_SERVER,
                DeviceClass.DATACENTER: NodeRole.INFERENCE_SERVER,
            }
            return role_map.get(device_class, NodeRole.INFERENCE_CLIENT)
        except ImportError:
            return NodeRole.INFERENCE_CLIENT
    
    def _detect_capabilities(self) -> list[str]:
        """Detect what this node can do."""
        caps = ["ping", "message"]
        
        if self.role in (NodeRole.INFERENCE_SERVER, NodeRole.HYBRID):
            caps.append("inference")
        
        if self.role in (NodeRole.AVATAR_DISPLAY, NodeRole.HYBRID):
            caps.append("avatar")
        
        if self.role == NodeRole.TRAINING_NODE:
            caps.append("training")
        
        if self.role == NodeRole.RELAY_NODE:
            caps.append("relay")
        
        # Check for specific hardware
        try:
            import torch
            if torch.cuda.is_available():
                caps.append("gpu")
        except ImportError:
            pass  # Intentionally silent
        
        return caps
    
    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception as e:
            logger.debug(f"Could not get local IP: {e}")
            return "127.0.0.1"
    
    @property
    def engine(self):
        """Lazy-load inference engine."""
        if self._engine is None and self.role in (NodeRole.INFERENCE_SERVER, NodeRole.HYBRID):
            try:
                from ..core.inference import EnigmaEngine
                self._engine = EnigmaEngine()
            except Exception as e:
                logger.error(f"Could not load engine: {e}")
        return self._engine
    
    def _register_default_handlers(self):
        """Register default message handlers."""
        self._handlers[MessageType.PING] = self._handle_ping
        self._handlers[MessageType.ANNOUNCE] = self._handle_announce
        self._handlers[MessageType.DISCOVER] = self._handle_discover
        self._handlers[MessageType.GENERATE_REQUEST] = self._handle_generate
    
    def _handle_ping(self, msg: ProtocolMessage) -> Optional[ProtocolMessage]:
        """Handle ping request."""
        return ProtocolMessage(
            msg_type=MessageType.PONG,
            payload={"time": time.time()},
            sender_id=self.node_id,
            target_id=msg.sender_id,
        )
    
    def _handle_announce(self, msg: ProtocolMessage) -> None:
        """Handle node announcement."""
        info = NodeInfo.from_dict(msg.payload)
        info.last_seen = datetime.utcnow().isoformat()
        self.peers[info.node_id] = info
        logger.info(f"Discovered peer: {info.name} ({info.role.name})")
    
    def _handle_discover(self, msg: ProtocolMessage) -> ProtocolMessage:
        """Handle discovery request."""
        my_info = NodeInfo(
            node_id=self.node_id,
            name=self.name,
            role=self.role,
            address=self.address,
            port=self.port,
            capabilities=self.capabilities,
        )
        return ProtocolMessage(
            msg_type=MessageType.ANNOUNCE,
            payload=my_info.to_dict(),
            sender_id=self.node_id,
            target_id=msg.sender_id,
        )
    
    def _handle_generate(self, msg: ProtocolMessage) -> ProtocolMessage:
        """Handle generation request."""
        if "inference" not in self.capabilities:
            return ProtocolMessage(
                msg_type=MessageType.ERROR,
                payload={"error": "This node cannot perform inference"},
                sender_id=self.node_id,
                target_id=msg.sender_id,
            )
        
        prompt = msg.payload.get("prompt", "")
        max_tokens = msg.payload.get("max_tokens", 100)
        temperature = msg.payload.get("temperature", 0.8)
        
        try:
            if self.engine:
                response = self.engine.generate(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
                return ProtocolMessage(
                    msg_type=MessageType.GENERATE_RESPONSE,
                    payload={"response": response, "prompt": prompt},
                    sender_id=self.node_id,
                    target_id=msg.sender_id,
                )
            else:
                return ProtocolMessage(
                    msg_type=MessageType.ERROR,
                    payload={"error": "Engine not available"},
                    sender_id=self.node_id,
                    target_id=msg.sender_id,
                )
        except Exception as e:
            return ProtocolMessage(
                msg_type=MessageType.ERROR,
                payload={"error": str(e)},
                sender_id=self.node_id,
                target_id=msg.sender_id,
            )
    
    def start(self, blocking: bool = False):
        """Start the node's network services."""
        try:
            from flask import Flask, jsonify, request
            from flask_cors import CORS
        except ImportError:
            logger.error("Flask not installed. Install with: pip install flask flask-cors")
            return
        
        self._app = Flask(f"forge_{self.name}")
        CORS(self._app)
        
        @self._app.route("/health")
        def health():
            return jsonify({
                "ok": True,
                "node_id": self.node_id,
                "name": self.name,
                "role": self.role.name,
            })
        
        @self._app.route("/info")
        def info():
            return jsonify(NodeInfo(
                node_id=self.node_id,
                name=self.name,
                role=self.role,
                address=self.address,
                port=self.port,
                capabilities=self.capabilities,
            ).to_dict())
        
        @self._app.route("/peers")
        def peers():
            return jsonify({
                nid: p.to_dict() for nid, p in self.peers.items()
            })
        
        @self._app.route("/message", methods=["POST"])
        def receive_message():
            try:
                msg = ProtocolMessage.from_json(request.data.decode())
                
                # Verify signature if secret is set
                if self.secret and msg.signature:
                    if not msg.verify(self.secret):
                        return jsonify({"error": "Invalid signature"}), 401
                
                # Handle message
                handler = self._handlers.get(msg.msg_type)
                if handler:
                    response = handler(msg)
                    if response:
                        if self.secret:
                            response.sign(self.secret)
                        return response.to_json()
                    return jsonify({"received": True})
                else:
                    return jsonify({"error": f"No handler for {msg.msg_type}"}), 400
                    
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self._app.route("/generate", methods=["POST"])
        def generate():
            """Direct generation endpoint for simple use."""
            data = request.json or {}
            prompt = data.get("prompt", "")
            max_tokens = data.get("max_tokens", 100)
            temperature = data.get("temperature", 0.8)
            
            msg = ProtocolMessage(
                msg_type=MessageType.GENERATE_REQUEST,
                payload={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                sender_id="direct_api",
            )
            
            response = self._handle_generate(msg)
            return jsonify(response.payload)
        
        logger.info(f"Starting DistributedNode '{self.name}' as {self.role.name} on {self.address}:{self.port}")
        
        if blocking:
            self._running = True
            self._app.run(host="0.0.0.0", port=self.port, debug=False)
        else:
            self._server_thread = threading.Thread(
                target=lambda: self._app.run(host="0.0.0.0", port=self.port, debug=False, use_reloader=False),
                daemon=True,
            )
            self._server_thread.start()
            self._running = True
    
    def stop(self):
        """Stop the node."""
        self._running = False
        # Flask doesn't have clean shutdown, daemon thread dies with process
    
    def connect(self, address: str) -> bool:
        """Connect to another node."""
        import urllib.request
        
        if not address.startswith("http"):
            address = f"http://{address}"
        
        try:
            # Get peer info
            req = urllib.request.Request(f"{address}/info")
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                info = NodeInfo.from_dict(data)
            
            # Announce ourselves
            my_info = NodeInfo(
                node_id=self.node_id,
                name=self.name,
                role=self.role,
                address=self.address,
                port=self.port,
                capabilities=self.capabilities,
            )
            
            msg = ProtocolMessage(
                msg_type=MessageType.ANNOUNCE,
                payload=my_info.to_dict(),
                sender_id=self.node_id,
            )
            if self.secret:
                msg.sign(self.secret)
            
            req = urllib.request.Request(
                f"{address}/message",
                data=msg.to_json().encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
            
            # Store peer
            info.last_seen = datetime.utcnow().isoformat()
            self.peers[info.node_id] = info
            
            logger.info(f"Connected to {info.name} ({info.role.name}) at {address}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to {address}: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.8) -> str:
        """
        Generate text, using local engine or remote server.
        
        Automatically routes to the best available inference server.
        """
        # If we can do inference locally, do it
        if "inference" in self.capabilities and self.engine:
            return self.engine.generate(prompt, max_gen=max_tokens, temperature=temperature)
        
        # Find a peer that can do inference
        for peer in self.peers.values():
            if "inference" in peer.capabilities:
                return self._remote_generate(peer, prompt, max_tokens, temperature)
        
        raise RuntimeError("No inference capability available locally or remotely")
    
    def _remote_generate(self, peer: NodeInfo, prompt: str, max_tokens: int, temperature: float) -> str:
        """Send generation request to remote peer."""
        import urllib.request
        
        address = f"http://{peer.address}:{peer.port}"
        
        msg = ProtocolMessage(
            msg_type=MessageType.GENERATE_REQUEST,
            payload={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            sender_id=self.node_id,
            target_id=peer.node_id,
        )
        if self.secret:
            msg.sign(self.secret)
        
        req = urllib.request.Request(
            f"{address}/message",
            data=msg.to_json().encode(),
            headers={"Content-Type": "application/json"},
        )
        
        with urllib.request.urlopen(req, timeout=60) as response:
            result = ProtocolMessage.from_json(response.read().decode())
            
            if result.msg_type == MessageType.ERROR:
                raise RuntimeError(result.payload.get("error", "Remote error"))
            
            return result.payload.get("response", "")
    
    def on_message(self, msg_type: MessageType, handler: Callable):
        """Register a custom message handler."""
        self._handlers[msg_type] = handler


def create_server(name: str = "forge_server", port: int = 5000) -> DistributedNode:
    """Quick helper to create an inference server."""
    node = DistributedNode(name, role=NodeRole.INFERENCE_SERVER, port=port)
    node.start()
    return node


def create_client(name: str = "forge_client", server_address: str = None) -> DistributedNode:
    """Quick helper to create an inference client."""
    node = DistributedNode(name, role=NodeRole.INFERENCE_CLIENT)
    if server_address:
        node.connect(server_address)
    return node


if __name__ == "__main__":
    print("Enigma AI Engine Distributed Protocol")
    print("="*40)
    print("\nServer mode:")
    print("  node = create_server('my_server', port=5000)")
    print("\nClient mode:")
    print("  node = create_client('my_client', 'server_ip:5000')")
    print("  response = node.generate('Hello!')")
