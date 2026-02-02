#!/usr/bin/env python3
"""
ForgeAI API Server & Networking Example
========================================

Complete example showing how to use ForgeAI's networking including:
- REST API server for remote access
- Multi-device networking (distributed inference)
- Client connections
- Security and authentication

ForgeAI can run as a server that other applications can connect to,
or participate in a network of ForgeAI instances for distributed computing.

Dependencies:
    pip install flask flask-cors requests

Run: python examples/networking_example.py
"""

import time
import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
import hashlib
import secrets


# =============================================================================
# API Server Configuration
# =============================================================================

@dataclass
class ServerConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    cors_enabled: bool = True
    require_auth: bool = True
    api_key: Optional[str] = None
    rate_limit: int = 100  # Requests per minute
    max_tokens: int = 2048


# =============================================================================
# Simulated API Server
# =============================================================================

class APIServerSimulator:
    """
    Simulated REST API server for demonstration.
    
    In real usage, use forge_ai.comms.api_server which provides:
    - Chat completions endpoint
    - Model info endpoint
    - Generation endpoints (image, code, etc.)
    - Health checks
    """
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.is_running = False
        self._request_count = 0
        self._start_time = None
        
        # Generate API key if auth required
        if config.require_auth and not config.api_key:
            config.api_key = secrets.token_urlsafe(32)
    
    def _log(self, message: str):
        """Log server message."""
        print(f"[Server] {message}")
    
    def start(self):
        """Start the API server."""
        self.is_running = True
        self._start_time = time.time()
        
        self._log(f"Starting server on {self.config.host}:{self.config.port}")
        
        if self.config.require_auth:
            self._log(f"API Key: {self.config.api_key[:20]}...")
        
        self._log("Available endpoints:")
        self._log("  POST /v1/chat/completions - Chat with AI")
        self._log("  GET  /v1/models           - List models")
        self._log("  GET  /health              - Health check")
        self._log("  POST /v1/images/generate  - Generate images")
    
    def stop(self):
        """Stop the API server."""
        self.is_running = False
        self._log("Server stopped")
    
    def handle_request(self, endpoint: str, method: str, 
                       data: Optional[Dict] = None,
                       headers: Optional[Dict] = None) -> Dict:
        """
        Handle an API request.
        
        Args:
            endpoint: API endpoint path
            method: HTTP method
            data: Request body
            headers: Request headers
            
        Returns:
            Response dict
        """
        self._request_count += 1
        
        # Check authentication
        if self.config.require_auth:
            auth = headers.get("Authorization", "") if headers else ""
            if not auth.startswith("Bearer ") or auth[7:] != self.config.api_key:
                return {"error": "Unauthorized", "status": 401}
        
        # Route request
        if endpoint == "/v1/chat/completions" and method == "POST":
            return self._handle_chat(data)
        elif endpoint == "/v1/models" and method == "GET":
            return self._handle_models()
        elif endpoint == "/health" and method == "GET":
            return self._handle_health()
        elif endpoint == "/v1/images/generate" and method == "POST":
            return self._handle_image_gen(data)
        else:
            return {"error": "Not found", "status": 404}
    
    def _handle_chat(self, data: Dict) -> Dict:
        """Handle chat completion request."""
        messages = data.get("messages", [])
        model = data.get("model", "forge-small")
        
        # Get last user message
        user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break
        
        # Generate response (simulated)
        response_text = f"This is a simulated response to: '{user_msg[:50]}...'"
        
        return {
            "id": f"chatcmpl-{secrets.token_hex(8)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(user_msg.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(user_msg.split()) + len(response_text.split())
            }
        }
    
    def _handle_models(self) -> Dict:
        """Handle models list request."""
        return {
            "object": "list",
            "data": [
                {"id": "forge-nano", "object": "model", "owned_by": "forgeai"},
                {"id": "forge-small", "object": "model", "owned_by": "forgeai"},
                {"id": "forge-medium", "object": "model", "owned_by": "forgeai"},
            ]
        }
    
    def _handle_health(self) -> Dict:
        """Handle health check."""
        uptime = time.time() - self._start_time if self._start_time else 0
        return {
            "status": "healthy",
            "uptime": uptime,
            "requests_served": self._request_count
        }
    
    def _handle_image_gen(self, data: Dict) -> Dict:
        """Handle image generation request."""
        prompt = data.get("prompt", "")
        
        return {
            "created": int(time.time()),
            "data": [{
                "url": f"https://example.com/generated_{secrets.token_hex(4)}.png",
                "prompt": prompt
            }]
        }


# =============================================================================
# API Client
# =============================================================================

class ForgeClient:
    """
    Client for connecting to ForgeAI API server.
    
    Compatible with OpenAI API format for easy integration.
    """
    
    def __init__(self, base_url: str = "http://localhost:5000",
                 api_key: Optional[str] = None):
        """
        Initialize client.
        
        Args:
            base_url: Server URL
            api_key: API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
    
    def _make_request(self, endpoint: str, method: str = "GET",
                      data: Optional[Dict] = None) -> Dict:
        """Make HTTP request to server."""
        try:
            import requests
            
            url = f"{self.base_url}{endpoint}"
            headers = {"Content-Type": "application/json"}
            
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            if method == "GET":
                response = requests.get(url, headers=headers)
            else:
                response = requests.post(url, headers=headers, json=data)
            
            return response.json()
            
        except ImportError:
            print("requests not installed. Install with: pip install requests")
            return {"error": "requests not installed"}
        except Exception as e:
            return {"error": str(e)}
    
    def chat(self, messages: List[Dict], model: str = "forge-small") -> str:
        """
        Send chat messages and get response.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use
            
        Returns:
            AI response text
        """
        response = self._make_request(
            "/v1/chat/completions",
            method="POST",
            data={"messages": messages, "model": model}
        )
        
        if "error" in response:
            return f"Error: {response['error']}"
        
        return response["choices"][0]["message"]["content"]
    
    def list_models(self) -> List[str]:
        """List available models."""
        response = self._make_request("/v1/models")
        
        if "error" in response:
            return []
        
        return [m["id"] for m in response.get("data", [])]
    
    def health_check(self) -> Dict:
        """Check server health."""
        return self._make_request("/health")
    
    def generate_image(self, prompt: str) -> Optional[str]:
        """
        Generate image from prompt.
        
        Args:
            prompt: Image description
            
        Returns:
            URL of generated image
        """
        response = self._make_request(
            "/v1/images/generate",
            method="POST",
            data={"prompt": prompt}
        )
        
        if "error" in response:
            return None
        
        return response["data"][0]["url"]


# =============================================================================
# Multi-Device Networking
# =============================================================================

@dataclass
class NetworkNode:
    """A node in the ForgeAI network."""
    node_id: str
    host: str
    port: int
    capabilities: List[str] = field(default_factory=list)
    is_online: bool = False
    last_seen: float = 0


class ForgeNetwork:
    """
    Multi-device networking for distributed ForgeAI.
    
    Allows multiple ForgeAI instances to:
    - Share model inference load
    - Specialize in different tasks
    - Provide redundancy
    """
    
    def __init__(self, node_id: Optional[str] = None):
        """
        Initialize network node.
        
        Args:
            node_id: Unique identifier for this node
        """
        self.node_id = node_id or secrets.token_hex(8)
        self.nodes: Dict[str, NetworkNode] = {}
        self.is_running = False
        
        # This node's capabilities
        self.capabilities = ["chat", "inference"]
    
    def _log(self, message: str):
        """Log network message."""
        print(f"[Network:{self.node_id[:8]}] {message}")
    
    def start(self, host: str = "0.0.0.0", port: int = 5001):
        """Start network services."""
        self.is_running = True
        self._log(f"Started on {host}:{port}")
        self._log(f"Capabilities: {self.capabilities}")
    
    def stop(self):
        """Stop network services."""
        self.is_running = False
        self._log("Stopped")
    
    def discover_nodes(self, broadcast_addr: str = "255.255.255.255"):
        """
        Discover other ForgeAI nodes on the network.
        
        Uses UDP broadcast to find nodes.
        """
        self._log("Discovering nodes...")
        
        # Simulated discovery
        discovered = [
            NetworkNode(
                node_id="node_001",
                host="192.168.1.100",
                port=5001,
                capabilities=["chat", "inference"],
                is_online=True,
                last_seen=time.time()
            ),
            NetworkNode(
                node_id="node_002",
                host="192.168.1.101",
                port=5001,
                capabilities=["image_gen"],
                is_online=True,
                last_seen=time.time()
            ),
        ]
        
        for node in discovered:
            self.nodes[node.node_id] = node
            self._log(f"  Found: {node.node_id} at {node.host}:{node.port}")
        
        self._log(f"Discovered {len(discovered)} nodes")
    
    def register_node(self, host: str, port: int, 
                      capabilities: Optional[List[str]] = None):
        """
        Manually register a node.
        
        Args:
            host: Node hostname/IP
            port: Node port
            capabilities: What the node can do
        """
        node_id = hashlib.sha256(f"{host}:{port}".encode()).hexdigest()[:16]
        
        self.nodes[node_id] = NetworkNode(
            node_id=node_id,
            host=host,
            port=port,
            capabilities=capabilities or ["chat"],
            is_online=True,
            last_seen=time.time()
        )
        
        self._log(f"Registered node: {node_id} at {host}:{port}")
    
    def find_node_for_task(self, task: str) -> Optional[NetworkNode]:
        """
        Find a node that can handle a specific task.
        
        Args:
            task: Task type (chat, image_gen, etc.)
            
        Returns:
            NetworkNode that can handle the task, or None
        """
        for node in self.nodes.values():
            if node.is_online and task in node.capabilities:
                return node
        return None
    
    def route_request(self, task: str, data: Dict) -> Optional[Dict]:
        """
        Route a request to the appropriate node.
        
        Args:
            task: Task type
            data: Request data
            
        Returns:
            Response from node, or None if no node available
        """
        node = self.find_node_for_task(task)
        
        if not node:
            self._log(f"No node available for task: {task}")
            return None
        
        self._log(f"Routing {task} to {node.node_id}")
        
        # Would make actual network request here
        return {"status": "success", "node": node.node_id}
    
    def get_network_status(self) -> Dict:
        """Get status of all nodes."""
        return {
            "local_node": self.node_id,
            "total_nodes": len(self.nodes) + 1,  # +1 for self
            "nodes": [
                {
                    "id": n.node_id,
                    "host": n.host,
                    "online": n.is_online,
                    "capabilities": n.capabilities
                }
                for n in self.nodes.values()
            ]
        }


# =============================================================================
# Example Usage
# =============================================================================

def example_api_server():
    """Start and use API server."""
    print("\n" + "="*60)
    print("Example 1: API Server")
    print("="*60)
    
    config = ServerConfig(
        host="0.0.0.0",
        port=5000,
        require_auth=True
    )
    
    server = APIServerSimulator(config)
    server.start()
    
    print("\n--- Making requests ---\n")
    
    # Test health endpoint
    response = server.handle_request("/health", "GET")
    print(f"Health check: {response}")
    
    # Test chat endpoint (with auth)
    headers = {"Authorization": f"Bearer {config.api_key}"}
    response = server.handle_request(
        "/v1/chat/completions",
        "POST",
        data={"messages": [{"role": "user", "content": "Hello!"}]},
        headers=headers
    )
    print(f"Chat response: {response['choices'][0]['message']['content']}")
    
    # Test without auth
    response = server.handle_request("/v1/models", "GET")
    print(f"Without auth: {response}")
    
    server.stop()


def example_client():
    """Use client to connect to server."""
    print("\n" + "="*60)
    print("Example 2: API Client")
    print("="*60)
    
    print("Client usage (requires running server):")
    print("""
    from examples.networking_example import ForgeClient
    
    # Connect to server
    client = ForgeClient(
        base_url="http://localhost:5000",
        api_key="your-api-key"
    )
    
    # Check health
    status = client.health_check()
    print(f"Server status: {status}")
    
    # List models
    models = client.list_models()
    print(f"Available models: {models}")
    
    # Chat
    response = client.chat([
        {"role": "user", "content": "What is Python?"}
    ])
    print(f"Response: {response}")
    
    # Generate image
    url = client.generate_image("A sunset over mountains")
    print(f"Image URL: {url}")
    """)


def example_multi_device():
    """Multi-device networking."""
    print("\n" + "="*60)
    print("Example 3: Multi-Device Network")
    print("="*60)
    
    # Create network
    network = ForgeNetwork()
    network.start()
    
    # Discover nodes
    network.discover_nodes()
    
    # Register additional node
    network.register_node("192.168.1.200", 5001, ["video_gen"])
    
    # Get status
    status = network.get_network_status()
    print(f"\nNetwork status:")
    print(f"  Local node: {status['local_node']}")
    print(f"  Total nodes: {status['total_nodes']}")
    
    # Route requests
    print("\n--- Routing requests ---")
    network.route_request("chat", {"messages": []})
    network.route_request("image_gen", {"prompt": "test"})
    network.route_request("video_gen", {"prompt": "test"})
    
    network.stop()


def example_openai_compatible():
    """OpenAI-compatible API usage."""
    print("\n" + "="*60)
    print("Example 4: OpenAI-Compatible API")
    print("="*60)
    
    print("ForgeAI server is OpenAI API compatible!")
    print("""
    # Start ForgeAI server
    python run.py --serve --port 5000
    
    # Use with OpenAI Python library
    import openai
    
    openai.api_base = "http://localhost:5000/v1"
    openai.api_key = "your-forge-api-key"
    
    response = openai.ChatCompletion.create(
        model="forge-small",
        messages=[
            {"role": "user", "content": "Hello!"}
        ]
    )
    
    print(response.choices[0].message.content)
    
    # Also works with LangChain, LlamaIndex, etc.!
    """)


def example_forge_integration():
    """Real ForgeAI integration."""
    print("\n" + "="*60)
    print("Example 5: ForgeAI Integration")
    print("="*60)
    
    print("For actual ForgeAI networking:")
    print("""
    # Start API server
    from forge_ai.comms.api_server import create_api_server
    
    app = create_api_server(
        model_path="models/forge-small",
        api_key="your-secret-key"
    )
    app.run(host="0.0.0.0", port=5000)
    
    # Or use command line
    python run.py --serve --port 5000
    
    # Multi-device networking
    from forge_ai.comms.network import ForgeNode
    
    node = ForgeNode(capabilities=["chat", "image_gen"])
    node.start(port=5001)
    node.discover_peers()
    
    # The network automatically:
    # - Discovers other ForgeAI instances
    # - Routes requests to appropriate nodes
    # - Balances load across nodes
    # - Handles node failures gracefully
    """)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("ForgeAI API Server & Networking Examples")
    print("="*60)
    
    example_api_server()
    example_client()
    example_multi_device()
    example_openai_compatible()
    example_forge_integration()
    
    print("\n" + "="*60)
    print("Networking Summary:")
    print("="*60)
    print("""
API Server:

1. Endpoints:
   - POST /v1/chat/completions - OpenAI-compatible chat
   - GET  /v1/models           - List available models
   - GET  /health              - Health check
   - POST /v1/images/generate  - Image generation

2. Authentication:
   - Bearer token authentication
   - Set FORGE_API_KEY environment variable
   - Pass in Authorization header

3. Starting Server:
   python run.py --serve --port 5000

Multi-Device Network:

1. Discovery:
   - Automatic UDP broadcast discovery
   - Manual node registration

2. Load Balancing:
   - Routes requests to capable nodes
   - Specialization (image node, chat node)
   - Redundancy for high availability

3. Capabilities:
   - chat: Text generation
   - inference: Model inference
   - image_gen: Image generation
   - video_gen: Video generation

OpenAI Compatibility:
   - Works with openai Python package
   - Works with LangChain
   - Works with any OpenAI-compatible client

Command Line:
   python run.py --serve                    # Start server
   python run.py --serve --port 8080        # Custom port
   python run.py --serve --api-key abc123   # Custom API key
""")
