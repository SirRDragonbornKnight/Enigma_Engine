"""
================================================================================
🌐 MULTI-DEVICE NETWORK - CONNECT YOUR AI ARMY
================================================================================

Connect multiple computers running Enigma AI Engine! Share models, distribute work,
and let AIs have conversations with each other.

📍 FILE: enigma_engine/comms/network.py
🏷️ TYPE: Multi-Device Networking
🎯 MAIN CLASSES: ForgeNode, Message, ModelExporter

┌─────────────────────────────────────────────────────────────────────────────┐
│  NETWORK ARCHITECTURE:                                                      │
│                                                                             │
│  Device A (PC with GPU)              Device B (Raspberry Pi)               │
│  ┌─────────────────────┐            ┌─────────────────────┐               │
│  │  Forge Instance    │◄──────────►│  Forge Instance    │               │
│  │  - Large Model      │   Network  │  - Small Model      │               │
│  │  - Training         │            │  - Inference        │               │
│  │  - API Server       │            │  - Remote Client    │               │
│  └─────────────────────┘            └─────────────────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘

🎮 MODES:
    1. NETWORKED MODE: Multiple instances talking to each other
       - Pi talks to PC, PC responds
       - AIs can have conversations with each other
       - Share context and memory across devices
    
    2. DISCONNECTED MODE: Run same AI on different devices
       - Export model as portable package
       - Import on another device
       - Sync memories when reconnected

🔗 CONNECTED FILES:
    → USES:      enigma_engine/core/inference.py (EnigmaEngine for responses)
    → USES:      enigma_engine/memory/manager.py (sync memories)
    ← USED BY:   enigma_engine/gui/tabs/network_tab.py (GUI controls)
    📄 RELATED:  enigma_engine/comms/discovery.py (auto-find nodes)

📖 USAGE:
    # On PC (server mode)
    from enigma_engine.comms.network import ForgeNode
    node = ForgeNode(name="pc_node", port=5000)
    node.start_server()
    
    # On Pi (client mode)
    node = ForgeNode(name="pi_node")
    node.connect_to("192.168.1.100:5000")
    response = node.ask_peer("pc_node", "What is the meaning of life?")
    
    # AI-to-AI conversation
    node.start_ai_conversation("pc_node", num_turns=10)

📖 SEE ALSO:
    • enigma_engine/comms/discovery.py    - Auto-discover other nodes
    • enigma_engine/comms/memory_sync.py  - Sync memories across devices
    • docs/multi_device_guide.md     - Multi-device setup guide
"""

import json
import logging
import queue
import socket
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from ..config import CONFIG

logger = logging.getLogger(__name__)


class Message:
    """A message between Forge nodes."""
    
    def __init__(
        self,
        msg_type: str,
        content: Any,
        sender: str = "unknown",
        recipient: str = "all",
        conversation_id: str = None
    ):
        self.msg_type = msg_type  # "chat", "query", "response", "sync", "heartbeat"
        self.content = content
        self.sender = sender
        self.recipient = recipient
        self.conversation_id = conversation_id or f"conv_{int(time.time())}"
        self.timestamp = datetime.now().isoformat()
        self.id = f"{sender}_{int(time.time()*1000)}"
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.msg_type,
            "content": self.content,
            "sender": self.sender,
            "recipient": self.recipient,
            "conversation_id": self.conversation_id,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        msg = cls(
            msg_type=data.get("type", "chat"),
            content=data.get("content", ""),
            sender=data.get("sender", "unknown"),
            recipient=data.get("recipient", "all"),
            conversation_id=data.get("conversation_id"),
        )
        msg.timestamp = data.get("timestamp", msg.timestamp)
        msg.id = data.get("id", msg.id)
        return msg


class ForgeNode:
    """
    A networked Forge instance that can communicate with other nodes.
    
    Can act as:
      - Server: Accept connections from other nodes
      - Client: Connect to other nodes
      - Both: Full peer-to-peer networking
    """
    
    def __init__(
        self,
        name: str = "enigma_engine_node",
        port: int = 5000,
        model_name: str = None,
    ):
        """
        Args:
            name: Unique name for this node
            port: Port to listen on when acting as server
            model_name: Name of the model to use (from registry)
        """
        self.name = name
        self.port = port
        self.model_name = model_name
        
        # Connected peers: {name: {"url": "...", "last_seen": ...}}
        self.peers: dict[str, dict] = {}
        
        # Message queues
        self.incoming_queue = queue.Queue()
        self.outgoing_queue = queue.Queue()
        
        # Conversation history (limited to prevent unbounded growth)
        self.conversations: dict[str, list[Message]] = {}
        self._max_conversation_messages = 100  # Max messages per conversation
        
        # Server thread
        self._server_thread = None
        self._running = False
        
        # Engine (lazy loaded)
        self._engine = None
    
    @property
    def engine(self):
        """Lazy-load the inference engine with device-aware configuration."""
        if self._engine is None:
            # Use device profiles for optimal device selection
            device = None
            try:
                from ..core.device_profiles import get_device_profiler
                profiler = get_device_profiler()
                device = profiler.get_torch_device()
            except ImportError:
                pass
            
            if device is None:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if self.model_name:
                from ..core.inference import EnigmaEngine
                from ..core.model_registry import ModelRegistry
                
                registry = ModelRegistry()
                model, config = registry.load_model(self.model_name)
                
                # Create engine with loaded model
                self._engine = EnigmaEngine.__new__(EnigmaEngine)
                self._engine.device = device
                self._engine.model = model
                self._engine.model.to(self._engine.device)
                self._engine.model.eval()
                from ..core.tokenizer import load_tokenizer
                self._engine.tokenizer = load_tokenizer()
            else:
                from ..core.inference import EnigmaEngine
                self._engine = EnigmaEngine(device=str(device))
        return self._engine
    
    # === Server Mode ===
    
    def start_server(self, host: str = "0.0.0.0", blocking: bool = False):
        """
        Start the API server to accept connections.
        
        Args:
            host: Interface to bind to (0.0.0.0 = all interfaces)
            blocking: If True, block until server stops
        """
        from flask import Flask, jsonify, request
        from flask_cors import CORS
        
        app = Flask(f"forge_{self.name}")
        CORS(app)
        
        @app.route("/health")
        def health():
            return jsonify({"ok": True, "node": self.name})
        
        @app.route("/info")
        def info():
            # Get device info if available
            device_info = {"device_class": "unknown", "device": "unknown"}
            try:
                from ..core.device_profiles import get_device_profiler
                profiler = get_device_profiler()
                caps = profiler.detect()
                device_info = {
                    "device_class": profiler.classify().name,
                    "device": profiler.get_torch_device(),
                    "cpu_cores": caps.cpu_cores,
                    "ram_mb": caps.ram_total_mb,
                    "has_gpu": caps.has_gpu,
                    "gpu_name": caps.gpu_name,
                    "vram_mb": caps.vram_total_mb,
                }
            except ImportError:
                pass
            
            return jsonify({
                "name": self.name,
                "model": self.model_name,
                "peers": list(self.peers.keys()),
                **device_info,
            })
        
        @app.route("/generate", methods=["POST"])
        def generate():
            data = request.json or {}
            prompt = data.get("prompt", "")
            max_gen = int(data.get("max_gen", 50))
            temp = float(data.get("temperature", 1.0))
            text = self.engine.generate(prompt, max_gen=max_gen, temperature=temp)
            return jsonify({"text": text, "from": self.name})
        
        @app.route("/message", methods=["POST"])
        def receive_message():
            data = request.json or {}
            msg = Message.from_dict(data)
            self.incoming_queue.put(msg)
            
            # Store in conversation history
            if msg.conversation_id not in self.conversations:
                self.conversations[msg.conversation_id] = []
            self.conversations[msg.conversation_id].append(msg)
            # Trim to prevent unbounded growth
            if len(self.conversations[msg.conversation_id]) > self._max_conversation_messages:
                self.conversations[msg.conversation_id] = self.conversations[msg.conversation_id][-self._max_conversation_messages:]
            
            # If it's a query, generate response
            if msg.msg_type == "query":
                response_text = self.engine.generate(msg.content, max_gen=50)
                response = Message(
                    msg_type="response",
                    content=response_text,
                    sender=self.name,
                    recipient=msg.sender,
                    conversation_id=msg.conversation_id
                )
                return jsonify(response.to_dict())
            
            return jsonify({"received": True})
        
        @app.route("/connect", methods=["POST"])
        def peer_connect():
            data = request.json or {}
            peer_name = data.get("name")
            peer_url = data.get("url")
            if peer_name and peer_url:
                self.peers[peer_name] = {
                    "url": peer_url,
                    "last_seen": datetime.now().isoformat()
                }
            return jsonify({"connected": True, "peers": list(self.peers.keys())})
        
        @app.route("/conversation/<conv_id>")
        def get_conversation(conv_id):
            msgs = self.conversations.get(conv_id, [])
            return jsonify({
                "conversation_id": conv_id,
                "messages": [m.to_dict() for m in msgs]
            })
        
        logger.info(f"Starting Forge node '{self.name}' on {host}:{self.port}")
        
        if blocking:
            app.run(host=host, port=self.port, debug=False)
        else:
            self._server_thread = threading.Thread(
                target=lambda: app.run(host=host, port=self.port, debug=False, use_reloader=False),
                daemon=True
            )
            self._server_thread.start()
            self._running = True
            logger.info(f"Server started in background on port {self.port}")
    
    def stop_server(self):
        """Stop the server (if running in background)."""
        self._running = False
        # Note: Flask doesn't have a clean shutdown, but daemon thread will die with main process
    
    # === Client Mode ===
    
    def connect_to(self, url: str, name: str = None) -> bool:
        """
        Connect to another Forge node.
        
        Args:
            url: URL of the peer (e.g., "192.168.1.100:5000")
            name: Optional name for the peer
            
        Returns:
            True if connection successful
        """
        if not url.startswith("http"):
            url = f"http://{url}"
        
        try:
            # Get peer info
            req = urllib.request.Request(f"{url}/info")
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                peer_name = name or data.get("name", "unknown")
            
            # Register ourselves with the peer
            register_data = json.dumps({
                "name": self.name,
                "url": f"http://{self._get_local_ip()}:{self.port}"
            }).encode()
            
            req = urllib.request.Request(
                f"{url}/connect",
                data=register_data,
                headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=5)
            
            # Store peer
            self.peers[peer_name] = {
                "url": url,
                "last_seen": datetime.now().isoformat()
            }
            
            logger.info(f"Connected to peer '{peer_name}' at {url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to {url}: {e}")
            return False
    
    def _get_local_ip(self) -> str:
        """Get this machine's local IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except OSError:
            return "127.0.0.1"
    
    # === Communication ===
    
    def ask_peer(self, peer_name: str, prompt: str, max_gen: int = 50) -> str:
        """
        Ask another Forge node a question.
        
        Args:
            peer_name: Name of the peer to ask
            prompt: The question/prompt
            max_gen: Maximum tokens to generate
            
        Returns:
            The peer's response
        """
        if peer_name not in self.peers:
            raise ValueError(f"Unknown peer: {peer_name}. Connected peers: {list(self.peers.keys())}")
        
        peer_url = self.peers[peer_name]["url"]
        
        data = json.dumps({
            "prompt": prompt,
            "max_gen": max_gen,
        }).encode()
        
        req = urllib.request.Request(
            f"{peer_url}/generate",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())
            return result.get("text", "")
    
    def send_message(self, peer_name: str, content: str, msg_type: str = "chat") -> Optional[Message]:
        """
        Send a message to another node.
        
        Args:
            peer_name: Peer to send to
            content: Message content
            msg_type: Type of message ("chat", "query", etc.)
            
        Returns:
            Response message if query, else None
        """
        if peer_name not in self.peers:
            raise ValueError(f"Unknown peer: {peer_name}")
        
        msg = Message(
            msg_type=msg_type,
            content=content,
            sender=self.name,
            recipient=peer_name,
        )
        
        peer_url = self.peers[peer_name]["url"]
        data = json.dumps(msg.to_dict()).encode()
        
        req = urllib.request.Request(
            f"{peer_url}/message",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())
            if msg_type == "query" and "content" in result:
                return Message.from_dict(result)
            return None
    
    # === AI-to-AI Conversation ===
    
    def start_ai_conversation(
        self,
        peer_name: str,
        initial_prompt: str = "Hello, let's have a conversation.",
        num_turns: int = 5,
        callback: Callable[[str, str], None] = None
    ) -> list[dict]:
        """
        Start an AI-to-AI conversation with another node.
        
        The AIs will take turns responding to each other.
        
        Args:
            peer_name: Peer to converse with
            initial_prompt: Starting message
            num_turns: Number of back-and-forth exchanges
            callback: Optional function called with (speaker, text) for each turn
            
        Returns:
            List of conversation turns
        """
        conversation = []
        current_prompt = initial_prompt
        
        print(f"\n{'='*60}")
        print(f"AI-to-AI Conversation: {self.name} <-> {peer_name}")
        print(f"{'='*60}\n")
        
        for turn in range(num_turns * 2):
            if turn % 2 == 0:
                # Our turn to speak
                if turn == 0:
                    response = current_prompt
                else:
                    response = self.engine.generate(current_prompt, max_gen=50)
                
                speaker = self.name
                print(f"{speaker}: {response}\n")
                
                if callback:
                    callback(speaker, response)
                
                conversation.append({"speaker": speaker, "text": response})
                current_prompt = response
                
            else:
                # Peer's turn
                response = self.ask_peer(peer_name, current_prompt)
                speaker = peer_name
                print(f"{speaker}: {response}\n")
                
                if callback:
                    callback(speaker, response)
                
                conversation.append({"speaker": speaker, "text": response})
                current_prompt = response
        
        print(f"{'='*60}")
        print("Conversation ended")
        print(f"{'='*60}\n")
        
        return conversation
    
    # === Local Generation ===
    
    def generate(self, prompt: str, max_gen: int = 50) -> str:
        """Generate a response locally."""
        return self.engine.generate(prompt, max_gen=max_gen)


class ModelExporter:
    """
    Export models as portable packages for disconnected devices.
    
    Creates a self-contained package that can be copied to another
    device and run without network access.
    """
    
    @staticmethod
    def export_model(
        model_name: str,
        output_path: str,
        include_tokenizer: bool = True,
        include_config: bool = True,
    ) -> str:
        """
        Export a model as a portable package.
        
        Args:
            model_name: Name of the model in registry
            output_path: Where to save the package
            include_tokenizer: Include tokenizer files
            include_config: Include configuration
            
        Returns:
            Path to the created package
        """
        import shutil
        import zipfile

        from ..core.model_registry import ModelRegistry
        
        registry = ModelRegistry()
        
        if model_name not in registry.registry.get("models", {}):
            raise ValueError(f"Model '{model_name}' not found")
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create package directory
        package_dir = output_path / f"{model_name}_package"
        package_dir.mkdir(exist_ok=True)
        
        # Copy model files
        model_dir = Path(registry.models_dir) / model_name
        shutil.copytree(model_dir, package_dir / "model", dirs_exist_ok=True)
        
        # Create manifest
        manifest = {
            "name": model_name,
            "exported": datetime.now().isoformat(),
            "includes": {
                "model": True,
                "tokenizer": include_tokenizer,
                "config": include_config,
            }
        }
        
        with open(package_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        
        # Create zip archive
        zip_path = output_path / f"{model_name}_package.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in package_dir.rglob("*"):
                if file.is_file():
                    arcname = file.relative_to(package_dir)
                    zipf.write(file, arcname)
        
        # Clean up directory
        shutil.rmtree(package_dir)
        
        logger.info(f"Exported model '{model_name}' to {zip_path}")
        return str(zip_path)
    
    @staticmethod
    def import_model(
        package_path: str,
        new_name: str = None,
    ) -> str:
        """
        Import a model from a portable package.
        
        Args:
            package_path: Path to the .zip package
            new_name: Optional new name for the model
            
        Returns:
            Name of the imported model
        """
        import shutil
        import zipfile

        from ..core.model_registry import ModelRegistry
        
        package_path = Path(package_path)
        
        if not package_path.exists():
            raise ValueError(f"Package not found: {package_path}")
        
        registry = ModelRegistry()
        
        # Extract package
        temp_dir = Path(CONFIG["models_dir"]) / "_temp_import"
        temp_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(package_path, 'r') as zipf:
            zipf.extractall(temp_dir)
        
        # Read manifest
        with open(temp_dir / "manifest.json") as f:
            manifest = json.load(f)
        
        original_name = manifest["name"]
        model_name = new_name or original_name
        
        # Check if name exists
        if model_name in registry.registry.get("models", {}):
            # Add suffix
            model_name = f"{model_name}_imported_{int(time.time())}"
        
        # Move model files
        model_dir = Path(registry.models_dir) / model_name
        shutil.move(temp_dir / "model", model_dir)
        
        # Update registry
        with open(model_dir / "config.json") as f:
            config = json.load(f)
        
        registry.registry["models"][model_name] = {
            "path": str(model_dir),
            "size": config.get("size", "unknown"),
            "created": manifest.get("exported"),
            "has_weights": (model_dir / "weights.pth").exists(),
            "imported_from": str(package_path),
        }
        registry._save_registry()
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        logger.info(f"Imported model as '{model_name}'")
        return model_name


# Convenience functions

def create_server_node(name: str, port: int = 5000, model_name: str = None) -> ForgeNode:
    """Create and start a server node."""
    node = ForgeNode(name=name, port=port, model_name=model_name)
    node.start_server()
    return node


def create_client_node(name: str, server_url: str) -> ForgeNode:
    """Create a client node and connect to a server."""
    node = ForgeNode(name=name)
    node.connect_to(server_url)
    return node


if __name__ == "__main__":
    # Example usage
    print("Forge Network System")
    print("="*40)
    print("\nServer mode:")
    print("  node = ForgeNode('my_server', port=5000)")
    print("  node.start_server()")
    print("\nClient mode:")
    print("  node = ForgeNode('my_client')")
    print("  node.connect_to('192.168.1.100:5000')")
    print("  response = node.ask_peer('my_server', 'Hello!')")
