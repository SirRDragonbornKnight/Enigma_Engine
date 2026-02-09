"""
Model Synchronization System

Automatically sync model weights across multiple devices.
When models are trained on one device, changes can be pushed to 
or pulled from other devices in the network.

Use cases:
- Train on PC, use on mobile/Raspberry Pi
- Keep a backup copy of your model on another machine
- Collaborative training - merge improvements from multiple devices

Usage:
    from enigma_engine.comms.model_sync import ModelSyncServer, ModelSyncClient
    
    # On the primary device (has the model)
    server = ModelSyncServer(port=5060)
    server.register_model("my_model", "models/my_model")
    server.start()
    
    # On secondary device (wants the model)
    client = ModelSyncClient()
    client.connect("192.168.1.100:5060")
    
    # Pull latest model version
    client.sync_model("my_model", "models/synced_model")
    
    # Or push local changes back
    client.push_model("my_model", "models/local_model")
"""

import hashlib
import json
import logging
import os
import shutil
import socket
import struct
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    from PyQt5.QtCore import QObject, pyqtSignal
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    QObject = object
    pyqtSignal = lambda *args: None

from ..config import CONFIG

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Represents a version of a model."""
    model_name: str
    version: int
    checksum: str  # MD5 of model file
    timestamp: str
    file_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "checksum": self.checksum,
            "timestamp": self.timestamp,
            "file_size": self.file_size,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ModelVersion":
        return cls(**data)


@dataclass
class SyncStatus:
    """Status of a sync operation."""
    model_name: str
    status: str  # syncing, completed, failed, up_to_date
    progress: float = 0.0
    local_version: int = 0
    remote_version: int = 0
    error: str = ""


class ModelSyncServer:
    """
    Server that hosts models for syncing.
    
    Register models to make them available for sync.
    Other devices can pull the latest version or push updates.
    """
    
    def __init__(self, port: int = 5060, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        
        # Registered models: name -> path
        self._models: Dict[str, str] = {}
        # Version tracking: name -> ModelVersion
        self._versions: Dict[str, ModelVersion] = {}
        
        self._running = False
        self._server_socket: Optional[socket.socket] = None
        
        # Config storage
        self._sync_dir = Path(CONFIG.get("data_dir", "data")) / "model_sync"
        self._sync_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing registrations
        self._load_registry()
    
    def register_model(self, name: str, path: str, metadata: Dict[str, Any] = None):
        """
        Register a model for syncing.
        
        Args:
            name: Model name/identifier
            path: Path to model file or directory
            metadata: Optional metadata about the model
        """
        model_path = Path(path)
        if not model_path.exists():
            logger.warning(f"Model path does not exist: {path}")
            return
        
        self._models[name] = str(model_path)
        
        # Calculate version info
        checksum = self._calculate_checksum(model_path)
        file_size = self._get_size(model_path)
        
        # Get next version number
        existing = self._versions.get(name)
        version = (existing.version + 1) if existing else 1
        
        # Only increment version if checksum changed
        if existing and existing.checksum == checksum:
            version = existing.version
        
        self._versions[name] = ModelVersion(
            model_name=name,
            version=version,
            checksum=checksum,
            timestamp=datetime.now().isoformat(),
            file_size=file_size,
            metadata=metadata or {}
        )
        
        self._save_registry()
        logger.info(f"Registered model '{name}' v{version} ({file_size} bytes)")
    
    def start(self):
        """Start the sync server."""
        if self._running:
            return
        
        self._running = True
        
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(5)
        self._server_socket.settimeout(1.0)
        
        threading.Thread(target=self._accept_loop, daemon=True).start()
        
        logger.info(f"Model sync server started on {self.host}:{self.port}")
    
    def stop(self):
        """Stop the sync server."""
        self._running = False
        if self._server_socket:
            self._server_socket.close()
    
    def _accept_loop(self):
        """Accept incoming connections."""
        while self._running:
            try:
                client_socket, address = self._server_socket.accept()
                logger.debug(f"Sync connection from {address}")
                threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address),
                    daemon=True
                ).start()
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"Accept error: {e}")
    
    def _handle_client(self, client_socket: socket.socket, address):
        """Handle a client connection."""
        try:
            # Set timeout for client socket to prevent indefinite blocking
            client_socket.settimeout(30.0)
            
            # Receive message
            length_data = client_socket.recv(4)
            if not length_data:
                return
            msg_length = struct.unpack(">I", length_data)[0]
            
            data = b""
            while len(data) < msg_length:
                chunk = client_socket.recv(min(4096, msg_length - len(data)))
                if not chunk:
                    break
                data += chunk
            
            message = json.loads(data.decode())
            cmd = message.get("command")
            
            response = {"success": False, "error": "Unknown command"}
            
            if cmd == "list_models":
                response = self._handle_list_models()
            elif cmd == "get_version":
                response = self._handle_get_version(message)
            elif cmd == "pull_model":
                response = self._handle_pull_model(message, client_socket)
            elif cmd == "push_model":
                response = self._handle_push_model(message, client_socket)
            elif cmd == "check_version":
                response = self._handle_check_version(message)
            
            if response:
                self._send_message(client_socket, response)
                
        except Exception as e:
            logger.error(f"Client handler error: {e}")
        finally:
            client_socket.close()
    
    def _send_message(self, sock: socket.socket, data: dict):
        """Send a message with length prefix."""
        msg = json.dumps(data).encode()
        sock.sendall(struct.pack(">I", len(msg)) + msg)
    
    def _handle_list_models(self) -> dict:
        """List available models."""
        return {
            "success": True,
            "models": [v.to_dict() for v in self._versions.values()]
        }
    
    def _handle_get_version(self, message: dict) -> dict:
        """Get version info for a model."""
        name = message.get("model_name")
        if name not in self._versions:
            return {"success": False, "error": "Model not found"}
        
        return {
            "success": True,
            "version": self._versions[name].to_dict()
        }
    
    def _handle_check_version(self, message: dict) -> dict:
        """Check if client's version is up to date."""
        name = message.get("model_name")
        client_checksum = message.get("checksum", "")
        
        if name not in self._versions:
            return {"success": False, "error": "Model not found"}
        
        server_version = self._versions[name]
        up_to_date = (client_checksum == server_version.checksum)
        
        return {
            "success": True,
            "up_to_date": up_to_date,
            "server_version": server_version.to_dict()
        }
    
    def _handle_pull_model(self, message: dict, client_socket: socket.socket) -> Optional[dict]:
        """Handle model pull request."""
        name = message.get("model_name")
        
        if name not in self._models:
            return {"success": False, "error": "Model not found"}
        
        model_path = Path(self._models[name])
        if not model_path.exists():
            return {"success": False, "error": "Model file not found"}
        
        version = self._versions[name]
        
        # Send version info first
        self._send_message(client_socket, {
            "success": True,
            "version": version.to_dict(),
            "file_size": version.file_size
        })
        
        # Send file data
        if model_path.is_dir():
            # Pack directory as tar
            import tarfile
            import io
            
            tar_buffer = io.BytesIO()
            with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
                tar.add(model_path, arcname=name)
            
            data = tar_buffer.getvalue()
            client_socket.sendall(data)
        else:
            # Send file directly
            with open(model_path, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    client_socket.sendall(chunk)
        
        logger.info(f"Sent model '{name}' v{version.version} to {client_socket.getpeername()}")
        return None
    
    def _handle_push_model(self, message: dict, client_socket: socket.socket) -> dict:
        """Handle model push request."""
        name = message.get("model_name")
        file_size = message.get("file_size", 0)
        is_directory = message.get("is_directory", False)
        
        if name not in self._models:
            return {"success": False, "error": "Model not registered"}
        
        # Acknowledge and receive file
        self._send_message(client_socket, {"success": True, "ready": True})
        
        # Receive data
        data = b""
        while len(data) < file_size:
            chunk = client_socket.recv(min(65536, file_size - len(data)))
            if not chunk:
                break
            data += chunk
        
        # Save to model path
        model_path = Path(self._models[name])
        
        if is_directory:
            # Unpack tar
            import tarfile
            import io
            
            tar_buffer = io.BytesIO(data)
            with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
                # Extract to parent directory
                tar.extractall(model_path.parent)
        else:
            # Save file
            with open(model_path, "wb") as f:
                f.write(data)
        
        # Update version
        checksum = self._calculate_checksum(model_path)
        version = self._versions[name].version + 1
        
        self._versions[name] = ModelVersion(
            model_name=name,
            version=version,
            checksum=checksum,
            timestamp=datetime.now().isoformat(),
            file_size=self._get_size(model_path),
            metadata=self._versions[name].metadata
        )
        
        self._save_registry()
        
        logger.info(f"Received model '{name}' - now v{version}")
        
        return {
            "success": True,
            "new_version": version
        }
    
    def _calculate_checksum(self, path: Path) -> str:
        """Calculate MD5 checksum of file or directory."""
        md5 = hashlib.md5()
        
        if path.is_dir():
            for file_path in sorted(path.rglob("*")):
                if file_path.is_file():
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(65536), b""):
                            md5.update(chunk)
        else:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    md5.update(chunk)
        
        return md5.hexdigest()
    
    def _get_size(self, path: Path) -> int:
        """Get total size of file or directory."""
        if path.is_dir():
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return path.stat().st_size
    
    def _save_registry(self):
        """Save model registry to disk."""
        registry = {
            "models": self._models,
            "versions": {k: v.to_dict() for k, v in self._versions.items()}
        }
        
        registry_file = self._sync_dir / "registry.json"
        with open(registry_file, "w") as f:
            json.dump(registry, f, indent=2)
    
    def _load_registry(self):
        """Load model registry from disk."""
        registry_file = self._sync_dir / "registry.json"
        if not registry_file.exists():
            return
        
        try:
            with open(registry_file) as f:
                registry = json.load(f)
            
            self._models = registry.get("models", {})
            self._versions = {
                k: ModelVersion.from_dict(v) 
                for k, v in registry.get("versions", {}).items()
            }
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")


class ModelSyncClient(QObject if HAS_PYQT else object):
    """
    Client for syncing models with a remote server.
    
    Can pull models from or push models to a ModelSyncServer.
    """
    
    # Signals
    if HAS_PYQT:
        sync_started = pyqtSignal(str)  # model_name
        sync_progress = pyqtSignal(str, float)  # model_name, progress
        sync_completed = pyqtSignal(str)  # model_name
        sync_failed = pyqtSignal(str, str)  # model_name, error
    
    def __init__(self):
        if HAS_PYQT:
            super().__init__()
        
        self._server_host: str = ""
        self._server_port: int = 5060
        self._connected = False
        
        # Local version cache
        self._local_versions: Dict[str, ModelVersion] = {}
    
    def connect(self, address: str) -> bool:
        """
        Connect to a sync server.
        
        Args:
            address: Server address (host:port or just host)
            
        Returns:
            True if connection successful
        """
        if ":" in address:
            host, port_str = address.rsplit(":", 1)
            port = int(port_str)
        else:
            host = address
            port = 5060
        
        self._server_host = host
        self._server_port = port
        
        # Test connection
        try:
            response = self._send_command({"command": "list_models"})
            self._connected = response.get("success", False)
            return self._connected
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def list_remote_models(self) -> List[ModelVersion]:
        """List models available on the server."""
        try:
            response = self._send_command({"command": "list_models"})
            if response.get("success"):
                return [
                    ModelVersion.from_dict(m) 
                    for m in response.get("models", [])
                ]
        except Exception as e:
            logger.error(f"List models error: {e}")
        return []
    
    def check_needs_sync(self, model_name: str, local_path: str) -> Tuple[bool, int, int]:
        """
        Check if a model needs syncing.
        
        Args:
            model_name: Name of the model
            local_path: Path to local model
            
        Returns:
            Tuple of (needs_sync, local_version, remote_version)
        """
        local_checksum = self._calculate_checksum(Path(local_path)) if Path(local_path).exists() else ""
        
        try:
            response = self._send_command({
                "command": "check_version",
                "model_name": model_name,
                "checksum": local_checksum
            })
            
            if response.get("success"):
                up_to_date = response.get("up_to_date", False)
                server_version = response.get("server_version", {})
                local_version = self._local_versions.get(model_name)
                
                return (
                    not up_to_date,
                    local_version.version if local_version else 0,
                    server_version.get("version", 0)
                )
        except Exception as e:
            logger.error(f"Check version error: {e}")
        
        return (True, 0, 0)  # Assume needs sync on error
    
    def sync_model(self, model_name: str, local_path: str) -> bool:
        """
        Pull the latest version of a model.
        
        Args:
            model_name: Name of the model to sync
            local_path: Where to save the model locally
            
        Returns:
            True if sync successful
        """
        if HAS_PYQT:
            self.sync_started.emit(model_name)
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self._server_host, self._server_port))
            
            # Request model
            self._send_message(sock, {
                "command": "pull_model",
                "model_name": model_name
            })
            
            # Get response with version info
            response = self._receive_message(sock)
            if not response.get("success"):
                if HAS_PYQT:
                    self.sync_failed.emit(model_name, response.get("error", "Unknown error"))
                sock.close()
                return False
            
            version = ModelVersion.from_dict(response.get("version", {}))
            file_size = response.get("file_size", 0)
            
            # Receive file data with progress
            output_path = Path(local_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            received = 0
            temp_file = output_path.with_suffix(".tmp")
            
            with open(temp_file, "wb") as f:
                while received < file_size:
                    chunk = sock.recv(min(65536, file_size - received))
                    if not chunk:
                        break
                    f.write(chunk)
                    received += len(chunk)
                    
                    progress = (received / file_size) * 100
                    if HAS_PYQT:
                        self.sync_progress.emit(model_name, progress)
            
            sock.close()
            
            # Check if it's a tar (directory)
            if temp_file.stat().st_size > 2 and open(temp_file, "rb").read(3)[:2] == b"\x1f\x8b":
                # It's gzipped tar - extract it
                import tarfile
                
                if output_path.exists():
                    shutil.rmtree(output_path)
                
                output_path.mkdir(parents=True, exist_ok=True)
                
                with tarfile.open(temp_file, "r:gz") as tar:
                    tar.extractall(output_path.parent)
                
                temp_file.unlink()
            else:
                # Regular file - move into place
                if output_path.exists():
                    output_path.unlink()
                temp_file.rename(output_path)
            
            # Update local version cache
            self._local_versions[model_name] = version
            
            if HAS_PYQT:
                self.sync_completed.emit(model_name)
            
            logger.info(f"Synced model '{model_name}' v{version.version} to {local_path}")
            return True
            
        except Exception as e:
            if HAS_PYQT:
                self.sync_failed.emit(model_name, str(e))
            logger.error(f"Sync error: {e}")
            return False
    
    def push_model(self, model_name: str, local_path: str) -> bool:
        """
        Push a local model to the server.
        
        Args:
            model_name: Name of the model
            local_path: Path to local model
            
        Returns:
            True if push successful
        """
        model_path = Path(local_path)
        if not model_path.exists():
            logger.error(f"Model not found: {local_path}")
            return False
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self._server_host, self._server_port))
            
            # Prepare data
            is_directory = model_path.is_dir()
            
            if is_directory:
                import tarfile
                import io
                
                tar_buffer = io.BytesIO()
                with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
                    tar.add(model_path, arcname=model_name)
                
                data = tar_buffer.getvalue()
            else:
                data = model_path.read_bytes()
            
            # Send push request
            self._send_message(sock, {
                "command": "push_model",
                "model_name": model_name,
                "file_size": len(data),
                "is_directory": is_directory
            })
            
            # Wait for acknowledgment
            response = self._receive_message(sock)
            if not response.get("ready"):
                sock.close()
                return False
            
            # Send data
            sock.sendall(data)
            
            # Get final response
            final_response = self._receive_message(sock)
            sock.close()
            
            if final_response.get("success"):
                new_version = final_response.get("new_version", 1)
                logger.info(f"Pushed model '{model_name}' - now v{new_version}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Push error: {e}")
            return False
    
    def _send_command(self, command: dict) -> dict:
        """Send a command and get response."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self._server_host, self._server_port))
        
        self._send_message(sock, command)
        response = self._receive_message(sock)
        
        sock.close()
        return response
    
    def _send_message(self, sock: socket.socket, data: dict):
        """Send a message with length prefix."""
        msg = json.dumps(data).encode()
        sock.sendall(struct.pack(">I", len(msg)) + msg)
    
    def _receive_message(self, sock: socket.socket) -> dict:
        """Receive a message with length prefix."""
        length_data = sock.recv(4)
        if not length_data:
            return {}
        msg_length = struct.unpack(">I", length_data)[0]
        
        data = b""
        while len(data) < msg_length:
            chunk = sock.recv(min(4096, msg_length - len(data)))
            if not chunk:
                break
            data += chunk
        
        return json.loads(data.decode())
    
    def _calculate_checksum(self, path: Path) -> str:
        """Calculate MD5 checksum."""
        md5 = hashlib.md5()
        
        if path.is_dir():
            for file_path in sorted(path.rglob("*")):
                if file_path.is_file():
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(65536), b""):
                            md5.update(chunk)
        else:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    md5.update(chunk)
        
        return md5.hexdigest()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_sync_server: Optional[ModelSyncServer] = None
_sync_client: Optional[ModelSyncClient] = None


def get_sync_server(port: int = 5060) -> ModelSyncServer:
    """Get or create global sync server."""
    global _sync_server
    if _sync_server is None:
        _sync_server = ModelSyncServer(port=port)
    return _sync_server


def get_sync_client() -> ModelSyncClient:
    """Get or create global sync client."""
    global _sync_client
    if _sync_client is None:
        _sync_client = ModelSyncClient()
    return _sync_client


def start_sync_server(port: int = 5060) -> ModelSyncServer:
    """Start a model sync server."""
    server = get_sync_server(port)
    server.start()
    return server


def sync_from_server(
    server_address: str,
    model_name: str,
    local_path: str
) -> bool:
    """
    Convenience function to sync a model from a server.
    
    Args:
        server_address: Server address (host:port)
        model_name: Name of the model
        local_path: Where to save locally
        
    Returns:
        True if successful
    """
    client = get_sync_client()
    if not client.connect(server_address):
        return False
    return client.sync_model(model_name, local_path)
