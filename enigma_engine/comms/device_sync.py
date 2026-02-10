"""
================================================================================
Device Sync - Synchronize state across all devices in real-time.
================================================================================

Keeps Pi robot, phone avatar, and gaming PC in sync:
- Avatar expressions sync to phone
- Robot position sync from Pi
- AI state sync across all
- Low-bandwidth delta updates

USAGE:
    from enigma_engine.comms.device_sync import DeviceSync, DeviceType
    
    # On PC (master)
    sync = DeviceSync(role="master", port=5050)
    sync.start()
    
    # On Pi (robot)
    sync = DeviceSync(role="client", master_url="http://pc:5050")
    sync.register(DeviceType.ROBOT, "pi-robot-1")
    
    # On Phone (avatar)
    sync = DeviceSync(role="client", master_url="http://pc:5050")
    sync.register(DeviceType.AVATAR_DISPLAY, "phone-1")
"""

import hashlib
import http.server
import json
import logging
import socketserver
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional
from urllib import request

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Types of devices in the system."""
    MASTER = auto()          # Main AI host (gaming PC)
    AVATAR_DISPLAY = auto()  # Phone displaying avatar
    ROBOT = auto()           # Pi controlling robot
    SENSOR_NODE = auto()     # Pi collecting sensor data
    VOICE_NODE = auto()      # Device handling voice
    COMPUTE_NODE = auto()    # Additional compute power


class SyncPriority(Enum):
    """Priority levels for sync updates."""
    CRITICAL = 1    # Emergency stop, errors
    HIGH = 2        # User commands, expressions
    NORMAL = 3      # Regular state updates
    LOW = 4         # Background data, stats


@dataclass
class SyncState:
    """State that gets synchronized across devices."""
    # Avatar state
    avatar_expression: str = "neutral"
    avatar_animation: str = "idle"
    avatar_speaking: bool = False
    avatar_text: str = ""
    
    # AI state
    ai_thinking: bool = False
    ai_response: str = ""
    current_tool: str = ""
    
    # Robot state
    robot_x: float = 0.0
    robot_y: float = 0.0
    robot_heading: float = 0.0
    robot_speed: float = 0.0
    robot_battery: float = 100.0
    robot_emergency_stop: bool = False
    
    # Sensor data
    obstacle_distance: float = float('inf')
    temperature: float = 0.0
    light_level: float = 0.0
    
    # System state
    gaming_mode: bool = False
    voice_enabled: bool = True
    connection_quality: str = "good"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "avatar_expression": self.avatar_expression,
            "avatar_animation": self.avatar_animation,
            "avatar_speaking": self.avatar_speaking,
            "avatar_text": self.avatar_text,
            "ai_thinking": self.ai_thinking,
            "ai_response": self.ai_response,
            "current_tool": self.current_tool,
            "robot_x": self.robot_x,
            "robot_y": self.robot_y,
            "robot_heading": self.robot_heading,
            "robot_speed": self.robot_speed,
            "robot_battery": self.robot_battery,
            "robot_emergency_stop": self.robot_emergency_stop,
            "obstacle_distance": self.obstacle_distance,
            "temperature": self.temperature,
            "light_level": self.light_level,
            "gaming_mode": self.gaming_mode,
            "voice_enabled": self.voice_enabled,
            "connection_quality": self.connection_quality,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'SyncState':
        """Create from dictionary."""
        state = cls()
        for key, value in data.items():
            if hasattr(state, key):
                setattr(state, key, value)
        return state
    
    def get_hash(self) -> str:
        """Get hash of current state for change detection."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_delta(self, other: 'SyncState') -> dict[str, Any]:
        """Get only changed fields compared to another state."""
        delta = {}
        self_dict = self.to_dict()
        other_dict = other.to_dict()
        
        for key, value in self_dict.items():
            if other_dict.get(key) != value:
                delta[key] = value
        
        return delta


@dataclass
class ConnectedDevice:
    """A device connected to the sync network."""
    device_id: str
    device_type: DeviceType
    address: str
    last_seen: float = field(default_factory=time.time)
    last_state_hash: str = ""
    subscriptions: set[str] = field(default_factory=set)  # State fields to receive
    capabilities: set[str] = field(default_factory=set)   # What it can do
    
    def is_alive(self, timeout: float = 30.0) -> bool:
        """Check if device is still responding."""
        return time.time() - self.last_seen < timeout


class DeviceSync:
    """
    Synchronizes state across multiple devices.
    
    Master mode: Runs on main PC, coordinates all devices
    Client mode: Runs on Pi/phone, connects to master
    """
    
    def __init__(
        self,
        role: str = "client",  # "master" or "client"
        master_url: str = "http://localhost:5050",
        port: int = 5050,
        sync_interval: float = 0.1,  # 100ms sync
    ):
        self.role = role
        self.master_url = master_url
        self.port = port
        self.sync_interval = sync_interval
        
        # Current state
        self._state = SyncState()
        self._state_lock = threading.Lock()
        self._last_state_hash = ""
        
        # Connected devices (master only)
        self._devices: dict[str, ConnectedDevice] = {}
        self._devices_lock = threading.Lock()
        
        # My device info (client only)
        self._my_device_id: str = ""
        self._my_device_type: Optional[DeviceType] = None
        
        # Update callbacks
        self._callbacks: dict[str, list[Callable[[str, Any], None]]] = {}
        
        # Update queue for outgoing changes
        self._update_queue: deque = deque(maxlen=100)
        
        # Server (master only)
        self._server: Optional[socketserver.TCPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        
        # Sync thread
        self._sync_thread: Optional[threading.Thread] = None
        self._running = False
    
    def start(self):
        """Start the sync service."""
        self._running = True
        
        if self.role == "master":
            self._start_server()
        
        # Start sync thread
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()
        
        logger.info(f"DeviceSync started as {self.role}")
    
    def stop(self):
        """Stop the sync service."""
        self._running = False
        
        if self._server:
            self._server.shutdown()
        
        if self._sync_thread:
            self._sync_thread.join(timeout=2.0)
    
    def register(self, device_type: DeviceType, device_id: str, capabilities: set[str] = None):
        """Register this device with the master (client mode)."""
        if self.role != "client":
            return
        
        self._my_device_id = device_id
        self._my_device_type = device_type
        
        try:
            data = {
                "action": "register",
                "device_id": device_id,
                "device_type": device_type.name,
                "capabilities": list(capabilities or set()),
            }
            
            json_data = json.dumps(data).encode('utf-8')
            req = request.Request(
                f"{self.master_url}/sync",
                data=json_data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with request.urlopen(req, timeout=5.0) as response:
                result = json.loads(response.read().decode('utf-8'))
                logger.info(f"Registered as {device_id}: {result}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register: {e}")
            return False
    
    def update(self, **kwargs):
        """Update state values."""
        with self._state_lock:
            for key, value in kwargs.items():
                if hasattr(self._state, key):
                    old_value = getattr(self._state, key)
                    if old_value != value:
                        setattr(self._state, key, value)
                        self._queue_update(key, value)
                        self._notify_callbacks(key, value)
    
    def get(self, key: str) -> Any:
        """Get a state value."""
        with self._state_lock:
            return getattr(self._state, key, None)
    
    def get_state(self) -> SyncState:
        """Get current state."""
        with self._state_lock:
            return SyncState.from_dict(self._state.to_dict())
    
    def subscribe(self, field: str, callback: Callable[[str, Any], None]):
        """Subscribe to changes on a specific field."""
        if field not in self._callbacks:
            self._callbacks[field] = []
        self._callbacks[field].append(callback)
    
    def subscribe_all(self, callback: Callable[[str, Any], None]):
        """Subscribe to all state changes."""
        self.subscribe("*", callback)
    
    def get_connected_devices(self) -> list[dict[str, Any]]:
        """Get list of connected devices (master only)."""
        with self._devices_lock:
            return [
                {
                    "device_id": d.device_id,
                    "device_type": d.device_type.name,
                    "address": d.address,
                    "alive": d.is_alive(),
                    "capabilities": list(d.capabilities),
                }
                for d in self._devices.values()
            ]
    
    def send_command(self, device_id: str, command: str, params: dict[str, Any] = None):
        """Send a command to a specific device."""
        if self.role == "master":
            self._queue_command(device_id, command, params or {})
        else:
            # Forward to master
            self._send_to_master({
                "action": "command",
                "target": device_id,
                "command": command,
                "params": params or {},
            })
    
    def _start_server(self):
        """Start HTTP server for sync (master only)."""
        handler = self._create_handler()
        self._server = socketserver.TCPServer(("0.0.0.0", self.port), handler)
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        logger.info(f"Sync server started on port {self.port}")
    
    def _create_handler(self):
        """Create HTTP request handler."""
        sync = self
        
        class SyncHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress logging
            
            def do_POST(self):
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)
                
                try:
                    data = json.loads(body.decode('utf-8'))
                    response = sync._handle_request(data, self.client_address[0])
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode('utf-8'))
                    
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
            
            def do_GET(self):
                # Return current state
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                
                with sync._state_lock:
                    self.wfile.write(json.dumps(sync._state.to_dict()).encode('utf-8'))
        
        return SyncHandler
    
    def _handle_request(self, data: dict[str, Any], address: str) -> dict[str, Any]:
        """Handle incoming sync request (master only)."""
        action = data.get("action", "sync")
        
        if action == "register":
            return self._handle_register(data, address)
        elif action == "sync":
            return self._handle_sync(data, address)
        elif action == "command":
            return self._handle_command(data)
        elif action == "heartbeat":
            return self._handle_heartbeat(data, address)
        else:
            return {"error": f"Unknown action: {action}"}
    
    def _handle_register(self, data: dict[str, Any], address: str) -> dict[str, Any]:
        """Handle device registration."""
        device_id = data.get("device_id", "")
        device_type_str = data.get("device_type", "COMPUTE_NODE")
        capabilities = set(data.get("capabilities", []))
        
        try:
            device_type = DeviceType[device_type_str]
        except KeyError:
            device_type = DeviceType.COMPUTE_NODE
        
        with self._devices_lock:
            self._devices[device_id] = ConnectedDevice(
                device_id=device_id,
                device_type=device_type,
                address=address,
                capabilities=capabilities,
            )
        
        logger.info(f"Device registered: {device_id} ({device_type.name}) from {address}")
        
        return {
            "status": "registered",
            "device_id": device_id,
            "state": self._state.to_dict(),
        }
    
    def _handle_sync(self, data: dict[str, Any], address: str) -> dict[str, Any]:
        """Handle sync request."""
        device_id = data.get("device_id", "")
        updates = data.get("updates", {})
        last_hash = data.get("last_hash", "")
        
        # Apply updates from device
        if updates:
            with self._state_lock:
                for key, value in updates.items():
                    if hasattr(self._state, key):
                        setattr(self._state, key, value)
                        self._notify_callbacks(key, value)
        
        # Update device last seen
        with self._devices_lock:
            if device_id in self._devices:
                self._devices[device_id].last_seen = time.time()
        
        # Return delta or full state
        with self._state_lock:
            current_hash = self._state.get_hash()
            
            if last_hash and last_hash == current_hash:
                return {"status": "no_change", "hash": current_hash}
            
            return {
                "status": "ok",
                "state": self._state.to_dict(),
                "hash": current_hash,
            }
    
    def _handle_heartbeat(self, data: dict[str, Any], address: str) -> dict[str, Any]:
        """Handle heartbeat from device."""
        device_id = data.get("device_id", "")
        
        with self._devices_lock:
            if device_id in self._devices:
                self._devices[device_id].last_seen = time.time()
        
        return {"status": "ok"}
    
    def _handle_command(self, data: dict[str, Any]) -> dict[str, Any]:
        """Handle command forwarding."""
        # Store command for target device
        target = data.get("target", "")
        command = data.get("command", "")
        params = data.get("params", {})
        
        self._queue_command(target, command, params)
        
        return {"status": "queued"}
    
    def _sync_loop(self):
        """Main sync loop."""
        while self._running:
            try:
                if self.role == "client":
                    self._sync_with_master()
                else:
                    self._broadcast_to_devices()
                
            except Exception as e:
                logger.debug(f"Sync error: {e}")
            
            time.sleep(self.sync_interval)
    
    def _sync_with_master(self):
        """Sync state with master (client mode)."""
        if not self._my_device_id:
            return
        
        try:
            # Gather local updates
            updates = {}
            while self._update_queue:
                key, value = self._update_queue.popleft()
                updates[key] = value
            
            data = {
                "action": "sync",
                "device_id": self._my_device_id,
                "updates": updates,
                "last_hash": self._last_state_hash,
            }
            
            json_data = json.dumps(data).encode('utf-8')
            req = request.Request(
                f"{self.master_url}/sync",
                data=json_data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with request.urlopen(req, timeout=5.0) as response:
                result = json.loads(response.read().decode('utf-8'))
                
                if result.get("status") == "ok":
                    # Apply state from master
                    with self._state_lock:
                        for key, value in result.get("state", {}).items():
                            if hasattr(self._state, key):
                                old = getattr(self._state, key)
                                if old != value:
                                    setattr(self._state, key, value)
                                    self._notify_callbacks(key, value)
                        
                        self._last_state_hash = result.get("hash", "")
                
        except Exception as e:
            logger.debug(f"Sync with master failed: {e}")
    
    def _broadcast_to_devices(self):
        """Broadcast state to all devices (master mode)."""
        # Clean up dead devices
        with self._devices_lock:
            dead = [d_id for d_id, d in self._devices.items() if not d.is_alive(timeout=60.0)]
            for d_id in dead:
                logger.info(f"Device disconnected: {d_id}")
                del self._devices[d_id]
    
    def _queue_update(self, key: str, value: Any):
        """Queue a state update for sync."""
        self._update_queue.append((key, value))
    
    def _queue_command(self, device_id: str, command: str, params: dict[str, Any]):
        """Queue a command for a device."""
        # Commands are delivered on next sync
    
    def _notify_callbacks(self, key: str, value: Any):
        """Notify subscribers of state change."""
        # Field-specific callbacks
        for callback in self._callbacks.get(key, []):
            try:
                callback(key, value)
            except Exception as e:
                logger.error(f"Callback error: {e}")
        
        # Wildcard callbacks
        for callback in self._callbacks.get("*", []):
            try:
                callback(key, value)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def _send_to_master(self, data: dict[str, Any]):
        """Send data to master."""
        try:
            json_data = json.dumps(data).encode('utf-8')
            req = request.Request(
                f"{self.master_url}/sync",
                data=json_data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with request.urlopen(req, timeout=5.0) as response:
                return json.loads(response.read().decode('utf-8'))
                
        except Exception as e:
            logger.debug(f"Send to master failed: {e}")
            return None


# Global sync instance
_sync: Optional[DeviceSync] = None


def get_device_sync(**kwargs) -> DeviceSync:
    """Get or create global device sync."""
    global _sync
    if _sync is None:
        _sync = DeviceSync(**kwargs)
    return _sync


__all__ = [
    'DeviceSync',
    'DeviceType',
    'SyncPriority',
    'SyncState',
    'ConnectedDevice',
    'get_device_sync',
]
