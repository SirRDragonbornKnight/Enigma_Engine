"""
================================================================================
Mobile Avatar - Lightweight avatar system for phones/tablets.
================================================================================

Optimized for mobile devices:
- Minimal memory footprint
- Touch-friendly controls
- Battery-efficient updates
- Network-synced with desktop

USAGE:
    from forge_ai.avatar.mobile_avatar import MobileAvatar
    
    # On phone
    avatar = MobileAvatar()
    avatar.connect_to_server("192.168.1.100:5000")
    avatar.start()  # Shows avatar on phone screen
    
    # Avatar can:
    # - Display expressions synced from PC
    # - Show speech bubbles
    # - React to touch
    # - Send commands back to PC

ARCHITECTURE:
    [Desktop PC]                    [Phone]
    ┌─────────────┐                ┌─────────────┐
    │ Full Avatar │  ──websocket─▶ │MobileAvatar │
    │ Controller  │  ◀─touch/cmd── │  (display)  │
    └─────────────┘                └─────────────┘
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class MobileAvatarState(Enum):
    """Mobile avatar states (simplified)."""
    DISCONNECTED = "disconnected"
    IDLE = "idle"
    SPEAKING = "speaking"
    REACTING = "reacting"


@dataclass
class MobileAvatarConfig:
    """Configuration for mobile avatar."""
    # Display settings
    avatar_size: int = 200          # Avatar size in pixels
    speech_bubble_enabled: bool = True
    touch_reactions: bool = True
    vibration_feedback: bool = True
    
    # Performance settings
    update_rate_ms: int = 100       # Update rate (lower = smoother, more battery)
    image_quality: int = 70         # JPEG quality for avatar images
    cache_expressions: bool = True  # Cache expression images
    
    # Network settings
    server_url: str = ""
    reconnect_interval: int = 5     # Seconds between reconnect attempts
    sync_interval_ms: int = 500     # State sync interval
    
    # Battery saving
    reduce_when_background: bool = True
    sleep_after_idle_mins: int = 5


@dataclass
class AvatarExpression:
    """A cached avatar expression."""
    name: str
    image_data: bytes = b""         # Pre-rendered image
    width: int = 200
    height: int = 200
    frame_count: int = 1            # For animated expressions


class MobileAvatar:
    """
    Lightweight avatar display for mobile devices.
    
    Features:
    - Receives state from desktop via websocket
    - Renders simple 2D avatar
    - Touch reactions
    - Speech bubble display
    - Battery-efficient operation
    """
    
    def __init__(self, config: MobileAvatarConfig = None):
        self.config = config or MobileAvatarConfig()
        self._state = MobileAvatarState.DISCONNECTED
        self._current_expression = "neutral"
        self._speech_text = ""
        self._speech_visible = False
        
        # Expression cache
        self._expressions: Dict[str, AvatarExpression] = {}
        
        # Network
        self._connected = False
        self._ws_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Touch handling
        self._touch_callbacks: List[Callable[[int, int], None]] = []
        
        # Battery saving
        self._is_background = False
        self._last_activity = time.time()
        
        # Initialize default expressions
        self._init_default_expressions()
    
    def _init_default_expressions(self):
        """Initialize default expression placeholders."""
        default_names = [
            "neutral", "happy", "sad", "surprised",
            "thinking", "speaking", "sleeping"
        ]
        for name in default_names:
            self._expressions[name] = AvatarExpression(name=name)
    
    @property
    def state(self) -> MobileAvatarState:
        """Get current state."""
        return self._state
    
    @property
    def connected(self) -> bool:
        """Check if connected to server."""
        return self._connected
    
    def connect_to_server(self, server_url: str) -> bool:
        """
        Connect to desktop avatar server.
        
        Args:
            server_url: WebSocket URL (e.g., "ws://192.168.1.100:5001/avatar")
            
        Returns:
            True if connection initiated
        """
        self.config.server_url = server_url
        
        # Start connection thread
        self._stop_event.clear()
        self._ws_thread = threading.Thread(
            target=self._connection_loop,
            daemon=True,
            name="MobileAvatarWS",
        )
        self._ws_thread.start()
        
        return True
    
    def disconnect(self):
        """Disconnect from server."""
        self._stop_event.set()
        self._connected = False
        self._state = MobileAvatarState.DISCONNECTED
        
        if self._ws_thread:
            self._ws_thread.join(timeout=2.0)
            self._ws_thread = None
    
    def _connection_loop(self):
        """WebSocket connection loop."""
        while not self._stop_event.is_set():
            try:
                self._connect_websocket()
            except Exception as e:
                logger.warning(f"Connection failed: {e}")
                self._connected = False
                self._state = MobileAvatarState.DISCONNECTED
            
            if not self._stop_event.is_set():
                self._stop_event.wait(self.config.reconnect_interval)
    
    def _connect_websocket(self):
        """Establish WebSocket connection."""
        try:
            import websocket
            
            def on_message(ws, message):
                self._handle_message(json.loads(message))
            
            def on_error(ws, error):
                logger.error(f"WebSocket error: {error}")
            
            def on_close(ws, close_code, close_msg):
                self._connected = False
                self._state = MobileAvatarState.DISCONNECTED
            
            def on_open(ws):
                self._connected = True
                self._state = MobileAvatarState.IDLE
                logger.info("Connected to avatar server")
                # Request current state
                ws.send(json.dumps({"type": "sync_request"}))
            
            ws = websocket.WebSocketApp(
                self.config.server_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open,
            )
            
            ws.run_forever()
            
        except ImportError:
            # Fall back to polling-based sync
            self._poll_based_sync()
    
    def _poll_based_sync(self):
        """Fallback: Poll-based state synchronization."""
        import urllib.request
        
        # Convert ws:// to http://
        http_url = self.config.server_url.replace("ws://", "http://").replace("/avatar", "")
        
        while not self._stop_event.is_set():
            try:
                req = urllib.request.Request(f"{http_url}/avatar/state")
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    self._handle_message(data)
                    self._connected = True
                    if self._state == MobileAvatarState.DISCONNECTED:
                        self._state = MobileAvatarState.IDLE
            except Exception as e:
                logger.debug(f"Poll failed: {e}")
                self._connected = False
            
            self._stop_event.wait(self.config.sync_interval_ms / 1000.0)
    
    def _handle_message(self, data: Dict[str, Any]):
        """Handle message from server."""
        msg_type = data.get("type", "")
        
        if msg_type == "state_update":
            self._update_state(data)
        elif msg_type == "expression_change":
            self._set_expression(data.get("expression", "neutral"))
        elif msg_type == "speak":
            self._show_speech(data.get("text", ""))
        elif msg_type == "expression_image":
            self._cache_expression_image(data)
        
        self._last_activity = time.time()
    
    def _update_state(self, data: Dict[str, Any]):
        """Update avatar state from server data."""
        state_str = data.get("state", "idle")
        try:
            self._state = MobileAvatarState[state_str.upper()]
        except KeyError:
            self._state = MobileAvatarState.IDLE
        
        if "expression" in data:
            self._current_expression = data["expression"]
    
    def _set_expression(self, expression: str):
        """Set current expression."""
        self._current_expression = expression
        if expression == "speaking":
            self._state = MobileAvatarState.SPEAKING
        elif self._state == MobileAvatarState.SPEAKING:
            self._state = MobileAvatarState.IDLE
    
    def _show_speech(self, text: str):
        """Show speech bubble."""
        self._speech_text = text
        self._speech_visible = bool(text)
        if text:
            self._state = MobileAvatarState.SPEAKING
    
    def _cache_expression_image(self, data: Dict[str, Any]):
        """Cache an expression image from server."""
        name = data.get("name", "")
        if not name:
            return
        
        import base64
        image_b64 = data.get("image", "")
        if image_b64:
            image_data = base64.b64decode(image_b64)
            self._expressions[name] = AvatarExpression(
                name=name,
                image_data=image_data,
                width=data.get("width", 200),
                height=data.get("height", 200),
            )
    
    def on_touch(self, x: int, y: int):
        """
        Handle touch event on avatar.
        
        Args:
            x: Touch X coordinate
            y: Touch Y coordinate
        """
        if not self.config.touch_reactions:
            return
        
        self._last_activity = time.time()
        
        # Trigger reaction
        self._state = MobileAvatarState.REACTING
        
        # Notify callbacks
        for callback in self._touch_callbacks:
            try:
                callback(x, y)
            except Exception as e:
                logger.error(f"Touch callback error: {e}")
        
        # Send touch event to server
        if self._connected:
            self._send_event({
                "type": "touch",
                "x": x,
                "y": y,
            })
    
    def _send_event(self, event: Dict[str, Any]):
        """Send event to server."""
        # TODO: Implement based on connection type
        pass
    
    def send_command(self, command: str, **kwargs):
        """
        Send command to desktop avatar.
        
        Args:
            command: Command name (e.g., "speak", "move", "expression")
            **kwargs: Command parameters
        """
        if not self._connected:
            logger.warning("Not connected to server")
            return
        
        self._send_event({
            "type": "command",
            "command": command,
            **kwargs,
        })
    
    def get_render_data(self) -> Dict[str, Any]:
        """
        Get data needed to render the avatar.
        
        Returns dict with:
        - expression: Current expression name
        - expression_image: Cached image bytes (if available)
        - speech_text: Current speech bubble text
        - speech_visible: Whether to show speech bubble
        - state: Current state
        """
        expr = self._expressions.get(self._current_expression)
        
        return {
            "expression": self._current_expression,
            "expression_image": expr.image_data if expr else b"",
            "speech_text": self._speech_text,
            "speech_visible": self._speech_visible,
            "state": self._state.value,
        }
    
    def register_touch_callback(self, callback: Callable[[int, int], None]):
        """Register callback for touch events."""
        self._touch_callbacks.append(callback)
    
    def set_background(self, is_background: bool):
        """
        Notify avatar that app is in background.
        
        When in background, reduces update rate to save battery.
        """
        self._is_background = is_background
        
        if is_background and self.config.reduce_when_background:
            # Reduce update rate to save battery (4x slower)
            self._background_update_rate = self.config.update_rate_ms * 4
            self._original_update_rate = self.config.update_rate_ms
            self.config.update_rate_ms = self._background_update_rate
            logger.debug(f"Background mode: reduced update rate to {self._background_update_rate}ms")
        elif not is_background and hasattr(self, '_original_update_rate'):
            # Restore normal update rate when returning to foreground
            self.config.update_rate_ms = self._original_update_rate
            logger.debug(f"Foreground mode: restored update rate to {self._original_update_rate}ms")
    
    def get_battery_stats(self) -> Dict[str, Any]:
        """Get battery usage statistics."""
        return {
            "update_rate_ms": self.config.update_rate_ms,
            "is_background": self._is_background,
            "last_activity_secs_ago": time.time() - self._last_activity,
            "cached_expressions": len(self._expressions),
        }


class MobileAvatarServer:
    """
    Server-side component to send avatar state to mobile.
    
    Run this on the desktop to sync avatar state to phone.
    """
    
    def __init__(self, avatar_controller, port: int = 5001):
        """
        Initialize mobile avatar server.
        
        Args:
            avatar_controller: Desktop AvatarController instance
            port: Port to listen on
        """
        self._avatar = avatar_controller
        self._port = port
        self._clients: List[Any] = []
        self._running = False
        self._server_thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start the server."""
        self._running = True
        self._server_thread = threading.Thread(
            target=self._run_server,
            daemon=True,
            name="MobileAvatarServer",
        )
        self._server_thread.start()
        logger.info(f"Mobile avatar server started on port {self._port}")
    
    def stop(self):
        """Stop the server."""
        self._running = False
        if self._server_thread:
            self._server_thread.join(timeout=2.0)
    
    def _run_server(self):
        """Run the WebSocket server."""
        try:
            from flask import Flask, jsonify
            from flask_cors import CORS
            
            app = Flask("mobile_avatar")
            CORS(app)
            
            @app.route("/avatar/state")
            def get_state():
                """Get current avatar state."""
                if self._avatar:
                    return jsonify({
                        "type": "state_update",
                        "state": self._avatar.state.value if hasattr(self._avatar, 'state') else "idle",
                        "expression": getattr(self._avatar, '_current_expression', 'neutral'),
                        "position": {
                            "x": getattr(self._avatar, 'x', 0),
                            "y": getattr(self._avatar, 'y', 0),
                        },
                    })
                return jsonify({"type": "state_update", "state": "idle"})
            
            @app.route("/avatar/command", methods=["POST"])
            def handle_command():
                """Handle command from mobile."""
                from flask import request
                data = request.json or {}
                cmd = data.get("command", "")
                
                if self._avatar and cmd:
                    if cmd == "speak" and hasattr(self._avatar, 'speak'):
                        self._avatar.speak(data.get("text", ""))
                    elif cmd == "expression" and hasattr(self._avatar, 'set_expression'):
                        self._avatar.set_expression(data.get("expression", "neutral"))
                
                return jsonify({"success": True})
            
            app.run(host="0.0.0.0", port=self._port, debug=False, use_reloader=False)
            
        except ImportError:
            logger.error("Flask not installed, mobile avatar server unavailable")
    
    def broadcast_state(self, state: Dict[str, Any]):
        """Broadcast state update to all connected clients via WebSocket."""
        if not hasattr(self, '_websocket_clients'):
            self._websocket_clients = set()
        
        # Try WebSocket broadcast (flask-socketio or websockets)
        try:
            from flask_socketio import emit
            emit('avatar_state', state, broadcast=True, namespace='/avatar')
            return
        except ImportError:
            pass
        
        # Fallback: HTTP Server-Sent Events (SSE) style
        # Store state for polling clients
        self._last_broadcast_state = state
        self._last_broadcast_time = time.time()
        
        # Also try simple socket broadcast for any raw WebSocket clients
        dead_clients = set()
        for ws_client in self._websocket_clients:
            try:
                import json
                ws_client.send(json.dumps({'type': 'avatar_state', 'data': state}))
            except Exception:
                dead_clients.add(ws_client)
        
        # Remove dead clients
        self._websocket_clients -= dead_clients


def create_mobile_avatar_server(avatar_controller, port: int = 5001) -> MobileAvatarServer:
    """Create and start mobile avatar server."""
    server = MobileAvatarServer(avatar_controller, port)
    server.start()
    return server


__all__ = [
    'MobileAvatar',
    'MobileAvatarConfig',
    'MobileAvatarState',
    'MobileAvatarServer',
    'create_mobile_avatar_server',
]
