"""
OBS Streaming Integration for Enigma AI Engine

Overlay control and streaming integration with OBS Studio.

Features:
- OBS WebSocket control
- Browser source overlays
- Chat display overlay
- Avatar status display
- Scene switching
- Audio routing

Usage:
    from enigma_engine.integrations.obs_streaming import OBSIntegration
    
    obs = OBSIntegration()
    obs.connect()
    
    # Show chat overlay
    obs.show_chat_message("Hello stream!", speaker="AI")
    
    # Switch scenes
    obs.switch_scene("Just Chatting")
"""

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class OverlayType(Enum):
    """Types of stream overlays."""
    CHAT = "chat"
    STATUS = "status"
    AVATAR = "avatar"
    SUBTITLE = "subtitle"
    ALERT = "alert"
    CUSTOM = "custom"


@dataclass
class OBSConfig:
    """OBS connection configuration."""
    host: str = "localhost"
    port: int = 4455
    password: str = ""
    
    # Overlay settings
    overlay_port: int = 8765
    overlay_host: str = "localhost"


@dataclass
class ChatMessage:
    """A chat message for overlay."""
    text: str
    speaker: str = "AI"
    timestamp: float = field(default_factory=time.time)
    color: str = "#ffffff"
    avatar_url: str = ""


@dataclass
class SceneInfo:
    """Information about an OBS scene."""
    name: str
    sources: List[str] = field(default_factory=list)
    is_current: bool = False


class OBSController:
    """Control OBS via WebSocket."""
    
    def __init__(self, config: OBSConfig):
        """
        Initialize OBS controller.
        
        Args:
            config: OBS configuration
        """
        self.config = config
        self._connected = False
        self._ws = None
        self._request_id = 0
        self._callbacks: Dict[str, Callable] = {}
    
    async def connect(self) -> bool:
        """Connect to OBS WebSocket."""
        try:
            import websockets
            
            uri = f"ws://{self.config.host}:{self.config.port}"
            
            self._ws = await websockets.connect(uri)
            
            # Handle OBS WebSocket 5.x authentication
            hello = await self._ws.recv()
            hello_data = json.loads(hello)
            
            if hello_data.get("op") == 0:  # Hello
                # Identify
                identify_msg = {
                    "op": 1,
                    "d": {
                        "rpcVersion": 1
                    }
                }
                
                if self.config.password:
                    import hashlib
                    import base64
                    
                    challenge = hello_data["d"].get("authentication", {})
                    if challenge:
                        salt = challenge.get("salt", "")
                        challenge_str = challenge.get("challenge", "")
                        
                        secret = base64.b64encode(
                            hashlib.sha256(
                                (self.config.password + salt).encode()
                            ).digest()
                        ).decode()
                        
                        auth = base64.b64encode(
                            hashlib.sha256(
                                (secret + challenge_str).encode()
                            ).digest()
                        ).decode()
                        
                        identify_msg["d"]["authentication"] = auth
                
                await self._ws.send(json.dumps(identify_msg))
                
                # Wait for Identified
                identified = await self._ws.recv()
                identified_data = json.loads(identified)
                
                if identified_data.get("op") == 2:  # Identified
                    self._connected = True
                    logger.info("Connected to OBS WebSocket")
                    return True
            
            return False
            
        except ImportError:
            logger.error("websockets not installed")
            return False
        except Exception as e:
            logger.error(f"OBS connection error: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from OBS."""
        if self._ws:
            await self._ws.close()
            self._ws = None
            self._connected = False
    
    async def send_request(
        self,
        request_type: str,
        request_data: Optional[Dict] = None
    ) -> Dict:
        """
        Send request to OBS.
        
        Args:
            request_type: OBS request type
            request_data: Request data
            
        Returns:
            Response data
        """
        if not self._connected:
            return {"error": "Not connected"}
        
        self._request_id += 1
        
        message = {
            "op": 6,  # Request
            "d": {
                "requestType": request_type,
                "requestId": str(self._request_id)
            }
        }
        
        if request_data:
            message["d"]["requestData"] = request_data
        
        await self._ws.send(json.dumps(message))
        
        # Wait for response
        while True:
            response = await self._ws.recv()
            response_data = json.loads(response)
            
            if response_data.get("op") == 7:  # RequestResponse
                if response_data["d"].get("requestId") == str(self._request_id):
                    return response_data["d"].get("responseData", {})
    
    async def get_scenes(self) -> List[SceneInfo]:
        """Get list of scenes."""
        response = await self.send_request("GetSceneList")
        
        scenes = []
        current = response.get("currentProgramSceneName", "")
        
        for scene in response.get("scenes", []):
            scenes.append(SceneInfo(
                name=scene["sceneName"],
                is_current=(scene["sceneName"] == current)
            ))
        
        return scenes
    
    async def switch_scene(self, scene_name: str):
        """Switch to a scene."""
        await self.send_request(
            "SetCurrentProgramScene",
            {"sceneName": scene_name}
        )
    
    async def set_source_visibility(
        self,
        scene_name: str,
        source_name: str,
        visible: bool
    ):
        """Set source visibility."""
        # Get scene item ID
        response = await self.send_request(
            "GetSceneItemId",
            {
                "sceneName": scene_name,
                "sourceName": source_name
            }
        )
        
        item_id = response.get("sceneItemId")
        if item_id:
            await self.send_request(
                "SetSceneItemEnabled",
                {
                    "sceneName": scene_name,
                    "sceneItemId": item_id,
                    "sceneItemEnabled": visible
                }
            )
    
    async def set_browser_source_url(
        self,
        source_name: str,
        url: str
    ):
        """Set browser source URL."""
        await self.send_request(
            "SetInputSettings",
            {
                "inputName": source_name,
                "inputSettings": {"url": url}
            }
        )
    
    async def refresh_browser_source(self, source_name: str):
        """Refresh a browser source."""
        await self.send_request(
            "PressInputPropertiesButton",
            {
                "inputName": source_name,
                "propertyName": "refreshnocache"
            }
        )


class OverlayServer:
    """HTTP server for browser source overlays."""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        """
        Initialize overlay server.
        
        Args:
            host: Server host
            port: Server port
        """
        self.host = host
        self.port = port
        
        # State
        self._messages: List[ChatMessage] = []
        self._status: Dict = {}
        self._alerts: List[Dict] = []
        
        # Server
        self._server = None
        self._app = None
    
    def start(self):
        """Start the overlay server."""
        try:
            from flask import Flask, render_template_string, jsonify
            
            self._app = Flask(__name__)
            
            @self._app.route("/")
            def index():
                return "Enigma AI Engine Stream Overlay Server"
            
            @self._app.route("/chat")
            def chat_overlay():
                return render_template_string(CHAT_OVERLAY_HTML)
            
            @self._app.route("/api/messages")
            def get_messages():
                return jsonify([{
                    "text": m.text,
                    "speaker": m.speaker,
                    "color": m.color,
                    "timestamp": m.timestamp
                } for m in self._messages[-50:]])
            
            @self._app.route("/status")
            def status_overlay():
                return render_template_string(STATUS_OVERLAY_HTML)
            
            @self._app.route("/api/status")
            def get_status():
                return jsonify(self._status)
            
            @self._app.route("/subtitle")
            def subtitle_overlay():
                return render_template_string(SUBTITLE_OVERLAY_HTML)
            
            @self._app.route("/api/subtitle")
            def get_subtitle():
                if self._messages:
                    last = self._messages[-1]
                    return jsonify({
                        "text": last.text,
                        "speaker": last.speaker,
                        "visible": time.time() - last.timestamp < 5.0
                    })
                return jsonify({"text": "", "visible": False})
            
            # Run in thread
            thread = threading.Thread(
                target=lambda: self._app.run(
                    host=self.host,
                    port=self.port,
                    debug=False,
                    use_reloader=False
                ),
                daemon=True
            )
            thread.start()
            
            logger.info(f"Overlay server started on {self.host}:{self.port}")
            
        except ImportError:
            logger.error("Flask not installed for overlay server")
    
    def add_message(self, message: ChatMessage):
        """Add a chat message."""
        self._messages.append(message)
        
        # Keep last 100
        if len(self._messages) > 100:
            self._messages = self._messages[-100:]
    
    def set_status(self, status: Dict):
        """Set status display."""
        self._status = status
    
    def add_alert(self, alert: Dict):
        """Add an alert."""
        self._alerts.append(alert)


class OBSIntegration:
    """Main OBS integration class."""
    
    def __init__(self, config: Optional[OBSConfig] = None):
        """
        Initialize OBS integration.
        
        Args:
            config: OBS configuration
        """
        self.config = config or OBSConfig()
        
        # Components
        self._controller = OBSController(self.config)
        self._overlay = OverlayServer(
            host=self.config.overlay_host,
            port=self.config.overlay_port
        )
        
        # State
        self._connected = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
    
    def connect(self) -> bool:
        """Connect to OBS and start overlay server."""
        # Start async loop in thread
        self._loop = asyncio.new_event_loop()
        
        def run_loop():
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()
        
        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        
        # Connect to OBS
        future = asyncio.run_coroutine_threadsafe(
            self._controller.connect(),
            self._loop
        )
        self._connected = future.result(timeout=10.0)
        
        # Start overlay server
        self._overlay.start()
        
        return self._connected
    
    def disconnect(self):
        """Disconnect from OBS."""
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._controller.disconnect(),
                self._loop
            )
            self._loop.call_soon_threadsafe(self._loop.stop)
    
    def show_chat_message(
        self,
        text: str,
        speaker: str = "AI",
        color: str = "#ffffff"
    ):
        """
        Show a chat message on overlay.
        
        Args:
            text: Message text
            speaker: Speaker name
            color: Text color
        """
        message = ChatMessage(
            text=text,
            speaker=speaker,
            color=color
        )
        self._overlay.add_message(message)
    
    def update_status(self, status: Dict):
        """
        Update status overlay.
        
        Args:
            status: Status data
        """
        self._overlay.set_status(status)
    
    def switch_scene(self, scene_name: str):
        """
        Switch OBS scene.
        
        Args:
            scene_name: Scene to switch to
        """
        if self._connected and self._loop:
            asyncio.run_coroutine_threadsafe(
                self._controller.switch_scene(scene_name),
                self._loop
            )
    
    def set_source_visible(
        self,
        scene_name: str,
        source_name: str,
        visible: bool
    ):
        """Set visibility of a source."""
        if self._connected and self._loop:
            asyncio.run_coroutine_threadsafe(
                self._controller.set_source_visibility(
                    scene_name, source_name, visible
                ),
                self._loop
            )
    
    def get_scenes(self) -> List[SceneInfo]:
        """Get list of scenes."""
        if self._connected and self._loop:
            future = asyncio.run_coroutine_threadsafe(
                self._controller.get_scenes(),
                self._loop
            )
            return future.result(timeout=5.0)
        return []
    
    def get_overlay_urls(self) -> Dict[str, str]:
        """Get URLs for browser sources."""
        base = f"http://{self.config.overlay_host}:{self.config.overlay_port}"
        return {
            "chat": f"{base}/chat",
            "status": f"{base}/status",
            "subtitle": f"{base}/subtitle"
        }


# HTML templates for overlays
CHAT_OVERLAY_HTML = """
<!DOCTYPE html>
<html>
<head>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: transparent;
            font-family: 'Segoe UI', sans-serif;
            overflow: hidden;
        }
        #chat {
            position: absolute;
            bottom: 20px;
            left: 20px;
            width: 400px;
            max-height: 300px;
            overflow: hidden;
        }
        .message {
            background: rgba(0, 0, 0, 0.7);
            border-radius: 8px;
            padding: 10px 15px;
            margin: 5px 0;
            animation: fadeIn 0.3s ease;
        }
        .speaker {
            font-weight: bold;
            color: #4fc3f7;
            margin-bottom: 3px;
        }
        .text {
            color: white;
            word-wrap: break-word;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div id="chat"></div>
    <script>
        let lastCount = 0;
        
        async function update() {
            const response = await fetch('/api/messages');
            const messages = await response.json();
            
            if (messages.length !== lastCount) {
                const chat = document.getElementById('chat');
                chat.innerHTML = messages.slice(-5).map(m => `
                    <div class="message">
                        <div class="speaker">${m.speaker}</div>
                        <div class="text" style="color: ${m.color}">${m.text}</div>
                    </div>
                `).join('');
                lastCount = messages.length;
            }
        }
        
        setInterval(update, 500);
        update();
    </script>
</body>
</html>
"""

STATUS_OVERLAY_HTML = """
<!DOCTYPE html>
<html>
<head>
    <style>
        * { margin: 0; padding: 0; }
        body {
            background: transparent;
            font-family: 'Segoe UI', sans-serif;
        }
        #status {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.7);
            border-radius: 8px;
            padding: 15px;
            color: white;
        }
        .label { color: #888; font-size: 12px; }
        .value { font-size: 18px; font-weight: bold; }
    </style>
</head>
<body>
    <div id="status">
        <div class="label">AI Status</div>
        <div class="value" id="state">Ready</div>
    </div>
    <script>
        async function update() {
            const response = await fetch('/api/status');
            const status = await response.json();
            document.getElementById('state').textContent = status.state || 'Ready';
        }
        setInterval(update, 1000);
        update();
    </script>
</body>
</html>
"""

SUBTITLE_OVERLAY_HTML = """
<!DOCTYPE html>
<html>
<head>
    <style>
        * { margin: 0; padding: 0; }
        body {
            background: transparent;
            font-family: 'Segoe UI', sans-serif;
        }
        #subtitle {
            position: absolute;
            bottom: 50px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.8);
            border-radius: 4px;
            padding: 10px 20px;
            color: white;
            font-size: 24px;
            max-width: 80%;
            text-align: center;
            opacity: 0;
            transition: opacity 0.3s;
        }
        #subtitle.visible { opacity: 1; }
    </style>
</head>
<body>
    <div id="subtitle"></div>
    <script>
        async function update() {
            const response = await fetch('/api/subtitle');
            const data = await response.json();
            const el = document.getElementById('subtitle');
            el.textContent = data.text;
            el.classList.toggle('visible', data.visible);
        }
        setInterval(update, 200);
        update();
    </script>
</body>
</html>
"""


# Global instance
_obs: Optional[OBSIntegration] = None


def get_obs_integration() -> OBSIntegration:
    """Get or create global OBS integration."""
    global _obs
    if _obs is None:
        _obs = OBSIntegration()
    return _obs
