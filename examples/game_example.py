"""
Game Integration Example for ForgeAI

This example shows how to connect ForgeAI to games.
The AI can play games, control characters, or act as an NPC.

SUPPORTED CONNECTIONS:
- WebSocket (real-time bidirectional)
- HTTP/REST API (request/response)
- Memory reading (direct game memory)
- Screen capture (vision-based)

USAGE:
    python examples/game_example.py
    
Or import in your own code:
    from examples.game_example import create_game_connection
"""

import json
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List, Callable


# =============================================================================
# GAME CONNECTION INTERFACES
# =============================================================================

class GameState(Enum):
    """Game connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    IN_GAME = "in_game"
    ERROR = "error"


@dataclass
class GameInfo:
    """Information about connected game."""
    name: str
    version: str = ""
    player_name: str = ""
    current_level: str = ""
    extra: Dict[str, Any] = None


class GameInterface(ABC):
    """
    Abstract base class for game connections.
    Implement this for your specific game.
    """
    
    def __init__(self, name: str = "game"):
        self.name = name
        self.state = GameState.DISCONNECTED
        self.info: Optional[GameInfo] = None
        self._callbacks: Dict[str, List[Callable]] = {}
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the game. Return True if successful."""
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the game."""
    
    @abstractmethod
    def send_command(self, command: str, **kwargs) -> bool:
        """Send a command to the game."""
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get current game state."""
    
    def on_event(self, event_type: str, callback: Callable):
        """Register callback for game events."""
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)
    
    def _emit(self, event_type: str, data: Any):
        """Emit event to registered callbacks."""
        for cb in self._callbacks.get(event_type, []):
            try:
                cb(data)
            except Exception as e:
                print(f"[GAME] Callback error: {e}")


# =============================================================================
# EXAMPLE 1: WebSocket Game Connection
# =============================================================================

class WebSocketGameInterface(GameInterface):
    """
    Connect to games via WebSocket.
    
    Many games and game mods expose WebSocket APIs.
    Examples: Minecraft mods, custom game servers, web games.
    
    Game should handle messages like:
        {"type": "command", "action": "move", "direction": "forward"}
        {"type": "command", "action": "say", "message": "Hello!"}
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765, name: str = "ws_game"):
        super().__init__(name)
        self.host = host
        self.port = port
        self.ws = None
        self._receive_thread = None
        self._running = False
    
    def connect(self) -> bool:
        try:
            import websocket
            
            url = f"ws://{self.host}:{self.port}"
            self.ws = websocket.create_connection(url, timeout=5)
            self.state = GameState.CONNECTED
            
            # Start receive thread
            self._running = True
            self._receive_thread = threading.Thread(target=self._receive_loop)
            self._receive_thread.daemon = True
            self._receive_thread.start()
            
            print(f"[GAME] Connected to {url}")
            return True
            
        except ImportError:
            print("[GAME] ERROR: Install websocket-client:")
            print("  pip install websocket-client")
            return False
        except Exception as e:
            print(f"[GAME] Connection failed: {e}")
            self.state = GameState.ERROR
            return False
    
    def disconnect(self) -> bool:
        self._running = False
        if self.ws:
            self.ws.close()
            self.ws = None
        self.state = GameState.DISCONNECTED
        return True
    
    def send_command(self, command: str, **kwargs) -> bool:
        if not self.ws:
            return False
        
        try:
            msg = {"type": "command", "action": command, **kwargs}
            self.ws.send(json.dumps(msg))
            return True
        except Exception as e:
            print(f"[GAME] Send failed: {e}")
            return False
    
    def get_state(self) -> Dict[str, Any]:
        if not self.ws:
            return {}
        
        try:
            self.ws.send(json.dumps({"type": "get_state"}))
            response = self.ws.recv()
            return json.loads(response)
        except Exception as e:
            print(f"[GAME] Get state failed: {e}")
            return {}
    
    def _receive_loop(self):
        """Background thread to receive game events."""
        while self._running and self.ws:
            try:
                msg = self.ws.recv()
                data = json.loads(msg)
                event_type = data.get("type", "unknown")
                self._emit(event_type, data)
            except Exception as e:
                if self._running:
                    print(f"[GAME] Receive error: {e}")
                break


# =============================================================================
# EXAMPLE 2: HTTP/REST API Game Connection
# =============================================================================

class HTTPGameInterface(GameInterface):
    """
    Connect to games via HTTP REST API.
    
    For games that expose HTTP endpoints.
    Examples: Game servers with REST APIs, browser games.
    """
    
    def __init__(self, base_url: str = "http://localhost:8080", name: str = "http_game"):
        super().__init__(name)
        self.base_url = base_url.rstrip('/')
    
    def connect(self) -> bool:
        try:
            import urllib.request
            
            # Test connection
            req = urllib.request.Request(f"{self.base_url}/status")
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                if data.get("ok"):
                    self.state = GameState.CONNECTED
                    self.info = GameInfo(
                        name=data.get("game", "Unknown"),
                        version=data.get("version", ""),
                    )
                    print(f"[GAME] Connected to {self.info.name}")
                    return True
        except Exception as e:
            print(f"[GAME] Connection failed: {e}")
        
        self.state = GameState.ERROR
        return False
    
    def disconnect(self) -> bool:
        self.state = GameState.DISCONNECTED
        return True
    
    def send_command(self, command: str, **kwargs) -> bool:
        try:
            import urllib.request
            
            data = json.dumps({"command": command, **kwargs}).encode()
            req = urllib.request.Request(
                f"{self.base_url}/command",
                data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())
                return result.get("success", False)
        except Exception as e:
            print(f"[GAME] Command failed: {e}")
            return False
    
    def get_state(self) -> Dict[str, Any]:
        try:
            import urllib.request
            
            req = urllib.request.Request(f"{self.base_url}/state")
            with urllib.request.urlopen(req, timeout=5) as response:
                return json.loads(response.read().decode())
        except Exception as e:
            print(f"[GAME] Get state failed: {e}")
            return {}


# =============================================================================
# EXAMPLE 3: Screen Capture Game (Vision-based)
# =============================================================================

class ScreenCaptureGameInterface(GameInterface):
    """
    Control games via screen capture and input simulation.
    
    Works with ANY game - no API needed!
    Uses ForgeAI's vision to understand what's on screen.
    
    Requirements:
        pip install pyautogui pillow mss
    """
    
    def __init__(self, name: str = "screen_game"):
        super().__init__(name)
        self.screen_region = None  # (x, y, width, height) or None for full screen
        self._last_frame = None
    
    def connect(self) -> bool:
        try:
            import pyautogui
            import mss
            
            self.state = GameState.CONNECTED
            print("[GAME] Screen capture ready")
            print("[GAME] Make sure your game window is visible!")
            return True
            
        except ImportError:
            print("[GAME] ERROR: Install required libraries:")
            print("  pip install pyautogui pillow mss")
            return False
    
    def disconnect(self) -> bool:
        self.state = GameState.DISCONNECTED
        return True
    
    def send_command(self, command: str, **kwargs) -> bool:
        """
        Send input commands to the game.
        
        Commands:
            key_press: Press a key (key="w")
            key_hold: Hold a key (key="shift", duration=1.0)
            mouse_click: Click (x=100, y=200, button="left")
            mouse_move: Move mouse (x=100, y=200)
            type_text: Type text (text="Hello")
        """
        try:
            import pyautogui
            
            if command == "key_press":
                pyautogui.press(kwargs.get("key", "space"))
            elif command == "key_hold":
                pyautogui.keyDown(kwargs.get("key", "shift"))
                time.sleep(kwargs.get("duration", 0.5))
                pyautogui.keyUp(kwargs.get("key", "shift"))
            elif command == "mouse_click":
                pyautogui.click(
                    x=kwargs.get("x"),
                    y=kwargs.get("y"),
                    button=kwargs.get("button", "left")
                )
            elif command == "mouse_move":
                pyautogui.moveTo(kwargs.get("x"), kwargs.get("y"))
            elif command == "type_text":
                pyautogui.typewrite(kwargs.get("text", ""), interval=0.05)
            else:
                print(f"[GAME] Unknown command: {command}")
                return False
            
            return True
            
        except Exception as e:
            print(f"[GAME] Input failed: {e}")
            return False
    
    def get_state(self) -> Dict[str, Any]:
        """Capture current screen state."""
        try:
            import mss
            from PIL import Image
            
            with mss.mss() as sct:
                if self.screen_region:
                    monitor = {
                        "left": self.screen_region[0],
                        "top": self.screen_region[1],
                        "width": self.screen_region[2],
                        "height": self.screen_region[3],
                    }
                else:
                    monitor = sct.monitors[1]  # Primary monitor
                
                screenshot = sct.grab(monitor)
                self._last_frame = Image.frombytes(
                    "RGB",
                    (screenshot.width, screenshot.height),
                    screenshot.rgb
                )
            
            return {
                "has_frame": True,
                "width": self._last_frame.width,
                "height": self._last_frame.height,
            }
            
        except Exception as e:
            print(f"[GAME] Screen capture failed: {e}")
            return {"has_frame": False}
    
    def get_frame(self):
        """Get the last captured frame (PIL Image)."""
        return self._last_frame
    
    def set_region(self, x: int, y: int, width: int, height: int):
        """Set screen capture region."""
        self.screen_region = (x, y, width, height)


# =============================================================================
# EXAMPLE 4: Simulated Game (for testing)
# =============================================================================

class SimulatedGameInterface(GameInterface):
    """
    Simulated game for testing without a real game.
    Logs all commands and maintains fake state.
    """
    
    def __init__(self, name: str = "sim_game"):
        super().__init__(name)
        self._state = {
            "player": {"x": 0, "y": 0, "health": 100},
            "score": 0,
            "level": 1,
        }
    
    def connect(self) -> bool:
        self.state = GameState.CONNECTED
        self.info = GameInfo(name="Simulated Game", version="1.0")
        print(f"[SIM GAME] Connected (simulated)")
        return True
    
    def disconnect(self) -> bool:
        self.state = GameState.DISCONNECTED
        print(f"[SIM GAME] Disconnected")
        return True
    
    def send_command(self, command: str, **kwargs) -> bool:
        print(f"[SIM GAME] Command: {command} {kwargs}")
        
        # Simulate some game logic
        if command == "move":
            direction = kwargs.get("direction", "forward")
            if direction == "forward":
                self._state["player"]["y"] += 1
            elif direction == "backward":
                self._state["player"]["y"] -= 1
            elif direction == "left":
                self._state["player"]["x"] -= 1
            elif direction == "right":
                self._state["player"]["x"] += 1
        elif command == "attack":
            self._state["score"] += 10
        elif command == "say":
            print(f"[SIM GAME] Player says: {kwargs.get('message', '')}")
        
        return True
    
    def get_state(self) -> Dict[str, Any]:
        return self._state.copy()


# =============================================================================
# FORGEAI GAME CONTROLLER
# =============================================================================

class ForgeGameController:
    """
    Main controller for ForgeAI game integration.
    Manages game connections and AI interaction.
    """
    
    def __init__(self):
        self._games: Dict[str, GameInterface] = {}
        self._active_game: Optional[str] = None
    
    def register(self, name: str, interface: GameInterface):
        """Register a game interface."""
        self._games[name] = interface
        if self._active_game is None:
            self._active_game = name
    
    def connect(self, name: str = None) -> bool:
        """Connect to a game."""
        game = self._get_game(name)
        if game:
            return game.connect()
        return False
    
    def disconnect(self, name: str = None) -> bool:
        """Disconnect from a game."""
        game = self._get_game(name)
        if game:
            return game.disconnect()
        return False
    
    def send(self, command: str, name: str = None, **kwargs) -> bool:
        """Send command to a game."""
        game = self._get_game(name)
        if game:
            return game.send_command(command, **kwargs)
        return False
    
    def get_state(self, name: str = None) -> Dict[str, Any]:
        """Get game state."""
        game = self._get_game(name)
        if game:
            return game.get_state()
        return {}
    
    def _get_game(self, name: str = None) -> Optional[GameInterface]:
        """Get game interface by name or active game."""
        name = name or self._active_game
        return self._games.get(name)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_websocket_game(host: str = "localhost", port: int = 8765) -> ForgeGameController:
    """Create WebSocket game connection."""
    controller = ForgeGameController()
    game = WebSocketGameInterface(host=host, port=port)
    controller.register("ws", game)
    return controller


def create_http_game(url: str = "http://localhost:8080") -> ForgeGameController:
    """Create HTTP game connection."""
    controller = ForgeGameController()
    game = HTTPGameInterface(base_url=url)
    controller.register("http", game)
    return controller


def create_screen_game() -> ForgeGameController:
    """Create screen capture game connection."""
    controller = ForgeGameController()
    game = ScreenCaptureGameInterface()
    controller.register("screen", game)
    return controller


def create_simulated_game() -> ForgeGameController:
    """Create simulated game for testing."""
    controller = ForgeGameController()
    game = SimulatedGameInterface()
    controller.register("sim", game)
    return controller


# =============================================================================
# MAIN - Run this file directly to test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ForgeAI Game Example")
    print("=" * 60)
    
    # Use simulated game for testing
    print("\n[1] Creating simulated game...")
    controller = create_simulated_game()
    
    print("\n[2] Connecting...")
    controller.connect("sim")
    
    print("\n[3] Getting initial state...")
    state = controller.get_state()
    print(f"State: {state}")
    
    print("\n[4] Sending commands...")
    controller.send("move", direction="forward")
    controller.send("move", direction="right")
    controller.send("attack")
    controller.send("say", message="Hello from ForgeAI!")
    
    print("\n[5] Getting updated state...")
    state = controller.get_state()
    print(f"State: {state}")
    
    print("\n[6] Disconnecting...")
    controller.disconnect()
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nTo use with real games, try:")
    print("  - create_websocket_game() for WebSocket games")
    print("  - create_http_game() for HTTP API games")
    print("  - create_screen_game() for any game (vision-based)")
