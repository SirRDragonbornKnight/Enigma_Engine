# comms package - Multi-device communication for Enigma

from .network import EnigmaNode, Message, ModelExporter, create_server_node, create_client_node
from .discovery import DeviceDiscovery, discover_enigma_nodes
from .memory_sync import MemorySync, OfflineSync, add_sync_routes
from .api_server import create_app
from .remote_client import RemoteClient
from .mobile_api import create_mobile_api

# Aliases
create_api = create_app
RemoteEnigmaClient = RemoteClient

# Web server with WebSocket support (optional - requires flask-socketio)
try:
    from .web_server import WebServer, create_web_server
    _WEB_SERVER_AVAILABLE = True
except ImportError:
    _WEB_SERVER_AVAILABLE = False

__all__ = [
    # Network nodes
    "EnigmaNode",
    "Message", 
    "ModelExporter",
    "create_server_node",
    "create_client_node",
    
    # Discovery
    "DeviceDiscovery",
    "discover_enigma_nodes",
    
    # Memory sync
    "MemorySync",
    "OfflineSync",
    "add_sync_routes",
    
    # API
    "create_app",
    "create_api",
    "RemoteEnigmaClient",
    "create_mobile_api",
]

# Add web server if available
if _WEB_SERVER_AVAILABLE:
    __all__.extend(["WebServer", "create_web_server"])
