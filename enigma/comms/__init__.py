# comms package - Multi-device communication for Enigma

from .network import EnigmaNode, Message, ModelExporter, create_server_node, create_client_node
from .discovery import DeviceDiscovery, discover_enigma_nodes
from .memory_sync import MemorySync, OfflineSync, add_sync_routes
from .multi_ai import AIConversation, AIParticipant, quick_ai_chat
from .protocol_manager import ProtocolManager, ProtocolConfig, get_protocol_manager
from .remote_client import RemoteClient
from .api_server import create_api_server

# Optional imports (may require Flask)
try:
    from .mobile_api import MobileAPI, create_mobile_api
    from .web_server import WebServer, create_web_server
    HAS_WEB = True
except ImportError:
    HAS_WEB = False

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
    
    # Multi-AI
    "AIConversation",
    "AIParticipant",
    "quick_ai_chat",
    
    # Protocol management
    "ProtocolManager",
    "ProtocolConfig",
    "get_protocol_manager",
    
    # API clients
    "RemoteClient",
    "create_api_server",
]

# Add Flask-dependent exports if available
if HAS_WEB:
    __all__.extend([
        "MobileAPI",
        "create_mobile_api",
        "WebServer", 
        "create_web_server",
    ])
