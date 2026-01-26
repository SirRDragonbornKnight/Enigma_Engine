# comms package - Multi-device communication for Forge

from .network import ForgeNode, Message, ModelExporter, create_server_node, create_client_node
from .discovery import DeviceDiscovery, discover_forge_ai_nodes
from .memory_sync import MemorySync, OfflineSync, add_sync_routes
from .multi_ai import AIConversation, AIParticipant, quick_ai_chat
from .protocol_manager import ProtocolManager, ProtocolConfig, get_protocol_manager
from .remote_client import RemoteClient
from .api_server import create_api_server

# Network optimizer for low-latency distributed AI
from .network_optimizer import (
    NetworkOptimizer, OptimizedRequest, RequestStats, ResponseCache,
    get_network_optimizer,
)

# Device sync for real-time state synchronization
from .device_sync import (
    DeviceSync, DeviceType, SyncPriority, SyncState, ConnectedDevice,
    get_device_sync,
)

# Distributed protocol (hardware-aware)
try:
    from .distributed import (
        DistributedNode, NodeRole, MessageType, ProtocolMessage, NodeInfo,
        create_server, create_client,
    )
    HAS_DISTRIBUTED = True
except ImportError:
    HAS_DISTRIBUTED = False

# Optional imports (may require Flask)
try:
    from .mobile_api import MobileAPI, create_mobile_api
    from .web_server import WebServer, create_web_server
    HAS_WEB = True
except ImportError:
    HAS_WEB = False

__all__ = [
    # Network nodes
    "ForgeNode",
    "Message", 
    "ModelExporter",
    "create_server_node",
    "create_client_node",
    
    # Distributed protocol (hardware-aware)
    "DistributedNode",
    "NodeRole",
    "MessageType",
    "ProtocolMessage",
    "NodeInfo",
    "create_server",
    "create_client",
    
    # Discovery
    "DeviceDiscovery",
    "discover_forge_ai_nodes",
    
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
    
    # Network optimizer
    "NetworkOptimizer",
    "OptimizedRequest",
    "RequestStats",
    "ResponseCache",
    "get_network_optimizer",
    
    # Device sync
    "DeviceSync",
    "DeviceType",
    "SyncPriority",
    "SyncState",
    "ConnectedDevice",
    "get_device_sync",
]

# Add Flask-dependent exports if available
if HAS_WEB:
    __all__.extend([
        "MobileAPI",
        "create_mobile_api",
        "WebServer", 
        "create_web_server",
    ])
