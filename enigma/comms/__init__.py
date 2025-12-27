# comms package - Multi-device communication for Enigma

from .network import EnigmaNode, Message, ModelExporter, create_server_node, create_client_node
from .discovery import DeviceDiscovery, discover_enigma_nodes
from .memory_sync import MemorySync, OfflineSync, add_sync_routes
from .multi_ai import AIConversation, AIParticipant, quick_ai_chat
from .protocol_manager import ProtocolManager, ProtocolConfig, get_protocol_manager

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
]
