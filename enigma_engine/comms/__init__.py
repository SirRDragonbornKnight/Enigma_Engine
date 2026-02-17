"""comms package - TRIMMED (distributed features removed)"""

# Core API (kept)
try:
    from .api_server import create_api_server
except ImportError:
    create_api_server = None

try:
    from .remote_client import RemoteClient
except ImportError:
    RemoteClient = None

# Stub classes for deleted features
class DeviceDiscovery: pass
def discover_enigma_engine_nodes(*a, **k): return []

class RemoteTrainer: pass
class TrainingServer: pass
class TrainingJob: pass
def get_remote_trainer(): return None
def start_training_server(*a, **k): pass
def stop_training_server(*a, **k): pass
def submit_remote_training(*a, **k): return None

class ModelSyncClient: pass
class ModelSyncServer: pass
class ModelVersion: pass
class SyncStatus: pass
def get_sync_client(): return None
def get_sync_server(): return None
def start_sync_server(*a, **k): pass
def sync_from_server(*a, **k): pass

class DistributedCoordinator: pass
class InferenceTask: pass
class InferenceWorker: pass
class WorkerInfo: pass
class WorkerStatus: pass
def distributed_generate(*a, **k): return ""
def get_coordinator(): return None
def start_inference_worker(*a, **k): pass
def stop_inference_worker(*a, **k): pass

class MemorySync: pass
class OfflineSync: pass
def add_sync_routes(*a, **k): pass

class AIConversation: pass
class AIParticipant: pass
def quick_ai_chat(*a, **k): return ""

class ForgeNode: pass
class Message: pass
class ModelExporter: pass
def create_server_node(*a, **k): return None
def create_client_node(*a, **k): return None

class ProtocolConfig: pass
class ProtocolManager: pass
def get_protocol_manager(): return None

class NetworkOptimizer: pass
class OptimizedRequest: pass
class RequestStats: pass
class ResponseCache: pass
def get_network_optimizer(): return None

class DeviceSync: pass
class DeviceType: pass
class SyncPriority: pass
class SyncState: pass
class ConnectedDevice: pass
def get_device_sync(): return None

class DistributedNode: pass
class MessageType: pass
class NodeInfo: pass
class NodeRole: pass
class ProtocolMessage: pass
def create_server(*a, **k): return None
def create_client(*a, **k): return None

class AICollaborationProtocol: pass
class AICapability: pass
class RoutingPreference: pass
class TaskRequest: pass
class TaskStatus: pass
def get_collaboration_protocol(): return None
def reset_protocol(*a, **k): pass

class MobileAPI: pass
def create_mobile_api(*a, **k): return None
class WebServer: pass
def create_web_server(*a, **k): return None

HAS_CORE = create_api_server is not None
HAS_REMOTE_TRAINING = False
HAS_MODEL_SYNC = False
HAS_DISTRIBUTED_INFERENCE = False
HAS_NETWORK_OPTIMIZER = False
HAS_DEVICE_SYNC = False
HAS_DISTRIBUTED = False
HAS_AI_COLLABORATION = False
HAS_WEB = False

__all__ = [
    "ForgeNode", "Message", "ModelExporter", "create_server_node", "create_client_node",
    "DistributedNode", "NodeRole", "MessageType", "ProtocolMessage", "NodeInfo", "create_server", "create_client",
    "AICollaborationProtocol", "AICapability", "TaskRequest", "TaskStatus", "RoutingPreference",
    "get_collaboration_protocol", "reset_protocol", "DeviceDiscovery", "discover_enigma_engine_nodes",
    "MemorySync", "OfflineSync", "add_sync_routes", "AIConversation", "AIParticipant", "quick_ai_chat",
    "ProtocolManager", "ProtocolConfig", "get_protocol_manager", "RemoteClient", "create_api_server",
    "NetworkOptimizer", "OptimizedRequest", "RequestStats", "ResponseCache", "get_network_optimizer",
    "DeviceSync", "DeviceType", "SyncPriority", "SyncState", "ConnectedDevice", "get_device_sync",
    "RemoteTrainer", "TrainingServer", "TrainingJob", "get_remote_trainer", "start_training_server",
    "stop_training_server", "submit_remote_training", "ModelSyncClient", "ModelSyncServer", "ModelVersion",
    "SyncStatus", "get_sync_client", "get_sync_server", "start_sync_server", "sync_from_server",
    "DistributedCoordinator", "InferenceWorker", "InferenceTask", "WorkerInfo", "WorkerStatus",
    "get_coordinator", "start_inference_worker", "stop_inference_worker", "distributed_generate",
    "MobileAPI", "create_mobile_api", "WebServer", "create_web_server",
]
