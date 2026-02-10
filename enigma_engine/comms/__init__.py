# comms package - Multi-device communication for Forge

# Discovery can be imported without torch dependency
from .discovery import DeviceDiscovery, discover_enigma_engine_nodes

# Remote training can be imported without torch dependency
try:
    from .remote_training import (
        RemoteTrainer,
        TrainingJob,
        TrainingServer,
        get_remote_trainer,
        start_training_server,
        stop_training_server,
        submit_remote_training,
    )
    HAS_REMOTE_TRAINING = True
except ImportError:
    HAS_REMOTE_TRAINING = False
    RemoteTrainer = None
    TrainingServer = None
    TrainingJob = None

# Model sync for keeping models in sync across devices
try:
    from .model_sync import (
        ModelSyncClient,
        ModelSyncServer,
        ModelVersion,
        SyncStatus,
        get_sync_client,
        get_sync_server,
        start_sync_server,
        sync_from_server,
    )
    HAS_MODEL_SYNC = True
except ImportError:
    HAS_MODEL_SYNC = False
    ModelSyncClient = None
    ModelSyncServer = None
    ModelVersion = None
    SyncStatus = None

# Distributed inference across multiple devices  
try:
    from .distributed_inference import (
        DistributedCoordinator,
        InferenceTask,
        InferenceWorker,
        WorkerInfo,
        WorkerStatus,
        distributed_generate,
        get_coordinator,
        start_inference_worker,
        stop_inference_worker,
    )
    HAS_DISTRIBUTED_INFERENCE = True
except ImportError:
    HAS_DISTRIBUTED_INFERENCE = False
    DistributedCoordinator = None
    InferenceWorker = None
    WorkerInfo = None
    InferenceTask = None
    WorkerStatus = None

# Other imports that may require torch - use lazy loading
try:
    from .api_server import create_api_server
    from .memory_sync import MemorySync, OfflineSync, add_sync_routes
    from .multi_ai import AIConversation, AIParticipant, quick_ai_chat
    from .network import (
        ForgeNode,
        Message,
        ModelExporter,
        create_client_node,
        create_server_node,
    )
    from .protocol_manager import ProtocolConfig, ProtocolManager, get_protocol_manager
    from .remote_client import RemoteClient
    HAS_CORE = True
except ImportError as e:
    HAS_CORE = False
    # These will be None if torch is not available
    ForgeNode = None
    Message = None
    ModelExporter = None
    create_server_node = None
    create_client_node = None
    MemorySync = None
    OfflineSync = None
    add_sync_routes = None
    AIConversation = None
    AIParticipant = None
    quick_ai_chat = None
    ProtocolManager = None
    ProtocolConfig = None
    get_protocol_manager = None
    RemoteClient = None
    create_api_server = None

# Network optimizer for low-latency distributed AI
try:
    from .network_optimizer import (
        NetworkOptimizer,
        OptimizedRequest,
        RequestStats,
        ResponseCache,
        get_network_optimizer,
    )
    HAS_NETWORK_OPTIMIZER = True
except ImportError:
    HAS_NETWORK_OPTIMIZER = False
    NetworkOptimizer = None
    OptimizedRequest = None
    RequestStats = None
    ResponseCache = None
    get_network_optimizer = None

# Device sync for real-time state synchronization
try:
    from .device_sync import (
        ConnectedDevice,
        DeviceSync,
        DeviceType,
        SyncPriority,
        SyncState,
        get_device_sync,
    )
    HAS_DEVICE_SYNC = True
except ImportError:
    HAS_DEVICE_SYNC = False
    DeviceSync = None
    DeviceType = None
    SyncPriority = None
    SyncState = None
    ConnectedDevice = None
    get_device_sync = None

# Distributed protocol (hardware-aware)
try:
    from .distributed import (
        DistributedNode,
        MessageType,
        NodeInfo,
        NodeRole,
        ProtocolMessage,
        create_client,
        create_server,
    )
    HAS_DISTRIBUTED = True
except ImportError:
    HAS_DISTRIBUTED = False

# AI Collaboration Protocol (AI-to-AI communication)
try:
    from .ai_collaboration import (
        AICapability,
        AICollaborationProtocol,
        RoutingPreference,
        TaskRequest,
        TaskStatus,
        get_collaboration_protocol,
        reset_protocol,
    )
    HAS_AI_COLLABORATION = True
except ImportError:
    HAS_AI_COLLABORATION = False

# Optional imports (may require Flask)
try:
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
    
    # AI Collaboration Protocol
    "AICollaborationProtocol",
    "AICapability",
    "TaskRequest",
    "TaskStatus",
    "RoutingPreference",
    "get_collaboration_protocol",
    "reset_protocol",
    
    # Discovery
    "DeviceDiscovery",
    "discover_enigma_engine_nodes",
    
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
    
    # Remote training
    "RemoteTrainer",
    "TrainingServer",
    "TrainingJob",
    "get_remote_trainer",
    "start_training_server",
    "stop_training_server",
    "submit_remote_training",
    
    # Model sync
    "ModelSyncClient",
    "ModelSyncServer",
    "ModelVersion",
    "SyncStatus",
    "get_sync_client",
    "get_sync_server",
    "start_sync_server",
    "sync_from_server",
    
    # Distributed inference
    "DistributedCoordinator",
    "InferenceWorker",
    "InferenceTask",
    "WorkerInfo",
    "WorkerStatus",
    "get_coordinator",
    "start_inference_worker",
    "stop_inference_worker",
    "distributed_generate",
]

# Add Flask-dependent exports if available
if HAS_WEB:
    __all__.extend([
        "MobileAPI",
        "create_mobile_api",
        "WebServer", 
        "create_web_server",
    ])
