"""
Network Module - Multi-Device Task Offloading

Enables task distribution across multiple ForgeAI devices:
- Raspberry Pi offloads heavy inference to PC
- Load balancing across multiple servers
- Automatic failover when servers go offline
"""

from .remote_offloading import (
    RemoteOffloader,
    OffloadDecision,
    get_remote_offloader,
)
from .load_balancer import (
    LoadBalancer,
    ServerInfo,
    BalancingStrategy,
)
from .task_queue import (
    NetworkTaskQueue,
    NetworkTask,
    TaskPriority,
)
from .failover import (
    FailoverManager,
    ServerHealth,
)
from .inference_gateway import (
    InferenceGateway,
    get_inference_gateway,
)

__all__ = [
    # Remote offloading
    "RemoteOffloader",
    "OffloadDecision",
    "get_remote_offloader",
    # Load balancing
    "LoadBalancer",
    "ServerInfo",
    "BalancingStrategy",
    # Task queue
    "NetworkTaskQueue",
    "NetworkTask",
    "TaskPriority",
    # Failover
    "FailoverManager",
    "ServerHealth",
    # Gateway
    "InferenceGateway",
    "get_inference_gateway",
]
