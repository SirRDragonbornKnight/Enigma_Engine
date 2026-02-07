"""
Network Module - Multi-Device Task Offloading

Enables task distribution across multiple Enigma AI Engine devices:
- Raspberry Pi offloads heavy inference to PC
- Load balancing across multiple servers
- Automatic failover when servers go offline
"""

from .failover import (
    FailoverManager,
    ServerHealth,
)
from .inference_gateway import (
    InferenceGateway,
    get_inference_gateway,
)
from .load_balancer import (
    BalancingStrategy,
    LoadBalancer,
    ServerInfo,
)
from .remote_offloading import (
    OffloadDecision,
    RemoteOffloader,
    get_remote_offloader,
)
from .task_queue import (
    NetworkTask,
    NetworkTaskQueue,
    TaskPriority,
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
