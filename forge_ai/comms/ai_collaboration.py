"""
================================================================================
🤝 AI COLLABORATION PROTOCOL - AI-TO-AI COMMUNICATION & TASK SHARING
================================================================================

Protocol for AI instances to communicate, negotiate tasks, and collaborate
on complex operations across multiple devices.

📍 FILE: forge_ai/comms/ai_collaboration.py
🏷️ TYPE: AI-to-AI Communication Protocol
🎯 MAIN CLASSES: AICapability, AICollaborationProtocol, TaskRequest

┌─────────────────────────────────────────────────────────────────────────────┐
│  COLLABORATION ARCHITECTURE:                                                │
│                                                                             │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │  Pi (pi_node)   │     │  PC (pc_node)   │     │ Cloud (cloud)   │       │
│  │  ├─ pi_5 model  │◄───►│  ├─ large model │◄───►│  ├─ xl model    │       │
│  │  ├─ Basic tools │     │  ├─ Image gen   │     │  ├─ Video gen   │       │
│  │  └─ Fast/local  │     │  └─ GPU power   │     │  └─ Max quality │       │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘       │
│           │                      │                       │                  │
│           └──────────────────────┴───────────────────────┘                  │
│                          COLLABORATION MESH                                  │
│                                                                             │
│  User: "Generate a high-quality video of a sunset"                         │
│         │                                                                   │
│         ▼                                                                   │
│  Pi receives request → Can't handle video locally                          │
│         │                                                                   │
│         ▼                                                                   │
│  Negotiate with peers → PC can do basic, Cloud can do best                 │
│         │                                                                   │
│         ▼                                                                   │
│  Route to best peer → Cloud generates video                                │
│         │                                                                   │
│         ▼                                                                   │
│  Return result to user                                                     │
└─────────────────────────────────────────────────────────────────────────────┘

🎯 ROUTING PREFERENCES:
    • local_first  - Try local, then peers (privacy-focused)
    • fastest      - Route to lowest-latency peer
    • quality_first - Route to most capable peer
    • distributed  - Split work across all available peers

🔗 CONNECTED FILES:
    → USES:      forge_ai/comms/network.py (ForgeNode for connectivity)
    → USES:      forge_ai/core/tool_router.py (tool execution)
    ← USED BY:   forge_ai/core/tool_router.py (smart routing)
    ← USED BY:   forge_ai/gui/tabs/network_tab.py (GUI controls)

📖 USAGE:
    from forge_ai.comms.ai_collaboration import AICollaborationProtocol
    
    # Initialize protocol
    protocol = AICollaborationProtocol()
    protocol.connect_to_network(node)
    
    # Announce capabilities
    protocol.announce_capabilities()
    
    # Request task handling
    best_peer = protocol.request_task_handling("video", {"prompt": "sunset"})
    if best_peer:
        result = protocol.delegate_task(best_peer, "video", params)

📖 SEE ALSO:
    • forge_ai/comms/network.py       - Network connectivity
    • forge_ai/core/tool_router.py    - Tool routing
    • docs/multi_device_guide.md      - Setup guide
"""

import json
import time
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# 📊 DATA STRUCTURES
# =============================================================================

class RoutingPreference(Enum):
    """Routing strategy for task distribution."""
    LOCAL_FIRST = "local_first"    # Try local, then peers
    FASTEST = "fastest"            # Lowest latency peer
    QUALITY_FIRST = "quality_first"  # Most capable peer
    DISTRIBUTED = "distributed"    # Split across peers


class TaskStatus(Enum):
    """Status of a collaborative task."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class AICapability:
    """
    Capabilities of an AI instance for task negotiation.
    
    📖 WHAT THIS REPRESENTS:
    When AIs negotiate who should handle a task, they share their
    capabilities so the best AI can be chosen.
    
    📐 CAPABILITY FACTORS:
    - model_size: Larger models generally produce better results
    - available_tools: What tasks this AI can perform
    - current_load: How busy the AI is (0.0 = idle, 1.0 = maxed)
    - estimated_latency_ms: Network + processing time estimate
    - hardware_type: Affects performance characteristics
    """
    # Identity
    node_name: str = "unknown"
    node_url: str = ""
    
    # Model info
    model_size: str = "small"
    model_params: int = 0  # Approximate parameter count
    
    # Tools
    available_tools: List[str] = field(default_factory=list)
    
    # Performance
    current_load: float = 0.0  # 0.0 to 1.0
    estimated_latency_ms: int = 100
    
    # Hardware
    hardware_type: str = "cpu"  # "gpu", "cpu", "pi", "cloud"
    ram_gb: float = 4.0
    vram_gb: float = 0.0
    is_raspberry_pi: bool = False
    
    # Timestamp
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for network transmission."""
        return {
            "node_name": self.node_name,
            "node_url": self.node_url,
            "model_size": self.model_size,
            "model_params": self.model_params,
            "available_tools": self.available_tools,
            "current_load": self.current_load,
            "estimated_latency_ms": self.estimated_latency_ms,
            "hardware_type": self.hardware_type,
            "ram_gb": self.ram_gb,
            "vram_gb": self.vram_gb,
            "is_raspberry_pi": self.is_raspberry_pi,
            "last_updated": self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AICapability':
        """Create from dictionary."""
        return cls(
            node_name=data.get("node_name", "unknown"),
            node_url=data.get("node_url", ""),
            model_size=data.get("model_size", "small"),
            model_params=data.get("model_params", 0),
            available_tools=data.get("available_tools", []),
            current_load=data.get("current_load", 0.0),
            estimated_latency_ms=data.get("estimated_latency_ms", 100),
            hardware_type=data.get("hardware_type", "cpu"),
            ram_gb=data.get("ram_gb", 4.0),
            vram_gb=data.get("vram_gb", 0.0),
            is_raspberry_pi=data.get("is_raspberry_pi", False),
            last_updated=data.get("last_updated", datetime.now().isoformat()),
        )
    
    def can_handle(self, tool_name: str) -> bool:
        """Check if this AI can handle a specific tool."""
        return tool_name in self.available_tools
    
    def get_score(self, tool_name: str, preference: RoutingPreference) -> float:
        """
        Calculate routing score for task assignment.
        
        Higher score = better candidate for the task.
        """
        if not self.can_handle(tool_name):
            return -1.0
        
        score = 0.0
        
        if preference == RoutingPreference.FASTEST:
            # Prioritize low latency and low load
            score = 1000 / (self.estimated_latency_ms + 1)
            score *= (1.0 - self.current_load)
            
        elif preference == RoutingPreference.QUALITY_FIRST:
            # Prioritize model size and hardware
            size_scores = {
                "pi_zero": 1, "pi_4": 2, "pi_5": 3, "nano": 1, "micro": 2,
                "tiny": 3, "mini": 4, "small": 5, "medium": 7, "base": 8,
                "large": 10, "xl": 12, "xxl": 15, "huge": 18, "giant": 20
            }
            score = size_scores.get(self.model_size, 5)
            if self.hardware_type == "gpu":
                score *= 1.5
            elif self.hardware_type == "cloud":
                score *= 1.3
            score *= (1.0 - self.current_load * 0.5)  # Less penalty for load
            
        elif preference == RoutingPreference.LOCAL_FIRST:
            # Prioritize local execution (this is scored differently)
            score = 10 if self.estimated_latency_ms < 50 else 5
            
        else:  # DISTRIBUTED
            # Even distribution based on available capacity
            score = (1.0 - self.current_load) * 10
        
        return score


@dataclass
class TaskRequest:
    """
    A request for collaborative task execution.
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    tool_name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    requester: str = ""  # Node name of requester
    assigned_to: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    timeout_seconds: int = 300  # 5 minutes default
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tool_name": self.tool_name,
            "params": self.params,
            "requester": self.requester,
            "assigned_to": self.assigned_to,
            "status": self.status.value,
            "result": self.result,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "timeout_seconds": self.timeout_seconds,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskRequest':
        return cls(
            task_id=data.get("task_id", str(uuid.uuid4())[:8]),
            tool_name=data.get("tool_name", ""),
            params=data.get("params", {}),
            requester=data.get("requester", ""),
            assigned_to=data.get("assigned_to"),
            status=TaskStatus(data.get("status", "pending")),
            result=data.get("result"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            completed_at=data.get("completed_at"),
            timeout_seconds=data.get("timeout_seconds", 300),
        )


# =============================================================================
# 🤝 COLLABORATION PROTOCOL
# =============================================================================

class AICollaborationProtocol:
    """
    Protocol for AI-to-AI task negotiation and collaboration.
    
    📖 WHAT THIS DOES:
    Manages communication between multiple ForgeAI instances to:
    - Share capabilities (what each AI can do)
    - Negotiate task handling (who should do what)
    - Execute tasks remotely (delegate to better-suited peers)
    - Report results back to requesters
    
    📐 PROTOCOL FLOW:
    
        ┌─────────────────────────────────────────────────────────────────────┐
        │  1. DISCOVERY: AIs announce capabilities on network                 │
        │                                                                     │
        │     Pi ──announce──► PC                                            │
        │     Pi ◄──announce── PC                                            │
        │                                                                     │
        │  2. NEGOTIATION: When task arrives, ask who can handle it          │
        │                                                                     │
        │     Pi ──"who can do video?"──► PC                                 │
        │     Pi ◄──"I can, score=8"────── PC                                │
        │                                                                     │
        │  3. DELEGATION: Send task to best peer                             │
        │                                                                     │
        │     Pi ──task_request──► PC                                        │
        │     Pi ◄──task_result─── PC                                        │
        └─────────────────────────────────────────────────────────────────────┘
    
    ⚡ ROUTING STRATEGIES:
    - local_first: Privacy, try local before network
    - fastest: Latency, pick lowest-latency peer
    - quality_first: Quality, pick most capable peer
    - distributed: Load balancing, spread across peers
    """
    
    # Timeouts
    CAPABILITY_TIMEOUT_SECONDS = 300  # 5 minutes
    TASK_TIMEOUT_SECONDS = 300
    NEGOTIATION_TIMEOUT_SECONDS = 10
    
    def __init__(self, node_name: str = "local"):
        """
        Initialize the collaboration protocol.
        
        Args:
            node_name: Name of this AI node
        """
        self.node_name = node_name
        self.node_url = ""
        
        # Network connection
        self._network_node = None
        
        # Peer capabilities (peer_name -> AICapability)
        self._peer_capabilities: Dict[str, AICapability] = {}
        self._capabilities_lock = threading.RLock()
        
        # Local capabilities
        self._local_capability: Optional[AICapability] = None
        
        # Active tasks
        self._active_tasks: Dict[str, TaskRequest] = {}
        self._tasks_lock = threading.RLock()
        
        # Routing preference
        self._routing_preference = RoutingPreference.LOCAL_FIRST
        
        # Callbacks
        self._on_task_received: Optional[Callable[[TaskRequest], Dict]] = None
        self._on_capability_update: Optional[Callable[[str, AICapability], None]] = None
        
        logger.info(f"AICollaborationProtocol initialized for node: {node_name}")
    
    # =========================================================================
    # 🔌 NETWORK CONNECTION
    # =========================================================================
    
    def connect_to_network(self, node: Any) -> bool:
        """
        Connect to the network using a ForgeNode.
        
        Args:
            node: ForgeNode instance for network communication
        
        Returns:
            True if connected successfully
        """
        try:
            self._network_node = node
            self.node_name = node.name
            self.node_url = f"http://{node._get_local_ip()}:{node.port}"
            
            # Detect and announce capabilities
            self._detect_local_capabilities()
            
            logger.info(f"Connected to network as: {self.node_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to network: {e}")
            return False
    
    def _detect_local_capabilities(self):
        """Detect this node's capabilities."""
        try:
            from ..core.hardware_detection import detect_hardware
            profile = detect_hardware()
            
            # Get available tools
            tools = self._get_available_tools()
            
            self._local_capability = AICapability(
                node_name=self.node_name,
                node_url=self.node_url,
                model_size=profile.recommended_model_size,
                available_tools=tools,
                current_load=0.0,
                hardware_type="pi" if profile.is_raspberry_pi else (
                    "gpu" if profile.has_cuda else "cpu"
                ),
                ram_gb=profile.total_ram_gb,
                vram_gb=profile.gpu_vram_gb or 0.0,
                is_raspberry_pi=profile.is_raspberry_pi,
            )
            
            logger.info(f"Local capabilities: {self._local_capability.model_size}, "
                       f"tools={len(tools)}, hardware={self._local_capability.hardware_type}")
            
        except ImportError:
            # Fallback if hardware detection unavailable
            self._local_capability = AICapability(
                node_name=self.node_name,
                node_url=self.node_url,
                model_size="small",
                available_tools=["chat"],
                hardware_type="cpu",
            )
    
    def _get_available_tools(self) -> List[str]:
        """Get list of locally available tools."""
        try:
            from ..core.tool_router import get_router
            router = get_router()
            return list(router.routing_rules.keys())
        except ImportError:
            return ["chat"]
    
    # =========================================================================
    # 📢 CAPABILITY ANNOUNCEMENT
    # =========================================================================
    
    def announce_capabilities(self) -> bool:
        """
        Broadcast this AI's capabilities to all connected peers.
        
        Returns:
            True if announcement was sent successfully
        """
        if self._network_node is None:
            logger.warning("Cannot announce: not connected to network")
            return False
        
        if self._local_capability is None:
            self._detect_local_capabilities()
        
        # Update timestamp
        self._local_capability.last_updated = datetime.now().isoformat()
        
        # Broadcast to all peers
        for peer_name, peer_info in self._network_node.peers.items():
            try:
                self._send_capability_to_peer(peer_name, peer_info["url"])
            except Exception as e:
                logger.warning(f"Failed to announce to {peer_name}: {e}")
        
        logger.info(f"Announced capabilities to {len(self._network_node.peers)} peers")
        return True
    
    def _send_capability_to_peer(self, peer_name: str, peer_url: str):
        """Send capability announcement to a specific peer."""
        import urllib.request
        
        data = json.dumps({
            "type": "capability_announce",
            "capability": self._local_capability.to_dict(),
        }).encode()
        
        req = urllib.request.Request(
            f"{peer_url}/ai_protocol",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        
        try:
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            logger.debug(f"Capability send to {peer_name} failed: {e}")
    
    def receive_capability_update(self, peer_name: str, capability: AICapability):
        """
        Handle capability update from a peer.
        
        Args:
            peer_name: Name of the peer
            capability: Updated capability information
        """
        with self._capabilities_lock:
            self._peer_capabilities[peer_name] = capability
            logger.debug(f"Updated capability for peer: {peer_name}")
        
        if self._on_capability_update:
            self._on_capability_update(peer_name, capability)
    
    def get_peer_capabilities(self, peer_name: str) -> Optional[AICapability]:
        """Get cached capabilities for a peer."""
        with self._capabilities_lock:
            cap = self._peer_capabilities.get(peer_name)
            
            # Check if stale
            if cap:
                age = (datetime.now() - datetime.fromisoformat(cap.last_updated)).total_seconds()
                if age > self.CAPABILITY_TIMEOUT_SECONDS:
                    logger.debug(f"Capability for {peer_name} is stale ({age:.0f}s old)")
                    return None
            
            return cap
    
    def list_all_capabilities(self) -> Dict[str, AICapability]:
        """Get all known peer capabilities."""
        with self._capabilities_lock:
            return self._peer_capabilities.copy()
    
    # =========================================================================
    # 🤝 TASK NEGOTIATION
    # =========================================================================
    
    def request_task_handling(self, tool_name: str, params: Dict[str, Any]) -> Optional[str]:
        """
        Ask peers who can best handle a task.
        
        📖 WHAT THIS DOES:
        Checks all connected peers to find who can handle the task,
        scores them based on routing preference, and returns the best peer.
        
        Args:
            tool_name: Name of the tool/task (e.g., "image", "video")
            params: Task parameters
        
        Returns:
            Name of best peer to handle task, or None if no peer available
        """
        if not self._peer_capabilities:
            logger.debug("No peer capabilities available")
            return None
        
        # Score all peers
        scores: List[Tuple[str, float]] = []
        
        with self._capabilities_lock:
            for peer_name, capability in self._peer_capabilities.items():
                score = capability.get_score(tool_name, self._routing_preference)
                if score > 0:
                    scores.append((peer_name, score))
        
        if not scores:
            logger.debug(f"No peers can handle tool: {tool_name}")
            return None
        
        # Sort by score (highest first)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        best_peer, best_score = scores[0]
        logger.info(f"Best peer for {tool_name}: {best_peer} (score={best_score:.2f})")
        
        return best_peer
    
    def negotiate_task(self, tool_name: str, params: Dict[str, Any]) -> str:
        """
        Negotiate with all peers to find the best handler for a task.
        
        This does active negotiation (queries peers in real-time) rather
        than using cached capabilities.
        
        Returns:
            Best peer name, or "local" if local is best
        """
        # Check local capability first
        local_score = 0
        if self._local_capability and self._local_capability.can_handle(tool_name):
            local_score = self._local_capability.get_score(tool_name, self._routing_preference)
        
        # Query peers
        best_peer = "local"
        best_score = local_score
        
        if self._network_node:
            for peer_name in self._network_node.peers:
                try:
                    score = self._query_peer_for_task(peer_name, tool_name)
                    if score > best_score:
                        best_score = score
                        best_peer = peer_name
                except Exception as e:
                    logger.debug(f"Failed to query {peer_name}: {e}")
        
        logger.info(f"Negotiation result for {tool_name}: {best_peer} (score={best_score:.2f})")
        return best_peer
    
    def _query_peer_for_task(self, peer_name: str, tool_name: str) -> float:
        """Query a peer's score for handling a task."""
        import urllib.request
        
        if peer_name not in self._network_node.peers:
            return -1
        
        peer_url = self._network_node.peers[peer_name]["url"]
        
        data = json.dumps({
            "type": "task_query",
            "tool_name": tool_name,
            "preference": self._routing_preference.value,
        }).encode()
        
        req = urllib.request.Request(
            f"{peer_url}/ai_protocol",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=self.NEGOTIATION_TIMEOUT_SECONDS) as response:
                result = json.loads(response.read().decode())
                return result.get("score", -1)
        except Exception:
            return -1
    
    # =========================================================================
    # 🚀 TASK EXECUTION
    # =========================================================================
    
    def accept_task(self, task_id: str) -> bool:
        """
        Accept a task from another AI.
        
        Args:
            task_id: ID of the task to accept
        
        Returns:
            True if task was accepted
        """
        with self._tasks_lock:
            if task_id not in self._active_tasks:
                logger.warning(f"Unknown task: {task_id}")
                return False
            
            task = self._active_tasks[task_id]
            task.status = TaskStatus.ACCEPTED
            task.assigned_to = self.node_name
            
            logger.info(f"Accepted task: {task_id}")
            return True
    
    def delegate_task(self, peer_name: str, tool_name: str, params: Dict[str, Any],
                      timeout: int = None) -> Dict[str, Any]:
        """
        Delegate a task to a peer for execution.
        
        Args:
            peer_name: Peer to delegate to
            tool_name: Tool to execute
            params: Task parameters
            timeout: Timeout in seconds (default: TASK_TIMEOUT_SECONDS)
        
        Returns:
            Result dictionary with success, result, error keys
        """
        import urllib.request
        
        if self._network_node is None or peer_name not in self._network_node.peers:
            return {"success": False, "error": f"Unknown peer: {peer_name}"}
        
        peer_url = self._network_node.peers[peer_name]["url"]
        timeout = timeout or self.TASK_TIMEOUT_SECONDS
        
        # Create task request
        task = TaskRequest(
            tool_name=tool_name,
            params=params,
            requester=self.node_name,
            timeout_seconds=timeout,
        )
        
        # Track task
        with self._tasks_lock:
            self._active_tasks[task.task_id] = task
        
        # Send to peer
        data = json.dumps({
            "type": "task_execute",
            "task": task.to_dict(),
        }).encode()
        
        req = urllib.request.Request(
            f"{peer_url}/ai_protocol",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                result = json.loads(response.read().decode())
                
                # Update task status
                with self._tasks_lock:
                    task.status = TaskStatus.COMPLETED if result.get("success") else TaskStatus.FAILED
                    task.result = result
                    task.completed_at = datetime.now().isoformat()
                
                return result
                
        except Exception as e:
            logger.error(f"Task delegation failed: {e}")
            with self._tasks_lock:
                task.status = TaskStatus.FAILED
                task.result = {"success": False, "error": str(e)}
            return {"success": False, "error": str(e)}
    
    def report_task_complete(self, task_id: str, result: Dict[str, Any]):
        """
        Report task completion back to the requester.
        
        Args:
            task_id: ID of completed task
            result: Task result
        """
        with self._tasks_lock:
            if task_id not in self._active_tasks:
                logger.warning(f"Unknown task to report: {task_id}")
                return
            
            task = self._active_tasks[task_id]
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.now().isoformat()
        
        # Notify requester if remote
        if task.requester != self.node_name and self._network_node:
            self._send_task_result(task)
        
        logger.info(f"Task {task_id} completed")
    
    def _send_task_result(self, task: TaskRequest):
        """Send task result back to requester."""
        import urllib.request
        
        if task.requester not in self._network_node.peers:
            return
        
        peer_url = self._network_node.peers[task.requester]["url"]
        
        data = json.dumps({
            "type": "task_result",
            "task": task.to_dict(),
        }).encode()
        
        req = urllib.request.Request(
            f"{peer_url}/ai_protocol",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        
        try:
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            logger.warning(f"Failed to send result to {task.requester}: {e}")
    
    # =========================================================================
    # 🔀 TASK SPLITTING
    # =========================================================================
    
    def split_task(self, complex_task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a complex task into subtasks for distributed execution.
        
        📖 WHAT THIS DOES:
        Takes a complex task and breaks it into smaller pieces that
        can be distributed across multiple AIs.
        
        📐 EXAMPLE:
        Complex: "Generate 10 images of cats"
        Split: [
            {"tool": "image", "prompt": "cat", "index": 0},
            {"tool": "image", "prompt": "cat", "index": 1},
            ...
        ]
        
        Args:
            complex_task: Task dictionary with tool_name and params
        
        Returns:
            List of subtask dictionaries
        """
        tool_name = complex_task.get("tool_name", "")
        params = complex_task.get("params", {})
        
        subtasks = []
        
        # Batch processing for image generation
        if tool_name == "image" and params.get("batch_size", 1) > 1:
            batch_size = params.get("batch_size", 1)
            for i in range(batch_size):
                subtask_params = params.copy()
                subtask_params["batch_size"] = 1
                subtask_params["batch_index"] = i
                subtasks.append({
                    "tool_name": tool_name,
                    "params": subtask_params,
                    "parent_task": complex_task.get("task_id"),
                })
        
        # Text processing - split by chunks
        elif tool_name == "chat" and len(params.get("prompt", "")) > 2000:
            prompt = params.get("prompt", "")
            chunk_size = 1000
            chunks = [prompt[i:i+chunk_size] for i in range(0, len(prompt), chunk_size)]
            
            for i, chunk in enumerate(chunks):
                subtask_params = params.copy()
                subtask_params["prompt"] = chunk
                subtask_params["chunk_index"] = i
                subtask_params["total_chunks"] = len(chunks)
                subtasks.append({
                    "tool_name": tool_name,
                    "params": subtask_params,
                    "parent_task": complex_task.get("task_id"),
                })
        
        # Default: single task
        if not subtasks:
            subtasks = [complex_task]
        
        logger.info(f"Split task into {len(subtasks)} subtasks")
        return subtasks
    
    def request_collaboration(self, task: str, sub_tasks: List[Dict]) -> Dict[str, Any]:
        """
        Request collaboration from peers for parallel subtask execution.
        
        Args:
            task: Parent task description
            sub_tasks: List of subtasks to distribute
        
        Returns:
            Combined results from all peers
        """
        results = []
        peer_assignments: Dict[str, List[Dict]] = {}
        
        # Assign subtasks to peers
        available_peers = list(self._peer_capabilities.keys())
        if not available_peers:
            # Execute all locally
            logger.info("No peers available, executing all subtasks locally")
            for subtask in sub_tasks:
                result = self._execute_locally(subtask)
                results.append(result)
        else:
            # Distribute across peers
            for i, subtask in enumerate(sub_tasks):
                peer = available_peers[i % len(available_peers)]
                if peer not in peer_assignments:
                    peer_assignments[peer] = []
                peer_assignments[peer].append(subtask)
            
            # Execute on each peer
            for peer, tasks in peer_assignments.items():
                for subtask in tasks:
                    result = self.delegate_task(
                        peer,
                        subtask.get("tool_name", "chat"),
                        subtask.get("params", {})
                    )
                    results.append(result)
        
        # Combine results
        success_count = sum(1 for r in results if r.get("success"))
        
        return {
            "success": success_count > 0,
            "total_subtasks": len(sub_tasks),
            "successful_subtasks": success_count,
            "results": results,
        }
    
    def _execute_locally(self, task: Dict) -> Dict[str, Any]:
        """Execute a task locally."""
        try:
            from ..core.tool_router import get_router
            router = get_router()
            return router.execute_tool(
                task.get("tool_name", "chat"),
                task.get("params", {})
            )
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # =========================================================================
    # ⚙️ CONFIGURATION
    # =========================================================================
    
    def set_routing_preference(self, preference: str):
        """
        Set the routing preference for task distribution.
        
        Args:
            preference: One of "local_first", "fastest", "quality_first", "distributed"
        """
        try:
            self._routing_preference = RoutingPreference(preference)
            logger.info(f"Routing preference set to: {preference}")
        except ValueError:
            logger.warning(f"Invalid routing preference: {preference}")
    
    def get_routing_preference(self) -> str:
        """Get current routing preference."""
        return self._routing_preference.value
    
    def set_task_callback(self, callback: Callable[[TaskRequest], Dict]):
        """Set callback for when a task is received from another AI."""
        self._on_task_received = callback
    
    def set_capability_callback(self, callback: Callable[[str, AICapability], None]):
        """Set callback for when a peer's capability is updated."""
        self._on_capability_update = callback
    
    def update_local_load(self, load: float):
        """
        Update local load indicator (0.0 to 1.0).
        
        Call this periodically to let peers know how busy this AI is.
        """
        if self._local_capability:
            self._local_capability.current_load = max(0.0, min(1.0, load))
            self._local_capability.last_updated = datetime.now().isoformat()


# =============================================================================
# 🏭 FACTORY FUNCTIONS
# =============================================================================

_protocol_instance: Optional[AICollaborationProtocol] = None


def get_collaboration_protocol(node_name: str = "local") -> AICollaborationProtocol:
    """
    Get or create the global collaboration protocol instance.
    
    Args:
        node_name: Name for this AI node
    
    Returns:
        AICollaborationProtocol instance
    """
    global _protocol_instance
    if _protocol_instance is None:
        _protocol_instance = AICollaborationProtocol(node_name)
    return _protocol_instance


def reset_protocol():
    """Reset the global protocol instance."""
    global _protocol_instance
    _protocol_instance = None
