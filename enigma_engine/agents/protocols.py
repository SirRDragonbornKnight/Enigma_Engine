"""
Agent Communication Protocols for Enigma AI Engine

Structured message passing between agents.

Features:
- Message schemas
- Protocol handlers
- Agent discovery
- Request/response patterns
- Event broadcasting

Usage:
    from enigma_engine.agents.protocols import AgentProtocol, MessageBus
    
    bus = MessageBus()
    
    # Register agent
    bus.register("researcher", researcher_agent)
    
    # Send message
    bus.send("researcher", "analyze", {"topic": "AI"})
    
    # Broadcast
    bus.broadcast("status_update", {"ready": True})
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Message types."""
    REQUEST = "request"  # Expecting response
    RESPONSE = "response"  # Reply to request
    NOTIFY = "notify"  # One-way notification
    BROADCAST = "broadcast"  # To all agents
    ERROR = "error"  # Error response
    ACK = "ack"  # Acknowledgment


class MessagePriority(Enum):
    """Message priorities."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class AgentMessage:
    """Message for inter-agent communication."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.NOTIFY
    sender: str = ""
    recipient: str = ""  # Empty for broadcast
    action: str = ""  # Method/action to invoke
    payload: Any = None
    reply_to: str = ""  # ID of message being replied to
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    correlation_id: str = ""  # For tracking related messages
    ttl: float = 60.0  # Time to live in seconds
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "action": self.action,
            "payload": self.payload,
            "reply_to": self.reply_to,
            "priority": self.priority.value,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "ttl": self.ttl
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentMessage':
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=MessageType(data.get("type", "notify")),
            sender=data.get("sender", ""),
            recipient=data.get("recipient", ""),
            action=data.get("action", ""),
            payload=data.get("payload"),
            reply_to=data.get("reply_to", ""),
            priority=MessagePriority(data.get("priority", 1)),
            timestamp=data.get("timestamp", time.time()),
            correlation_id=data.get("correlation_id", ""),
            ttl=data.get("ttl", 60.0)
        )
    
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl


@dataclass
class AgentCapability:
    """Capability that an agent provides."""
    name: str
    description: str = ""
    input_schema: Dict = field(default_factory=dict)
    output_schema: Dict = field(default_factory=dict)


@dataclass
class AgentInfo:
    """Information about a registered agent."""
    id: str
    name: str
    capabilities: List[AgentCapability] = field(default_factory=list)
    status: str = "active"  # active, busy, offline
    metadata: Dict = field(default_factory=dict)


class AgentProtocol(ABC):
    """Base protocol for agent communication."""
    
    @abstractmethod
    def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle incoming message."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[AgentCapability]:
        """Return list of capabilities."""
        pass
    
    def get_agent_info(self) -> AgentInfo:
        """Get agent info."""
        return AgentInfo(
            id=getattr(self, 'id', str(uuid.uuid4())),
            name=getattr(self, 'name', 'Unknown'),
            capabilities=self.get_capabilities()
        )


class MessageHandler:
    """Decorator-based message handler."""
    
    def __init__(self):
        self._handlers: Dict[str, Callable] = {}
    
    def on(self, action: str):
        """Decorator to register handler for action."""
        def decorator(func):
            self._handlers[action] = func
            return func
        return decorator
    
    def handle(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Route message to handler."""
        handler = self._handlers.get(message.action)
        if handler:
            try:
                result = handler(message.payload)
                return AgentMessage(
                    type=MessageType.RESPONSE,
                    sender=message.recipient,
                    recipient=message.sender,
                    action=message.action,
                    payload=result,
                    reply_to=message.id,
                    correlation_id=message.correlation_id
                )
            except Exception as e:
                return AgentMessage(
                    type=MessageType.ERROR,
                    sender=message.recipient,
                    recipient=message.sender,
                    action=message.action,
                    payload={"error": str(e)},
                    reply_to=message.id
                )
        return None


class MessageBus:
    """Central message bus for agent communication."""
    
    def __init__(self, async_mode: bool = False):
        """
        Initialize message bus.
        
        Args:
            async_mode: Use async message handling
        """
        self.async_mode = async_mode
        
        # Registered agents
        self._agents: Dict[str, AgentProtocol] = {}
        self._agent_info: Dict[str, AgentInfo] = {}
        
        # Message queues per agent
        self._queues: Dict[str, Queue] = {}
        
        # Subscriptions (action -> list of agent IDs)
        self._subscriptions: Dict[str, Set[str]] = {}
        
        # Pending requests (message ID -> callback)
        self._pending: Dict[str, Callable] = {}
        
        # Background processing
        self._running = False
        self._processor_thread: Optional[threading.Thread] = None
        
        logger.info("MessageBus initialized")
    
    def start(self):
        """Start message processing."""
        if self._running:
            return
        
        self._running = True
        self._processor_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._processor_thread.start()
    
    def stop(self):
        """Stop message processing."""
        self._running = False
        if self._processor_thread:
            self._processor_thread.join(timeout=2.0)
    
    def register(
        self,
        agent_id: str,
        agent: AgentProtocol,
        name: Optional[str] = None
    ):
        """
        Register an agent.
        
        Args:
            agent_id: Unique agent identifier
            agent: Agent implementing AgentProtocol
            name: Display name
        """
        self._agents[agent_id] = agent
        self._queues[agent_id] = Queue()
        
        # Store agent info
        info = agent.get_agent_info()
        info.id = agent_id
        if name:
            info.name = name
        self._agent_info[agent_id] = info
        
        logger.info(f"Registered agent: {agent_id}")
    
    def unregister(self, agent_id: str):
        """Unregister an agent."""
        self._agents.pop(agent_id, None)
        self._queues.pop(agent_id, None)
        self._agent_info.pop(agent_id, None)
        
        # Remove from subscriptions
        for subscribers in self._subscriptions.values():
            subscribers.discard(agent_id)
    
    def subscribe(self, agent_id: str, action: str):
        """Subscribe agent to action broadcasts."""
        if action not in self._subscriptions:
            self._subscriptions[action] = set()
        self._subscriptions[action].add(agent_id)
    
    def unsubscribe(self, agent_id: str, action: str):
        """Unsubscribe agent from action."""
        if action in self._subscriptions:
            self._subscriptions[action].discard(agent_id)
    
    def send(
        self,
        recipient: str,
        action: str,
        payload: Any = None,
        sender: str = "system",
        priority: MessagePriority = MessagePriority.NORMAL,
        callback: Optional[Callable[[AgentMessage], None]] = None,
        timeout: float = 30.0
    ) -> Optional[AgentMessage]:
        """
        Send message to specific agent.
        
        Args:
            recipient: Target agent ID
            action: Action to invoke
            payload: Message payload
            sender: Sender ID
            priority: Message priority
            callback: Async callback for response
            timeout: Timeout for sync response
            
        Returns:
            Response message if sync, None if async
        """
        message = AgentMessage(
            type=MessageType.REQUEST if callback or timeout else MessageType.NOTIFY,
            sender=sender,
            recipient=recipient,
            action=action,
            payload=payload,
            priority=priority
        )
        
        return self._deliver(message, callback, timeout)
    
    def broadcast(
        self,
        action: str,
        payload: Any = None,
        sender: str = "system"
    ):
        """
        Broadcast message to all subscribed agents.
        
        Args:
            action: Action name
            payload: Message payload
            sender: Sender ID
        """
        message = AgentMessage(
            type=MessageType.BROADCAST,
            sender=sender,
            action=action,
            payload=payload
        )
        
        # Get subscribers
        subscribers = self._subscriptions.get(action, set())
        
        # Also send to all agents if no subscribers
        targets = subscribers if subscribers else set(self._agents.keys())
        
        for agent_id in targets:
            if agent_id in self._queues:
                msg_copy = AgentMessage(**asdict(message))
                msg_copy.recipient = agent_id
                self._queues[agent_id].put(msg_copy)
    
    def request(
        self,
        recipient: str,
        action: str,
        payload: Any = None,
        sender: str = "system",
        timeout: float = 30.0
    ) -> Optional[Any]:
        """
        Send request and wait for response.
        
        Args:
            recipient: Target agent
            action: Action to invoke
            payload: Request payload
            sender: Sender ID
            timeout: Response timeout
            
        Returns:
            Response payload or None
        """
        response = self.send(
            recipient=recipient,
            action=action,
            payload=payload,
            sender=sender,
            timeout=timeout
        )
        
        if response and response.type == MessageType.RESPONSE:
            return response.payload
        return None
    
    def discover(self, capability: Optional[str] = None) -> List[AgentInfo]:
        """
        Discover available agents.
        
        Args:
            capability: Filter by capability name
            
        Returns:
            List of agent info
        """
        agents = list(self._agent_info.values())
        
        if capability:
            agents = [
                a for a in agents
                if any(c.name == capability for c in a.capabilities)
            ]
        
        return agents
    
    def _deliver(
        self,
        message: AgentMessage,
        callback: Optional[Callable] = None,
        timeout: float = 30.0
    ) -> Optional[AgentMessage]:
        """Deliver message to recipient."""
        recipient = message.recipient
        
        if recipient not in self._queues:
            logger.warning(f"Unknown recipient: {recipient}")
            return None
        
        # Queue message
        self._queues[recipient].put(message)
        
        # Async callback
        if callback:
            self._pending[message.id] = callback
            return None
        
        # Sync wait for response
        if message.type == MessageType.REQUEST:
            start = time.time()
            response_event = threading.Event()
            response_holder = [None]
            
            def capture_response(resp):
                response_holder[0] = resp
                response_event.set()
            
            self._pending[message.id] = capture_response
            
            # Wait
            if response_event.wait(timeout):
                return response_holder[0]
            else:
                self._pending.pop(message.id, None)
                logger.warning(f"Request timeout: {message.action}")
                return None
        
        return None
    
    def _process_loop(self):
        """Process messages in queues."""
        while self._running:
            for agent_id, queue in list(self._queues.items()):
                if queue.empty():
                    continue
                
                try:
                    message = queue.get_nowait()
                    
                    # Skip expired messages
                    if message.is_expired():
                        continue
                    
                    # Get agent
                    agent = self._agents.get(agent_id)
                    if not agent:
                        continue
                    
                    # Handle message
                    response = agent.handle_message(message)
                    
                    # Route response
                    if response:
                        # Check for pending callback
                        callback = self._pending.pop(message.id, None)
                        if callback:
                            callback(response)
                        elif response.recipient:
                            self._deliver(response)
                            
                except Exception as e:
                    logger.error(f"Message processing error: {e}")
            
            time.sleep(0.01)


class SimpleAgent(AgentProtocol):
    """Simple agent implementation."""
    
    def __init__(self, agent_id: str, name: str = ""):
        self.id = agent_id
        self.name = name or agent_id
        self._handler = MessageHandler()
    
    def on(self, action: str):
        """Register action handler."""
        return self._handler.on(action)
    
    def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        return self._handler.handle(message)
    
    def get_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(name=action)
            for action in self._handler._handlers.keys()
        ]


class TaskCoordinator:
    """Coordinate tasks across multiple agents."""
    
    def __init__(self, bus: MessageBus):
        self.bus = bus
        self._tasks: Dict[str, Dict] = {}
    
    def submit_task(
        self,
        task_id: str,
        steps: List[Dict],
        on_complete: Optional[Callable] = None
    ):
        """
        Submit multi-step task.
        
        Args:
            task_id: Unique task ID
            steps: List of steps with {agent, action, payload}
            on_complete: Callback when all steps done
        """
        self._tasks[task_id] = {
            "steps": steps,
            "current": 0,
            "results": [],
            "on_complete": on_complete
        }
        
        self._execute_step(task_id)
    
    def _execute_step(self, task_id: str):
        """Execute current step of task."""
        task = self._tasks.get(task_id)
        if not task:
            return
        
        current_idx = task["current"]
        
        if current_idx >= len(task["steps"]):
            # Task complete
            if task["on_complete"]:
                task["on_complete"](task["results"])
            del self._tasks[task_id]
            return
        
        step = task["steps"][current_idx]
        
        def on_response(response: AgentMessage):
            task["results"].append(response.payload if response else None)
            task["current"] += 1
            self._execute_step(task_id)
        
        self.bus.send(
            recipient=step["agent"],
            action=step["action"],
            payload=step.get("payload"),
            callback=on_response
        )


# Global message bus
_message_bus: Optional[MessageBus] = None


def get_message_bus() -> MessageBus:
    """Get or create global MessageBus."""
    global _message_bus
    if _message_bus is None:
        _message_bus = MessageBus()
        _message_bus.start()
    return _message_bus
