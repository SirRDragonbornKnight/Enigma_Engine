"""
Visual Agent Workspace for Enigma AI Engine

See agents thinking and collaborating visually.

Features:
- Agent state visualization
- Message flow tracking
- Thinking/processing indicators
- Collaboration diagrams
- Real-time updates

Usage:
    from enigma_engine.agents.visual_workspace import AgentWorkspace
    
    workspace = AgentWorkspace()
    
    # Add agents
    workspace.add_agent("researcher", ResearchAgent())
    workspace.add_agent("coder", CodingAgent())
    
    # Run task and visualize
    workspace.run_task("Build a web scraper")
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent processing status."""
    IDLE = "idle"
    THINKING = "thinking"
    PROCESSING = "processing"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"


class MessageType(Enum):
    """Message types between agents."""
    TASK = "task"
    RESPONSE = "response"
    QUERY = "query"
    DATA = "data"
    STATUS = "status"


@dataclass
class AgentMessage:
    """A message between agents."""
    id: str
    from_agent: str
    to_agent: str
    message_type: MessageType
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """Current state of an agent."""
    agent_id: str
    name: str
    status: AgentStatus = AgentStatus.IDLE
    current_task: str = ""
    thinking: str = ""
    progress: float = 0.0
    messages_sent: int = 0
    messages_received: int = 0
    last_active: float = field(default_factory=time.time)


@dataclass
class WorkspaceSnapshot:
    """Snapshot of workspace state."""
    timestamp: float
    agents: Dict[str, AgentState]
    messages: List[AgentMessage]
    active_task: str = ""


class AgentWorkspace:
    """Visual workspace for agent collaboration."""
    
    def __init__(self):
        """Initialize workspace."""
        self._agents: Dict[str, Any] = {}
        self._agent_states: Dict[str, AgentState] = {}
        self._messages: List[AgentMessage] = []
        self._snapshots: List[WorkspaceSnapshot] = []
        
        # Event listeners
        self._listeners: List[Callable[[str, Any], None]] = []
        
        # Current task
        self._current_task: Optional[str] = None
        self._task_results: Dict[str, Any] = {}
    
    def add_agent(self, agent_id: str, agent: Any, name: Optional[str] = None):
        """
        Add an agent to the workspace.
        
        Args:
            agent_id: Unique agent identifier
            agent: Agent instance
            name: Display name
        """
        self._agents[agent_id] = agent
        self._agent_states[agent_id] = AgentState(
            agent_id=agent_id,
            name=name or agent_id
        )
        
        self._emit("agent_added", {"agent_id": agent_id})
    
    def remove_agent(self, agent_id: str):
        """Remove an agent from workspace."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            del self._agent_states[agent_id]
            self._emit("agent_removed", {"agent_id": agent_id})
    
    def get_agent_state(self, agent_id: str) -> Optional[AgentState]:
        """Get agent state."""
        return self._agent_states.get(agent_id)
    
    def get_all_states(self) -> Dict[str, AgentState]:
        """Get all agent states."""
        return dict(self._agent_states)
    
    def update_status(self, agent_id: str, status: AgentStatus, thinking: str = ""):
        """
        Update agent status.
        
        Args:
            agent_id: Agent ID
            status: New status
            thinking: Current thinking/reasoning
        """
        if agent_id in self._agent_states:
            state = self._agent_states[agent_id]
            state.status = status
            state.thinking = thinking
            state.last_active = time.time()
            
            self._emit("status_changed", {
                "agent_id": agent_id,
                "status": status.value,
                "thinking": thinking
            })
    
    def update_progress(self, agent_id: str, progress: float, task: str = ""):
        """Update agent progress."""
        if agent_id in self._agent_states:
            state = self._agent_states[agent_id]
            state.progress = progress
            if task:
                state.current_task = task
            
            self._emit("progress_changed", {
                "agent_id": agent_id,
                "progress": progress
            })
    
    def send_message(
        self,
        from_agent: str,
        to_agent: str,
        content: str,
        message_type: MessageType = MessageType.DATA
    ) -> AgentMessage:
        """
        Send message between agents.
        
        Args:
            from_agent: Sender ID
            to_agent: Receiver ID
            content: Message content
            message_type: Message type
            
        Returns:
            Created message
        """
        message = AgentMessage(
            id=str(uuid.uuid4())[:8],
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            content=content
        )
        
        self._messages.append(message)
        
        # Update message counts
        if from_agent in self._agent_states:
            self._agent_states[from_agent].messages_sent += 1
        if to_agent in self._agent_states:
            self._agent_states[to_agent].messages_received += 1
        
        self._emit("message_sent", {
            "from": from_agent,
            "to": to_agent,
            "type": message_type.value,
            "content_preview": content[:100]
        })
        
        return message
    
    def run_task(
        self,
        task: str,
        callback: Optional[Callable[[str, Any], None]] = None
    ) -> Dict[str, Any]:
        """
        Run a task through the agent workspace.
        
        Args:
            task: Task description
            callback: Progress callback
            
        Returns:
            Task results
        """
        self._current_task = task
        self._task_results = {}
        
        if callback:
            self._listeners.append(callback)
        
        self._emit("task_started", {"task": task})
        
        # Simple sequential execution
        for agent_id, agent in self._agents.items():
            self.update_status(agent_id, AgentStatus.THINKING, "Analyzing task...")
            
            try:
                # Execute agent
                if hasattr(agent, 'execute'):
                    result = agent.execute(task)
                elif hasattr(agent, 'run'):
                    result = agent.run(task)
                elif hasattr(agent, '__call__'):
                    result = agent(task)
                else:
                    result = None
                
                self._task_results[agent_id] = result
                self.update_status(agent_id, AgentStatus.COMPLETED)
                
            except Exception as e:
                logger.error(f"Agent {agent_id} failed: {e}")
                self.update_status(agent_id, AgentStatus.ERROR, str(e))
                self._task_results[agent_id] = {"error": str(e)}
        
        self._emit("task_completed", {"results": self._task_results})
        
        # Take snapshot
        self._take_snapshot()
        
        # Remove callback
        if callback and callback in self._listeners:
            self._listeners.remove(callback)
        
        self._current_task = None
        return self._task_results
    
    def _take_snapshot(self):
        """Take a workspace snapshot."""
        snapshot = WorkspaceSnapshot(
            timestamp=time.time(),
            agents={k: AgentState(**v.__dict__) for k, v in self._agent_states.items()},
            messages=list(self._messages),
            active_task=self._current_task or ""
        )
        self._snapshots.append(snapshot)
    
    def get_snapshots(self) -> List[WorkspaceSnapshot]:
        """Get all snapshots."""
        return self._snapshots
    
    def add_listener(self, callback: Callable[[str, Any], None]):
        """Add event listener."""
        self._listeners.append(callback)
    
    def remove_listener(self, callback: Callable[[str, Any], None]):
        """Remove event listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)
    
    def _emit(self, event: str, data: Any):
        """Emit event to listeners."""
        for listener in self._listeners:
            try:
                listener(event, data)
            except Exception as e:
                logger.error(f"Listener error: {e}")
    
    def get_message_flow(self) -> List[Dict]:
        """Get message flow for visualization."""
        return [
            {
                "id": m.id,
                "from": m.from_agent,
                "to": m.to_agent,
                "type": m.message_type.value,
                "time": m.timestamp,
                "content": m.content[:50]
            }
            for m in self._messages[-100:]  # Last 100 messages
        ]
    
    def generate_visualization_data(self) -> Dict[str, Any]:
        """Generate data for visualization."""
        agents = []
        for agent_id, state in self._agent_states.items():
            agents.append({
                "id": agent_id,
                "name": state.name,
                "status": state.status.value,
                "progress": state.progress,
                "thinking": state.thinking,
                "messages_sent": state.messages_sent,
                "messages_received": state.messages_received
            })
        
        connections = []
        for m in self._messages:
            connections.append({
                "source": m.from_agent,
                "target": m.to_agent,
                "type": m.message_type.value
            })
        
        return {
            "agents": agents,
            "connections": connections,
            "current_task": self._current_task,
            "timestamp": time.time()
        }
    
    def render_text_view(self) -> str:
        """Render text-based visualization."""
        lines = ["Agent Workspace", "=" * 40, ""]
        
        # Agents
        lines.append("Agents:")
        for agent_id, state in self._agent_states.items():
            status_icon = {
                AgentStatus.IDLE: "[ ]",
                AgentStatus.THINKING: "[?]",
                AgentStatus.PROCESSING: "[*]",
                AgentStatus.WAITING: "[.]",
                AgentStatus.COMPLETED: "[v]",
                AgentStatus.ERROR: "[X]"
            }.get(state.status, "[ ]")
            
            lines.append(f"  {status_icon} {state.name}: {state.status.value}")
            if state.thinking:
                lines.append(f"      Thinking: {state.thinking[:60]}...")
        
        lines.append("")
        
        # Recent messages
        lines.append("Recent Messages:")
        for m in self._messages[-5:]:
            lines.append(f"  {m.from_agent} -> {m.to_agent}: {m.content[:40]}...")
        
        if self._current_task:
            lines.append("")
            lines.append(f"Current Task: {self._current_task}")
        
        return "\n".join(lines)


class WorkspaceUI:
    """Simple text-based workspace UI."""
    
    def __init__(self, workspace: AgentWorkspace):
        self.workspace = workspace
        workspace.add_listener(self._on_event)
    
    def _on_event(self, event: str, data: Any):
        """Handle workspace events."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if event == "status_changed":
            print(f"[{timestamp}] {data['agent_id']}: {data['status']}")
            if data.get('thinking'):
                print(f"           Thinking: {data['thinking'][:60]}")
        
        elif event == "message_sent":
            print(f"[{timestamp}] {data['from']} -> {data['to']}: {data['content_preview']}")
        
        elif event == "task_started":
            print(f"\n[{timestamp}] TASK STARTED: {data['task']}")
            print("-" * 40)
        
        elif event == "task_completed":
            print("-" * 40)
            print(f"[{timestamp}] TASK COMPLETED")
    
    def display(self):
        """Display current workspace state."""
        print(self.workspace.render_text_view())


# Convenience function
def create_workspace() -> AgentWorkspace:
    """Create a new agent workspace."""
    return AgentWorkspace()
