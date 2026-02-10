"""
Visual Workspace for Enigma AI Engine

Visual canvas for multi-agent collaboration.

Features:
- Node-based agent visualization
- Connection mapping
- Task flow diagrams
- Real-time updates
- Interactive canvas

Usage:
    from enigma_engine.gui.workspace import VisualWorkspace
    
    workspace = VisualWorkspace()
    
    # Add agent nodes
    workspace.add_agent_node("agent1", x=100, y=100)
    workspace.add_agent_node("agent2", x=300, y=100)
    
    # Connect agents
    workspace.connect("agent1", "agent2", "collaborates")
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of workspace nodes."""
    AGENT = "agent"
    TASK = "task"
    MEMORY = "memory"
    DATA = "data"
    TOOL = "tool"
    GROUP = "group"


class ConnectionType(Enum):
    """Types of connections."""
    DATA_FLOW = "data_flow"
    CONTROL = "control"
    COMMUNICATION = "communication"
    DEPENDENCY = "dependency"
    REFERENCE = "reference"


class NodeState(Enum):
    """States of a node."""
    IDLE = "idle"
    ACTIVE = "active"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETE = "complete"


@dataclass
class Position:
    """2D position."""
    x: float = 0.0
    y: float = 0.0


@dataclass
class NodeStyle:
    """Visual style for a node."""
    color: str = "#4A90D9"
    border_color: str = "#2E5A8A"
    text_color: str = "#FFFFFF"
    width: int = 120
    height: int = 60
    border_width: int = 2
    corner_radius: int = 8


@dataclass
class ConnectionStyle:
    """Visual style for a connection."""
    color: str = "#888888"
    width: int = 2
    arrow_size: int = 8
    dashed: bool = False


@dataclass
class WorkspaceNode:
    """A node in the workspace."""
    id: str
    node_type: NodeType
    label: str
    
    # Position
    position: Position = field(default_factory=Position)
    
    # Style
    style: NodeStyle = field(default_factory=NodeStyle)
    
    # State
    state: NodeState = NodeState.IDLE
    
    # Data
    data: Dict[str, Any] = field(default_factory=dict)
    
    # UI
    selected: bool = False
    visible: bool = True
    locked: bool = False


@dataclass
class Connection:
    """A connection between nodes."""
    id: str
    source_id: str
    target_id: str
    connection_type: ConnectionType
    
    label: str = ""
    style: ConnectionStyle = field(default_factory=ConnectionStyle)
    
    # State
    active: bool = False
    data_flowing: bool = False


@dataclass
class NodeGroup:
    """A group of nodes."""
    id: str
    label: str
    node_ids: Set[str] = field(default_factory=set)
    
    # Position (bounding box)
    position: Position = field(default_factory=Position)
    width: float = 200
    height: float = 150
    
    # Style
    color: str = "#E8E8E8"
    collapsed: bool = False


class Layout:
    """Layout algorithms for nodes."""
    
    @staticmethod
    def grid(
        nodes: List[WorkspaceNode],
        start_x: float = 50,
        start_y: float = 50,
        cols: int = 4,
        spacing_x: float = 150,
        spacing_y: float = 100
    ):
        """Arrange nodes in a grid."""
        for i, node in enumerate(nodes):
            row = i // cols
            col = i % cols
            node.position.x = start_x + col * spacing_x
            node.position.y = start_y + row * spacing_y
    
    @staticmethod
    def circular(
        nodes: List[WorkspaceNode],
        center_x: float = 300,
        center_y: float = 300,
        radius: float = 200
    ):
        """Arrange nodes in a circle."""
        n = len(nodes)
        if n == 0:
            return
        
        for i, node in enumerate(nodes):
            angle = 2 * math.pi * i / n
            node.position.x = center_x + radius * math.cos(angle)
            node.position.y = center_y + radius * math.sin(angle)
    
    @staticmethod
    def force_directed(
        nodes: List[WorkspaceNode],
        connections: List[Connection],
        iterations: int = 50,
        repulsion: float = 5000,
        attraction: float = 0.1
    ):
        """Force-directed layout."""
        if not nodes:
            return
        
        # Initialize velocities
        velocities = {n.id: Position() for n in nodes}
        
        for _ in range(iterations):
            # Calculate repulsive forces
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i+1:]:
                    dx = node1.position.x - node2.position.x
                    dy = node1.position.y - node2.position.y
                    dist = math.sqrt(dx*dx + dy*dy) + 0.01
                    
                    force = repulsion / (dist * dist)
                    fx = force * dx / dist
                    fy = force * dy / dist
                    
                    velocities[node1.id].x += fx
                    velocities[node1.id].y += fy
                    velocities[node2.id].x -= fx
                    velocities[node2.id].y -= fy
            
            # Calculate attractive forces for connections
            node_map = {n.id: n for n in nodes}
            for conn in connections:
                if conn.source_id in node_map and conn.target_id in node_map:
                    n1 = node_map[conn.source_id]
                    n2 = node_map[conn.target_id]
                    
                    dx = n2.position.x - n1.position.x
                    dy = n2.position.y - n1.position.y
                    dist = math.sqrt(dx*dx + dy*dy) + 0.01
                    
                    force = attraction * dist
                    fx = force * dx / dist
                    fy = force * dy / dist
                    
                    velocities[n1.id].x += fx
                    velocities[n1.id].y += fy
                    velocities[n2.id].x -= fx
                    velocities[n2.id].y -= fy
            
            # Apply velocities
            for node in nodes:
                if not node.locked:
                    node.position.x += velocities[node.id].x * 0.1
                    node.position.y += velocities[node.id].y * 0.1
                    
                    # Damping
                    velocities[node.id].x *= 0.5
                    velocities[node.id].y *= 0.5
    
    @staticmethod
    def hierarchical(
        nodes: List[WorkspaceNode],
        connections: List[Connection],
        start_x: float = 50,
        start_y: float = 50,
        level_height: float = 100,
        node_spacing: float = 150
    ):
        """Arrange nodes in hierarchical levels."""
        if not nodes:
            return
        
        # Build adjacency
        outgoing: Dict[str, Set[str]] = {n.id: set() for n in nodes}
        incoming: Dict[str, Set[str]] = {n.id: set() for n in nodes}
        
        for conn in connections:
            if conn.source_id in outgoing:
                outgoing[conn.source_id].add(conn.target_id)
            if conn.target_id in incoming:
                incoming[conn.target_id].add(conn.source_id)
        
        # Find roots (nodes with no incoming)
        roots = [n for n in nodes if not incoming[n.id]]
        if not roots:
            roots = nodes[:1]
        
        # Assign levels
        levels: Dict[str, int] = {}
        visited = set()
        
        def assign_level(node_id: str, level: int):
            if node_id in visited:
                return
            visited.add(node_id)
            levels[node_id] = level
            for target in outgoing.get(node_id, []):
                assign_level(target, level + 1)
        
        for root in roots:
            assign_level(root.id, 0)
        
        # Position nodes by level
        level_nodes: Dict[int, List[WorkspaceNode]] = {}
        node_map = {n.id: n for n in nodes}
        
        for node_id, level in levels.items():
            if level not in level_nodes:
                level_nodes[level] = []
            level_nodes[level].append(node_map[node_id])
        
        for level, level_list in level_nodes.items():
            for i, node in enumerate(level_list):
                node.position.x = start_x + i * node_spacing
                node.position.y = start_y + level * level_height


class VisualWorkspace:
    """Visual canvas for multi-agent collaboration."""
    
    def __init__(
        self,
        width: int = 800,
        height: int = 600
    ):
        """
        Initialize workspace.
        
        Args:
            width: Canvas width
            height: Canvas height
        """
        self._width = width
        self._height = height
        
        self._nodes: Dict[str, WorkspaceNode] = {}
        self._connections: Dict[str, Connection] = {}
        self._groups: Dict[str, NodeGroup] = {}
        
        # Callbacks
        self._on_node_click: Optional[Callable[[WorkspaceNode], None]] = None
        self._on_connection_click: Optional[Callable[[Connection], None]] = None
        self._on_update: Optional[Callable[[], None]] = None
        
        # Selection
        self._selected_nodes: Set[str] = set()
        self._selected_connections: Set[str] = set()
    
    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        label: str,
        x: float = 0,
        y: float = 0,
        data: Optional[Dict[str, Any]] = None,
        style: Optional[NodeStyle] = None
    ) -> WorkspaceNode:
        """
        Add a node to the workspace.
        
        Returns:
            The created node
        """
        node = WorkspaceNode(
            id=node_id,
            node_type=node_type,
            label=label,
            position=Position(x, y),
            data=data or {},
            style=style or self._default_style(node_type)
        )
        
        self._nodes[node_id] = node
        self._trigger_update()
        
        return node
    
    def add_agent_node(
        self,
        agent_id: str,
        label: Optional[str] = None,
        x: float = 0,
        y: float = 0,
        role: str = "agent"
    ) -> WorkspaceNode:
        """Add an agent node."""
        style = NodeStyle(
            color="#4A90D9",
            border_color="#2E5A8A"
        )
        
        return self.add_node(
            agent_id,
            NodeType.AGENT,
            label or agent_id,
            x, y,
            data={"role": role},
            style=style
        )
    
    def add_task_node(
        self,
        task_id: str,
        label: str,
        x: float = 0,
        y: float = 0,
        status: str = "pending"
    ) -> WorkspaceNode:
        """Add a task node."""
        style = NodeStyle(
            color="#E8A838",
            border_color="#B87828"
        )
        
        return self.add_node(
            task_id,
            NodeType.TASK,
            label,
            x, y,
            data={"status": status},
            style=style
        )
    
    def add_memory_node(
        self,
        memory_id: str,
        label: str,
        x: float = 0,
        y: float = 0
    ) -> WorkspaceNode:
        """Add a memory node."""
        style = NodeStyle(
            color="#68B368",
            border_color="#488848"
        )
        
        return self.add_node(
            memory_id,
            NodeType.MEMORY,
            label,
            x, y,
            style=style
        )
    
    def remove_node(self, node_id: str):
        """Remove a node and its connections."""
        if node_id in self._nodes:
            del self._nodes[node_id]
            
            # Remove connected connections
            to_remove = [
                cid for cid, conn in self._connections.items()
                if conn.source_id == node_id or conn.target_id == node_id
            ]
            for cid in to_remove:
                del self._connections[cid]
            
            self._trigger_update()
    
    def connect(
        self,
        source_id: str,
        target_id: str,
        label: str = "",
        connection_type: ConnectionType = ConnectionType.DATA_FLOW,
        style: Optional[ConnectionStyle] = None
    ) -> Optional[Connection]:
        """
        Connect two nodes.
        
        Returns:
            The connection or None if invalid
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            return None
        
        conn_id = f"{source_id}_to_{target_id}"
        
        conn = Connection(
            id=conn_id,
            source_id=source_id,
            target_id=target_id,
            connection_type=connection_type,
            label=label,
            style=style or ConnectionStyle()
        )
        
        self._connections[conn_id] = conn
        self._trigger_update()
        
        return conn
    
    def disconnect(self, source_id: str, target_id: str):
        """Remove connection between nodes."""
        conn_id = f"{source_id}_to_{target_id}"
        if conn_id in self._connections:
            del self._connections[conn_id]
            self._trigger_update()
    
    def create_group(
        self,
        group_id: str,
        label: str,
        node_ids: List[str]
    ) -> NodeGroup:
        """Create a group of nodes."""
        group = NodeGroup(
            id=group_id,
            label=label,
            node_ids=set(node_ids)
        )
        
        # Calculate bounding box
        self._update_group_bounds(group)
        
        self._groups[group_id] = group
        self._trigger_update()
        
        return group
    
    def add_to_group(self, group_id: str, node_id: str):
        """Add a node to a group."""
        if group_id in self._groups and node_id in self._nodes:
            self._groups[group_id].node_ids.add(node_id)
            self._update_group_bounds(self._groups[group_id])
            self._trigger_update()
    
    def _update_group_bounds(self, group: NodeGroup):
        """Update group bounding box based on nodes."""
        if not group.node_ids:
            return
        
        nodes = [self._nodes[nid] for nid in group.node_ids if nid in self._nodes]
        if not nodes:
            return
        
        min_x = min(n.position.x for n in nodes)
        min_y = min(n.position.y for n in nodes)
        max_x = max(n.position.x + n.style.width for n in nodes)
        max_y = max(n.position.y + n.style.height for n in nodes)
        
        padding = 20
        group.position.x = min_x - padding
        group.position.y = min_y - padding
        group.width = max_x - min_x + 2 * padding
        group.height = max_y - min_y + 2 * padding
    
    def move_node(self, node_id: str, x: float, y: float):
        """Move a node to new position."""
        if node_id in self._nodes:
            node = self._nodes[node_id]
            if not node.locked:
                node.position.x = x
                node.position.y = y
                self._trigger_update()
    
    def set_node_state(self, node_id: str, state: NodeState):
        """Set node state."""
        if node_id in self._nodes:
            self._nodes[node_id].state = state
            self._trigger_update()
    
    def set_connection_active(self, conn_id: str, active: bool):
        """Set connection active state (for animation)."""
        if conn_id in self._connections:
            self._connections[conn_id].active = active
            self._connections[conn_id].data_flowing = active
            self._trigger_update()
    
    def select_node(self, node_id: str, add: bool = False):
        """Select a node."""
        if not add:
            self._selected_nodes.clear()
        
        if node_id in self._nodes:
            self._selected_nodes.add(node_id)
            self._nodes[node_id].selected = True
            self._trigger_update()
    
    def deselect_all(self):
        """Clear selection."""
        for nid in self._selected_nodes:
            if nid in self._nodes:
                self._nodes[nid].selected = False
        self._selected_nodes.clear()
        self._selected_connections.clear()
        self._trigger_update()
    
    def auto_layout(
        self,
        layout_type: str = "force",
        **kwargs
    ):
        """Apply automatic layout."""
        nodes = list(self._nodes.values())
        connections = list(self._connections.values())
        
        if layout_type == "grid":
            Layout.grid(nodes, **kwargs)
        elif layout_type == "circular":
            Layout.circular(nodes, **kwargs)
        elif layout_type == "hierarchical":
            Layout.hierarchical(nodes, connections, **kwargs)
        else:
            Layout.force_directed(nodes, connections, **kwargs)
        
        self._trigger_update()
    
    def get_node(self, node_id: str) -> Optional[WorkspaceNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)
    
    def get_all_nodes(self) -> List[WorkspaceNode]:
        """Get all nodes."""
        return list(self._nodes.values())
    
    def get_connections(self, node_id: str) -> List[Connection]:
        """Get all connections for a node."""
        return [
            c for c in self._connections.values()
            if c.source_id == node_id or c.target_id == node_id
        ]
    
    def on_node_click(self, callback: Callable[[WorkspaceNode], None]):
        """Set callback for node clicks."""
        self._on_node_click = callback
    
    def on_update(self, callback: Callable[[], None]):
        """Set callback for workspace updates."""
        self._on_update = callback
    
    def _trigger_update(self):
        """Trigger update callback."""
        if self._on_update:
            self._on_update()
    
    def _default_style(self, node_type: NodeType) -> NodeStyle:
        """Get default style for node type."""
        styles = {
            NodeType.AGENT: NodeStyle(color="#4A90D9"),
            NodeType.TASK: NodeStyle(color="#E8A838"),
            NodeType.MEMORY: NodeStyle(color="#68B368"),
            NodeType.DATA: NodeStyle(color="#9B59B6"),
            NodeType.TOOL: NodeStyle(color="#E74C3C"),
            NodeType.GROUP: NodeStyle(color="#95A5A6")
        }
        return styles.get(node_type, NodeStyle())
    
    def export_state(self) -> Dict[str, Any]:
        """Export workspace state."""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "type": n.node_type.value,
                    "label": n.label,
                    "x": n.position.x,
                    "y": n.position.y,
                    "state": n.state.value,
                    "data": n.data
                }
                for n in self._nodes.values()
            ],
            "connections": [
                {
                    "id": c.id,
                    "source": c.source_id,
                    "target": c.target_id,
                    "type": c.connection_type.value,
                    "label": c.label
                }
                for c in self._connections.values()
            ],
            "groups": [
                {
                    "id": g.id,
                    "label": g.label,
                    "nodes": list(g.node_ids)
                }
                for g in self._groups.values()
            ]
        }
    
    def import_state(self, state: Dict[str, Any]):
        """Import workspace state."""
        self._nodes.clear()
        self._connections.clear()
        self._groups.clear()
        
        for node_data in state.get("nodes", []):
            self.add_node(
                node_data["id"],
                NodeType(node_data["type"]),
                node_data["label"],
                node_data.get("x", 0),
                node_data.get("y", 0),
                node_data.get("data", {})
            )
            if "state" in node_data:
                self.set_node_state(node_data["id"], NodeState(node_data["state"]))
        
        for conn_data in state.get("connections", []):
            self.connect(
                conn_data["source"],
                conn_data["target"],
                conn_data.get("label", ""),
                ConnectionType(conn_data.get("type", "data_flow"))
            )
        
        for group_data in state.get("groups", []):
            self.create_group(
                group_data["id"],
                group_data["label"],
                group_data.get("nodes", [])
            )


# Global instance
_workspace: Optional[VisualWorkspace] = None


def get_workspace() -> VisualWorkspace:
    """Get or create global workspace."""
    global _workspace
    if _workspace is None:
        _workspace = VisualWorkspace()
    return _workspace
