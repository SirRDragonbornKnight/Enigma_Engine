"""
Workflow Builder - Visual workflow creation and execution.

Features:
- Visual workflow designer
- Node-based connections
- Workflow templates
- Execution engine
- Import/export workflows

Part of the ForgeAI automation suite.
"""

import json
import time
import threading
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Set, Union
from pathlib import Path
from enum import Enum
from datetime import datetime
import logging
import copy

logger = logging.getLogger(__name__)


# =============================================================================
# WORKFLOW NODE TYPES
# =============================================================================

class NodeType(Enum):
    """Types of workflow nodes."""
    TRIGGER = "trigger"      # Starts workflow
    ACTION = "action"        # Performs action
    CONDITION = "condition"  # Branch logic
    TRANSFORM = "transform"  # Data transformation
    DELAY = "delay"          # Wait/pause
    LOOP = "loop"            # Iteration
    SUBFLOW = "subflow"      # Nested workflow
    OUTPUT = "output"        # Final output


class TriggerType(Enum):
    """Types of triggers."""
    MANUAL = "manual"
    SCHEDULE = "schedule"
    MESSAGE = "message"
    KEYWORD = "keyword"
    FILE = "file"
    API = "api"


class ActionType(Enum):
    """Types of actions."""
    SEND_MESSAGE = "send_message"
    GENERATE_IMAGE = "generate_image"
    GENERATE_CODE = "generate_code"
    GENERATE_AUDIO = "generate_audio"
    RUN_COMMAND = "run_command"
    CALL_API = "call_api"
    SAVE_FILE = "save_file"
    LOAD_FILE = "load_file"
    SET_VARIABLE = "set_variable"
    SEND_NOTIFICATION = "send_notification"


# =============================================================================
# WORKFLOW NODE
# =============================================================================

@dataclass
class NodePort:
    """Input/output port on a node."""
    id: str
    name: str
    port_type: str  # "input" or "output"
    data_type: str = "any"  # any, string, number, boolean, object, array
    required: bool = True
    default: Any = None


@dataclass
class NodePosition:
    """Position of node in canvas."""
    x: float = 0.0
    y: float = 0.0


@dataclass
class WorkflowNode:
    """A node in the workflow."""
    id: str
    name: str
    node_type: NodeType
    config: Dict[str, Any] = field(default_factory=dict)
    inputs: List[NodePort] = field(default_factory=list)
    outputs: List[NodePort] = field(default_factory=list)
    position: NodePosition = field(default_factory=NodePosition)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create_trigger(
        cls,
        trigger_type: TriggerType,
        name: str = "Trigger",
        config: Optional[Dict] = None
    ) -> "WorkflowNode":
        """Create a trigger node."""
        return cls(
            id=f"trigger_{uuid.uuid4().hex[:8]}",
            name=name,
            node_type=NodeType.TRIGGER,
            config={"trigger_type": trigger_type.value, **(config or {})},
            outputs=[NodePort(id="out", name="Output", port_type="output")]
        )
    
    @classmethod
    def create_action(
        cls,
        action_type: ActionType,
        name: str = "Action",
        config: Optional[Dict] = None
    ) -> "WorkflowNode":
        """Create an action node."""
        return cls(
            id=f"action_{uuid.uuid4().hex[:8]}",
            name=name,
            node_type=NodeType.ACTION,
            config={"action_type": action_type.value, **(config or {})},
            inputs=[NodePort(id="in", name="Input", port_type="input")],
            outputs=[NodePort(id="out", name="Output", port_type="output")]
        )
    
    @classmethod
    def create_condition(
        cls,
        condition: str,
        name: str = "Condition"
    ) -> "WorkflowNode":
        """Create a condition node."""
        return cls(
            id=f"cond_{uuid.uuid4().hex[:8]}",
            name=name,
            node_type=NodeType.CONDITION,
            config={"condition": condition},
            inputs=[NodePort(id="in", name="Input", port_type="input")],
            outputs=[
                NodePort(id="true", name="True", port_type="output"),
                NodePort(id="false", name="False", port_type="output")
            ]
        )
    
    @classmethod
    def create_transform(
        cls,
        transform: str,
        name: str = "Transform"
    ) -> "WorkflowNode":
        """Create a transform node."""
        return cls(
            id=f"trans_{uuid.uuid4().hex[:8]}",
            name=name,
            node_type=NodeType.TRANSFORM,
            config={"transform": transform},
            inputs=[NodePort(id="in", name="Input", port_type="input")],
            outputs=[NodePort(id="out", name="Output", port_type="output")]
        )
    
    @classmethod  
    def create_delay(cls, seconds: float, name: str = "Delay") -> "WorkflowNode":
        """Create a delay node."""
        return cls(
            id=f"delay_{uuid.uuid4().hex[:8]}",
            name=name,
            node_type=NodeType.DELAY,
            config={"seconds": seconds},
            inputs=[NodePort(id="in", name="Input", port_type="input")],
            outputs=[NodePort(id="out", name="Output", port_type="output")]
        )
    
    @classmethod
    def create_output(cls, name: str = "Output") -> "WorkflowNode":
        """Create an output node."""
        return cls(
            id=f"output_{uuid.uuid4().hex[:8]}",
            name=name,
            node_type=NodeType.OUTPUT,
            inputs=[NodePort(id="in", name="Input", port_type="input")]
        )


# =============================================================================
# WORKFLOW CONNECTION
# =============================================================================

@dataclass
class Connection:
    """Connection between two nodes."""
    id: str
    from_node: str
    from_port: str
    to_node: str
    to_port: str
    
    @classmethod
    def create(
        cls,
        from_node: str,
        from_port: str,
        to_node: str,
        to_port: str
    ) -> "Connection":
        """Create a connection."""
        return cls(
            id=f"conn_{uuid.uuid4().hex[:8]}",
            from_node=from_node,
            from_port=from_port,
            to_node=to_node,
            to_port=to_port
        )


# =============================================================================
# WORKFLOW
# =============================================================================

@dataclass
class Workflow:
    """A complete workflow definition."""
    id: str
    name: str
    description: str = ""
    nodes: Dict[str, WorkflowNode] = field(default_factory=dict)
    connections: List[Connection] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    version: int = 1
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)
    
    @classmethod
    def create(cls, name: str, description: str = "") -> "Workflow":
        """Create a new workflow."""
        return cls(
            id=f"workflow_{uuid.uuid4().hex[:8]}",
            name=name,
            description=description
        )
    
    def add_node(self, node: WorkflowNode) -> str:
        """Add a node to the workflow."""
        self.nodes[node.id] = node
        self.updated_at = datetime.now().isoformat()
        return node.id
    
    def remove_node(self, node_id: str):
        """Remove a node and its connections."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            # Remove related connections
            self.connections = [
                c for c in self.connections
                if c.from_node != node_id and c.to_node != node_id
            ]
            self.updated_at = datetime.now().isoformat()
    
    def connect(
        self,
        from_node: str,
        from_port: str,
        to_node: str,
        to_port: str
    ) -> Optional[str]:
        """Create a connection between nodes."""
        # Validate
        if from_node not in self.nodes or to_node not in self.nodes:
            return None
        
        conn = Connection.create(from_node, from_port, to_node, to_port)
        self.connections.append(conn)
        self.updated_at = datetime.now().isoformat()
        return conn.id
    
    def disconnect(self, connection_id: str):
        """Remove a connection."""
        self.connections = [c for c in self.connections if c.id != connection_id]
        self.updated_at = datetime.now().isoformat()
    
    def get_start_nodes(self) -> List[WorkflowNode]:
        """Get trigger/start nodes."""
        return [n for n in self.nodes.values() if n.node_type == NodeType.TRIGGER]
    
    def get_next_nodes(self, node_id: str) -> List[tuple]:
        """Get next connected nodes from a node."""
        result = []
        for conn in self.connections:
            if conn.from_node == node_id:
                if conn.to_node in self.nodes:
                    result.append((self.nodes[conn.to_node], conn.to_port))
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tags": self.tags,
            "variables": self.variables,
            "nodes": {
                nid: {
                    "id": n.id,
                    "name": n.name,
                    "node_type": n.node_type.value,
                    "config": n.config,
                    "position": {"x": n.position.x, "y": n.position.y},
                    "inputs": [{"id": p.id, "name": p.name, "port_type": p.port_type,
                               "data_type": p.data_type, "required": p.required}
                              for p in n.inputs],
                    "outputs": [{"id": p.id, "name": p.name, "port_type": p.port_type,
                                "data_type": p.data_type}
                               for p in n.outputs],
                    "metadata": n.metadata
                }
                for nid, n in self.nodes.items()
            },
            "connections": [
                {"id": c.id, "from_node": c.from_node, "from_port": c.from_port,
                 "to_node": c.to_node, "to_port": c.to_port}
                for c in self.connections
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Workflow":
        """Create workflow from dictionary."""
        workflow = cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", 1),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            tags=data.get("tags", []),
            variables=data.get("variables", {})
        )
        
        # Restore nodes
        for nid, ndata in data.get("nodes", {}).items():
            node = WorkflowNode(
                id=ndata["id"],
                name=ndata["name"],
                node_type=NodeType(ndata["node_type"]),
                config=ndata.get("config", {}),
                position=NodePosition(
                    x=ndata.get("position", {}).get("x", 0),
                    y=ndata.get("position", {}).get("y", 0)
                ),
                inputs=[
                    NodePort(
                        id=p["id"],
                        name=p["name"],
                        port_type=p["port_type"],
                        data_type=p.get("data_type", "any"),
                        required=p.get("required", True)
                    )
                    for p in ndata.get("inputs", [])
                ],
                outputs=[
                    NodePort(
                        id=p["id"],
                        name=p["name"],
                        port_type=p["port_type"],
                        data_type=p.get("data_type", "any")
                    )
                    for p in ndata.get("outputs", [])
                ],
                metadata=ndata.get("metadata", {})
            )
            workflow.nodes[node.id] = node
        
        # Restore connections
        for cdata in data.get("connections", []):
            conn = Connection(
                id=cdata["id"],
                from_node=cdata["from_node"],
                from_port=cdata["from_port"],
                to_node=cdata["to_node"],
                to_port=cdata["to_port"]
            )
            workflow.connections.append(conn)
        
        return workflow


# =============================================================================
# WORKFLOW TEMPLATES
# =============================================================================

class WorkflowTemplates:
    """Pre-built workflow templates."""
    
    @staticmethod
    def simple_chat_response() -> Workflow:
        """Simple message trigger to response."""
        wf = Workflow.create("Simple Chat Response", "Respond to a message")
        
        trigger = WorkflowNode.create_trigger(TriggerType.MESSAGE, "On Message")
        action = WorkflowNode.create_action(ActionType.SEND_MESSAGE, "Reply")
        output = WorkflowNode.create_output("Done")
        
        trigger.position = NodePosition(100, 100)
        action.position = NodePosition(300, 100)
        output.position = NodePosition(500, 100)
        
        wf.add_node(trigger)
        wf.add_node(action)
        wf.add_node(output)
        
        wf.connect(trigger.id, "out", action.id, "in")
        wf.connect(action.id, "out", output.id, "in")
        
        return wf
    
    @staticmethod
    def conditional_response() -> Workflow:
        """Conditional branching based on keyword."""
        wf = Workflow.create("Conditional Response", "Branch based on input")
        
        trigger = WorkflowNode.create_trigger(TriggerType.MESSAGE, "On Message")
        condition = WorkflowNode.create_condition("'image' in input.lower()", "Check Keyword")
        image_action = WorkflowNode.create_action(ActionType.GENERATE_IMAGE, "Generate Image")
        text_action = WorkflowNode.create_action(ActionType.SEND_MESSAGE, "Text Reply")
        output = WorkflowNode.create_output("Done")
        
        trigger.position = NodePosition(100, 150)
        condition.position = NodePosition(300, 150)
        image_action.position = NodePosition(500, 50)
        text_action.position = NodePosition(500, 250)
        output.position = NodePosition(700, 150)
        
        wf.add_node(trigger)
        wf.add_node(condition)
        wf.add_node(image_action)
        wf.add_node(text_action)
        wf.add_node(output)
        
        wf.connect(trigger.id, "out", condition.id, "in")
        wf.connect(condition.id, "true", image_action.id, "in")
        wf.connect(condition.id, "false", text_action.id, "in")
        wf.connect(image_action.id, "out", output.id, "in")
        wf.connect(text_action.id, "out", output.id, "in")
        
        return wf
    
    @staticmethod
    def scheduled_task() -> Workflow:
        """Scheduled task workflow."""
        wf = Workflow.create("Scheduled Task", "Run task on schedule")
        
        trigger = WorkflowNode.create_trigger(
            TriggerType.SCHEDULE, 
            "Every Hour",
            {"cron": "0 * * * *"}
        )
        action = WorkflowNode.create_action(
            ActionType.RUN_COMMAND,
            "Run Script",
            {"command": "python cleanup.py"}
        )
        notify = WorkflowNode.create_action(ActionType.SEND_NOTIFICATION, "Notify")
        output = WorkflowNode.create_output("Done")
        
        trigger.position = NodePosition(100, 100)
        action.position = NodePosition(300, 100)
        notify.position = NodePosition(500, 100)
        output.position = NodePosition(700, 100)
        
        wf.add_node(trigger)
        wf.add_node(action)
        wf.add_node(notify)
        wf.add_node(output)
        
        wf.connect(trigger.id, "out", action.id, "in")
        wf.connect(action.id, "out", notify.id, "in")
        wf.connect(notify.id, "out", output.id, "in")
        
        return wf
    
    @staticmethod
    def data_pipeline() -> Workflow:
        """Data processing pipeline."""
        wf = Workflow.create("Data Pipeline", "Process and transform data")
        
        trigger = WorkflowNode.create_trigger(TriggerType.FILE, "On File")
        load = WorkflowNode.create_action(ActionType.LOAD_FILE, "Load Data")
        transform = WorkflowNode.create_transform("json.loads(data)", "Parse JSON")
        process = WorkflowNode.create_transform("process_data(data)", "Process")
        save = WorkflowNode.create_action(ActionType.SAVE_FILE, "Save Result")
        output = WorkflowNode.create_output("Done")
        
        trigger.position = NodePosition(100, 100)
        load.position = NodePosition(250, 100)
        transform.position = NodePosition(400, 100)
        process.position = NodePosition(550, 100)
        save.position = NodePosition(700, 100)
        output.position = NodePosition(850, 100)
        
        for node in [trigger, load, transform, process, save, output]:
            wf.add_node(node)
        
        wf.connect(trigger.id, "out", load.id, "in")
        wf.connect(load.id, "out", transform.id, "in")
        wf.connect(transform.id, "out", process.id, "in")
        wf.connect(process.id, "out", save.id, "in")
        wf.connect(save.id, "out", output.id, "in")
        
        return wf
    
    @classmethod
    def get_all(cls) -> Dict[str, Workflow]:
        """Get all available templates."""
        return {
            "simple_chat": cls.simple_chat_response(),
            "conditional": cls.conditional_response(),
            "scheduled": cls.scheduled_task(),
            "data_pipeline": cls.data_pipeline()
        }


# =============================================================================
# WORKFLOW EXECUTION
# =============================================================================

class ExecutionStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionContext:
    """Context for workflow execution."""
    workflow_id: str
    run_id: str
    variables: Dict[str, Any] = field(default_factory=dict)
    input_data: Any = None
    output_data: Any = None
    current_node: Optional[str] = None
    node_outputs: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    status: ExecutionStatus = ExecutionStatus.PENDING


class WorkflowExecutor:
    """
    Execute workflows.
    
    Features:
    - Sequential execution
    - Condition evaluation
    - Transform execution
    - Error handling
    """
    
    def __init__(self):
        """Initialize executor."""
        self._action_handlers: Dict[ActionType, Callable] = {}
        self._running: Dict[str, ExecutionContext] = {}
        self._lock = threading.Lock()
    
    def register_action(
        self,
        action_type: ActionType,
        handler: Callable[[Dict, ExecutionContext], Any]
    ):
        """Register an action handler."""
        self._action_handlers[action_type] = handler
    
    def execute(
        self,
        workflow: Workflow,
        input_data: Any = None,
        async_exec: bool = False
    ) -> ExecutionContext:
        """
        Execute a workflow.
        
        Args:
            workflow: Workflow to execute
            input_data: Initial input data
            async_exec: Run asynchronously
            
        Returns:
            Execution context
        """
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        
        ctx = ExecutionContext(
            workflow_id=workflow.id,
            run_id=run_id,
            variables=copy.deepcopy(workflow.variables),
            input_data=input_data,
            started_at=datetime.now().isoformat(),
            status=ExecutionStatus.PENDING
        )
        
        with self._lock:
            self._running[run_id] = ctx
        
        if async_exec:
            threading.Thread(
                target=self._execute_workflow,
                args=(workflow, ctx)
            ).start()
        else:
            self._execute_workflow(workflow, ctx)
        
        return ctx
    
    def _execute_workflow(self, workflow: Workflow, ctx: ExecutionContext):
        """Execute workflow nodes."""
        ctx.status = ExecutionStatus.RUNNING
        
        try:
            # Find start nodes
            start_nodes = workflow.get_start_nodes()
            if not start_nodes:
                raise ValueError("No trigger nodes found")
            
            # Execute from each start node
            for start in start_nodes:
                self._execute_node(start, ctx.input_data, workflow, ctx)
            
            ctx.status = ExecutionStatus.COMPLETED
            
        except Exception as e:
            ctx.errors.append(str(e))
            ctx.status = ExecutionStatus.FAILED
            logger.error(f"Workflow execution failed: {e}")
        
        finally:
            ctx.completed_at = datetime.now().isoformat()
    
    def _execute_node(
        self,
        node: WorkflowNode,
        input_data: Any,
        workflow: Workflow,
        ctx: ExecutionContext
    ) -> Any:
        """Execute a single node."""
        ctx.current_node = node.id
        
        try:
            # Execute based on node type
            if node.node_type == NodeType.TRIGGER:
                output = input_data
                
            elif node.node_type == NodeType.ACTION:
                output = self._execute_action(node, input_data, ctx)
                
            elif node.node_type == NodeType.CONDITION:
                output = self._evaluate_condition(node, input_data, ctx)
                
            elif node.node_type == NodeType.TRANSFORM:
                output = self._execute_transform(node, input_data, ctx)
                
            elif node.node_type == NodeType.DELAY:
                seconds = node.config.get("seconds", 1)
                time.sleep(seconds)
                output = input_data
                
            elif node.node_type == NodeType.OUTPUT:
                ctx.output_data = input_data
                return input_data
                
            else:
                output = input_data
            
            # Store node output
            ctx.node_outputs[node.id] = output
            
            # Get next nodes
            if node.node_type == NodeType.CONDITION:
                port = "true" if output else "false"
                next_nodes = [
                    (n, p) for n, p in workflow.get_next_nodes(node.id)
                    if any(c.from_port == port and c.to_node == n.id 
                          for c in workflow.connections)
                ]
            else:
                next_nodes = workflow.get_next_nodes(node.id)
            
            # Execute next nodes
            for next_node, _ in next_nodes:
                self._execute_node(next_node, output, workflow, ctx)
            
            return output
            
        except Exception as e:
            ctx.errors.append(f"Node {node.id}: {str(e)}")
            raise
    
    def _execute_action(
        self,
        node: WorkflowNode,
        input_data: Any,
        ctx: ExecutionContext
    ) -> Any:
        """Execute an action node."""
        action_type_str = node.config.get("action_type")
        if not action_type_str:
            return input_data
        
        try:
            action_type = ActionType(action_type_str)
        except ValueError:
            logger.warning(f"Unknown action type: {action_type_str}")
            return input_data
        
        handler = self._action_handlers.get(action_type)
        if handler:
            return handler(node.config, ctx)
        
        # Default handlers
        if action_type == ActionType.SET_VARIABLE:
            var_name = node.config.get("variable", "var")
            ctx.variables[var_name] = input_data
            return input_data
            
        elif action_type == ActionType.SEND_MESSAGE:
            message = node.config.get("message", str(input_data))
            logger.info(f"Workflow message: {message}")
            return message
        
        return input_data
    
    def _evaluate_condition(
        self,
        node: WorkflowNode,
        input_data: Any,
        ctx: ExecutionContext
    ) -> bool:
        """Evaluate a condition node."""
        condition = node.config.get("condition", "True")
        
        # Create evaluation context
        eval_ctx = {
            "input": input_data,
            "data": input_data,
            "vars": ctx.variables,
            "True": True,
            "False": False,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool
        }
        
        try:
            result = eval(condition, {"__builtins__": {}}, eval_ctx)
            return bool(result)
        except Exception as e:
            logger.warning(f"Condition evaluation failed: {e}")
            return False
    
    def _execute_transform(
        self,
        node: WorkflowNode,
        input_data: Any,
        ctx: ExecutionContext
    ) -> Any:
        """Execute a transform node."""
        transform = node.config.get("transform", "data")
        
        # Create evaluation context
        eval_ctx = {
            "data": input_data,
            "input": input_data,
            "vars": ctx.variables,
            "json": __import__("json"),
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "len": len,
            "sum": sum,
            "min": min,
            "max": max
        }
        
        try:
            return eval(transform, {"__builtins__": {}}, eval_ctx)
        except Exception as e:
            logger.warning(f"Transform failed: {e}")
            return input_data
    
    def cancel(self, run_id: str):
        """Cancel a running workflow."""
        with self._lock:
            if run_id in self._running:
                self._running[run_id].status = ExecutionStatus.CANCELLED
    
    def get_status(self, run_id: str) -> Optional[ExecutionContext]:
        """Get execution status."""
        return self._running.get(run_id)


# =============================================================================
# WORKFLOW MANAGER
# =============================================================================

class WorkflowManager:
    """
    Manage workflows.
    
    Features:
    - Create/edit workflows
    - Save/load workflows
    - Template library
    - Execution management
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize workflow manager.
        
        Args:
            storage_path: Path to store workflows
        """
        self.storage_path = storage_path or Path("data/workflows")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._workflows: Dict[str, Workflow] = {}
        self._executor = WorkflowExecutor()
        self._templates = WorkflowTemplates()
        
        # Load existing workflows
        self._load_workflows()
    
    def create(self, name: str, description: str = "") -> Workflow:
        """Create a new workflow."""
        workflow = Workflow.create(name, description)
        self._workflows[workflow.id] = workflow
        self._save_workflow(workflow)
        return workflow
    
    def create_from_template(self, template_name: str) -> Optional[Workflow]:
        """Create workflow from template."""
        templates = self._templates.get_all()
        if template_name in templates:
            workflow = copy.deepcopy(templates[template_name])
            workflow.id = f"workflow_{uuid.uuid4().hex[:8]}"
            self._workflows[workflow.id] = workflow
            self._save_workflow(workflow)
            return workflow
        return None
    
    def get(self, workflow_id: str) -> Optional[Workflow]:
        """Get a workflow by ID."""
        return self._workflows.get(workflow_id)
    
    def list(self) -> List[Workflow]:
        """List all workflows."""
        return list(self._workflows.values())
    
    def update(self, workflow: Workflow):
        """Update a workflow."""
        workflow.updated_at = datetime.now().isoformat()
        self._workflows[workflow.id] = workflow
        self._save_workflow(workflow)
    
    def delete(self, workflow_id: str):
        """Delete a workflow."""
        if workflow_id in self._workflows:
            del self._workflows[workflow_id]
            file_path = self.storage_path / f"{workflow_id}.json"
            if file_path.exists():
                file_path.unlink()
    
    def execute(self, workflow_id: str, input_data: Any = None) -> Optional[ExecutionContext]:
        """Execute a workflow."""
        workflow = self.get(workflow_id)
        if workflow:
            return self._executor.execute(workflow, input_data)
        return None
    
    def export_workflow(self, workflow_id: str) -> Optional[str]:
        """Export workflow as JSON string."""
        workflow = self.get(workflow_id)
        if workflow:
            return json.dumps(workflow.to_dict(), indent=2)
        return None
    
    def import_workflow(self, json_str: str) -> Optional[Workflow]:
        """Import workflow from JSON string."""
        try:
            data = json.loads(json_str)
            workflow = Workflow.from_dict(data)
            # Generate new ID to avoid conflicts
            workflow.id = f"workflow_{uuid.uuid4().hex[:8]}"
            self._workflows[workflow.id] = workflow
            self._save_workflow(workflow)
            return workflow
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return None
    
    def get_templates(self) -> Dict[str, Workflow]:
        """Get available templates."""
        return self._templates.get_all()
    
    def _save_workflow(self, workflow: Workflow):
        """Save workflow to file."""
        file_path = self.storage_path / f"{workflow.id}.json"
        with open(file_path, "w") as f:
            json.dump(workflow.to_dict(), f, indent=2)
    
    def _load_workflows(self):
        """Load workflows from storage."""
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                workflow = Workflow.from_dict(data)
                self._workflows[workflow.id] = workflow
            except Exception as e:
                logger.warning(f"Failed to load workflow {file_path}: {e}")


# Singleton
_workflow_manager: Optional[WorkflowManager] = None


def get_workflow_manager() -> WorkflowManager:
    """Get or create workflow manager."""
    global _workflow_manager
    if _workflow_manager is None:
        _workflow_manager = WorkflowManager()
    return _workflow_manager
