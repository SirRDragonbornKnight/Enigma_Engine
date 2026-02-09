"""
Inter-Agent Tool Sharing for Enigma AI Engine

Share tool outputs between agents.

Features:
- Tool output piping
- Data transformation
- Cache for shared results
- Pipeline composition
- Error propagation

Usage:
    from enigma_engine.agents.tool_sharing import ToolPipeline, SharedToolContext
    
    pipeline = ToolPipeline()
    
    # Agent A's output feeds to Agent B
    pipeline.add_step("researcher", "search", {"query": "AI news"})
    pipeline.add_step("summarizer", "summarize", use_previous=True)
    
    result = pipeline.execute()
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolOutput:
    """Output from a tool execution."""
    tool_name: str
    agent_id: str
    output: Any
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineStep:
    """A step in a tool pipeline."""
    step_id: str
    agent_id: str
    tool_name: str
    params: Dict[str, Any]
    use_previous: bool = False
    transform: Optional[Callable[[Any], Any]] = None
    
    # Execution results
    output: Optional[ToolOutput] = None
    executed: bool = False


class SharedToolContext:
    """Shared context for tool outputs between agents."""
    
    def __init__(self, ttl_seconds: float = 3600):
        """
        Initialize shared context.
        
        Args:
            ttl_seconds: Time to live for cached outputs
        """
        self.ttl = ttl_seconds
        self._outputs: Dict[str, ToolOutput] = {}
        self._listeners: List[Callable[[str, ToolOutput], None]] = []
    
    def store(self, key: str, output: ToolOutput):
        """
        Store a tool output.
        
        Args:
            key: Unique key for the output
            output: Tool output to store
        """
        self._outputs[key] = output
        
        # Notify listeners
        for listener in self._listeners:
            try:
                listener(key, output)
            except Exception as e:
                logger.error(f"Listener error: {e}")
    
    def get(self, key: str) -> Optional[ToolOutput]:
        """
        Get a stored output.
        
        Args:
            key: Output key
            
        Returns:
            Tool output or None if not found/expired
        """
        output = self._outputs.get(key)
        
        if output:
            # Check TTL
            if time.time() - output.timestamp > self.ttl:
                del self._outputs[key]
                return None
        
        return output
    
    def get_by_agent(self, agent_id: str) -> List[ToolOutput]:
        """Get all outputs from an agent."""
        return [
            o for o in self._outputs.values()
            if o.agent_id == agent_id and time.time() - o.timestamp <= self.ttl
        ]
    
    def get_by_tool(self, tool_name: str) -> List[ToolOutput]:
        """Get all outputs from a specific tool."""
        return [
            o for o in self._outputs.values()
            if o.tool_name == tool_name and time.time() - o.timestamp <= self.ttl
        ]
    
    def add_listener(self, callback: Callable[[str, ToolOutput], None]):
        """Add listener for new outputs."""
        self._listeners.append(callback)
    
    def clear(self):
        """Clear all stored outputs."""
        self._outputs.clear()
    
    def cleanup_expired(self):
        """Remove expired entries."""
        now = time.time()
        expired = [
            k for k, v in self._outputs.items()
            if now - v.timestamp > self.ttl
        ]
        for k in expired:
            del self._outputs[k]


class ToolPipeline:
    """Pipeline for chaining tool outputs between agents."""
    
    def __init__(self, context: Optional[SharedToolContext] = None):
        """
        Initialize pipeline.
        
        Args:
            context: Shared context for outputs
        """
        self.context = context or SharedToolContext()
        self._steps: List[PipelineStep] = []
        self._executors: Dict[str, Callable] = {}
    
    def register_executor(self, agent_id: str, executor: Callable):
        """
        Register a tool executor for an agent.
        
        Args:
            agent_id: Agent ID
            executor: Function that takes (tool_name, params) and returns result
        """
        self._executors[agent_id] = executor
    
    def add_step(
        self,
        agent_id: str,
        tool_name: str,
        params: Optional[Dict[str, Any]] = None,
        use_previous: bool = False,
        transform: Optional[Callable[[Any], Any]] = None
    ) -> str:
        """
        Add a step to the pipeline.
        
        Args:
            agent_id: Agent to execute
            tool_name: Tool to run
            params: Tool parameters
            use_previous: Whether to use previous step's output
            transform: Transform function for previous output
            
        Returns:
            Step ID
        """
        step = PipelineStep(
            step_id=str(uuid.uuid4())[:8],
            agent_id=agent_id,
            tool_name=tool_name,
            params=params or {},
            use_previous=use_previous,
            transform=transform
        )
        
        self._steps.append(step)
        return step.step_id
    
    def execute(self) -> List[ToolOutput]:
        """
        Execute the entire pipeline.
        
        Returns:
            List of outputs from all steps
        """
        outputs = []
        previous_output = None
        
        for step in self._steps:
            try:
                # Prepare parameters
                params = dict(step.params)
                
                # Inject previous output if requested
                if step.use_previous and previous_output:
                    data = previous_output.output
                    
                    # Apply transform if provided
                    if step.transform:
                        data = step.transform(data)
                    
                    params["input"] = data
                
                # Execute
                output = self._execute_step(step, params)
                outputs.append(output)
                
                # Store in context
                self.context.store(f"{step.agent_id}:{step.tool_name}", output)
                
                previous_output = output
                step.output = output
                step.executed = True
                
            except Exception as e:
                logger.error(f"Pipeline step {step.step_id} failed: {e}")
                outputs.append(ToolOutput(
                    tool_name=step.tool_name,
                    agent_id=step.agent_id,
                    output=None,
                    success=False,
                    error=str(e)
                ))
                break
        
        return outputs
    
    def _execute_step(self, step: PipelineStep, params: Dict) -> ToolOutput:
        """Execute a single step."""
        executor = self._executors.get(step.agent_id)
        
        if executor:
            result = executor(step.tool_name, params)
        else:
            # Default: return params as output (for testing)
            result = params
        
        return ToolOutput(
            tool_name=step.tool_name,
            agent_id=step.agent_id,
            output=result
        )
    
    def reset(self):
        """Reset pipeline for re-execution."""
        for step in self._steps:
            step.output = None
            step.executed = False
    
    def clear(self):
        """Clear all steps."""
        self._steps.clear()
    
    def get_steps(self) -> List[PipelineStep]:
        """Get all steps."""
        return list(self._steps)


class ToolRouter:
    """Route tool calls between agents based on capabilities."""
    
    def __init__(self, context: Optional[SharedToolContext] = None):
        """
        Initialize router.
        
        Args:
            context: Shared context
        """
        self.context = context or SharedToolContext()
        self._capabilities: Dict[str, List[str]] = {}  # agent_id -> tool names
        self._executors: Dict[str, Callable] = {}
    
    def register_agent(
        self,
        agent_id: str,
        tools: List[str],
        executor: Callable
    ):
        """
        Register an agent with its capabilities.
        
        Args:
            agent_id: Agent ID
            tools: List of tool names the agent can execute
            executor: Tool executor function
        """
        self._capabilities[agent_id] = tools
        self._executors[agent_id] = executor
    
    def find_agent_for_tool(self, tool_name: str) -> Optional[str]:
        """Find an agent that can execute a tool."""
        for agent_id, tools in self._capabilities.items():
            if tool_name in tools:
                return agent_id
        return None
    
    def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        preferred_agent: Optional[str] = None
    ) -> ToolOutput:
        """
        Execute a tool, routing to appropriate agent.
        
        Args:
            tool_name: Tool to execute
            params: Tool parameters
            preferred_agent: Preferred agent (if capable)
            
        Returns:
            Tool output
        """
        # Find agent
        agent_id = preferred_agent
        if not agent_id or tool_name not in self._capabilities.get(agent_id, []):
            agent_id = self.find_agent_for_tool(tool_name)
        
        if not agent_id:
            return ToolOutput(
                tool_name=tool_name,
                agent_id="none",
                output=None,
                success=False,
                error=f"No agent found for tool: {tool_name}"
            )
        
        # Execute
        executor = self._executors[agent_id]
        try:
            result = executor(tool_name, params)
            output = ToolOutput(
                tool_name=tool_name,
                agent_id=agent_id,
                output=result
            )
        except Exception as e:
            output = ToolOutput(
                tool_name=tool_name,
                agent_id=agent_id,
                output=None,
                success=False,
                error=str(e)
            )
        
        # Store in context
        self.context.store(f"{agent_id}:{tool_name}", output)
        
        return output
    
    def get_available_tools(self) -> Dict[str, List[str]]:
        """Get all available tools and their providers."""
        return dict(self._capabilities)


# Global instances
_global_context: Optional[SharedToolContext] = None
_global_router: Optional[ToolRouter] = None


def get_shared_context() -> SharedToolContext:
    """Get global shared context."""
    global _global_context
    if _global_context is None:
        _global_context = SharedToolContext()
    return _global_context


def get_tool_router() -> ToolRouter:
    """Get global tool router."""
    global _global_router
    if _global_router is None:
        _global_router = ToolRouter(get_shared_context())
    return _global_router


def share_output(agent_id: str, tool_name: str, output: Any):
    """Share a tool output to the global context."""
    context = get_shared_context()
    context.store(
        f"{agent_id}:{tool_name}",
        ToolOutput(tool_name=tool_name, agent_id=agent_id, output=output)
    )


def get_shared_output(agent_id: str, tool_name: str) -> Optional[Any]:
    """Get a shared output from the global context."""
    context = get_shared_context()
    output = context.get(f"{agent_id}:{tool_name}")
    return output.output if output else None
