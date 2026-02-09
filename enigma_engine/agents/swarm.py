"""
Agent Swarm System for Enigma AI Engine

Coordinate multiple AI agents working together on complex tasks.

Features:
- Agent spawning and lifecycle
- Task distribution
- Inter-agent communication
- Result aggregation
- Hierarchical organization

Usage:
    from enigma_engine.agents.swarm import AgentSwarm, SwarmAgent
    
    # Create swarm
    swarm = AgentSwarm()
    
    # Add agents
    swarm.spawn("researcher", role="research")
    swarm.spawn("writer", role="writing")
    swarm.spawn("reviewer", role="review")
    
    # Execute task
    result = await swarm.execute("Write a research paper on AI")
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent lifecycle status."""
    IDLE = auto()
    WORKING = auto()
    WAITING = auto()
    COMPLETED = auto()
    FAILED = auto()
    TERMINATED = auto()


class AgentRole(Enum):
    """Standard agent roles."""
    COORDINATOR = auto()   # Orchestrates other agents
    RESEARCHER = auto()    # Gathers information
    WRITER = auto()        # Generates content
    REVIEWER = auto()      # Reviews and critiques
    CODER = auto()         # Writes code
    ANALYST = auto()       # Analyzes data
    PLANNER = auto()       # Creates plans
    EXECUTOR = auto()      # Executes tasks
    SPECIALIST = auto()    # Domain-specific tasks


@dataclass
class Message:
    """Inter-agent message."""
    sender: str
    recipient: str  # "all" for broadcast
    content: str
    message_type: str = "info"  # info, request, response, command
    timestamp: float = field(default_factory=time.time)
    reply_to: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class AgentTask:
    """A task for an agent."""
    id: str
    description: str
    assigned_to: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    priority: int = 5  # 1-10, higher is more important
    status: str = "pending"
    result: Optional[str] = None
    created: float = field(default_factory=time.time)
    completed: Optional[float] = None


@dataclass
class AgentConfig:
    """Agent configuration."""
    name: str
    role: AgentRole
    persona: str = ""
    model: Any = None
    temperature: float = 0.7
    max_retries: int = 3
    timeout: float = 120.0
    
    # Capabilities
    can_spawn: bool = False  # Can spawn sub-agents
    can_delegate: bool = False  # Can delegate tasks
    max_concurrent: int = 3


class SwarmAgent:
    """
    An agent in the swarm.
    """
    
    def __init__(self, config: AgentConfig, swarm: "AgentSwarm"):
        """
        Initialize agent.
        
        Args:
            config: Agent configuration
            swarm: Parent swarm
        """
        self.id = str(uuid.uuid4())[:8]
        self.config = config
        self.name = config.name
        self.role = config.role
        self._swarm = swarm
        
        self.status = AgentStatus.IDLE
        self.current_task: Optional[AgentTask] = None
        self.completed_tasks: List[str] = []
        self.message_queue: List[Message] = []
        
        # Memory
        self.context: List[Dict[str, str]] = []
        self.knowledge: Dict[str, Any] = {}
    
    async def process_task(self, task: AgentTask) -> str:
        """
        Process a task.
        
        Args:
            task: Task to process
            
        Returns:
            Task result
        """
        self.status = AgentStatus.WORKING
        self.current_task = task
        
        logger.info(f"Agent {self.name} processing: {task.description[:50]}...")
        
        try:
            # Build prompt with context
            prompt = self._build_prompt(task)
            
            # Generate response
            result = await self._generate(prompt)
            
            self.completed_tasks.append(task.id)
            self.status = AgentStatus.IDLE
            self.current_task = None
            
            # Update context
            self.context.append({
                "task": task.description,
                "result": result[:500]
            })
            
            return result
        
        except Exception as e:
            logger.error(f"Agent {self.name} failed: {e}")
            self.status = AgentStatus.FAILED
            raise
    
    def _build_prompt(self, task: AgentTask) -> str:
        """Build prompt for task."""
        role_prompts = {
            AgentRole.COORDINATOR: "You are a coordinator. Organize and delegate work efficiently.",
            AgentRole.RESEARCHER: "You are a researcher. Find and synthesize information thoroughly.",
            AgentRole.WRITER: "You are a writer. Create clear, engaging content.",
            AgentRole.REVIEWER: "You are a reviewer. Critique thoroughly and suggest improvements.",
            AgentRole.CODER: "You are a coder. Write clean, working code.",
            AgentRole.ANALYST: "You are an analyst. Analyze data and provide insights.",
            AgentRole.PLANNER: "You are a planner. Create detailed, actionable plans.",
            AgentRole.EXECUTOR: "You are an executor. Complete tasks efficiently."
        }
        
        system = role_prompts.get(self.role, "You are an AI assistant.")
        if self.config.persona:
            system += f" {self.config.persona}"
        
        # Add context from previous tasks
        context_text = ""
        if self.context:
            context_text = "\n\nPrevious context:\n"
            for ctx in self.context[-3:]:  # Last 3 items
                context_text += f"- Task: {ctx['task'][:100]}\n"
                context_text += f"  Result: {ctx['result'][:200]}\n"
        
        # Add messages
        messages_text = ""
        if self.message_queue:
            messages_text = "\n\nMessages from other agents:\n"
            for msg in self.message_queue[-5:]:
                messages_text += f"- From {msg.sender}: {msg.content[:200]}\n"
        
        prompt = f"""{system}
{context_text}
{messages_text}

Task: {task.description}

Your response:"""
        
        return prompt
    
    async def _generate(self, prompt: str) -> str:
        """Generate response."""
        model = self.config.model
        
        if model is None:
            # Mock response
            return f"[Agent {self.name}] Completed task."
        
        if hasattr(model, "generate"):
            # Run in executor for async
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: model.generate(prompt, temperature=self.config.temperature)
            )
        
        return str(model(prompt))
    
    def receive_message(self, message: Message):
        """Receive a message."""
        self.message_queue.append(message)
    
    def send_message(self, recipient: str, content: str, msg_type: str = "info"):
        """Send a message through the swarm."""
        msg = Message(
            sender=self.name,
            recipient=recipient,
            content=content,
            message_type=msg_type
        )
        self._swarm.route_message(msg)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.name,
            "status": self.status.name,
            "completed_tasks": len(self.completed_tasks)
        }


class AgentSwarm:
    """
    Swarm of coordinated agents.
    """
    
    def __init__(
        self,
        default_model: Any = None,
        max_agents: int = 10
    ):
        """
        Initialize swarm.
        
        Args:
            default_model: Default model for agents
            max_agents: Maximum number of agents
        """
        self._default_model = default_model
        self._max_agents = max_agents
        
        self._agents: Dict[str, SwarmAgent] = {}
        self._tasks: Dict[str, AgentTask] = {}
        self._message_log: List[Message] = []
        
        self._coordinator: Optional[SwarmAgent] = None
    
    def spawn(
        self,
        name: str,
        role: str | AgentRole = AgentRole.EXECUTOR,
        **kwargs
    ) -> SwarmAgent:
        """
        Spawn a new agent.
        
        Args:
            name: Agent name
            role: Agent role
            **kwargs: Additional config
            
        Returns:
            New agent
        """
        if len(self._agents) >= self._max_agents:
            raise RuntimeError(f"Maximum agents ({self._max_agents}) reached")
        
        if isinstance(role, str):
            role = AgentRole[role.upper()]
        
        config = AgentConfig(
            name=name,
            role=role,
            model=kwargs.get("model", self._default_model),
            persona=kwargs.get("persona", ""),
            temperature=kwargs.get("temperature", 0.7)
        )
        
        agent = SwarmAgent(config, self)
        self._agents[name] = agent
        
        # Set as coordinator if first coordinator role
        if role == AgentRole.COORDINATOR and self._coordinator is None:
            self._coordinator = agent
        
        logger.info(f"Spawned agent: {name} ({role.name})")
        return agent
    
    def terminate(self, name: str):
        """Terminate an agent."""
        if name in self._agents:
            self._agents[name].status = AgentStatus.TERMINATED
            del self._agents[name]
            logger.info(f"Terminated agent: {name}")
    
    def get_agent(self, name: str) -> Optional[SwarmAgent]:
        """Get agent by name."""
        return self._agents.get(name)
    
    def list_agents(self) -> List[SwarmAgent]:
        """List all agents."""
        return list(self._agents.values())
    
    def route_message(self, message: Message):
        """Route a message to recipient(s)."""
        self._message_log.append(message)
        
        if message.recipient == "all":
            for agent in self._agents.values():
                if agent.name != message.sender:
                    agent.receive_message(message)
        else:
            agent = self._agents.get(message.recipient)
            if agent:
                agent.receive_message(message)
    
    def create_task(
        self,
        description: str,
        assign_to: Optional[str] = None,
        priority: int = 5,
        dependencies: Optional[List[str]] = None
    ) -> AgentTask:
        """
        Create a task.
        
        Args:
            description: Task description
            assign_to: Agent to assign to
            priority: Task priority (1-10)
            dependencies: Task IDs that must complete first
            
        Returns:
            New task
        """
        task = AgentTask(
            id=str(uuid.uuid4())[:8],
            description=description,
            assigned_to=assign_to,
            priority=priority,
            dependencies=dependencies or []
        )
        
        self._tasks[task.id] = task
        return task
    
    async def execute(
        self,
        goal: str,
        auto_plan: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a complex goal using the swarm.
        
        Args:
            goal: Overall goal to achieve
            auto_plan: Automatically decompose into tasks
            
        Returns:
            Results from all agents
        """
        start_time = time.time()
        results = {}
        
        # Ensure we have agents
        if not self._agents:
            self._spawn_default_team()
        
        # Auto-plan if requested
        if auto_plan:
            tasks = await self._decompose_goal(goal)
        else:
            tasks = [self.create_task(goal)]
        
        # Execute tasks
        for task in tasks:
            # Find available agent
            agent = self._select_agent(task)
            task.assigned_to = agent.name
            task.status = "in_progress"
            
            try:
                result = await agent.process_task(task)
                task.result = result
                task.status = "completed"
                task.completed = time.time()
                results[agent.name] = result
            except Exception as e:
                task.status = "failed"
                results[agent.name] = f"Error: {e}"
        
        # Synthesize results
        final_result = await self._synthesize_results(goal, results)
        
        return {
            "goal": goal,
            "final_result": final_result,
            "agent_results": results,
            "tasks_completed": len([t for t in tasks if t.status == "completed"]),
            "duration": time.time() - start_time
        }
    
    def _spawn_default_team(self):
        """Spawn a default team of agents."""
        self.spawn("Coordinator", AgentRole.COORDINATOR)
        self.spawn("Researcher", AgentRole.RESEARCHER)
        self.spawn("Writer", AgentRole.WRITER)
        self.spawn("Reviewer", AgentRole.REVIEWER)
    
    async def _decompose_goal(self, goal: str) -> List[AgentTask]:
        """Decompose a goal into tasks using coordinator."""
        coordinator = self._coordinator or list(self._agents.values())[0]
        
        decompose_prompt = f"""Break down this goal into 3-5 specific tasks:

Goal: {goal}

List the tasks, one per line:
1."""
        
        response = await coordinator._generate(decompose_prompt)
        
        # Parse tasks from response
        tasks = []
        lines = response.strip().split("\n")
        
        for line in lines:
            # Remove numbering
            line = line.strip()
            if line and line[0].isdigit():
                line = line[2:].strip() if len(line) > 2 else line
            
            if line:
                task = self.create_task(line)
                tasks.append(task)
        
        if not tasks:
            tasks = [self.create_task(goal)]
        
        logger.info(f"Decomposed goal into {len(tasks)} tasks")
        return tasks
    
    def _select_agent(self, task: AgentTask) -> SwarmAgent:
        """Select best agent for a task."""
        # If explicitly assigned
        if task.assigned_to and task.assigned_to in self._agents:
            return self._agents[task.assigned_to]
        
        # Match by role keywords
        task_lower = task.description.lower()
        
        role_keywords = {
            AgentRole.RESEARCHER: ["research", "find", "search", "gather", "investigate"],
            AgentRole.WRITER: ["write", "create", "draft", "compose", "generate text"],
            AgentRole.CODER: ["code", "implement", "program", "develop", "script"],
            AgentRole.REVIEWER: ["review", "check", "verify", "critique", "improve"],
            AgentRole.ANALYST: ["analyze", "examine", "study", "evaluate", "assess"],
            AgentRole.PLANNER: ["plan", "organize", "schedule", "structure", "outline"]
        }
        
        for agent in self._agents.values():
            if agent.status == AgentStatus.IDLE:
                keywords = role_keywords.get(agent.role, [])
                if any(kw in task_lower for kw in keywords):
                    return agent
        
        # Return any idle agent
        for agent in self._agents.values():
            if agent.status == AgentStatus.IDLE:
                return agent
        
        # Fall back to first agent
        return list(self._agents.values())[0]
    
    async def _synthesize_results(
        self,
        goal: str,
        results: Dict[str, str]
    ) -> str:
        """Synthesize results from all agents."""
        if not results:
            return "No results to synthesize."
        
        # Use coordinator or first agent
        synthesizer = self._coordinator or list(self._agents.values())[0]
        
        results_text = "\n".join(
            f"- {agent}: {result[:500]}"
            for agent, result in results.items()
        )
        
        prompt = f"""Synthesize these agent results into a cohesive response.

Original goal: {goal}

Agent results:
{results_text}

Synthesized response:"""
        
        return await synthesizer._generate(prompt)
    
    def get_status(self) -> Dict[str, Any]:
        """Get swarm status."""
        return {
            "total_agents": len(self._agents),
            "active_agents": len([a for a in self._agents.values() if a.status == AgentStatus.WORKING]),
            "total_tasks": len(self._tasks),
            "completed_tasks": len([t for t in self._tasks.values() if t.status == "completed"]),
            "agents": [a.to_dict() for a in self._agents.values()]
        }


# Convenience function
async def run_swarm(
    goal: str,
    model: Any = None,
    agents: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Quick function to run a swarm task.
    
    Args:
        goal: Task to accomplish
        model: Language model
        agents: Optional agent configurations
        
    Returns:
        Swarm results
    """
    swarm = AgentSwarm(default_model=model)
    
    if agents:
        for agent_config in agents:
            swarm.spawn(**agent_config)
    
    return await swarm.execute(goal)
