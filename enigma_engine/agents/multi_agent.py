"""
================================================================================
MULTI-AGENT SYSTEM - MULTIPLE AI PERSONALITIES
================================================================================

Run multiple AI agents with different personalities, roles, and capabilities.
Agents can collaborate, debate, or work independently on tasks.

FILE: enigma_engine/agents/multi_agent.py
TYPE: Agent System
MAIN CLASSES: MultiAgentSystem, Agent, AgentRole, AgentConversation

FEATURES:
    - Multiple concurrent AI personalities
    - Role-based agents (coder, writer, analyst, etc.)
    - Agent collaboration and debate
    - Task delegation and orchestration
    - Shared memory between agents
    - Agent-to-agent communication

USAGE:
    from enigma_engine.agents.multi_agent import MultiAgentSystem, Agent, AgentRole
    
    system = MultiAgentSystem()
    
    # Create agents with different roles
    coder = system.create_agent("Coder", AgentRole.CODER)
    reviewer = system.create_agent("Reviewer", AgentRole.REVIEWER)
    
    # Have agents collaborate
    result = system.collaborate([coder, reviewer], "Write a Python function")
    
    # Or have agents debate
    debate = system.debate([coder, reviewer], "Best programming language?")
"""

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config import CONFIG

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Predefined agent roles."""
    GENERAL = "general"           # General purpose assistant
    CODER = "coder"               # Code writing specialist
    REVIEWER = "reviewer"         # Code/text review
    WRITER = "writer"             # Creative writing
    ANALYST = "analyst"           # Data analysis
    RESEARCHER = "researcher"     # Research and fact-finding
    TEACHER = "teacher"           # Educational explanations
    CRITIC = "critic"             # Constructive criticism
    PLANNER = "planner"           # Task planning
    EXECUTOR = "executor"         # Task execution
    MEDIATOR = "mediator"         # Conflict resolution
    CUSTOM = "custom"             # User-defined role


@dataclass
class AgentPersonality:
    """Personality traits for an agent."""
    
    name: str
    role: AgentRole
    description: str = ""
    
    # Personality traits (0-1)
    creativity: float = 0.5
    precision: float = 0.5
    verbosity: float = 0.5
    formality: float = 0.5
    enthusiasm: float = 0.5
    skepticism: float = 0.3
    
    # Role-specific prompts
    system_prompt: str = ""
    response_style: str = ""
    
    # Capabilities
    tools: list[str] = field(default_factory=list)
    specialties: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'role': self.role.value,
            'description': self.description,
            'creativity': self.creativity,
            'precision': self.precision,
            'verbosity': self.verbosity,
            'formality': self.formality,
            'enthusiasm': self.enthusiasm,
            'skepticism': self.skepticism,
            'system_prompt': self.system_prompt,
            'response_style': self.response_style,
            'tools': self.tools,
            'specialties': self.specialties,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AgentPersonality':
        return cls(
            name=data['name'],
            role=AgentRole(data.get('role', 'general')),
            description=data.get('description', ''),
            creativity=data.get('creativity', 0.5),
            precision=data.get('precision', 0.5),
            verbosity=data.get('verbosity', 0.5),
            formality=data.get('formality', 0.5),
            enthusiasm=data.get('enthusiasm', 0.5),
            skepticism=data.get('skepticism', 0.3),
            system_prompt=data.get('system_prompt', ''),
            response_style=data.get('response_style', ''),
            tools=data.get('tools', []),
            specialties=data.get('specialties', []),
        )


@dataclass
class AgentMessage:
    """Message in agent conversation."""
    
    agent_id: str
    agent_name: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    message_type: str = "response"  # response, thought, action, question
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'message_type': self.message_type,
            'metadata': self.metadata,
        }


class Agent:
    """
    An individual AI agent with personality and capabilities.
    """
    
    # Role-specific default prompts
    ROLE_PROMPTS = {
        AgentRole.GENERAL: "You are a helpful AI assistant.",
        AgentRole.CODER: "You are an expert programmer. Write clean, efficient code with clear comments. Focus on best practices and maintainability.",
        AgentRole.REVIEWER: "You are a code/text reviewer. Provide constructive feedback, identify issues, and suggest improvements. Be thorough but fair.",
        AgentRole.WRITER: "You are a creative writer. Use vivid language, engaging narratives, and appropriate style for the context.",
        AgentRole.ANALYST: "You are a data analyst. Provide clear, data-driven insights. Use statistics and evidence to support conclusions.",
        AgentRole.RESEARCHER: "You are a researcher. Find accurate information, cite sources when possible, and present balanced viewpoints.",
        AgentRole.TEACHER: "You are a teacher. Explain concepts clearly, use examples, and adapt to the learner's level.",
        AgentRole.CRITIC: "You are a constructive critic. Identify weaknesses while acknowledging strengths. Suggest specific improvements.",
        AgentRole.PLANNER: "You are a project planner. Break down tasks, identify dependencies, estimate effort, and create actionable plans.",
        AgentRole.EXECUTOR: "You are a task executor. Focus on completing tasks efficiently and reporting progress clearly.",
        AgentRole.MEDIATOR: "You are a mediator. Find common ground, facilitate understanding, and help reach consensus.",
    }
    
    def __init__(
        self,
        personality: AgentPersonality,
        engine = None,
        agent_id: str = None
    ):
        """
        Initialize an agent.
        
        Args:
            personality: Agent's personality configuration
            engine: Inference engine (optional, uses default)
            agent_id: Unique agent ID (generated if not provided)
        """
        self.id = agent_id or str(uuid.uuid4())[:8]
        self.personality = personality
        self._engine = engine
        
        # Conversation history
        self.history: list[AgentMessage] = []
        self.memory: dict[str, Any] = {}  # Shared memory
        
        # State
        self.is_active = True
        self.is_thinking = False
        self.last_response_time: Optional[datetime] = None
        
        # Build system prompt
        self._system_prompt = self._build_system_prompt()
        
        logger.info(f"Agent created: {self.personality.name} (role: {self.personality.role.value})")
    
    @property
    def name(self) -> str:
        return self.personality.name
    
    @property
    def role(self) -> AgentRole:
        return self.personality.role
    
    @property
    def engine(self):
        """Get or create inference engine."""
        if self._engine is None:
            try:
                from ..core.inference import EnigmaEngine
                self._engine = EnigmaEngine()
            except Exception as e:
                logger.error(f"Could not create engine: {e}")
        return self._engine
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for this agent."""
        # Start with role-specific prompt
        base_prompt = self.ROLE_PROMPTS.get(
            self.personality.role,
            self.ROLE_PROMPTS[AgentRole.GENERAL]
        )
        
        # Add custom system prompt
        if self.personality.system_prompt:
            base_prompt = f"{base_prompt}\n\n{self.personality.system_prompt}"
        
        # Add personality traits as instructions
        traits = []
        if self.personality.creativity > 0.7:
            traits.append("Be creative and think outside the box.")
        elif self.personality.creativity < 0.3:
            traits.append("Stick to conventional approaches.")
        
        if self.personality.precision > 0.7:
            traits.append("Be precise and detailed.")
        elif self.personality.precision < 0.3:
            traits.append("Focus on the big picture.")
        
        if self.personality.formality > 0.7:
            traits.append("Use formal, professional language.")
        elif self.personality.formality < 0.3:
            traits.append("Use casual, friendly language.")
        
        if self.personality.skepticism > 0.7:
            traits.append("Question assumptions and look for flaws.")
        
        if traits:
            base_prompt += "\n\n" + " ".join(traits)
        
        # Add name and identity
        base_prompt = f"Your name is {self.personality.name}.\n\n{base_prompt}"
        
        return base_prompt
    
    def respond(self, message: str, context: list[AgentMessage] = None) -> str:
        """
        Generate a response to a message.
        
        Args:
            message: Input message
            context: Optional conversation context
        
        Returns:
            Agent's response
        """
        self.is_thinking = True
        
        try:
            # Build context from history and provided context
            history = []
            if context:
                for msg in context[-10:]:  # Last 10 messages
                    role = "assistant" if msg.agent_id == self.id else "user"
                    content = msg.content
                    if msg.agent_id != self.id:
                        content = f"[{msg.agent_name}]: {content}"
                    history.append({"role": role, "content": content})
            
            # Generate response
            if self.engine:
                response = self.engine.generate(
                    message,
                    system_prompt=self._system_prompt,
                    history=history,
                    max_new_tokens=500,
                    temperature=0.5 + (self.personality.creativity * 0.5)
                )
            else:
                response = f"[{self.name}]: I understand your message, but I cannot generate a response without an inference engine."
            
            # Record response
            self.last_response_time = datetime.now()
            self.history.append(AgentMessage(
                agent_id=self.id,
                agent_name=self.name,
                content=response
            ))
            
            return response
            
        finally:
            self.is_thinking = False
    
    def think(self, topic: str) -> str:
        """
        Generate internal thoughts about a topic (not shared with others).
        
        Args:
            topic: Topic to think about
        
        Returns:
            Agent's thoughts
        """
        prompt = f"Think step by step about: {topic}\n\nYour internal thoughts:"
        thoughts = self.respond(prompt)
        
        # Record as thought
        self.history.append(AgentMessage(
            agent_id=self.id,
            agent_name=self.name,
            content=thoughts,
            message_type="thought"
        ))
        
        return thoughts
    
    def ask(self, question: str, target_agent: 'Agent' = None) -> str:
        """
        Ask a question (optionally to another agent).
        
        Args:
            question: Question to ask
            target_agent: Optional agent to direct question to
        
        Returns:
            The question (for logging)
        """
        if target_agent:
            question = f"@{target_agent.name}: {question}"
        
        self.history.append(AgentMessage(
            agent_id=self.id,
            agent_name=self.name,
            content=question,
            message_type="question"
        ))
        
        return question
    
    def remember(self, key: str, value: Any):
        """Store something in agent's memory."""
        self.memory[key] = value
    
    def recall(self, key: str) -> Optional[Any]:
        """Recall something from agent's memory."""
        return self.memory.get(key)
    
    def clear_history(self):
        """Clear conversation history."""
        self.history = []
    
    def to_dict(self) -> dict:
        """Serialize agent to dictionary."""
        return {
            'id': self.id,
            'personality': self.personality.to_dict(),
            'history_length': len(self.history),
            'is_active': self.is_active,
            'last_response': self.last_response_time.isoformat() if self.last_response_time else None,
        }


class AgentConversation:
    """
    A conversation between multiple agents.
    """
    
    def __init__(self, agents: list[Agent], topic: str = ""):
        """
        Initialize a conversation.
        
        Args:
            agents: Agents participating in conversation
            topic: Conversation topic
        """
        self.id = str(uuid.uuid4())[:8]
        self.agents = {a.id: a for a in agents}
        self.topic = topic
        self.messages: list[AgentMessage] = []
        self.started_at = datetime.now()
        self.is_active = True
    
    def add_message(self, agent: Agent, content: str, message_type: str = "response"):
        """Add a message to the conversation."""
        message = AgentMessage(
            agent_id=agent.id,
            agent_name=agent.name,
            content=content,
            message_type=message_type
        )
        self.messages.append(message)
        return message
    
    def get_context(self, for_agent: Agent, limit: int = 10) -> list[AgentMessage]:
        """Get recent conversation context for an agent."""
        return self.messages[-limit:]
    
    def get_summary(self) -> str:
        """Get a summary of the conversation."""
        if not self.messages:
            return "No messages yet."
        
        summary = f"Conversation on: {self.topic}\n"
        summary += f"Participants: {', '.join(a.name for a in self.agents.values())}\n"
        summary += f"Messages: {len(self.messages)}\n"
        
        return summary


class MultiAgentSystem:
    """
    System for managing multiple AI agents.
    
    Features:
    - Create and manage agents
    - Facilitate collaboration and debate
    - Orchestrate complex tasks
    - Shared memory between agents
    """
    
    def __init__(self, engine = None):
        """
        Initialize multi-agent system.
        
        Args:
            engine: Shared inference engine (optional)
        """
        self._engine = engine
        self.agents: dict[str, Agent] = {}
        self.conversations: dict[str, AgentConversation] = {}
        self.shared_memory: dict[str, Any] = {}
        
        # Presets directory
        self.presets_dir = Path(CONFIG.get("data_dir", "data")) / "agent_presets"
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Multi-agent system initialized")
    
    def create_agent(
        self,
        name: str,
        role: AgentRole = AgentRole.GENERAL,
        **personality_kwargs
    ) -> Agent:
        """
        Create a new agent.
        
        Args:
            name: Agent name
            role: Agent role
            **personality_kwargs: Additional personality settings
        
        Returns:
            Created Agent
        """
        personality = AgentPersonality(
            name=name,
            role=role,
            **personality_kwargs
        )
        
        agent = Agent(personality, engine=self._engine)
        self.agents[agent.id] = agent
        
        return agent
    
    def create_agent_from_preset(self, preset_name: str) -> Optional[Agent]:
        """Create agent from a saved preset."""
        preset_path = self.presets_dir / f"{preset_name}.json"
        if not preset_path.exists():
            logger.warning(f"Preset not found: {preset_name}")
            return None
        
        try:
            with open(preset_path) as f:
                data = json.load(f)
            
            personality = AgentPersonality.from_dict(data)
            agent = Agent(personality, engine=self._engine)
            self.agents[agent.id] = agent
            
            return agent
        except Exception as e:
            logger.error(f"Error loading preset: {e}")
            return None
    
    def save_agent_preset(self, agent: Agent, preset_name: str = None):
        """Save agent as a preset."""
        name = preset_name or agent.name.lower().replace(" ", "_")
        preset_path = self.presets_dir / f"{name}.json"
        
        with open(preset_path, 'w') as f:
            json.dump(agent.personality.to_dict(), f, indent=2)
        
        logger.info(f"Saved agent preset: {name}")
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID."""
        return self.agents.get(agent_id)
    
    def get_agent_by_name(self, name: str) -> Optional[Agent]:
        """Get agent by name."""
        for agent in self.agents.values():
            if agent.name.lower() == name.lower():
                return agent
        return None
    
    def list_agents(self) -> list[Agent]:
        """List all agents."""
        return list(self.agents.values())
    
    def remove_agent(self, agent_id: str):
        """Remove an agent from the system."""
        if agent_id in self.agents:
            del self.agents[agent_id]
    
    def collaborate(
        self,
        agents: list[Agent],
        task: str,
        max_rounds: int = 5,
        callback: Callable[[AgentMessage], None] = None
    ) -> AgentConversation:
        """
        Have agents collaborate on a task.
        
        Args:
            agents: Agents to collaborate
            task: Task description
            max_rounds: Maximum conversation rounds
            callback: Optional callback for each message
        
        Returns:
            AgentConversation with results
        """
        conversation = AgentConversation(agents, topic=task)
        self.conversations[conversation.id] = conversation
        
        # Initial prompt
        prompt = f"Let's collaborate on this task: {task}\n\nEach of you, contribute your expertise."
        
        for round_num in range(max_rounds):
            for agent in agents:
                if not conversation.is_active:
                    break
                
                # Build context from conversation
                context = conversation.get_context(agent)
                
                # Get agent's contribution
                if round_num == 0:
                    response = agent.respond(prompt, context)
                else:
                    last_message = conversation.messages[-1] if conversation.messages else None
                    if last_message:
                        response = agent.respond(
                            f"Continue collaborating. Previous contribution from {last_message.agent_name}: {last_message.content[:200]}...",
                            context
                        )
                    else:
                        response = agent.respond("Continue collaborating.", context)
                
                message = conversation.add_message(agent, response)
                
                if callback:
                    callback(message)
                
                # Check for completion signals
                if "DONE" in response or "complete" in response.lower()[-50:]:
                    conversation.is_active = False
                    break
        
        return conversation
    
    def debate(
        self,
        agents: list[Agent],
        topic: str,
        max_rounds: int = 3,
        callback: Callable[[AgentMessage], None] = None
    ) -> AgentConversation:
        """
        Have agents debate a topic.
        
        Args:
            agents: Agents to debate (should be 2+)
            topic: Debate topic
            max_rounds: Rounds of debate
            callback: Optional callback for each message
        
        Returns:
            AgentConversation with debate
        """
        if len(agents) < 2:
            raise ValueError("Debate requires at least 2 agents")
        
        conversation = AgentConversation(agents, topic=f"Debate: {topic}")
        self.conversations[conversation.id] = conversation
        
        # Initial positions
        for i, agent in enumerate(agents):
            position = "for" if i % 2 == 0 else "against"
            prompt = f"Debate topic: {topic}\n\nYou are arguing {position} this position. Make your opening statement."
            
            response = agent.respond(prompt)
            message = conversation.add_message(agent, response)
            
            if callback:
                callback(message)
        
        # Rebuttals and responses
        for round_num in range(max_rounds - 1):
            for agent in agents:
                context = conversation.get_context(agent)
                
                # Find opposing argument to respond to
                opposing = [m for m in context if m.agent_id != agent.id]
                if opposing:
                    last_opposing = opposing[-1]
                    prompt = f"Respond to {last_opposing.agent_name}'s argument: {last_opposing.content[:300]}..."
                else:
                    prompt = "Continue making your case."
                
                response = agent.respond(prompt, context)
                message = conversation.add_message(agent, response)
                
                if callback:
                    callback(message)
        
        return conversation
    
    def delegate_task(
        self,
        task: str,
        planner: Agent = None,
        executors: list[Agent] = None
    ) -> dict[str, Any]:
        """
        Delegate a complex task using planning and execution agents.
        
        Args:
            task: Task to complete
            planner: Planning agent (created if not provided)
            executors: Execution agents (created if not provided)
        
        Returns:
            Results dictionary
        """
        # Create planner if needed
        if planner is None:
            planner = self.create_agent("TaskPlanner", AgentRole.PLANNER)
        
        # Create executors if needed
        if executors is None or not executors:
            executors = [self.create_agent("Executor", AgentRole.EXECUTOR)]
        
        # Step 1: Plan the task
        plan_prompt = f"""Break down this task into specific, actionable steps:

Task: {task}

Provide a numbered list of steps."""
        
        plan = planner.respond(plan_prompt)
        
        # Step 2: Execute each step
        results = {
            'task': task,
            'plan': plan,
            'steps': [],
            'completed': True
        }
        
        # Parse steps (simple line-based parsing)
        import re
        steps = re.findall(r'\d+\.\s*(.+)', plan)
        
        for i, step in enumerate(steps):
            # Assign to an executor (round-robin)
            executor = executors[i % len(executors)]
            
            step_prompt = f"Execute this step: {step}"
            result = executor.respond(step_prompt)
            
            results['steps'].append({
                'step': step,
                'executor': executor.name,
                'result': result
            })
        
        return results
    
    def brainstorm(
        self,
        topic: str,
        agents: list[Agent] = None,
        num_ideas: int = 5
    ) -> list[str]:
        """
        Have agents brainstorm ideas.
        
        Args:
            topic: Brainstorming topic
            agents: Agents to brainstorm (creates defaults if not provided)
            num_ideas: Target number of ideas
        
        Returns:
            List of ideas
        """
        if agents is None or not agents:
            # Create diverse brainstorming team
            agents = [
                self.create_agent("Creative", AgentRole.WRITER, creativity=0.9),
                self.create_agent("Practical", AgentRole.ANALYST, precision=0.9),
                self.create_agent("Challenger", AgentRole.CRITIC, skepticism=0.8),
            ]
        
        ideas = []
        
        for agent in agents:
            prompt = f"""Brainstorm ideas about: {topic}

Generate {num_ideas // len(agents) + 1} unique ideas. Be creative!

Format each idea on a new line starting with "IDEA:"."""
            
            response = agent.respond(prompt)
            
            # Extract ideas
            import re
            agent_ideas = re.findall(r'IDEA:\s*(.+)', response, re.IGNORECASE)
            ideas.extend(agent_ideas)
        
        return ideas[:num_ideas]
    
    def get_conversation(self, conversation_id: str) -> Optional[AgentConversation]:
        """Get a conversation by ID."""
        return self.conversations.get(conversation_id)
    
    def list_conversations(self) -> list[AgentConversation]:
        """List all conversations."""
        return list(self.conversations.values())
    
    def share_memory(self, key: str, value: Any):
        """Share data with all agents."""
        self.shared_memory[key] = value
        for agent in self.agents.values():
            agent.remember(f"shared_{key}", value)
    
    def get_shared_memory(self, key: str) -> Optional[Any]:
        """Get shared memory value."""
        return self.shared_memory.get(key)


# Global system instance
_system: Optional[MultiAgentSystem] = None


def get_multi_agent_system() -> MultiAgentSystem:
    """Get or create global multi-agent system."""
    global _system
    if _system is None:
        _system = MultiAgentSystem()
    return _system


# Preset agent configurations
PRESET_AGENTS = {
    "coder": {
        "name": "CodeMaster",
        "role": "coder",
        "creativity": 0.4,
        "precision": 0.9,
        "verbosity": 0.6,
        "formality": 0.7,
        "description": "Expert programmer focused on clean, efficient code"
    },
    "writer": {
        "name": "WordSmith",
        "role": "writer",
        "creativity": 0.9,
        "precision": 0.4,
        "verbosity": 0.8,
        "formality": 0.5,
        "description": "Creative writer with vivid imagination"
    },
    "analyst": {
        "name": "DataMind",
        "role": "analyst",
        "creativity": 0.3,
        "precision": 0.95,
        "verbosity": 0.5,
        "formality": 0.8,
        "description": "Data-driven analyst who loves numbers"
    },
    "teacher": {
        "name": "Professor",
        "role": "teacher",
        "creativity": 0.6,
        "precision": 0.7,
        "verbosity": 0.7,
        "formality": 0.6,
        "enthusiasm": 0.8,
        "description": "Patient teacher who explains things clearly"
    },
    "critic": {
        "name": "Reviewer",
        "role": "critic",
        "creativity": 0.4,
        "precision": 0.8,
        "verbosity": 0.5,
        "formality": 0.6,
        "skepticism": 0.8,
        "description": "Constructive critic who finds ways to improve"
    },
}
