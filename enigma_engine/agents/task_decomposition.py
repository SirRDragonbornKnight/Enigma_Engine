"""
Task Decomposition for Enigma AI Engine

Break complex tasks into agent subtasks.

Features:
- Automatic task analysis
- Subtask generation
- Dependency mapping
- Agent assignment
- Progress tracking

Usage:
    from enigma_engine.agents.task_decomposition import TaskDecomposer
    
    decomposer = TaskDecomposer()
    
    # Decompose a complex task
    subtasks = decomposer.decompose("Build a web scraper that extracts product prices")
    
    # Execute with agents
    results = decomposer.execute(subtasks)
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Subtask:
    """A decomposed subtask."""
    id: str
    title: str
    description: str
    agent_type: str
    dependencies: List[str] = field(default_factory=list)
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecomposedTask:
    """A fully decomposed task."""
    id: str
    original_task: str
    subtasks: List[Subtask]
    total_subtasks: int
    completed_subtasks: int = 0
    
    @property
    def progress(self) -> float:
        """Get completion percentage."""
        if self.total_subtasks == 0:
            return 0.0
        return self.completed_subtasks / self.total_subtasks


class TaskDecomposer:
    """Decompose complex tasks into subtasks."""
    
    def __init__(self):
        """Initialize task decomposer."""
        # Agent type mappings
        self._agent_capabilities = {
            "researcher": ["search", "research", "find", "lookup", "investigate", "analyze"],
            "coder": ["code", "implement", "program", "develop", "build", "create"],
            "writer": ["write", "draft", "compose", "document", "explain", "describe"],
            "critic": ["review", "evaluate", "critique", "assess", "validate", "verify"],
            "planner": ["plan", "design", "architect", "organize", "structure"],
            "executor": ["run", "execute", "test", "deploy", "launch"]
        }
        
        # Task patterns for decomposition
        self._task_patterns = [
            (r"build\s+(.+)", self._decompose_build_task),
            (r"create\s+(.+)", self._decompose_create_task),
            (r"analyze\s+(.+)", self._decompose_analysis_task),
            (r"research\s+(.+)", self._decompose_research_task),
            (r"write\s+(.+)", self._decompose_writing_task),
        ]
    
    def decompose(self, task: str) -> DecomposedTask:
        """
        Decompose a task into subtasks.
        
        Args:
            task: Task description
            
        Returns:
            Decomposed task with subtasks
        """
        task_id = str(uuid.uuid4())[:8]
        task_lower = task.lower()
        
        # Try pattern-based decomposition
        for pattern, decomposer in self._task_patterns:
            match = re.search(pattern, task_lower)
            if match:
                subtasks = decomposer(task, match)
                return DecomposedTask(
                    id=task_id,
                    original_task=task,
                    subtasks=subtasks,
                    total_subtasks=len(subtasks)
                )
        
        # Generic decomposition
        subtasks = self._generic_decompose(task)
        
        return DecomposedTask(
            id=task_id,
            original_task=task,
            subtasks=subtasks,
            total_subtasks=len(subtasks)
        )
    
    def _decompose_build_task(self, task: str, match: re.Match) -> List[Subtask]:
        """Decompose a build/development task."""
        target = match.group(1)
        
        subtasks = [
            Subtask(
                id=str(uuid.uuid4())[:8],
                title="Requirements Analysis",
                description=f"Analyze requirements for building {target}",
                agent_type="planner",
                priority=TaskPriority.HIGH
            ),
            Subtask(
                id=str(uuid.uuid4())[:8],
                title="Research Existing Solutions",
                description=f"Research existing solutions and best practices for {target}",
                agent_type="researcher",
                priority=TaskPriority.MEDIUM
            ),
            Subtask(
                id=str(uuid.uuid4())[:8],
                title="Design Architecture",
                description=f"Design the architecture for {target}",
                agent_type="planner",
                dependencies=["Requirements Analysis"],
                priority=TaskPriority.HIGH
            ),
            Subtask(
                id=str(uuid.uuid4())[:8],
                title="Implement Core",
                description=f"Implement the core functionality of {target}",
                agent_type="coder",
                dependencies=["Design Architecture"],
                priority=TaskPriority.HIGH
            ),
            Subtask(
                id=str(uuid.uuid4())[:8],
                title="Write Tests",
                description=f"Write tests for {target}",
                agent_type="coder",
                dependencies=["Implement Core"],
                priority=TaskPriority.MEDIUM
            ),
            Subtask(
                id=str(uuid.uuid4())[:8],
                title="Code Review",
                description=f"Review implementation of {target}",
                agent_type="critic",
                dependencies=["Implement Core"],
                priority=TaskPriority.MEDIUM
            ),
            Subtask(
                id=str(uuid.uuid4())[:8],
                title="Documentation",
                description=f"Document {target}",
                agent_type="writer",
                dependencies=["Code Review"],
                priority=TaskPriority.LOW
            )
        ]
        
        # Resolve dependencies by ID
        title_to_id = {s.title: s.id for s in subtasks}
        for subtask in subtasks:
            subtask.dependencies = [
                title_to_id.get(dep, dep) for dep in subtask.dependencies
            ]
        
        return subtasks
    
    def _decompose_create_task(self, task: str, match: re.Match) -> List[Subtask]:
        """Decompose a creation task."""
        target = match.group(1)
        
        subtasks = [
            Subtask(
                id=str(uuid.uuid4())[:8],
                title="Define Scope",
                description=f"Define the scope and requirements for creating {target}",
                agent_type="planner",
                priority=TaskPriority.HIGH
            ),
            Subtask(
                id=str(uuid.uuid4())[:8],
                title="Create Initial Version",
                description=f"Create the initial version of {target}",
                agent_type="coder",
                dependencies=["Define Scope"],
                priority=TaskPriority.HIGH
            ),
            Subtask(
                id=str(uuid.uuid4())[:8],
                title="Review and Iterate",
                description=f"Review {target} and make improvements",
                agent_type="critic",
                dependencies=["Create Initial Version"],
                priority=TaskPriority.MEDIUM
            ),
            Subtask(
                id=str(uuid.uuid4())[:8],
                title="Finalize",
                description=f"Finalize {target}",
                agent_type="executor",
                dependencies=["Review and Iterate"],
                priority=TaskPriority.MEDIUM
            )
        ]
        
        title_to_id = {s.title: s.id for s in subtasks}
        for subtask in subtasks:
            subtask.dependencies = [
                title_to_id.get(dep, dep) for dep in subtask.dependencies
            ]
        
        return subtasks
    
    def _decompose_analysis_task(self, task: str, match: re.Match) -> List[Subtask]:
        """Decompose an analysis task."""
        target = match.group(1)
        
        subtasks = [
            Subtask(
                id=str(uuid.uuid4())[:8],
                title="Gather Data",
                description=f"Gather data related to {target}",
                agent_type="researcher",
                priority=TaskPriority.HIGH
            ),
            Subtask(
                id=str(uuid.uuid4())[:8],
                title="Data Processing",
                description=f"Process and clean data for {target} analysis",
                agent_type="coder",
                dependencies=["Gather Data"],
                priority=TaskPriority.MEDIUM
            ),
            Subtask(
                id=str(uuid.uuid4())[:8],
                title="Perform Analysis",
                description=f"Perform detailed analysis of {target}",
                agent_type="researcher",
                dependencies=["Data Processing"],
                priority=TaskPriority.HIGH
            ),
            Subtask(
                id=str(uuid.uuid4())[:8],
                title="Generate Report",
                description=f"Generate analysis report for {target}",
                agent_type="writer",
                dependencies=["Perform Analysis"],
                priority=TaskPriority.MEDIUM
            )
        ]
        
        title_to_id = {s.title: s.id for s in subtasks}
        for subtask in subtasks:
            subtask.dependencies = [
                title_to_id.get(dep, dep) for dep in subtask.dependencies
            ]
        
        return subtasks
    
    def _decompose_research_task(self, task: str, match: re.Match) -> List[Subtask]:
        """Decompose a research task."""
        target = match.group(1)
        
        return [
            Subtask(
                id=str(uuid.uuid4())[:8],
                title="Define Questions",
                description=f"Define research questions for {target}",
                agent_type="planner",
                priority=TaskPriority.HIGH
            ),
            Subtask(
                id=str(uuid.uuid4())[:8],
                title="Literature Search",
                description=f"Search for existing literature on {target}",
                agent_type="researcher",
                priority=TaskPriority.HIGH
            ),
            Subtask(
                id=str(uuid.uuid4())[:8],
                title="Synthesize Findings",
                description=f"Synthesize findings about {target}",
                agent_type="researcher",
                priority=TaskPriority.MEDIUM
            ),
            Subtask(
                id=str(uuid.uuid4())[:8],
                title="Write Summary",
                description=f"Write research summary for {target}",
                agent_type="writer",
                priority=TaskPriority.MEDIUM
            )
        ]
    
    def _decompose_writing_task(self, task: str, match: re.Match) -> List[Subtask]:
        """Decompose a writing task."""
        target = match.group(1)
        
        return [
            Subtask(
                id=str(uuid.uuid4())[:8],
                title="Outline",
                description=f"Create outline for {target}",
                agent_type="planner",
                priority=TaskPriority.HIGH
            ),
            Subtask(
                id=str(uuid.uuid4())[:8],
                title="Research",
                description=f"Research content for {target}",
                agent_type="researcher",
                priority=TaskPriority.MEDIUM
            ),
            Subtask(
                id=str(uuid.uuid4())[:8],
                title="Draft",
                description=f"Write first draft of {target}",
                agent_type="writer",
                priority=TaskPriority.HIGH
            ),
            Subtask(
                id=str(uuid.uuid4())[:8],
                title="Edit",
                description=f"Edit and refine {target}",
                agent_type="critic",
                priority=TaskPriority.MEDIUM
            ),
            Subtask(
                id=str(uuid.uuid4())[:8],
                title="Finalize",
                description=f"Finalize {target}",
                agent_type="writer",
                priority=TaskPriority.LOW
            )
        ]
    
    def _generic_decompose(self, task: str) -> List[Subtask]:
        """Generic task decomposition."""
        # Determine primary agent type
        agent_type = self._detect_agent_type(task)
        
        return [
            Subtask(
                id=str(uuid.uuid4())[:8],
                title="Understand Task",
                description=f"Understand and break down: {task}",
                agent_type="planner",
                priority=TaskPriority.HIGH
            ),
            Subtask(
                id=str(uuid.uuid4())[:8],
                title="Execute Task",
                description=f"Execute the main task: {task}",
                agent_type=agent_type,
                priority=TaskPriority.HIGH
            ),
            Subtask(
                id=str(uuid.uuid4())[:8],
                title="Verify Result",
                description=f"Verify the result of: {task}",
                agent_type="critic",
                priority=TaskPriority.MEDIUM
            )
        ]
    
    def _detect_agent_type(self, task: str) -> str:
        """Detect the most appropriate agent type for a task."""
        task_lower = task.lower()
        
        for agent_type, keywords in self._agent_capabilities.items():
            for keyword in keywords:
                if keyword in task_lower:
                    return agent_type
        
        return "executor"  # Default
    
    def get_ready_subtasks(self, decomposed: DecomposedTask) -> List[Subtask]:
        """Get subtasks that are ready to execute (all dependencies met)."""
        completed_ids = {
            s.id for s in decomposed.subtasks
            if s.status == TaskStatus.COMPLETED
        }
        
        ready = []
        for subtask in decomposed.subtasks:
            if subtask.status != TaskStatus.PENDING:
                continue
            
            # Check if all dependencies are completed
            if all(dep in completed_ids for dep in subtask.dependencies):
                ready.append(subtask)
        
        return ready
    
    def update_progress(self, decomposed: DecomposedTask):
        """Update progress counter."""
        decomposed.completed_subtasks = sum(
            1 for s in decomposed.subtasks
            if s.status == TaskStatus.COMPLETED
        )


class TaskExecutor:
    """Execute decomposed tasks with agents."""
    
    def __init__(self):
        """Initialize executor."""
        self._agents: Dict[str, Any] = {}
    
    def register_agent(self, agent_type: str, agent: Any):
        """Register an agent for a type."""
        self._agents[agent_type] = agent
    
    def execute(
        self,
        decomposed: DecomposedTask,
        callback: Optional[Callable[[Subtask], None]] = None
    ) -> List[Subtask]:
        """
        Execute all subtasks.
        
        Args:
            decomposed: Decomposed task
            callback: Progress callback
            
        Returns:
            List of completed subtasks
        """
        decomposer = TaskDecomposer()
        
        while True:
            ready = decomposer.get_ready_subtasks(decomposed)
            
            if not ready:
                # Check if we're done or blocked
                pending = [
                    s for s in decomposed.subtasks
                    if s.status == TaskStatus.PENDING
                ]
                if not pending:
                    break  # All done
                else:
                    # Some tasks are blocked
                    logger.warning("Some tasks are blocked")
                    break
            
            for subtask in ready:
                self._execute_subtask(subtask)
                decomposer.update_progress(decomposed)
                
                if callback:
                    callback(subtask)
        
        return decomposed.subtasks
    
    def _execute_subtask(self, subtask: Subtask):
        """Execute a single subtask."""
        subtask.status = TaskStatus.IN_PROGRESS
        
        agent = self._agents.get(subtask.agent_type)
        
        if not agent:
            logger.warning(f"No agent for type: {subtask.agent_type}")
            subtask.status = TaskStatus.FAILED
            subtask.error = f"No agent of type {subtask.agent_type}"
            return
        
        try:
            # Execute with agent
            if hasattr(agent, 'execute'):
                result = agent.execute(subtask.description)
            elif hasattr(agent, 'run'):
                result = agent.run(subtask.description)
            else:
                result = str(agent(subtask.description))
            
            subtask.result = result
            subtask.status = TaskStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Subtask failed: {e}")
            subtask.status = TaskStatus.FAILED
            subtask.error = str(e)


# Convenience function
def decompose_task(task: str) -> DecomposedTask:
    """Decompose a task into subtasks."""
    decomposer = TaskDecomposer()
    return decomposer.decompose(task)
