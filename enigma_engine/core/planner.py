"""
Multi-Turn Planning for Enigma AI Engine

Complex task decomposition and execution planning.

Features:
- Task decomposition
- Step-by-step execution
- Progress tracking
- Rollback support
- Dependency handling

Usage:
    from enigma_engine.core.planner import TaskPlanner, get_planner
    
    planner = get_planner()
    
    # Plan a complex task
    plan = planner.plan("Build a REST API with authentication")
    
    # Execute steps
    for step in plan.execute():
        print(f"Completed: {step.name}")
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Set

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of a planning step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class StepType(Enum):
    """Type of planning step."""
    THINK = "think"  # Reasoning step
    ACTION = "action"  # Tool/action execution
    QUERY = "query"  # Information gathering
    VALIDATE = "validate"  # Check results
    DECISION = "decision"  # Branch point
    PARALLEL = "parallel"  # Run multiple steps


@dataclass
class PlanStep:
    """A single step in a plan."""
    id: str
    name: str
    description: str
    step_type: StepType = StepType.ACTION
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    
    # Execution
    action: Optional[Callable] = None
    action_args: Dict[str, Any] = field(default_factory=dict)
    
    # Results
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    
    # Timing
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Metadata
    priority: int = 0
    optional: bool = False
    retry_count: int = 0
    max_retries: int = 3
    
    @property
    def duration(self) -> Optional[float]:
        """Get step duration."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    def can_execute(self, completed: Set[str]) -> bool:
        """Check if step can execute."""
        return all(dep in completed for dep in self.depends_on)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.step_type.value,
            "status": self.status.value,
            "depends_on": self.depends_on,
            "priority": self.priority,
            "optional": self.optional
        }


@dataclass
class Plan:
    """A complete execution plan."""
    id: str
    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    
    # State
    current_step_idx: int = 0
    created_at: float = field(default_factory=time.time)
    
    # Metadata
    estimated_duration: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    
    def add_step(
        self,
        name: str,
        description: str,
        step_type: StepType = StepType.ACTION,
        depends_on: Optional[List[str]] = None,
        action: Optional[Callable] = None,
        **kwargs
    ) -> PlanStep:
        """Add a step to the plan."""
        step_id = f"step_{len(self.steps) + 1}"
        step = PlanStep(
            id=step_id,
            name=name,
            description=description,
            step_type=step_type,
            depends_on=depends_on or [],
            action=action,
            **kwargs
        )
        self.steps.append(step)
        return step
    
    def get_step(self, step_id: str) -> Optional[PlanStep]:
        """Get step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None
    
    def get_ready_steps(self) -> List[PlanStep]:
        """Get steps ready for execution."""
        completed = {s.id for s in self.steps if s.status == StepStatus.COMPLETED}
        ready = []
        
        for step in self.steps:
            if step.status == StepStatus.PENDING and step.can_execute(completed):
                ready.append(step)
        
        return sorted(ready, key=lambda s: -s.priority)
    
    @property
    def progress(self) -> float:
        """Get completion progress (0-1)."""
        if not self.steps:
            return 0.0
        completed = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        return completed / len(self.steps)
    
    @property
    def is_complete(self) -> bool:
        """Check if plan is complete."""
        return all(
            s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)
            for s in self.steps
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "progress": self.progress,
            "created_at": self.created_at,
            "tags": self.tags
        }


class TaskDecomposer:
    """Decomposes tasks into steps."""
    
    # Task patterns and their typical steps
    PATTERNS = {
        "api": [
            ("design", "Design API endpoints and data models"),
            ("setup", "Set up project structure and dependencies"),
            ("implement", "Implement core functionality"),
            ("auth", "Add authentication/authorization"),
            ("validate", "Add input validation"),
            ("test", "Write tests"),
            ("document", "Create documentation")
        ],
        "website": [
            ("plan", "Plan site structure and pages"),
            ("design", "Create wireframes/mockups"),
            ("setup", "Set up project and tooling"),
            ("frontend", "Build frontend components"),
            ("backend", "Implement backend logic"),
            ("style", "Apply styling and responsiveness"),
            ("test", "Test across browsers"),
            ("deploy", "Deploy to hosting")
        ],
        "data": [
            ("collect", "Gather and load data"),
            ("clean", "Clean and preprocess data"),
            ("explore", "Exploratory data analysis"),
            ("transform", "Feature engineering"),
            ("model", "Build and train model"),
            ("evaluate", "Evaluate model performance"),
            ("optimize", "Tune hyperparameters"),
            ("deploy", "Deploy model")
        ],
        "general": [
            ("understand", "Understand the requirements"),
            ("plan", "Create implementation plan"),
            ("implement", "Execute the plan"),
            ("review", "Review and refine"),
            ("complete", "Finalize and deliver")
        ]
    }
    
    def __init__(self, model=None) -> None:
        self._model = model
    
    def detect_task_type(self, goal: str) -> str:
        """Detect task type from goal."""
        goal_lower = goal.lower()
        
        if any(w in goal_lower for w in ["api", "rest", "endpoint", "server"]):
            return "api"
        elif any(w in goal_lower for w in ["website", "web app", "frontend", "ui"]):
            return "website"
        elif any(w in goal_lower for w in ["data", "analysis", "model", "ml", "train"]):
            return "data"
        else:
            return "general"
    
    def decompose(self, goal: str) -> List[Dict[str, str]]:
        """Decompose goal into steps."""
        task_type = self.detect_task_type(goal)
        pattern = self.PATTERNS.get(task_type, self.PATTERNS["general"])
        
        steps = []
        for name, desc in pattern:
            steps.append({
                "name": name,
                "description": f"{desc} for: {goal}"
            })
        
        return steps
    
    def decompose_with_model(self, goal: str) -> List[Dict[str, str]]:
        """Decompose using AI model."""
        if self._model is None:
            return self.decompose(goal)
        
        prompt = f"""Break down this task into clear steps:
Goal: {goal}

Format each step as:
STEP: <name> - <description>

Steps:"""
        
        try:
            response = self._model.generate(prompt, max_new_tokens=300)
            steps = self._parse_model_steps(response, goal)
            return steps if steps else self.decompose(goal)
        except Exception:
            return self.decompose(goal)
    
    def _parse_model_steps(self, response: str, goal: str) -> List[Dict[str, str]]:
        """Parse model response into steps."""
        steps = []
        lines = response.strip().split("\n")
        
        for line in lines:
            line = line.strip()
            # Parse "STEP: name - description" format
            if line.upper().startswith("STEP:"):
                content = line[5:].strip()
                if " - " in content:
                    name, desc = content.split(" - ", 1)
                    steps.append({
                        "name": name.strip().lower().replace(" ", "_"),
                        "description": desc.strip()
                    })
            # Also parse numbered format "1. name - description"
            elif line and line[0].isdigit() and ". " in line:
                content = line.split(". ", 1)[1] if ". " in line else line
                if " - " in content:
                    name, desc = content.split(" - ", 1)
                    steps.append({
                        "name": name.strip().lower().replace(" ", "_"),
                        "description": desc.strip()
                    })
                elif ":" in content:
                    name, desc = content.split(":", 1)
                    steps.append({
                        "name": name.strip().lower().replace(" ", "_"),
                        "description": desc.strip()
                    })
        
        return steps


class PlanExecutor:
    """Executes plans step by step."""
    
    def __init__(self) -> None:
        self._callbacks: Dict[str, List[Callable]] = {
            "step_started": [],
            "step_completed": [],
            "step_failed": [],
            "plan_completed": []
        }
        self._stop_requested = False
    
    def add_callback(self, event: str, callback: Callable) -> None:
        """Add event callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _emit(self, event: str, *args) -> None:
        """Emit event to callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def stop(self) -> None:
        """Request execution stop."""
        self._stop_requested = True
    
    def execute_step(self, step: PlanStep) -> bool:
        """Execute a single step."""
        step.status = StepStatus.IN_PROGRESS
        step.started_at = time.time()
        
        self._emit("step_started", step)
        
        try:
            if step.action:
                step.result = step.action(**step.action_args)
            else:
                # Placeholder for manual steps
                step.result = {"message": "Step requires manual completion"}
            
            step.status = StepStatus.COMPLETED
            step.completed_at = time.time()
            
            self._emit("step_completed", step)
            return True
            
        except Exception as e:
            step.error = str(e)
            step.retry_count += 1
            
            if step.retry_count < step.max_retries:
                step.status = StepStatus.PENDING
                logger.warning(f"Step {step.id} failed, retrying ({step.retry_count}/{step.max_retries})")
            else:
                step.status = StepStatus.FAILED
                step.completed_at = time.time()
                self._emit("step_failed", step)
            
            return False
    
    def execute(self, plan: Plan) -> Iterator[PlanStep]:
        """
        Execute plan, yielding completed steps.
        
        Args:
            plan: Plan to execute
            
        Yields:
            Completed PlanStep objects
        """
        self._stop_requested = False
        
        while not plan.is_complete and not self._stop_requested:
            ready_steps = plan.get_ready_steps()
            
            if not ready_steps:
                # Check for stuck plan
                pending = [s for s in plan.steps if s.status == StepStatus.PENDING]
                if pending:
                    # Mark remaining as blocked
                    for step in pending:
                        step.status = StepStatus.BLOCKED
                break
            
            for step in ready_steps:
                if self._stop_requested:
                    break
                
                success = self.execute_step(step)
                if success:
                    yield step
                elif not step.optional:
                    # Required step failed
                    logger.error(f"Required step {step.id} failed")
                    return
        
        if plan.is_complete:
            self._emit("plan_completed", plan)


class TaskPlanner:
    """High-level task planning interface."""
    
    def __init__(self, model=None) -> None:
        """
        Initialize planner.
        
        Args:
            model: Optional AI model for decomposition
        """
        self._decomposer = TaskDecomposer(model)
        self._executor = PlanExecutor()
        
        # History
        self._plans: Dict[str, Plan] = {}
        self._plan_counter = 0
    
    def plan(
        self,
        goal: str,
        use_ai: bool = False,
        tags: Optional[List[str]] = None
    ) -> Plan:
        """
        Create a plan for a goal.
        
        Args:
            goal: What to accomplish
            use_ai: Use AI for decomposition
            tags: Optional tags for organization
            
        Returns:
            Execution plan
        """
        self._plan_counter += 1
        plan_id = f"plan_{self._plan_counter}"
        
        plan = Plan(
            id=plan_id,
            goal=goal,
            tags=tags or []
        )
        
        # Decompose
        if use_ai:
            steps = self._decomposer.decompose_with_model(goal)
        else:
            steps = self._decomposer.decompose(goal)
        
        # Build dependency chain
        prev_id = None
        for step_data in steps:
            step = plan.add_step(
                name=step_data["name"],
                description=step_data["description"],
                depends_on=[prev_id] if prev_id else []
            )
            prev_id = step.id
        
        self._plans[plan_id] = plan
        logger.info(f"Created plan '{plan_id}' with {len(steps)} steps")
        
        return plan
    
    def execute(self, plan: Plan) -> Iterator[PlanStep]:
        """Execute a plan step by step."""
        return self._executor.execute(plan)
    
    def run(self, goal: str, **kwargs) -> Plan:
        """Plan and execute in one call."""
        plan = self.plan(goal, **kwargs)
        
        for step in self.execute(plan):
            logger.info(f"Completed: {step.name}")
        
        return plan
    
    def add_callback(self, event: str, callback: Callable) -> None:
        """Add execution callback."""
        self._executor.add_callback(event, callback)
    
    def get_plan(self, plan_id: str) -> Optional[Plan]:
        """Get plan by ID."""
        return self._plans.get(plan_id)
    
    def get_plans(self) -> List[Plan]:
        """Get all plans."""
        return list(self._plans.values())
    
    def export_plan(self, plan: Plan, path: str) -> None:
        """Export plan to JSON."""
        with open(path, "w") as f:
            json.dump(plan.to_dict(), f, indent=2)
    
    def get_status_summary(self, plan: Plan) -> Dict[str, Any]:
        """Get plan status summary."""
        status_counts = {}
        for step in plan.steps:
            status = step.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "plan_id": plan.id,
            "goal": plan.goal,
            "total_steps": len(plan.steps),
            "progress": plan.progress,
            "is_complete": plan.is_complete,
            "status_counts": status_counts
        }


# Global instance
_planner: Optional[TaskPlanner] = None


def get_planner() -> TaskPlanner:
    """Get or create global planner."""
    global _planner
    if _planner is None:
        _planner = TaskPlanner()
    return _planner
