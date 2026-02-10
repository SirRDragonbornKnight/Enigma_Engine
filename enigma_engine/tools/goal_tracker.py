"""
Goal Tracking System - Track user goals across sessions.

Supports:
- Goal inference from conversation
- Goal decomposition into subgoals
- Progress tracking
- Goal prioritization
- Completion detection
"""

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class GoalStatus(Enum):
    """Status of a goal."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class GoalPriority(Enum):
    """Priority levels for goals."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Goal:
    """A user goal to track."""
    id: str
    title: str
    description: str = ""
    status: GoalStatus = GoalStatus.NOT_STARTED
    priority: GoalPriority = GoalPriority.MEDIUM
    parent_id: Optional[str] = None  # For subgoals
    subgoal_ids: list[str] = field(default_factory=list)
    progress: float = 0.0  # 0-100
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    due_date: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['status'] = self.status.value
        d['priority'] = self.priority.value
        return d
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Goal':
        """Create from dictionary."""
        data['status'] = GoalStatus(data.get('status', 'not_started'))
        data['priority'] = GoalPriority(data.get('priority', 2))
        return cls(**data)


class GoalTracker:
    """
    Tracks user goals across sessions.
    
    Features:
    - Create and manage goals
    - Decompose goals into subgoals
    - Track progress automatically
    - Infer goals from conversation
    - Detect goal completion
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/goals.json")
        self.goals: dict[str, Goal] = {}
        self._load()
        
    def _load(self):
        """Load goals from storage."""
        try:
            if self.storage_path.exists():
                with open(self.storage_path) as f:
                    data = json.load(f)
                    for gid, gdata in data.get('goals', {}).items():
                        self.goals[gid] = Goal.from_dict(gdata)
        except Exception as e:
            logger.warning(f"Failed to load goals: {e}")
            
    def _save(self):
        """Save goals to storage."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'goals': {gid: g.to_dict() for gid, g in self.goals.items()},
                'updated_at': time.time()
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save goals: {e}")
    
    # ========== Goal Creation ==========
    
    def create_goal(
        self,
        title: str,
        description: str = "",
        priority: GoalPriority = GoalPriority.MEDIUM,
        parent_id: Optional[str] = None,
        due_date: Optional[str] = None,
        tags: Optional[list[str]] = None,
        success_criteria: Optional[list[str]] = None
    ) -> Goal:
        """Create a new goal."""
        goal = Goal(
            id=str(uuid.uuid4())[:8],
            title=title,
            description=description,
            priority=priority,
            parent_id=parent_id,
            due_date=due_date,
            tags=tags or [],
            success_criteria=success_criteria or []
        )
        
        self.goals[goal.id] = goal
        
        # Link to parent if subgoal
        if parent_id and parent_id in self.goals:
            self.goals[parent_id].subgoal_ids.append(goal.id)
            self.goals[parent_id].updated_at = time.time()
            
        self._save()
        return goal
    
    def create_subgoal(self, parent_id: str, title: str, **kwargs) -> Optional[Goal]:
        """Create a subgoal under a parent goal."""
        if parent_id not in self.goals:
            return None
        return self.create_goal(title, parent_id=parent_id, **kwargs)
    
    # ========== Goal Retrieval ==========
    
    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID."""
        return self.goals.get(goal_id)
    
    def get_all_goals(self) -> list[Goal]:
        """Get all goals."""
        return list(self.goals.values())
    
    def get_active_goals(self) -> list[Goal]:
        """Get goals that are in progress or not started."""
        return [g for g in self.goals.values() 
                if g.status in (GoalStatus.NOT_STARTED, GoalStatus.IN_PROGRESS)]
    
    def get_top_level_goals(self) -> list[Goal]:
        """Get goals that are not subgoals."""
        return [g for g in self.goals.values() if g.parent_id is None]
    
    def get_subgoals(self, goal_id: str) -> list[Goal]:
        """Get subgoals of a goal."""
        goal = self.goals.get(goal_id)
        if not goal:
            return []
        return [self.goals[sid] for sid in goal.subgoal_ids if sid in self.goals]
    
    def get_goals_by_status(self, status: GoalStatus) -> list[Goal]:
        """Get goals by status."""
        return [g for g in self.goals.values() if g.status == status]
    
    def get_goals_by_priority(self, priority: GoalPriority) -> list[Goal]:
        """Get goals by priority."""
        return [g for g in self.goals.values() if g.priority == priority]
    
    def get_goals_by_tag(self, tag: str) -> list[Goal]:
        """Get goals with a specific tag."""
        return [g for g in self.goals.values() if tag in g.tags]
    
    def search_goals(self, query: str) -> list[Goal]:
        """Search goals by title or description."""
        q = query.lower()
        return [g for g in self.goals.values() 
                if q in g.title.lower() or q in g.description.lower()]
    
    # ========== Goal Updates ==========
    
    def update_goal(self, goal_id: str, **kwargs) -> Optional[Goal]:
        """Update a goal's attributes."""
        goal = self.goals.get(goal_id)
        if not goal:
            return None
            
        for key, value in kwargs.items():
            if hasattr(goal, key):
                setattr(goal, key, value)
                
        goal.updated_at = time.time()
        self._save()
        return goal
    
    def set_status(self, goal_id: str, status: GoalStatus) -> Optional[Goal]:
        """Set goal status."""
        goal = self.goals.get(goal_id)
        if not goal:
            return None
            
        goal.status = status
        goal.updated_at = time.time()
        
        if status == GoalStatus.COMPLETED:
            goal.completed_at = time.time()
            goal.progress = 100.0
            
        self._save()
        self._update_parent_progress(goal)
        return goal
    
    def start_goal(self, goal_id: str) -> Optional[Goal]:
        """Mark a goal as in progress."""
        return self.set_status(goal_id, GoalStatus.IN_PROGRESS)
    
    def complete_goal(self, goal_id: str) -> Optional[Goal]:
        """Mark a goal as completed."""
        return self.set_status(goal_id, GoalStatus.COMPLETED)
    
    def abandon_goal(self, goal_id: str, reason: str = "") -> Optional[Goal]:
        """Abandon a goal."""
        goal = self.set_status(goal_id, GoalStatus.ABANDONED)
        if goal and reason:
            goal.notes.append(f"Abandoned: {reason}")
            self._save()
        return goal
    
    def block_goal(self, goal_id: str, blocker: str) -> Optional[Goal]:
        """Mark a goal as blocked."""
        goal = self.goals.get(goal_id)
        if not goal:
            return None
            
        goal.status = GoalStatus.BLOCKED
        goal.blockers.append(blocker)
        goal.updated_at = time.time()
        self._save()
        return goal
    
    def unblock_goal(self, goal_id: str) -> Optional[Goal]:
        """Unblock a goal."""
        goal = self.goals.get(goal_id)
        if not goal:
            return None
            
        goal.status = GoalStatus.IN_PROGRESS
        goal.blockers.clear()
        goal.updated_at = time.time()
        self._save()
        return goal
    
    def update_progress(self, goal_id: str, progress: float) -> Optional[Goal]:
        """Update goal progress (0-100)."""
        goal = self.goals.get(goal_id)
        if not goal:
            return None
            
        goal.progress = max(0, min(100, progress))
        goal.updated_at = time.time()
        
        if goal.progress >= 100:
            goal.status = GoalStatus.COMPLETED
            goal.completed_at = time.time()
            
        self._save()
        self._update_parent_progress(goal)
        return goal
    
    def add_note(self, goal_id: str, note: str) -> Optional[Goal]:
        """Add a note to a goal."""
        goal = self.goals.get(goal_id)
        if not goal:
            return None
            
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        goal.notes.append(f"[{timestamp}] {note}")
        goal.updated_at = time.time()
        self._save()
        return goal
    
    def _update_parent_progress(self, goal: Goal):
        """Update parent goal progress based on subgoals."""
        if not goal.parent_id:
            return
            
        parent = self.goals.get(goal.parent_id)
        if not parent or not parent.subgoal_ids:
            return
            
        # Calculate average progress of subgoals
        total = 0.0
        for sid in parent.subgoal_ids:
            if sid in self.goals:
                total += self.goals[sid].progress
                
        parent.progress = total / len(parent.subgoal_ids)
        parent.updated_at = time.time()
        
        if parent.progress >= 100:
            parent.status = GoalStatus.COMPLETED
            parent.completed_at = time.time()
            
        self._save()
    
    # ========== Goal Decomposition ==========
    
    def decompose_goal(self, goal_id: str, subtasks: list[str]) -> list[Goal]:
        """Break a goal into subgoals."""
        goal = self.goals.get(goal_id)
        if not goal:
            return []
            
        subgoals = []
        for i, task in enumerate(subtasks):
            subgoal = self.create_subgoal(
                goal_id,
                title=task,
                priority=goal.priority
            )
            if subgoal:
                subgoals.append(subgoal)
                
        return subgoals
    
    def suggest_decomposition(self, goal_id: str) -> list[str]:
        """
        Suggest how to break down a goal.
        Returns list of suggested subtask titles.
        """
        goal = self.goals.get(goal_id)
        if not goal:
            return []
            
        # Simple heuristic-based suggestions
        title = goal.title.lower()
        
        if 'learn' in title or 'study' in title:
            return [
                "Research fundamentals",
                "Find learning resources",
                "Practice basics",
                "Build a small project",
                "Review and solidify knowledge"
            ]
        elif 'build' in title or 'create' in title or 'make' in title:
            return [
                "Define requirements",
                "Design solution",
                "Implement core features",
                "Test and debug",
                "Polish and document"
            ]
        elif 'fix' in title or 'debug' in title or 'solve' in title:
            return [
                "Reproduce the issue",
                "Identify root cause",
                "Develop fix",
                "Test fix",
                "Deploy and verify"
            ]
        elif 'improve' in title or 'optimize' in title:
            return [
                "Measure current state",
                "Identify bottlenecks",
                "Research solutions",
                "Implement improvements",
                "Measure results"
            ]
        else:
            return [
                "Define scope",
                "Research options",
                "Plan approach",
                "Execute",
                "Review results"
            ]
    
    # ========== Goal Inference ==========
    
    def infer_goal_from_text(self, text: str) -> Optional[dict[str, Any]]:
        """
        Infer a potential goal from user text.
        Returns goal data if a goal is detected.
        """
        text_lower = text.lower()
        
        # Goal indicators
        goal_phrases = [
            "i want to", "i need to", "i'm trying to", "i'd like to",
            "my goal is", "help me", "i wish", "can you help",
            "how do i", "i'm working on", "i have to"
        ]
        
        has_goal_phrase = any(phrase in text_lower for phrase in goal_phrases)
        
        if not has_goal_phrase:
            return None
            
        # Extract the goal
        for phrase in goal_phrases:
            if phrase in text_lower:
                idx = text_lower.find(phrase)
                goal_text = text[idx + len(phrase):].strip()
                # Take first sentence
                for end in ['.', '!', '?', '\n']:
                    if end in goal_text:
                        goal_text = goal_text[:goal_text.find(end)]
                        break
                        
                if len(goal_text) > 5:
                    # Determine priority from urgency words
                    priority = GoalPriority.MEDIUM
                    if any(w in text_lower for w in ['urgent', 'asap', 'immediately', 'critical']):
                        priority = GoalPriority.CRITICAL
                    elif any(w in text_lower for w in ['important', 'need to', 'must']):
                        priority = GoalPriority.HIGH
                    elif any(w in text_lower for w in ['eventually', 'someday', 'maybe']):
                        priority = GoalPriority.LOW
                        
                    return {
                        'title': goal_text.strip().capitalize(),
                        'description': text,
                        'priority': priority,
                        'inferred': True
                    }
                    
        return None
    
    # ========== Goal Completion Detection ==========
    
    def check_completion(self, goal_id: str) -> bool:
        """
        Check if a goal's success criteria are met.
        Returns True if goal appears complete.
        """
        goal = self.goals.get(goal_id)
        if not goal:
            return False
            
        # Already completed
        if goal.status == GoalStatus.COMPLETED:
            return True
            
        # Check subgoals
        if goal.subgoal_ids:
            all_complete = all(
                self.goals.get(sid, Goal(id='', title='')).status == GoalStatus.COMPLETED
                for sid in goal.subgoal_ids
            )
            if all_complete:
                return True
                
        # Progress-based
        if goal.progress >= 100:
            return True
            
        return False
    
    def auto_complete_check(self) -> list[Goal]:
        """Check all goals and auto-complete those that are done."""
        completed = []
        for goal in self.get_active_goals():
            if self.check_completion(goal.id):
                self.complete_goal(goal.id)
                completed.append(goal)
        return completed
    
    # ========== Prioritization ==========
    
    def get_prioritized_goals(self) -> list[Goal]:
        """Get active goals sorted by priority and urgency."""
        active = self.get_active_goals()
        
        def score(g: Goal) -> float:
            s = g.priority.value * 100
            # Boost blocked goals
            if g.status == GoalStatus.BLOCKED:
                s += 50
            # Boost overdue goals
            if g.due_date:
                try:
                    due = datetime.fromisoformat(g.due_date)
                    days_left = (due - datetime.now()).days
                    if days_left < 0:
                        s += 200  # Overdue
                    elif days_left < 3:
                        s += 100  # Due soon
                except ValueError:
                    pass  # Invalid date format
            return s
            
        return sorted(active, key=score, reverse=True)
    
    def get_next_action(self) -> Optional[Goal]:
        """Get the highest priority actionable goal."""
        prioritized = self.get_prioritized_goals()
        for goal in prioritized:
            if goal.status != GoalStatus.BLOCKED:
                return goal
        return None
    
    # ========== Reporting ==========
    
    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all goals."""
        goals = list(self.goals.values())
        
        by_status = {}
        for status in GoalStatus:
            by_status[status.value] = len([g for g in goals if g.status == status])
            
        by_priority = {}
        for priority in GoalPriority:
            by_priority[priority.name.lower()] = len([g for g in goals if g.priority == priority])
            
        return {
            'total': len(goals),
            'by_status': by_status,
            'by_priority': by_priority,
            'active': len(self.get_active_goals()),
            'completed_today': len([
                g for g in goals 
                if g.completed_at and 
                datetime.fromtimestamp(g.completed_at).date() == datetime.now().date()
            ]),
            'overdue': len([
                g for g in goals
                if g.due_date and g.status not in (GoalStatus.COMPLETED, GoalStatus.ABANDONED)
                and datetime.fromisoformat(g.due_date) < datetime.now()
            ])
        }
    
    def describe(self) -> str:
        """Get a natural language description of goals."""
        summary = self.get_summary()
        active = self.get_active_goals()
        next_goal = self.get_next_action()
        
        parts = [f"You have {summary['total']} goals total."]
        
        if summary['active'] > 0:
            parts.append(f"{summary['active']} are active.")
            
        if summary['by_status']['completed'] > 0:
            parts.append(f"{summary['by_status']['completed']} completed.")
            
        if summary['overdue'] > 0:
            parts.append(f"{summary['overdue']} overdue!")
            
        if next_goal:
            parts.append(f"Top priority: {next_goal.title}")
            
        return " ".join(parts)
    
    def delete_goal(self, goal_id: str) -> bool:
        """Delete a goal."""
        if goal_id not in self.goals:
            return False
            
        goal = self.goals[goal_id]
        
        # Remove from parent
        if goal.parent_id and goal.parent_id in self.goals:
            parent = self.goals[goal.parent_id]
            if goal_id in parent.subgoal_ids:
                parent.subgoal_ids.remove(goal_id)
                
        # Delete subgoals
        for sid in goal.subgoal_ids:
            if sid in self.goals:
                del self.goals[sid]
                
        del self.goals[goal_id]
        self._save()
        return True


# Singleton
_tracker: Optional[GoalTracker] = None

def get_tracker() -> GoalTracker:
    """Get the goal tracker singleton."""
    global _tracker
    if _tracker is None:
        _tracker = GoalTracker()
    return _tracker


# Convenience functions
def add_goal(title: str, **kwargs) -> Goal:
    """Quick add a goal."""
    return get_tracker().create_goal(title, **kwargs)

def complete(goal_id: str) -> Optional[Goal]:
    """Mark goal complete."""
    return get_tracker().complete_goal(goal_id)

def my_goals() -> list[Goal]:
    """Get active goals."""
    return get_tracker().get_active_goals()

def next_action() -> Optional[Goal]:
    """Get next priority action."""
    return get_tracker().get_next_action()

def goal_status() -> str:
    """Get goal status description."""
    return get_tracker().describe()
