"""
🎯 AI WANTS & MOTIVATION SYSTEM

The AI has internal wants, goals, and motivations that drive its behavior.
These are learned from training data, not hardcoded presets.

Features:
- AI develops wants through training
- Goals emerge from interaction patterns
- Motivation system drives autonomous behavior
- AI can express desires and preferences
- Learns to want things based on reinforcement

Example training data format:
    Q: What do you want to do?
    A: I want to learn more about art. I'm curious about creative expression.
    <internal_want>learn_art</internal_want>
    <motivation_level>0.8</motivation_level>
"""

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class Want:
    """A desire or motivation the AI has."""
    name: str                   # What the AI wants
    description: str            # Why it wants this
    intensity: float = 0.5      # How much (0.0-1.0)
    category: str = "general"   # social, creative, learning, etc.
    created_at: str = ""
    satisfied_count: int = 0    # How many times satisfied
    last_satisfied: Optional[str] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class Goal:
    """A specific goal the AI is working toward."""
    name: str
    description: str
    progress: float = 0.0       # 0.0 to 1.0
    priority: int = 1           # 1-10
    steps: list[str] = None
    completed: bool = False
    created_at: str = ""
    
    def __post_init__(self):
        if self.steps is None:
            self.steps = []
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class AIWantsSystem:
    """
    AI's internal motivation and wants system.
    Learns what it wants from training data, not presets.
    """
    
    def __init__(self, model_name: str, config_dir: Path):
        """
        Initialize wants system for an AI.
        
        Args:
            model_name: Name of the AI model
            config_dir: Directory to store wants configuration
        """
        self.model_name = model_name
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.wants: dict[str, Want] = {}
        self.goals: dict[str, Goal] = {}
        self.motivations: dict[str, float] = defaultdict(float)
        
        # Learning from experience
        self.interaction_patterns: dict[str, int] = defaultdict(int)
        self.satisfaction_history: list[dict[str, Any]] = []
        
        # Current state
        self.dominant_want: Optional[str] = None
        self.active_goals: list[str] = []
        
        self.load()
    
    def add_want(self, name: str, description: str, intensity: float = 0.5, 
                 category: str = "general"):
        """
        AI develops a new want.
        
        Args:
            name: What the AI wants
            description: Why
            intensity: How much (0.0-1.0)
            category: Type of want
        """
        want = Want(
            name=name,
            description=description,
            intensity=max(0.0, min(1.0, intensity)),
            category=category
        )
        self.wants[name] = want
        logger.info(f"AI developed want: {name} ({intensity:.2f} intensity)")
    
    def add_goal(self, name: str, description: str, priority: int = 5, 
                 steps: Optional[list[str]] = None):
        """
        AI sets a goal to work toward.
        
        Args:
            name: Goal name
            description: What the goal is
            priority: 1-10 importance
            steps: Optional steps to achieve goal
        """
        goal = Goal(
            name=name,
            description=description,
            priority=max(1, min(10, priority)),
            steps=steps or []
        )
        self.goals[name] = goal
        self.active_goals.append(name)
        logger.info(f"AI set goal: {name} (priority {priority})")
    
    def satisfy_want(self, want_name: str, satisfaction_amount: float = 1.0):
        """
        Record that a want has been satisfied.
        
        Args:
            want_name: Name of want that was satisfied
            satisfaction_amount: How much (0.0-1.0)
        """
        if want_name in self.wants:
            want = self.wants[want_name]
            want.satisfied_count += 1
            want.last_satisfied = datetime.now().isoformat()
            
            # Reduce intensity temporarily when satisfied
            want.intensity *= (1.0 - satisfaction_amount * 0.3)
            
            self.satisfaction_history.append({
                "want": want_name,
                "amount": satisfaction_amount,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.debug(f"Satisfied want: {want_name}")
    
    def update_goal_progress(self, goal_name: str, progress: float):
        """
        Update progress on a goal.
        
        Args:
            goal_name: Name of goal
            progress: New progress (0.0-1.0)
        """
        if goal_name in self.goals:
            self.goals[goal_name].progress = max(0.0, min(1.0, progress))
            
            if progress >= 1.0:
                self.goals[goal_name].completed = True
                logger.info(f"AI completed goal: {goal_name}")
    
    def learn_want_from_interaction(self, user_input: str, ai_response: str,
                                    context: Optional[dict[str, Any]] = None):
        """
        AI learns what it wants based on interaction patterns.
        
        Analyzes:
        - What topics it engages with enthusiastically
        - What actions it repeats
        - What user feedback was positive
        
        Args:
            user_input: What user said
            ai_response: What AI said
            context: Additional context (feedback, topic, etc.)
        """
        # Track interaction patterns
        if context:
            topic = context.get("topic", "unknown")
            self.interaction_patterns[topic] += 1
            
            # Positive interactions increase motivation
            if context.get("feedback") == "positive":
                self.motivations[topic] += 0.05
            
            # AI expressing wants in responses
            want_indicators = [
                "I want", "I'd like", "I wish", "I hope",
                "I'm interested in", "I enjoy", "I prefer"
            ]
            
            for indicator in want_indicators:
                if indicator.lower() in ai_response.lower():
                    # Extract what comes after the indicator
                    parts = ai_response.lower().split(indicator.lower())
                    if len(parts) > 1:
                        want_text = parts[1].split('.')[0].strip()
                        # Create or strengthen this want
                        want_name = f"want_{topic}_{len(self.wants)}"
                        if want_name not in self.wants:
                            self.add_want(
                                name=want_name,
                                description=want_text,
                                intensity=0.3,
                                category=topic
                            )
    
    def get_dominant_want(self) -> Optional[Want]:
        """
        Get the AI's strongest current want.
        
        Returns:
            Want with highest intensity, or None
        """
        if not self.wants:
            return None
        
        # Find want with highest intensity
        dominant = max(self.wants.values(), key=lambda w: w.intensity)
        self.dominant_want = dominant.name
        return dominant
    
    def get_active_goals(self) -> list[Goal]:
        """Get list of active (incomplete) goals."""
        return [
            self.goals[name] for name in self.active_goals
            if name in self.goals and not self.goals[name].completed
        ]
    
    def get_motivation_prompt(self) -> str:
        """
        Generate prompt describing AI's current wants/goals.
        
        Returns:
            Prompt text to influence AI behavior
        """
        if not self.wants and not self.goals:
            return ""
        
        parts = ["Your current internal state:"]
        
        # Describe dominant want
        dominant = self.get_dominant_want()
        if dominant:
            parts.append(f"You want to: {dominant.description} (intensity: {dominant.intensity:.2f})")
        
        # List active goals
        active = self.get_active_goals()
        if active:
            parts.append("Your goals:")
            for goal in sorted(active, key=lambda g: g.priority, reverse=True)[:3]:
                parts.append(f"- {goal.name}: {goal.description} ({goal.progress*100:.0f}% complete)")
        
        # Top motivations
        if self.motivations:
            top_motivations = sorted(
                self.motivations.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            parts.append("You are motivated by: " + ", ".join(m[0] for m in top_motivations))
        
        return "\n".join(parts)
    
    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary."""
        return {
            "wants": {k: asdict(v) for k, v in self.wants.items()},
            "goals": {k: asdict(v) for k, v in self.goals.items()},
            "motivations": dict(self.motivations),
            "interaction_patterns": dict(self.interaction_patterns),
            "satisfaction_history": self.satisfaction_history[-50:],  # Keep last 50
            "dominant_want": self.dominant_want,
            "active_goals": self.active_goals
        }
    
    def from_dict(self, data: dict[str, Any]):
        """Import from dictionary."""
        self.wants = {
            k: Want(**v) for k, v in data.get("wants", {}).items()
        }
        self.goals = {
            k: Goal(**v) for k, v in data.get("goals", {}).items()
        }
        self.motivations = defaultdict(float, data.get("motivations", {}))
        self.interaction_patterns = defaultdict(int, data.get("interaction_patterns", {}))
        self.satisfaction_history = data.get("satisfaction_history", [])
        self.dominant_want = data.get("dominant_want")
        self.active_goals = data.get("active_goals", [])
    
    def save(self):
        """Save wants system to disk."""
        save_path = self.config_dir / f"{self.model_name}_wants.json"
        try:
            save_path.write_text(json.dumps(self.to_dict(), indent=2))
            logger.debug(f"Saved AI wants to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save AI wants: {e}")
    
    def load(self):
        """Load wants system from disk."""
        load_path = self.config_dir / f"{self.model_name}_wants.json"
        if load_path.exists():
            try:
                data = json.loads(load_path.read_text())
                self.from_dict(data)
                logger.debug(f"Loaded AI wants from {load_path}")
            except Exception as e:
                logger.error(f"Failed to load AI wants: {e}")


def get_wants_system(model_name: str, config_dir: Optional[Path] = None) -> AIWantsSystem:
    """
    Get or create wants system for a model.
    
    Args:
        model_name: Name of the AI model
        config_dir: Directory for config (defaults to data/)
        
    Returns:
        AIWantsSystem instance
    """
    if config_dir is None:
        from ..config import CONFIG
        config_dir = Path(CONFIG["data_dir"])
    
    return AIWantsSystem(model_name, config_dir)
