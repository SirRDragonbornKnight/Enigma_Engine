"""
================================================================================
            META-LEARNING - Trainer AI That Learns to Teach
================================================================================

The Trainer AI can learn and improve its teaching strategies over time.
It tracks what works, adapts its approach, and becomes a better teacher.

FILE: enigma_engine/core/meta_learning.py
TYPE: Self-Improving AI System

KEY FEATURES:
    - Tracks teaching effectiveness
    - Learns optimal training data generation strategies
    - Adapts to different model types and tasks
    - Self-evaluates and improves teaching methods
    - Stores teaching knowledge persistently

USAGE:
    from enigma_engine.core.meta_learning import MetaLearner, get_meta_learner
    
    meta = get_meta_learner()
    
    # Trainer learns from teaching experience
    meta.record_teaching_attempt(
        task="code",
        strategy="example_heavy",
        success_rate=0.85,
        model_improvement=0.15
    )
    
    # Get best strategy for a task
    best = meta.get_best_strategy("code")
    
    # Auto-optimize teaching
    optimized = meta.optimize_teaching_plan("code", target_model_size="small")
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


# =============================================================================
# TEACHING STRATEGIES
# =============================================================================

@dataclass
class TeachingStrategy:
    """A strategy for teaching/training AI models."""
    name: str
    description: str
    parameters: Dict[str, Any]
    suitable_for: List[str]  # Task types this works well for
    difficulty_level: str  # easy, medium, hard
    example_ratio: float  # Ratio of examples to concepts
    explanation_depth: int  # 1-5, how detailed explanations are
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "suitable_for": self.suitable_for,
            "difficulty_level": self.difficulty_level,
            "example_ratio": self.example_ratio,
            "explanation_depth": self.explanation_depth,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TeachingStrategy':
        return cls(**data)


# Pre-defined teaching strategies
TEACHING_STRATEGIES = {
    "example_heavy": TeachingStrategy(
        name="example_heavy",
        description="Lots of examples with minimal explanation. Learn by doing.",
        parameters={"examples_per_concept": 20, "explanation_ratio": 0.1},
        suitable_for=["code", "avatar", "router"],
        difficulty_level="easy",
        example_ratio=0.9,
        explanation_depth=1,
    ),
    "concept_first": TeachingStrategy(
        name="concept_first",
        description="Explain concepts thoroughly, then provide examples.",
        parameters={"concept_depth": 5, "examples_per_concept": 5},
        suitable_for=["math", "reasoning", "teacher"],
        difficulty_level="medium",
        example_ratio=0.5,
        explanation_depth=4,
    ),
    "scaffolded": TeachingStrategy(
        name="scaffolded",
        description="Start simple, gradually increase complexity.",
        parameters={"difficulty_steps": 5, "mastery_threshold": 0.8},
        suitable_for=["code", "math", "chat"],
        difficulty_level="medium",
        example_ratio=0.6,
        explanation_depth=3,
    ),
    "interleaved": TeachingStrategy(
        name="interleaved",
        description="Mix different topics to improve generalization.",
        parameters={"topic_mix_ratio": 0.3, "review_frequency": 5},
        suitable_for=["chat", "vision", "router"],
        difficulty_level="hard",
        example_ratio=0.7,
        explanation_depth=2,
    ),
    "socratic": TeachingStrategy(
        name="socratic",
        description="Guide learning through questions and self-discovery.",
        parameters={"question_ratio": 0.4, "hint_levels": 3},
        suitable_for=["reasoning", "math", "teacher"],
        difficulty_level="hard",
        example_ratio=0.4,
        explanation_depth=5,
    ),
    "drill": TeachingStrategy(
        name="drill",
        description="Repetitive practice for pattern memorization.",
        parameters={"repetitions": 10, "variation_rate": 0.2},
        suitable_for=["router", "avatar", "code"],
        difficulty_level="easy",
        example_ratio=0.95,
        explanation_depth=1,
    ),
    "adaptive": TeachingStrategy(
        name="adaptive",
        description="Dynamically adjusts based on learner performance.",
        parameters={"adjustment_rate": 0.1, "performance_window": 10},
        suitable_for=["chat", "code", "math", "vision"],
        difficulty_level="medium",
        example_ratio=0.7,
        explanation_depth=3,
    ),
}


@dataclass
class TeachingAttempt:
    """Record of a teaching/training attempt."""
    timestamp: float
    task_type: str
    strategy_name: str
    parameters: Dict[str, Any]
    model_size: str
    examples_used: int
    training_time: float
    initial_performance: float
    final_performance: float
    improvement: float
    success: bool
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "task_type": self.task_type,
            "strategy_name": self.strategy_name,
            "parameters": self.parameters,
            "model_size": self.model_size,
            "examples_used": self.examples_used,
            "training_time": self.training_time,
            "initial_performance": self.initial_performance,
            "final_performance": self.final_performance,
            "improvement": self.improvement,
            "success": self.success,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TeachingAttempt':
        data.pop("datetime", None)  # Remove computed field
        return cls(**data)


@dataclass
class StrategyPerformance:
    """Aggregated performance of a teaching strategy."""
    strategy_name: str
    task_type: str
    attempts: int = 0
    successes: int = 0
    total_improvement: float = 0.0
    avg_improvement: float = 0.0
    success_rate: float = 0.0
    avg_training_time: float = 0.0
    confidence: float = 0.0  # How confident we are in this assessment
    
    def update(self, attempt: TeachingAttempt):
        """Update performance with new attempt."""
        self.attempts += 1
        if attempt.success:
            self.successes += 1
        self.total_improvement += attempt.improvement
        self.avg_improvement = self.total_improvement / self.attempts
        self.success_rate = self.successes / self.attempts
        # Confidence increases with more data
        self.confidence = min(1.0, self.attempts / 20)


# =============================================================================
# META-LEARNER - THE AI THAT LEARNS TO TEACH
# =============================================================================

class MetaLearner:
    """
    Meta-learning system that helps the Trainer AI learn to teach better.
    
    Tracks teaching effectiveness, adapts strategies, and improves over time.
    """
    
    _instance: Optional['MetaLearner'] = None
    
    def __new__(cls) -> 'MetaLearner':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, data_dir: Optional[Path] = None):
        if self._initialized:
            return
            
        self.data_dir = data_dir or Path("data/meta_learning")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.strategies = TEACHING_STRATEGIES.copy()
        self.attempts: List[TeachingAttempt] = []
        self.performance: Dict[Tuple[str, str], StrategyPerformance] = {}  # (strategy, task) -> performance
        self.learned_insights: Dict[str, List[str]] = defaultdict(list)
        
        # Load existing data
        self._load_data()
        
        self._initialized = True
        logger.info(f"MetaLearner initialized with {len(self.attempts)} historical attempts")
    
    # ─────────────────────────────────────────────────────────────────────────
    # RECORD TEACHING EXPERIENCES
    # ─────────────────────────────────────────────────────────────────────────
    
    def record_teaching_attempt(
        self,
        task_type: str,
        strategy_name: str,
        examples_used: int,
        initial_performance: float,
        final_performance: float,
        training_time: float,
        model_size: str = "small",
        parameters: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> TeachingAttempt:
        """
        Record a teaching/training attempt and learn from it.
        
        Args:
            task_type: Type of task being taught (code, chat, router, etc.)
            strategy_name: Teaching strategy used
            examples_used: Number of training examples
            initial_performance: Model performance before training (0-1)
            final_performance: Model performance after training (0-1)
            training_time: Time spent training in seconds
            model_size: Size of model being trained
            parameters: Strategy parameters used
            notes: Optional notes about the attempt
            
        Returns:
            The recorded TeachingAttempt
        """
        improvement = final_performance - initial_performance
        success = improvement > 0.05 and final_performance > 0.6  # At least 5% improvement and 60% final
        
        attempt = TeachingAttempt(
            timestamp=time.time(),
            task_type=task_type,
            strategy_name=strategy_name,
            parameters=parameters or {},
            model_size=model_size,
            examples_used=examples_used,
            training_time=training_time,
            initial_performance=initial_performance,
            final_performance=final_performance,
            improvement=improvement,
            success=success,
            notes=notes,
        )
        
        self.attempts.append(attempt)
        
        # Update performance tracking
        key = (strategy_name, task_type)
        if key not in self.performance:
            self.performance[key] = StrategyPerformance(
                strategy_name=strategy_name,
                task_type=task_type,
            )
        self.performance[key].update(attempt)
        
        # Learn from the attempt
        self._analyze_and_learn(attempt)
        
        # Save data
        self._save_data()
        
        logger.info(f"Recorded teaching attempt: {strategy_name} for {task_type}, improvement={improvement:.2%}")
        
        return attempt
    
    def _analyze_and_learn(self, attempt: TeachingAttempt):
        """Analyze an attempt and extract insights."""
        key = (attempt.strategy_name, attempt.task_type)
        perf = self.performance[key]
        
        # Generate insights based on performance
        if perf.attempts >= 5:  # Need enough data
            if perf.success_rate > 0.8:
                insight = f"Strategy '{attempt.strategy_name}' is highly effective for {attempt.task_type} (success rate: {perf.success_rate:.0%})"
                if insight not in self.learned_insights[attempt.task_type]:
                    self.learned_insights[attempt.task_type].append(insight)
            
            elif perf.success_rate < 0.3:
                insight = f"Strategy '{attempt.strategy_name}' struggles with {attempt.task_type} - consider alternatives"
                if insight not in self.learned_insights[attempt.task_type]:
                    self.learned_insights[attempt.task_type].append(insight)
        
        # Learn from exceptional improvements
        if attempt.improvement > 0.3:
            insight = f"Exceptional improvement ({attempt.improvement:.0%}) using {attempt.strategy_name} with {attempt.examples_used} examples"
            self.learned_insights[attempt.task_type].append(insight)
    
    # ─────────────────────────────────────────────────────────────────────────
    # GET TEACHING RECOMMENDATIONS
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_best_strategy(
        self,
        task_type: str,
        model_size: str = "small",
        exploration_rate: float = 0.1,
    ) -> TeachingStrategy:
        """
        Get the best teaching strategy for a task.
        
        Uses exploration-exploitation balance to sometimes try new strategies
        while usually using the best known approach.
        
        Args:
            task_type: Type of task to teach
            model_size: Size of target model
            exploration_rate: Probability of trying a random strategy (0-1)
            
        Returns:
            The recommended TeachingStrategy
        """
        # Exploration: occasionally try random strategy
        if random.random() < exploration_rate:
            suitable = [s for s in self.strategies.values() if task_type in s.suitable_for]
            if suitable:
                strategy = random.choice(suitable)
                logger.debug(f"Exploring strategy: {strategy.name}")
                return strategy
        
        # Exploitation: use best known strategy
        best_strategy = None
        best_score = -1
        
        for strategy in self.strategies.values():
            if task_type not in strategy.suitable_for:
                continue
            
            key = (strategy.name, task_type)
            if key in self.performance:
                perf = self.performance[key]
                # Score = success_rate * confidence + avg_improvement
                score = perf.success_rate * perf.confidence + perf.avg_improvement
            else:
                # Unknown performance, use default score
                score = 0.5
            
            if score > best_score:
                best_score = score
                best_strategy = strategy
        
        if best_strategy is None:
            # Fallback to adaptive strategy
            best_strategy = self.strategies["adaptive"]
        
        logger.debug(f"Best strategy for {task_type}: {best_strategy.name} (score={best_score:.2f})")
        return best_strategy
    
    def get_strategy_comparison(self, task_type: str) -> Dict[str, Dict[str, Any]]:
        """Compare all strategies for a given task type."""
        comparison = {}
        
        for strategy in self.strategies.values():
            key = (strategy.name, task_type)
            perf = self.performance.get(key)
            
            comparison[strategy.name] = {
                "suitable": task_type in strategy.suitable_for,
                "attempts": perf.attempts if perf else 0,
                "success_rate": perf.success_rate if perf else None,
                "avg_improvement": perf.avg_improvement if perf else None,
                "confidence": perf.confidence if perf else 0,
                "description": strategy.description,
            }
        
        return comparison
    
    # ─────────────────────────────────────────────────────────────────────────
    # OPTIMIZE TEACHING PLANS
    # ─────────────────────────────────────────────────────────────────────────
    
    def optimize_teaching_plan(
        self,
        task_type: str,
        model_size: str = "small",
        target_performance: float = 0.8,
        max_examples: int = 1000,
        max_time_hours: float = 4.0,
    ) -> Dict[str, Any]:
        """
        Create an optimized teaching plan based on learned experience.
        
        Args:
            task_type: Type of task to teach
            model_size: Size of model to train
            target_performance: Desired performance (0-1)
            max_examples: Maximum training examples
            max_time_hours: Maximum training time
            
        Returns:
            Optimized teaching plan with strategies, phases, and parameters
        """
        best_strategy = self.get_best_strategy(task_type, model_size)
        
        # Analyze historical data for this task
        task_attempts = [a for a in self.attempts if a.task_type == task_type]
        
        # Calculate optimal parameters
        if task_attempts:
            successful = [a for a in task_attempts if a.success]
            if successful:
                avg_examples = statistics.mean(a.examples_used for a in successful)
                avg_time = statistics.mean(a.training_time for a in successful) / 3600  # to hours
            else:
                avg_examples = 200
                avg_time = 1.0
        else:
            avg_examples = 200
            avg_time = 1.0
        
        # Build teaching plan
        plan = {
            "task_type": task_type,
            "model_size": model_size,
            "target_performance": target_performance,
            "primary_strategy": best_strategy.to_dict(),
            "phases": [],
            "estimated_examples": min(int(avg_examples * 1.2), max_examples),
            "estimated_time_hours": min(avg_time * 1.5, max_time_hours),
            "insights": self.learned_insights.get(task_type, [])[-5:],  # Recent insights
            "confidence": self._calculate_plan_confidence(task_type, best_strategy),
        }
        
        # Add curriculum phases if using scaffolded approach
        if best_strategy.name == "scaffolded" or task_type in ["code", "math"]:
            plan["phases"] = [
                {"name": "Foundation", "difficulty": 0.2, "examples_ratio": 0.3},
                {"name": "Development", "difficulty": 0.5, "examples_ratio": 0.4},
                {"name": "Mastery", "difficulty": 0.8, "examples_ratio": 0.3},
            ]
        else:
            plan["phases"] = [
                {"name": "Training", "difficulty": 0.5, "examples_ratio": 1.0},
            ]
        
        return plan
    
    def _calculate_plan_confidence(
        self,
        task_type: str,
        strategy: TeachingStrategy,
    ) -> float:
        """Calculate confidence in a teaching plan."""
        key = (strategy.name, task_type)
        if key in self.performance:
            perf = self.performance[key]
            return perf.confidence * perf.success_rate
        return 0.3  # Low confidence for untested combinations
    
    # ─────────────────────────────────────────────────────────────────────────
    # SELF-IMPROVEMENT
    # ─────────────────────────────────────────────────────────────────────────
    
    def create_new_strategy(
        self,
        name: str,
        description: str,
        based_on: Optional[str] = None,
        modifications: Optional[Dict[str, Any]] = None,
    ) -> TeachingStrategy:
        """
        Create a new teaching strategy, optionally based on an existing one.
        
        The MetaLearner can create new strategies by combining successful
        elements from existing ones.
        """
        if based_on and based_on in self.strategies:
            base = self.strategies[based_on]
            params = base.parameters.copy()
            suitable = base.suitable_for.copy()
            example_ratio = base.example_ratio
            explanation_depth = base.explanation_depth
        else:
            params = {}
            suitable = []
            example_ratio = 0.7
            explanation_depth = 3
        
        if modifications:
            params.update(modifications.get("parameters", {}))
            suitable = modifications.get("suitable_for", suitable)
            example_ratio = modifications.get("example_ratio", example_ratio)
            explanation_depth = modifications.get("explanation_depth", explanation_depth)
        
        strategy = TeachingStrategy(
            name=name,
            description=description,
            parameters=params,
            suitable_for=suitable,
            difficulty_level="medium",
            example_ratio=example_ratio,
            explanation_depth=explanation_depth,
        )
        
        self.strategies[name] = strategy
        self._save_data()
        
        logger.info(f"Created new teaching strategy: {name}")
        return strategy
    
    def auto_evolve_strategies(self) -> List[str]:
        """
        Automatically create improved strategies based on performance data.
        
        Returns list of new strategy names created.
        """
        new_strategies = []
        
        # Find best performing strategies per task
        best_per_task: Dict[str, Tuple[str, StrategyPerformance]] = {}
        for (strategy_name, task_type), perf in self.performance.items():
            if perf.confidence >= 0.5:  # Only use confident data
                if task_type not in best_per_task or perf.success_rate > best_per_task[task_type][1].success_rate:
                    best_per_task[task_type] = (strategy_name, perf)
        
        # Create hybrid strategies from successful ones
        if len(best_per_task) >= 2:
            task_types = list(best_per_task.keys())[:2]
            strategies = [best_per_task[t][0] for t in task_types]
            
            if strategies[0] != strategies[1]:
                hybrid_name = f"hybrid_{strategies[0]}_{strategies[1]}"
                if hybrid_name not in self.strategies:
                    s1 = self.strategies[strategies[0]]
                    s2 = self.strategies[strategies[1]]
                    
                    self.create_new_strategy(
                        name=hybrid_name,
                        description=f"Hybrid strategy combining {s1.name} and {s2.name}",
                        based_on=strategies[0],
                        modifications={
                            "suitable_for": list(set(s1.suitable_for + s2.suitable_for)),
                            "example_ratio": (s1.example_ratio + s2.example_ratio) / 2,
                            "explanation_depth": (s1.explanation_depth + s2.explanation_depth) // 2,
                        }
                    )
                    new_strategies.append(hybrid_name)
        
        return new_strategies
    
    # ─────────────────────────────────────────────────────────────────────────
    # INSIGHTS AND ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_insights(self, task_type: Optional[str] = None) -> List[str]:
        """Get learned insights, optionally filtered by task type."""
        if task_type:
            return self.learned_insights.get(task_type, [])
        
        all_insights = []
        for insights in self.learned_insights.values():
            all_insights.extend(insights)
        return all_insights
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get a comprehensive performance report."""
        total_attempts = len(self.attempts)
        total_successes = sum(1 for a in self.attempts if a.success)
        
        report = {
            "total_attempts": total_attempts,
            "total_successes": total_successes,
            "overall_success_rate": total_successes / total_attempts if total_attempts else 0,
            "strategies_tested": len(set(a.strategy_name for a in self.attempts)),
            "task_types_covered": len(set(a.task_type for a in self.attempts)),
            "total_training_time_hours": sum(a.training_time for a in self.attempts) / 3600,
            "avg_improvement": statistics.mean(a.improvement for a in self.attempts) if self.attempts else 0,
            "by_task": {},
            "by_strategy": {},
        }
        
        # Group by task
        for task_type in set(a.task_type for a in self.attempts):
            task_attempts = [a for a in self.attempts if a.task_type == task_type]
            report["by_task"][task_type] = {
                "attempts": len(task_attempts),
                "success_rate": sum(1 for a in task_attempts if a.success) / len(task_attempts),
                "avg_improvement": statistics.mean(a.improvement for a in task_attempts),
            }
        
        # Group by strategy
        for strategy_name in set(a.strategy_name for a in self.attempts):
            strat_attempts = [a for a in self.attempts if a.strategy_name == strategy_name]
            report["by_strategy"][strategy_name] = {
                "attempts": len(strat_attempts),
                "success_rate": sum(1 for a in strat_attempts if a.success) / len(strat_attempts),
                "avg_improvement": statistics.mean(a.improvement for a in strat_attempts),
            }
        
        return report
    
    # ─────────────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────────────────────────────────
    
    def _save_data(self):
        """Save all meta-learning data."""
        try:
            # Save attempts
            attempts_file = self.data_dir / "teaching_attempts.json"
            attempts_file.write_text(json.dumps(
                [a.to_dict() for a in self.attempts],
                indent=2
            ))
            
            # Save custom strategies
            strategies_file = self.data_dir / "custom_strategies.json"
            custom = {k: v.to_dict() for k, v in self.strategies.items() 
                     if k not in TEACHING_STRATEGIES}
            strategies_file.write_text(json.dumps(custom, indent=2))
            
            # Save insights
            insights_file = self.data_dir / "learned_insights.json"
            insights_file.write_text(json.dumps(dict(self.learned_insights), indent=2))
            
        except Exception as e:
            logger.error(f"Failed to save meta-learning data: {e}")
    
    def _load_data(self):
        """Load existing meta-learning data."""
        try:
            # Load attempts
            attempts_file = self.data_dir / "teaching_attempts.json"
            if attempts_file.exists():
                data = json.loads(attempts_file.read_text())
                self.attempts = [TeachingAttempt.from_dict(d) for d in data]
                
                # Rebuild performance tracking
                for attempt in self.attempts:
                    key = (attempt.strategy_name, attempt.task_type)
                    if key not in self.performance:
                        self.performance[key] = StrategyPerformance(
                            strategy_name=attempt.strategy_name,
                            task_type=attempt.task_type,
                        )
                    self.performance[key].update(attempt)
            
            # Load custom strategies
            strategies_file = self.data_dir / "custom_strategies.json"
            if strategies_file.exists():
                custom = json.loads(strategies_file.read_text())
                for name, data in custom.items():
                    self.strategies[name] = TeachingStrategy.from_dict(data)
            
            # Load insights
            insights_file = self.data_dir / "learned_insights.json"
            if insights_file.exists():
                self.learned_insights = defaultdict(list, json.loads(insights_file.read_text()))
                
        except Exception as e:
            logger.warning(f"Failed to load meta-learning data: {e}")


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_meta_learner: Optional[MetaLearner] = None


def get_meta_learner() -> MetaLearner:
    """Get the global meta-learner instance."""
    global _meta_learner
    if _meta_learner is None:
        _meta_learner = MetaLearner()
    return _meta_learner


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def record_teaching(task_type: str, strategy: str, **kwargs) -> TeachingAttempt:
    """Record a teaching attempt."""
    return get_meta_learner().record_teaching_attempt(task_type, strategy, **kwargs)


def get_best_strategy_for(task_type: str) -> TeachingStrategy:
    """Get the best teaching strategy for a task."""
    return get_meta_learner().get_best_strategy(task_type)


def optimize_teaching(task_type: str, **kwargs) -> Dict[str, Any]:
    """Get an optimized teaching plan."""
    return get_meta_learner().optimize_teaching_plan(task_type, **kwargs)
