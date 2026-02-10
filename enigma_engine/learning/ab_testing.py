"""
A/B Testing System for Personality Traits
=========================================

Data-driven personality optimization through controlled experiments.
Test different personality configurations and measure user satisfaction.

Features:
- Define personality variants (A/B/n testing)
- Random or weighted assignment to variants
- Track user satisfaction per variant
- Statistical significance analysis
- Automatic promotion of winning variants

Usage:
    from enigma_engine.learning.ab_testing import PersonalityABTest
    
    test = PersonalityABTest("friendliness_test")
    
    # Define variants
    test.add_variant("formal", {"tone": "formal", "emoji_use": 0.0})
    test.add_variant("casual", {"tone": "casual", "emoji_use": 0.3})
    
    # Assign user to variant
    variant = test.get_variant(user_id)
    
    # Record outcome
    test.record_outcome(user_id, "positive")
    
    # Analyze results
    results = test.analyze()
    print(f"Best variant: {results['winner']}")
"""

import logging
import json
import hashlib
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import math

logger = logging.getLogger(__name__)


class OutcomeType(Enum):
    """Types of outcomes to track."""
    POSITIVE = "positive"  # Thumbs up, explicit positive feedback
    NEGATIVE = "negative"  # Thumbs down, explicit negative feedback
    NEUTRAL = "neutral"  # No feedback
    ENGAGEMENT = "engagement"  # Continued conversation
    ABANDONMENT = "abandonment"  # User left without feedback


class AssignmentStrategy(Enum):
    """How to assign users to variants."""
    RANDOM = "random"  # Pure random assignment
    WEIGHTED = "weighted"  # Based on variant weights
    DETERMINISTIC = "deterministic"  # Hash-based, same user = same variant
    EPSILON_GREEDY = "epsilon_greedy"  # Explore/exploit
    THOMPSON = "thompson"  # Thompson sampling


@dataclass
class PersonalityVariant:
    """A personality configuration variant."""
    name: str
    traits: Dict[str, Any]
    weight: float = 1.0  # Relative weight for weighted assignment
    description: str = ""
    
    # Tracking
    assignments: int = 0
    positive_outcomes: int = 0
    negative_outcomes: int = 0
    neutral_outcomes: int = 0
    engagement_score: float = 0.0
    total_conversations: int = 0
    total_messages: int = 0
    
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.positive_outcomes + self.negative_outcomes
        if total == 0:
            return 0.5  # Prior
        return self.positive_outcomes / total
    
    def engagement_rate(self) -> float:
        """Calculate engagement rate."""
        if self.total_conversations == 0:
            return 0.0
        return self.total_messages / self.total_conversations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "traits": self.traits,
            "weight": self.weight,
            "description": self.description,
            "assignments": self.assignments,
            "positive_outcomes": self.positive_outcomes,
            "negative_outcomes": self.negative_outcomes,
            "neutral_outcomes": self.neutral_outcomes,
            "engagement_score": self.engagement_score,
            "total_conversations": self.total_conversations,
            "total_messages": self.total_messages,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonalityVariant':
        """Create from dictionary."""
        variant = cls(
            name=data["name"],
            traits=data["traits"],
            weight=data.get("weight", 1.0),
            description=data.get("description", ""),
        )
        variant.assignments = data.get("assignments", 0)
        variant.positive_outcomes = data.get("positive_outcomes", 0)
        variant.negative_outcomes = data.get("negative_outcomes", 0)
        variant.neutral_outcomes = data.get("neutral_outcomes", 0)
        variant.engagement_score = data.get("engagement_score", 0.0)
        variant.total_conversations = data.get("total_conversations", 0)
        variant.total_messages = data.get("total_messages", 0)
        return variant


@dataclass
class TestConfig:
    """A/B test configuration."""
    name: str
    description: str = ""
    assignment_strategy: AssignmentStrategy = AssignmentStrategy.DETERMINISTIC
    
    # Epsilon-greedy params
    epsilon: float = 0.1  # Exploration rate
    
    # Statistical params
    min_samples_per_variant: int = 30  # Min samples before declaring winner
    significance_level: float = 0.05  # p-value threshold
    
    # Auto-promotion
    auto_promote: bool = False  # Automatically promote winner
    promote_threshold: float = 0.95  # Confidence threshold for promotion
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "assignment_strategy": self.assignment_strategy.value,
            "epsilon": self.epsilon,
            "min_samples_per_variant": self.min_samples_per_variant,
            "significance_level": self.significance_level,
            "auto_promote": self.auto_promote,
            "promote_threshold": self.promote_threshold,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestConfig':
        """Create from dictionary."""
        config = cls(
            name=data["name"],
            description=data.get("description", ""),
            epsilon=data.get("epsilon", 0.1),
            min_samples_per_variant=data.get("min_samples_per_variant", 30),
            significance_level=data.get("significance_level", 0.05),
            auto_promote=data.get("auto_promote", False),
            promote_threshold=data.get("promote_threshold", 0.95),
        )
        if "assignment_strategy" in data:
            config.assignment_strategy = AssignmentStrategy(data["assignment_strategy"])
        if "created_at" in data:
            config.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("expires_at"):
            config.expires_at = datetime.fromisoformat(data["expires_at"])
        return config


@dataclass
class UserAssignment:
    """Record of user's variant assignment."""
    user_id: str
    variant_name: str
    assigned_at: datetime = field(default_factory=datetime.now)
    outcomes: List[Tuple[OutcomeType, datetime]] = field(default_factory=list)
    messages_sent: int = 0


class PersonalityABTest:
    """
    A/B testing for personality configurations.
    
    Example:
        test = PersonalityABTest("tone_test")
        
        # Add variants
        test.add_variant("formal", {"tone": "formal", "verbosity": 0.8})
        test.add_variant("casual", {"tone": "casual", "verbosity": 0.5})
        
        # Get variant for user
        variant = test.get_variant("user_123")
        
        # Apply traits
        apply_personality_traits(variant.traits)
        
        # Record outcome after interaction
        test.record_outcome("user_123", OutcomeType.POSITIVE)
        
        # Check results
        results = test.analyze()
        if results["significant"]:
            print(f"Winner: {results['winner']}")
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[TestConfig] = None,
        storage_path: Optional[str] = None,
    ):
        """
        Initialize A/B test.
        
        Args:
            name: Test name
            config: Test configuration
            storage_path: Path to store test data
        """
        self.name = name
        self.config = config or TestConfig(name=name)
        
        self.variants: Dict[str, PersonalityVariant] = {}
        self.assignments: Dict[str, UserAssignment] = {}
        
        # Storage
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path("data/ab_tests") / f"{name}.json"
        
        # Load existing data
        self._load()
    
    def add_variant(
        self,
        name: str,
        traits: Dict[str, Any],
        weight: float = 1.0,
        description: str = "",
    ) -> PersonalityVariant:
        """
        Add a personality variant to test.
        
        Args:
            name: Variant name (e.g., "control", "variant_a")
            traits: Personality trait configuration
            weight: Relative weight for assignment
            description: Human-readable description
        
        Returns:
            Created variant
        """
        variant = PersonalityVariant(
            name=name,
            traits=traits,
            weight=weight,
            description=description,
        )
        self.variants[name] = variant
        self._save()
        
        logger.info(f"Added variant '{name}' to test '{self.name}'")
        return variant
    
    def get_variant(self, user_id: str) -> PersonalityVariant:
        """
        Get variant for a user.
        
        Assigns to variant if not already assigned.
        
        Args:
            user_id: User identifier
        
        Returns:
            Assigned variant
        """
        if not self.variants:
            raise ValueError("No variants defined. Add variants before getting assignment.")
        
        # Check existing assignment
        if user_id in self.assignments:
            variant_name = self.assignments[user_id].variant_name
            return self.variants[variant_name]
        
        # Assign to variant
        variant = self._select_variant(user_id)
        
        # Record assignment
        self.assignments[user_id] = UserAssignment(
            user_id=user_id,
            variant_name=variant.name,
        )
        variant.assignments += 1
        variant.total_conversations += 1
        
        self._save()
        logger.debug(f"Assigned user {user_id} to variant '{variant.name}'")
        
        return variant
    
    def _select_variant(self, user_id: str) -> PersonalityVariant:
        """Select variant based on assignment strategy."""
        variants = list(self.variants.values())
        
        if self.config.assignment_strategy == AssignmentStrategy.DETERMINISTIC:
            # Hash-based: same user always gets same variant
            hash_val = int(hashlib.md5(f"{self.name}:{user_id}".encode()).hexdigest(), 16)
            total_weight = sum(v.weight for v in variants)
            target = (hash_val % 1000) / 1000 * total_weight
            
            cumulative = 0
            for variant in variants:
                cumulative += variant.weight
                if target < cumulative:
                    return variant
            return variants[-1]
        
        elif self.config.assignment_strategy == AssignmentStrategy.RANDOM:
            return random.choice(variants)
        
        elif self.config.assignment_strategy == AssignmentStrategy.WEIGHTED:
            weights = [v.weight for v in variants]
            return random.choices(variants, weights=weights)[0]
        
        elif self.config.assignment_strategy == AssignmentStrategy.EPSILON_GREEDY:
            if random.random() < self.config.epsilon:
                # Explore: random
                return random.choice(variants)
            else:
                # Exploit: best performing
                return max(variants, key=lambda v: v.success_rate())
        
        elif self.config.assignment_strategy == AssignmentStrategy.THOMPSON:
            # Thompson sampling with Beta distribution
            samples = []
            for v in variants:
                alpha = v.positive_outcomes + 1  # Prior: 1
                beta = v.negative_outcomes + 1  # Prior: 1
                sample = random.betavariate(alpha, beta)
                samples.append((sample, v))
            return max(samples, key=lambda x: x[0])[1]
        
        return random.choice(variants)
    
    def record_outcome(
        self,
        user_id: str,
        outcome: OutcomeType,
    ) -> None:
        """
        Record outcome for a user's interaction.
        
        Args:
            user_id: User identifier
            outcome: Type of outcome
        """
        if user_id not in self.assignments:
            logger.warning(f"User {user_id} not assigned to any variant")
            return
        
        assignment = self.assignments[user_id]
        variant = self.variants[assignment.variant_name]
        
        # Record outcome
        assignment.outcomes.append((outcome, datetime.now()))
        
        # Update variant stats
        if outcome == OutcomeType.POSITIVE:
            variant.positive_outcomes += 1
        elif outcome == OutcomeType.NEGATIVE:
            variant.negative_outcomes += 1
        elif outcome == OutcomeType.NEUTRAL:
            variant.neutral_outcomes += 1
        elif outcome == OutcomeType.ENGAGEMENT:
            variant.engagement_score += 1
        
        self._save()
        logger.debug(f"Recorded {outcome.value} outcome for user {user_id}")
    
    def record_message(self, user_id: str) -> None:
        """Record a message in the conversation."""
        if user_id not in self.assignments:
            return
        
        assignment = self.assignments[user_id]
        assignment.messages_sent += 1
        
        variant = self.variants[assignment.variant_name]
        variant.total_messages += 1
        
        self._save()
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze test results.
        
        Returns:
            Analysis results including:
            - variants: Per-variant statistics
            - significant: Whether results are statistically significant
            - winner: Best performing variant (if significant)
            - p_value: Statistical p-value
            - confidence: Confidence in the winner
        """
        if len(self.variants) < 2:
            return {
                "error": "Need at least 2 variants for analysis",
                "variants": {},
                "significant": False,
            }
        
        # Calculate per-variant metrics
        variant_stats = {}
        for name, variant in self.variants.items():
            total = variant.positive_outcomes + variant.negative_outcomes
            rate = variant.success_rate()
            
            # Confidence interval (Wilson score)
            if total > 0:
                z = 1.96  # 95% CI
                denominator = 1 + z**2/total
                center = (rate + z**2/(2*total)) / denominator
                spread = z * math.sqrt((rate*(1-rate) + z**2/(4*total))/total) / denominator
                ci_low = max(0, center - spread)
                ci_high = min(1, center + spread)
            else:
                ci_low, ci_high = 0, 1
            
            variant_stats[name] = {
                "assignments": variant.assignments,
                "positive": variant.positive_outcomes,
                "negative": variant.negative_outcomes,
                "neutral": variant.neutral_outcomes,
                "success_rate": rate,
                "ci_95": (ci_low, ci_high),
                "engagement_rate": variant.engagement_rate(),
                "messages_per_conversation": variant.total_messages / max(variant.total_conversations, 1),
            }
        
        # Check for sufficient samples
        min_samples = self.config.min_samples_per_variant
        sufficient_data = all(
            (v.positive_outcomes + v.negative_outcomes) >= min_samples
            for v in self.variants.values()
        )
        
        # Statistical test (chi-square for proportions)
        significant = False
        p_value = 1.0
        winner = None
        confidence = 0.0
        
        if sufficient_data:
            p_value, significant = self._chi_square_test()
            
            if significant:
                # Find winner by success rate
                best_variant = max(self.variants.values(), key=lambda v: v.success_rate())
                winner = best_variant.name
                
                # Calculate confidence via Bayesian comparison
                confidence = self._bayesian_confidence(best_variant.name)
        
        # Auto-promote if configured
        if (self.config.auto_promote and 
            significant and 
            confidence >= self.config.promote_threshold):
            logger.info(f"Auto-promoting winner '{winner}' with {confidence:.1%} confidence")
        
        return {
            "test_name": self.name,
            "variants": variant_stats,
            "sufficient_data": sufficient_data,
            "significant": significant,
            "p_value": p_value,
            "winner": winner,
            "confidence": confidence,
            "total_users": len(self.assignments),
            "total_outcomes": sum(
                v.positive_outcomes + v.negative_outcomes
                for v in self.variants.values()
            ),
        }
    
    def _chi_square_test(self) -> Tuple[float, bool]:
        """Perform chi-square test for independence."""
        # Build contingency table
        observed = []
        for variant in self.variants.values():
            observed.append([variant.positive_outcomes, variant.negative_outcomes])
        
        # Calculate expected frequencies
        total_pos = sum(row[0] for row in observed)
        total_neg = sum(row[1] for row in observed)
        total = total_pos + total_neg
        
        if total == 0:
            return 1.0, False
        
        chi_square = 0
        for i, variant in enumerate(self.variants.values()):
            row_total = observed[i][0] + observed[i][1]
            expected_pos = row_total * total_pos / total
            expected_neg = row_total * total_neg / total
            
            if expected_pos > 0:
                chi_square += (observed[i][0] - expected_pos)**2 / expected_pos
            if expected_neg > 0:
                chi_square += (observed[i][1] - expected_neg)**2 / expected_neg
        
        # Degrees of freedom
        df = len(self.variants) - 1
        
        # Approximate p-value using chi-square distribution
        # Using simplified approximation
        p_value = math.exp(-chi_square / 2) if chi_square < 20 else 0.0
        
        return p_value, p_value < self.config.significance_level
    
    def _bayesian_confidence(self, variant_name: str) -> float:
        """Calculate Bayesian confidence that variant is best."""
        target = self.variants[variant_name]
        others = [v for v in self.variants.values() if v.name != variant_name]
        
        # Monte Carlo simulation
        n_samples = 10000
        wins = 0
        
        for _ in range(n_samples):
            target_sample = random.betavariate(
                target.positive_outcomes + 1,
                target.negative_outcomes + 1
            )
            
            is_best = True
            for other in others:
                other_sample = random.betavariate(
                    other.positive_outcomes + 1,
                    other.negative_outcomes + 1
                )
                if other_sample >= target_sample:
                    is_best = False
                    break
            
            if is_best:
                wins += 1
        
        return wins / n_samples
    
    def get_winning_traits(self) -> Optional[Dict[str, Any]]:
        """Get traits of the winning variant."""
        results = self.analyze()
        if results.get("winner"):
            return self.variants[results["winner"]].traits
        return None
    
    def promote_winner(self) -> Optional[str]:
        """
        Promote the winning variant as the new default.
        
        Returns name of promoted variant or None if no clear winner.
        """
        results = self.analyze()
        
        if not results.get("significant"):
            logger.warning("Cannot promote: results not significant")
            return None
        
        winner = results["winner"]
        if not winner:
            return None
        
        logger.info(f"Promoting variant '{winner}' as new default")
        
        # Return winner name - integration code should apply the traits
        return winner
    
    def reset(self) -> None:
        """Reset all test data."""
        for variant in self.variants.values():
            variant.assignments = 0
            variant.positive_outcomes = 0
            variant.negative_outcomes = 0
            variant.neutral_outcomes = 0
            variant.engagement_score = 0.0
            variant.total_conversations = 0
            variant.total_messages = 0
        
        self.assignments.clear()
        self._save()
        logger.info(f"Reset test '{self.name}'")
    
    def _save(self) -> None:
        """Save test data to file."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "config": self.config.to_dict(),
            "variants": {name: v.to_dict() for name, v in self.variants.items()},
            "assignments": {
                uid: {
                    "user_id": a.user_id,
                    "variant_name": a.variant_name,
                    "assigned_at": a.assigned_at.isoformat(),
                    "outcomes": [(o.value, t.isoformat()) for o, t in a.outcomes],
                    "messages_sent": a.messages_sent,
                }
                for uid, a in self.assignments.items()
            },
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load(self) -> None:
        """Load test data from file."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
            
            self.config = TestConfig.from_dict(data.get("config", {"name": self.name}))
            
            self.variants = {
                name: PersonalityVariant.from_dict(v)
                for name, v in data.get("variants", {}).items()
            }
            
            self.assignments = {}
            for uid, a in data.get("assignments", {}).items():
                assignment = UserAssignment(
                    user_id=a["user_id"],
                    variant_name=a["variant_name"],
                    assigned_at=datetime.fromisoformat(a["assigned_at"]),
                    messages_sent=a.get("messages_sent", 0),
                )
                assignment.outcomes = [
                    (OutcomeType(o), datetime.fromisoformat(t))
                    for o, t in a.get("outcomes", [])
                ]
                self.assignments[uid] = assignment
            
            logger.debug(f"Loaded test '{self.name}' with {len(self.variants)} variants")
            
        except Exception as e:
            logger.warning(f"Failed to load test data: {e}")


class ABTestManager:
    """
    Manager for multiple A/B tests.
    
    Coordinates multiple concurrent personality tests.
    """
    
    def __init__(self, storage_dir: str = "data/ab_tests"):
        """Initialize manager."""
        self.storage_dir = Path(storage_dir)
        self.tests: Dict[str, PersonalityABTest] = {}
        
        # Load existing tests
        self._load_all()
    
    def _load_all(self) -> None:
        """Load all existing tests."""
        if not self.storage_dir.exists():
            return
        
        for path in self.storage_dir.glob("*.json"):
            name = path.stem
            if name not in self.tests:
                self.tests[name] = PersonalityABTest(
                    name=name,
                    storage_path=str(path)
                )
    
    def create_test(
        self,
        name: str,
        config: Optional[TestConfig] = None,
    ) -> PersonalityABTest:
        """Create a new A/B test."""
        if name in self.tests:
            logger.warning(f"Test '{name}' already exists, returning existing")
            return self.tests[name]
        
        test = PersonalityABTest(
            name=name,
            config=config,
            storage_path=str(self.storage_dir / f"{name}.json"),
        )
        self.tests[name] = test
        return test
    
    def get_test(self, name: str) -> Optional[PersonalityABTest]:
        """Get a test by name."""
        return self.tests.get(name)
    
    def list_tests(self) -> List[Dict[str, Any]]:
        """List all tests with summary."""
        return [
            {
                "name": test.name,
                "variants": len(test.variants),
                "users": len(test.assignments),
                "analysis": test.analyze(),
            }
            for test in self.tests.values()
        ]
    
    def delete_test(self, name: str) -> bool:
        """Delete a test."""
        if name not in self.tests:
            return False
        
        test = self.tests.pop(name)
        if test.storage_path.exists():
            test.storage_path.unlink()
        
        return True
    
    def get_active_traits(self, user_id: str) -> Dict[str, Any]:
        """
        Get combined traits for a user from all active tests.
        
        Merges traits from all tests the user is assigned to.
        """
        combined = {}
        
        for test in self.tests.values():
            if user_id in test.assignments:
                variant = test.get_variant(user_id)
                combined.update(variant.traits)
        
        return combined


# Convenience functions
def create_tone_test() -> PersonalityABTest:
    """Create a standard tone variation test."""
    test = PersonalityABTest("tone_test")
    
    test.add_variant("formal", {
        "tone": "formal",
        "use_contractions": False,
        "emoji_frequency": 0.0,
        "verbosity": 0.8,
    }, description="Formal, professional tone")
    
    test.add_variant("casual", {
        "tone": "casual",
        "use_contractions": True,
        "emoji_frequency": 0.2,
        "verbosity": 0.5,
    }, description="Casual, friendly tone")
    
    test.add_variant("enthusiastic", {
        "tone": "enthusiastic",
        "use_contractions": True,
        "emoji_frequency": 0.3,
        "verbosity": 0.6,
        "exclamation_frequency": 0.2,
    }, description="Enthusiastic, energetic tone")
    
    return test


def create_verbosity_test() -> PersonalityABTest:
    """Create a standard verbosity test."""
    test = PersonalityABTest("verbosity_test")
    
    test.add_variant("concise", {
        "verbosity": 0.3,
        "max_response_sentences": 3,
        "include_examples": False,
    }, description="Short, direct responses")
    
    test.add_variant("balanced", {
        "verbosity": 0.5,
        "max_response_sentences": 5,
        "include_examples": True,
    }, description="Balanced response length")
    
    test.add_variant("detailed", {
        "verbosity": 0.8,
        "max_response_sentences": 10,
        "include_examples": True,
        "include_caveats": True,
    }, description="Detailed, comprehensive responses")
    
    return test


# Global manager singleton
_manager: Optional[ABTestManager] = None

def get_ab_manager() -> ABTestManager:
    """Get global A/B test manager."""
    global _manager
    if _manager is None:
        _manager = ABTestManager()
    return _manager


# Export public API
__all__ = [
    'PersonalityABTest',
    'PersonalityVariant',
    'TestConfig',
    'OutcomeType',
    'AssignmentStrategy',
    'ABTestManager',
    'create_tone_test',
    'create_verbosity_test',
    'get_ab_manager',
]
