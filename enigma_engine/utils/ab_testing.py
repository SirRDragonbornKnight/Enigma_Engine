"""
A/B Testing Framework - Test feature variants.

Provides experiment infrastructure for:
- Feature flag management
- Variant assignment
- Statistical analysis
- Conversion tracking
- Experiment persistence

Part of the Enigma AI Engine testing utilities.
"""

import hashlib
import json
import random
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


class ExperimentStatus(Enum):
    """Experiment lifecycle status."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class AllocationStrategy(Enum):
    """How to assign users to variants."""
    RANDOM = "random"           # Pure random
    HASH = "hash"               # Deterministic hash-based
    WEIGHTED = "weighted"       # Weighted random
    SEQUENTIAL = "sequential"   # Round-robin


@dataclass
class Variant:
    """An experiment variant."""
    id: str
    name: str
    weight: float = 1.0
    config: dict[str, Any] = field(default_factory=dict)
    
    # Metrics
    impressions: int = 0
    conversions: int = 0
    total_value: float = 0.0
    
    @property
    def conversion_rate(self) -> float:
        """Calculate conversion rate."""
        return self.conversions / self.impressions if self.impressions > 0 else 0.0
    
    @property
    def average_value(self) -> float:
        """Calculate average conversion value."""
        return self.total_value / self.conversions if self.conversions > 0 else 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "weight": self.weight,
            "config": self.config,
            "impressions": self.impressions,
            "conversions": self.conversions,
            "total_value": self.total_value,
            "conversion_rate": self.conversion_rate,
            "average_value": self.average_value
        }


@dataclass
class Experiment:
    """An A/B test experiment."""
    id: str
    name: str
    description: str = ""
    variants: list[Variant] = field(default_factory=list)
    status: ExperimentStatus = ExperimentStatus.DRAFT
    allocation: AllocationStrategy = AllocationStrategy.HASH
    
    # Targeting
    target_percentage: float = 100.0  # % of users in experiment
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Assignment tracking
    _assignments: dict[str, str] = field(default_factory=dict)
    _sequential_index: int = 0
    
    def add_variant(
        self,
        id: str,
        name: str,
        weight: float = 1.0,
        config: Optional[dict[str, Any]] = None
    ) -> Variant:
        """Add a variant to the experiment."""
        variant = Variant(
            id=id,
            name=name,
            weight=weight,
            config=config or {}
        )
        self.variants.append(variant)
        return variant
    
    def get_variant(self, variant_id: str) -> Optional[Variant]:
        """Get a variant by ID."""
        for v in self.variants:
            if v.id == variant_id:
                return v
        return None
    
    def start(self) -> bool:
        """Start the experiment."""
        if self.status == ExperimentStatus.DRAFT:
            self.status = ExperimentStatus.RUNNING
            self.started_at = datetime.now()
            return True
        return False
    
    def pause(self) -> bool:
        """Pause the experiment."""
        if self.status == ExperimentStatus.RUNNING:
            self.status = ExperimentStatus.PAUSED
            return True
        return False
    
    def resume(self) -> bool:
        """Resume the experiment."""
        if self.status == ExperimentStatus.PAUSED:
            self.status = ExperimentStatus.RUNNING
            return True
        return False
    
    def complete(self) -> bool:
        """Mark experiment as complete."""
        if self.status in (ExperimentStatus.RUNNING, ExperimentStatus.PAUSED):
            self.status = ExperimentStatus.COMPLETED
            self.completed_at = datetime.now()
            return True
        return False
    
    def assign(self, user_id: str) -> Optional[Variant]:
        """
        Assign a user to a variant.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            Assigned variant or None if not in experiment
        """
        if self.status != ExperimentStatus.RUNNING:
            return None
        
        if not self.variants:
            return None
        
        # Check if already assigned
        if user_id in self._assignments:
            return self.get_variant(self._assignments[user_id])
        
        # Check targeting percentage
        if self.target_percentage < 100:
            hash_val = int(hashlib.md5(f"{self.id}:{user_id}:target".encode()).hexdigest(), 16)
            if (hash_val % 100) >= self.target_percentage:
                return None
        
        # Assign based on strategy
        variant = self._select_variant(user_id)
        if variant:
            self._assignments[user_id] = variant.id
            variant.impressions += 1
        
        return variant
    
    def _select_variant(self, user_id: str) -> Optional[Variant]:
        """Select variant based on allocation strategy."""
        if not self.variants:
            return None
        
        if self.allocation == AllocationStrategy.RANDOM:
            return random.choice(self.variants)
        
        elif self.allocation == AllocationStrategy.HASH:
            hash_val = int(hashlib.md5(f"{self.id}:{user_id}".encode()).hexdigest(), 16)
            total_weight = sum(v.weight for v in self.variants)
            threshold = (hash_val % 1000) / 1000 * total_weight
            
            cumulative = 0.0
            for variant in self.variants:
                cumulative += variant.weight
                if cumulative >= threshold:
                    return variant
            return self.variants[-1]
        
        elif self.allocation == AllocationStrategy.WEIGHTED:
            total_weight = sum(v.weight for v in self.variants)
            r = random.random() * total_weight
            
            cumulative = 0.0
            for variant in self.variants:
                cumulative += variant.weight
                if cumulative >= r:
                    return variant
            return self.variants[-1]
        
        elif self.allocation == AllocationStrategy.SEQUENTIAL:
            variant = self.variants[self._sequential_index % len(self.variants)]
            self._sequential_index += 1
            return variant
        
        return self.variants[0]
    
    def convert(
        self,
        user_id: str,
        value: float = 1.0
    ) -> bool:
        """
        Record a conversion.
        
        Args:
            user_id: User who converted
            value: Conversion value
            
        Returns:
            True if conversion recorded
        """
        if user_id not in self._assignments:
            return False
        
        variant = self.get_variant(self._assignments[user_id])
        if variant:
            variant.conversions += 1
            variant.total_value += value
            return True
        return False
    
    def get_winner(self, metric: str = "conversion_rate") -> Optional[Variant]:
        """
        Get the winning variant.
        
        Args:
            metric: Metric to compare (conversion_rate, average_value)
            
        Returns:
            Best performing variant
        """
        if not self.variants:
            return None
        
        return max(
            self.variants,
            key=lambda v: getattr(v, metric, 0.0)
        )
    
    def get_stats(self) -> dict[str, Any]:
        """Get experiment statistics."""
        total_impressions = sum(v.impressions for v in self.variants)
        total_conversions = sum(v.conversions for v in self.variants)
        
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "total_impressions": total_impressions,
            "total_conversions": total_conversions,
            "overall_conversion_rate": total_conversions / total_impressions if total_impressions > 0 else 0.0,
            "variants": [v.to_dict() for v in self.variants],
            "winner": (winner.id if (winner := self.get_winner()) else None),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }
    
    def to_dict(self) -> dict[str, Any]:
        """Convert experiment to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "variants": [v.to_dict() for v in self.variants],
            "status": self.status.value,
            "allocation": self.allocation.value,
            "target_percentage": self.target_percentage,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "assignments": self._assignments,
            "sequential_index": self._sequential_index
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Experiment':
        """Create experiment from dictionary."""
        exp = cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            status=ExperimentStatus(data.get("status", "draft")),
            allocation=AllocationStrategy(data.get("allocation", "hash")),
            target_percentage=data.get("target_percentage", 100.0)
        )
        
        for v_data in data.get("variants", []):
            variant = Variant(
                id=v_data["id"],
                name=v_data["name"],
                weight=v_data.get("weight", 1.0),
                config=v_data.get("config", {}),
                impressions=v_data.get("impressions", 0),
                conversions=v_data.get("conversions", 0),
                total_value=v_data.get("total_value", 0.0)
            )
            exp.variants.append(variant)
        
        exp._assignments = data.get("assignments", {})
        exp._sequential_index = data.get("sequential_index", 0)
        
        if data.get("created_at"):
            exp.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("started_at"):
            exp.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            exp.completed_at = datetime.fromisoformat(data["completed_at"])
        
        return exp


class ABTestingManager:
    """
    Manage A/B testing experiments.
    
    Usage:
        manager = ABTestingManager("data/experiments.json")
        
        # Create experiment
        exp = manager.create_experiment(
            id="new-chat-ui",
            name="New Chat UI Test",
            variants=[
                {"id": "control", "name": "Original UI"},
                {"id": "treatment", "name": "New UI"}
            ]
        )
        
        # Start
        manager.start_experiment("new-chat-ui")
        
        # Assign user
        variant = manager.assign("new-chat-ui", "user-123")
        if variant and variant.id == "treatment":
            # Show new UI
            ...
        
        # Record conversion
        manager.convert("new-chat-ui", "user-123", value=1.0)
        
        # Get stats
        stats = manager.get_stats("new-chat-ui")
    """
    
    def __init__(self, persistence_file: Optional[str] = None):
        """
        Initialize A/B testing manager.
        
        Args:
            persistence_file: Path to persist experiments
        """
        self._experiments: dict[str, Experiment] = {}
        self._persistence_file = Path(persistence_file) if persistence_file else None
        self._lock = threading.Lock()
        
        # Callbacks
        self._on_assignment: Optional[Callable[[str, str, Variant], None]] = None
        self._on_conversion: Optional[Callable[[str, str, float], None]] = None
        
        # Load persisted experiments
        if self._persistence_file and self._persistence_file.exists():
            self._load()
    
    def create_experiment(
        self,
        id: str,
        name: str,
        description: str = "",
        variants: Optional[list[dict[str, Any]]] = None,
        allocation: AllocationStrategy = AllocationStrategy.HASH,
        target_percentage: float = 100.0
    ) -> Experiment:
        """
        Create a new experiment.
        
        Args:
            id: Unique experiment ID
            name: Display name
            description: Experiment description
            variants: List of variant configs
            allocation: Assignment strategy
            target_percentage: % of users to include
            
        Returns:
            Created experiment
        """
        exp = Experiment(
            id=id,
            name=name,
            description=description,
            allocation=allocation,
            target_percentage=target_percentage
        )
        
        if variants:
            for v in variants:
                exp.add_variant(
                    id=v["id"],
                    name=v["name"],
                    weight=v.get("weight", 1.0),
                    config=v.get("config", {})
                )
        
        with self._lock:
            self._experiments[id] = exp
        
        self._save()
        return exp
    
    def get_experiment(self, id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        return self._experiments.get(id)
    
    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None
    ) -> list[Experiment]:
        """List all experiments, optionally filtered by status."""
        exps = list(self._experiments.values())
        if status:
            exps = [e for e in exps if e.status == status]
        return exps
    
    def start_experiment(self, id: str) -> bool:
        """Start an experiment."""
        exp = self._experiments.get(id)
        if exp and exp.start():
            self._save()
            return True
        return False
    
    def pause_experiment(self, id: str) -> bool:
        """Pause an experiment."""
        exp = self._experiments.get(id)
        if exp and exp.pause():
            self._save()
            return True
        return False
    
    def resume_experiment(self, id: str) -> bool:
        """Resume an experiment."""
        exp = self._experiments.get(id)
        if exp and exp.resume():
            self._save()
            return True
        return False
    
    def complete_experiment(self, id: str) -> bool:
        """Complete an experiment."""
        exp = self._experiments.get(id)
        if exp and exp.complete():
            self._save()
            return True
        return False
    
    def delete_experiment(self, id: str) -> bool:
        """Delete an experiment."""
        if id in self._experiments:
            del self._experiments[id]
            self._save()
            return True
        return False
    
    def assign(
        self,
        experiment_id: str,
        user_id: str
    ) -> Optional[Variant]:
        """
        Assign user to experiment variant.
        
        Args:
            experiment_id: Experiment ID
            user_id: User identifier
            
        Returns:
            Assigned variant or None
        """
        exp = self._experiments.get(experiment_id)
        if not exp:
            return None
        
        variant = exp.assign(user_id)
        
        if variant:
            self._save()
            if self._on_assignment:
                self._on_assignment(experiment_id, user_id, variant)
        
        return variant
    
    def convert(
        self,
        experiment_id: str,
        user_id: str,
        value: float = 1.0
    ) -> bool:
        """
        Record a conversion.
        
        Args:
            experiment_id: Experiment ID
            user_id: User who converted
            value: Conversion value
            
        Returns:
            True if recorded
        """
        exp = self._experiments.get(experiment_id)
        if exp and exp.convert(user_id, value):
            self._save()
            if self._on_conversion:
                self._on_conversion(experiment_id, user_id, value)
            return True
        return False
    
    def get_stats(self, experiment_id: str) -> Optional[dict[str, Any]]:
        """Get experiment statistics."""
        exp = self._experiments.get(experiment_id)
        return exp.get_stats() if exp else None
    
    def get_user_variants(self, user_id: str) -> dict[str, str]:
        """
        Get all variants assigned to a user.
        
        Returns:
            Dict of experiment_id -> variant_id
        """
        result = {}
        for exp_id, exp in self._experiments.items():
            if user_id in exp._assignments:
                result[exp_id] = exp._assignments[user_id]
        return result
    
    def on_assignment(
        self,
        callback: Callable[[str, str, Variant], None]
    ):
        """Set callback for variant assignments."""
        self._on_assignment = callback
    
    def on_conversion(
        self,
        callback: Callable[[str, str, float], None]
    ):
        """Set callback for conversions."""
        self._on_conversion = callback
    
    def _save(self):
        """Save experiments to file."""
        if not self._persistence_file:
            return
        
        try:
            data = {
                id: exp.to_dict()
                for id, exp in self._experiments.items()
            }
            self._persistence_file.parent.mkdir(parents=True, exist_ok=True)
            self._persistence_file.write_text(json.dumps(data, indent=2))
        except Exception:
            pass
    
    def _load(self):
        """Load experiments from file."""
        if not self._persistence_file or not self._persistence_file.exists():
            return
        
        try:
            data = json.loads(self._persistence_file.read_text())
            for id, exp_data in data.items():
                self._experiments[id] = Experiment.from_dict(exp_data)
        except Exception:
            pass


# Feature flag support
class FeatureFlag:
    """
    Simple feature flag with targeting.
    
    Usage:
        flag = FeatureFlag("dark-mode", default=False)
        flag.enable_for_users(["user-123", "user-456"])
        flag.enable_percentage(10)  # 10% of users
        
        if flag.is_enabled("user-789"):
            # Show dark mode
            ...
    """
    
    def __init__(
        self,
        id: str,
        default: bool = False,
        description: str = ""
    ):
        """
        Initialize feature flag.
        
        Args:
            id: Unique flag identifier
            default: Default enabled state
            description: Flag description
        """
        self.id = id
        self.default = default
        self.description = description
        
        self._enabled_globally = False
        self._disabled_globally = False
        self._enabled_users: set = set()
        self._disabled_users: set = set()
        self._percentage: float = 0.0
    
    def enable(self) -> 'FeatureFlag':
        """Enable flag globally."""
        self._enabled_globally = True
        self._disabled_globally = False
        return self
    
    def disable(self) -> 'FeatureFlag':
        """Disable flag globally."""
        self._disabled_globally = True
        self._enabled_globally = False
        return self
    
    def enable_for_users(self, user_ids: list[str]) -> 'FeatureFlag':
        """Enable for specific users."""
        self._enabled_users.update(user_ids)
        return self
    
    def disable_for_users(self, user_ids: list[str]) -> 'FeatureFlag':
        """Disable for specific users."""
        self._disabled_users.update(user_ids)
        return self
    
    def enable_percentage(self, percentage: float) -> 'FeatureFlag':
        """Enable for a percentage of users (0-100)."""
        self._percentage = max(0.0, min(100.0, percentage))
        return self
    
    def is_enabled(self, user_id: Optional[str] = None) -> bool:
        """
        Check if flag is enabled for a user.
        
        Args:
            user_id: Optional user identifier
            
        Returns:
            True if enabled
        """
        # Global overrides
        if self._disabled_globally:
            return False
        if self._enabled_globally:
            return True
        
        # User-specific
        if user_id:
            if user_id in self._disabled_users:
                return False
            if user_id in self._enabled_users:
                return True
            
            # Percentage rollout
            if self._percentage > 0:
                hash_val = int(hashlib.md5(f"{self.id}:{user_id}".encode()).hexdigest(), 16)
                if (hash_val % 100) < self._percentage:
                    return True
        
        return self.default
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "default": self.default,
            "description": self.description,
            "enabled_globally": self._enabled_globally,
            "disabled_globally": self._disabled_globally,
            "enabled_users": list(self._enabled_users),
            "disabled_users": list(self._disabled_users),
            "percentage": self._percentage
        }


# Global manager
_global_manager: Optional[ABTestingManager] = None


def get_ab_manager() -> ABTestingManager:
    """Get the global A/B testing manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = ABTestingManager()
    return _global_manager


def create_experiment(
    id: str,
    name: str,
    variants: list[dict[str, Any]],
    **kwargs
) -> Experiment:
    """Create experiment via global manager."""
    return get_ab_manager().create_experiment(id, name, variants=variants, **kwargs)


def assign_variant(
    experiment_id: str,
    user_id: str
) -> Optional[Variant]:
    """Assign variant via global manager."""
    return get_ab_manager().assign(experiment_id, user_id)


def record_conversion(
    experiment_id: str,
    user_id: str,
    value: float = 1.0
) -> bool:
    """Record conversion via global manager."""
    return get_ab_manager().convert(experiment_id, user_id, value)
