"""
Hyperparameter Search

Grid, random, and Bayesian optimization for hyperparameter tuning.
Supports parallel evaluation and early stopping.

FILE: enigma_engine/core/hyperparam_search.py
TYPE: Training
MAIN CLASSES: HyperparameterSearch, GridSearch, RandomSearch, BayesianSearch
"""

import logging
import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """Search strategy types."""
    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"


@dataclass
class HyperparameterSpace:
    """Definition of hyperparameter search space."""
    name: str
    type: str  # "continuous", "discrete", "categorical"
    low: float = 0.0
    high: float = 1.0
    values: list[Any] = field(default_factory=list)
    log_scale: bool = False


@dataclass
class Trial:
    """A single hyperparameter trial."""
    trial_id: int
    params: dict[str, Any]
    score: Optional[float] = None
    metrics: dict[str, float] = field(default_factory=dict)
    status: str = "pending"  # pending, running, completed, failed
    duration_seconds: float = 0.0
    start_time: float = 0.0


@dataclass
class SearchResult:
    """Results from hyperparameter search."""
    best_params: dict[str, Any]
    best_score: float
    all_trials: list[Trial]
    total_time_seconds: float


class HyperparameterSearch(ABC):
    """Abstract base class for hyperparameter search."""
    
    def __init__(self,
                 space: list[HyperparameterSpace],
                 objective: Callable[[dict[str, Any]], float],
                 maximize: bool = True,
                 n_trials: int = 10):
        """
        Initialize search.
        
        Args:
            space: Hyperparameter search space
            objective: Function to optimize (params -> score)
            maximize: Whether to maximize (True) or minimize
            n_trials: Number of trials
        """
        self.space = {s.name: s for s in space}
        self.objective = objective
        self.maximize = maximize
        self.n_trials = n_trials
        
        self.trials: list[Trial] = []
        self._best_trial: Optional[Trial] = None
    
    @abstractmethod
    def suggest(self) -> dict[str, Any]:
        """Suggest next hyperparameters to try."""
    
    def _sample_param(self, param: HyperparameterSpace) -> Any:
        """Sample a parameter value."""
        if param.type == "categorical":
            return random.choice(param.values)
        elif param.type == "discrete":
            return random.choice(param.values)
        elif param.type == "continuous":
            if param.log_scale:
                log_low = math.log(param.low)
                log_high = math.log(param.high)
                return math.exp(random.uniform(log_low, log_high))
            else:
                return random.uniform(param.low, param.high)
        return None
    
    def run(self, verbose: bool = True) -> SearchResult:
        """
        Run hyperparameter search.
        
        Args:
            verbose: Print progress
            
        Returns:
            Search results
        """
        start_time = time.time()
        
        for i in range(self.n_trials):
            # Suggest parameters
            params = self.suggest()
            
            # Create trial
            trial = Trial(
                trial_id=i,
                params=params,
                status="running",
                start_time=time.time()
            )
            
            if verbose:
                logger.info(f"Trial {i+1}/{self.n_trials}: {params}")
            
            try:
                # Run objective
                score = self.objective(params)
                
                trial.score = score
                trial.status = "completed"
                trial.duration_seconds = time.time() - trial.start_time
                
                # Update best
                if self._best_trial is None:
                    self._best_trial = trial
                elif self.maximize and score > self._best_trial.score:
                    self._best_trial = trial
                elif not self.maximize and score < self._best_trial.score:
                    self._best_trial = trial
                
                if verbose:
                    logger.info(f"  Score: {score:.4f} (best: {self._best_trial.score:.4f})")
                
            except Exception as e:
                logger.error(f"Trial {i} failed: {e}")
                trial.status = "failed"
                trial.duration_seconds = time.time() - trial.start_time
            
            self.trials.append(trial)
        
        total_time = time.time() - start_time
        
        return SearchResult(
            best_params=self._best_trial.params if self._best_trial else {},
            best_score=self._best_trial.score if self._best_trial else 0.0,
            all_trials=self.trials,
            total_time_seconds=total_time
        )


class GridSearch(HyperparameterSearch):
    """Exhaustive grid search over parameter combinations."""
    
    def __init__(self,
                 space: list[HyperparameterSpace],
                 objective: Callable[[dict[str, Any]], float],
                 maximize: bool = True):
        # Calculate total combinations
        total = 1
        for s in space:
            if s.type == "categorical" or s.type == "discrete":
                total *= len(s.values)
            else:
                raise ValueError("Grid search requires discrete/categorical params")
        
        super().__init__(space, objective, maximize, n_trials=total)
        
        self._grid = self._generate_grid()
        self._grid_index = 0
    
    def _generate_grid(self) -> list[dict[str, Any]]:
        """Generate all parameter combinations."""
        import itertools
        
        param_values = []
        param_names = []
        
        for name, space in self.space.items():
            param_names.append(name)
            param_values.append(space.values)
        
        combinations = []
        for combo in itertools.product(*param_values):
            combinations.append(dict(zip(param_names, combo)))
        
        return combinations
    
    def suggest(self) -> dict[str, Any]:
        """Get next grid point."""
        if self._grid_index >= len(self._grid):
            raise StopIteration("Grid exhausted")
        
        params = self._grid[self._grid_index]
        self._grid_index += 1
        return params


class RandomSearch(HyperparameterSearch):
    """Random search over parameter space."""
    
    def __init__(self,
                 space: list[HyperparameterSpace],
                 objective: Callable[[dict[str, Any]], float],
                 maximize: bool = True,
                 n_trials: int = 100,
                 seed: int = None):
        super().__init__(space, objective, maximize, n_trials)
        
        if seed is not None:
            random.seed(seed)
    
    def suggest(self) -> dict[str, Any]:
        """Sample random parameters."""
        params = {}
        for name, space in self.space.items():
            params[name] = self._sample_param(space)
        return params


class BayesianSearch(HyperparameterSearch):
    """
    Bayesian optimization using Gaussian Process surrogate.
    
    Uses expected improvement acquisition function.
    """
    
    def __init__(self,
                 space: list[HyperparameterSpace],
                 objective: Callable[[dict[str, Any]], float],
                 maximize: bool = True,
                 n_trials: int = 50,
                 n_initial: int = 5,
                 seed: int = None):
        """
        Initialize Bayesian search.
        
        Args:
            space: Parameter space
            objective: Objective function
            maximize: Maximize or minimize
            n_trials: Total trials
            n_initial: Initial random samples
            seed: Random seed
        """
        super().__init__(space, objective, maximize, n_trials)
        
        self.n_initial = n_initial
        self._trial_count = 0
        
        # Gaussian process model (simple implementation)
        self._X: list[list[float]] = []
        self._y: list[float] = []
        
        if seed is not None:
            random.seed(seed)
    
    def _encode_params(self, params: dict[str, Any]) -> list[float]:
        """Encode parameters to numeric vector."""
        encoded = []
        for name, space in self.space.items():
            value = params[name]
            if space.type == "categorical":
                # One-hot style encoding
                idx = space.values.index(value)
                encoded.append(idx / (len(space.values) - 1) if len(space.values) > 1 else 0)
            elif space.type == "continuous":
                if space.log_scale:
                    normalized = (math.log(value) - math.log(space.low)) / (math.log(space.high) - math.log(space.low))
                else:
                    normalized = (value - space.low) / (space.high - space.low)
                encoded.append(normalized)
            elif space.type == "discrete":
                idx = space.values.index(value)
                encoded.append(idx / (len(space.values) - 1) if len(space.values) > 1 else 0)
        return encoded
    
    def _decode_params(self, encoded: list[float]) -> dict[str, Any]:
        """Decode numeric vector to parameters."""
        params = {}
        i = 0
        for name, space in self.space.items():
            if space.type == "categorical":
                idx = round(encoded[i] * (len(space.values) - 1))
                idx = max(0, min(idx, len(space.values) - 1))
                params[name] = space.values[idx]
            elif space.type == "continuous":
                if space.log_scale:
                    log_value = encoded[i] * (math.log(space.high) - math.log(space.low)) + math.log(space.low)
                    params[name] = math.exp(log_value)
                else:
                    params[name] = encoded[i] * (space.high - space.low) + space.low
            elif space.type == "discrete":
                idx = round(encoded[i] * (len(space.values) - 1))
                idx = max(0, min(idx, len(space.values) - 1))
                params[name] = space.values[idx]
            i += 1
        return params
    
    def _rbf_kernel(self, x1: list[float], x2: list[float], length_scale: float = 1.0) -> float:
        """RBF kernel for GP."""
        sq_dist = sum((a - b) ** 2 for a, b in zip(x1, x2))
        return math.exp(-sq_dist / (2 * length_scale ** 2))
    
    def _predict(self, x: list[float]) -> tuple[float, float]:
        """Predict mean and variance at point x using GP."""
        if not self._X:
            return 0.0, 1.0
        
        # Kernel values
        K = [[self._rbf_kernel(xi, xj) for xj in self._X] for xi in self._X]
        k_star = [self._rbf_kernel(x, xi) for xi in self._X]
        
        # Add noise to diagonal
        noise = 1e-6
        for i in range(len(K)):
            K[i][i] += noise
        
        # Simple matrix inversion (for small n)
        try:
            # Use numpy if available
            import numpy as np
            K_np = np.array(K)
            k_np = np.array(k_star)
            y_np = np.array(self._y)
            
            K_inv = np.linalg.inv(K_np)
            mean = float(k_np @ K_inv @ y_np)
            var = float(1.0 - k_np @ K_inv @ k_np.T)
        except ImportError:
            # Fallback to simple prediction
            mean = sum(y * k for y, k in zip(self._y, k_star)) / (sum(k_star) + 1e-6)
            var = 1.0 - sum(k ** 2 for k in k_star) / (len(k_star) + 1e-6)
        
        return mean, max(var, 1e-6)
    
    def _expected_improvement(self, x: list[float]) -> float:
        """Calculate expected improvement at point x."""
        if not self._y:
            return 0.0
        
        mean, var = self._predict(x)
        std = math.sqrt(var)
        
        best = max(self._y) if self.maximize else min(self._y)
        
        if self.maximize:
            z = (mean - best) / std
        else:
            z = (best - mean) / std
        
        # Approximate normal CDF and PDF
        def norm_cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))
        
        def norm_pdf(x):
            return math.exp(-x ** 2 / 2) / math.sqrt(2 * math.pi)
        
        ei = std * (z * norm_cdf(z) + norm_pdf(z))
        return ei
    
    def suggest(self) -> dict[str, Any]:
        """Suggest next hyperparameters using Bayesian optimization."""
        self._trial_count += 1
        
        # Initial random samples
        if len(self._X) < self.n_initial:
            params = {}
            for name, space in self.space.items():
                params[name] = self._sample_param(space)
            return params
        
        # Optimize acquisition function with random restarts
        best_ei = -float('inf')
        best_params = None
        
        for _ in range(100):  # Random samples
            x = [random.random() for _ in self.space]
            ei = self._expected_improvement(x)
            
            if ei > best_ei:
                best_ei = ei
                best_params = self._decode_params(x)
        
        return best_params
    
    def observe(self, params: dict[str, Any], score: float):
        """Record observation for GP model."""
        x = self._encode_params(params)
        self._X.append(x)
        self._y.append(score)


class EarlyStopping:
    """Early stopping for trials that aren't promising."""
    
    def __init__(self,
                 patience: int = 5,
                 min_delta: float = 0.001,
                 maximize: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Epochs without improvement before stopping
            min_delta: Minimum change to count as improvement
            maximize: Whether higher is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.maximize = maximize
        
        self.best_score = None
        self.counter = 0
    
    def should_stop(self, score: float) -> bool:
        """Check if training should stop."""
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.maximize:
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
    
    def reset(self):
        """Reset state for new trial."""
        self.best_score = None
        self.counter = 0


def create_search_space() -> list[HyperparameterSpace]:
    """Create default hyperparameter search space for LLM training."""
    return [
        HyperparameterSpace(
            name="learning_rate",
            type="continuous",
            low=1e-6,
            high=1e-3,
            log_scale=True
        ),
        HyperparameterSpace(
            name="batch_size",
            type="discrete",
            values=[4, 8, 16, 32, 64]
        ),
        HyperparameterSpace(
            name="weight_decay",
            type="continuous",
            low=0.0,
            high=0.1
        ),
        HyperparameterSpace(
            name="warmup_ratio",
            type="continuous",
            low=0.0,
            high=0.1
        ),
        HyperparameterSpace(
            name="dropout",
            type="continuous",
            low=0.0,
            high=0.5
        ),
        HyperparameterSpace(
            name="optimizer",
            type="categorical",
            values=["adam", "adamw", "sgd"]
        )
    ]


def tune_hyperparameters(objective: Callable,
                         space: list[HyperparameterSpace] = None,
                         strategy: SearchStrategy = SearchStrategy.BAYESIAN,
                         n_trials: int = 50) -> SearchResult:
    """
    Convenience function for hyperparameter tuning.
    
    Args:
        objective: Objective function to optimize
        space: Search space (defaults to LLM training params)
        strategy: Search strategy
        n_trials: Number of trials
        
    Returns:
        Search results
    """
    space = space or create_search_space()
    
    if strategy == SearchStrategy.GRID:
        search = GridSearch(space, objective)
    elif strategy == SearchStrategy.RANDOM:
        search = RandomSearch(space, objective, n_trials=n_trials)
    elif strategy == SearchStrategy.BAYESIAN:
        search = BayesianSearch(space, objective, n_trials=n_trials)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return search.run()


__all__ = [
    'HyperparameterSearch',
    'GridSearch',
    'RandomSearch',
    'BayesianSearch',
    'HyperparameterSpace',
    'Trial',
    'SearchResult',
    'SearchStrategy',
    'EarlyStopping',
    'create_search_space',
    'tune_hyperparameters'
]
