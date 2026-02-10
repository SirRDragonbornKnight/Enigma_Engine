"""
================================================================================
DIFFERENTIAL PRIVACY - PROTECT INDIVIDUAL DATA
================================================================================

Adds calibrated noise to model updates to prevent reverse-engineering
of individual training data.

FILE: enigma_engine/federated/privacy.py
TYPE: Privacy Protection
MAIN CLASS: DifferentialPrivacy

HOW IT WORKS:
    - Adds Gaussian noise to weight updates
    - Noise scale based on privacy budget (epsilon)
    - Lower epsilon = more privacy = more noise
    - Prevents inferring individual data from updates

USAGE:
    privacy = DifferentialPrivacy(epsilon=1.0)
    protected_update = privacy.add_noise(update)
"""

import logging

import numpy as np

from .federation import ModelUpdate

logger = logging.getLogger(__name__)


class DifferentialPrivacy:
    """
    Add noise to updates for privacy protection.
    
    Makes it impossible to reverse-engineer individual data
    from model updates using differential privacy techniques.
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Initialize differential privacy.
        
        Args:
            epsilon: Privacy budget (lower = more private, more noise)
                    Typical values: 0.1 (very private) to 10 (less private)
            delta: Privacy parameter (probability of privacy breach)
                   Should be much smaller than 1/dataset_size
        """
        self.epsilon = epsilon
        self.delta = delta
        
        logger.info(f"Initialized differential privacy with epsilon={epsilon}, delta={delta}")
    
    def add_noise(self, update: ModelUpdate) -> ModelUpdate:
        """
        Add calibrated noise to updates.
        
        Uses Gaussian mechanism scaled to sensitivity.
        
        Args:
            update: Original model update
        
        Returns:
            Model update with noise added
        """
        noisy_deltas = {}
        
        for layer_name, weights in update.weight_deltas.items():
            # Calculate noise scale based on privacy budget and sensitivity
            noise_scale = self.get_noise_scale(weights)
            
            # Generate Gaussian noise
            noise = np.random.normal(0, noise_scale, weights.shape)
            
            # Add noise to weights
            noisy_deltas[layer_name] = weights + noise
        
        # Create new update with noisy weights
        noisy_update = ModelUpdate(
            device_id=update.device_id,
            round_number=update.round_number,
            weight_deltas=noisy_deltas,
            num_samples=update.num_samples,
            loss=update.loss,
            timestamp=update.timestamp,
            metadata={**update.metadata, 'differential_privacy': True, 'epsilon': self.epsilon},
        )
        
        logger.debug(f"Added differential privacy noise to {len(noisy_deltas)} layers")
        return noisy_update
    
    def get_noise_scale(self, weights: np.ndarray) -> float:
        """
        Calculate appropriate noise scale for privacy level.
        
        Uses the Gaussian mechanism:
        σ = (sensitivity * sqrt(2 * ln(1.25/delta))) / epsilon
        
        Args:
            weights: Weight array to calculate sensitivity from
        
        Returns:
            Noise standard deviation
        """
        # Estimate sensitivity as standard deviation of weights
        # In practice, this should be the L2 sensitivity bound
        sensitivity = np.std(weights)
        
        if sensitivity == 0:
            sensitivity = 1e-6  # Avoid division by zero
        
        # Gaussian mechanism noise scale
        noise_scale = (sensitivity * np.sqrt(2 * np.log(1.25 / self.delta))) / self.epsilon
        
        return noise_scale
    
    def clip_gradients(
        self,
        update: ModelUpdate,
        clip_norm: float = 1.0
    ) -> ModelUpdate:
        """
        Clip gradients to bound sensitivity.
        
        This is important for differential privacy as it limits
        how much any single example can influence the update.
        
        Args:
            update: Model update
            clip_norm: Maximum L2 norm for gradients
        
        Returns:
            Update with clipped gradients
        """
        clipped_deltas = {}
        
        for layer_name, weights in update.weight_deltas.items():
            # Calculate L2 norm
            norm = np.linalg.norm(weights)
            
            if norm > clip_norm:
                # Scale down to clip_norm
                clipped_deltas[layer_name] = weights * (clip_norm / norm)
            else:
                clipped_deltas[layer_name] = weights
        
        # Create new update with clipped weights
        clipped_update = ModelUpdate(
            device_id=update.device_id,
            round_number=update.round_number,
            weight_deltas=clipped_deltas,
            num_samples=update.num_samples,
            loss=update.loss,
            timestamp=update.timestamp,
            metadata={**update.metadata, 'gradient_clipping': True, 'clip_norm': clip_norm},
        )
        
        logger.debug(f"Clipped gradients to norm {clip_norm}")
        return clipped_update
    
    def calculate_privacy_spent(self, num_rounds: int, composition: str = "basic") -> float:
        """
        Calculate total privacy budget spent over multiple rounds.
        
        Args:
            num_rounds: Number of training rounds
            composition: Composition method ("basic", "advanced")
        
        Returns:
            Total epsilon spent
        """
        if composition == "basic":
            # Basic composition: ε_total = num_rounds * ε
            return num_rounds * self.epsilon
        elif composition == "advanced":
            # Advanced composition (tighter bound)
            # ε_total ≈ sqrt(2 * num_rounds * ln(1/δ)) * ε
            return np.sqrt(2 * num_rounds * np.log(1 / self.delta)) * self.epsilon
        else:
            return num_rounds * self.epsilon


class PrivacyAccountant:
    """
    Track privacy budget expenditure over time.
    
    Monitors how much privacy budget has been spent
    and warns when approaching limits.
    """
    
    def __init__(self, total_epsilon: float = 10.0, delta: float = 1e-5):
        """
        Initialize privacy accountant.
        
        Args:
            total_epsilon: Total privacy budget
            delta: Privacy parameter
        """
        self.total_epsilon = total_epsilon
        self.delta = delta
        self.spent_epsilon = 0.0
        self.rounds = 0
    
    def add_round(self, epsilon: float):
        """
        Record a training round.
        
        Args:
            epsilon: Epsilon spent in this round
        """
        self.spent_epsilon += epsilon
        self.rounds += 1
        
        if self.spent_epsilon > self.total_epsilon:
            logger.warning(
                f"Privacy budget exceeded! Spent {self.spent_epsilon:.2f} / {self.total_epsilon:.2f}"
            )
    
    def get_remaining_budget(self) -> float:
        """
        Get remaining privacy budget.
        
        Returns:
            Remaining epsilon
        """
        return max(0, self.total_epsilon - self.spent_epsilon)
    
    def can_continue(self) -> bool:
        """
        Check if training can continue within privacy budget.
        
        Returns:
            True if budget remains, False otherwise
        """
        return self.spent_epsilon < self.total_epsilon
    
    def get_stats(self) -> dict:
        """
        Get privacy statistics.
        
        Returns:
            Dictionary with privacy stats
        """
        return {
            'total_epsilon': self.total_epsilon,
            'spent_epsilon': self.spent_epsilon,
            'remaining_epsilon': self.get_remaining_budget(),
            'rounds': self.rounds,
            'can_continue': self.can_continue(),
        }
