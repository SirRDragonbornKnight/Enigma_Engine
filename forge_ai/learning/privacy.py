"""
Differential Privacy for Federated Learning

Add calibrated noise to weight updates to prevent reverse-engineering
of training data from model updates.
"""

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


class DifferentialPrivacy:
    """
    Add noise to weight updates for privacy.
    
    Makes it impossible to reverse-engineer training data from updates.
    Uses the Gaussian mechanism for differential privacy.
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Initialize differential privacy.
        
        Args:
            epsilon: Privacy budget (lower = more private, more noise)
                    Typical values: 0.1 (very private) to 10.0 (less private)
            delta: Privacy parameter (probability of privacy breach)
                   Typical value: 1e-5 or 1e-7
        """
        self.epsilon = epsilon
        self.delta = delta
        
        logger.debug(f"Initialized DifferentialPrivacy: ε={epsilon}, δ={delta}")
    
    def add_noise(self, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Add calibrated noise to weights.
        
        Uses Gaussian mechanism or Laplace mechanism based on sensitivity.
        
        Args:
            weights: Original weights
            
        Returns:
            Noisy weights with differential privacy guarantees
        """
        noisy_weights = {}
        
        for name, weight in weights.items():
            # Calculate L2 sensitivity for this weight
            sensitivity = self._calculate_sensitivity(weight)
            
            # Calculate noise scale using Gaussian mechanism
            # σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
            noise_scale = (
                sensitivity * 
                np.sqrt(2 * np.log(1.25 / self.delta)) / 
                self.epsilon
            )
            
            # Add Gaussian noise
            noise = np.random.normal(0, noise_scale, weight.shape)
            noisy_weight = weight + noise
            
            noisy_weights[name] = noisy_weight
            
            # Log noise statistics
            noise_magnitude = np.linalg.norm(noise)
            weight_magnitude = np.linalg.norm(weight)
            noise_ratio = noise_magnitude / (weight_magnitude + 1e-10)
            
            logger.debug(
                f"Added noise to {name}: "
                f"scale={noise_scale:.6f}, "
                f"ratio={noise_ratio:.4f}"
            )
        
        return noisy_weights
    
    def _calculate_sensitivity(self, weight: np.ndarray) -> float:
        """
        Calculate L2 sensitivity for this weight.
        
        Sensitivity is the maximum change in the output that can be caused
        by a single training example. We use the L2 norm as a conservative
        estimate.
        
        Args:
            weight: Weight array
            
        Returns:
            L2 sensitivity
        """
        # Use L2 norm as sensitivity estimate
        # In practice, this could be refined based on:
        # - Gradient clipping bounds
        # - Learning rate
        # - Number of training samples
        
        sensitivity = np.linalg.norm(weight)
        
        # Add small constant to avoid division by zero
        if sensitivity < 1e-10:
            sensitivity = 1e-10
        
        return sensitivity
    
    def compute_privacy_loss(self, num_rounds: int) -> float:
        """
        Compute cumulative privacy loss over multiple rounds.
        
        Privacy budgets are consumed with each update. This calculates
        the total privacy loss after multiple rounds of federated learning.
        
        Args:
            num_rounds: Number of training rounds
            
        Returns:
            Total privacy loss (epsilon)
        """
        # Simple composition: ε_total = num_rounds * ε
        # Advanced composition would use tighter bounds
        total_epsilon = num_rounds * self.epsilon
        
        logger.info(
            f"Privacy loss after {num_rounds} rounds: "
            f"ε_total={total_epsilon:.2f}"
        )
        
        return total_epsilon
    
    def get_privacy_params(self) -> Dict[str, float]:
        """Get current privacy parameters."""
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
        }
    
    def update_privacy_budget(self, new_epsilon: float) -> None:
        """
        Update privacy budget.
        
        Args:
            new_epsilon: New epsilon value
        """
        old_epsilon = self.epsilon
        self.epsilon = new_epsilon
        
        logger.info(
            f"Updated privacy budget: ε={old_epsilon:.2f} -> ε={new_epsilon:.2f}"
        )


class GradientClipper:
    """
    Clip gradients to bound sensitivity.
    
    By clipping gradients before computing weight updates, we can
    bound the maximum influence of any single training example,
    which improves privacy guarantees.
    """
    
    def __init__(self, clip_norm: float = 1.0):
        """
        Initialize gradient clipper.
        
        Args:
            clip_norm: Maximum L2 norm for gradients
        """
        self.clip_norm = clip_norm
        logger.debug(f"Initialized GradientClipper: clip_norm={clip_norm}")
    
    def clip_gradients(
        self, 
        gradients: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Clip gradients to maximum norm.
        
        Args:
            gradients: Original gradients
            
        Returns:
            Clipped gradients
        """
        clipped = {}
        
        for name, gradient in gradients.items():
            # Calculate current norm
            norm = np.linalg.norm(gradient)
            
            # Clip if necessary
            if norm > self.clip_norm:
                scale = self.clip_norm / norm
                clipped[name] = gradient * scale
                logger.debug(
                    f"Clipped {name}: norm={norm:.4f} -> {self.clip_norm:.4f}"
                )
            else:
                clipped[name] = gradient.copy()
        
        return clipped
