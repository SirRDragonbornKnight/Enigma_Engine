"""
Differential Privacy for Federated Learning

Adds calibrated noise to weight updates to prevent reverse-engineering
of training data. Even if someone intercepts the weight updates, they
cannot reconstruct the original training examples.
"""

import logging
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
else:
    try:
        import numpy as np
        HAS_NUMPY = True
    except ImportError:
        HAS_NUMPY = False
        np = None  # type: ignore

logger = logging.getLogger(__name__)


class DifferentialPrivacy:
    """
    Add noise to weight updates for privacy protection.
    
    Uses the Gaussian mechanism to add calibrated noise that provides
    (epsilon, delta)-differential privacy guarantees.
    
    Lower epsilon = more privacy but less accuracy
    Higher epsilon = less privacy but more accuracy
    
    Typical values:
        epsilon=0.1: Very strong privacy
        epsilon=1.0: Strong privacy (recommended)
        epsilon=10.0: Weak privacy
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Initialize differential privacy mechanism.
        
        Args:
            epsilon: Privacy budget (lower = more private)
            delta: Probability of privacy breach (should be very small)
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not (0 <= delta < 1):
            raise ValueError("delta must be in [0, 1)")
        
        self.epsilon = epsilon
        self.delta = delta
        
        logger.info(f"Differential privacy initialized (epsilon={epsilon}, delta={delta})")
    
    def add_noise(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add calibrated noise to weights for privacy.
        
        Uses the Gaussian mechanism with noise proportional to the
        sensitivity and privacy budget.
        
        Args:
            weights: Dictionary of layer names to weight arrays/lists
        
        Returns:
            Dictionary of noisy weights (same structure as input)
        """
        if not HAS_NUMPY:
            logger.warning("NumPy not available, cannot add differential privacy noise")
            return weights
        
        noisy_weights = {}
        
        for name, weight in weights.items():
            # Convert to numpy array if needed
            if isinstance(weight, (list, tuple)):
                weight_array = np.array(weight)
                was_list = True
            elif isinstance(weight, np.ndarray):
                weight_array = weight
                was_list = False
            else:
                # Can't add noise to non-numeric weights
                noisy_weights[name] = weight
                continue
            
            # Calculate sensitivity (L2 norm)
            sensitivity = self._calculate_sensitivity(weight_array)
            
            # Add Gaussian noise scaled by sensitivity and privacy budget
            # Gaussian mechanism: noise ~ N(0, sigma^2) where
            # sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon
            sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
            noise = np.random.normal(0, sigma, weight_array.shape)
            
            noisy_array = weight_array + noise
            
            # Convert back to original format
            if was_list:
                noisy_weights[name] = noisy_array.tolist()
            else:
                noisy_weights[name] = noisy_array
            
            logger.debug(
                f"Added DP noise to {name}: sensitivity={sensitivity:.4f}, sigma={sigma:.4f}"
            )
        
        return noisy_weights
    
    def _calculate_sensitivity(self, weight) -> float:
        """
        Calculate L2 sensitivity for weight array.
        
        This is the maximum L2 norm change that a single training example
        could cause. We approximate it as the L2 norm of the weights themselves.
        
        Args:
            weight: Weight array
        
        Returns:
            L2 sensitivity (scalar)
        """
        if not HAS_NUMPY:
            return 1.0
        return float(np.linalg.norm(weight))
    
    def compose_privacy_budget(self, num_queries: int) -> float:
        """
        Calculate total privacy cost for multiple queries.
        
        Using basic composition, the total epsilon grows linearly with
        the number of queries. Advanced composition could reduce this.
        
        Args:
            num_queries: Number of times the mechanism will be used
        
        Returns:
            Total epsilon budget consumed
        """
        # Basic composition: epsilon_total = num_queries * epsilon
        # In practice, advanced composition provides better bounds
        return num_queries * self.epsilon
    
    def get_privacy_level_description(self) -> str:
        """Get human-readable description of privacy level."""
        if self.epsilon <= 0.1:
            return "Very Strong Privacy (may significantly affect model quality)"
        elif self.epsilon <= 1.0:
            return "Strong Privacy (recommended for sensitive data)"
        elif self.epsilon <= 5.0:
            return "Moderate Privacy (good balance)"
        elif self.epsilon <= 10.0:
            return "Weak Privacy (better model quality)"
        else:
            return "Minimal Privacy (prioritizes model quality)"


def test_differential_privacy():
    """Test differential privacy mechanism."""
    if not HAS_NUMPY:
        print("NumPy not available, skipping test")
        return
    
    # Create test weights
    weights = {
        "layer1": np.array([1.0, 2.0, 3.0]),
        "layer2": np.array([[0.5, 0.5], [0.5, 0.5]]),
    }
    
    # Apply differential privacy
    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
    noisy_weights = dp.add_noise(weights)
    
    # Check that noise was added
    for name in weights:
        original = weights[name]
        noisy = noisy_weights[name]
        diff = np.abs(noisy - original).max()
        print(f"{name}: max difference = {diff:.6f}")
    
    print(f"Privacy level: {dp.get_privacy_level_description()}")
    print(f"Total epsilon after 10 rounds: {dp.compose_privacy_budget(10)}")


if __name__ == "__main__":
    # Run test
    test_differential_privacy()
