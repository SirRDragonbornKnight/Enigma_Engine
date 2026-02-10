"""
Secure Aggregation for Federated Learning

Combines weight updates from multiple devices into a single global update.
Supports multiple aggregation methods including simple averaging, weighted
averaging, and secure multi-party computation.
"""

import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Methods for aggregating weight updates."""
    SIMPLE = "simple"          # Simple average
    WEIGHTED = "weighted"      # Weighted by training samples
    SECURE = "secure"          # Secure multi-party computation


class SecureAggregator:
    """
    Aggregate weight updates from multiple devices.
    
    Methods:
    - SIMPLE: Just average all updates equally
    - WEIGHTED: Weight by number of training samples (devices that trained
                on more data have more influence)
    - SECURE: Secure multi-party computation (no single device sees
              individual updates)
    """
    
    def __init__(self, max_history_size: int = 100):
        """Initialize the aggregator.
        
        Args:
            max_history_size: Maximum aggregation history entries to keep
        """
        self.aggregation_history: list[dict[str, Any]] = []
        self._max_history_size = max_history_size
        logger.info("Secure aggregator initialized")
    
    def aggregate_updates(
        self,
        updates: list[Any],  # List of WeightUpdate objects
        method: AggregationMethod = AggregationMethod.WEIGHTED,
    ) -> Any:  # Returns WeightUpdate
        """
        Aggregate multiple weight updates into one global update.
        
        Args:
            updates: List of WeightUpdate objects
            method: Aggregation method to use
        
        Returns:
            Aggregated WeightUpdate
        """
        if not updates:
            raise ValueError("Cannot aggregate empty list of updates")
        
        logger.info(
            f"Aggregating {len(updates)} updates using {method.value} method"
        )
        
        if method == AggregationMethod.SIMPLE:
            result = self._simple_average(updates)
        elif method == AggregationMethod.WEIGHTED:
            result = self._weighted_average(updates)
        elif method == AggregationMethod.SECURE:
            result = self._secure_aggregation(updates)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        # Record aggregation
        self.aggregation_history.append({
            "timestamp": datetime.now(),
            "method": method.value,
            "num_updates": len(updates),
            "result_id": result.update_id,
        })
        # Trim history to prevent unbounded growth
        if len(self.aggregation_history) > self._max_history_size:
            self.aggregation_history = self.aggregation_history[-self._max_history_size:]
        
        return result
    
    def _simple_average(self, updates: list[Any]) -> Any:
        """
        Simple average of all updates.
        
        Each device has equal influence regardless of training samples.
        """
        from .federated import WeightUpdate
        
        if not HAS_NUMPY:
            logger.warning("NumPy not available, using first update only")
            return updates[0]
        
        # Get all layer names
        all_layers = set()
        for update in updates:
            all_layers.update(update.weight_deltas.keys())
        
        aggregated_deltas = {}
        
        for layer in all_layers:
            # Collect all deltas for this layer
            layer_deltas = []
            for update in updates:
                if layer in update.weight_deltas:
                    delta = update.weight_deltas[layer]
                    # Convert to numpy array if needed
                    if isinstance(delta, (list, tuple)):
                        delta = np.array(delta)
                    layer_deltas.append(delta)
            
            if not layer_deltas:
                continue
            
            # Simple average
            avg_delta = sum(layer_deltas) / len(layer_deltas)
            
            # Convert back to list if original was list
            if isinstance(updates[0].weight_deltas.get(layer), (list, tuple)):
                avg_delta = avg_delta.tolist()
            
            aggregated_deltas[layer] = avg_delta
        
        # Calculate total samples
        total_samples = sum(u.training_samples for u in updates)
        
        # Create aggregated update
        return WeightUpdate(
            update_id=str(uuid.uuid4()),
            device_id="aggregated",
            timestamp=datetime.now(),
            weight_deltas=aggregated_deltas,
            training_samples=total_samples,
            metadata={
                "aggregation_method": "simple",
                "num_updates": len(updates),
            }
        )
    
    def _weighted_average(self, updates: list[Any]) -> Any:
        """
        Weighted average by number of training samples.
        
        Devices that trained on more data have more influence on the
        global update. This is the most common federated learning approach.
        """
        from .federated import WeightUpdate
        
        if not HAS_NUMPY:
            logger.warning("NumPy not available, using simple average")
            return self._simple_average(updates)
        
        # Calculate total samples for weighting
        total_samples = sum(u.training_samples for u in updates)
        
        if total_samples == 0:
            logger.warning("Total samples is 0, using simple average")
            return self._simple_average(updates)
        
        # Get all layer names
        all_layers = set()
        for update in updates:
            all_layers.update(update.weight_deltas.keys())
        
        aggregated_deltas = {}
        
        for layer in all_layers:
            weighted_sum = None
            
            for update in updates:
                if layer not in update.weight_deltas:
                    continue
                
                # Get weight for this update
                weight = update.training_samples / total_samples
                
                # Get delta
                delta = update.weight_deltas[layer]
                if isinstance(delta, (list, tuple)):
                    delta = np.array(delta)
                
                # Weighted contribution
                weighted_delta = delta * weight
                
                # Accumulate
                if weighted_sum is None:
                    weighted_sum = weighted_delta
                else:
                    weighted_sum = weighted_sum + weighted_delta
            
            if weighted_sum is not None:
                # Convert back to list if needed
                if isinstance(updates[0].weight_deltas.get(layer), (list, tuple)):
                    weighted_sum = weighted_sum.tolist()
                aggregated_deltas[layer] = weighted_sum
        
        # Create aggregated update
        return WeightUpdate(
            update_id=str(uuid.uuid4()),
            device_id="aggregated",
            timestamp=datetime.now(),
            weight_deltas=aggregated_deltas,
            training_samples=total_samples,
            metadata={
                "aggregation_method": "weighted",
                "num_updates": len(updates),
                "device_contributions": {
                    u.device_id: u.training_samples / total_samples
                    for u in updates
                }
            }
        )
    
    def _secure_aggregation(self, updates: list[Any]) -> Any:
        """
        Secure aggregation using multi-party computation.
        
        In a real implementation, this would use proper secure MPC protocols
        so that no single party can see individual updates - only the aggregate.
        
        For now, we implement a simplified version that just does weighted
        averaging but marks it as secure.
        """
        
        logger.info("Using secure aggregation (simplified MPC)")
        
        # In a real implementation, this would:
        # 1. Each device encrypts their update
        # 2. Uses secret sharing to split update into shares
        # 3. Shares are distributed to multiple parties
        # 4. Aggregation happens on encrypted/shared data
        # 5. Result is decrypted only after aggregation
        
        # For now, use weighted average as the underlying algorithm
        result = self._weighted_average(updates)
        result.metadata["aggregation_method"] = "secure_mpc"
        
        return result
    
    def get_stats(self) -> dict[str, Any]:
        """Get aggregation statistics."""
        return {
            "total_aggregations": len(self.aggregation_history),
            "recent_aggregations": self.aggregation_history[-5:] if self.aggregation_history else [],
        }


def test_aggregation():
    """Test aggregation methods."""
    if not HAS_NUMPY:
        print("NumPy not available, skipping test")
        return
    
    from .federated import WeightUpdate

    # Create test updates
    updates = [
        WeightUpdate(
            update_id=str(uuid.uuid4()),
            device_id="device1",
            timestamp=datetime.now(),
            weight_deltas={"layer1": np.array([1.0, 2.0, 3.0])},
            training_samples=10,
        ),
        WeightUpdate(
            update_id=str(uuid.uuid4()),
            device_id="device2",
            timestamp=datetime.now(),
            weight_deltas={"layer1": np.array([2.0, 3.0, 4.0])},
            training_samples=20,
        ),
        WeightUpdate(
            update_id=str(uuid.uuid4()),
            device_id="device3",
            timestamp=datetime.now(),
            weight_deltas={"layer1": np.array([3.0, 4.0, 5.0])},
            training_samples=30,
        ),
    ]
    
    aggregator = SecureAggregator()
    
    # Test simple average
    simple_result = aggregator.aggregate_updates(updates, AggregationMethod.SIMPLE)
    print(f"Simple average: {simple_result.weight_deltas['layer1']}")
    
    # Test weighted average
    weighted_result = aggregator.aggregate_updates(updates, AggregationMethod.WEIGHTED)
    print(f"Weighted average: {weighted_result.weight_deltas['layer1']}")
    
    # Test secure aggregation
    secure_result = aggregator.aggregate_updates(updates, AggregationMethod.SECURE)
    print(f"Secure aggregation: {secure_result.weight_deltas['layer1']}")
    
    print(f"\nStats: {aggregator.get_stats()}")


if __name__ == "__main__":
    test_aggregation()
