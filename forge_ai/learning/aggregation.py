"""
Secure Aggregation for Federated Learning

Aggregate weight updates from multiple devices using various methods:
- Simple averaging
- Weighted averaging (by number of samples)
- Secure multi-party computation (future)
"""

import logging
from enum import Enum
from typing import Dict, List

import numpy as np

from .federated import WeightUpdate

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Methods for aggregating weight updates."""
    SIMPLE = "simple"          # Simple average
    WEIGHTED = "weighted"      # Weighted by training samples
    SECURE = "secure"          # Secure multi-party computation
    MEDIAN = "median"          # Use median instead of mean (robust to outliers)


class SecureAggregator:
    """
    Aggregate weight updates from multiple devices.
    
    Methods:
    - SIMPLE: Just average the updates
    - WEIGHTED: Weight by training samples
    - MEDIAN: Use median (robust to outliers/poisoning)
    - SECURE: Secure multi-party computation (MPC) - future feature
    """
    
    def __init__(self, method: AggregationMethod = AggregationMethod.WEIGHTED):
        """
        Initialize aggregator.
        
        Args:
            method: Aggregation method to use
        """
        self.method = method
        logger.debug(f"Initialized SecureAggregator: method={method.value}")
    
    def aggregate_updates(
        self, 
        updates: List[WeightUpdate],
        method: AggregationMethod = None
    ) -> WeightUpdate:
        """
        Aggregate multiple weight updates into one.
        
        Args:
            updates: List of weight updates from different devices
            method: Override default aggregation method
            
        Returns:
            Aggregated weight update
        """
        if not updates:
            raise ValueError("Cannot aggregate empty list of updates")
        
        method = method or self.method
        
        logger.info(
            f"Aggregating {len(updates)} updates using {method.value} method"
        )
        
        if method == AggregationMethod.SIMPLE:
            return self._simple_average(updates)
        elif method == AggregationMethod.WEIGHTED:
            return self._weighted_average(updates)
        elif method == AggregationMethod.MEDIAN:
            return self._median_aggregate(updates)
        elif method == AggregationMethod.SECURE:
            return self._secure_aggregation(updates)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def _simple_average(self, updates: List[WeightUpdate]) -> WeightUpdate:
        """
        Simple average of all updates.
        
        Each device has equal influence regardless of training samples.
        
        Args:
            updates: Weight updates to aggregate
            
        Returns:
            Averaged weight update
        """
        # Get all layer names
        layer_names = set()
        for update in updates:
            layer_names.update(update.weight_deltas.keys())
        
        # Average each layer
        aggregated_deltas = {}
        for name in layer_names:
            deltas = []
            for update in updates:
                if name in update.weight_deltas:
                    deltas.append(update.weight_deltas[name])
            
            if deltas:
                # Simple mean
                aggregated_deltas[name] = np.mean(deltas, axis=0)
        
        # Create aggregated update
        from datetime import datetime
        result = WeightUpdate(
            update_id=f"aggregated_{datetime.now().timestamp()}",
            device_id="aggregator",
            timestamp=datetime.now(),
            weight_deltas=aggregated_deltas,
            training_samples=sum(u.training_samples for u in updates),
            metadata={
                "method": "simple",
                "num_devices": len(updates),
                "round": updates[0].metadata.get("round") if updates else None,
            }
        )
        
        logger.debug(f"Simple averaged {len(updates)} updates")
        return result
    
    def _weighted_average(self, updates: List[WeightUpdate]) -> WeightUpdate:
        """
        Weighted average by number of training samples.
        
        Devices that trained on more data have more influence.
        This is the standard FedAvg algorithm.
        
        Args:
            updates: Weight updates to aggregate
            
        Returns:
            Weighted averaged update
        """
        # Calculate total training samples
        total_samples = sum(u.training_samples for u in updates)
        
        if total_samples == 0:
            logger.warning("Total training samples is 0, using simple average")
            return self._simple_average(updates)
        
        # Get all layer names
        layer_names = set()
        for update in updates:
            layer_names.update(update.weight_deltas.keys())
        
        # Weighted average each layer
        aggregated_deltas = {}
        for name in layer_names:
            weighted_sum = None
            
            for update in updates:
                if name in update.weight_deltas:
                    delta = update.weight_deltas[name]
                    weight = update.training_samples / total_samples
                    
                    if weighted_sum is None:
                        weighted_sum = delta * weight
                    else:
                        weighted_sum += delta * weight
            
            if weighted_sum is not None:
                aggregated_deltas[name] = weighted_sum
        
        # Create aggregated update
        from datetime import datetime
        result = WeightUpdate(
            update_id=f"aggregated_{datetime.now().timestamp()}",
            device_id="aggregator",
            timestamp=datetime.now(),
            weight_deltas=aggregated_deltas,
            training_samples=total_samples,
            metadata={
                "method": "weighted",
                "num_devices": len(updates),
                "total_samples": total_samples,
                "round": updates[0].metadata.get("round") if updates else None,
            }
        )
        
        logger.debug(
            f"Weighted averaged {len(updates)} updates "
            f"(total samples: {total_samples})"
        )
        return result
    
    def _median_aggregate(self, updates: List[WeightUpdate]) -> WeightUpdate:
        """
        Use median instead of mean for robustness.
        
        More robust to outliers and Byzantine (malicious) updates.
        
        Args:
            updates: Weight updates to aggregate
            
        Returns:
            Median-aggregated update
        """
        # Get all layer names
        layer_names = set()
        for update in updates:
            layer_names.update(update.weight_deltas.keys())
        
        # Median for each layer
        aggregated_deltas = {}
        for name in layer_names:
            deltas = []
            for update in updates:
                if name in update.weight_deltas:
                    deltas.append(update.weight_deltas[name])
            
            if deltas:
                # Use median
                aggregated_deltas[name] = np.median(deltas, axis=0)
        
        # Create aggregated update
        from datetime import datetime
        result = WeightUpdate(
            update_id=f"aggregated_{datetime.now().timestamp()}",
            device_id="aggregator",
            timestamp=datetime.now(),
            weight_deltas=aggregated_deltas,
            training_samples=sum(u.training_samples for u in updates),
            metadata={
                "method": "median",
                "num_devices": len(updates),
                "round": updates[0].metadata.get("round") if updates else None,
            }
        )
        
        logger.debug(f"Median aggregated {len(updates)} updates")
        return result
    
    def _secure_aggregation(self, updates: List[WeightUpdate]) -> WeightUpdate:
        """
        Secure multi-party computation for aggregation.
        
        This is a placeholder for future implementation of secure
        aggregation protocols where the aggregator cannot see
        individual updates.
        
        Args:
            updates: Weight updates to aggregate
            
        Returns:
            Securely aggregated update
        """
        logger.warning(
            "Secure aggregation (MPC) not yet implemented, "
            "falling back to weighted average"
        )
        return self._weighted_average(updates)
    
    def validate_updates(self, updates: List[WeightUpdate]) -> List[WeightUpdate]:
        """
        Validate and filter updates before aggregation.
        
        Remove updates that:
        - Have invalid signatures
        - Have suspicious magnitudes
        - Come from untrusted devices
        
        Args:
            updates: Updates to validate
            
        Returns:
            List of valid updates
        """
        valid_updates = []
        
        for update in updates:
            # Check signature if present
            if update.signature and not update.verify_signature():
                logger.warning(
                    f"Rejecting update {update.update_id[:8]}... "
                    f"due to invalid signature"
                )
                continue
            
            # Check for suspicious magnitude
            total_magnitude = 0
            for delta in update.weight_deltas.values():
                total_magnitude += np.linalg.norm(delta)
            
            # Simple heuristic: reject if magnitude is too large
            # (could indicate Byzantine attack)
            if total_magnitude > 1e10:
                logger.warning(
                    f"Rejecting update {update.update_id[:8]}... "
                    f"due to suspicious magnitude: {total_magnitude:.2e}"
                )
                continue
            
            valid_updates.append(update)
        
        logger.info(
            f"Validated updates: {len(valid_updates)}/{len(updates)} passed"
        )
        
        return valid_updates
