"""
================================================================================
FEDERATED AGGREGATION - COMBINE UPDATES FROM MULTIPLE DEVICES
================================================================================

Aggregates model updates from multiple devices using Federated Averaging (FedAvg)
and secure aggregation techniques.

FILE: forge_ai/federated/aggregation.py
TYPE: Update Aggregation
MAIN CLASSES: FederatedAggregator, SecureAggregation

ALGORITHM (FedAvg):
    1. Weight each update by num_samples
    2. Average all weighted updates
    3. Apply to base model
    4. Return improved model

USAGE:
    aggregator = FederatedAggregator()
    updates = [update1, update2, update3]
    aggregated = aggregator.aggregate_updates(updates)
"""

import logging
from typing import Dict, List
import numpy as np

from .federation import ModelUpdate

logger = logging.getLogger(__name__)


class FederatedAggregator:
    """
    Aggregate updates from multiple devices.
    
    Uses Federated Averaging (FedAvg) algorithm:
    - Weight each update by number of samples
    - Average all weighted updates
    - Return aggregated weights
    """
    
    def __init__(self):
        """Initialize aggregator."""
        self.total_rounds = 0
        self.total_updates = 0
    
    def aggregate_updates(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """
        Aggregate updates from all participants.
        
        Algorithm:
        1. Weight each update by num_samples
        2. Average all weighted updates
        3. Return aggregated weights
        
        Args:
            updates: List of model updates from participants
        
        Returns:
            Aggregated weight deltas
        """
        if not updates:
            logger.warning("No updates to aggregate")
            return {}
        
        # Calculate total samples across all updates
        total_samples = sum(u.num_samples for u in updates)
        
        if total_samples == 0:
            logger.warning("Total samples is 0, using equal weights")
            total_samples = len(updates)
            for update in updates:
                update.num_samples = 1
        
        # Get layer names from first update
        layer_names = list(updates[0].weight_deltas.keys())
        
        # Aggregate each layer separately
        aggregated_weights = {}
        
        for layer_name in layer_names:
            # Weighted average of updates for this layer
            layer_updates = []
            
            for update in updates:
                if layer_name not in update.weight_deltas:
                    logger.warning(f"Layer {layer_name} missing in update from {update.device_id}")
                    continue
                
                # Weight by number of samples
                weight = update.num_samples / total_samples
                weighted_update = update.weight_deltas[layer_name] * weight
                layer_updates.append(weighted_update)
            
            if layer_updates:
                # Sum all weighted updates
                aggregated_weights[layer_name] = sum(layer_updates)
        
        self.total_rounds += 1
        self.total_updates += len(updates)
        
        logger.info(
            f"Aggregated {len(updates)} updates ({total_samples} total samples) "
            f"across {len(aggregated_weights)} layers"
        )
        
        return aggregated_weights
    
    def apply_updates(
        self,
        base_weights: Dict[str, np.ndarray],
        aggregated_deltas: Dict[str, np.ndarray],
        learning_rate: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Apply aggregated updates to base model weights.
        
        Args:
            base_weights: Current model weights
            aggregated_deltas: Aggregated weight deltas
            learning_rate: Learning rate for applying updates
        
        Returns:
            Updated model weights
        """
        updated_weights = {}
        
        for layer_name, base_weight in base_weights.items():
            if layer_name in aggregated_deltas:
                # Apply delta with learning rate
                delta = aggregated_deltas[layer_name]
                updated_weights[layer_name] = base_weight + (delta * learning_rate)
            else:
                # No update for this layer, keep as is
                updated_weights[layer_name] = base_weight
        
        logger.info(f"Applied updates to {len(updated_weights)} layers")
        return updated_weights
    
    def get_stats(self) -> Dict:
        """
        Get aggregation statistics.
        
        Returns:
            Dictionary with stats
        """
        return {
            'total_rounds': self.total_rounds,
            'total_updates': self.total_updates,
            'avg_updates_per_round': self.total_updates / max(1, self.total_rounds),
        }


class SecureAggregation:
    """
    Secure aggregation using cryptographic techniques.
    
    Additional privacy: Even coordinator can't see individual updates.
    Only the aggregate is visible.
    
    NOTE: This is a placeholder for future implementation.
    Real secure aggregation requires complex cryptographic protocols.
    """
    
    def __init__(self):
        """Initialize secure aggregation."""
        logger.warning(
            "SecureAggregation is not yet fully implemented. "
            "Individual updates will be visible to coordinator. "
            "Use DifferentialPrivacy for basic protection."
        )
    
    def encrypt_update(
        self,
        update: ModelUpdate,
        public_keys: List[str]
    ) -> 'EncryptedUpdate':
        """
        Encrypt update so only aggregate is visible.
        
        Args:
            update: Model update to encrypt
            public_keys: Public keys of other participants
        
        Returns:
            Encrypted update
        
        NOTE: Not yet implemented. Returns plaintext update.
        """
        logger.warning("SecureAggregation.encrypt_update not implemented, returning plaintext")
        
        # Placeholder - would use cryptographic protocol
        return EncryptedUpdate(
            device_id=update.device_id,
            round_number=update.round_number,
            encrypted_data={},  # Would contain encrypted shares
            num_samples=update.num_samples,
        )
    
    def decrypt_aggregate(
        self,
        encrypted_updates: List['EncryptedUpdate']
    ) -> Dict[str, np.ndarray]:
        """
        Decrypt only the aggregated result.
        
        Args:
            encrypted_updates: List of encrypted updates
        
        Returns:
            Aggregated weights (decrypted)
        
        NOTE: Not yet implemented. Returns empty dict.
        """
        logger.warning("SecureAggregation.decrypt_aggregate not implemented")
        return {}


class EncryptedUpdate:
    """
    Encrypted model update.
    
    In a real implementation, this would contain cryptographic shares
    that can only be decrypted in aggregate.
    """
    
    def __init__(
        self,
        device_id: str,
        round_number: int,
        encrypted_data: Dict,
        num_samples: int
    ):
        """
        Initialize encrypted update.
        
        Args:
            device_id: Device ID
            round_number: Training round number
            encrypted_data: Encrypted weight shares
            num_samples: Number of samples (not encrypted)
        """
        self.device_id = device_id
        self.round_number = round_number
        self.encrypted_data = encrypted_data
        self.num_samples = num_samples


class FederatedMedian:
    """
    Alternative aggregation using median instead of mean.
    
    More robust to outliers and Byzantine attacks.
    """
    
    def __init__(self):
        """Initialize median aggregator."""
        pass
    
    def aggregate_updates(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """
        Aggregate updates using coordinate-wise median.
        
        Args:
            updates: List of model updates
        
        Returns:
            Aggregated weight deltas using median
        """
        if not updates:
            return {}
        
        layer_names = list(updates[0].weight_deltas.keys())
        aggregated_weights = {}
        
        for layer_name in layer_names:
            # Collect all updates for this layer
            layer_updates = []
            
            for update in updates:
                if layer_name in update.weight_deltas:
                    layer_updates.append(update.weight_deltas[layer_name])
            
            if layer_updates:
                # Stack and take median
                stacked = np.stack(layer_updates, axis=0)
                aggregated_weights[layer_name] = np.median(stacked, axis=0)
        
        logger.info(f"Aggregated {len(updates)} updates using median")
        return aggregated_weights
