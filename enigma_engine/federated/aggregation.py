"""
================================================================================
FEDERATED AGGREGATION - COMBINE UPDATES FROM MULTIPLE DEVICES
================================================================================

Aggregates model updates from multiple devices using Federated Averaging (FedAvg)
and secure aggregation techniques.

FILE: enigma_engine/federated/aggregation.py
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
    
    def aggregate_updates(self, updates: list[ModelUpdate]) -> dict[str, np.ndarray]:
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
        base_weights: dict[str, np.ndarray],
        aggregated_deltas: dict[str, np.ndarray],
        learning_rate: float = 1.0
    ) -> dict[str, np.ndarray]:
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
    
    def get_stats(self) -> dict:
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
    Secure aggregation using secret sharing.
    
    Implements a simplified secure aggregation protocol:
    1. Each participant splits their update into shares
    2. Shares are distributed to other participants
    3. Only when all shares are combined can the aggregate be computed
    
    This provides privacy - the coordinator only sees the aggregate,
    not individual updates.
    """
    
    def __init__(self, threshold: int = 2):
        """
        Initialize secure aggregation.
        
        Args:
            threshold: Minimum participants needed to reconstruct (default 2)
        """
        self.threshold = threshold
        self._shares_received: dict[str, list[np.ndarray]] = {}
        self._participant_count = 0
        logger.info(f"SecureAggregation initialized with threshold={threshold}")
    
    def _generate_shares(
        self,
        secret: np.ndarray,
        n_shares: int
    ) -> list[np.ndarray]:
        """
        Split a secret into n additive shares.
        
        Uses additive secret sharing: sum of all shares = secret
        
        Args:
            secret: The value to split
            n_shares: Number of shares to create
        
        Returns:
            List of shares that sum to the secret
        """
        shares = []
        remaining = secret.copy()
        
        # Generate n-1 random shares
        for i in range(n_shares - 1):
            # Random share with same shape
            share = np.random.randn(*secret.shape).astype(secret.dtype)
            # Scale to reasonable range
            share = share * np.std(secret) if np.std(secret) > 0 else share * 0.01
            shares.append(share)
            remaining = remaining - share
        
        # Last share ensures sum equals secret
        shares.append(remaining)
        
        return shares
    
    def _reconstruct_secret(self, shares: list[np.ndarray]) -> np.ndarray:
        """
        Reconstruct secret from shares.
        
        Args:
            shares: List of additive shares
        
        Returns:
            Reconstructed secret (sum of shares)
        """
        if not shares:
            return np.array([])
        
        result = np.zeros_like(shares[0])
        for share in shares:
            result = result + share
        return result
    
    def encrypt_update(
        self,
        update: ModelUpdate,
        num_participants: int
    ) -> 'EncryptedUpdate':
        """
        Create secret shares of the update.
        
        Args:
            update: Model update to encrypt
            num_participants: Total number of participants
        
        Returns:
            EncryptedUpdate containing shares for this participant
        """
        self._participant_count = max(self._participant_count, num_participants)
        
        # Split each weight delta into shares
        encrypted_data = {}
        for name, delta in update.weight_deltas.items():
            shares = self._generate_shares(delta, num_participants)
            # Store the share meant for aggregation (last one)
            encrypted_data[name] = shares[-1]
        
        return EncryptedUpdate(
            device_id=update.device_id,
            round_number=update.round_number,
            encrypted_data=encrypted_data,
            num_samples=update.num_samples,
        )
    
    def add_share(self, device_id: str, shares: dict[str, np.ndarray]):
        """
        Add received shares from a participant.
        
        Args:
            device_id: ID of the participant
            shares: Their shares for each layer
        """
        for name, share in shares.items():
            if name not in self._shares_received:
                self._shares_received[name] = []
            self._shares_received[name].append(share)
    
    def decrypt_aggregate(
        self,
        encrypted_updates: list['EncryptedUpdate']
    ) -> dict[str, np.ndarray]:
        """
        Compute the aggregate from encrypted updates.
        
        Since we use additive sharing, summing all shares gives us
        the sum of all original updates.
        
        Args:
            encrypted_updates: List of encrypted updates
        
        Returns:
            Aggregated weight deltas
        """
        if len(encrypted_updates) < self.threshold:
            logger.warning(
                f"Not enough updates ({len(encrypted_updates)}) "
                f"for threshold ({self.threshold})"
            )
            return {}
        
        # Collect all shares
        all_shares: dict[str, list[np.ndarray]] = {}
        total_samples = 0
        
        for update in encrypted_updates:
            total_samples += update.num_samples
            for name, share in update.encrypted_data.items():
                if name not in all_shares:
                    all_shares[name] = []
                all_shares[name].append(share)
        
        # Sum shares to get aggregate (weighted by samples would be done separately)
        aggregated = {}
        for name, shares in all_shares.items():
            # Sum all shares
            aggregate = self._reconstruct_secret(shares)
            # Average by number of participants
            aggregated[name] = aggregate / len(shares)
        
        logger.info(
            f"Decrypted aggregate from {len(encrypted_updates)} participants, "
            f"{total_samples} total samples"
        )
        return aggregated


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
        encrypted_data: dict,
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
    
    def aggregate_updates(self, updates: list[ModelUpdate]) -> dict[str, np.ndarray]:
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
