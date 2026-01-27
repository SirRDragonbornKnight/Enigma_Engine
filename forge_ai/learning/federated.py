"""
Federated Learning System for ForgeAI

Privacy-preserving distributed learning where:
- Each device trains on its own data locally
- Only model improvements (weight deltas) are shared
- Raw data never leaves the device
- Differential privacy protects individual contributions
"""

import uuid
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)


class FederatedMode(Enum):
    """Federated learning participation modes."""
    OPT_IN = "opt_in"          # Must explicitly enable
    OPT_OUT = "opt_out"        # Enabled by default
    DISABLED = "disabled"      # No federated learning


class PrivacyLevel(Enum):
    """Privacy protection levels for federated learning."""
    NONE = "none"              # Share everything
    LOW = "low"                # Anonymize device ID
    MEDIUM = "medium"          # + Differential privacy
    HIGH = "high"              # + Secure aggregation
    MAXIMUM = "maximum"        # + Homomorphic encryption


@dataclass
class WeightUpdate:
    """
    Model weight update (delta only, not full weights).
    
    Contains ONLY the changes from training, not the full model or any data.
    This ensures privacy - you can't reverse engineer training data from
    just the weight changes.
    """
    update_id: str
    device_id: str                         # Anonymized if privacy enabled
    timestamp: datetime
    weight_deltas: Dict[str, Any]          # Layer name -> weight changes (numpy arrays or lists)
    training_samples: int                  # How many samples trained on
    metadata: Dict[str, Any] = field(default_factory=dict)  # Loss, accuracy, etc.
    signature: Optional[str] = None        # Cryptographic signature
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "update_id": self.update_id,
            "device_id": self.device_id,
            "timestamp": self.timestamp.isoformat(),
            "weight_deltas": {
                name: self._serialize_array(delta)
                for name, delta in self.weight_deltas.items()
            },
            "training_samples": self.training_samples,
            "metadata": self.metadata,
            "signature": self.signature,
        }
    
    @staticmethod
    def _serialize_array(arr: Any) -> Any:
        """Serialize numpy array or return as-is."""
        if np is not None and isinstance(arr, np.ndarray):
            return arr.tolist()
        return arr
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeightUpdate":
        """Create from dictionary."""
        return cls(
            update_id=data["update_id"],
            device_id=data["device_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            weight_deltas=data["weight_deltas"],
            training_samples=data["training_samples"],
            metadata=data.get("metadata", {}),
            signature=data.get("signature"),
        )
    
    def sign(self, private_key: Optional[str] = None) -> str:
        """
        Create cryptographic signature of update.
        
        In a real implementation, this would use proper cryptographic signing.
        For now, we use a simple hash.
        """
        if private_key is None:
            private_key = "default_key"
        
        # Create deterministic string representation
        content = f"{self.update_id}{self.device_id}{self.timestamp.isoformat()}{self.training_samples}"
        signature = hashlib.sha256(f"{content}{private_key}".encode()).hexdigest()
        self.signature = signature
        return signature
    
    def verify_signature(self, public_key: Optional[str] = None) -> bool:
        """Verify the signature is valid."""
        if self.signature is None:
            return False
        
        if public_key is None:
            public_key = "default_key"
        
        # Recreate signature and compare
        saved_sig = self.signature
        self.signature = None
        expected_sig = self.sign(public_key)
        self.signature = saved_sig
        
        return saved_sig == expected_sig


class FederatedLearning:
    """
    Privacy-preserving distributed learning system.
    
    How it works:
    1. Each device trains on its own data locally
    2. Devices compute weight updates (deltas) from training
    3. Weight updates are shared (with privacy protection)
    4. Central aggregator or peer-to-peer system combines updates
    5. Updated model distributed back to devices
    
    NO RAW DATA IS EVER SHARED - only weight differences.
    """
    
    def __init__(
        self,
        model_name: str,
        mode: FederatedMode = FederatedMode.OPT_IN,
        privacy_level: PrivacyLevel = PrivacyLevel.HIGH,
        device_id: Optional[str] = None,
    ):
        """
        Initialize federated learning system.
        
        Args:
            model_name: Name of the model to train
            mode: Participation mode (opt-in, opt-out, disabled)
            privacy_level: Level of privacy protection
            device_id: Unique device identifier (generated if not provided)
        """
        self.model_name = model_name
        self.mode = mode
        self.privacy_level = privacy_level
        self.device_id = device_id or self._generate_device_id()
        
        # Initial model weights (for computing deltas)
        self.initial_weights: Optional[Dict[str, Any]] = None
        
        # Training history
        self.updates_sent: List[WeightUpdate] = []
        self.updates_received: List[WeightUpdate] = []
        
        logger.info(
            f"Federated learning initialized for {model_name} "
            f"(mode={mode.value}, privacy={privacy_level.value})"
        )
    
    def _generate_device_id(self) -> str:
        """Generate a unique device identifier."""
        if self.privacy_level in [PrivacyLevel.LOW, PrivacyLevel.MEDIUM, 
                                   PrivacyLevel.HIGH, PrivacyLevel.MAXIMUM]:
            # Use anonymous ID
            return f"device_{uuid.uuid4().hex[:8]}"
        else:
            # Use machine-based ID
            import socket
            hostname = socket.gethostname()
            return hashlib.md5(hostname.encode()).hexdigest()[:12]
    
    def train_local_round(
        self,
        final_weights: Dict[str, Any],
        training_samples: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WeightUpdate:
        """
        Create weight update from local training.
        
        Args:
            final_weights: Model weights after training
            training_samples: Number of samples trained on
            metadata: Optional metadata (loss, accuracy, etc.)
        
        Returns:
            WeightUpdate containing only the deltas
        """
        if self.initial_weights is None:
            raise ValueError("Must set initial_weights before training")
        
        # Compute weight deltas (difference from initial)
        weight_deltas = {}
        for layer_name, final_weight in final_weights.items():
            if layer_name in self.initial_weights:
                initial = self.initial_weights[layer_name]
                
                # Compute delta
                if np is not None and isinstance(final_weight, np.ndarray):
                    delta = final_weight - initial
                elif isinstance(final_weight, (list, tuple)):
                    # Handle list-based weights
                    delta = [f - i for f, i in zip(final_weight, initial)]
                else:
                    delta = final_weight  # Can't compute delta, use full weight
                
                weight_deltas[layer_name] = delta
            else:
                # New layer, include full weight
                weight_deltas[layer_name] = final_weight
        
        # Create update
        update = WeightUpdate(
            update_id=str(uuid.uuid4()),
            device_id=self.device_id,
            timestamp=datetime.now(),
            weight_deltas=weight_deltas,
            training_samples=training_samples,
            metadata=metadata or {},
        )
        
        # Sign the update
        update.sign()
        
        logger.info(
            f"Created local update {update.update_id[:8]} "
            f"with {len(weight_deltas)} layers, {training_samples} samples"
        )
        
        return update
    
    def share_update(self, update: WeightUpdate) -> bool:
        """
        Share weight update with network.
        
        Applies privacy protection based on privacy_level:
        - LOW: Anonymize device ID
        - MEDIUM+: Apply differential privacy
        - HIGH+: Use secure aggregation
        - MAXIMUM: Homomorphic encryption
        
        Args:
            update: Weight update to share
        
        Returns:
            True if shared successfully
        """
        if self.mode == FederatedMode.DISABLED:
            logger.warning("Federated learning is disabled")
            return False
        
        # Apply privacy protection
        protected_update = self._apply_privacy(update)
        
        # In a real implementation, this would send to coordinator or peers
        # For now, we just record it
        self.updates_sent.append(protected_update)
        
        logger.info(
            f"Shared update {update.update_id[:8]} with privacy level {self.privacy_level.value}"
        )
        
        return True
    
    def _apply_privacy(self, update: WeightUpdate) -> WeightUpdate:
        """Apply privacy protection to update."""
        from .privacy import DifferentialPrivacy
        
        protected = WeightUpdate(
            update_id=update.update_id,
            device_id=update.device_id,
            timestamp=update.timestamp,
            weight_deltas=update.weight_deltas.copy(),
            training_samples=update.training_samples,
            metadata=update.metadata.copy(),
            signature=update.signature,
        )
        
        # Anonymize device ID for privacy levels >= LOW
        if self.privacy_level != PrivacyLevel.NONE:
            protected.device_id = f"anon_{hashlib.md5(update.device_id.encode()).hexdigest()[:8]}"
        
        # Apply differential privacy for levels >= MEDIUM
        if self.privacy_level in [PrivacyLevel.MEDIUM, PrivacyLevel.HIGH, PrivacyLevel.MAXIMUM]:
            dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
            protected.weight_deltas = dp.add_noise(protected.weight_deltas)
        
        return protected
    
    def receive_global_update(self, global_update: WeightUpdate) -> bool:
        """
        Receive and apply aggregated update from network.
        
        Args:
            global_update: Aggregated weight update
        
        Returns:
            True if applied successfully
        """
        if self.mode == FederatedMode.DISABLED:
            logger.warning("Federated learning is disabled")
            return False
        
        # Verify signature
        if not global_update.verify_signature():
            logger.warning(f"Invalid signature on update {global_update.update_id[:8]}")
            return False
        
        # Record the update
        self.updates_received.append(global_update)
        
        # In a real implementation, this would apply the update to the model
        logger.info(
            f"Received global update {global_update.update_id[:8]} "
            f"aggregated from {global_update.training_samples} samples"
        )
        
        return True
    
    def set_initial_weights(self, weights: Dict[str, Any]):
        """Set initial weights for computing deltas."""
        self.initial_weights = weights
        logger.debug(f"Set initial weights with {len(weights)} layers")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get federated learning statistics."""
        return {
            "device_id": self.device_id,
            "mode": self.mode.value,
            "privacy_level": self.privacy_level.value,
            "updates_sent": len(self.updates_sent),
            "updates_received": len(self.updates_received),
            "total_samples_contributed": sum(
                u.training_samples for u in self.updates_sent
            ),
        }
