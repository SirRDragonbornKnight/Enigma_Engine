"""
Federated Learning System - Privacy-Preserving Distributed Learning

This module implements the core federated learning system where:
1. Each device trains on its own data locally
2. Devices share weight updates (not data)
3. Central aggregator (or peer-to-peer) combines updates
4. Updated model distributed back to devices
"""

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import CONFIG

logger = logging.getLogger(__name__)


class FederatedMode(Enum):
    """Participation mode for federated learning."""
    OPT_IN = "opt_in"          # Must explicitly enable
    OPT_OUT = "opt_out"        # Enabled by default
    DISABLED = "disabled"      # No federated learning


class PrivacyLevel(Enum):
    """Privacy protection level for updates."""
    NONE = "none"              # Share everything
    LOW = "low"                # Anonymize device ID
    MEDIUM = "medium"          # + Differential privacy
    HIGH = "high"              # + Secure aggregation
    MAXIMUM = "maximum"        # + Homomorphic encryption


@dataclass
class WeightUpdate:
    """
    Model weight update (delta).
    
    Contains ONLY the changes, not the full model or any data.
    """
    update_id: str
    device_id: str                    # Anonymized if privacy enabled
    timestamp: datetime
    weight_deltas: Dict[str, np.ndarray]  # Layer name -> weight changes
    training_samples: int             # How many samples trained on
    metadata: Dict[str, Any] = field(default_factory=dict)  # Loss, accuracy, etc.
    signature: Optional[str] = None   # Cryptographic signature
    
    def __post_init__(self):
        """Validate the update."""
        if not self.update_id:
            self.update_id = str(uuid.uuid4())
        if not isinstance(self.timestamp, datetime):
            self.timestamp = datetime.now()
    
    def get_size(self) -> int:
        """Get the total size of weight deltas in bytes."""
        total = 0
        for delta in self.weight_deltas.values():
            total += delta.nbytes
        return total
    
    def sign(self, private_key: Optional[str] = None) -> str:
        """
        Generate cryptographic signature for this update.
        
        Args:
            private_key: Optional private key for signing
            
        Returns:
            Signature string
        """
        # Create hash of update data
        data = f"{self.update_id}|{self.device_id}|{self.timestamp.isoformat()}"
        
        # Add weight delta hashes
        for name in sorted(self.weight_deltas.keys()):
            delta_hash = hashlib.sha256(self.weight_deltas[name].tobytes()).hexdigest()
            data += f"|{name}:{delta_hash}"
        
        # Sign with private key if provided, otherwise just hash
        if private_key:
            data += f"|{private_key}"
        
        signature = hashlib.sha256(data.encode()).hexdigest()
        self.signature = signature
        return signature
    
    def verify_signature(self, public_key: Optional[str] = None) -> bool:
        """
        Verify the signature of this update.
        
        Args:
            public_key: Optional public key for verification
            
        Returns:
            True if signature is valid
        """
        if not self.signature:
            return False
        
        # Recreate signature and compare
        old_sig = self.signature
        self.signature = None
        new_sig = self.sign(public_key)
        self.signature = old_sig
        
        return old_sig == new_sig


class FederatedLearning:
    """
    Privacy-preserving distributed learning.
    
    How it works:
    1. Each device trains on its own data locally
    2. Devices share weight updates (not data)
    3. Central aggregator (or peer-to-peer) combines updates
    4. Updated model distributed back to devices
    """
    
    def __init__(
        self,
        mode: FederatedMode = FederatedMode.OPT_IN,
        privacy_level: PrivacyLevel = PrivacyLevel.HIGH,
        device_id: Optional[str] = None
    ):
        """
        Initialize federated learning system.
        
        Args:
            mode: Participation mode
            privacy_level: Privacy protection level
            device_id: Unique device identifier (auto-generated if None)
        """
        self.mode = mode
        self.privacy_level = privacy_level
        self.device_id = device_id or self._generate_device_id()
        
        # Track local training state
        self.current_round = 0
        self.local_epochs = 0
        self.last_update_time = None
        
        # Model weights before training (for computing deltas)
        self.base_weights: Optional[Dict[str, np.ndarray]] = None
        
        # Privacy components (lazy loaded)
        self._privacy = None
        
        logger.info(
            f"Initialized FederatedLearning: mode={mode.value}, "
            f"privacy={privacy_level.value}, device={self.device_id[:8]}..."
        )
    
    def _generate_device_id(self) -> str:
        """Generate a unique device ID."""
        # Use machine-specific info if available
        try:
            import platform
            info = f"{platform.node()}|{platform.machine()}|{time.time()}"
            return hashlib.sha256(info.encode()).hexdigest()
        except Exception:
            return str(uuid.uuid4())
    
    def _anonymize_device_id(self) -> str:
        """Anonymize device ID based on privacy level."""
        if self.privacy_level == PrivacyLevel.NONE:
            return self.device_id
        
        # Hash the device ID
        anon_id = hashlib.sha256(self.device_id.encode()).hexdigest()
        
        # Truncate based on privacy level
        if self.privacy_level == PrivacyLevel.LOW:
            return anon_id[:16]
        else:
            return anon_id[:8]
    
    def start_local_round(self, model_weights: Dict[str, np.ndarray]) -> None:
        """
        Start a new local training round.
        
        Args:
            model_weights: Current model weights before training
        """
        self.base_weights = {
            name: weight.copy() for name, weight in model_weights.items()
        }
        self.current_round += 1
        logger.debug(f"Started local training round {self.current_round}")
    
    def train_local_round(
        self,
        model_weights: Dict[str, np.ndarray],
        training_samples: int,
        loss: float,
        accuracy: Optional[float] = None
    ) -> WeightUpdate:
        """
        Complete local training and create weight update.
        
        Args:
            model_weights: Model weights after training
            training_samples: Number of samples trained on
            loss: Training loss
            accuracy: Optional training accuracy
            
        Returns:
            Weight update containing only the deltas
        """
        if self.base_weights is None:
            raise RuntimeError("Must call start_local_round() before train_local_round()")
        
        # Compute weight deltas
        weight_deltas = {}
        for name, weight in model_weights.items():
            if name in self.base_weights:
                delta = weight - self.base_weights[name]
                weight_deltas[name] = delta
            else:
                logger.warning(f"New layer '{name}' found - including full weights")
                weight_deltas[name] = weight.copy()
        
        # Apply privacy if enabled
        if self.privacy_level in (PrivacyLevel.MEDIUM, PrivacyLevel.HIGH, PrivacyLevel.MAXIMUM):
            weight_deltas = self._apply_differential_privacy(weight_deltas)
        
        # Create update
        update = WeightUpdate(
            update_id=str(uuid.uuid4()),
            device_id=self._anonymize_device_id(),
            timestamp=datetime.now(),
            weight_deltas=weight_deltas,
            training_samples=training_samples,
            metadata={
                "round": self.current_round,
                "loss": loss,
                "accuracy": accuracy,
                "privacy_level": self.privacy_level.value,
            }
        )
        
        # Sign the update
        update.sign()
        
        self.local_epochs += 1
        self.last_update_time = datetime.now()
        
        logger.info(
            f"Created weight update: round={self.current_round}, "
            f"samples={training_samples}, loss={loss:.4f}"
        )
        
        return update
    
    def _apply_differential_privacy(
        self, 
        weight_deltas: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Apply differential privacy to weight deltas.
        
        Args:
            weight_deltas: Original weight deltas
            
        Returns:
            Noisy weight deltas
        """
        if self._privacy is None:
            from .privacy import DifferentialPrivacy
            
            # Get privacy parameters from config or use defaults
            epsilon = CONFIG.get("federated", {}).get("epsilon", 1.0)
            delta = CONFIG.get("federated", {}).get("delta", 1e-5)
            
            self._privacy = DifferentialPrivacy(epsilon=epsilon, delta=delta)
        
        return self._privacy.add_noise(weight_deltas)
    
    def share_update(self, update: WeightUpdate) -> bool:
        """
        Share weight update with network.
        
        Args:
            update: Weight update to share
            
        Returns:
            True if sharing was successful
        """
        # Check if federated learning is enabled
        if self.mode == FederatedMode.DISABLED:
            logger.debug("Federated learning is disabled - not sharing update")
            return False
        
        try:
            # This would integrate with the network module
            # For now, just log the intent
            logger.info(
                f"Sharing update {update.update_id[:8]}... "
                f"(size: {update.get_size()} bytes)"
            )
            
            # TODO: Integrate with forge_ai.comms.network to actually share
            # from ..comms.network import ForgeNode
            # node = ForgeNode()
            # node.broadcast_update(update)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to share update: {e}")
            return False
    
    def receive_global_update(self, global_update: WeightUpdate) -> Dict[str, np.ndarray]:
        """
        Receive aggregated update from network and apply to local model.
        
        Args:
            global_update: Aggregated update from all devices
            
        Returns:
            New model weights after applying update
        """
        logger.info(
            f"Received global update {global_update.update_id[:8]}... "
            f"from round {global_update.metadata.get('round', 'unknown')}"
        )
        
        # Verify signature if present
        if global_update.signature:
            if not global_update.verify_signature():
                logger.warning("Global update has invalid signature!")
        
        # Apply deltas to current weights
        if self.base_weights is None:
            logger.warning("No base weights - cannot apply global update")
            return {}
        
        new_weights = {}
        for name, base_weight in self.base_weights.items():
            if name in global_update.weight_deltas:
                delta = global_update.weight_deltas[name]
                new_weights[name] = base_weight + delta
            else:
                new_weights[name] = base_weight.copy()
        
        # Update base weights for next round
        self.base_weights = new_weights
        
        return new_weights
    
    def get_status(self) -> Dict[str, Any]:
        """Get current federated learning status."""
        return {
            "device_id": self.device_id[:16] + "...",
            "mode": self.mode.value,
            "privacy_level": self.privacy_level.value,
            "current_round": self.current_round,
            "local_epochs": self.local_epochs,
            "last_update": self.last_update_time.isoformat() if self.last_update_time else None,
        }
