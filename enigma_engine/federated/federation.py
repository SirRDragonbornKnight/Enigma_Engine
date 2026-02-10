"""
================================================================================
FEDERATED LEARNING - CORE FEDERATION LOGIC
================================================================================

Main federated learning coordinator and participant interfaces.

FILE: enigma_engine/federated/federation.py
TYPE: Federated Learning Core
MAIN CLASSES: FederatedLearning, FederationRole, FederationMode, ModelUpdate

HOW IT WORKS:
    1. Each device trains on its own data
    2. Devices share only weight updates (gradients)
    3. Coordinator aggregates updates
    4. Improved model distributed back to devices
    5. No raw data ever leaves device

USAGE:
    # Create federation
    fl = FederatedLearning(role=FederationRole.COORDINATOR)
    fed_id = fl.create_federation("MyFed", FederationMode.PRIVATE)
    
    # Join federation
    fl = FederatedLearning(role=FederationRole.PARTICIPANT)
    fl.join_federation(fed_id)
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..config import CONFIG

logger = logging.getLogger(__name__)


class FederationRole(Enum):
    """Role a device plays in federated learning."""
    COORDINATOR = "coordinator"  # Aggregates updates (usually main PC)
    PARTICIPANT = "participant"  # Contributes updates (all devices)
    OBSERVER = "observer"        # Receives updates but doesn't contribute


class FederationMode(Enum):
    """Different federation modes for different use cases."""
    PRIVATE = "private"      # Your devices only
    TRUSTED = "trusted"      # Friends/trusted users
    PUBLIC = "public"        # Anyone can join
    HYBRID = "hybrid"        # Mix of trusted and public


@dataclass
class ModelUpdate:
    """
    Update to share with federation.
    
    Contains only weight deltas, not full weights or data.
    """
    device_id: str
    round_number: int
    weight_deltas: dict[str, np.ndarray]  # Only changes, not full weights
    num_samples: int                      # How much data used
    loss: float                           # Training loss
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'device_id': self.device_id,
            'round_number': self.round_number,
            'weight_deltas': {k: v.tolist() for k, v in self.weight_deltas.items()},
            'num_samples': self.num_samples,
            'loss': self.loss,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ModelUpdate':
        """Deserialize from dictionary."""
        return cls(
            device_id=data['device_id'],
            round_number=data['round_number'],
            weight_deltas={k: np.array(v) for k, v in data['weight_deltas'].items()},
            num_samples=data['num_samples'],
            loss=data['loss'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {}),
        )


@dataclass
class FederationInfo:
    """Information about a federation."""
    id: str
    name: str
    coordinator: str          # Device ID
    mode: FederationMode
    participants: int
    current_round: int
    requires_password: bool
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'coordinator': self.coordinator,
            'mode': self.mode.value,
            'participants': self.participants,
            'current_round': self.current_round,
            'requires_password': self.requires_password,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FederationInfo':
        """Deserialize from dictionary."""
        return cls(
            id=data['id'],
            name=data['name'],
            coordinator=data['coordinator'],
            mode=FederationMode(data['mode']),
            participants=data['participants'],
            current_round=data['current_round'],
            requires_password=data['requires_password'],
            description=data.get('description', ''),
            created_at=datetime.fromisoformat(data['created_at']),
        )


class FederatedLearning:
    """
    Federated learning coordinator.
    
    How it works:
    1. Each device trains on its own data
    2. Devices share only weight updates (gradients)
    3. Coordinator aggregates updates
    4. Improved model distributed back to devices
    5. No raw data ever leaves device
    """
    
    def __init__(self, role: FederationRole = FederationRole.PARTICIPANT):
        """
        Initialize federated learning.
        
        Args:
            role: Role this device plays (coordinator, participant, observer)
        """
        self.role = role
        self.device_id = self._get_device_id()
        self.current_federation: Optional[str] = None
        self.current_round = 0
        self.federations: dict[str, FederationInfo] = {}
        
        # Lazy imports to avoid circular dependencies
        self._coordinator = None
        self._participant = None
        self._aggregator = None
        self._privacy = None
        self._compressor = None
        
        logger.info(f"Federated learning initialized with role: {role.value}")
    
    def _get_device_id(self) -> str:
        """Get or generate unique device ID."""
        device_id_file = Path(CONFIG.get("info_dir", "information")) / "device_id.txt"
        device_id_file.parent.mkdir(parents=True, exist_ok=True)
        
        if device_id_file.exists():
            return device_id_file.read_text().strip()
        else:
            device_id = str(uuid.uuid4())
            device_id_file.write_text(device_id)
            return device_id
    
    @property
    def coordinator(self):
        """Lazy load coordinator."""
        if self._coordinator is None:
            from .coordinator import FederatedCoordinator
            self._coordinator = FederatedCoordinator()
        return self._coordinator
    
    @property
    def participant(self):
        """Lazy load participant."""
        if self._participant is None:
            from .participant import FederatedParticipant
            self._participant = FederatedParticipant()
        return self._participant
    
    @property
    def aggregator(self):
        """Lazy load aggregator."""
        if self._aggregator is None:
            from .aggregation import FederatedAggregator
            self._aggregator = FederatedAggregator()
        return self._aggregator
    
    @property
    def privacy(self):
        """Lazy load privacy module."""
        if self._privacy is None:
            from .privacy import DifferentialPrivacy
            epsilon = CONFIG.get("federated_learning", {}).get("privacy", {}).get("epsilon", 1.0)
            self._privacy = DifferentialPrivacy(epsilon=epsilon)
        return self._privacy
    
    @property
    def compressor(self):
        """Lazy load compressor."""
        if self._compressor is None:
            from .compression import UpdateCompressor
            self._compressor = UpdateCompressor()
        return self._compressor
    
    def create_federation(
        self,
        name: str,
        mode: FederationMode,
        description: str = "",
        password: Optional[str] = None
    ) -> str:
        """
        Create a new federation.
        
        Args:
            name: Federation name
            mode: Federation mode (private, trusted, public, hybrid)
            description: Optional description
            password: Optional password for joining
        
        Returns:
            Federation ID
        """
        if self.role != FederationRole.COORDINATOR:
            raise ValueError("Only coordinators can create federations")
        
        federation_id = str(uuid.uuid4())
        
        federation_info = FederationInfo(
            id=federation_id,
            name=name,
            coordinator=self.device_id,
            mode=mode,
            participants=1,  # Creator is first participant
            current_round=0,
            requires_password=password is not None,
            description=description,
        )
        
        self.federations[federation_id] = federation_info
        self.current_federation = federation_id
        
        logger.info(f"Created federation '{name}' with ID {federation_id}")
        return federation_id
    
    def join_federation(self, federation_id: str, password: Optional[str] = None):
        """
        Join an existing federation.
        
        Args:
            federation_id: Federation ID to join
            password: Password if required
        """
        # In real implementation, this would contact coordinator
        # For now, just mark as joined
        self.current_federation = federation_id
        logger.info(f"Joined federation {federation_id}")
    
    def leave_federation(self, federation_id: str):
        """
        Leave a federation.
        
        Args:
            federation_id: Federation ID to leave
        """
        if self.current_federation == federation_id:
            self.current_federation = None
        logger.info(f"Left federation {federation_id}")
    
    def get_federation_info(self, federation_id: str) -> Optional[FederationInfo]:
        """
        Get information about a federation.
        
        Args:
            federation_id: Federation ID
        
        Returns:
            Federation info or None if not found
        """
        return self.federations.get(federation_id)
    
    def list_federations(self) -> list[FederationInfo]:
        """
        List all available federations.
        
        Returns:
            List of federation info
        """
        return list(self.federations.values())
    
    def train_local(self, model, dataset, **kwargs) -> ModelUpdate:
        """
        Train on local data using Enigma AI Engine training infrastructure.
        
        Args:
            model: Model to train
            dataset: Local dataset (list of samples or DataLoader)
            **kwargs: Training parameters (epochs, lr, etc.)
        
        Returns:
            ModelUpdate with weight deltas (not full weights for privacy)
        """
        import torch

        # Store initial weights
        initial_weights = {
            name: param.detach().cpu().numpy().copy()
            for name, param in model.named_parameters()
        }
        
        # Get training parameters
        epochs = kwargs.get('epochs', 1)
        learning_rate = kwargs.get('lr', kwargs.get('learning_rate', 1e-4))
        batch_size = kwargs.get('batch_size', 4)
        
        logger.info(f"Training locally on {len(dataset)} samples for {epochs} epoch(s)")
        
        # Try to use Enigma AI Engine training infrastructure
        try:
            from enigma_engine.core.training import Trainer, TrainingConfig
            
            config = TrainingConfig(
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                save_checkpoints=False,  # Don't save during federated training
            )
            
            trainer = Trainer(model, config)
            result = trainer.train(dataset)
            final_loss = result.get('final_loss', 0.0) if result else 0.0
            
        except ImportError:
            # Fallback to basic training loop
            logger.debug("Using fallback training loop")
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            
            total_loss = 0.0
            num_batches = 0
            
            for epoch in range(epochs):
                for i in range(0, len(dataset), batch_size):
                    batch = dataset[i:i + batch_size]
                    
                    # Handle different dataset formats
                    if isinstance(batch, dict):
                        input_ids = batch.get('input_ids')
                        labels = batch.get('labels', input_ids)
                    elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        input_ids, labels = batch[0], batch[1]
                    else:
                        continue
                    
                    if input_ids is None:
                        continue
                    
                    optimizer.zero_grad()
                    
                    try:
                        outputs = model(input_ids)
                        if hasattr(outputs, 'loss') and outputs.loss is not None:
                            loss = outputs.loss
                        else:
                            # Compute cross-entropy loss
                            loss = torch.nn.functional.cross_entropy(
                                outputs.view(-1, outputs.size(-1)),
                                labels.view(-1)
                            )
                        
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item()
                        num_batches += 1
                    except Exception as e:
                        logger.debug(f"Batch training error: {e}")
                        continue
            
            final_loss = total_loss / max(num_batches, 1)
        
        # Calculate weight deltas (what changed during training)
        weight_deltas = {}
        for name, param in model.named_parameters():
            current_weight = param.detach().cpu().numpy()
            weight_deltas[name] = current_weight - initial_weights[name]
        
        # Create update
        update = ModelUpdate(
            device_id=self.device_id,
            round_number=self.current_round,
            weight_deltas=weight_deltas,
            num_samples=len(dataset),
            loss=final_loss,
        )
        
        logger.info(f"Training completed: {len(weight_deltas)} layers, loss={final_loss:.4f}")
        return update
    
    def send_update(self, update: ModelUpdate):
        """
        Send weight updates to coordinator.
        
        Args:
            update: Model update to send
        """
        # Apply privacy protection
        if CONFIG.get("federated_learning", {}).get("privacy", {}).get("differential_privacy", True):
            update = self.privacy.add_noise(update)
        
        # Compress update
        if CONFIG.get("federated_learning", {}).get("compression", {}).get("enabled", True):
            compressed = self.compressor.compress(update)
            logger.info(f"Compressed update from {self._estimate_size(update)} to {self._estimate_size(compressed)} bytes")
        
        # In real implementation, send over network
        logger.info(f"Sending update for round {update.round_number}")
    
    def receive_aggregated_model(self, model_weights: dict[str, np.ndarray]):
        """
        Receive improved model from coordinator.
        
        Args:
            model_weights: Aggregated model weights
        """
        logger.info(f"Received aggregated model with {len(model_weights)} layers")
        # In real implementation, apply to local model
    
    def _estimate_size(self, obj) -> int:
        """Estimate size of object in bytes."""
        # Rough estimate
        if hasattr(obj, 'weight_deltas'):
            return sum(v.nbytes for v in obj.weight_deltas.values())
        return 0


class FederationManager:
    """
    Manage federation membership and policies.
    """
    
    def __init__(self):
        """Initialize federation manager."""
        self.federations: dict[str, FederationInfo] = {}
        self.invites: dict[str, str] = {}  # invite_code -> federation_id
        
    def create_federation(
        self,
        name: str,
        mode: FederationMode,
        coordinator_id: str,
        description: str = ""
    ) -> str:
        """
        Create a new federation.
        
        Args:
            name: Federation name
            mode: Federation mode
            coordinator_id: Device ID of coordinator
            description: Optional description
        
        Returns:
            Federation ID
        """
        federation_id = str(uuid.uuid4())
        
        federation_info = FederationInfo(
            id=federation_id,
            name=name,
            coordinator=coordinator_id,
            mode=mode,
            participants=1,
            current_round=0,
            requires_password=False,
            description=description,
        )
        
        self.federations[federation_id] = federation_info
        logger.info(f"Created federation '{name}' with ID {federation_id}")
        return federation_id
    
    def join_federation(self, federation_id: str, password: Optional[str] = None):
        """
        Join an existing federation.
        
        Args:
            federation_id: Federation ID to join
            password: Password if required
        """
        federation = self.federations.get(federation_id)
        if not federation:
            raise ValueError(f"Federation {federation_id} not found")
        
        if federation.requires_password and not password:
            raise ValueError("Password required to join this federation")
        
        federation.participants += 1
        logger.info(f"Joined federation {federation_id}")
    
    def leave_federation(self, federation_id: str):
        """
        Leave federation.
        
        Args:
            federation_id: Federation ID to leave
        """
        federation = self.federations.get(federation_id)
        if federation:
            federation.participants = max(0, federation.participants - 1)
            logger.info(f"Left federation {federation_id}")
    
    def invite_device(self, federation_id: str) -> str:
        """
        Generate invite code for device.
        
        Args:
            federation_id: Federation ID
        
        Returns:
            Invite code
        """
        invite_code = str(uuid.uuid4())[:8]
        self.invites[invite_code] = federation_id
        logger.info(f"Generated invite code {invite_code} for federation {federation_id}")
        return invite_code
    
    def redeem_invite(self, invite_code: str) -> Optional[str]:
        """
        Redeem invite code.
        
        Args:
            invite_code: Invite code
        
        Returns:
            Federation ID or None if invalid
        """
        return self.invites.get(invite_code)
