"""
Federated Learning Coordinator

Manages federated learning rounds and coordinates training across devices.
Supports both centralized and decentralized (peer-to-peer) modes.
"""

import logging
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class CoordinatorMode(Enum):
    """Coordination modes for federated learning."""
    CENTRALIZED = "centralized"    # One coordinator
    PEER_TO_PEER = "p2p"           # Decentralized
    BLOCKCHAIN = "blockchain"      # Blockchain-based (future)


class RoundState(Enum):
    """States of a federated learning round."""
    IDLE = "idle"
    TRAINING = "training"
    AGGREGATING = "aggregating"
    COMPLETE = "complete"


class FederatedCoordinator:
    """
    Coordinate federated learning rounds.
    
    In CENTRALIZED mode:
        - This node acts as the coordinator
        - Other devices connect and submit updates
        - Coordinator aggregates and distributes results
    
    In PEER_TO_PEER mode:
        - No central coordinator
        - Devices communicate directly
        - Consensus algorithm determines global update
    """
    
    def __init__(
        self,
        mode: CoordinatorMode = CoordinatorMode.PEER_TO_PEER,
        min_participants: int = 2,
        round_timeout: int = 300,
        max_history_size: int = 100,
    ):
        """
        Initialize coordinator.
        
        Args:
            mode: Coordination mode (centralized or peer-to-peer)
            min_participants: Minimum devices needed for a round
            round_timeout: Seconds to wait for updates before finalizing
            max_history_size: Maximum round history entries to keep
        """
        self.mode = mode
        self.min_participants = min_participants
        self.round_timeout = round_timeout
        self._max_history_size = max_history_size
        
        # Round management
        self.current_round = 0
        self.round_state = RoundState.IDLE
        self.round_start_time: Optional[float] = None
        
        # Participants
        self.participants: set[str] = set()  # Device IDs
        self.authorized_devices: set[str] = set()  # Authorized device IDs
        
        # Updates for current round
        self.pending_updates: dict[str, Any] = {}  # device_id -> WeightUpdate
        
        # Round history
        self.round_history: list[dict[str, Any]] = []
        
        logger.info(
            f"Federated coordinator initialized "
            f"(mode={mode.value}, min_participants={min_participants})"
        )
    
    def register_device(self, device_id: str) -> bool:
        """
        Register a device as authorized to participate.
        
        Args:
            device_id: Unique device identifier
        
        Returns:
            True if registered successfully
        """
        self.authorized_devices.add(device_id)
        logger.info(f"Registered device: {device_id}")
        return True
    
    def unregister_device(self, device_id: str) -> bool:
        """
        Unregister a device.
        
        Args:
            device_id: Device to unregister
        
        Returns:
            True if unregistered successfully
        """
        self.authorized_devices.discard(device_id)
        self.participants.discard(device_id)
        logger.info(f"Unregistered device: {device_id}")
        return True
    
    def start_round(self) -> str:
        """
        Start a new training round.
        
        Returns:
            Round ID
        """
        if self.round_state != RoundState.IDLE:
            raise RuntimeError(
                f"Cannot start round in state {self.round_state.value}"
            )
        
        self.current_round += 1
        self.round_state = RoundState.TRAINING
        self.round_start_time = time.time()
        self.pending_updates.clear()
        
        round_id = f"round_{self.current_round}_{uuid.uuid4().hex[:8]}"
        
        logger.info(
            f"Started round {self.current_round} ({round_id}) "
            f"with {len(self.authorized_devices)} authorized devices"
        )
        
        # In a real implementation, would broadcast round start to all devices
        
        return round_id
    
    def accept_update(self, update: Any) -> bool:  # update: WeightUpdate
        """
        Accept a weight update from a device.
        
        Validates:
        - Update is for current round
        - Device is authorized
        - Update passes sanity checks
        
        Args:
            update: WeightUpdate from device
        
        Returns:
            True if accepted
        """
        if self.round_state != RoundState.TRAINING:
            logger.warning(
                f"Rejected update in state {self.round_state.value}"
            )
            return False
        
        device_id = update.device_id
        
        # Check authorization (skip for anonymized IDs)
        if not device_id.startswith("anon_"):
            if device_id not in self.authorized_devices:
                logger.warning(f"Rejected update from unauthorized device: {device_id}")
                return False
        
        # Verify signature
        if not update.verify_signature():
            logger.warning(f"Rejected update with invalid signature from {device_id}")
            return False
        
        # Sanity check: reasonable number of training samples
        if update.training_samples <= 0:
            logger.warning(f"Rejected update with invalid sample count: {update.training_samples}")
            return False
        
        # Sanity check: has weight deltas
        if not update.weight_deltas:
            logger.warning(f"Rejected update with no weight deltas")
            return False
        
        # Accept the update
        self.pending_updates[device_id] = update
        self.participants.add(device_id)
        
        logger.info(
            f"Accepted update from {device_id} "
            f"({len(self.pending_updates)}/{self.min_participants} received)"
        )
        
        # Check if we should finalize
        if self._should_finalize_round():
            self._finalize_round()
        
        return True
    
    def _should_finalize_round(self) -> bool:
        """Check if round should be finalized."""
        if self.round_state != RoundState.TRAINING:
            return False
        
        # Enough participants?
        if len(self.pending_updates) < self.min_participants:
            return False
        
        # Timeout reached?
        if self.round_start_time is not None:
            elapsed = time.time() - self.round_start_time
            if elapsed >= self.round_timeout:
                return True
        
        # All authorized devices submitted?
        if len(self.pending_updates) >= len(self.authorized_devices):
            return True
        
        return False
    
    def finalize_round(self) -> Optional[Any]:  # Returns Optional[WeightUpdate]
        """
        Manually finalize the current round.
        
        Returns:
            Aggregated update, or None if not enough participants
        """
        return self._finalize_round()
    
    def _finalize_round(self) -> Optional[Any]:
        """Internal round finalization."""
        if self.round_state != RoundState.TRAINING:
            logger.warning(f"Cannot finalize round in state {self.round_state.value}")
            return None
        
        if len(self.pending_updates) < self.min_participants:
            logger.warning(
                f"Not enough participants to finalize round "
                f"({len(self.pending_updates)}/{self.min_participants})"
            )
            self.round_state = RoundState.IDLE
            return None
        
        self.round_state = RoundState.AGGREGATING
        
        # Aggregate updates
        from .aggregation import AggregationMethod, SecureAggregator
        
        aggregator = SecureAggregator()
        updates = list(self.pending_updates.values())
        
        global_update = aggregator.aggregate_updates(
            updates,
            method=AggregationMethod.WEIGHTED,
        )
        
        # Sign the global update
        global_update.sign()
        
        # Record round completion
        elapsed = time.time() - self.round_start_time if self.round_start_time else 0
        
        round_info = {
            "round": self.current_round,
            "timestamp": datetime.now(),
            "participants": len(self.pending_updates),
            "total_samples": global_update.training_samples,
            "duration_seconds": elapsed,
            "update_id": global_update.update_id,
        }
        
        self.round_history.append(round_info)
        # Trim history to prevent unbounded growth
        if len(self.round_history) > self._max_history_size:
            self.round_history = self.round_history[-self._max_history_size:]
        
        logger.info(
            f"Finalized round {self.current_round}: "
            f"{len(self.pending_updates)} participants, "
            f"{global_update.training_samples} total samples, "
            f"{elapsed:.1f}s elapsed"
        )
        
        # Reset for next round
        self.round_state = RoundState.COMPLETE
        self.pending_updates.clear()
        
        # In a real implementation, would distribute global_update to all devices
        
        # Ready for next round
        self.round_state = RoundState.IDLE
        
        return global_update
    
    def get_round_status(self) -> dict[str, Any]:
        """Get current round status."""
        return {
            "current_round": self.current_round,
            "state": self.round_state.value,
            "mode": self.mode.value,
            "participants": len(self.participants),
            "pending_updates": len(self.pending_updates),
            "authorized_devices": len(self.authorized_devices),
            "min_participants": self.min_participants,
            "elapsed_seconds": (
                time.time() - self.round_start_time
                if self.round_start_time else 0
            ),
        }
    
    def get_stats(self) -> dict[str, Any]:
        """Get coordinator statistics."""
        return {
            "mode": self.mode.value,
            "total_rounds": self.current_round,
            "current_state": self.round_state.value,
            "authorized_devices": len(self.authorized_devices),
            "recent_rounds": self.round_history[-5:] if self.round_history else [],
        }


def test_coordinator():
    """Test coordinator functionality."""
    from .federated import WeightUpdate

    # Create coordinator
    coordinator = FederatedCoordinator(
        mode=CoordinatorMode.CENTRALIZED,
        min_participants=2,
        round_timeout=60,
    )
    
    # Register devices
    coordinator.register_device("device1")
    coordinator.register_device("device2")
    coordinator.register_device("device3")
    
    # Start round
    round_id = coordinator.start_round()
    print(f"Started {round_id}")
    print(f"Status: {coordinator.get_round_status()}")
    
    # Submit updates
    try:
        import numpy as np
        
        update1 = WeightUpdate(
            update_id=str(uuid.uuid4()),
            device_id="device1",
            timestamp=datetime.now(),
            weight_deltas={"layer1": np.array([1.0, 2.0])},
            training_samples=10,
        )
        update1.sign()
        
        update2 = WeightUpdate(
            update_id=str(uuid.uuid4()),
            device_id="device2",
            timestamp=datetime.now(),
            weight_deltas={"layer1": np.array([2.0, 3.0])},
            training_samples=20,
        )
        update2.sign()
        
        coordinator.accept_update(update1)
        coordinator.accept_update(update2)
        
        print(f"\nFinal status: {coordinator.get_round_status()}")
        print(f"Stats: {coordinator.get_stats()}")
        
    except ImportError:
        print("NumPy not available, skipping update submission test")


if __name__ == "__main__":
    test_coordinator()
