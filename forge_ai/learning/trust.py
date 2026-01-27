"""
Trust Management for Federated Learning

Manage trust between devices and detect malicious/poisoned updates:
- Track device reputation
- Detect Byzantine attacks
- Filter suspicious updates
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

from .federated import WeightUpdate

logger = logging.getLogger(__name__)


@dataclass
class DeviceTrust:
    """Trust information for a device."""
    device_id: str
    trust_score: float = 1.0  # 0.0 (untrusted) to 1.0 (fully trusted)
    total_updates: int = 0
    accepted_updates: int = 0
    rejected_updates: int = 0
    last_update: Optional[datetime] = None
    reputation_history: List[float] = field(default_factory=list)
    
    def update_trust(self, accepted: bool, weight: float = 0.1) -> None:
        """
        Update trust score based on update acceptance.
        
        Args:
            accepted: Whether the update was accepted
            weight: Weight for exponential moving average (0-1)
        """
        self.total_updates += 1
        
        if accepted:
            self.accepted_updates += 1
            # Increase trust slightly
            new_score = 1.0
        else:
            self.rejected_updates += 1
            # Decrease trust more significantly
            new_score = 0.0
        
        # Exponential moving average
        self.trust_score = (1 - weight) * self.trust_score + weight * new_score
        
        # Clamp to [0, 1]
        self.trust_score = max(0.0, min(1.0, self.trust_score))
        
        # Record in history
        self.reputation_history.append(self.trust_score)
        self.last_update = datetime.now()


class TrustManager:
    """
    Manage trust between devices and detect malicious updates.
    
    Defends against:
    - Byzantine attacks (malicious devices)
    - Model poisoning
    - Data poisoning
    - Sybil attacks (fake devices)
    """
    
    def __init__(
        self,
        min_trust_score: float = 0.3,
        byzantine_threshold: float = 3.0,
        history_window: int = 10
    ):
        """
        Initialize trust manager.
        
        Args:
            min_trust_score: Minimum trust score to accept updates
            byzantine_threshold: Standard deviations for Byzantine detection
            history_window: Number of recent updates to track per device
        """
        self.min_trust_score = min_trust_score
        self.byzantine_threshold = byzantine_threshold
        self.history_window = history_window
        
        # Track device trust
        self.devices: Dict[str, DeviceTrust] = {}
        
        logger.info(
            f"Initialized TrustManager: "
            f"min_trust={min_trust_score}, "
            f"byzantine_threshold={byzantine_threshold}"
        )
    
    def evaluate_update(self, update: WeightUpdate) -> bool:
        """
        Evaluate if an update should be trusted.
        
        Args:
            update: Weight update to evaluate
            
        Returns:
            True if update is trusted
        """
        device_id = update.device_id
        
        # Get or create device trust
        if device_id not in self.devices:
            self.devices[device_id] = DeviceTrust(device_id=device_id)
        
        device_trust = self.devices[device_id]
        
        # Check trust score
        if device_trust.trust_score < self.min_trust_score:
            logger.warning(
                f"Device {device_id[:8]}... has low trust score: "
                f"{device_trust.trust_score:.2f}"
            )
            device_trust.update_trust(accepted=False)
            return False
        
        # Check for Byzantine behavior
        if self._is_byzantine(update):
            logger.warning(
                f"Update {update.update_id[:8]}... appears Byzantine"
            )
            device_trust.update_trust(accepted=False)
            return False
        
        # Check signature
        if update.signature and not update.verify_signature():
            logger.warning(
                f"Update {update.update_id[:8]}... has invalid signature"
            )
            device_trust.update_trust(accepted=False)
            return False
        
        # Update accepted
        device_trust.update_trust(accepted=True)
        return True
    
    def _is_byzantine(self, update: WeightUpdate) -> bool:
        """
        Detect if update is potentially Byzantine (malicious).
        
        Uses statistical analysis to detect outliers.
        
        Args:
            update: Update to check
            
        Returns:
            True if update appears Byzantine
        """
        # Calculate magnitude of update
        magnitude = 0.0
        for delta in update.weight_deltas.values():
            magnitude += np.linalg.norm(delta)
        
        # Check if magnitude is suspiciously large
        # (In practice, would compare to historical distribution)
        
        # Simple heuristic: flag if magnitude > 1e10
        if magnitude > 1e10:
            logger.debug(f"Byzantine check: Large magnitude {magnitude:.2e}")
            return True
        
        # Check for NaN or Inf values
        for name, delta in update.weight_deltas.items():
            if np.any(np.isnan(delta)) or np.any(np.isinf(delta)):
                logger.debug(f"Byzantine check: Invalid values in {name}")
                return True
        
        return False
    
    def detect_poisoning(
        self, 
        updates: List[WeightUpdate]
    ) -> List[WeightUpdate]:
        """
        Detect and filter poisoned updates.
        
        Uses statistical methods to identify outliers that could be
        poisoning attacks.
        
        Args:
            updates: List of updates
            
        Returns:
            Filtered list without poisoned updates
        """
        if len(updates) < 3:
            # Not enough updates for statistical analysis
            return updates
        
        # Calculate magnitude for each update
        magnitudes = []
        for update in updates:
            mag = 0.0
            for delta in update.weight_deltas.values():
                mag += np.linalg.norm(delta)
            magnitudes.append(mag)
        
        # Calculate statistics
        mean_mag = np.mean(magnitudes)
        std_mag = np.std(magnitudes)
        
        # Filter outliers
        filtered = []
        for update, mag in zip(updates, magnitudes):
            # Z-score
            if std_mag > 0:
                z_score = abs(mag - mean_mag) / std_mag
            else:
                z_score = 0.0
            
            # Flag if z-score exceeds threshold
            if z_score > self.byzantine_threshold:
                logger.warning(
                    f"Update {update.update_id[:8]}... flagged as outlier: "
                    f"z-score={z_score:.2f}, magnitude={mag:.2e}"
                )
                
                # Update device trust
                device_id = update.device_id
                if device_id in self.devices:
                    self.devices[device_id].update_trust(accepted=False)
            else:
                filtered.append(update)
        
        logger.info(
            f"Poisoning detection: {len(filtered)}/{len(updates)} updates kept"
        )
        
        return filtered
    
    def get_device_trust(self, device_id: str) -> Optional[DeviceTrust]:
        """Get trust information for a device."""
        return self.devices.get(device_id)
    
    def get_trusted_devices(self) -> List[str]:
        """Get list of trusted device IDs."""
        return [
            device_id 
            for device_id, trust in self.devices.items()
            if trust.trust_score >= self.min_trust_score
        ]
    
    def ban_device(self, device_id: str) -> None:
        """
        Ban a device (set trust to 0).
        
        Args:
            device_id: Device to ban
        """
        if device_id in self.devices:
            self.devices[device_id].trust_score = 0.0
            logger.warning(f"Banned device {device_id[:8]}...")
        else:
            self.devices[device_id] = DeviceTrust(
                device_id=device_id,
                trust_score=0.0
            )
    
    def get_statistics(self) -> Dict:
        """Get trust statistics."""
        total_devices = len(self.devices)
        trusted_devices = len(self.get_trusted_devices())
        
        if total_devices > 0:
            avg_trust = np.mean([d.trust_score for d in self.devices.values()])
        else:
            avg_trust = 0.0
        
        return {
            "total_devices": total_devices,
            "trusted_devices": trusted_devices,
            "untrusted_devices": total_devices - trusted_devices,
            "average_trust_score": avg_trust,
            "min_trust_threshold": self.min_trust_score,
        }
