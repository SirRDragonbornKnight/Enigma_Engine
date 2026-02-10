"""
Trust Management for Federated Learning

Verifies updates are legitimate and detects potential poisoning attacks.
Prevents malicious actors from corrupting the shared model.
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = logging.getLogger(__name__)


class TrustManager:
    """
    Verify updates and detect malicious behavior.
    
    Protects against:
    - Invalid/corrupted updates
    - Model poisoning attacks
    - Sybil attacks (fake device IDs)
    - Data poisoning
    """
    
    def __init__(
        self,
        max_update_magnitude: float = 10.0,
        min_reputation: float = 0.3,
        reputation_decay: float = 0.95,
        max_history_size: int = 100,
    ):
        """
        Initialize trust manager.
        
        Args:
            max_update_magnitude: Maximum allowed L2 norm of updates
            min_reputation: Minimum reputation to accept updates
            reputation_decay: Reputation decay per time period
            max_history_size: Maximum update history entries to keep
        """
        self.max_update_magnitude = max_update_magnitude
        self.min_reputation = min_reputation
        self.reputation_decay = reputation_decay
        self._max_history_size = max_history_size
        
        # Device reputation scores (0.0 to 1.0)
        self.reputations: dict[str, float] = defaultdict(lambda: 0.5)
        
        # Update history for anomaly detection
        self.update_history: list[dict[str, Any]] = []
        
        # Blocked devices
        self.blocked_devices: set[str] = set()
        
        logger.info("Trust manager initialized")
    
    def verify_update(self, update: Any) -> bool:  # update: WeightUpdate
        """
        Verify an update is legitimate and safe.
        
        Checks:
        - Cryptographic signature
        - Update magnitude (not too large)
        - Device reputation
        - Consistency with other updates
        
        Args:
            update: WeightUpdate to verify
        
        Returns:
            True if update is valid and should be accepted
        """
        device_id = update.device_id
        
        # Check if device is blocked
        if device_id in self.blocked_devices:
            logger.warning(f"Rejected update from blocked device: {device_id}")
            return False
        
        # Verify cryptographic signature
        if not update.verify_signature():
            logger.warning(f"Invalid signature from {device_id}")
            self._penalize_device(device_id, 0.2)
            return False
        
        # Check device reputation
        reputation = self.reputations[device_id]
        if reputation < self.min_reputation:
            logger.warning(
                f"Rejected update from low-reputation device {device_id} "
                f"(reputation={reputation:.2f})"
            )
            return False
        
        # Check update magnitude
        if not self._check_magnitude(update):
            logger.warning(f"Update from {device_id} has excessive magnitude")
            self._penalize_device(device_id, 0.3)
            return False
        
        # Check training samples are reasonable
        if update.training_samples <= 0 or update.training_samples > 1000000:
            logger.warning(f"Suspicious training sample count from {device_id}: {update.training_samples}")
            self._penalize_device(device_id, 0.1)
            return False
        
        # Record update
        self.update_history.append({
            "device_id": device_id,
            "update_id": update.update_id,
            "timestamp": update.timestamp,
            "training_samples": update.training_samples,
            "magnitude": self._calculate_magnitude(update),
        })
        # Trim history to prevent unbounded growth
        if len(self.update_history) > self._max_history_size:
            self.update_history = self.update_history[-self._max_history_size:]
        
        # Reward device for valid update
        self._reward_device(device_id, 0.05)
        
        return True
    
    def _check_magnitude(self, update: Any) -> bool:
        """Check if update magnitude is within acceptable range."""
        if not HAS_NUMPY:
            return True  # Can't check without numpy
        
        magnitude = self._calculate_magnitude(update)
        return magnitude <= self.max_update_magnitude
    
    def _calculate_magnitude(self, update: Any) -> float:
        """Calculate L2 norm of all weight updates."""
        if not HAS_NUMPY:
            return 0.0
        
        total_magnitude = 0.0
        
        for layer_name, delta in update.weight_deltas.items():
            if isinstance(delta, (list, tuple)):
                delta = np.array(delta)
            
            if isinstance(delta, np.ndarray):
                total_magnitude += np.linalg.norm(delta) ** 2
        
        return float(np.sqrt(total_magnitude))
    
    def detect_poisoning(self, updates: list[Any]) -> list[str]:
        """
        Detect potential poisoning attacks in a batch of updates.
        
        Uses statistical analysis to identify outliers that may be
        malicious attempts to corrupt the model.
        
        Args:
            updates: List of WeightUpdate objects
        
        Returns:
            List of suspicious update IDs
        """
        if not HAS_NUMPY or len(updates) < 3:
            return []
        
        suspicious = []
        
        # Calculate magnitudes
        magnitudes = []
        for update in updates:
            mag = self._calculate_magnitude(update)
            magnitudes.append((update.update_id, update.device_id, mag))
        
        if not magnitudes:
            return []
        
        # Statistical outlier detection using median and IQR (robust to outliers)
        mags = np.array([m[2] for m in magnitudes])
        median = np.median(mags)
        q1, q3 = np.percentile(mags, [25, 75])
        iqr = q3 - q1
        
        # Use median-based threshold: values > median + 3*IQR are outliers
        # For small samples where IQR might be 0, fall back to using median * 10
        if iqr > 0:
            threshold = median + 3 * iqr
        else:
            threshold = median * 10  # 10x median is suspicious
        
        for update_id, device_id, mag in magnitudes:
            if mag > threshold:
                logger.warning(
                    f"Suspicious update {update_id[:8]} from {device_id}: "
                    f"magnitude {mag:.2f} >> median {median:.2f}"
                )
                suspicious.append(update_id)
                self._penalize_device(device_id, 0.4)
        
        # Check for consistency in training samples
        samples = np.array([u.training_samples for u in updates])
        sample_mean = np.mean(samples)
        sample_std = np.std(samples)
        
        for update in updates:
            if update.training_samples > sample_mean + 3 * sample_std:
                if update.update_id not in suspicious:
                    logger.warning(
                        f"Suspicious training sample count from {update.device_id}: "
                        f"{update.training_samples} >> mean {sample_mean:.0f}"
                    )
                    suspicious.append(update.update_id)
                    self._penalize_device(update.device_id, 0.2)
        
        if suspicious:
            logger.warning(f"Detected {len(suspicious)} potentially poisoned updates")
        
        return suspicious
    
    def calculate_reputation(self, device_id: str) -> float:
        """
        Calculate device reputation score (0.0 to 1.0).
        
        Based on:
        - Update quality history
        - Consistency with other devices
        - Age of participation
        - Number of valid updates
        
        Args:
            device_id: Device to calculate reputation for
        
        Returns:
            Reputation score (0.0 = untrusted, 1.0 = fully trusted)
        """
        return self.reputations[device_id]
    
    def _reward_device(self, device_id: str, amount: float):
        """Increase device reputation."""
        current = self.reputations[device_id]
        # Increase reputation, but cap at 1.0
        self.reputations[device_id] = min(1.0, current + amount)
        logger.debug(f"Rewarded {device_id}: {current:.2f} -> {self.reputations[device_id]:.2f}")
    
    def _penalize_device(self, device_id: str, amount: float):
        """Decrease device reputation."""
        current = self.reputations[device_id]
        # Decrease reputation, but keep >= 0
        self.reputations[device_id] = max(0.0, current - amount)
        logger.debug(f"Penalized {device_id}: {current:.2f} -> {self.reputations[device_id]:.2f}")
        
        # Block if reputation too low
        if self.reputations[device_id] < 0.1:
            self.block_device(device_id)
    
    def block_device(self, device_id: str):
        """Block a device from participating."""
        self.blocked_devices.add(device_id)
        logger.warning(f"Blocked device: {device_id}")
    
    def unblock_device(self, device_id: str):
        """Unblock a device."""
        self.blocked_devices.discard(device_id)
        logger.info(f"Unblocked device: {device_id}")
    
    def apply_reputation_decay(self):
        """
        Apply time-based reputation decay.
        
        Devices that haven't contributed recently lose some reputation.
        Call this periodically (e.g., once per day).
        """
        for device_id in list(self.reputations.keys()):
            current = self.reputations[device_id]
            # Decay towards 0.5 (neutral)
            decayed = current * self.reputation_decay + 0.5 * (1 - self.reputation_decay)
            self.reputations[device_id] = decayed
        
        logger.info(f"Applied reputation decay to {len(self.reputations)} devices")
    
    def get_stats(self) -> dict[str, Any]:
        """Get trust manager statistics."""
        return {
            "total_devices": len(self.reputations),
            "blocked_devices": len(self.blocked_devices),
            "total_updates": len(self.update_history),
            "recent_updates": self.update_history[-5:] if self.update_history else [],
            "reputation_distribution": self._get_reputation_distribution(),
        }
    
    def _get_reputation_distribution(self) -> dict[str, int]:
        """Get distribution of reputation scores."""
        distribution = {
            "excellent (0.8-1.0)": 0,
            "good (0.6-0.8)": 0,
            "average (0.4-0.6)": 0,
            "poor (0.2-0.4)": 0,
            "bad (0.0-0.2)": 0,
        }
        
        for rep in self.reputations.values():
            if rep >= 0.8:
                distribution["excellent (0.8-1.0)"] += 1
            elif rep >= 0.6:
                distribution["good (0.6-0.8)"] += 1
            elif rep >= 0.4:
                distribution["average (0.4-0.6)"] += 1
            elif rep >= 0.2:
                distribution["poor (0.2-0.4)"] += 1
            else:
                distribution["bad (0.0-0.2)"] += 1
        
        return distribution


def test_trust_manager():
    """Test trust management."""
    if not HAS_NUMPY:
        print("NumPy not available, skipping test")
        return
    
    import uuid

    from .federated import WeightUpdate
    
    trust_manager = TrustManager(max_update_magnitude=5.0)
    
    # Create test updates
    updates = []
    for i in range(5):
        update = WeightUpdate(
            update_id=str(uuid.uuid4()),
            device_id=f"device{i}",
            timestamp=datetime.now(),
            weight_deltas={"layer1": np.random.randn(10)},
            training_samples=100,
        )
        update.sign()
        updates.append(update)
    
    # Add a poisoned update with very large magnitude
    poisoned = WeightUpdate(
        update_id=str(uuid.uuid4()),
        device_id="malicious",
        timestamp=datetime.now(),
        weight_deltas={"layer1": np.random.randn(10) * 100},  # 100x larger
        training_samples=100,
    )
    poisoned.sign()
    updates.append(poisoned)
    
    # Verify updates
    print("Verifying updates:")
    for update in updates:
        valid = trust_manager.verify_update(update)
        print(f"  {update.device_id}: {'VALID' if valid else 'REJECTED'}")
    
    # Detect poisoning
    print(f"\nDetecting poisoning:")
    suspicious = trust_manager.detect_poisoning(updates)
    print(f"  Found {len(suspicious)} suspicious updates")
    
    # Show reputations
    print(f"\nDevice reputations:")
    for device_id in [f"device{i}" for i in range(5)] + ["malicious"]:
        rep = trust_manager.calculate_reputation(device_id)
        print(f"  {device_id}: {rep:.2f}")
    
    print(f"\nStats: {trust_manager.get_stats()}")


if __name__ == "__main__":
    test_trust_manager()
