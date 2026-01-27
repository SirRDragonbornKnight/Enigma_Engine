"""Tests for federated learning module."""

import numpy as np
import pytest
from datetime import datetime

from forge_ai.learning import (
    FederatedLearning,
    WeightUpdate,
    FederatedMode,
    PrivacyLevel,
    DifferentialPrivacy,
    SecureAggregator,
    AggregationMethod,
    DataFilter,
    TrustManager,
)


class TestWeightUpdate:
    """Tests for WeightUpdate class."""
    
    def test_create_update(self):
        """Test creating a weight update."""
        deltas = {
            "layer1": np.array([1.0, 2.0, 3.0]),
            "layer2": np.array([[1.0, 2.0], [3.0, 4.0]]),
        }
        
        update = WeightUpdate(
            update_id="test-123",
            device_id="device-1",
            timestamp=datetime.now(),
            weight_deltas=deltas,
            training_samples=100,
        )
        
        assert update.update_id == "test-123"
        assert update.device_id == "device-1"
        assert update.training_samples == 100
        assert "layer1" in update.weight_deltas
        assert "layer2" in update.weight_deltas
    
    def test_sign_and_verify(self):
        """Test signing and verifying updates."""
        update = WeightUpdate(
            update_id="test-123",
            device_id="device-1",
            timestamp=datetime.now(),
            weight_deltas={"layer1": np.array([1.0, 2.0])},
            training_samples=50,
        )
        
        # Sign
        signature = update.sign()
        assert signature is not None
        assert update.signature == signature
        
        # Verify
        assert update.verify_signature()
    
    def test_get_size(self):
        """Test getting update size."""
        update = WeightUpdate(
            update_id="test-123",
            device_id="device-1",
            timestamp=datetime.now(),
            weight_deltas={
                "layer1": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            },
            training_samples=50,
        )
        
        size = update.get_size()
        assert size == 3 * 4  # 3 floats * 4 bytes each


class TestFederatedLearning:
    """Tests for FederatedLearning class."""
    
    def test_init(self):
        """Test initialization."""
        fl = FederatedLearning(
            mode=FederatedMode.OPT_IN,
            privacy_level=PrivacyLevel.HIGH
        )
        
        assert fl.mode == FederatedMode.OPT_IN
        assert fl.privacy_level == PrivacyLevel.HIGH
        assert fl.current_round == 0
    
    def test_local_training_round(self):
        """Test local training round."""
        fl = FederatedLearning()
        
        # Base weights
        base_weights = {
            "layer1": np.array([1.0, 2.0, 3.0]),
            "layer2": np.array([[1.0, 2.0], [3.0, 4.0]]),
        }
        
        fl.start_local_round(base_weights)
        
        # Updated weights (after training)
        updated_weights = {
            "layer1": np.array([1.1, 2.1, 3.1]),
            "layer2": np.array([[1.1, 2.1], [3.1, 4.1]]),
        }
        
        # Create update
        update = fl.train_local_round(
            model_weights=updated_weights,
            training_samples=100,
            loss=0.5,
            accuracy=0.9
        )
        
        assert update is not None
        assert update.training_samples == 100
        assert update.metadata["loss"] == 0.5
        assert update.metadata["accuracy"] == 0.9
        
        # Check deltas are computed correctly
        assert "layer1" in update.weight_deltas
        expected_delta = np.array([0.1, 0.1, 0.1])
        np.testing.assert_array_almost_equal(
            update.weight_deltas["layer1"], 
            expected_delta
        )


class TestDifferentialPrivacy:
    """Tests for DifferentialPrivacy class."""
    
    def test_add_noise(self):
        """Test adding noise to weights."""
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        
        weights = {
            "layer1": np.array([1.0, 2.0, 3.0]),
        }
        
        noisy_weights = dp.add_noise(weights)
        
        # Weights should be different (noise added)
        assert not np.array_equal(noisy_weights["layer1"], weights["layer1"])
        
        # But should be similar
        diff = np.abs(noisy_weights["layer1"] - weights["layer1"])
        assert np.all(diff < 10.0)  # Reasonable noise level
    
    def test_privacy_loss(self):
        """Test computing privacy loss."""
        dp = DifferentialPrivacy(epsilon=0.5)
        
        loss = dp.compute_privacy_loss(num_rounds=10)
        assert loss == 10 * 0.5  # Simple composition


class TestSecureAggregator:
    """Tests for SecureAggregator class."""
    
    def test_simple_average(self):
        """Test simple averaging."""
        agg = SecureAggregator(method=AggregationMethod.SIMPLE)
        
        # Create test updates
        updates = [
            WeightUpdate(
                update_id=f"update-{i}",
                device_id=f"device-{i}",
                timestamp=datetime.now(),
                weight_deltas={"layer1": np.array([float(i), float(i)])},
                training_samples=10,
            )
            for i in range(1, 4)  # 1, 2, 3
        ]
        
        result = agg.aggregate_updates(updates)
        
        # Average of [1, 1], [2, 2], [3, 3] should be [2, 2]
        expected = np.array([2.0, 2.0])
        np.testing.assert_array_almost_equal(
            result.weight_deltas["layer1"],
            expected
        )
    
    def test_weighted_average(self):
        """Test weighted averaging."""
        agg = SecureAggregator(method=AggregationMethod.WEIGHTED)
        
        # Create test updates with different sample counts
        updates = [
            WeightUpdate(
                update_id="update-1",
                device_id="device-1",
                timestamp=datetime.now(),
                weight_deltas={"layer1": np.array([1.0, 1.0])},
                training_samples=10,  # 10 samples
            ),
            WeightUpdate(
                update_id="update-2",
                device_id="device-2",
                timestamp=datetime.now(),
                weight_deltas={"layer1": np.array([3.0, 3.0])},
                training_samples=30,  # 30 samples (more weight)
            ),
        ]
        
        result = agg.aggregate_updates(updates)
        
        # Weighted average: (1*10 + 3*30) / 40 = 100/40 = 2.5
        expected = np.array([2.5, 2.5])
        np.testing.assert_array_almost_equal(
            result.weight_deltas["layer1"],
            expected
        )


class TestDataFilter:
    """Tests for DataFilter class."""
    
    def test_filter_pii(self):
        """Test filtering PII."""
        filter = DataFilter()
        
        data = [
            {
                "input": "Contact me at john@example.com",
                "output": "I will call you at 555-123-4567"
            },
        ]
        
        filtered = filter.filter_training_data(data)
        
        assert len(filtered) == 1
        assert "[EMAIL]" in filtered[0]["input"]
        assert "[PHONE]" in filtered[0]["output"]
    
    def test_filter_length(self):
        """Test filtering by length."""
        filter = DataFilter(min_length=10, max_length=50)
        
        data = [
            {"input": "short", "output": "test"},  # Too short
            {"input": "This is a good length message", "output": "response"},
            {"input": "a" * 100, "output": "response"},  # Too long
        ]
        
        filtered = filter.filter_training_data(data)
        
        assert len(filtered) == 1
        assert "good length" in filtered[0]["input"]


class TestTrustManager:
    """Tests for TrustManager class."""
    
    def test_evaluate_trusted_update(self):
        """Test evaluating a trusted update."""
        tm = TrustManager()
        
        update = WeightUpdate(
            update_id="test-123",
            device_id="device-1",
            timestamp=datetime.now(),
            weight_deltas={"layer1": np.array([0.1, 0.2])},
            training_samples=100,
        )
        
        # Should be trusted initially
        assert tm.evaluate_update(update)
    
    def test_detect_byzantine(self):
        """Test detecting Byzantine updates."""
        tm = TrustManager()
        
        # Normal update
        normal = WeightUpdate(
            update_id="normal",
            device_id="device-1",
            timestamp=datetime.now(),
            weight_deltas={"layer1": np.array([0.1, 0.2])},
            training_samples=100,
        )
        
        # Byzantine update (huge magnitude)
        byzantine = WeightUpdate(
            update_id="byzantine",
            device_id="device-2",
            timestamp=datetime.now(),
            weight_deltas={"layer1": np.array([1e12, 1e12])},
            training_samples=100,
        )
        
        assert tm.evaluate_update(normal)
        assert not tm.evaluate_update(byzantine)
    
    def test_ban_device(self):
        """Test banning a device."""
        tm = TrustManager()
        
        device_id = "bad-device"
        tm.ban_device(device_id)
        
        trust = tm.get_device_trust(device_id)
        assert trust is not None
        assert trust.trust_score == 0.0
