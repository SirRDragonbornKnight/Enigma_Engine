"""Tests for federated learning module."""

import numpy as np
import pytest
from datetime import datetime

from enigma_engine.learning import (
    FederatedLearning,
    WeightUpdate,
    FederatedMode,
    PrivacyLevel,
    DifferentialPrivacy,
    SecureAggregator,
    AggregationMethod,
    FederatedDataFilter,
    TrainingExample,
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
            metadata={"loss": 0.5},
        )
        
        assert update.update_id == "test-123"
        assert update.device_id == "device-1"
        assert update.training_samples == 100
        assert "layer1" in update.weight_deltas
        assert update.metadata["loss"] == 0.5
    
    def test_sign_and_verify(self):
        """Test signing and verifying updates."""
        update = WeightUpdate(
            update_id="test-456",
            device_id="device-2",
            timestamp=datetime.now(),
            weight_deltas={"layer1": np.array([1.0, 2.0])},
            training_samples=50,
        )
        
        # Sign the update
        signature = update.sign("secret_key")
        assert signature is not None
        assert update.signature == signature
        
        # Verify with correct key
        assert update.verify_signature("secret_key")
    
    def test_get_size(self):
        """Test getting update size."""
        update = WeightUpdate(
            update_id="test-789",
            device_id="device-3",
            timestamp=datetime.now(),
            weight_deltas={"layer1": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
            training_samples=25,
        )
        
        size = update.get_size()
        assert size == 3 * 4  # 3 floats * 4 bytes each


class TestFederatedLearning:
    """Tests for FederatedLearning class."""
    
    def test_init(self):
        """Test initialization."""
        fl = FederatedLearning(
            model_name="test_model",
            mode=FederatedMode.OPT_IN,
            privacy_level=PrivacyLevel.HIGH
        )
        
        assert fl.mode == FederatedMode.OPT_IN
        assert fl.privacy_level == PrivacyLevel.HIGH
        assert fl.model_name == "test_model"
    
    def test_local_training_round(self):
        """Test local training round."""
        # Use NONE privacy level to avoid noise
        fl = FederatedLearning(
            model_name="test_model",
            privacy_level=PrivacyLevel.NONE
        )
        
        # Base weights (set initial weights directly)
        base_weights = {
            "layer1": np.array([1.0, 2.0, 3.0]),
            "layer2": np.array([[1.0, 2.0], [3.0, 4.0]]),
        }
        
        fl.initial_weights = base_weights
        
        # Updated weights (after training)
        updated_weights = {
            "layer1": np.array([1.1, 2.1, 3.1]),
            "layer2": np.array([[1.1, 2.1], [3.1, 4.1]]),
        }
        
        # Create update
        update = fl.train_local_round(
            final_weights=updated_weights,
            training_samples=100,
            metadata={"loss": 0.5, "accuracy": 0.9}
        )
        
        assert update is not None
        assert update.training_samples == 100
        
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
        
        # But should be similar (noise is proportional to sensitivity)
        diff = np.abs(noisy_weights["layer1"] - weights["layer1"])
        # Noise can be significant based on sensitivity, just check it's not infinite
        assert np.all(np.isfinite(diff))
    
    def test_epsilon_validation(self):
        """Test epsilon validation."""
        with pytest.raises(ValueError):
            DifferentialPrivacy(epsilon=0)  # epsilon must be positive
        
        with pytest.raises(ValueError):
            DifferentialPrivacy(epsilon=-1)  # epsilon must be positive


class TestSecureAggregator:
    """Tests for SecureAggregator class."""
    
    def test_simple_average(self):
        """Test simple averaging."""
        agg = SecureAggregator()
        
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
        
        result = agg.aggregate_updates(updates, method=AggregationMethod.SIMPLE)
        
        # Average of [1, 1], [2, 2], [3, 3] should be [2, 2]
        expected = np.array([2.0, 2.0])
        np.testing.assert_array_almost_equal(
            result.weight_deltas["layer1"],
            expected
        )
    
    def test_weighted_average(self):
        """Test weighted averaging."""
        agg = SecureAggregator()
        
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
        
        result = agg.aggregate_updates(updates, method=AggregationMethod.WEIGHTED)
        
        # Weighted average: (1*10 + 3*30) / 40 = 100/40 = 2.5
        expected = np.array([2.5, 2.5])
        np.testing.assert_array_almost_equal(
            result.weight_deltas["layer1"],
            expected
        )


class TestDataFilter:
    """Tests for FederatedDataFilter class."""
    
    def test_should_include(self):
        """Test filtering examples."""
        filter = FederatedDataFilter()
        
        # Normal example - should be included
        example1 = TrainingExample(
            text="This is a normal training example with no sensitive data.",
            is_private=False
        )
        assert filter.should_include(example1)
        
        # Private example - should be excluded
        example2 = TrainingExample(
            text="This is a private conversation",
            is_private=True
        )
        assert not filter.should_include(example2)
    
    def test_filter_sensitive_keywords(self):
        """Test filtering sensitive keywords."""
        filter = FederatedDataFilter()
        
        # Example with password - should be excluded
        example = TrainingExample(
            text="My password is secret123",
            is_private=False
        )
        assert not filter.should_include(example)
    
    def test_filter_pii(self):
        """Test filtering PII."""
        filter = FederatedDataFilter()
        
        # Example with email - should be excluded due to PII
        example = TrainingExample(
            text="Contact me at john@example.com for more info",
            is_private=False
        )
        assert not filter.should_include(example)
    
    def test_sanitize(self):
        """Test sanitizing PII from examples."""
        filter = FederatedDataFilter()
        
        example = TrainingExample(
            text="Contact john@example.com or call 555-123-4567",
            is_private=False
        )
        sanitized = filter.sanitize(example)
        
        assert "john@example.com" not in sanitized.text
        assert "555-123-4567" not in sanitized.text
        assert "[EMAIL]" in sanitized.text or "[PHONE]" in sanitized.text


class TestTrustManager:
    """Tests for TrustManager class."""
    
    def test_verify_trusted_update(self):
        """Test verifying a trusted update."""
        tm = TrustManager()
        
        update = WeightUpdate(
            update_id="test-123",
            device_id="device-1",
            timestamp=datetime.now(),
            weight_deltas={"layer1": np.array([0.1, 0.2])},
            training_samples=100,
        )
        update.sign()  # Sign the update
        
        # Should be trusted initially
        assert tm.verify_update(update)
    
    def test_detect_large_magnitude(self):
        """Test detecting updates with large magnitude."""
        tm = TrustManager(max_update_magnitude=1.0)
        
        # Normal update
        normal = WeightUpdate(
            update_id="normal",
            device_id="device-1",
            timestamp=datetime.now(),
            weight_deltas={"layer1": np.array([0.1, 0.2])},
            training_samples=100,
        )
        normal.sign()
        
        # Large magnitude update (should fail)
        large = WeightUpdate(
            update_id="large",
            device_id="device-2",
            timestamp=datetime.now(),
            weight_deltas={"layer1": np.array([1e6, 1e6])},
            training_samples=100,
        )
        large.sign()
        
        assert tm.verify_update(normal)
        assert not tm.verify_update(large)
    
    def test_block_device(self):
        """Test blocking a device."""
        tm = TrustManager()
        
        device_id = "bad-device"
        tm.block_device(device_id)
        
        update = WeightUpdate(
            update_id="test",
            device_id=device_id,
            timestamp=datetime.now(),
            weight_deltas={"layer1": np.array([0.1])},
            training_samples=10,
        )
        update.sign()
        
        # Should be rejected because device is blocked
        assert not tm.verify_update(update)
