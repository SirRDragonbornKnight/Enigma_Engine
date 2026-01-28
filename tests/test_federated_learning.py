#!/usr/bin/env python3
"""
Tests for federated learning system.

Run with: python -m pytest tests/test_federated_learning.py -v
"""
import pytest
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestWeightUpdate:
    """Tests for WeightUpdate class."""
    
    def test_weight_update_creation(self):
        """Test creating a weight update."""
        from forge_ai.learning import WeightUpdate
        
        update = WeightUpdate(
            update_id="test123",
            device_id="device1",
            timestamp=datetime.now(),
            weight_deltas={"layer1": [1.0, 2.0, 3.0]},
            training_samples=100,
        )
        
        assert update.update_id == "test123"
        assert update.device_id == "device1"
        assert update.training_samples == 100
        assert "layer1" in update.weight_deltas
    
    def test_weight_update_signature(self):
        """Test signature creation and verification."""
        from forge_ai.learning import WeightUpdate
        
        update = WeightUpdate(
            update_id="test123",
            device_id="device1",
            timestamp=datetime.now(),
            weight_deltas={"layer1": [1.0, 2.0]},
            training_samples=10,
        )
        
        # Sign the update
        signature = update.sign()
        assert signature is not None
        assert update.signature == signature
        
        # Verify signature
        assert update.verify_signature()
    
    def test_weight_update_serialization(self):
        """Test converting weight update to/from dict."""
        from forge_ai.learning import WeightUpdate
        
        original = WeightUpdate(
            update_id="test123",
            device_id="device1",
            timestamp=datetime.now(),
            weight_deltas={"layer1": [1.0, 2.0]},
            training_samples=10,
            metadata={"loss": 0.5},
        )
        
        # Convert to dict
        data = original.to_dict()
        assert isinstance(data, dict)
        assert data["update_id"] == "test123"
        
        # Convert back
        restored = WeightUpdate.from_dict(data)
        assert restored.update_id == original.update_id
        assert restored.device_id == original.device_id
        assert restored.training_samples == original.training_samples


class TestFederatedLearning:
    """Tests for FederatedLearning class."""
    
    def test_federated_learning_init(self):
        """Test initializing federated learning."""
        from forge_ai.learning import FederatedLearning, FederatedMode, PrivacyLevel
        
        fl = FederatedLearning(
            model_name="test_model",
            mode=FederatedMode.OPT_IN,
            privacy_level=PrivacyLevel.HIGH,
        )
        
        assert fl.model_name == "test_model"
        assert fl.mode == FederatedMode.OPT_IN
        assert fl.privacy_level == PrivacyLevel.HIGH
        assert fl.device_id is not None
    
    def test_device_id_generation(self):
        """Test device ID generation."""
        from forge_ai.learning import FederatedLearning, FederatedMode, PrivacyLevel
        
        fl = FederatedLearning(
            model_name="test_model",
            mode=FederatedMode.OPT_IN,
            privacy_level=PrivacyLevel.HIGH,
        )
        
        # With high privacy, should be anonymized
        assert fl.device_id.startswith("device_")
    
    def test_train_local_round(self):
        """Test creating weight update from training."""
        from forge_ai.learning import FederatedLearning, FederatedMode, PrivacyLevel
        
        fl = FederatedLearning(
            model_name="test_model",
            mode=FederatedMode.OPT_IN,
            privacy_level=PrivacyLevel.LOW,
        )
        
        # Set initial weights
        initial = {"layer1": [1.0, 2.0, 3.0]}
        fl.set_initial_weights(initial)
        
        # Create update after training
        final = {"layer1": [1.1, 2.2, 3.3]}
        update = fl.train_local_round(final, training_samples=50)
        
        assert update is not None
        assert update.training_samples == 50
        assert "layer1" in update.weight_deltas
        # Delta should be approximately [0.1, 0.2, 0.3]
    
    def test_get_stats(self):
        """Test getting federated learning statistics."""
        from forge_ai.learning import FederatedLearning, FederatedMode, PrivacyLevel
        
        fl = FederatedLearning(
            model_name="test_model",
            mode=FederatedMode.OPT_IN,
            privacy_level=PrivacyLevel.HIGH,
        )
        
        stats = fl.get_stats()
        assert isinstance(stats, dict)
        assert "device_id" in stats
        assert "mode" in stats
        assert "privacy_level" in stats
        assert "updates_sent" in stats
        assert "updates_received" in stats


class TestDifferentialPrivacy:
    """Tests for DifferentialPrivacy class."""
    
    def test_differential_privacy_init(self):
        """Test initializing differential privacy."""
        from forge_ai.learning import DifferentialPrivacy
        
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        assert dp.epsilon == 1.0
        assert dp.delta == 1e-5
    
    def test_add_noise(self):
        """Test adding noise to weights."""
        pytest.importorskip("numpy")
        import numpy as np
        from forge_ai.learning import DifferentialPrivacy
        
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        
        weights = {
            "layer1": np.array([1.0, 2.0, 3.0]),
            "layer2": np.array([4.0, 5.0, 6.0]),
        }
        
        noisy_weights = dp.add_noise(weights)
        
        assert "layer1" in noisy_weights
        assert "layer2" in noisy_weights
        
        # Noise should have been added
        assert not np.allclose(weights["layer1"], noisy_weights["layer1"])
        assert not np.allclose(weights["layer2"], noisy_weights["layer2"])
    
    def test_privacy_level_description(self):
        """Test getting privacy level description."""
        from forge_ai.learning import DifferentialPrivacy
        
        dp_strong = DifferentialPrivacy(epsilon=0.1, delta=1e-5)
        dp_weak = DifferentialPrivacy(epsilon=15.0, delta=1e-5)
        
        assert "Strong" in dp_strong.get_privacy_level_description()
        assert "Minimal" in dp_weak.get_privacy_level_description()


class TestSecureAggregator:
    """Tests for SecureAggregator class."""
    
    def test_aggregator_init(self):
        """Test initializing aggregator."""
        from forge_ai.learning import SecureAggregator
        
        aggregator = SecureAggregator()
        assert aggregator is not None
    
    def test_simple_average(self):
        """Test simple averaging aggregation."""
        pytest.importorskip("numpy")
        import numpy as np
        from forge_ai.learning import SecureAggregator, AggregationMethod, WeightUpdate
        
        aggregator = SecureAggregator()
        
        # Create test updates
        updates = [
            WeightUpdate(
                update_id="u1",
                device_id="d1",
                timestamp=datetime.now(),
                weight_deltas={"layer1": np.array([1.0, 2.0])},
                training_samples=10,
            ),
            WeightUpdate(
                update_id="u2",
                device_id="d2",
                timestamp=datetime.now(),
                weight_deltas={"layer1": np.array([3.0, 4.0])},
                training_samples=10,
            ),
        ]
        
        result = aggregator.aggregate_updates(updates, AggregationMethod.SIMPLE)
        
        assert result is not None
        assert "layer1" in result.weight_deltas
        # Average should be [2.0, 3.0]
        assert np.allclose(result.weight_deltas["layer1"], [2.0, 3.0])
    
    def test_weighted_average(self):
        """Test weighted averaging aggregation."""
        pytest.importorskip("numpy")
        import numpy as np
        from forge_ai.learning import SecureAggregator, AggregationMethod, WeightUpdate
        
        aggregator = SecureAggregator()
        
        # Create test updates with different sample counts
        updates = [
            WeightUpdate(
                update_id="u1",
                device_id="d1",
                timestamp=datetime.now(),
                weight_deltas={"layer1": np.array([1.0, 2.0])},
                training_samples=10,  # 10% weight
            ),
            WeightUpdate(
                update_id="u2",
                device_id="d2",
                timestamp=datetime.now(),
                weight_deltas={"layer1": np.array([3.0, 4.0])},
                training_samples=90,  # 90% weight
            ),
        ]
        
        result = aggregator.aggregate_updates(updates, AggregationMethod.WEIGHTED)
        
        assert result is not None
        assert result.training_samples == 100  # Total samples
        # Weighted average should be closer to [3.0, 4.0]
        expected = np.array([1.0*0.1 + 3.0*0.9, 2.0*0.1 + 4.0*0.9])
        assert np.allclose(result.weight_deltas["layer1"], expected)


class TestFederatedCoordinator:
    """Tests for FederatedCoordinator class."""
    
    def test_coordinator_init(self):
        """Test initializing coordinator."""
        from forge_ai.learning import FederatedCoordinator, CoordinatorMode
        
        coordinator = FederatedCoordinator(
            mode=CoordinatorMode.CENTRALIZED,
            min_participants=2,
        )
        
        assert coordinator.mode == CoordinatorMode.CENTRALIZED
        assert coordinator.min_participants == 2
        assert coordinator.current_round == 0
    
    def test_device_registration(self):
        """Test registering devices."""
        from forge_ai.learning import FederatedCoordinator
        
        coordinator = FederatedCoordinator()
        
        assert coordinator.register_device("device1")
        assert "device1" in coordinator.authorized_devices
        
        assert coordinator.unregister_device("device1")
        assert "device1" not in coordinator.authorized_devices
    
    def test_start_round(self):
        """Test starting a training round."""
        from forge_ai.learning import FederatedCoordinator
        
        coordinator = FederatedCoordinator()
        
        round_id = coordinator.start_round()
        
        assert round_id is not None
        assert coordinator.current_round == 1
        
        status = coordinator.get_round_status()
        assert status["current_round"] == 1
        assert status["state"] == "training"
    
    def test_accept_update(self):
        """Test accepting weight updates."""
        pytest.importorskip("numpy")
        import numpy as np
        from forge_ai.learning import FederatedCoordinator, WeightUpdate
        
        coordinator = FederatedCoordinator(min_participants=1)
        coordinator.register_device("device1")
        coordinator.start_round()
        
        update = WeightUpdate(
            update_id="u1",
            device_id="device1",
            timestamp=datetime.now(),
            weight_deltas={"layer1": np.array([1.0, 2.0])},
            training_samples=10,
        )
        update.sign()
        
        assert coordinator.accept_update(update)
        assert len(coordinator.pending_updates) == 1


class TestFederatedDataFilter:
    """Tests for FederatedDataFilter class."""
    
    def test_filter_init(self):
        """Test initializing data filter."""
        from forge_ai.learning import FederatedDataFilter
        
        filter = FederatedDataFilter()
        assert filter is not None
        assert len(filter.excluded_keywords) > 0
    
    def test_should_include(self):
        """Test filtering decisions."""
        from forge_ai.learning import FederatedDataFilter, TrainingExample
        
        filter = FederatedDataFilter()
        
        # Normal example should be included
        normal = TrainingExample(
            text="This is a normal conversation about programming.",
            category="coding",
        )
        assert filter.should_include(normal)
        
        # Private example should be excluded
        private = TrainingExample(
            text="Private conversation",
            is_private=True,
        )
        assert not filter.should_include(private)
        
        # Example with sensitive keyword should be excluded
        sensitive = TrainingExample(
            text="My password is secret123",
        )
        assert not filter.should_include(sensitive)
    
    def test_sanitize(self):
        """Test sanitizing personal information."""
        from forge_ai.learning import FederatedDataFilter, TrainingExample
        
        filter = FederatedDataFilter()
        
        example = TrainingExample(
            text="Contact me at john@example.com or 555-123-4567",
        )
        
        sanitized = filter.sanitize(example)
        
        # Email and phone should be redacted
        assert "[EMAIL]" in sanitized.text
        assert "[PHONE]" in sanitized.text
        assert "john@example.com" not in sanitized.text
        assert "555-123-4567" not in sanitized.text


class TestTrustManager:
    """Tests for TrustManager class."""
    
    def test_trust_manager_init(self):
        """Test initializing trust manager."""
        from forge_ai.learning import TrustManager
        
        trust_mgr = TrustManager()
        assert trust_mgr is not None
        assert trust_mgr.max_update_magnitude > 0
    
    def test_verify_update(self):
        """Test verifying updates."""
        pytest.importorskip("numpy")
        import numpy as np
        from forge_ai.learning import TrustManager, WeightUpdate
        
        trust_mgr = TrustManager()
        
        # Valid update
        update = WeightUpdate(
            update_id="u1",
            device_id="device1",
            timestamp=datetime.now(),
            weight_deltas={"layer1": np.array([1.0, 2.0])},
            training_samples=10,
        )
        update.sign()
        
        assert trust_mgr.verify_update(update)
    
    def test_detect_poisoning(self):
        """Test detecting poisoning attacks."""
        pytest.importorskip("numpy")
        import numpy as np
        from forge_ai.learning import TrustManager, WeightUpdate
        
        trust_mgr = TrustManager()
        
        # Create normal updates
        normal_updates = []
        for i in range(5):
            update = WeightUpdate(
                update_id=f"u{i}",
                device_id=f"device{i}",
                timestamp=datetime.now(),
                weight_deltas={"layer1": np.random.randn(10)},
                training_samples=100,
            )
            update.sign()
            normal_updates.append(update)
        
        # Create poisoned update with very large magnitude
        poisoned = WeightUpdate(
            update_id="malicious",
            device_id="attacker",
            timestamp=datetime.now(),
            weight_deltas={"layer1": np.random.randn(10) * 100},
            training_samples=100,
        )
        poisoned.sign()
        
        all_updates = normal_updates + [poisoned]
        
        suspicious = trust_mgr.detect_poisoning(all_updates)
        
        # Poisoned update should be detected
        assert len(suspicious) > 0
        assert "malicious" in suspicious
    
    def test_reputation_system(self):
        """Test device reputation system."""
        pytest.importorskip("numpy")
        import numpy as np
        from forge_ai.learning import TrustManager, WeightUpdate
        
        trust_mgr = TrustManager()
        
        # Valid update should increase reputation
        update = WeightUpdate(
            update_id="u1",
            device_id="good_device",
            timestamp=datetime.now(),
            weight_deltas={"layer1": np.array([1.0, 2.0])},
            training_samples=10,
        )
        update.sign()
        
        initial_rep = trust_mgr.calculate_reputation("good_device")
        trust_mgr.verify_update(update)
        final_rep = trust_mgr.calculate_reputation("good_device")
        
        assert final_rep > initial_rep


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
