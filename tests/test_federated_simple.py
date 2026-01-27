#!/usr/bin/env python3
"""
Simple test runner for federated learning (no pytest required).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from forge_ai.learning import (
            FederatedLearning,
            WeightUpdate,
            FederatedMode,
            PrivacyLevel,
            DifferentialPrivacy,
            SecureAggregator,
            AggregationMethod,
            FederatedCoordinator,
            CoordinatorMode,
            FederatedDataFilter,
            TrainingExample,
            TrustManager,
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_weight_update():
    """Test WeightUpdate class."""
    print("\nTesting WeightUpdate...")
    
    from forge_ai.learning import WeightUpdate
    from datetime import datetime
    
    try:
        update = WeightUpdate(
            update_id="test123",
            device_id="device1",
            timestamp=datetime.now(),
            weight_deltas={"layer1": [1.0, 2.0, 3.0]},
            training_samples=100,
        )
        
        # Test signature
        signature = update.sign()
        assert signature is not None, "Signature should not be None"
        assert update.verify_signature(), "Signature verification failed"
        
        # Test serialization
        data = update.to_dict()
        assert isinstance(data, dict), "to_dict should return dict"
        
        restored = WeightUpdate.from_dict(data)
        assert restored.update_id == update.update_id, "Deserialization failed"
        
        print("✓ WeightUpdate tests passed")
        return True
    except Exception as e:
        print(f"✗ WeightUpdate test failed: {e}")
        return False


def test_federated_learning():
    """Test FederatedLearning class."""
    print("\nTesting FederatedLearning...")
    
    from forge_ai.learning import FederatedLearning, FederatedMode, PrivacyLevel
    
    try:
        fl = FederatedLearning(
            model_name="test_model",
            mode=FederatedMode.OPT_IN,
            privacy_level=PrivacyLevel.HIGH,
        )
        
        assert fl.model_name == "test_model", "Model name mismatch"
        assert fl.mode == FederatedMode.OPT_IN, "Mode mismatch"
        assert fl.device_id is not None, "Device ID not generated"
        
        # Test stats
        stats = fl.get_stats()
        assert isinstance(stats, dict), "Stats should be dict"
        assert "device_id" in stats, "Stats missing device_id"
        
        print("✓ FederatedLearning tests passed")
        return True
    except Exception as e:
        print(f"✗ FederatedLearning test failed: {e}")
        return False


def test_differential_privacy():
    """Test DifferentialPrivacy class."""
    print("\nTesting DifferentialPrivacy...")
    
    from forge_ai.learning import DifferentialPrivacy
    
    try:
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        
        assert dp.epsilon == 1.0, "Epsilon mismatch"
        assert dp.delta == 1e-5, "Delta mismatch"
        
        desc = dp.get_privacy_level_description()
        assert isinstance(desc, str), "Description should be string"
        
        # Test with numpy if available
        try:
            import numpy as np
            
            weights = {"layer1": np.array([1.0, 2.0, 3.0])}
            noisy = dp.add_noise(weights)
            
            assert "layer1" in noisy, "Noisy weights missing layer"
            print("✓ DifferentialPrivacy tests passed (with numpy)")
        except ImportError:
            print("✓ DifferentialPrivacy tests passed (numpy not available)")
        
        return True
    except Exception as e:
        print(f"✗ DifferentialPrivacy test failed: {e}")
        return False


def test_secure_aggregator():
    """Test SecureAggregator class."""
    print("\nTesting SecureAggregator...")
    
    from forge_ai.learning import SecureAggregator, AggregationMethod
    
    try:
        aggregator = SecureAggregator()
        assert aggregator is not None, "Aggregator creation failed"
        
        stats = aggregator.get_stats()
        assert isinstance(stats, dict), "Stats should be dict"
        
        print("✓ SecureAggregator tests passed")
        return True
    except Exception as e:
        print(f"✗ SecureAggregator test failed: {e}")
        return False


def test_coordinator():
    """Test FederatedCoordinator class."""
    print("\nTesting FederatedCoordinator...")
    
    from forge_ai.learning import FederatedCoordinator, CoordinatorMode
    
    try:
        coordinator = FederatedCoordinator(
            mode=CoordinatorMode.CENTRALIZED,
            min_participants=2,
        )
        
        # Test device registration
        assert coordinator.register_device("device1"), "Device registration failed"
        assert "device1" in coordinator.authorized_devices, "Device not in authorized list"
        
        # Test round management
        round_id = coordinator.start_round()
        assert round_id is not None, "Round ID should not be None"
        assert coordinator.current_round == 1, "Round number incorrect"
        
        status = coordinator.get_round_status()
        assert status["current_round"] == 1, "Status round mismatch"
        
        print("✓ FederatedCoordinator tests passed")
        return True
    except Exception as e:
        print(f"✗ FederatedCoordinator test failed: {e}")
        return False


def test_data_filter():
    """Test FederatedDataFilter class."""
    print("\nTesting FederatedDataFilter...")
    
    from forge_ai.learning import FederatedDataFilter, TrainingExample
    
    try:
        filter = FederatedDataFilter()
        assert filter is not None, "Filter creation failed"
        
        # Test normal example
        normal = TrainingExample(text="This is a normal conversation.")
        assert filter.should_include(normal), "Normal example should be included"
        
        # Test private example
        private = TrainingExample(text="Private text", is_private=True)
        assert not filter.should_include(private), "Private example should be excluded"
        
        # Test sanitization
        example = TrainingExample(text="Contact me at test@example.com")
        sanitized = filter.sanitize(example)
        assert "[EMAIL]" in sanitized.text, "Email should be sanitized"
        
        print("✓ FederatedDataFilter tests passed")
        return True
    except Exception as e:
        print(f"✗ FederatedDataFilter test failed: {e}")
        return False


def test_trust_manager():
    """Test TrustManager class."""
    print("\nTesting TrustManager...")
    
    from forge_ai.learning import TrustManager
    
    try:
        trust_mgr = TrustManager()
        assert trust_mgr is not None, "TrustManager creation failed"
        
        # Test reputation
        rep = trust_mgr.calculate_reputation("device1")
        assert 0.0 <= rep <= 1.0, "Reputation out of range"
        
        stats = trust_mgr.get_stats()
        assert isinstance(stats, dict), "Stats should be dict"
        
        print("✓ TrustManager tests passed")
        return True
    except Exception as e:
        print(f"✗ TrustManager test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Federated Learning Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_weight_update,
        test_federated_learning,
        test_differential_privacy,
        test_secure_aggregator,
        test_coordinator,
        test_data_filter,
        test_trust_manager,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
