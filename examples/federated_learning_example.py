"""
Federated Learning Example for ForgeAI

This example demonstrates how to use the federated learning system
to share model improvements without sharing private data.

What is Federated Learning?
----------------------------
Federated learning allows multiple devices to collaboratively train a model
while keeping all training data on the device. Only model improvements
(weight updates) are shared, never the raw data.

Privacy Protection:
- Differential privacy adds noise to weight updates
- PII is automatically detected and sanitized
- Device IDs are anonymized
- Secure aggregation prevents inspection of individual updates
- Trust management detects and blocks malicious updates

Usage Example
-------------
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def example_basic_usage():
    """Basic federated learning setup."""
    from forge_ai.learning import (
        FederatedLearning,
        FederatedMode,
        PrivacyLevel,
    )
    
    print("=== Basic Federated Learning Setup ===\n")
    
    # Create federated learning instance
    fl = FederatedLearning(
        model_name="my_model",
        mode=FederatedMode.OPT_IN,  # Must explicitly enable
        privacy_level=PrivacyLevel.HIGH,  # Strong privacy protection
    )
    
    print(f"Device ID: {fl.device_id}")
    print(f"Mode: {fl.mode.value}")
    print(f"Privacy Level: {fl.privacy_level.value}\n")
    
    # Set initial model weights (before training)
    initial_weights = {
        "layer1": [1.0, 2.0, 3.0],
        "layer2": [4.0, 5.0, 6.0],
    }
    fl.set_initial_weights(initial_weights)
    
    # After local training, create a weight update
    final_weights = {
        "layer1": [1.1, 2.2, 3.3],  # Trained weights
        "layer2": [4.1, 5.1, 6.1],
    }
    
    update = fl.train_local_round(
        final_weights=final_weights,
        training_samples=50,
        metadata={"loss": 0.45, "accuracy": 0.85}
    )
    
    print(f"Created update: {update.update_id[:8]}...")
    print(f"Training samples: {update.training_samples}")
    print(f"Layers updated: {list(update.weight_deltas.keys())}\n")
    
    # Share the update (applies privacy protection)
    success = fl.share_update(update)
    print(f"Update shared: {success}\n")
    
    # Get statistics
    stats = fl.get_stats()
    print("Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def example_data_filtering():
    """Data filtering and sanitization."""
    from forge_ai.learning import (
        FederatedDataFilter,
        TrainingExample,
    )
    
    print("\n\n=== Data Filtering & Privacy ===\n")
    
    # Create data filter
    filter = FederatedDataFilter()
    
    # Add custom excluded keywords
    filter.add_excluded_keyword("company_secret")
    
    # Example 1: Normal conversation (included)
    example1 = TrainingExample(
        text="Let's discuss how to implement recursion in Python.",
        category="coding",
    )
    print(f"Example 1: '{example1.text[:50]}...'")
    print(f"  Should include? {filter.should_include(example1)}\n")
    
    # Example 2: Contains password (excluded)
    example2 = TrainingExample(
        text="My password is secret123, please don't share it.",
    )
    print(f"Example 2: '{example2.text[:50]}...'")
    print(f"  Should include? {filter.should_include(example2)}\n")
    
    # Example 3: Contains PII (sanitize it)
    example3 = TrainingExample(
        text="Contact me at john.doe@example.com or 555-123-4567.",
    )
    print(f"Example 3 (original): '{example3.text}'")
    sanitized = filter.sanitize(example3)
    print(f"Example 3 (sanitized): '{sanitized.text}'\n")
    
    # Example 4: Private conversation (excluded)
    example4 = TrainingExample(
        text="This is a private conversation.",
        is_private=True,
    )
    print(f"Example 4: '{example4.text}'")
    print(f"  Should include? {filter.should_include(example4)}\n")


def example_coordinator():
    """Federated learning coordination."""
    from forge_ai.learning import (
        FederatedCoordinator,
        CoordinatorMode,
    )
    
    print("\n\n=== Federated Learning Coordinator ===\n")
    
    # Create coordinator
    coordinator = FederatedCoordinator(
        mode=CoordinatorMode.CENTRALIZED,
        min_participants=2,
        round_timeout=300,  # 5 minutes
    )
    
    # Register devices
    coordinator.register_device("device1")
    coordinator.register_device("device2")
    coordinator.register_device("device3")
    
    print(f"Registered {len(coordinator.authorized_devices)} devices")
    
    # Start a training round
    round_id = coordinator.start_round()
    print(f"\nStarted round: {round_id}")
    
    # Get round status
    status = coordinator.get_round_status()
    print(f"Round state: {status['state']}")
    print(f"Pending updates: {status['pending_updates']}/{status['min_participants']}")
    print(f"Authorized devices: {status['authorized_devices']}\n")


def example_differential_privacy():
    """Differential privacy demonstration."""
    print("\n\n=== Differential Privacy ===\n")
    
    try:
        import numpy as np
        from forge_ai.learning import DifferentialPrivacy
        
        # Create differential privacy mechanism
        dp = DifferentialPrivacy(
            epsilon=1.0,  # Privacy budget (lower = more private)
            delta=1e-5,   # Privacy delta
        )
        
        print(f"Privacy Level: {dp.get_privacy_level_description()}\n")
        
        # Original weights
        weights = {
            "layer1": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "layer2": np.array([0.5, 0.5, 0.5, 0.5]),
        }
        
        print("Original weights:")
        for name, w in weights.items():
            print(f"  {name}: {w}")
        
        # Add differential privacy noise
        noisy_weights = dp.add_noise(weights)
        
        print("\nNoisy weights (with privacy protection):")
        for name, w in noisy_weights.items():
            print(f"  {name}: {w}")
        
        print("\nDifference (noise added):")
        for name in weights:
            diff = noisy_weights[name] - weights[name]
            print(f"  {name}: {diff}")
        
        print(f"\nTotal privacy budget after 10 rounds: ε = {dp.compose_privacy_budget(10)}")
        
    except ImportError:
        print("NumPy not available - differential privacy requires NumPy")


def example_trust_management():
    """Trust management and poisoning detection."""
    print("\n\n=== Trust Management ===\n")
    
    try:
        import numpy as np
        from forge_ai.learning import TrustManager, WeightUpdate
        from datetime import datetime
        
        trust_mgr = TrustManager(
            max_update_magnitude=10.0,
            min_reputation=0.5,
        )
        
        # Create a valid update
        valid_update = WeightUpdate(
            update_id="valid123",
            device_id="good_device",
            timestamp=datetime.now(),
            weight_deltas={"layer1": np.array([0.1, 0.2, 0.3])},
            training_samples=100,
        )
        valid_update.sign()
        
        print("Verifying valid update...")
        is_valid = trust_mgr.verify_update(valid_update)
        print(f"  Valid: {is_valid}")
        print(f"  Device reputation: {trust_mgr.calculate_reputation('good_device'):.2f}\n")
        
        # Create a suspicious update (very large magnitude)
        suspicious_update = WeightUpdate(
            update_id="suspicious456",
            device_id="suspicious_device",
            timestamp=datetime.now(),
            weight_deltas={"layer1": np.array([100.0, 200.0, 300.0])},
            training_samples=100,
        )
        suspicious_update.sign()
        
        print("Verifying suspicious update (large magnitude)...")
        is_valid = trust_mgr.verify_update(suspicious_update)
        print(f"  Valid: {is_valid}")
        print(f"  Device reputation: {trust_mgr.calculate_reputation('suspicious_device'):.2f}\n")
        
        # Get trust statistics
        stats = trust_mgr.get_stats()
        print("Trust Manager Statistics:")
        print(f"  Total devices tracked: {stats['total_devices']}")
        print(f"  Blocked devices: {stats['blocked_devices']}")
        print(f"  Total updates processed: {stats['total_updates']}")
        
    except ImportError:
        print("NumPy not available - trust management requires NumPy for update verification")


def example_configuration():
    """Configuration example."""
    print("\n\n=== Configuration ===\n")
    
    print("To enable federated learning, add to forge_config.json:\n")
    print("""{
  "federated_learning": {
    "enabled": true,
    "mode": "peer_to_peer",
    "privacy_level": "high",
    "differential_privacy": {
      "enabled": true,
      "epsilon": 1.0,
      "delta": 1e-5
    },
    "data_filtering": {
      "exclude_private_chats": true,
      "sanitize_pii": true,
      "excluded_keywords": ["password", "credit card", "ssn"]
    },
    "participation": {
      "auto_join_rounds": true,
      "max_rounds_per_day": 3,
      "min_training_samples": 10
    },
    "trust": {
      "verify_signatures": true,
      "detect_poisoning": true
    }
  }
}""")
    
    print("\n\nOr configure via GUI:")
    print("  Settings Tab -> Federated Learning section")
    print("  - Enable/disable participation")
    print("  - Set privacy level")
    print("  - Configure data filtering")
    print("  - View contribution statistics")


def main():
    """Run all examples."""
    print("=" * 70)
    print(" ForgeAI Federated Learning - Usage Examples")
    print("=" * 70)
    
    try:
        example_basic_usage()
        example_data_filtering()
        example_coordinator()
        example_differential_privacy()
        example_trust_management()
        example_configuration()
        
        print("\n" + "=" * 70)
        print(" Examples completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
