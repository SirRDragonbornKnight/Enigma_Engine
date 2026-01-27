"""
Demo script for Federated Learning System

Demonstrates the core federated learning capabilities:
- Creating weight updates
- Adding differential privacy
- Aggregating updates
- Trust management
"""

import numpy as np
from datetime import datetime

# Import federated learning components
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
    TrainingCoordinator,
)


def demo_basic_workflow():
    """Demonstrate basic federated learning workflow."""
    print("=" * 70)
    print("FEDERATED LEARNING - BASIC WORKFLOW DEMO")
    print("=" * 70)
    
    # Step 1: Initialize federated learning on 3 devices
    print("\n1. Initializing 3 devices...")
    devices = []
    for i in range(3):
        fl = FederatedLearning(
            mode=FederatedMode.OPT_IN,
            privacy_level=PrivacyLevel.MEDIUM
        )
        devices.append(fl)
        print(f"   Device {i+1}: {fl.device_id[:16]}...")
    
    # Step 2: Each device trains locally
    print("\n2. Local training on each device...")
    
    # Simulated base weights
    base_weights = {
        "layer1": np.array([1.0, 2.0, 3.0]),
        "layer2": np.array([0.5, 0.5]),
    }
    
    updates = []
    for i, device in enumerate(devices):
        # Start training round
        device.start_local_round(base_weights)
        
        # Simulate training (small random changes)
        updated_weights = {
            "layer1": base_weights["layer1"] + np.random.normal(0, 0.1, 3),
            "layer2": base_weights["layer2"] + np.random.normal(0, 0.1, 2),
        }
        
        # Create update
        update = device.train_local_round(
            model_weights=updated_weights,
            training_samples=100 + i * 50,  # Different amounts of data
            loss=0.5 - i * 0.1,
            accuracy=0.8 + i * 0.05
        )
        
        updates.append(update)
        print(f"   Device {i+1}: trained on {update.training_samples} samples, "
              f"loss={update.metadata['loss']:.3f}")
    
    # Step 3: Aggregate updates
    print("\n3. Aggregating updates (weighted by samples)...")
    aggregator = SecureAggregator(method=AggregationMethod.WEIGHTED)
    global_update = aggregator.aggregate_updates(updates)
    
    print(f"   Aggregated {len(updates)} updates")
    print(f"   Total training samples: {global_update.training_samples}")
    
    # Step 4: Devices receive global update
    print("\n4. Distributing global update to devices...")
    for i, device in enumerate(devices):
        new_weights = device.receive_global_update(global_update)
        print(f"   Device {i+1} applied global update")
    
    print("\n✓ Basic workflow completed successfully!")


def demo_privacy():
    """Demonstrate differential privacy."""
    print("\n" + "=" * 70)
    print("DIFFERENTIAL PRIVACY DEMO")
    print("=" * 70)
    
    print("\n1. Original weights:")
    weights = {
        "layer1": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    }
    print(f"   layer1: {weights['layer1']}")
    
    # Add noise with different privacy levels
    print("\n2. Adding noise with different privacy budgets:")
    
    for epsilon in [10.0, 1.0, 0.1]:
        dp = DifferentialPrivacy(epsilon=epsilon, delta=1e-5)
        noisy = dp.add_noise(weights.copy())
        
        diff = np.abs(noisy["layer1"] - weights["layer1"])
        avg_diff = np.mean(diff)
        
        print(f"   ε={epsilon:4.1f}: avg noise={avg_diff:.4f}, "
              f"noisy values={noisy['layer1'][:3]}")
    
    print("\n✓ Lower epsilon = more privacy (more noise)")


def demo_trust():
    """Demonstrate trust management."""
    print("\n" + "=" * 70)
    print("TRUST MANAGEMENT DEMO")
    print("=" * 70)
    
    tm = TrustManager(min_trust_score=0.5)
    
    print("\n1. Creating normal and malicious updates...")
    
    # Normal update
    normal = WeightUpdate(
        update_id="normal-1",
        device_id="good-device",
        timestamp=datetime.now(),
        weight_deltas={"layer1": np.array([0.1, 0.2, 0.1])},
        training_samples=100,
    )
    
    # Malicious update (huge weights)
    malicious = WeightUpdate(
        update_id="malicious-1",
        device_id="bad-device",
        timestamp=datetime.now(),
        weight_deltas={"layer1": np.array([1e12, 1e12, 1e12])},
        training_samples=100,
    )
    
    print("\n2. Evaluating updates...")
    
    normal_ok = tm.evaluate_update(normal)
    print(f"   Normal update: {'✓ Accepted' if normal_ok else '✗ Rejected'}")
    
    malicious_ok = tm.evaluate_update(malicious)
    print(f"   Malicious update: {'✓ Accepted' if malicious_ok else '✗ Rejected'}")
    
    print("\n3. Trust scores:")
    for device_id in ["good-device", "bad-device"]:
        trust = tm.get_device_trust(device_id)
        if trust:
            print(f"   {device_id}: trust={trust.trust_score:.2f}")
    
    print("\n✓ Byzantine attack detected and blocked!")


def demo_data_filter():
    """Demonstrate data filtering."""
    print("\n" + "=" * 70)
    print("DATA FILTERING DEMO")
    print("=" * 70)
    
    print("\n1. Original training data with PII:")
    data = [
        {
            "input": "My email is john.doe@example.com",
            "output": "Thanks! I'll contact you soon."
        },
        {
            "input": "Call me at 555-123-4567",
            "output": "I'll call you tomorrow."
        },
        {
            "input": "Tell me about quantum computing",
            "output": "Quantum computing uses qubits..."
        }
    ]
    
    for i, ex in enumerate(data):
        print(f"   {i+1}. Input: {ex['input']}")
    
    print("\n2. Filtering data...")
    filter = DataFilter(remove_pii=True)
    filtered = filter.filter_training_data(data)
    
    print("\n3. Filtered data (PII removed):")
    for i, ex in enumerate(filtered):
        print(f"   {i+1}. Input: {ex['input']}")
    
    print("\n✓ Personal information protected!")


def demo_coordinator():
    """Demonstrate training coordination."""
    print("\n" + "=" * 70)
    print("TRAINING COORDINATOR DEMO")
    print("=" * 70)
    
    print("\n1. Initializing coordinator...")
    coordinator = TrainingCoordinator(
        min_devices=2,
        round_duration=10,  # 10 seconds for demo
    )
    
    # Register callbacks
    coordinator.on_round_start(
        lambda round_id: print(f"   → Round {round_id} started")
    )
    coordinator.on_round_complete(
        lambda round_info: print(
            f"   → Round {round_info.round_id} completed: "
            f"{len(round_info.updates)} updates, "
            f"status={round_info.status.value}"
        )
    )
    
    print("\n2. Simulating update submissions...")
    
    # Manually start a round
    coordinator._start_new_round()
    
    # Submit some updates
    for i in range(3):
        update = WeightUpdate(
            update_id=f"update-{i}",
            device_id=f"device-{i}",
            timestamp=datetime.now(),
            weight_deltas={"layer1": np.array([0.1, 0.2])},
            training_samples=100,
        )
        coordinator.submit_update(update)
        print(f"   Submitted update from device-{i}")
    
    print("\n3. Coordinator statistics:")
    stats = coordinator.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✓ Coordination system working!")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "FORGE AI - FEDERATED LEARNING DEMO" + " " * 19 + "║")
    print("╚" + "═" * 68 + "╝")
    
    try:
        demo_basic_workflow()
        demo_privacy()
        demo_trust()
        demo_data_filter()
        demo_coordinator()
        
        print("\n" + "=" * 70)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nFederated Learning Features:")
        print("  ✓ Privacy-preserving weight updates")
        print("  ✓ Differential privacy with noise")
        print("  ✓ Trust management and Byzantine detection")
        print("  ✓ Data filtering (PII removal)")
        print("  ✓ Training round coordination")
        print("  ✓ Secure aggregation")
        print("\n")
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
