# Federated Learning System

Privacy-preserving distributed learning for ForgeAI where devices train locally and share only model improvements (weight updates), never raw data.

## Overview

The federated learning system allows multiple ForgeAI instances to collaboratively improve their models without sharing private training data. Each device:

1. **Trains locally** on its own data
2. **Computes weight deltas** (only the changes, not the full model)
3. **Applies privacy protection** (differential privacy, anonymization)
4. **Shares updates** with other devices
5. **Receives aggregated updates** and applies them to local model

## Key Features

### Privacy Protection
- **Differential Privacy**: Add calibrated noise to prevent data reconstruction
- **Data Filtering**: Remove PII and inappropriate content automatically
- **Anonymization**: Device IDs are hashed based on privacy level
- **No Raw Data Sharing**: Only weight deltas are transmitted

### Security
- **Trust Management**: Track device reputation and detect malicious updates
- **Byzantine Detection**: Identify and filter poisoned/corrupted updates
- **Cryptographic Signatures**: Verify update authenticity
- **Secure Aggregation**: Multiple aggregation methods including median (robust to outliers)

### Coordination
- **Training Rounds**: Automatic coordination of training cycles
- **Peer Discovery**: Find other federated-capable devices on network
- **Flexible Aggregation**: Simple average, weighted by samples, or median
- **Configurable Privacy**: Adjust privacy vs utility trade-off

## Components

### Core Classes

#### `FederatedLearning`
Main federated learning system that handles local training and update creation.

```python
from forge_ai.learning import FederatedLearning, FederatedMode, PrivacyLevel

fl = FederatedLearning(
    mode=FederatedMode.OPT_IN,
    privacy_level=PrivacyLevel.HIGH
)

# Start training round
fl.start_local_round(model_weights)

# After training
update = fl.train_local_round(
    model_weights=updated_weights,
    training_samples=100,
    loss=0.5,
    accuracy=0.9
)

# Share update
fl.share_update(update)

# Apply global update
new_weights = fl.receive_global_update(global_update)
```

#### `WeightUpdate`
Represents a model update containing only weight deltas.

```python
from forge_ai.learning import WeightUpdate
import numpy as np
from datetime import datetime

update = WeightUpdate(
    update_id="unique-id",
    device_id="device-1",
    timestamp=datetime.now(),
    weight_deltas={"layer1": np.array([0.1, 0.2, 0.3])},
    training_samples=100,
    metadata={"loss": 0.5, "accuracy": 0.9}
)

# Sign and verify
update.sign()
assert update.verify_signature()
```

#### `DifferentialPrivacy`
Add noise to weights for privacy protection.

```python
from forge_ai.learning import DifferentialPrivacy

dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
noisy_weights = dp.add_noise(weights)

# Check privacy loss
total_loss = dp.compute_privacy_loss(num_rounds=10)
```

#### `SecureAggregator`
Aggregate updates from multiple devices.

```python
from forge_ai.learning import SecureAggregator, AggregationMethod

aggregator = SecureAggregator(method=AggregationMethod.WEIGHTED)

# Aggregate updates
global_update = aggregator.aggregate_updates(updates)

# Validate first
valid_updates = aggregator.validate_updates(updates)
```

#### `TrustManager`
Manage device trust and detect malicious updates.

```python
from forge_ai.learning import TrustManager

tm = TrustManager(min_trust_score=0.3)

# Evaluate update
is_trusted = tm.evaluate_update(update)

# Detect poisoning
clean_updates = tm.detect_poisoning(updates)

# Ban malicious device
tm.ban_device("bad-device-id")
```

#### `DataFilter`
Filter training data for privacy and quality.

```python
from forge_ai.learning import DataFilter

filter = DataFilter(
    remove_pii=True,
    remove_inappropriate=True,
    min_length=10,
    max_length=1000
)

filtered_data = filter.filter_training_data(data)
```

#### `TrainingCoordinator`
Coordinate training rounds across devices.

```python
from forge_ai.learning import TrainingCoordinator

coordinator = TrainingCoordinator(
    min_devices=2,
    round_duration=300
)

# Register callbacks
coordinator.on_round_start(lambda round_id: print(f"Round {round_id}"))
coordinator.on_round_complete(lambda info: print(f"Completed!"))

# Start coordination
coordinator.start_coordination()

# Submit updates
coordinator.submit_update(update)
```

## Configuration

Add to `forge_config.json` or use environment variables:

```json
{
  "federated": {
    "mode": "opt_in",
    "privacy_level": "high",
    "epsilon": 1.0,
    "delta": 1e-5,
    "min_devices": 2,
    "round_duration": 300,
    "aggregation_method": "weighted",
    "min_trust_score": 0.3,
    "byzantine_threshold": 3.0,
    "enable_data_filtering": true,
    "remove_pii": true,
    "remove_inappropriate": true
  }
}
```

### Configuration Options

- **mode**: `"opt_in"`, `"opt_out"`, or `"disabled"`
- **privacy_level**: `"none"`, `"low"`, `"medium"`, `"high"`, or `"maximum"`
- **epsilon**: Privacy budget (lower = more private, more noise)
- **delta**: Privacy parameter (probability of breach, typically 1e-5)
- **min_devices**: Minimum devices required for aggregation
- **round_duration**: Training round duration in seconds
- **aggregation_method**: `"simple"`, `"weighted"`, `"median"`, or `"secure"`
- **min_trust_score**: Minimum trust score to accept updates (0.0-1.0)
- **byzantine_threshold**: Standard deviations for outlier detection
- **enable_data_filtering**: Filter training data for privacy
- **remove_pii**: Remove personally identifiable information
- **remove_inappropriate**: Filter inappropriate content

## Privacy Levels

### None
- No privacy protection
- Full device ID visible
- No noise added to updates

### Low
- Anonymize device ID (truncated hash)
- No differential privacy

### Medium
- Anonymize device ID
- **Differential privacy** with moderate noise

### High (Recommended)
- Anonymize device ID
- Differential privacy
- **Secure aggregation**

### Maximum
- Anonymize device ID
- Differential privacy
- Secure aggregation
- **Homomorphic encryption** (future)

## Peer Discovery

Find other federated learning capable devices:

```python
from forge_ai.comms.discovery import discover_federated_learning_peers

peers = discover_federated_learning_peers(
    node_name="my-device",
    timeout=3.0,
    min_trust_score=0.5
)

for name, info in peers.items():
    print(f"{name}: {info['ip']}:{info['port']}")
    print(f"  Federated: {info['federated']}")
```

## GUI Integration

Use the federated widget in your GUI:

```python
from forge_ai.gui.widgets.federated_widget import FederatedWidget

widget = FederatedWidget()
layout.addWidget(widget)

# Get current settings
settings = widget.get_settings()
```

## Demo

Run the included demo to see all features:

```bash
python demo_federated_learning.py
```

This demonstrates:
- Basic workflow (train → aggregate → apply)
- Differential privacy with different epsilon values
- Trust management and Byzantine attack detection
- Data filtering (PII removal)
- Training coordination

## Testing

Run the test suite:

```bash
pytest tests/learning/test_federated.py -v
```

## Architecture

```
forge_ai/learning/
├── __init__.py           # Public API
├── federated.py          # Main FL system, WeightUpdate
├── privacy.py            # Differential privacy
├── aggregation.py        # Secure aggregation
├── coordinator.py        # Training coordination
├── data_filter.py        # Data privacy filtering
└── trust.py             # Trust management
```

## Security Considerations

1. **Data Privacy**: Training data never leaves the device
2. **Model Privacy**: Weight updates protected with differential privacy
3. **Byzantine Resilience**: Trust system and outlier detection
4. **Secure Communication**: Use HTTPS for update transmission (when deployed)
5. **Access Control**: Opt-in by default, configurable participation

## Performance Tips

1. **Privacy vs Utility**: Lower epsilon = more privacy but noisier updates
2. **Aggregation Method**: Use weighted for heterogeneous data sizes
3. **Round Duration**: Longer rounds allow more devices to participate
4. **Trust Threshold**: Lower threshold accepts more devices but increases risk
5. **Data Filtering**: Enable for privacy but may reduce training data

## Future Enhancements

- [ ] Secure multi-party computation (MPC)
- [ ] Homomorphic encryption
- [ ] Personalized federated learning
- [ ] Adaptive privacy budgets
- [ ] Cross-device model testing
- [ ] Federated analytics

## References

- McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data" (2017)
- Abadi et al. "Deep Learning with Differential Privacy" (2016)
- Bonawitz et al. "Practical Secure Aggregation for Privacy-Preserving Machine Learning" (2017)

## License

Part of ForgeAI - See LICENSE file for details.
