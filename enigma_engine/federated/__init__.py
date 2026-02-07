"""
================================================================================
FEDERATED LEARNING - SHARE IMPROVEMENTS WITHOUT SHARING DATA
================================================================================

Federated learning allows devices to collaboratively train models by sharing
only model updates (gradients), never raw data. This preserves privacy while
allowing collective improvement.

FILE: enigma_engine/federated/__init__.py
TYPE: Package Initialization
EXPORTS: Main federated learning classes

HOW IT WORKS:
    1. Each device trains on its own private data
    2. Devices share only weight updates (deltas)
    3. Coordinator aggregates updates from all participants
    4. Improved model distributed back to all devices
    5. No raw data ever leaves any device

MAIN COMPONENTS:
    - FederatedLearning: Main coordinator/participant interface
    - FederatedAggregator: Aggregates updates using FedAvg
    - DifferentialPrivacy: Adds noise for privacy protection
    - UpdateCompressor: Compresses updates for bandwidth efficiency
    - FederatedCoordinator: Manages training rounds
    - FederatedParticipant: Client that contributes updates
    - FederationDiscovery: Discovers available federations

USAGE:
    # Create a federation (coordinator)
    from enigma_engine.federated import FederatedLearning, FederationMode
    
    fl = FederatedLearning(role="coordinator")
    federation_id = fl.create_federation("MyFederation", FederationMode.PRIVATE)
    
    # Join a federation (participant)
    fl = FederatedLearning(role="participant")
    fl.join_federation(federation_id)
    
    # Start training round (coordinator)
    await fl.coordinator.run_round()
    
    # Participate in round (participant)
    await fl.participant.participate_in_round(round_number)

PRIVACY FEATURES:
    - Differential privacy: Adds calibrated noise to updates
    - Secure aggregation: Even coordinator can't see individual updates
    - Update compression: Reduces bandwidth and limits information leakage
    - Local training only: Raw data never leaves device

SEE ALSO:
    • enigma_engine/federated/federation.py - Core classes
    • enigma_engine/federated/aggregation.py - Update aggregation
    • enigma_engine/federated/privacy.py - Privacy protection
    • enigma_engine/federated/compression.py - Update compression
"""

from .aggregation import (
    FederatedAggregator,
    SecureAggregation,
)
from .compression import (
    CompressedUpdate,
    SparseUpdate,
    UpdateCompressor,
)
from .coordinator import (
    FederatedCoordinator,
)
from .discovery import (
    FederationDiscovery,
)
from .federation import (
    FederatedLearning,
    FederationInfo,
    FederationMode,
    FederationRole,
    ModelUpdate,
)
from .participant import (
    FederatedParticipant,
)
from .privacy import (
    DifferentialPrivacy,
)

__all__ = [
    'FederatedLearning',
    'FederationRole',
    'FederationMode',
    'ModelUpdate',
    'FederationInfo',
    'FederatedAggregator',
    'SecureAggregation',
    'DifferentialPrivacy',
    'UpdateCompressor',
    'CompressedUpdate',
    'SparseUpdate',
    'FederatedCoordinator',
    'FederatedParticipant',
    'FederationDiscovery',
]
