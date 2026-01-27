"""
Federated Learning Module for ForgeAI

Privacy-preserving distributed learning that allows devices to learn
from collective experience without sharing raw data.
"""

from .federated import (
    FederatedLearning,
    WeightUpdate,
    FederatedMode,
    PrivacyLevel,
)
from .privacy import DifferentialPrivacy
from .aggregation import SecureAggregator, AggregationMethod
from .coordinator import FederatedCoordinator, CoordinatorMode
from .data_filter import FederatedDataFilter, TrainingExample
from .trust import TrustManager

__all__ = [
    "FederatedLearning",
    "WeightUpdate",
    "FederatedMode",
    "PrivacyLevel",
    "DifferentialPrivacy",
    "SecureAggregator",
    "AggregationMethod",
    "FederatedCoordinator",
    "CoordinatorMode",
    "FederatedDataFilter",
    "TrainingExample",
    "TrustManager",
]
