"""
Federated Learning Module for ForgeAI

Privacy-preserving distributed learning where devices train locally
and share only model improvements (weight updates), never raw data.
"""

from .federated import (
    FederatedLearning,
    WeightUpdate,
    FederatedMode,
    PrivacyLevel,
)
from .privacy import DifferentialPrivacy
from .aggregation import SecureAggregator, AggregationMethod
from .coordinator import TrainingCoordinator
from .data_filter import DataFilter
from .trust import TrustManager

__all__ = [
    "FederatedLearning",
    "WeightUpdate",
    "FederatedMode",
    "PrivacyLevel",
    "DifferentialPrivacy",
    "SecureAggregator",
    "AggregationMethod",
    "TrainingCoordinator",
    "DataFilter",
    "TrustManager",
]
