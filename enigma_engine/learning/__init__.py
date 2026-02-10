"""
================================================================================
Enigma AI Engine LEARNING MODULE
================================================================================

Comprehensive learning system for Enigma AI Engine combining:
- Real-time conversation learning (corrections, teaching, feedback)
- Model bootstrapping (blank/quick/conversational/knowledgeable)
- Federated learning (privacy-preserving distributed learning)

BUILDS ON EXISTING SYSTEMS:
    → enigma_engine.core.self_improvement (LearningEngine, LearningExample)
    → enigma_engine.core.autonomous (AutonomousLearner)
    → enigma_engine.core.training (Trainer)

NEW COMPONENTS:
    → ModelBootstrap: Initialize models with different starting points
    → ConversationDetector: Detect learning opportunities in real-time
    → LearningChatIntegration: Hook learning into chat system

USAGE:
    # Bootstrap a new model
    from enigma_engine.learning import ModelBootstrap, StartingPoint
    model = ModelBootstrap.initialize("small", StartingPoint.CONVERSATIONAL)
    
    # Add learning to chat
    from enigma_engine.learning import LearningChatIntegration
    integration = LearningChatIntegration(model)
    
    # In chat loop:
    detected = integration.before_response(user_msg)
    response = generate_response(user_msg)
    integration.after_response(response)
================================================================================
"""

# =============================================================================
# NEW LEARNING COMPONENTS
# =============================================================================

# Model bootstrap (from core module)
from ..core.model_bootstrap import (
    ModelBootstrap,
    StartingPoint,
    bootstrap_model,
    list_starting_points,
)
from .aggregation import AggregationMethod, SecureAggregator

# Chat system integration
from .chat_integration import (
    LearningChatIntegration,
    LearningChatWrapper,
    create_chat_integration,
)

# Conversation learning detection
from .conversation_detector import (
    ConversationDetector,
    DetectedLearning,
    detect_learning,
    is_correction,
    is_feedback,
    is_teaching,
)

# User-teachable behavior preferences

from .coordinator import CoordinatorMode, FederatedCoordinator
from .data_filter import FederatedDataFilter, TrainingExample
from .federated import (
    FederatedLearning,
    FederatedMode,
    PrivacyLevel,
    WeightUpdate,
)
from .privacy import DifferentialPrivacy
from .trust import TrustManager

# =============================================================================
# FEDERATED LEARNING COMPONENTS (Existing)
# =============================================================================


# Alias for backwards compatibility
DataFilter = FederatedDataFilter

__all__ = [
    # New learning components
    "ConversationDetector",
    "DetectedLearning",
    "detect_learning",
    "is_correction",
    "is_teaching",
    "is_feedback",
    "LearningChatIntegration",
    "LearningChatWrapper",
    "create_chat_integration",
    "ModelBootstrap",
    "StartingPoint",
    "bootstrap_model",
    "list_starting_points",
    # Federated learning (existing)
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
    "DataFilter",  # Alias
    "TrainingExample",
    "TrustManager",
]
