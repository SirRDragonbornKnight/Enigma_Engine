"""
================================================================================
FORGEAI LEARNING MODULE
================================================================================

Comprehensive learning system for ForgeAI combining:
- Real-time conversation learning (corrections, teaching, feedback)
- Model bootstrapping (blank/quick/conversational/knowledgeable)
- Federated learning (privacy-preserving distributed learning)

BUILDS ON EXISTING SYSTEMS:
    → forge_ai.core.self_improvement (LearningEngine, LearningExample)
    → forge_ai.core.autonomous (AutonomousLearner)
    → forge_ai.core.training (Trainer)

NEW COMPONENTS:
    → ModelBootstrap: Initialize models with different starting points
    → ConversationDetector: Detect learning opportunities in real-time
    → LearningChatIntegration: Hook learning into chat system

USAGE:
    # Bootstrap a new model
    from forge_ai.learning import ModelBootstrap, StartingPoint
    model = ModelBootstrap.initialize("small", StartingPoint.CONVERSATIONAL)
    
    # Add learning to chat
    from forge_ai.learning import LearningChatIntegration
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

# Conversation learning detection
from .conversation_detector import (
    ConversationDetector,
    DetectedLearning,
    detect_learning,
    is_correction,
    is_teaching,
    is_feedback,
)

# Chat system integration
from .chat_integration import (
    LearningChatIntegration,
    LearningChatWrapper,
    create_chat_integration,
)

# Model bootstrap (from core module)
from ..core.model_bootstrap import (
    ModelBootstrap,
    StartingPoint,
    bootstrap_model,
    list_starting_points,
)

# =============================================================================
# FEDERATED LEARNING COMPONENTS (Existing)
# =============================================================================

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
