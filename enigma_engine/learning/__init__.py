"""Learning Module - STUBBED (legacy federated/chat learning removed)"""

# Stub classes and functions
class ModelBootstrap: pass
class StartingPoint: BLANK = "blank"; CONVERSATIONAL = "conversational"
def bootstrap_model(*a, **k): return None
def list_starting_points(): return []
class AggregationMethod: pass
class SecureAggregator: pass
class LearningChatIntegration:
    def __init__(self, *args, **kwargs): pass
class LearningChatWrapper: pass
def create_chat_integration(*a, **k): return None
class ConversationDetector: pass
class DetectedLearning: pass
def detect_learning(*a, **k): return None
def is_correction(*a, **k): return False
def is_feedback(*a, **k): return False
def is_teaching(*a, **k): return False
class CoordinatorMode: pass
class FederatedCoordinator: pass
class FederatedDataFilter: pass
class TrainingExample: pass
class FederatedLearning: pass
class FederatedMode: pass
class PrivacyLevel: pass
class WeightUpdate: pass
class DifferentialPrivacy: pass
class TrustManager: pass
DataFilter = FederatedDataFilter

__all__ = [
    "ConversationDetector", "DetectedLearning", "detect_learning", "is_correction",
    "is_teaching", "is_feedback", "LearningChatIntegration", "LearningChatWrapper",
    "create_chat_integration", "ModelBootstrap", "StartingPoint", "bootstrap_model",
    "list_starting_points", "FederatedLearning", "WeightUpdate", "FederatedMode",
    "PrivacyLevel", "DifferentialPrivacy", "SecureAggregator", "AggregationMethod",
    "FederatedCoordinator", "CoordinatorMode", "FederatedDataFilter", "DataFilter",
    "TrainingExample", "TrustManager",
]
