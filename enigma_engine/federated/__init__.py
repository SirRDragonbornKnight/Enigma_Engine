"""Federated learning stub - feature disabled."""

from enum import Enum, auto


class FederationMode(Enum):
    LOCAL = auto()
    PEER_TO_PEER = auto()
    CENTRALIZED = auto()


class FederationRole(Enum):
    CLIENT = auto()
    SERVER = auto()
    COORDINATOR = auto()


class FederatedLearning:
    """Stub - federated learning disabled."""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def create_federation(self, *args, **kwargs):
        return None
    
    def join_federation(self, *args, **kwargs):
        return False
    
    def leave_federation(self, *args, **kwargs):
        pass
    
    def get_stats(self):
        return {}


class FederationDiscovery:
    """Stub - federation discovery disabled."""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def discover(self, *args, **kwargs):
        return []
    
    def start(self):
        pass
    
    def stop(self):
        pass
