# Stub file - original module deleted in Feb 2026 cleanup

class EffectManager:
    def __init__(self, *args, **kwargs): pass
    def add_effect(self, *args, **kwargs): pass
    def remove_effect(self, *args, **kwargs): pass
    def clear(self): pass

_instance = None
def get_effect_manager():
    global _instance
    if _instance is None:
        _instance = EffectManager()
    return _instance
