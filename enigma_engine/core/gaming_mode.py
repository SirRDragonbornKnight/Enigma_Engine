# Stub file - original module deleted in Feb 2026 cleanup

class GamingProfile:
    def __init__(self, *args, **kwargs): pass

class GamingMode:
    def __init__(self, *args, **kwargs): pass
    def start(self): pass
    def stop(self): pass
    def is_active(self): return False

DEFAULT_GAMING_PROFILES = {}

def get_gaming_mode(): return GamingMode()
