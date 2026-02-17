# Stub file - original module deleted in Feb 2026 cleanup

class AutonomousMode:
    def __init__(self, *args, **kwargs): pass
    def start(self): pass
    def stop(self): pass
    def is_active(self): return False

def get_autonomous_mode(): return AutonomousMode()
