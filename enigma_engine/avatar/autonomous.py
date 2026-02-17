# Stub file - original module deleted in Feb 2026 cleanup

class AutonomousConfig:
    def __init__(self, *args, **kwargs):
        self.enabled = False

class AvatarMood:
    NEUTRAL = "neutral"

class ScreenRegion:
    def __init__(self, *args, **kwargs): pass

class AutonomousAvatar:
    def __init__(self, *args, **kwargs): pass
    def start(self): pass
    def stop(self): pass

_instance = None
def get_autonomous_avatar():
    global _instance
    if _instance is None:
        _instance = AutonomousAvatar()
    return _instance
