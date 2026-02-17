# Stub file - original module deleted in Feb 2026 cleanup

class SpawnableObject:
    def __init__(self, *args, **kwargs):
        self.blocked = False
        self.blocked_reason = ""

class ObjectSpawner:
    def __init__(self, *args, **kwargs): pass
    def spawn(self, *args, **kwargs): return SpawnableObject()
    def despawn(self, *args, **kwargs): pass

_instance = None
def get_spawner():
    global _instance
    if _instance is None:
        _instance = ObjectSpawner()
    return _instance
