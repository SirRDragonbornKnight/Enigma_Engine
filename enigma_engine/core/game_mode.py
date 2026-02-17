# Stub file - original module deleted in Feb 2026 cleanup

class GameMode:
    def __init__(self, *args, **kwargs):
        self._callbacks_detected = []
        self._callbacks_ended = []
    def is_active(self): return False
    def enter(self): pass
    def exit(self): pass
    def on_game_detected(self, callback):
        self._callbacks_detected.append(callback)
    def on_game_ended(self, callback):
        self._callbacks_ended.append(callback)
    def on_game_started(self, callback): pass
    def on_limits_changed(self, callback): pass

_instance = None

def get_game_mode():
    global _instance
    if _instance is None:
        _instance = GameMode()
    return _instance
