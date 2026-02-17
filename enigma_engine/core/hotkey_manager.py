# Stub file - original module deleted in Feb 2026 cleanup

DEFAULT_HOTKEYS = {}

class HotkeyManager:
    def __init__(self, *args, **kwargs): pass
    def register(self, key, callback): pass
    def unregister(self, key): pass
    def start(self): pass
    def stop(self): pass
    def list_registered(self): return []
    def get_hotkey(self, name): return None
    def set_hotkey(self, name, key): pass

def get_hotkey_manager(): return HotkeyManager()
