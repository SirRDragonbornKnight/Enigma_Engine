# Stub file - original module deleted in Feb 2026 cleanup

class SpeechSyncConfig:
    def __init__(self, *args, **kwargs): pass

class SpeechSync:
    def __init__(self, *args, **kwargs): pass
    def start(self): pass
    def stop(self): pass
    def speak(self, text): pass

_instance = None
def get_speech_sync():
    global _instance
    if _instance is None:
        _instance = SpeechSync()
    return _instance

def sync_speak(text): pass
def set_avatar_for_sync(avatar): pass
def create_voice_avatar_bridge(*args, **kwargs): return None
