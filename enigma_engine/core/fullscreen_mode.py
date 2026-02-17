# Stub file - original module deleted in Feb 2026 cleanup

class FullscreenSettings:
    def __init__(self):
        self.auto_hide_on_fullscreen = False
        self.fade_transparency = False
        self.hide_avatar = False
        self.hide_objects = False
        self.hide_effects = False
        self.hide_particles = False
        self.hotkey = 'F11'

class FullscreenController:
    def __init__(self, *args, **kwargs):
        self._settings = FullscreenSettings()
    def enter(self): pass
    def exit(self): pass
    def is_active(self): return False
    def load_settings(self): pass
    def save_settings(self): pass

def get_fullscreen_controller(): return FullscreenController()
