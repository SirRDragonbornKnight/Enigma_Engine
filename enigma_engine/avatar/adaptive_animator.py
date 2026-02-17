# Stub file - original module deleted in Feb 2026 cleanup

class AnimationState:
    IDLE = "idle"
    TALKING = "talking"
    THINKING = "thinking"

class AnimationStrategy:
    BONE_BASED = "bone_based"
    MORPH_BASED = "morph_based"

class ModelCapabilities:
    def __init__(self, *args, **kwargs):
        self.has_bones = True
        self.has_morphs = False

class AdaptiveAnimator:
    def __init__(self, *args, **kwargs): 
        self._transform_callbacks = []
        self._model_info = None
    def set_state(self, state): pass
    def update(self, dt=0): pass
    def on_transform_update(self, callback): 
        self._transform_callbacks.append(callback)
    def get_transform(self): 
        return {"position": (0, 0, 0), "rotation": (0, 0, 0), "scale": (1, 1, 1)}
    def set_animation(self, name): pass
    def stop(self): pass
    def start(self): pass
    def set_model_info(self, info): 
        self._model_info = info
    def get_model_info(self): 
        return self._model_info
    def analyze_model(self, *args): 
        return ModelCapabilities()
    def get_capabilities_for_ai(self):
        return {"bones": True, "morphs": False, "animations": []}
    def set_idle_animation(self, name): pass
    def play_animation(self, name, loop=False): pass
    def blend_to(self, name, duration=0.5): pass
