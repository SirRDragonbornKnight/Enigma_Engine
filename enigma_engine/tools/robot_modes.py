# Stub file - original module deleted in Feb 2026 cleanup

class RobotMode:
    MANUAL = "manual"
    AUTO = "auto"

class CameraConfig:
    def __init__(self, *args, **kwargs): pass

class RobotModeController:
    def __init__(self, *args, **kwargs): pass
    def set_mode(self, mode): pass

def get_mode_controller(): return RobotModeController()
