# Stub file - original module deleted in Feb 2026 cleanup

DEFAULT_RATE_LIMITS = {
    'default': 60,  # calls per minute
    'web_search': 10,
    'file_write': 30,
    'screenshot': 20,
}

class RateLimiter:
    def __init__(self, *args, **kwargs): pass
    def check(self): return True
    def wait(self): pass
