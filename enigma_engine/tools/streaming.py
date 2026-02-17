# Stub file - original module deleted in Feb 2026 cleanup

class StreamingToolResult:
    def __init__(self, *args, **kwargs): pass

class StreamingToolExecutor:
    def __init__(self, *args, **kwargs): pass
    def execute(self, tool): return None

class StreamState:
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETE = 'complete'
    ERROR = 'error'
    def __init__(self, *args, **kwargs):
        self.state = self.PENDING
