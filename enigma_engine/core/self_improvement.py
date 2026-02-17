# Stub file - original module deleted in Feb 2026 cleanup

class LearningSource:
    CONVERSATION = "conversation"
    FEEDBACK = "feedback"
    SELF = "self"

class Priority:
    LOW = 0
    MEDIUM = 1
    HIGH = 2

class LearningEngine:
    def __init__(self, *args, **kwargs): pass
    def learn(self, *args, **kwargs): pass

def get_learning_engine(): return LearningEngine()
