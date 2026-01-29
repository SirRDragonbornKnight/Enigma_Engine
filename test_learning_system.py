"""Test script for the learning system."""
from forge_ai.learning import ConversationDetector, detect_learning, is_correction, is_teaching

# Test correction detection
detector = ConversationDetector()
detector.last_ai_response = 'Nice to meet you, Alice!'

tests = [
    ('No, my name is Bob', 'correction'),
    ("That's wrong, I asked about Python", 'correction'),
    ('Remember that I prefer dark mode', 'teaching'),
    ('My name is Alex', 'teaching'),
    ('Good job!', 'positive_feedback'),
    ('Thanks, that helped!', 'positive_feedback'),
    ("That's not helpful", 'negative_feedback'),
    ("I don't understand", 'negative_feedback'),
]

print('Testing ConversationDetector:')
print('-' * 50)
for msg, expected in tests:
    detected = detector.on_user_message(msg)
    result = detected.type if detected else 'None'
    status = '[OK]' if result == expected else '[!!]'
    print(f'{status} "{msg[:40]}..." -> {result} (expected: {expected})')
    detector.reset_context()

print('-' * 50)
print('Stats:', detector.get_stats())
