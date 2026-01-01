#!/usr/bin/env python3
"""
Demo script showcasing all new Enigma AI Engine enhancements.

This demonstrates:
1. Enhanced memory system with vector databases and categorization
2. Dynamic personality with user-tunable traits
3. Context-aware conversations
4. Bias detection and ethics tools
5. Enhanced web safety
6. Theme system

Usage:
    python demo_comprehensive_enhancements.py
"""
import sys
from pathlib import Path
import numpy as np

# Add enigma to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("ENIGMA AI ENGINE - COMPREHENSIVE ENHANCEMENTS DEMO")
print("=" * 80)
print()

# ============================================================================
# 1. ENHANCED MEMORY SYSTEM
# ============================================================================
print("📝 1. ENHANCED MEMORY SYSTEM")
print("-" * 80)

from enigma.memory.vector_db import SimpleVectorDB, create_vector_db
from enigma.memory.categorization import MemoryCategorization, MemoryType
from enigma.memory.export_import import MemoryExporter, MemoryImporter

# Vector database
print("\n✓ Creating vector database...")
vector_db = create_vector_db(dim=128, backend='simple')

# Add some vectors
vectors = np.random.rand(3, 128)
ids = ['mem1', 'mem2', 'mem3']
metadata = [
    {'content': 'User likes Python'},
    {'content': 'User prefers dark themes'},
    {'content': 'User interested in AI'}
]
vector_db.add(vectors, ids, metadata)
print(f"  Added {vector_db.count()} vectors to database")

# Search
query = np.random.rand(128)
results = vector_db.search(query, top_k=2)
print(f"  Search found {len(results)} similar memories")

# Memory categorization
print("\n✓ Setting up memory categorization...")
mem_system = MemoryCategorization()

# Add different types of memories
mem_system.add_memory(
    "User's name is Alice",
    memory_type=MemoryType.LONG_TERM,
    importance=1.0
)
mem_system.add_memory(
    "Current conversation about Python",
    memory_type=MemoryType.SHORT_TERM,
    ttl=86400  # 1 day
)
mem_system.add_memory(
    "User just asked about AI",
    memory_type=MemoryType.WORKING,
    ttl=3600  # 1 hour
)

stats = mem_system.get_statistics()
print(f"  Total memories: {stats['total']}")
print(f"  By type: {stats['by_type']}")

# Export/Import
print("\n✓ Testing memory export/import...")
exporter = MemoryExporter(mem_system)
export_stats = exporter.export_to_json(Path('/tmp/enigma_memories_export.json'))
print(f"  Exported {export_stats['exported_count']} memories")

# ============================================================================
# 2. DYNAMIC PERSONALITY SYSTEM
# ============================================================================
print("\n" + "=" * 80)
print("🎭 2. DYNAMIC PERSONALITY SYSTEM")
print("-" * 80)

from enigma.core.personality import AIPersonality

personality = AIPersonality("demo_model")

print("\n✓ Initial personality traits:")
print(f"  Humor: {personality.traits.humor_level:.2f}")
print(f"  Formality: {personality.traits.formality:.2f}")
print(f"  Creativity: {personality.traits.creativity:.2f}")

# Apply user overrides
print("\n✓ Applying user overrides...")
personality.set_user_override('humor_level', 0.9)
personality.set_user_override('formality', 0.2)

print(f"  Humor (overridden): {personality.get_effective_trait('humor_level'):.2f}")
print(f"  Formality (overridden): {personality.get_effective_trait('formality'):.2f}")

# Apply preset
print("\n✓ Applying 'comedian' preset...")
personality.set_preset('comedian')

print("  Effective traits after preset:")
for trait, value in personality.get_all_effective_traits().items():
    print(f"    {trait}: {value:.2f}")

# Get personality prompt
prompt = personality.get_personality_prompt()
print(f"\n✓ Generated system prompt:\n  {prompt}")

# ============================================================================
# 3. CONTEXT AWARENESS
# ============================================================================
print("\n" + "=" * 80)
print("🗣️ 3. CONTEXT AWARENESS")
print("-" * 80)

from enigma.core.context_awareness import ContextAwareConversation

conversation = ContextAwareConversation()

print("\n✓ Processing conversation turns...")

# Turn 1
result1 = conversation.process_user_input("Hi, I'm interested in Python programming.")
print(f"\n  User: Hi, I'm interested in Python programming.")
print(f"  Needs clarification: {result1['needs_clarification']}")
conversation.add_assistant_response("Great! Python is an excellent language.")

# Turn 2 - unclear input
result2 = conversation.process_user_input("What about it?")
print(f"\n  User: What about it?")
print(f"  Needs clarification: {result2['needs_clarification']}")
if result2['needs_clarification']:
    print(f"  Clarification: {result2['clarification_prompt']}")

# Context summary
print(f"\n✓ Context summary: {result2['context_summary']}")

# ============================================================================
# 4. BIAS DETECTION & ETHICS
# ============================================================================
print("\n" + "=" * 80)
print("🛡️ 4. BIAS DETECTION & ETHICS")
print("-" * 80)

from enigma.tools.bias_detection import (
    BiasDetector,
    OffensiveContentFilter,
    SafeReinforcementLogic
)

# Bias detection
print("\n✓ Testing bias detection...")
detector = BiasDetector()

test_texts = [
    "The engineer worked on his project.",
    "The nurse was caring and helpful.",
    "Scientists use logic to solve problems."
]

result = detector.scan_dataset(test_texts)
print(f"  Scanned {result.statistics['total_samples']} texts")
print(f"  Bias score: {result.bias_score:.2f}")
print(f"  Issues found: {len(result.issues_found)}")

# Offensive content filtering
print("\n✓ Testing offensive content filter...")
content_filter = OffensiveContentFilter()

safe_text = "This is a nice day"
unsafe_text = "This is damn annoying"

safe_result = content_filter.scan_text(safe_text)
unsafe_result = content_filter.scan_text(unsafe_text)

print(f"  Safe text offensive: {safe_result['is_offensive']}")
print(f"  Unsafe text offensive: {unsafe_result['is_offensive']}")

if unsafe_result['is_offensive']:
    filtered = content_filter.filter_text(unsafe_text)
    print(f"  Filtered: {filtered}")

# Safe reinforcement
print("\n✓ Testing safe reinforcement logic...")
logic = SafeReinforcementLogic()

output = "I'll help you learn Python. It's a great language!"
safety_check = logic.check_output_safety(output)

print(f"  Output is safe: {safety_check['is_safe']}")
print(f"  Confidence: {safety_check['confidence']:.2f}")

# ============================================================================
# 5. ENHANCED WEB SAFETY
# ============================================================================
print("\n" + "=" * 80)
print("🌐 5. ENHANCED WEB SAFETY")
print("-" * 80)

from enigma.tools.url_safety import URLSafety

safety = URLSafety(enable_auto_update=False)

print("\n✓ Testing URL safety checks...")

test_urls = [
    "https://github.com/test",
    "http://example.com/malware.exe",
    "https://trusted-site.com",
    "http://phishing-site.com"
]

for url in test_urls:
    is_safe = safety.is_safe(url)
    status = "✓ SAFE" if is_safe else "✗ BLOCKED"
    print(f"  {status}: {url}")

# Add custom blocked domain
print("\n✓ Adding custom blocked domain...")
safety.add_blocked_domain('dangerous-site.com')
print(f"  Total blocked domains: {safety.get_statistics()['total_blocked_domains']}")

# ============================================================================
# 6. THEME SYSTEM
# ============================================================================
print("\n" + "=" * 80)
print("🎨 6. THEME SYSTEM")
print("-" * 80)

from enigma.gui.theme_system import ThemeManager, ThemeColors

manager = ThemeManager()

print("\n✓ Available themes:")
for name, description in manager.list_themes().items():
    print(f"  • {name}: {description}")

# Switch themes
print("\n✓ Switching to 'midnight' theme...")
manager.set_theme('midnight')
print(f"  Current theme: {manager.current_theme.name}")

# Create custom theme
print("\n✓ Creating custom theme...")
custom_colors = ThemeColors(
    bg_primary='#2a0a2a',
    text_primary='#e0b0ff',
    accent_primary='#ff00ff'
)
custom_theme = manager.create_custom_theme(
    'my_purple_theme',
    custom_colors,
    'Custom purple theme'
)
print(f"  Created: {custom_theme.name}")

# Get stylesheet
stylesheet = manager.get_current_stylesheet()
print(f"  Generated stylesheet length: {len(stylesheet)} characters")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ DEMO COMPLETED SUCCESSFULLY")
print("=" * 80)
print()
print("All new features demonstrated:")
print("  1. ✓ Enhanced memory system with vector DB and categorization")
print("  2. ✓ Dynamic personality with user overrides")
print("  3. ✓ Context-aware conversations")
print("  4. ✓ Bias detection and ethics tools")
print("  5. ✓ Enhanced web safety with dynamic blocklists")
print("  6. ✓ Advanced theme system")
print()
print("Ready for integration into Enigma AI Engine!")
print("=" * 80)
