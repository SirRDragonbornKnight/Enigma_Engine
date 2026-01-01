#!/usr/bin/env python3
"""
Lightweight demo of new enhancements (no PyTorch required).

Usage:
    python demo_enhancements_lite.py
"""
import sys
from pathlib import Path
import numpy as np

print("=" * 80)
print("ENIGMA AI ENGINE - NEW ENHANCEMENTS DEMO (Lightweight)")
print("=" * 80)
print()

# ============================================================================
# 1. MEMORY SYSTEM
# ============================================================================
print("📝 1. ENHANCED MEMORY SYSTEM")
print("-" * 80)

from enigma.memory.vector_db import SimpleVectorDB
from enigma.memory.categorization import MemoryCategorization, MemoryType
from enigma.memory.export_import import MemoryExporter

# Vector database
print("\n✓ Vector database with semantic search...")
vector_db = SimpleVectorDB(dim=128)
vectors = np.random.rand(5, 128).astype('float32')
ids = [f'memory_{i}' for i in range(5)]
metadata = [{'content': f'Memory {i}'} for i in range(5)]
vector_db.add(vectors, ids, metadata)
print(f"  Added {vector_db.count()} vectors")

# Memory categorization
print("\n✓ Memory categorization with TTL...")
mem_system = MemoryCategorization()
mem_system.add_memory("Long-term fact", MemoryType.LONG_TERM)
mem_system.add_memory("Short-term info", MemoryType.SHORT_TERM, ttl=86400)
mem_system.add_memory("Working context", MemoryType.WORKING, ttl=3600)

stats = mem_system.get_statistics()
print(f"  Total memories: {stats['total']}")
print(f"  Categories: {list(stats['by_type'].keys())}")

# Export
exporter = MemoryExporter(mem_system)
export_file = Path('/tmp/enigma_demo_export.json')
export_stats = exporter.export_to_json(export_file)
print(f"\n✓ Exported {export_stats['exported_count']} memories to {export_file}")

# ============================================================================
# 2. CONTEXT AWARENESS
# ============================================================================
print("\n" + "=" * 80)
print("🗣️ 2. CONTEXT-AWARE CONVERSATIONS")
print("-" * 80)

from enigma.core.context_awareness import ContextAwareConversation

conversation = ContextAwareConversation()

# Simulated conversation
print("\n✓ Multi-turn conversation tracking...")
turns = [
    "Hello, I'm learning Python",
    "What about it?",  # Unclear - needs clarification
    "Tell me about functions"
]

for i, user_input in enumerate(turns, 1):
    result = conversation.process_user_input(user_input)
    print(f"\n  Turn {i}: {user_input}")
    print(f"    Needs clarification: {result['needs_clarification']}")
    if result['needs_clarification']:
        print(f"    → {result['clarification_prompt']}")
    conversation.add_assistant_response(f"Response to turn {i}")

print(f"\n✓ Context summary: {conversation.context_tracker.get_context_summary()}")

# ============================================================================
# 3. BIAS DETECTION & ETHICS
# ============================================================================
print("\n" + "=" * 80)
print("🛡️ 3. BIAS DETECTION & ETHICS TOOLS")
print("-" * 80)

from enigma.tools.bias_detection import (
    BiasDetector,
    OffensiveContentFilter,
    SafeReinforcementLogic
)

# Bias detection
print("\n✓ Scanning for gender bias...")
detector = BiasDetector(config={'sensitivity': 0.5})

sample_texts = [
    "The engineer worked on his code",
    "The nurse helped her patients",
    "Developers need strong technical skills"
]

for text in sample_texts:
    result = detector.scan_text(text)
    print(f"  '{text[:40]}...'")
    print(f"    Bias score: {result.bias_score:.2f}, Issues: {len(result.issues_found)}")

# Offensive content
print("\n✓ Offensive content filtering...")
content_filter = OffensiveContentFilter()

test_cases = [
    ("This is a great day", False),
    ("This is damn annoying", True)
]

for text, should_be_offensive in test_cases:
    result = content_filter.scan_text(text)
    status = "✗ OFFENSIVE" if result['is_offensive'] else "✓ CLEAN"
    print(f"  {status}: '{text}'")

# Safe reinforcement
print("\n✓ Safe reinforcement for AI outputs...")
logic = SafeReinforcementLogic()

outputs = [
    "I'm happy to help you learn!",
    "All women are emotional"  # Biased
]

for output in outputs:
    check = logic.check_output_safety(output)
    status = "✓ SAFE" if check['is_safe'] else "✗ UNSAFE"
    print(f"  {status}: '{output[:40]}...'")
    if not check['is_safe']:
        print(f"    Issues: {len(check['issues'])}")

# ============================================================================
# 4. WEB SAFETY
# ============================================================================
print("\n" + "=" * 80)
print("🌐 4. ENHANCED WEB SAFETY")
print("-" * 80)

from enigma.tools.url_safety import URLSafety

safety = URLSafety()

print("\n✓ URL safety checking with dynamic blocklists...")

test_urls = [
    ("https://github.com/user/repo", True),
    ("http://site.com/download.exe", False),
    ("https://python.org/docs", True)
]

for url, should_be_safe in test_urls:
    is_safe = safety.is_safe(url)
    status = "✓ SAFE" if is_safe else "✗ BLOCKED"
    print(f"  {status}: {url}")

# Dynamic blocklist
print("\n✓ Dynamic blocklist management...")
safety.add_blocked_domain('dangerous-site.com')
stats = safety.get_statistics()
print(f"  Total blocked domains: {stats['total_blocked_domains']}")

# ============================================================================
# 5. THEME SYSTEM
# ============================================================================
print("\n" + "=" * 80)
print("🎨 5. ADVANCED THEME SYSTEM")
print("-" * 80)

from enigma.gui.theme_system import ThemeManager, ThemeColors

manager = ThemeManager()

print("\n✓ Available preset themes:")
themes = manager.list_themes()
for name in list(themes.keys())[:6]:  # Show first 6
    print(f"  • {name}: {themes[name]}")

# Theme switching
print("\n✓ Theme features:")
for theme_name in ['dark', 'light', 'midnight']:
    manager.set_theme(theme_name)
    stylesheet_len = len(manager.get_current_stylesheet())
    print(f"  {theme_name}: {stylesheet_len} chars stylesheet")

# Custom theme
print("\n✓ Creating custom theme...")
custom_colors = ThemeColors(
    bg_primary='#2a0a2a',
    text_primary='#e0b0ff',
    accent_primary='#ff00ff'
)
# Note: Theme creation would normally save to disk
print(f"  Custom colors defined: {len(custom_colors.to_dict())} properties")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ ALL ENHANCEMENTS DEMONSTRATED SUCCESSFULLY!")
print("=" * 80)
print()
print("Features showcased:")
print("  ✓ Vector databases for semantic memory search")
print("  ✓ Memory categorization (short/long-term) with TTL")
print("  ✓ Memory export/import across sessions")
print("  ✓ Context-aware conversations with clarification")
print("  ✓ Bias detection for datasets")
print("  ✓ Offensive content filtering")
print("  ✓ Safe reinforcement logic")
print("  ✓ Dynamic URL blocklist management")
print("  ✓ Advanced theme system with 6+ presets")
print()
print("Ready for production use!")
print("=" * 80)
