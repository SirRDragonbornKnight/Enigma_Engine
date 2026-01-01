# Enigma AI Engine - Comprehensive Enhancements Documentation

## Overview

This document describes the comprehensive enhancements added to Enigma AI Engine, making it more robust, ethical, and universally accessible.

---

## 🧠 Enhanced Memory System

### Vector Databases
Support for advanced vector databases for semantic memory search:

- **SimpleVectorDB**: Built-in, no dependencies, good for small-scale use
- **FAISSVectorDB**: Production-ready, fast, supports millions of vectors
- **PineconeVectorDB**: Cloud-based, managed, infinitely scalable

```python
from enigma.memory.vector_db import create_vector_db

# Create vector database
db = create_vector_db(dim=128, backend='simple')  # or 'faiss', 'pinecone'

# Add vectors
import numpy as np
vectors = np.array([[1.0, 0.0, ...], [0.0, 1.0, ...]])
db.add(vectors, ids=['mem1', 'mem2'], metadata=[...])

# Search
results = db.search(query_vector, top_k=5)
```

### Memory Categorization

Organize memories by type with automatic TTL-based pruning:

```python
from enigma.memory.categorization import MemoryCategorization, MemoryType

mem_system = MemoryCategorization()

# Add memories with categories
mem_system.add_memory(
    "User's name is Alice",
    memory_type=MemoryType.LONG_TERM,  # Permanent
    importance=1.0
)

mem_system.add_memory(
    "Current conversation topic",
    memory_type=MemoryType.SHORT_TERM,  # Expires in 1 day
    ttl=86400
)

mem_system.add_memory(
    "Just asked about Python",
    memory_type=MemoryType.WORKING,  # Expires in 1 hour
    ttl=3600
)

# Auto-prune expired memories
mem_system.prune_all()

# Promote important memories
mem_system.promote_to_long_term(memory_id)
```

### Memory Export/Import

Export and import memories across sessions:

```python
from enigma.memory.export_import import MemoryExporter, MemoryImporter

# Export to JSON
exporter = MemoryExporter(mem_system)
exporter.export_to_json(Path('memories.json'))

# Export to CSV
exporter.export_to_csv(Path('memories.csv'))

# Export complete archive
exporter.export_to_archive(Path('memories.zip'), include_vectors=True)

# Import
importer = MemoryImporter(mem_system)
importer.import_from_json(Path('memories.json'), merge=True)
```

---

## 🎭 Dynamic Personality System

User-tunable personality traits with preset personalities and auto-evolution:

```python
from enigma.core.personality import AIPersonality

personality = AIPersonality("my_model")

# Set user overrides (takes precedence over evolved traits)
personality.set_user_override('humor_level', 0.9)
personality.set_user_override('formality', 0.2)

# Apply presets
personality.set_preset('comedian')  # professional, friendly, creative, analytical, teacher, coach

# Get effective traits
traits = personality.get_all_effective_traits()

# Generate personality-aware system prompt
system_prompt = personality.get_personality_prompt()

# Allow/disallow evolution
personality.allow_evolution = False  # Freeze personality

# Evolution happens automatically during conversations
personality.evolve_from_interaction(
    user_input="Tell me a joke!",
    ai_response="Why did the chicken cross the road? ...",
    feedback="positive"
)
```

### Available Personality Presets

- **professional**: Formal, efficient, analytical
- **friendly**: Warm, casual, supportive
- **creative**: Imaginative, playful, artistic
- **analytical**: Logical, precise, data-driven
- **teacher**: Patient, thorough, educational
- **comedian**: Funny, entertaining, lighthearted
- **coach**: Motivational, energetic, supportive

---

## 🗣️ Context-Aware Conversations

Improved multi-turn conversation tracking with clarification fallbacks:

```python
from enigma.core.context_awareness import ContextAwareConversation

conversation = ContextAwareConversation()

# Process user input
result = conversation.process_user_input("What about it?")

if result['needs_clarification']:
    print(result['clarification_prompt'])  # "Can you be more specific?"

# Get formatted context for AI prompt
context = conversation.get_context_for_prompt()

# Add assistant response
conversation.add_assistant_response("I'll help explain...")

# Get context summary
summary = result['context_summary']
```

Features:
- Entity extraction from conversations
- Topic tracking
- Unclear query detection
- Automatic clarification suggestions
- Configurable context window size

---

## 🛡️ Ethics and Safety Tools

### Bias Detection

Scan datasets and outputs for biased patterns:

```python
from enigma.tools.bias_detection import BiasDetector

detector = BiasDetector(config={'sensitivity': 0.5})

# Scan single text
result = detector.scan_text("The engineer worked on his code")

print(f"Bias score: {result.bias_score}")
print(f"Issues: {result.issues_found}")
print(f"Recommendations: {result.recommendations}")

# Scan entire dataset
dataset_result = detector.scan_dataset(texts)
print(f"Biased samples: {dataset_result.statistics['biased_samples']}")
```

### Offensive Content Filtering

```python
from enigma.tools.bias_detection import OffensiveContentFilter

content_filter = OffensiveContentFilter()

# Scan text
result = content_filter.scan_text("Some text")
if result['is_offensive']:
    print(f"Found terms: {result['found_terms']}")

# Filter text
filtered = content_filter.filter_text("Some text", replacement="[FILTERED]")
```

### Safe Reinforcement Logic

Ensure AI outputs are safe and ethical:

```python
from enigma.tools.bias_detection import SafeReinforcementLogic

logic = SafeReinforcementLogic()

# Check output safety before returning to user
result = logic.check_output_safety(ai_output)

if not result['is_safe']:
    print(f"Issues: {result['issues']}")
    if result['should_regenerate']:
        # Regenerate response

# Add safety guidelines to system prompt
safety_prompt = logic.get_safety_prompt_additions()
```

---

## 🌐 Enhanced Web Safety

Dynamic blocklist management with auto-updates:

```python
from enigma.tools.url_safety import URLSafety

safety = URLSafety(
    enable_auto_update=True,
    update_interval_hours=24
)

# Check URL safety
if safety.is_safe("https://example.com"):
    # Proceed with request
    pass

# Manually add blocked domain
safety.add_blocked_domain('dangerous-site.com')

# Import blocklist from file
safety.import_blocklist_from_file(Path('blocklist.txt'))

# Update from online sources
stats = safety.update_blocklist_from_sources()
print(f"Added {stats['added_count']} new domains")

# Get statistics
stats = safety.get_statistics()
print(f"Total blocked: {stats['total_blocked_domains']}")
```

### Content Filtering

Remove ads, popups, and trackers:

```python
from enigma.tools.url_safety import ContentFilter

filter = ContentFilter()

# Check for ad content
if filter.is_ad_content(text):
    # Skip or filter

# Filter content
clean_text = filter.filter_content(text_with_ads)

# Extract main content from HTML
main_content = filter.extract_main_content(html)
```

---

## 🎨 Advanced Theme System

Multiple theme presets and custom theme support:

```python
from enigma.gui.theme_system import ThemeManager, ThemeColors

manager = ThemeManager()

# List available themes
themes = manager.list_themes()
# dark, light, high_contrast, midnight, forest, sunset

# Switch theme
manager.set_theme('midnight')

# Get stylesheet for Qt application
stylesheet = manager.get_current_stylesheet()
app.setStyleSheet(stylesheet)

# Create custom theme
custom_colors = ThemeColors(
    bg_primary='#2a0a2a',
    text_primary='#e0b0ff',
    accent_primary='#ff00ff'
)
manager.create_custom_theme('my_theme', custom_colors, 'My custom theme')
```

### Available Theme Presets

1. **Dark (Catppuccin Mocha)**: Soft, easy on the eyes (default)
2. **Light**: Bright, for well-lit environments
3. **High Contrast**: Maximum readability, accessibility-focused
4. **Midnight**: Deep blue, professional
5. **Forest**: Nature-inspired green
6. **Sunset**: Warm, cozy colors

---

## 📦 Installation

### Basic Installation

```bash
pip install -r requirements.txt
```

### Optional Dependencies

For advanced features, install optional dependencies:

```bash
# For FAISS vector database
pip install faiss-cpu  # or faiss-gpu for GPU support

# For Pinecone vector database
pip install pinecone-client

# For development/testing
pip install -r requirements-dev.txt
```

---

## 🚀 Quick Start

```python
# 1. Enhanced Memory
from enigma.memory import MemoryCategorization, MemoryType
mem = MemoryCategorization()
mem.add_memory("Important fact", MemoryType.LONG_TERM)

# 2. Dynamic Personality
from enigma.core.personality import AIPersonality
personality = AIPersonality("my_ai")
personality.set_preset('friendly')

# 3. Context Awareness
from enigma.core.context_awareness import ContextAwareConversation
conv = ContextAwareConversation()
result = conv.process_user_input("Hello!")

# 4. Bias Detection
from enigma.tools.bias_detection import BiasDetector
detector = BiasDetector()
result = detector.scan_text("Some text")

# 5. Web Safety
from enigma.tools.url_safety import URLSafety
safety = URLSafety()
if safety.is_safe(url):
    # Safe to visit
    pass

# 6. Themes
from enigma.gui.theme_system import ThemeManager
themes = ThemeManager()
themes.set_theme('midnight')
```

---

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_enhanced_memory.py -v
python -m pytest tests/test_personality_enhancements.py -v
python -m pytest tests/test_context_and_ethics.py -v
python -m pytest tests/test_web_safety_and_themes.py -v
```

Run demos:

```bash
# Lightweight demo (no PyTorch required)
python demo_enhancements_lite.py

# Full demo (requires PyTorch)
python demo_comprehensive_enhancements.py
```

---

## 📖 API Reference

See individual module documentation:
- `enigma/memory/` - Memory system APIs
- `enigma/core/personality.py` - Personality system
- `enigma/core/context_awareness.py` - Context tracking
- `enigma/tools/bias_detection.py` - Ethics tools
- `enigma/tools/url_safety.py` - Web safety
- `enigma/gui/theme_system.py` - Theme management

---

## 🤝 Contributing

When contributing enhancements:

1. Follow the modular architecture
2. Add comprehensive tests
3. Update documentation
4. Run bias/ethics checks on any data
5. Ensure backward compatibility

---

## 📄 License

Same license as Enigma AI Engine.

---

## 🙏 Credits

Enhanced by the Enigma AI Engine community to make AI more ethical, robust, and accessible.
