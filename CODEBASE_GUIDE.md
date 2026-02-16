# Enigma AI Engine - Codebase Guide

**Version:** 3.0 | **Last Updated:** February 15, 2026

This file helps AI assistants quickly understand the codebase before making changes.

---

## Quick Reference

### DO NOT DELETE These Files
These files appear unused but ARE imported somewhere:
```
core/meta_learning.py      → Used by trainer_ai.py
core/prompt_builder.py     → Used by game_router.py, tests
core/moe.py                → Used by test_moe.py
core/paged_attention.py    → Used by continuous_batching.py
utils/battery_manager.py   → Used by __init__.py, integration.py
utils/api_key_encryption.py → Used by build_ai_tab.py, trainer_ai.py
utils/starter_kits.py      → Used by quick_create.py
```

### Removed Packages (Feb 15, 2026)
Don't look for these - entire packages deleted:
```
federated/, robotics/, docs/, monitoring/, hub/, deploy/, collab/
testing/, scripts/, training/, sync/, prompts/, data/, edge/, personality/, integrations/
```

### Removed Files (Feb 15, 2026)
```
# Phase 1 (core/)
core/gguf_export.py, core/gguf_exporter.py, core/moe_router.py, core/moe_routing.py
core/dpo_training.py, core/rlhf_training.py, core/speculative_decoding.py
core/curriculum_learning.py, core/kv_compression.py, core/kv_cache_compression.py
core/kv_cache_quantization.py, core/prompts.py, core/prompt_manager.py, core/prompt_templates.py

# Phase 2
tools/battery_manager.py, tools/home_assistant.py, tools/manipulation.py, tools/slam.py
tools/goal_tracker.py, tools/robot_platforms.py, tools/system_awareness.py
+ 15 voice files, 20 memory files, 20 avatar files, 15 comms files

# Phase 3
utils/ - 59 dead files (ab_testing, backup, circuit_breaker, encryption, hotkeys, etc.)
gui/tabs/ - 16 unused tabs (dashboard, personality, build_ai, modules, etc.)
core/ - 49 dead files (flash_attention, multi_gpu, dpo, rlhf, distillation, etc.)

# Phase 4
tools/ - 21 dead files (game_*, sensor_*, replay_analysis, etc.)
agents/ - 10 dead files (debate, swarm, tournament, visual_workspace, etc.)
web/ - 3 dead files (session_middleware, api_docs, training_dashboard)
security/ - 3 dead files (gdpr, pii_scrubber, tls)
learning/ - 5 dead files (ab_testing, critic_model, model_coordination, etc.)
gui/widgets/ - 4 dead files (image_paste, split_view, feedback_widget, quick_settings)
```

---

## Current Package Structure (512 files)

| Package | Files | Purpose |
|---------|-------|---------|
| core/ | 121 | AI model, inference, training |
| gui/ | 115 | PyQt5 interface (28 tabs) |
| tools/ | 44 | AI tool implementations |
| avatar/ | 42 | Avatar control system |
| voice/ | 29 | TTS/STT features |
| utils/ | 23 | Utilities and helpers |
| memory/ | 19 | Conversation/vector storage |
| comms/ | 17 | API server, networking |
| learning/ | 11 | Learning system |
| agents/ | 2 | Multi-agent system |
| builtin/ | 11 | Fallback generators |
| web/ | 6 | Web dashboard |
| modules/ | 7 | Module manager |
| self_improvement/ | 7 | Self-training |
| game/ | 6 | Game overlay |
| network/ | 6 | Network offloading |
| security/ | 3 | Auth/security |
| plugins/ | 5 | Plugin system |
| cli/ | 4 | Command line |
| config/ | 4 | Configuration |
| marketplace/ | 4 | Plugin marketplace |
| auth/ | 2 | Authentication |
| companion/ | 2 | Companion mode |
| i18n/ | 2 | Translations |
| mobile/ | 2 | Mobile API |

---

## Architecture Overview

```
enigma_engine/
├── core/           # AI model, inference, training (188 files)
├── gui/            # PyQt5 interface
│   └── tabs/       # 44 tab files (*_tab.py)
├── modules/        # Module manager system
├── tools/          # AI tools (65 files)
├── memory/         # Conversation storage, vector DB
├── voice/          # TTS/STT
├── avatar/         # Avatar control
├── comms/          # API server, networking
├── learning/       # Learning utilities
├── config/         # Global CONFIG
└── utils/          # Helpers (82 files)
```

---

## Package Details

### core/ - AI Engine (188 files)
| Category | Key Files | Purpose |
|----------|-----------|---------|
| **Model** | model.py, layers.py, nn/* | Transformer architecture |
| **Inference** | inference.py, streaming.py, kv_cache.py | Text generation |
| **Training** | training.py, trainer.py, lora_training.py | Model training |
| **Tokenizer** | tokenizer.py, bpe_tokenizer.py, char_tokenizer.py | Text processing |
| **Quantization** | quantization.py, awq_quantization.py, gptq_quantization.py | Model compression |
| **Routing** | tool_router.py, universal_router.py, orchestrator.py | Intent detection |
| **Context** | context_extender.py, infinite_context.py, paged_attention.py | Long context |
| **Advanced** | moe.py, ssm.py, flash_attention.py | Experimental architectures |
| **Loading** | huggingface_loader.py, gguf_loader.py, ollama_loader.py | External models |

### tools/ - AI Capabilities (65 files)
| Category | Key Files | Purpose |
|----------|-----------|---------|
| **Execution** | tool_executor.py, tool_registry.py, tool_manager.py | Tool system core |
| **Vision** | vision.py, simple_ocr.py | Image analysis, OCR |
| **Web** | web_tools.py, browser_tools.py, url_safety.py | Web access |
| **Files** | file_tools.py, document_tools.py, document_ingestion.py | File operations |
| **System** | system_tools.py | System control |
| **Gaming** | game_router.py, game_detector.py, game_state.py | Game AI |
| **Robot** | robot_tools.py, robot_modes.py, pi_robot.py | Robot control |

### utils/ - Helpers (82 files)
| Category | Key Files | Purpose |
|----------|-----------|---------|
| **Security** | security.py, encryption.py, api_key_encryption.py | Security utilities |
| **Performance** | lazy_import.py, performance.py, memory_profiler.py | Optimization |
| **Storage** | storage.py, backup.py, json_cache.py | Data persistence |
| **Networking** | network_fallback.py, circuit_breaker.py, bulkhead.py | Resilience |

---

## Core Files (CRITICAL - Be Careful)

| File | Purpose | Lines |
|------|---------|-------|
| `core/model.py` | Transformer model (Enigma class) | 3,214 |
| `core/inference.py` | Text generation (EnigmaEngine) | 2,167 |
| `core/training.py` | Model training (Trainer) | 1,974 |
| `core/tokenizer.py` | Text ↔ tokens | 826 |
| `core/tool_router.py` | Intent detection & routing | 3,374 |
| `modules/manager.py` | Module lifecycle | 1,874 |
| `modules/registry.py` | Module definitions | 3,095 |
| `tools/tool_executor.py` | Safe tool execution | 2,877 |

---

## GUI Tabs (44 total)

### Largest/Most Complex
| Tab | Lines | Purpose |
|-----|-------|---------|
| settings_tab.py | 4,721 | App settings, API keys |
| chat_tab.py | 2,869 | Main chat interface |
| build_ai_tab.py | 2,498 | AI creation wizard |
| image_tab.py | 1,449 | Image generation |
| training_tab.py | 1,424 | Model training |
| modules_tab.py | 1,383 | Module toggles |

### Tab Naming Pattern
All tabs follow: `{name}_tab.py` with `create_{name}_tab()` or `{Name}Tab` class.

---

## Import Patterns

### Lazy Loading (core/__init__.py)
Heavy imports are lazy-loaded. Don't import torch at module level in __init__.py:
```python
from ..utils.lazy_import import LazyLoader
_loader = LazyLoader(__name__)
_loader.register('EnigmaEngine', '.inference', 'EnigmaEngine')
```

### Relative Imports
Within enigma_engine, use relative imports:
```python
from ..config import CONFIG
from .model import create_model
from ...utils.security import is_path_blocked
```

### Common Imports
```python
from enigma_engine.config import CONFIG
from enigma_engine.core.inference import EnigmaEngine
from enigma_engine.core.model import create_model, ForgeConfig
from enigma_engine.modules import ModuleManager
from enigma_engine.tools.tool_executor import ToolExecutor
```

---

## Testing Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_model.py -v

# Quick import check
python -c "from enigma_engine.core import EnigmaEngine; print('OK')"

# Check for syntax errors
python -m py_compile enigma_engine/core/model.py
```

---

## Before Making Changes

### 1. Check if file is used
```bash
# Search for imports of a file
grep -r "from .filename import" enigma_engine/
grep -r "from enigma_engine.package.filename" enigma_engine/
```

### 2. Check for duplicates
Before creating new functionality, search:
```bash
grep -r "class MyClassName" enigma_engine/
grep -r "def my_function" enigma_engine/
```

### 3. Verify after changes
```bash
python -c "from enigma_engine.core import EnigmaEngine"
python -c "from enigma_engine.gui.enhanced_window import EnhancedMainWindow"
python -m pytest tests/test_model.py -v
```

---

## Configuration

### Main Config Object
```python
from enigma_engine.config import CONFIG

# Common settings
models_dir = CONFIG.get("models_dir", "models")
data_dir = CONFIG.get("data_dir", "data")
max_len = CONFIG.get("max_len", 1024)
```

### Model Presets
15 presets from tiny to massive:
```
pi_zero, pi_4, pi_5       # Raspberry Pi optimized
nano, micro, tiny         # Embedded/IoT
small, medium, large      # Desktop
xl, xxl, titan           # Server
huge, giant, omega       # Datacenter
```

---

## Future Features (Complete but Not Integrated)

These are ready to use when needed:
- `core/dpo.py` - Direct Preference Optimization training
- `core/rlhf.py` - RLHF training
- `core/ssm.py` - Mamba/S4 state space models
- `core/speculative.py` - Speculative decoding (faster inference)
- `core/tree_attention.py` - Tree-based attention
- `core/infinite_context.py` - Streaming unlimited context
- `tools/sensor_fusion.py` - Multi-sensor fusion
- `tools/achievement_tracker.py` - Game achievement tracking

---

## File Statistics

| Category | Count |
|----------|-------|
| **Total Python files** | **682** |
| GUI tabs | 44 |
| Core modules | 188 |
| Test files | 50+ |

---

## Quick Fixes for Common Issues

### Import Error
```python
# Wrong (absolute in same package)
from enigma_engine.core.model import Enigma

# Right (relative within package)
from .model import Enigma
```

### Circular Import
Use lazy imports or move import inside function:
```python
def my_function():
    from .other_module import SomeClass  # Import when needed
    return SomeClass()
```

### Missing Dependency
Check requirements.txt. Core dependencies:
- torch, numpy, PyQt5, psutil, requests

---

## Contacts & Resources

- **Main Entry:** `run.py`
- **GUI Entry:** `python run.py --gui`
- **API Server:** `python run.py --serve`
- **Training:** `python run.py --train`
- **Config File:** `forge_config.json`
- **Module Config:** `forge_modules.json`

---

*This guide should be updated when major structural changes are made.*
