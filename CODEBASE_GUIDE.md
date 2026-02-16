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

# Phase 5
gui/ - 14 dead files (accessibility, chat_features, notification_system, etc.)
gui/tabs/avatar/ - 2 dead files (avatar_management, widgets)
config/ - 2 dead files (migration, validation)

# Phase 6 (final)
modules/ - 1 dead file (error_messages)
core/nn/ - 4 dead files (activations, attention, embeddings, normalization)
```

---

## Current Package Structure (489 files)

| Package | Files | Purpose |
|---------|-------|---------|
| core/ | 135 | AI model, inference, training |
| gui/ | 95 | PyQt5 interface (28 tabs) |
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
| modules/ | 6 | Module manager |
| self_improvement/ | 7 | Self-training |
| game/ | 6 | Game overlay |
| network/ | 6 | Network offloading |
| security/ | 3 | Auth/security |
| plugins/ | 5 | Plugin system |
| cli/ | 4 | Command line |
| config/ | 2 | Configuration |
| marketplace/ | 4 | Plugin marketplace |
| auth/ | 2 | Authentication |
| companion/ | 2 | Companion mode |
| i18n/ | 2 | Translations |
| mobile/ | 2 | Mobile API |

---

## Architecture Overview

```
enigma_engine/
├── core/           # AI model, inference, training (135 files)
├── gui/            # PyQt5 interface
│   └── tabs/       # 28 tab files (*_tab.py)
├── modules/        # Module manager system
├── tools/          # AI tools (44 files)
├── memory/         # Conversation storage, vector DB
├── voice/          # TTS/STT
├── avatar/         # Avatar control
├── comms/          # API server, networking
├── learning/       # Learning utilities
├── config/         # Global CONFIG
└── utils/          # Helpers (23 files)
```

---

## Package Details - How Each File Works

### core/ - AI Engine (135 files)

| File | How It Works |
|------|-------------|
| `model.py` | The neural network brain. Stacks transformer layers: input → embedding → N attention blocks → output probabilities. Each layer does: RMSNorm → self-attention (queries/keys/values) → feedforward (SwiGLU activation). Uses RoPE for position encoding so model knows word order. KV-cache stores past computations to avoid recalculating during generation. |
| `inference.py` | The voice - converts your text to AI response. Loop: tokenize input → run through model → get probability distribution for next token → sample one (greedy/top-k/top-p) → add to sequence → repeat until EOS or max length. Supports streaming (yield each token as generated). |
| `training.py` | Teaches the model. Forward pass: run input through model, compare output to target with cross-entropy loss. Backward pass: compute gradients, update weights via AdamW optimizer. Handles batching, gradient accumulation, learning rate scheduling (warmup + cosine decay). |
| `tokenizer.py` | Converts text↔numbers. BPE (byte-pair encoding): learns common subword patterns, encodes unknown words as pieces. "playing" might become ["play", "ing"]. Vocabulary typically 32K-100K tokens. Special tokens: [PAD], [EOS], [BOS]. |
| `tool_router.py` | The dispatcher that reads user intent. Scans input for keywords ("draw" → image, "explain" → chat). Each tool has priority-ordered model assignments. Routes request to tool_executor with parameters extracted from user text. |
| `kv_cache.py` | Speed optimization. Stores key/value tensors from previous tokens so attention doesn't recalculate them. On token N, only computes attention for new token vs all previous. Turns O(n²) into O(n). |
| `quantization.py` | Shrinks models to use less memory. Converts float32 weights to int8/int4. AWQ: finds important weights, keeps them higher precision. GPTQ: layer-by-layer quantization with calibration data. |
| `streaming.py` | Yields tokens as they're generated instead of waiting for full response. Uses Python generators. GUI hooks into this for live typing effect. |
| `lora_training.py` | Efficient fine-tuning. Instead of updating all weights, adds small trainable matrices (A×B) to attention layers. Original weights frozen, only LoRA adapters trained. Saves 90%+ memory vs full fine-tuning. |
| `huggingface_loader.py` | Loads external models. Downloads from HuggingFace Hub, converts weight names to Enigma format, handles different architectures (LLaMA, Mistral, GPT-2). Auto-detects model type from config.json. |
| `gguf_loader.py` | Loads llama.cpp quantized models. Parses GGUF binary format, reads metadata (vocab, architecture), dequantizes weights on-the-fly or keeps quantized for inference. |
| `moe.py` | Mixture of Experts. Routes each token to 2 of N expert networks (instead of one big feedforward). Gating network decides which experts. Increases capacity without proportional compute. |
| `paged_attention.py` | Memory-efficient attention for long sequences. Splits KV cache into pages, only allocates pages as needed. Allows variable-length sequences without wasting memory on padding. |

### tools/ - AI Capabilities (44 files)

| File | How It Works |
|------|-------------|
| `tool_executor.py` | Safe execution sandbox. Parses `<tool_call>{JSON}</tool_call>` from AI output. Validates parameters against tool schema. Runs with timeout (Unix: SIGALRM, Windows: threading.Timer). Blocks dangerous paths via security.py. Returns structured result or error. |
| `tool_definitions.py` | Schema registry. Each tool defined with: name, description, parameters (name, type, required, description), handler function. Used by executor for validation and by AI for knowing what tools exist. |
| `vision.py` | Screenshot + image analysis. Uses PIL for capture, sends to AI vision model (or API like GPT-4V) for description. OCR via simple_ocr.py: binarize → find text regions → tesseract. |
| `web_tools.py` | Web access. `web_search`: queries DuckDuckGo/Google API, parses results. `fetch_url`: requests + BeautifulSoup to extract text content. Rate limiting prevents abuse. |
| `file_tools.py` | File operations with security. read_file/write_file/list_dir with path blocking (can't access ~/.ssh, system dirs). Operations go through is_path_blocked() check first. |
| `document_ingestion.py` | Extracts text from files. PDF: PyPDF2 or pdfplumber. DOCX: python-docx. EPUB: ebooklib. Chunks large documents into overlapping segments for vector search. |
| `game_router.py` | Game-specific AI routing. Detects running game via window title. Loads game config (keybinds, state patterns). Routes inputs through game-specific prompt templates. |
| `game_detector.py` | Identifies running games. Scans window titles against known game patterns. Checks process names. Returns game ID and config path. |
| `robot_tools.py` | Hardware control abstraction. Sends commands to robot backends (pi_robot.py for Raspberry Pi GPIO, serial for Arduino). Actions: move, gripper_open/close, home. |

### modules/ - Module System (6 files)

| File | How It Works |
|------|-------------|
| `manager.py` | Lifecycle controller. State machine: UNLOADED → LOADING → LOADED → ACTIVE. Checks dependencies before loading (inference needs model+tokenizer). Prevents conflicts (can't load image_gen_local AND image_gen_api). Tracks memory usage per module. |
| `registry.py` | Module catalog. Each module class defines: init (setup), load (allocate resources), unload (cleanup), dependencies, conflicts. 50+ modules registered covering all capabilities. |
| `sandbox.py` | Isolated execution. Runs module code with restricted imports, timeout, memory limits. Catches crashes without taking down main app. |

### utils/ - Helpers (23 files)

| File | How It Works |
|------|-------------|
| `lazy_import.py` | Deferred loading. `LazyLoader` class replaces heavy imports (torch, transformers) with proxies. Actual import happens on first attribute access. Speeds up startup 5-10x. |
| `security.py` | Path blocking. `is_path_blocked()` checks path against blocklist (~/.ssh, /etc/passwd, C:\\Windows). Called by file_tools before any operation. Config in blocked_paths.json. |
| `api_key_encryption.py` | Secure key storage. Encrypts API keys with Fernet (AES-128). Master key derived from machine ID + user password. Keys stored encrypted in ~/.enigma_engine/keys. |

### memory/ - Conversation Storage (19 files)

| File | How It Works |
|------|-------------|
| `manager.py` | Conversation history. Stores messages as list of {role, content, timestamp}. Auto-truncates to fit context window. Saves/loads conversations as JSON. |
| `vector_db.py` | Semantic search. Embeds text chunks via sentence-transformers. Stores vectors in FAISS index. Query: embed search text → find k nearest neighbors → return matching chunks. |

### voice/ - TTS/STT (29 files)

| File | How It Works |
|------|-------------|
| `voice_generator.py` | TTS synthesis. Backend options: pyttsx3 (local), ElevenLabs API, built-in. Queues text chunks, plays audio via sounddevice. Voice profile sets pitch/rate/voice ID. |
| `listener.py` | Speech recognition. Captures mic audio via sounddevice. Sends to Whisper (local) or cloud STT. Returns transcribed text. Wake word detection optional. |
| `voice_profile.py` | Voice identity storage. Dataclass with: voice_id, pitch, rate, volume, custom settings. Save/load as JSON. Used by TTS and avatar. |

### comms/ - Networking (17 files)

| File | How It Works |
|------|-------------|
| `api_server.py` | REST API. Flask app with routes: /generate (text), /chat, /tools/{name}. Auth via API key header. CORS enabled for web clients. Runs in background thread. |
| `discovery.py` | Network device finder. UDP broadcast on local network. Other Enigma instances respond with capabilities. Used for distributed inference. |
| `network.py` | Multi-device coordination. ForgeNode class represents a peer. Message passing via TCP. ModelExporter sends model shards to peers for parallel inference. |

---

## Data Flow - How a Chat Message is Processed

```
USER TYPES: "Draw me a sunset over mountains"
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  chat_tab.py                                                        │
│  - Captures text from input box                                     │
│  - Adds to conversation history                                     │
│  - Calls EnigmaEngine.chat() or .generate()                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  inference.py (EnigmaEngine)                                        │
│  - Tokenizes input text → [15496, 502, 257, 24536, ...]            │
│  - If use_routing=True, sends to tool_router first                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  tool_router.py                                                     │
│  - Scans for keywords: "draw" detected                              │
│  - Matches to "image" tool                                          │
│  - Extracts parameters: prompt="sunset over mountains"              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  tool_executor.py                                                   │
│  - Validates parameters against tool schema                         │
│  - Checks security (no blocked paths)                               │
│  - Calls image generation handler with timeout                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  image_tab.py (generator)                                           │
│  - Runs Stable Diffusion locally OR calls DALL-E API                │
│  - Saves image to outputs/images/                                   │
│  - Returns path to generated image                                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
RESULT: Image of sunset displayed in chat + saved to disk
```

---

## GUI Tabs (28 total) - How They Work

### Core Interface Tabs

| Tab | How It Works |
|-----|-------------|
| `chat_tab.py` | Main conversation UI. QTextEdit for input, QScrollArea for messages. On send: adds user message to display → calls EnigmaEngine.generate() in QThread → streams response tokens via signal/slot → appends AI response. Manages conversation history via memory/manager.py. |
| `settings_tab.py` | Config editor. Loads forge_config.json on init. QFormLayout with input fields for each setting. API keys encrypted via api_key_encryption.py before saving. Emits configChanged signal so other tabs can react. |
| `training_tab.py` | Training UI. File picker for dataset → validates format → creates Trainer instance → runs training in QThread with progress callback → updates progress bar → saves model checkpoint. |

### Generation Tabs (all inherit BaseGenerationTab)

| Tab | How It Works |
|-----|-------------|
| `image_tab.py` | Provider pattern: StableDiffusionLocal (diffusers), OpenAIImage (DALL-E API), ReplicateImage. User picks provider, enters prompt → QThread runs generate() → emits signal with result path → displays in QLabel with resize handles. |
| `code_tab.py` | Same provider pattern. ForgeCode uses local model with code-specific prompt template. OpenAICode calls GPT-4 API. Output displayed in QTextEdit with syntax highlighting (QSyntaxHighlighter). |
| `video_tab.py` | LocalVideo wraps AnimateDiff pipeline. Chains multiple image generations, interpolates frames. ReplicateVideo calls external API. Saves as MP4 or GIF. |
| `audio_tab.py` | LocalTTS wraps pyttsx3 engine. ElevenLabsTTS calls their API with voice_id. Plays audio via QMediaPlayer or sounddevice. |
| `threed_tab.py` | Local3DGen wraps Shap-E model. Generates point cloud → converts to mesh → exports OBJ. Preview via QOpenGLWidget (if available) or matplotlib fallback. |

### Model Management Tabs

| Tab | How It Works |
|-----|-------------|
| `modules_tab.py` | Displays all modules from registry.py as toggle switches. On toggle: calls ModuleManager.load()/unload() in QThread. Shows dependencies (grayed if missing prereqs). Filters by category dropdown. |
| `model_router_tab.py` | Visual tool→model assignment. Dropdowns for each tool category. Saves to tool_routing.json. Changes take effect immediately via ToolRouter.reload_config(). |
| `training_data_tab.py` | AI-assisted dataset generation. User enters topic → calls Claude/GPT-4 API to generate Q&A pairs → formats as training data → appends to data file. |

### Utility Tabs

| Tab | How It Works |
|-----|-------------|
| `network_tab.py` | Scans local network via discovery.py UDP broadcast. Displays found devices in QListWidget. "Connect" button establishes TCP connection for distributed inference. |
| `camera_tab.py` | OpenCV camera feed in QLabel. QThread captures frames at 30fps, emits as QPixmap. "Capture" saves current frame. "Analyze" sends to vision model. |
| `analytics_tab.py` | Reads JSON logs from ~/.enigma_engine/analytics/. Aggregates by day/week. Displays charts via pyqtgraph or matplotlib. |

### Tab Creation Pattern

All tabs follow one of two patterns:

```python
# Pattern 1: Factory function (older style)
def create_my_tab(parent):
    tab = QWidget()
    layout = QVBoxLayout(tab)
    # ... setup widgets ...
    return tab

# Pattern 2: Class-based (newer style)
class MyTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        # ... setup widgets ...
```

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
| **Total Python files** | **489** |
| GUI tabs | 28 |
| Core modules | 135 |
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
