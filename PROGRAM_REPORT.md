# Enigma AI Engine - Program Report

**Generated:** February 15, 2026

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [GUI Tabs Summary](#gui-tabs-summary)
4. [GUI Tabs Detailed](#gui-tabs-detailed)
5. [File Dependencies](#file-dependencies)

---

## Architecture Overview

Enigma AI Engine is a **fully modular AI framework** where everything is a toggleable module. The system supports running on hardware from Raspberry Pi to datacenter servers.

### Main Packages
```
enigma_engine/
├── core/          # AI model, inference, training, tokenizer
├── gui/           # PyQt5 interface with 28 tabs
├── modules/       # Module manager, registry, state handling
├── tools/         # Vision, web, file, document tools
├── memory/        # Conversation storage, vector DB
├── voice/         # TTS/STT wrappers
├── avatar/        # Avatar control and rendering
├── comms/         # API server, networking
├── config/        # Global configuration
└── utils/         # Helpers, security, lazy imports
```

### Key Technologies
- **Framework:** PyTorch for AI, PyQt5 for GUI
- **Model:** Custom transformer with RoPE, RMSNorm, SwiGLU, GQA, KV-cache
- **Sizes:** 15 presets from ~500K params (pi_zero) to 70B+ params (omega)

### Package List (558 Python files)

| Category | Packages | Files | Status |
|----------|----------|-------|--------|
| **Core** | core, gui/tabs, tools, utils, memory, voice, avatar, comms, modules, builtin, config, web, cli, i18n, game, plugins | ~450 | Active |
| **Features** | learning, self_improvement, marketplace, network | ~35 | Integrated |
| **Test/Internal** | security, agents, auth | ~20 | Tests only |
| **Optional** | companion, mobile | ~4 | Complete but minimal use |

---

## Core Components

| Component | File | Purpose |
|-----------|------|---------|
| Model | `core/model.py` | Transformer neural network |
| Inference | `core/inference.py` | Text generation engine |
| Training | `core/training.py` | Model training system |
| Tokenizer | `core/tokenizer.py` | Text ↔ numbers conversion |
| Tool Router | `core/tool_router.py` | Intent detection and routing |
| Module Manager | `modules/manager.py` | Module lifecycle management |
| Tool Executor | `tools/tool_executor.py` | Safe tool execution |

---

## GUI Tabs Summary

**Total Tabs:** 28 (16 unused tabs removed Feb 15, 2026)

### Quick Reference

| Tab | Lines | Purpose |
|-----|-------|---------|
| chat_tab | 2869 | Main AI conversation interface |
| settings_tab | 4721 | Application settings & API keys |
| image_tab | 1449 | Image generation |
| training_tab | 1424 | Model training interface |
| workspace_tab | 1001 | Training data preparation |
| bundle_manager_tab | 991 | Package models for sharing |
| training_data_tab | 987 | Generate training data |
| voice_clone_tab | 923 | Voice cloning |
| persona_tab | 761 | Prompt/persona management |
| audio_tab | 747 | Text-to-speech |
| threed_tab | 718 | 3D model generation |
| network_tab | 717 | Multi-device networking |
| scheduler_tab | 694 | Task scheduling |
| analytics_tab | 671 | Usage analytics |
| base_generation_tab | 655 | Base class for generation tabs |
| embeddings_tab | 633 | Vector embeddings |
| video_tab | 623 | Video generation |
| code_tab | 607 | Code generation |
| model_router_tab | 599 | Assign models to tasks |
| federation_tab | 526 | Federated learning |
| notes_tab | 523 | Notes and bookmarks |
| gif_tab | 485 | GIF generation |
| camera_tab | 468 | Webcam capture |
| instructions_tab | 364 | Help and documentation |
| vision_tab | 212 | Image/screen analysis |
| terminal_tab | 189 | Embedded terminal |
| sessions_tab | 178 | Session management |
| avatar_tab | 46 | Avatar display |
| vision_tab | 212 | Image/screen analysis |
| ai_tab | 193 | AI assistant panel |
| terminal_tab | 189 | Embedded terminal |
| create_tab | 179 | Quick creation shortcuts |
| sessions_tab | 178 | Session management |
| avatar_tab | 46 | Avatar display |

---

## GUI Tabs Detailed

### chat_tab.py (2869 lines)
**Purpose:** Main chat interface for conversing with the AI

**Main Components:**
- `create_chat_tab()` - Factory function

**Key Features:**
- Model selection dropdown
- Message display scroll area
- Conversation management (new, clear, save)
- Thinking/status indicator
- Tool calling support

**Connected Files:**
- `enigma_engine/core/inference.py` - AI responses
- `enigma_engine/memory/manager.py` - Conversation history

---

### settings_tab.py (4721 lines)
**Purpose:** Central application settings and API key management

**Main Components:**
- `create_settings_tab()` - Factory function

**Key Features:**
- API key management (OpenAI, Anthropic, HuggingFace, etc.)
- Feature toggles
- Theme/UI customization
- Robot mode control
- Game mode detection
- Device profile selection

**Connected Files:**
- `enigma_engine/config/` - CONFIG object
- `gui/shared_components.py` - UI components

---

### build_ai_tab.py (2498 lines)
**Purpose:** Step-by-step wizard for creating custom AI models

**Main Components:**
- `create_build_ai_tab()` - Factory function

**Key Features:**
- 7-step wizard workflow
- Manual vs AI Trainer modes
- Tool capabilities selection
- Router position definitions
- Training with progress tracking

**Connected Files:**
- `enigma_engine/config/` - Settings
- `enigma_engine/utils/api_key_encryption.py` - API keys

---

### image_tab.py (1449 lines)
**Purpose:** Generate images from text prompts

**Main Components:**
- `ImageTab` - Main tab class
- `ResizableImagePreview` - Preview widget

**Key Features:**
- Providers: Stable Diffusion (local), DALL-E 3, Replicate
- Resizable preview
- Auto-save to outputs/images/
- Provider-specific controls

**Connected Files:**
- `enigma_engine/config/` - Output directory

---

### training_tab.py (1424 lines)
**Purpose:** Train AI models with simplified workflow

**Main Components:**
- `create_training_tab()` - Factory function

**Key Features:**
- 3-step quick start workflow
- Text-to-QA conversion
- URL data import
- Hyperparameter controls

**Connected Files:**
- `enigma_engine/config/` - Training settings

---

### modules_tab.py (1383 lines)
**Purpose:** Toggle AI capabilities on/off

**Main Components:**
- `ModuleListItem` - Module toggle widget

**Key Features:**
- Module ON/OFF switches
- Category organization
- GPU/API requirements display
- Dependency tracking
- Search/filter

**Connected Files:**
- `enigma_engine/modules/manager.py` - Module control
- `enigma_engine/modules/registry.py` - Module definitions

---

### workspace_tab.py (1001 lines)
**Purpose:** Unified workspace for data preparation

**Main Components:**
- `create_workspace_tab()` - Factory function

**Key Features:**
- Training data editing
- Prompt template library
- Notes management
- Sub-tab organization

**Connected Files:**
- `enigma_engine/config/` - Paths

---

### bundle_manager_tab.py (991 lines)
**Purpose:** Package AI models for sharing

**Main Components:**
- `BundleManagerTab` - Tab class
- `BundleDisplayInfo` - Dataclass

**Key Features:**
- Create/load .enigma-bundle files
- Export for sharing
- Clone bundles
- Metadata display

**Connected Files:**
- `enigma_engine/config/` - Bundle paths

---

### training_data_tab.py (987 lines)
**Purpose:** Generate training data using AI

**Main Components:**
- `GeneratorWorker` - QThread worker

**Key Features:**
- API providers (Claude, GPT-4)
- Local models (TinyLlama, Phi-2, Mistral)
- Q&A, conversation, instruction formats
- Topic-based generation

**Connected Files:**
- `enigma_engine/config/` - API keys

---

### character_trainer_tab.py (980 lines)
**Purpose:** Train specialized character models

**Main Components:**
- `CharacterTrainerTab` - Tab class
- `ScanWorker`, `ExtractWorker` - Thread workers

**Key Features:**
- Scan data for characters
- Extract personality traits
- Generate character datasets
- Task training

**Connected Files:**
- `enigma_engine/tools/data_trainer.py` - Character tools

---

### voice_clone_tab.py (923 lines)
**Purpose:** Create custom voices

**Main Components:**
- `VoiceGenerationWorker` - QThread worker

**Key Features:**
- Clone voice from audio clips
- AI-generated voices from text
- Voice profile management
- Real-time preview

**Connected Files:**
- `enigma_engine/voice/audio_analyzer.py`
- `enigma_engine/voice/voice_generator.py`
- `enigma_engine/voice/voice_profile.py`

---

### dashboard_tab.py (831 lines)
**Purpose:** Visual system overview

**Main Components:**
- `CircularGauge` - Resource gauge widget
- `MiniLineChart` - Usage chart widget
- `QuickActionButton` - Action button

**Key Features:**
- CPU/RAM/Disk monitoring
- Historical charts
- Module status
- System alerts

**Connected Files:**
- `psutil` - System monitoring

---

### audio_tab.py (747 lines)
**Purpose:** Text-to-speech generation

**Main Components:**
- `LocalTTS` - Local TTS provider
- `AudioTab` - Tab class

**Key Features:**
- Local TTS (pyttsx3)
- ElevenLabs cloud TTS
- Replicate audio
- Voice/rate/volume controls

**Connected Files:**
- `enigma_engine/builtin/` - Fallback TTS
- `enigma_engine/config/` - Output paths

---

### persona_tab.py (761 lines)
**Purpose:** Manage AI personas/prompts

**Main Components:**
- `create_persona_tab()` - Factory function

**Key Features:**
- Create/edit personas
- Copy to create variants
- Export/import
- Quick switching

**Connected Files:**
- `enigma_engine/core/persona.py` - AIPersona class

---

### task_offloading_tab.py (773 lines)
**Purpose:** Distribute tasks across devices

**Main Components:**
- `TaskType` - Task enum
- `RoutingMode` - Routing enum
- `TaskConfig`, `QueuedTask` - Dataclasses

**Key Features:**
- Per-task device assignment
- Task queue visualization
- Routing modes (auto, round-robin, etc.)
- Priority/timeout configuration

**Connected Files:**
- (Standalone tab)

---

### network_tab.py (717 lines)
**Purpose:** Multi-device management

**Main Components:**
- `NetworkTab` - Tab class
- `NetworkScanner` - QThread scanner

**Key Features:**
- Scan for Forge devices
- Start/stop API server
- Connect to remote instances
- Model sync

**Connected Files:**
- `enigma_engine/comms/discovery.py` - Device discovery

---

### threed_tab.py (718 lines)
**Purpose:** Generate 3D models

**Main Components:**
- `Local3DGen` - Local provider
- `ThreeDTab` - Tab class

**Key Features:**
- Shap-E local generation
- Replicate cloud generation
- OBJ fallback
- GPU/CPU mode

**Connected Files:**
- `enigma_engine/builtin/` - Fallback 3D
- `enigma_engine/config/` - Output paths

---

### analytics_tab.py (697 lines)
**Purpose:** Usage statistics and performance insights

**Main Components:**
- `AnalyticsRecorder` - Global analytics helper
- `AnalyticsTab` - Main tab class

**Key Features:**
- Tool usage statistics
- Chat session metrics
- Model performance tracking
- Training history charts
- Visual graphs

**Connected Files:**
- Standalone (stores in ~/.enigma_engine/analytics/)

---

### import_models_tab.py (721 lines)
**Purpose:** Import external AI models into Enigma Engine

**Main Components:**
- `ImportWorker` - Background import thread
- Factory function

**Key Features:**
- HuggingFace model import
- GGUF (llama.cpp) model loading
- Local model file import
- Conversion for training

**Connected Files:**
- `enigma_engine/core/huggingface_loader.py`
- `enigma_engine/core/gguf_loader.py`

---

### scheduler_tab.py (708 lines)
**Purpose:** View and manage scheduled tasks

**Main Components:**
- Task management dialogs

**Key Features:**
- Create scheduled tasks
- Edit/delete existing tasks
- Run tasks manually
- View task history
- Cron-like scheduling

**Connected Files:**
- Standalone (stores in ~/.enigma_engine/scheduled_tasks.json)

---

### model_comparison_tab.py (671 lines)
**Purpose:** Compare responses from multiple models

**Main Components:**
- Comparison worker thread
- Side-by-side display

**Key Features:**
- Select 2-4 models for comparison
- Simultaneous response generation
- Latency and token count metrics
- Rating system for responses
- Save comparison results

**Connected Files:**
- `enigma_engine/core/inference.py`

---

### base_generation_tab.py (690 lines)
**Purpose:** Base class for all generation tabs

**Main Components:**
- `BaseGenerationTab` - Abstract base class

**Key Features:**
- Consistent header styling
- Standard progress/status layout
- Provider management patterns
- Auto-open functionality
- Device-aware styling

**Connected Files:**
- Child tabs: image_tab, code_tab, video_tab, audio_tab, etc.
- `unified_patterns.py` - Styling

---

### embeddings_tab.py (656 lines)
**Purpose:** Generate and compare text embeddings

**Main Components:**
- `LocalEmbedding` - Local provider
- `EmbeddingsTab` - Tab class

**Key Features:**
- Local: sentence-transformers
- Cloud: OpenAI embeddings API
- Similarity comparison
- Export embeddings to JSON

**Connected Files:**
- `enigma_engine/memory/vector_db.py`

---

### video_tab.py (646 lines)
**Purpose:** Generate videos from prompts

**Main Components:**
- `LocalVideo` - AnimateDiff provider
- `VideoTab` - Tab class

**Key Features:**
- Local: AnimateDiff generation
- Cloud: Replicate video API
- GIF fallback for simple animations
- Frame count and FPS control

**Connected Files:**
- `enigma_engine/config/` - Output paths

---

### code_tab.py (631 lines)
**Purpose:** Generate code from prompts

**Main Components:**
- `ForgeCode` - Local code provider
- `OpenAICode` - GPT-4 provider
- `CodeTab` - Tab class

**Key Features:**
- Local model code generation
- GPT-4 cloud generation
- Language selection
- Syntax highlighting
- Export to file

**Connected Files:**
- `enigma_engine/core/inference.py`

---

### model_router_tab.py (611 lines)
**Purpose:** Assign models to specific tasks

**Main Components:**
- Model-to-tool assignment UI
- Category-organized tools

**Key Features:**
- Map models to tool categories
- Generation, perception, memory tasks
- Visual category color coding
- Load/save configurations

**Connected Files:**
- `enigma_engine/core/tool_router.py`

---

### scaling_tab.py (592 lines)
**Purpose:** Select model sizes

**Main Components:**
- Model card list
- Spec detail panel

**Key Features:**
- 15 model size presets
- Visual size comparison
- RAM/VRAM requirements
- Hardware tier recommendations

**Connected Files:**
- `enigma_engine/core/model.py` - ForgeConfig

---

### marketplace_tab.py (561 lines)
**Purpose:** Browse and install community plugins

**Main Components:**
- `MarketplaceTab` - Browse interface
- Plugin cards with ratings

**Key Features:**
- Search plugins
- One-click installation
- Rating display
- Category filtering

**Connected Files:**
- `enigma_engine/plugins/` - Plugin directory

---

### devices_tab.py (544 lines)
**Purpose:** Manage network devices

**Main Components:**
- `DeviceItem` - Device display widget
- `DevicesTab` - Main tab class

**Key Features:**
- View connected devices
- Device capability display
- Task offloading settings
- Device status monitoring

**Connected Files:**
- `enigma_engine/comms/discovery.py`

---

### federation_tab.py (539 lines)
**Purpose:** Federated learning management

**Main Components:**
- `FederationTab` - Main tab class

**Key Features:**
- Create/join federations
- View federation statistics
- Privacy settings
- Monitor training rounds
- Participant management

**Connected Files:**
- `enigma_engine/federated/` - Federated learning package

---

### notes_tab.py (538 lines)
**Purpose:** Quick notes and bookmarks

**Main Components:**
- `NotesManager` - Storage backend
- Notes list and editor

**Key Features:**
- Create/edit/delete notes
- Tag-based organization
- Search functionality
- Markdown preview
- Bookmarks viewer

**Connected Files:**
- Standalone (stores in ~/.enigma_engine/notes/)

---

### learning_tab.py (524 lines)
**Purpose:** Self-improvement dashboard

**Main Components:**
- `LearningTab` - Main tab class

**Key Features:**
- Real-time metrics display
- Training examples count
- Learning progress visualization
- Manual training trigger
- Autonomous learning toggle

**Connected Files:**
- `enigma_engine/learning/` - Learning package

---

### gif_tab.py (502 lines)
**Purpose:** Create animated GIFs

**Main Components:**
- `GIFGenerationWorker` - Background thread
- `GIFTab` - Tab class

**Key Features:**
- Generate GIFs from prompts
- Animate image sequences
- Morph between prompts
- FPS and loop control

**Connected Files:**
- `enigma_engine/config/` - Output paths

---

### personality_tab.py (500 lines)
**Purpose:** Configure AI personality

**Main Components:**
- Trait sliders
- Preset selector

**Key Features:**
- Adjustable personality traits
- Preset personalities (Professional, Friendly, etc.)
- Personality evolution toggle
- Live description preview

**Connected Files:**
- `enigma_engine/core/persona.py`

---

### camera_tab.py (482 lines)
**Purpose:** Live camera preview and capture

**Main Components:**
- `CameraThread` - OpenCV capture thread
- `CameraTab` - Tab class

**Key Features:**
- Webcam preview
- Photo capture
- Resolution controls
- Camera device selection
- AI analysis integration

**Connected Files:**
- Requires OpenCV (cv2)
- `enigma_engine/config/` - Image storage

---

### tool_manager_tab.py (434 lines)
**Purpose:** Enable/disable AI tools

**Main Components:**
- `ToolManagerTab` - Main tab class
- Tool tree widget

**Key Features:**
- Tool enable/disable toggles
- Preset profiles (minimal, standard, full)
- Dependency visualization
- Save/load custom profiles

**Connected Files:**
- `enigma_engine/tools/tool_manager.py`

---

### instructions_tab.py (411 lines)
**Purpose:** Help and documentation viewer

**Main Components:**
- File tree browser
- Text editor

**Key Features:**
- Quick start guide
- Training data format help
- Model size documentation
- Edit model data files

**Connected Files:**
- Standalone help content

---

### logs_tab.py (402 lines)
**Purpose:** View system logs

**Main Components:**
- `LogViewerWidget` - Log display
- Multiple log tabs

**Key Features:**
- Real-time log viewing
- Filter by level (DEBUG, INFO, etc.)
- Search within logs
- Export logs to file
- Clear logs

**Connected Files:**
- Project logs/ directory
- ~/.enigma_engine/logs/

---

### vision_tab.py (230 lines)
**Purpose:** Screen capture and image analysis

**Main Components:**
- `create_vision_tab()` - Factory function

**Key Features:**
- Screen capture
- Load images for analysis
- AI image description
- History list

**Connected Files:**
- `enigma_engine/tools/vision.py`

---

### ai_tab.py (205 lines)
**Purpose:** Consolidated AI configuration

**Main Components:**
- `AITab` - Tab class with sub-tabs

**Key Features:**
- Combines: Avatar, Modules, Scaling, Training
- Sub-tab navigation
- Standard mode interface

**Connected Files:**
- References modules_tab, scaling_tab, training_tab

---

### terminal_tab.py (197 lines)
**Purpose:** AI processing terminal

**Main Components:**
- `create_terminal_tab()` - Factory function

**Key Features:**
- View AI processing in real-time
- Auto-scroll toggle
- Clear terminal
- Monospace font display

**Connected Files:**
- Standalone (receives logs from inference)

---

### create_tab.py (189 lines)
**Purpose:** Consolidated creation interface

**Main Components:**
- Sub-tab container

**Key Features:**
- Combines: Image, Code, Video, Audio tabs
- Cleaner Standard mode interface

**Connected Files:**
- References image_tab, code_tab, video_tab, audio_tab

---

### sessions_tab.py (183 lines)
**Purpose:** View chat history

**Main Components:**
- `create_sessions_tab()` - Factory function

**Key Features:**
- Per-AI chat history
- Session list sidebar
- Chat content viewer
- AI model selector

**Connected Files:**
- `enigma_engine/memory/manager.py`

---

### avatar_tab.py (54 lines)
**Purpose:** Avatar/Game/Robot control container

**Main Components:**
- Sub-tab container only

**Key Features:**
- Avatar display sub-tab
- Game connection sub-tab
- Robot control sub-tab

**Connected Files:**
- `gui/tabs/avatar/avatar_display.py`
- `gui/tabs/game/game_connection.py`
- `gui/tabs/robot/robot_control.py`

---

## Core Package Summary (~170 files)

### Model & Inference
| File | Lines | Purpose |
|------|-------|---------|
| model.py | 3,214 | Transformer model (Enigma class, ForgeConfig) |
| inference.py | 2,167 | Text generation engine (EnigmaEngine) |
| training.py | 1,974 | Training loops (Trainer, TrainingConfig) |
| tokenizer.py | 826 | Text ↔ tokens (get_tokenizer, SimpleTokenizer) |
| tool_router.py | 3,374 | Intent detection, specialized model routing |

### Advanced Features
| File | Purpose |
|------|---------|
| moe.py | Mixture of Experts architecture |
| ssm.py | Mamba/S4 state space models |
| flash_attention.py | Flash attention for speed |
| infinite_context.py | Streaming context extension |
| paged_attention.py | Memory-efficient KV cache |

### Quantization & Loading
| File | Purpose |
|------|---------|
| quantization.py | INT8/4-bit quantization |
| awq_quantization.py | AWQ quantization method |
| gptq_quantization.py | GPTQ quantization method |
| huggingface_loader.py | Load HuggingFace models |
| gguf_loader.py | Load GGUF (llama.cpp) models |
| ollama_loader.py | Load Ollama models |

### Training Variants
| File | Purpose |
|------|---------|
| lora_training.py | LoRA fine-tuning |
| qlora.py | QLoRA (quantized LoRA) |
| dpo.py | Direct Preference Optimization |
| rlhf.py | RLHF training |
| distillation.py | Model distillation |

---

## Tools Package Summary (~70 files)

### Core Tool System
| File | Lines | Purpose |
|------|-------|---------|
| tool_executor.py | 2,877 | Safe tool execution with timeouts |
| tool_definitions.py | ~1,500 | Tool schemas (ToolDefinition, ToolParameter) |
| tool_registry.py | ~800 | Tool registration and lookup |
| tool_manager.py | ~600 | Enable/disable tools, presets |

### Tool Categories
| Category | Files | Examples |
|----------|-------|----------|
| Vision | vision.py, simple_ocr.py | Screen capture, image analysis |
| Web | web_tools.py, browser_tools.py | Web search, fetch URLs |
| Files | file_tools.py, document_tools.py | Read/write files |
| System | system_tools.py | Run commands, system info |
| Gaming | game_router.py, game_state.py | Game AI routing |
| Robot | robot_tools.py, robot_modes.py | Robot control |

---

## Utils Package Summary (~80 files)

### Key Utilities
| File | Purpose |
|------|---------|
| security.py | Path blocking, sandboxing |
| lazy_import.py | LazyLoader for fast startup |
| api_key_encryption.py | Secure API key storage |
| battery_manager.py | Power management |
| backup.py | Model/data backup |

---

## File Dependencies

### Core Module Dependencies
```
model.py ← inference.py ← chat_tab.py
         ← training.py ← training_tab.py
         ← modules/registry.py

tokenizer.py ← training.py
             ← inference.py

tool_router.py ← inference.py (with use_routing=True)
              ← model_router_tab.py
```

### GUI Tab Dependencies
```
All tabs ← enigma_engine/config/ (CONFIG)
All tabs ← gui/shared_components.py

chat_tab ← core/inference.py, memory/manager.py
settings_tab ← config/, utils/api_key_encryption.py
modules_tab ← modules/manager.py, modules/registry.py
training_tab ← core/training.py
image_tab ← gui/tabs/output_helpers.py
audio_tab ← builtin/, voice/
voice_clone_tab ← voice/*
character_trainer_tab ← tools/data_trainer.py
network_tab ← comms/discovery.py
persona_tab ← core/persona.py
dashboard_tab ← psutil (external)
```

### External Dependencies
- **PyTorch** - All AI operations
- **PyQt5** - All GUI components
- **psutil** - System monitoring (dashboard_tab)
- **pyttsx3** - Local TTS (audio_tab)
- **requests** - HTTP requests
- **transformers** - HuggingFace model loading
- **diffusers** - Stable Diffusion (image_tab)

---

## Statistics

| Category | Count |
|----------|-------|
| **GUI** | |
| Total GUI Tabs | 44 |
| GUI Lines (estimated) | ~45,000 |
| Largest Tab | settings_tab.py (4,721 lines) |
| **Core** | |
| Core Files | ~170 |
| Model Presets | 15 (nano → omega) |
| **Tools** | |
| Tool Files | ~70 |
| Tool Categories | 8 |
| **Utils** | |
| Utils Files | ~80 |
| **Total** | |
| **Python Files** | **816** |
| **Total Lines** | **458,255** |
| Dead Code Removed | 15 files (~8,000 lines) |

---

*Report generated by Enigma AI Engine Code Analysis - February 15, 2026*
