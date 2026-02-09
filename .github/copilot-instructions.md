# Enigma AI Engine - AI Coding Guidelines

## Architecture Overview

Enigma AI Engine is a **fully modular AI framework** where EVERYTHING is a toggleable module. This prevents conflicts and allows flexible configuration from Raspberry Pi to datacenter.

### System Architecture
```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MODULE MANAGER                                 │
│            enigma_engine/modules/manager.py - Central Control                 │
├─────────────────────────────────────────────────────────────────────────┤
│                           MODULE REGISTRY                                │
│        enigma_engine/modules/registry.py - All Available Modules              │
├──────────────┬──────────────┬──────────────┬───────────────────────────┤
│    CORE      │  GENERATION  │   MEMORY     │    PERCEPTION/OUTPUT      │
│  - model     │  - image_gen │  - memory    │  - voice_input/output     │
│  - tokenizer │  - code_gen  │  - embedding │  - vision                 │
│  - training  │  - video_gen │              │  - avatar                 │
│  - inference │  - audio_gen │              │  - camera                 │
│  - gguf      │  - threed    │              │                           │
├──────────────┴──────────────┴──────────────┴───────────────────────────┤
│    TOOLS              │    NETWORK              │    INTERFACE          │
│  - web_tools          │  - api_server           │  - gui (with tabs)    │
│  - file_tools         │  - network (multi-dev)  │                       │
│  - tool_router        │                         │                       │
└───────────────────────┴─────────────────────────┴───────────────────────┘
```

### Core Packages
- **enigma_engine.core**: Forge transformer model with RoPE, RMSNorm, SwiGLU, GQA, KV-cache, optional Flash Attention, tool routing
- **enigma_engine.modules**: Module system - manager, registry, state handling
- **enigma_engine.gui.tabs**: Generation capabilities in standalone tabs (image, code, video, audio, 3D, embeddings, camera)
- **enigma_engine.memory**: Conversation storage (JSON/SQLite), vector search
- **enigma_engine.comms**: API server, remote client, multi-device networking
- **enigma_engine.gui**: PyQt5 interface with Module Manager tab
- **enigma_engine.voice**: TTS/STT wrappers
- **enigma_engine.avatar**: Avatar control and rendering
- **enigma_engine.tools**: Vision, web, file, document, robot, game tools

### Model Sizes (15 presets)
| Size | Params | Use Case |
|------|--------|----------|
| nano | ~1M | Embedded/testing |
| micro | ~2M | Raspberry Pi |
| tiny | ~5M | Light devices |
| small | ~27M | Desktop default |
| medium | ~85M | Good balance |
| large | ~300M | Quality focus |
| xl-omega | 1B-70B+ | Datacenter |

## Key Files & Classes Reference

### Module Management
- **enigma_engine/modules/manager.py**: `ModuleManager`, `Module`, `ModuleInfo`, `ModuleState`, `ModuleCategory` classes - Central module system (auto-registers 50+ modules on init)
- **enigma_engine/modules/registry.py**: Module classes (`ModelModule`, `TokenizerModule`, `ToolRouterModule`, etc.) + `MODULE_REGISTRY` dict, `register_all()`
- **enigma_engine/modules/sandbox.py**: Sandboxed module execution

### Core AI Components
- **enigma_engine/core/model.py**: `Enigma`, `create_model()` - Main transformer model implementation
- **enigma_engine/core/tokenizer.py**: `get_tokenizer()`, `SimpleTokenizer`, `TiktokenWrapper` - Text tokenization
- **enigma_engine/core/training.py**: `Trainer`, `TrainingConfig`, `train_model()` - Model training
- **enigma_engine/core/inference.py**: `EnigmaEngine` class - Model inference with optional `use_routing` for specialized models
- **enigma_engine/core/model_registry.py**: `ModelRegistry` class - Manages multiple loaded models, `export_to_huggingface()` for uploading to HF Hub
- **enigma_engine/core/tool_router.py**: `ToolRouter`, `get_router()`, `classify_intent()`, `describe_image()`, `generate_code()` - Specialized model routing
- **enigma_engine/core/huggingface_loader.py**: `load_huggingface_model()` - Load HuggingFace models
- **enigma_engine/core/huggingface_exporter.py**: `HuggingFaceExporter`, `export_model_to_hub()`, `export_model_locally()` - Upload Enigma AI Engine models to HuggingFace
- **enigma_engine/core/gguf_loader.py**: GGUF format model loading

### AI Generation Tabs (in enigma_engine/gui/tabs/)
Each tab contains both the implementation (provider classes) and the GUI:
- **image_tab.py**: `StableDiffusionLocal`, `OpenAIImage`, `ReplicateImage` + `ImageTab`
- **code_tab.py**: `ForgeCode`, `OpenAICode` + `CodeTab`
- **video_tab.py**: `LocalVideo`, `ReplicateVideo` + `VideoTab`
- **audio_tab.py**: `LocalTTS`, `ElevenLabsTTS`, `ReplicateAudio` + `AudioTab`
- **embeddings_tab.py**: `LocalEmbedding`, `OpenAIEmbedding` + `EmbeddingsTab`
- **threed_tab.py**: `Local3DGen`, `Cloud3DGen` + `ThreeDTab`
- **gif_tab.py**: `GIFTab` class - Animated GIF generation
- **camera_tab.py**: `CameraTab` class - Webcam capture and analysis
- **vision_tab.py**: `VisionTab` class - Image/screen analysis
- **model_router_tab.py**: `ModelRouterTab` class - Tool-to-model assignment UI

### Memory System
- **enigma_engine/memory/manager.py**: `ConversationManager` class - Stores chat history
- **enigma_engine/memory/vector_db.py**: `VectorDBInterface`, `FAISSVectorDB`, `SimpleVectorDB` - Semantic search over memories

### Communication & Networking
- **enigma_engine/comms/api_server.py**: `create_api_server()` function - REST API for remote access
- **enigma_engine/comms/network.py**: `ForgeNode`, `Message`, `ModelExporter` classes - Multi-device networking

### Voice System
- **enigma_engine/voice/voice_generator.py**: `AIVoiceGenerator`, `VoiceEvolution` classes - Voice synthesis
- **enigma_engine/voice/listener.py**: `VoiceListener` class - Speech-to-text input

### User Interface
- **enigma_engine/gui/enhanced_window.py**: `EnhancedMainWindow` class - PyQt5 main application window
- **enigma_engine/gui/tabs/modules_tab.py**: `ModulesTab` class - UI for toggling modules on/off
- **enigma_engine/gui/system_tray.py**: `QuickCommandOverlay` class - Mini chat window

### Web Interface
- **enigma_engine/web/app.py**: `run_web()` function - Flask web dashboard, `app` Flask instance

### Autonomous Systems (Avatar/Robot/Game Control)
- **enigma_engine/avatar/controller.py**: `AvatarController`, `ControlPriority` - Main avatar control with priority system
- **enigma_engine/avatar/bone_control.py**: `BoneController`, `get_bone_controller()` - PRIMARY avatar control via bone rigging (priority 100)
- **enigma_engine/avatar/autonomous.py**: `AutonomousAvatar`, `AutonomousConfig`, `ScreenRegion` - FALLBACK auto-behavior (priority 50)
- **enigma_engine/tools/robot_modes.py**: `RobotModeController`, `get_mode_controller()` - Robot hardware control
- **enigma_engine/tools/game_router.py**: `GameAIRouter`, `GameConfig`, `get_game_router()` - Game-specific AI routing

### Avatar Control Priority System
Avatar control uses a priority system to prevent conflicts:
```python
ControlPriority.BONE_ANIMATION = 100   # PRIMARY - Direct bone control (rigged models)
ControlPriority.USER_MANUAL = 80       # User dragging/clicking
ControlPriority.AI_TOOL_CALL = 70      # AI explicit commands
ControlPriority.AUTONOMOUS = 50        # Autonomous behaviors (FALLBACK)
ControlPriority.IDLE_ANIMATION = 30    # Background animations
ControlPriority.FALLBACK = 10          # For non-avatar-trained models
```
**Bone animation is PRIMARY** - all other systems are fallbacks for models without bone control.

### Tools System
- **enigma_engine/tools/tool_executor.py**: `ToolExecutor` class - Executes AI tool calls
- **enigma_engine/tools/tool_definitions.py**: `ToolDefinition`, `ToolParameter` classes, `get_all_tools()` function
- **enigma_engine/tools/tool_registry.py**: `ToolRegistry` class - Manages available tools

### Security
- **enigma_engine/utils/security.py**: `is_path_blocked()`, `get_blocked_paths()` functions - Path blocking for AI safety

### Configuration
- **enigma_engine/config/**: `CONFIG` object - Global configuration (paths, model sizes, hyperparameters)

## Module System

### How It Works
```python
from enigma_engine.modules import ModuleManager

manager = ModuleManager()

# Load modules (checks dependencies & conflicts)
manager.load('model')
manager.load('tokenizer')
manager.load('inference')

# Load generation capability (LOCAL or API - not both)
manager.load('image_gen_local')  # Uses Stable Diffusion

# Use the module
image_mod = manager.get_module('image_gen_local')
result = image_mod.generate("a sunset over mountains", width=512, height=512)

# Unload when done
manager.unload('image_gen_local')
```

### Generation Modules (AI Capabilities)
Each generation capability has LOCAL and API variants. **Only one can be loaded at a time** (they provide the same capability).

| Module | Local | API |
|--------|-------|-----|
| Image Gen | `image_gen_local` (Stable Diffusion) | `image_gen_api` (DALL-E, Replicate) |
| Code Gen | `code_gen_local` (Enigma model) | `code_gen_api` (GPT-4) |
| Video Gen | `video_gen_local` (AnimateDiff) | `video_gen_api` (Replicate) |
| Audio/TTS | `audio_gen_local` (pyttsx3) | `audio_gen_api` (ElevenLabs) |
| Embeddings | `embedding_local` (sentence-transformers) | `embedding_api` (OpenAI) |
| 3D Gen | `threed_gen_local` (Shap-E) | `threed_gen_api` (Replicate) |

### Conflict Prevention
The module manager automatically prevents:
- Loading two modules that provide the same capability
- Loading modules without their dependencies
- Loading modules that exceed hardware limits
- Resource conflicts between modules

## Developer Workflows
- **Setup (Linux/macOS)**: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
- **Setup (Windows)**: `python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt`
- **Train Model**: `python run.py --train`
- **Train Specialized**: `python scripts/train_specialized_model.py --type router --data data/specialized/router_training.txt`
- **Run Inference**: `python run.py --run` (CLI) or `python run.py --serve` (API)
- **GUI**: `python run.py --gui` - Module Manager tab to toggle capabilities

## Conventions
- **Imports**: Relative within enigma_engine (`from ..config import CONFIG`)
- **Paths**: Use `pathlib.Path`, dirs auto-created via CONFIG
- **Modules**: Always use ModuleManager for loading capabilities
- **Tabs**: Generation implementations live directly in their GUI tabs

## UI Guidelines
- **NO EMOJIS**: Never use emojis, unicode symbols, or special characters in GUI text (buttons, labels, menus, status messages, tooltips). Keep UI text clean and professional.
- **Button Text**: Keep button text short (1-2 words max). Use tooltips for longer descriptions.
- **Font Sizes**: Use 10-12px for most UI elements. Never exceed 14px except for headings.
- **Consistent Styling**: Follow existing PyQt5 stylesheet patterns in the codebase.

## What NOT To Do

❌ **DON'T load conflicting modules** - e.g., both `image_gen_local` AND `image_gen_api`

❌ **DON'T bypass ModuleManager** - Always use manager.load() for proper dependency handling

❌ **DON'T train with very small datasets** - Need 1000+ lines minimum

❌ **DON'T use extremely high learning rates** - Default 0.0001 is safe

❌ **DON'T delete model folders manually** - Use Model Manager

❌ **DON'T train during inference** - Close chat first or use separate instances

❌ **DON'T run multiple instances on same model** - Can corrupt files

❌ **DON'T block the UI thread** - Use QThread for long operations

❌ **DON'T mix model sizes carelessly** - Requires proper conversion