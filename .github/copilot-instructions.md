# Enigma Engine - AI Coding Guidelines

## Architecture Overview

Enigma Engine is a **fully modular AI framework** where EVERYTHING is a toggleable module. This prevents conflicts and allows flexible configuration from Raspberry Pi to datacenter.

### System Architecture
```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MODULE MANAGER                                 │
│              enigma/modules/manager.py - Central Control                 │
├─────────────────────────────────────────────────────────────────────────┤
│                           MODULE REGISTRY                                │
│          enigma/modules/registry.py - All Available Modules              │
├──────────────┬──────────────┬──────────────┬───────────────────────────┤
│    CORE      │  GENERATION  │   MEMORY     │    PERCEPTION/OUTPUT      │
│  - model     │  - image_gen │  - memory    │  - voice_input/output     │
│  - tokenizer │  - code_gen  │  - embedding │  - vision                 │
│  - training  │  - video_gen │              │  - avatar                 │
│  - inference │  - audio_gen │              │                           │
├──────────────┴──────────────┴──────────────┴───────────────────────────┤
│    TOOLS              │    NETWORK              │    INTERFACE          │
│  - web_tools          │  - api_server           │  - gui                │
│  - file_tools         │  - network (multi-dev)  │                       │
└───────────────────────┴─────────────────────────┴───────────────────────┘
```

### Core Packages
- **enigma.core**: Enigma transformer model with RoPE, RMSNorm, SwiGLU, KV-cache
- **enigma.modules**: Module system - manager, registry, state handling
- **enigma.addons**: AI generation capabilities (image, code, video, audio, embeddings)
- **enigma.memory**: Conversation storage (JSON/SQLite), vector search
- **enigma.comms**: API server, remote client, multi-device networking
- **enigma.gui**: PyQt5 interface with Module Manager tab
- **enigma.voice**: TTS/STT wrappers
- **enigma.avatar**: Avatar control
- **enigma.tools**: Vision, web, file, document tools

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

## Module System

### How It Works
```python
from enigma.modules import ModuleManager

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

### Conflict Prevention
The module manager automatically prevents:
- Loading two modules that provide the same capability
- Loading modules without their dependencies
- Loading modules that exceed hardware limits
- Resource conflicts between modules

## Developer Workflows
- **Setup**: `python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt`
- **Train Model**: `python run.py --train`
- **Run Inference**: `python run.py --run` (CLI) or `python run.py --serve` (API)
- **GUI**: `python run.py --gui` - Module Manager tab to toggle capabilities

## Conventions
- **Imports**: Relative within enigma (`from ..config import CONFIG`)
- **Paths**: Use `pathlib.Path`, dirs auto-created via CONFIG
- **Modules**: Always use ModuleManager for loading capabilities
- **Addons**: Wrap external AI via addon base classes, integrate as modules

## What NOT To Do

❌ **DON'T load conflicting modules** - e.g., both `image_gen_local` AND `image_gen_api`

❌ **DON'T bypass ModuleManager** - Always use manager.load() for proper dependency handling

❌ **DON'T train with very small datasets** - Need 1000+ lines minimum

❌ **DON'T use extremely high learning rates** - Default 0.0001 is safe

❌ **DON'T delete model folders manually** - Use Model Manager

❌ **DON'T train during inference** - Close chat first or use separate instances

❌ **DON'T run multiple instances on same model** - Can corrupt files

❌ **DON'T block the UI thread** - Use QThread for long operations

❌ **DON'T mix model sizes carelessly** - Requires proper conversion</content>
<parameter name="filePath">c:\Users\sirkn_gbhnunq\Documents\GitHub\enigma_engine\.github\copilot-instructions.md