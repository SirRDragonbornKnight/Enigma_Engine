# AI Tester Engine - Architecture

*Last Updated: January 2026*

## Overview

Enigma Engine is a **fully modular AI framework** where everything is a toggleable module. This prevents conflicts and allows flexible configuration from Raspberry Pi to datacenter.

## Package Structure

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
│              │  - threed    │              │                           │
├──────────────┴──────────────┴──────────────┴───────────────────────────┤
│    TOOLS              │    NETWORK              │    INTERFACE          │
│  - web_tools          │  - api_server           │  - gui (with tabs)    │
│  - file_tools         │  - network (multi-dev)  │                       │
└───────────────────────┴─────────────────────────┴───────────────────────┘
```

## Core Packages

### enigma.core
The brain of the system - contains:
- **model.py**: Enigma transformer with RoPE, RMSNorm, SwiGLU, GQA, KV-cache
- **inference.py**: High-performance generation engine with streaming
- **training.py**: Mixed-precision training with cosine warmup
- **tokenizer.py**: Unified tokenizer interface
- **advanced_tokenizer.py**: Custom BPE with `[E:token]` format

### enigma.modules
Module system for dynamic capability loading:
- **manager.py**: ModuleManager - central controller
- **registry.py**: All available modules with metadata
- **base.py**: BaseModule abstract class

### enigma.memory
Conversation storage and retrieval:
- **manager.py**: ConversationManager
- **memory_db.py**: SQLite persistence
- **vector_db.py**: FAISS/Pinecone vector search
- **categorization.py**: Memory type classification

### enigma.gui
PyQt5-based graphical interface:
- **enhanced_window.py**: Main window with all tabs
- **theme_system.py**: Dark/Light/Shadow/Midnight themes
- **tabs/**: Individual capability tabs (chat, training, image, video, etc.)

### enigma.comms
Networking and API:
- **api_server.py**: Flask REST API
- **remote.py**: Remote client for distributed setups
- **network.py**: Multi-device synchronization

### enigma.voice
Speech capabilities:
- **tts.py**: Text-to-speech (pyttsx3, ElevenLabs)
- **stt.py**: Speech-to-text (Vosk, SpeechRecognition)
- **voice_profile.py**: Custom voice configurations

### enigma.avatar
Visual character system:
- **controller.py**: Avatar state management
- **animation.py**: Gesture and expression control

### enigma.tools
AI capabilities:
- **vision.py**: Screen capture, image analysis
- **web.py**: URL safety, web search
- **file_ops.py**: File system operations

### enigma.utils
Helper utilities:
- **system_messages.py**: Formatted output
- **text_formatting.py**: Code highlighting, markdown

## Model Sizes

| Size | Params | Use Case |
|------|--------|----------|
| nano | ~1M | Embedded/testing |
| micro | ~2M | Raspberry Pi |
| tiny | ~5M | Light devices |
| small | ~27M | Desktop default |
| medium | ~85M | Good balance |
| large | ~300M | Quality focus |
| xl-omega | 1B-70B+ | Datacenter |

## Design Principles

1. **Modularity**: Everything is a toggleable module
2. **Conflict Prevention**: Module manager prevents incompatible loads
3. **Hardware Awareness**: Auto-adjusts to available resources
4. **Local-First**: Works entirely offline by default
5. **API Compatible**: Can use cloud providers when needed

This layout keeps responsibilities isolated so you can replace modules independently.
