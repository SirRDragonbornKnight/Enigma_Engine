# ForgeAI Module Dependency Graph

This document describes the relationships between ForgeAI modules - which modules depend on others to function.

## Dependency Graph (Visual)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CORE FOUNDATION LAYER                               │
│  ┌─────────┐    ┌───────────┐                                               │
│  │  model  │    │ tokenizer │   (no dependencies - load first)              │
│  └────┬────┘    └─────┬─────┘                                               │
│       │               │                                                     │
│       └───────┬───────┘                                                     │
│               ▼                                                             │
│       ┌───────────────┐    ┌──────────┐                                     │
│       │   training    │    │   gguf   │                                     │
│       │ (model+token) │    │ (loader) │                                     │
│       └───────────────┘    └──────────┘                                     │
│               │                                                             │
│               ▼                                                             │
│       ┌───────────────┐                                                     │
│       │   inference   │                                                     │
│       │ (model+token) │                                                     │
│       └───────┬───────┘                                                     │
│               │                                                             │
├───────────────┼─────────────────────────────────────────────────────────────┤
│               ▼           DEPENDENT MODULES                                 │
│       ┌───────────────┐                                                     │
│       │  api_server   │────► Provides REST API                              │
│       │  (inference)  │                                                     │
│       └───────────────┘                                                     │
│                                                                             │
│       ┌───────────────┐                                                     │
│       │  tool_router  │────► Routes to specialized models                   │
│       │  (tokenizer)  │                                                     │
│       └───────────────┘                                                     │
│                                                                             │
│       ┌───────────────┐                                                     │
│       │code_gen_local │────► Local code generation                          │
│       │(model+token+  │                                                     │
│       │  inference)   │                                                     │
│       └───────────────┘                                                     │
│                                                                             │
│       ┌───────────────┐    ┌───────────────┐                                │
│       │   sessions    │    │    scaling    │                                │
│       │   (memory)    │    │    (model)    │                                │
│       └───────────────┘    └───────────────┘                                │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                       STANDALONE MODULES                                     │
│                    (No dependencies - can load anytime)                      │
│                                                                             │
│  Memory:        memory, notes, personality                                   │
│  Voice:         voice_input, voice_output, voice_clone                       │
│  Vision:        vision, camera, motion_tracking                              │
│  Generation:    image_gen_*, video_gen_*, audio_gen_*, threed_gen_*,        │
│                 embedding_*, gif_gen (all standalone)                        │
│  Network:       network, tunnel                                              │
│  Interface:     gui, dashboard, terminal, analytics                          │
│  Tools:         web_tools, file_tools, game_ai, robot_control               │
│  Other:         scheduler, examples, instructions, logs, workspace,         │
│                 model_router, huggingface, avatar                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONFLICT GROUPS                                       │
│             (Only ONE from each group can be loaded at a time)               │
│                                                                             │
│  Image Generation:     image_gen_local  ←CONFLICT→  image_gen_api           │
│  Code Generation:      code_gen_local   ←CONFLICT→  code_gen_api            │
│  Video Generation:     video_gen_local  ←CONFLICT→  video_gen_api           │
│  Audio/TTS:            audio_gen_local  ←CONFLICT→  audio_gen_api           │
│  3D Generation:        threed_gen_local ←CONFLICT→  threed_gen_api          │
│  Embeddings:           embedding_local  ←CONFLICT→  embedding_api           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Complete Module List

### Core Modules (Load These First)

| Module ID | Dependencies | Provides | Description |
|-----------|--------------|----------|-------------|
| `model` | None | language_model, model_embeddings | Forge transformer neural network |
| `tokenizer` | None | tokenization | Text to token conversion |
| `training` | model, tokenizer | training_capability | Model training and fine-tuning |
| `inference` | model, tokenizer | text_generation | Text generation from the model |
| `gguf_loader` | None | gguf_loading | Load GGUF format models |
| `chat_api` | None | openai_compatible_api | OpenAI-compatible chat API |

### Memory & Storage

| Module ID | Dependencies | Description |
|-----------|--------------|-------------|
| `memory` | None | Conversation storage (JSON/SQLite) |
| `notes` | None | Note-taking and organization |
| `sessions` | memory | Multiple conversation sessions |
| `personality` | None | AI personality configuration |

### Voice & Audio

| Module ID | Dependencies | Description |
|-----------|--------------|-------------|
| `voice_input` | None | Speech-to-text (microphone) |
| `voice_output` | None | Text-to-speech (speakers) |
| `voice_clone` | None | Voice cloning capabilities |

### Vision & Camera

| Module ID | Dependencies | Description |
|-----------|--------------|-------------|
| `vision` | None | Image analysis and understanding |
| `camera` | None | Webcam capture and processing |
| `motion_tracking` | None | Body and face tracking |
| `avatar` | None | Visual avatar rendering |

### Generation Modules

**Image Generation (choose one):**
| Module ID | Dependencies | Backend |
|-----------|--------------|---------|
| `image_gen_local` | None | Stable Diffusion (local) |
| `image_gen_api` | None | DALL-E, Replicate (cloud) |

**Code Generation (choose one):**
| Module ID | Dependencies | Backend |
|-----------|--------------|---------|
| `code_gen_local` | model, tokenizer, inference | Local Enigma model |
| `code_gen_api` | None | GPT-4, Claude (cloud) |

**Video Generation (choose one):**
| Module ID | Dependencies | Backend |
|-----------|--------------|---------|
| `video_gen_local` | None | AnimateDiff (local) |
| `video_gen_api` | None | Replicate (cloud) |

**Audio/TTS Generation (choose one):**
| Module ID | Dependencies | Backend |
|-----------|--------------|---------|
| `audio_gen_local` | None | pyttsx3 (local) |
| `audio_gen_api` | None | ElevenLabs (cloud) |

**3D Generation (choose one):**
| Module ID | Dependencies | Backend |
|-----------|--------------|---------|
| `threed_gen_local` | None | Shap-E (local) |
| `threed_gen_api` | None | Replicate (cloud) |

**Embeddings (choose one):**
| Module ID | Dependencies | Backend |
|-----------|--------------|---------|
| `embedding_local` | None | sentence-transformers (local) |
| `embedding_api` | None | OpenAI embeddings (cloud) |

**Other Generation:**
| Module ID | Dependencies | Description |
|-----------|--------------|-------------|
| `gif_gen` | None | Animated GIF generation |

### Network & API

| Module ID | Dependencies | Description |
|-----------|--------------|-------------|
| `api_server` | inference | REST API server (Flask) |
| `network` | None | Multi-device networking |
| `tunnel` | None | Expose to internet (ngrok/localtunnel) |

### Interface

| Module ID | Dependencies | Description |
|-----------|--------------|-------------|
| `gui` | None | PyQt5 graphical interface |
| `dashboard` | None | Web dashboard |
| `terminal` | None | Terminal/console interface |
| `analytics` | None | Usage analytics |

### Tools & Utilities

| Module ID | Dependencies | Description |
|-----------|--------------|-------------|
| `web_tools` | None | Web scraping, search |
| `file_tools` | None | File operations |
| `tool_router` | tokenizer | Route requests to specialized models |
| `model_router` | None | UI for tool-to-model assignment |
| `game_ai` | None | Game-specific AI |
| `robot_control` | None | Hardware robot control |
| `scheduler` | None | Task scheduling |

### Other Modules

| Module ID | Dependencies | Description |
|-----------|--------------|-------------|
| `examples` | None | Example prompts and demos |
| `instructions` | None | Built-in instructions |
| `logs` | None | Logging viewer |
| `workspace` | None | Workspace management |
| `huggingface` | None | HuggingFace Hub integration |
| `scaling` | model | Model scaling utilities |

## Common Module Combinations

### Minimal Chat (CPU Friendly)
```python
manager.load('model')
manager.load('tokenizer')
manager.load('inference')
manager.load('memory')
```

### Full Local AI
```python
# Core
manager.load('model')
manager.load('tokenizer')
manager.load('inference')

# Features
manager.load('memory')
manager.load('voice_input')
manager.load('voice_output')
manager.load('image_gen_local')
manager.load('gui')
```

### API-Based (Minimal Local Resources)
```python
manager.load('memory')
manager.load('chat_api')           # OpenAI-compatible
manager.load('image_gen_api')      # DALL-E
manager.load('audio_gen_api')      # ElevenLabs
manager.load('gui')
```

### Developer Setup
```python
manager.load('model')
manager.load('tokenizer')
manager.load('training')
manager.load('inference')
manager.load('code_gen_local')
manager.load('tool_router')
manager.load('terminal')
```

### Server Deployment
```python
manager.load('model')
manager.load('tokenizer')
manager.load('inference')
manager.load('api_server')
manager.load('network')
manager.load('tunnel')
```

## Loading Order Best Practices

1. **Always load `model` before `tokenizer`** (or together)
2. **Load `training` or `inference` after model+tokenizer**
3. **Load `api_server` after inference**
4. **Generation modules** can be loaded anytime
5. **GUI** should be loaded last for best startup performance

## Troubleshooting

### "Module requires X"
Load the required dependency first:
```python
manager.load('model')      # Load dependency
manager.load('training')   # Now this works
```

### "Conflicting modules"
Unload the conflicting module first:
```python
manager.unload('image_gen_local')  # Unload local
manager.load('image_gen_api')       # Now load API version
```

### "Out of memory"
Use smaller model size or unload unused modules:
```python
manager.unload('video_gen_local')   # Free up GPU memory
manager.unload('threed_gen_local')  # More memory freed
```

### Check what's loaded
```python
print(manager.list_loaded())  # See all active modules
```
