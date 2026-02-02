# ForgeAI Examples

This directory contains comprehensive examples for all ForgeAI features.

## Quick Start

```bash
# Run any example
python examples/complete_example.py      # Full integration demo
python examples/robot_example.py         # Robot control
python examples/game_example.py          # Game AI
```

## Example Index

### Complete Integration
| File | Description |
|------|-------------|
| [complete_example.py](complete_example.py) | Full application showing all components together |

### Core AI
| File | Description |
|------|-------------|
| [inference_example.py](inference_example.py) | Text generation, streaming, tool routing |
| [training_example.py](training_example.py) | Train models from nano to omega sizes |
| [modules_example.py](modules_example.py) | Module system, dependencies, conflicts |

### Generation Capabilities
| File | Description |
|------|-------------|
| [image_gen_example.py](image_gen_example.py) | Stable Diffusion, DALL-E, image generation |
| [video_example.py](video_example.py) | Video generation with AnimateDiff |
| [audio_example.py](audio_example.py) | TTS, voice synthesis, audio generation |
| [threed_example.py](threed_example.py) | 3D model generation (Shap-E) |
| [voice_clone_example.py](voice_clone_example.py) | Voice cloning and conversion |

### Memory and Context
| File | Description |
|------|-------------|
| [memory_example.py](memory_example.py) | Conversations, vector DB, RAG, embeddings |

### Perception
| File | Description |
|------|-------------|
| [vision_example.py](vision_example.py) | Image analysis, OCR, screenshot analysis |
| [camera_example.py](camera_example.py) | Webcam capture and real-time analysis |

### Tools
| File | Description |
|------|-------------|
| [web_tools_example.py](web_tools_example.py) | Web search, scraping, summarization |
| [file_tools_example.py](file_tools_example.py) | File operations, search, watching |
| [document_example.py](document_example.py) | PDF, DOCX, Markdown processing |
| [iot_example.py](iot_example.py) | Home Assistant, smart devices, sensors |

### Robot and Game Control
| File | Description |
|------|-------------|
| [robot_example.py](robot_example.py) | Servo control, animatronics, safety |
| [game_example.py](game_example.py) | Game AI, mod integration, strategies |

### Avatar and Interface
| File | Description |
|------|-------------|
| [avatar_example.py](avatar_example.py) | Desktop pet, lip sync, emotions |
| [networking_example.py](networking_example.py) | API server, multi-device networking |

## Example Structure

Each example follows this pattern:

```python
# 1. Simulated classes (for standalone testing)
class SimulatedClass:
    """Works without ForgeAI installed"""
    pass

# 2. Real ForgeAI imports (commented)
# from forge_ai.module import RealClass

# 3. Example functions
def example_basic_usage():
    """Demonstrates basic usage"""
    pass

# 4. Main with all examples
if __name__ == "__main__":
    example_basic_usage()
```

## Running Examples

### Without ForgeAI Installed
Examples use simulated classes by default:
```bash
python examples/robot_example.py
```

### With ForgeAI Installed
Uncomment the real imports:
```python
# Change from:
# from forge_ai.tools.robot_tools import RobotInterface

# To:
from forge_ai.tools.robot_tools import RobotInterface
```

## Component Map

```
ForgeAI Components:
├── Core
│   ├── model.py        → Training, inference
│   ├── tokenizer.py    → Text processing  
│   └── inference.py    → Generation engine
├── Modules
│   ├── manager.py      → Module loading
│   └── registry.py     → Available modules
├── Memory
│   ├── manager.py      → Conversation storage
│   └── vector_db.py    → Semantic search
├── Voice
│   ├── listener.py     → Speech-to-text
│   └── generator.py    → Text-to-speech
├── Avatar
│   ├── desktop_pet.py  → Desktop companion
│   └── lip_sync.py     → Animation sync
├── GUI Tabs (Generation)
│   ├── image_tab.py    → Image generation
│   ├── video_tab.py    → Video generation
│   ├── audio_tab.py    → Audio/TTS
│   ├── code_tab.py     → Code generation
│   └── threed_tab.py   → 3D generation
├── Tools
│   ├── web_tools.py    → Internet access
│   ├── file_tools.py   → File operations
│   ├── robot_tools.py  → Hardware control
│   └── game_router.py  → Game integration
└── Comms
    ├── api_server.py   → REST API
    └── network.py      → Multi-device
```

## Hardware Requirements

| Use Case | RAM | GPU | Notes |
|----------|-----|-----|-------|
| Text chat (nano/micro) | 512MB | None | Raspberry Pi Zero |
| Text chat (small) | 2GB | None | Raspberry Pi 4 |
| Voice assistant | 4GB | None | + microphone/speaker |
| Image generation | 8GB | 6GB VRAM | Stable Diffusion |
| Video generation | 16GB | 12GB VRAM | AnimateDiff |
| Robot control | 1GB | None | + servo controller |
| Full desktop app | 8GB | Optional | All features |

## Getting Help

- **Documentation**: See `/docs` directory
- **Quick Start**: See `GETTING_STARTED.md` in root
- **Troubleshooting**: See `TROUBLESHOOTING.md` in root
- **GUI Guide**: `python run.py --gui` and explore tabs
