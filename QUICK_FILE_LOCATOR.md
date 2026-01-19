# ForgeAI - Quick File Locator 🔍

Find the file you need in seconds!

---

## 💬 I want to CHAT with the AI

| What | File |
|------|------|
| Start | `python run.py --gui` (or `--run` for terminal) |
| Engine | `forge_ai/core/inference.py` |
| Model | `forge_ai/core/model.py` |
| GUI | `forge_ai/gui/tabs/chat_tab.py` |

---

## 🎨 I want to GENERATE IMAGES

| What | File |
|------|------|
| Tab | `forge_ai/gui/tabs/image_tab.py` |
| Output | `outputs/images/` |

**Providers:**
| Provider | Notes |
|----------|-------|
| `PlaceholderImage` | Built-in, no GPU needed |
| `StableDiffusionLocal` | Local, needs GPU |
| `OpenAIImage` | DALL-E, needs API key |
| `ReplicateImage` | Cloud, needs API key |

---

## 💻 I want to GENERATE CODE

| What | File |
|------|------|
| Tab | `forge_ai/gui/tabs/code_tab.py` |
| Router | `forge_ai/core/tool_router.py` |

---

## 🎬 I want to GENERATE VIDEOS

| What | File |
|------|------|
| Tab | `forge_ai/gui/tabs/video_tab.py` |
| Output | `outputs/videos/` |

---

## 🔊 I want TEXT-TO-SPEECH

| What | File |
|------|------|
| Voice | `forge_ai/voice/voice_generator.py` |
| Simple | `forge_ai/voice/tts_simple.py` |
| Tab | `forge_ai/gui/tabs/audio_tab.py` |

---

## 📚 I want to TRAIN the model

| What | File |
|------|------|
| Command | `python run.py --train` |
| Trainer | `forge_ai/core/training.py` |
| Data | `data/training.txt` |
| Tab | `forge_ai/gui/tabs/training_tab.py` |

---

## 🤖 I want to create an AVATAR

| What | File |
|------|------|
| Control | `forge_ai/avatar/controller.py` |
| Autonomous | `forge_ai/avatar/autonomous.py` |
| Desktop | `forge_ai/avatar/desktop_pet.py` |
| Tab | `forge_ai/gui/tabs/avatar_tab.py` |

---

## ⚙️ I want to LOAD/UNLOAD modules

| What | File |
|------|------|
| Manager | `forge_ai/modules/manager.py` |
| Registry | `forge_ai/modules/registry.py` |
| Tab | `forge_ai/gui/tabs/modules_tab.py` |
| Config | `data/module_config.json` |

---

## 💾 I want to SAVE/LOAD conversations

| What | File |
|------|------|
| Manager | `forge_ai/memory/manager.py` |
| Storage | `data/conversations/*.json` |
| Search | `forge_ai/memory/vector_db.py` |

---

## 🌐 I want to create an API SERVER

| What | File |
|------|------|
| Command | `python run.py --serve` |
| Server | `forge_ai/comms/api_server.py` |
| Network | `forge_ai/comms/network.py` |

**Endpoints:**
| Method | URL | Purpose |
|--------|-----|---------|
| GET | `/health` | Check if running |
| GET | `/info` | Server info |
| POST | `/generate` | Send prompts |

---

## 🔧 I want to add TOOLS for the AI

| What | File |
|------|------|
| Executor | `forge_ai/tools/tool_executor.py` |
| Definitions | `forge_ai/tools/tool_definitions.py` |
| Router | `forge_ai/core/tool_router.py` |

**Built-in tools:**
| File | Purpose |
|------|---------|
| `vision.py` | See images/screens |
| `web_tools.py` | Browse web |
| `file_tools.py` | Read/write files |
| `browser_tools.py` | Automate browsers |

---

## ⚡ I want to CONFIGURE settings

| What | File |
|------|------|
| Config | `forge_ai/config/` |
| Paths | `forge_ai/__init__.py` |
| GUI | `data/gui_settings.json` |
| AI | `data/ai_self_config.json` |

---

## 📂 Folder Overview

```
forge_ai/
├── core/      🧠 AI brain (model, inference, training)
├── gui/       🖥️ User interface
│   └── tabs/  📑 All tab panels
├── memory/    💾 Storage and recall
├── comms/     🌐 Networking and API
├── tools/     🔧 AI capabilities
├── avatar/    🎭 Virtual character
├── voice/     🔊 Speech (TTS/STT)
├── modules/   ⚙️ Load/unload features
├── config/    ⚡ Configuration
└── builtin/   📦 Built-in fallbacks

data/
├── training.txt       📚 Training data
├── conversations/     💬 Saved chats
├── module_config.json ⚙️ Module settings
└── gui_settings.json  🎨 GUI preferences

models/
└── *.pth              🤖 Saved model weights

outputs/
├── images/            🖼️ Generated images
├── videos/            🎬 Generated videos
└── audio/             🔊 Generated audio
```

---

## 📦 Common Imports

```python
# AI Brain
from forge_ai.core.model import Forge, create_model
from forge_ai.core.inference import ForgeEngine
from forge_ai.core.tokenizer import get_tokenizer
from forge_ai.core.training import train_model
from forge_ai.core.tool_router import ToolRouter, get_router

# Modules
from forge_ai.modules import ModuleManager

# Memory
from forge_ai.memory.manager import ConversationManager
from forge_ai.memory.vector_db import SimpleVectorDB

# Voice
from forge_ai.voice.voice_generator import AIVoiceGenerator

# Avatar
from forge_ai.avatar import get_avatar
from forge_ai.avatar.autonomous import AutonomousAvatar

# Network
from forge_ai.comms.api_server import create_api_server

# Config
from forge_ai.config import CONFIG
```

---

## 🚀 Common Workflows

### 1. Start chatting
```
python run.py --gui → Chat Tab → Type message
```

### 2. Train custom AI
```
Edit data/training.txt → python run.py --train → Wait → Chat
```

### 3. Generate images
```
python run.py --gui → Image Tab → Enter prompt → Click Generate
```

### 4. Create avatar
```
python run.py --gui → Avatar Tab → Enable → Start Autonomous
```

### 5. Start API server
```
python run.py --serve → Access http://localhost:5000
```
