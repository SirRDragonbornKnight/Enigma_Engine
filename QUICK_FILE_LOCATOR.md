# Enigma AI Engine - Quick File Locator 🔍

Find the file you need in seconds!

---

## 💬 I want to CHAT with the AI

| What | File |
|------|------|
| Start | `python run.py --gui` (or `--run` for terminal) |
| Engine | `enigma_engine/core/inference.py` |
| Model | `enigma_engine/core/model.py` |
| GUI | `enigma_engine/gui/tabs/chat_tab.py` |

---

## 🎨 I want to GENERATE IMAGES

| What | File |
|------|------|
| Tab | `enigma_engine/gui/tabs/image_tab.py` |
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
| Tab | `enigma_engine/gui/tabs/code_tab.py` |
| Router | `enigma_engine/core/tool_router.py` |

---

## 🎬 I want to GENERATE VIDEOS

| What | File |
|------|------|
| Tab | `enigma_engine/gui/tabs/video_tab.py` |
| Output | `outputs/videos/` |

---

## 🔊 I want TEXT-TO-SPEECH

| What | File |
|------|------|
| Voice | `enigma_engine/voice/voice_generator.py` |
| Simple | `enigma_engine/voice/tts_simple.py` |
| Tab | `enigma_engine/gui/tabs/audio_tab.py` |

---

## 📚 I want to TRAIN the model

| What | File |
|------|------|
| Command | `python run.py --train` |
| Trainer | `enigma_engine/core/training.py` |
| Data | `data/training.txt` |
| Tab | `enigma_engine/gui/tabs/training_tab.py` |

---

## 🤖 I want to create an AVATAR

| What | File |
|------|------|
| Control | `enigma_engine/avatar/controller.py` |
| Autonomous | `enigma_engine/avatar/autonomous.py` |
| Desktop | `enigma_engine/avatar/desktop_pet.py` |
| Tab | `enigma_engine/gui/tabs/avatar_tab.py` |

---

## ⚙️ I want to LOAD/UNLOAD modules

| What | File |
|------|------|
| Manager | `enigma_engine/modules/manager.py` |
| Registry | `enigma_engine/modules/registry.py` |
| Tab | `enigma_engine/gui/tabs/modules_tab.py` |
| Config | `data/module_config.json` |

---

## 💾 I want to SAVE/LOAD conversations

| What | File |
|------|------|
| Manager | `enigma_engine/memory/manager.py` |
| Storage | `data/conversations/*.json` |
| Search | `enigma_engine/memory/vector_db.py` |

---

## 🌐 I want to create an API SERVER

| What | File |
|------|------|
| Command | `python run.py --serve` |
| Server | `enigma_engine/comms/api_server.py` |
| Network | `enigma_engine/comms/network.py` |

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
| Executor | `enigma_engine/tools/tool_executor.py` |
| Definitions | `enigma_engine/tools/tool_definitions.py` |
| Router | `enigma_engine/core/tool_router.py` |

**Built-in tools:**
| File | Purpose |
|------|---------|
| `vision.py` | See images/screens |
| `web_tools.py` | Browse web |
| `file_tools.py` | Read/write files |
| `browser_tools.py` | Automate browsers |

**GUI Control tools:**
| Tool | Purpose |
|------|---------|
| `switch_tab` | Navigate GUI tabs |
| `adjust_setting` | Change preferences |
| `get_setting` | Read preferences |
| `manage_conversation` | Save/load/list chats |
| `show_help` | Display contextual help |
| `optimize_for_hardware` | Auto-configure for system |

**GUI State Manager:** `enigma_engine/gui/gui_state.py`

---

## ⚡ I want to CONFIGURE settings

| What | File |
|------|------|
| Config | `enigma_engine/config/` |
| Paths | `enigma_engine/__init__.py` |
| GUI | `data/gui_settings.json` |
| AI | `data/ai_self_config.json` |

---

## 📂 Folder Overview

```
enigma_engine/
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
from enigma_engine.core.model import Enigma, create_model, get_model, register_model
from enigma_engine.core.inference import EnigmaEngine
from enigma_engine.core.tokenizer import get_tokenizer, TokenizerProtocol, encode_text, decode_tokens
from enigma_engine.core.training import train_model
from enigma_engine.core.tool_router import ToolRouter, get_router

# Modules
from enigma_engine.modules import ModuleManager

# Memory
from enigma_engine.memory.manager import ConversationManager
from enigma_engine.memory.vector_db import SimpleVectorDB

# Voice
from enigma_engine.voice.voice_generator import AIVoiceGenerator

# Avatar
from enigma_engine.avatar import get_avatar
from enigma_engine.avatar.autonomous import AutonomousAvatar

# Network
from enigma_engine.comms.api_server import create_api_server

# Config
from enigma_engine.config import CONFIG
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
