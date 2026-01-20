# ForgeAI - Code Adventure Tour 🗺️

Your guide through the ForgeAI codebase!

Think of this like a "Choose Your Own Adventure" book. Follow the paths that interest you.

---

## 🚀 Where to Start

Your journey begins at: **`run.py`**

This file is the front door. Run it with one of these commands:

| Command | What it does |
|---------|--------------|
| `python run.py --gui` | Opens the graphical interface **(BEST FOR BEGINNERS)** |
| `python run.py --train` | Train your AI model |
| `python run.py --run` | Chat in terminal |
| `python run.py --serve` | Start REST API server |
| `python run.py --build` | Build new model from scratch |

---

## 🗂️ The Map

Here's how ForgeAI is organized:

```
run.py                   <-- You start here!
    │
    ▼
forge_ai/                <-- Main package folder
    ├── core/            The AI brain (models, inference)
    ├── gui/             The visual interface
    ├── memory/          Saves conversations
    ├── comms/           Networking and APIs
    ├── tools/           AI actions (web, files, vision)
    ├── avatar/          Virtual character
    ├── voice/           Speech (TTS, STT)
    └── modules/         Load/unload features
```

---

## 🧠 Chapter 1: The Brain (`forge_ai/core/`)

This folder contains the AI intelligence.

### `model.py`
The neural network itself. Called "Forge".

**What it does:**
- Takes numbers in, predicts next word
- Uses modern techniques: RoPE, RMSNorm, SwiGLU, GQA
- Optional Flash Attention (auto-detected, 2-4x faster on modern GPUs)

**Model sizes:**
| Size | Params | Best For |
|------|--------|----------|
| nano | ~1M | tiny devices, testing |
| micro | ~2M | Raspberry Pi |
| tiny | ~5M | light devices |
| small | ~27M | desktop default |
| medium | ~85M | good balance |
| large | ~300M | quality focus |
| xl+ | 1B+ | datacenter |

**How to use:**
```python
from forge_ai.core.model import create_model
model = create_model('small')
```

---

### `inference.py`
Generates text from the model.

**Main class:** `ForgeEngine`

**What it does:**
- Takes your text prompt
- Runs it through the model
- Returns the AI's response

**How to use:**
```python
from forge_ai.core.inference import ForgeEngine
engine = ForgeEngine()
response = engine.generate("Hello, my name is")
print(response)
```

---

### `tokenizer.py`
Converts text to numbers and back.

> Why? The AI only understands numbers, not letters.

```
"Hello world" --> [15496, 995]  (encode)
[15496, 995]  --> "Hello world" (decode)
```

**How to use:**
```python
from forge_ai.core.tokenizer import get_tokenizer
tokenizer = get_tokenizer()
tokens = tokenizer.encode("Hello")
text = tokenizer.decode(tokens)
```

---

### `training.py`
Teaches the model to be smarter.

**How to use:**
```bash
python run.py --train
```

Or in code:
```python
from forge_ai.core.training import train_model
train_model("data/training.txt")
```

---

### `tool_router.py`
Figures out what the user wants and sends it to the right place.

**Example:**
- User says "Draw me a cat" → Router sees "draw" → sends to IMAGE generator
- User says "Write Python code" → Router sees "code" → sends to CODE generator

**Built-in tools:**
| Tool | Purpose |
|------|---------|
| chat | normal conversation |
| image | generate pictures |
| code | write programs |
| video | make videos |
| audio | text-to-speech, music |
| 3d | 3D models |

---

## 🖥️ Chapter 2: The Interface (`forge_ai/gui/`)

This folder contains the visual application.

### `enhanced_window.py`
The main application window.

**Class:** `EnhancedMainWindow`

**Contains:**
- First-time setup wizard
- Tab system for all features
- Model selection
- Theme switching

### Tabs (`forge_ai/gui/tabs/`)

| File | Purpose |
|------|---------|
| `chat_tab.py` | Talk to the AI |
| `image_tab.py` | Generate images |
| `code_tab.py` | Generate code |
| `video_tab.py` | Generate videos |
| `audio_tab.py` | Text-to-speech |
| `threed_tab.py` | Generate 3D models |
| `training_tab.py` | Train your model |
| `modules_tab.py` | Turn features on/off |
| `avatar_tab.py` | Control your avatar |
| `settings_tab.py` | App settings |

---

## 💾 Chapter 3: Memory (`forge_ai/memory/`)

Saves conversations so the AI remembers things.

### `manager.py`
Saves and loads chat history.

**Class:** `ConversationManager`

**How to use:**
```python
from forge_ai.memory.manager import ConversationManager
manager = ConversationManager()
manager.save_conversation("my_chat", messages)
data = manager.load_conversation("my_chat")
```

Conversations saved to: `data/conversations/*.json`

---

### `vector_db.py`
Searches memories by meaning, not just keywords.

**Example:**
- Search for "feline pets"
- Finds "Tell me about cats" (similar meaning!)

**Class:** `SimpleVectorDB`

---

## 🌐 Chapter 4: Networking (`forge_ai/comms/`)

Connect your AI to the internet or other devices.

### `api_server.py`
Creates a REST API so other programs can use your AI.

**Start it:**
```bash
python run.py --serve
```

**Then access:**
- `http://localhost:5000/health` - Check if running
- `http://localhost:5000/generate` - Send prompts (POST)

---

### `network.py`
Connect multiple computers running ForgeAI.

**Class:** `ForgeNode`

**Use cases:**
- Share models between machines
- Run the brain on a server, UI on a tablet

---

## 🔧 Chapter 5: Tools (`forge_ai/tools/`)

Actions the AI can perform.

### `tool_executor.py`
Runs AI tool calls safely.

**What it does:**
1. Parses the AI's request
2. Validates parameters
3. Runs the tool
4. Returns results

**Security features:**
- Timeout protection
- Blocked file paths
- Parameter validation

### Other tools:
| File | Purpose |
|------|---------|
| `vision.py` | See images and screens |
| `web_tools.py` | Search the web, fetch pages |
| `file_tools.py` | Read and write files |
| `browser_tools.py` | Automate web browsers |

---

## 🎭 Chapter 6: Avatar (`forge_ai/avatar/`)

Create a virtual character that reacts and moves.

### `controller.py`
Main avatar control.

**Class:** `AvatarController`

**How to use:**
```python
from forge_ai.avatar import get_avatar
avatar = get_avatar()
avatar.enable()
avatar.set_expression("happy")
```

---

### `autonomous.py`
Makes the avatar act on its own!

**Class:** `AutonomousAvatar`

**Features:**
- Watches your screen and reacts
- Random idle animations
- Mood system (happy, bored, curious, etc.)

**How to use:**
```python
from forge_ai.avatar.autonomous import AutonomousAvatar
auto = AutonomousAvatar(avatar)
auto.start()   # Avatar does its own thing!
auto.stop()    # Back to manual control
```

---

## ⚙️ Chapter 7: Modules (`forge_ai/modules/`)

Turn features on and off like switches.

**Why this exists:**
- Save memory by only loading what you need
- Prevent conflicts between features
- Works on devices from Raspberry Pi to servers

### `manager.py`
Loads and unloads modules.

**Class:** `ModuleManager`

**How to use:**
```python
from forge_ai.modules import ModuleManager

manager = ModuleManager()
manager.load('model')
manager.load('tokenizer')
manager.load('inference')

# Use the module
engine = manager.get_interface('inference')

# Free memory when done
manager.unload('inference')
```

---

### `registry.py`
Lists all available modules.

| Module | Purpose |
|--------|---------|
| `model` | The neural network |
| `tokenizer` | Text to numbers |
| `inference` | Generate text |
| `image_gen_local` | Stable Diffusion (local) |
| `image_gen_api` | DALL-E (cloud) |
| `code_gen_local` | Local code generation |
| `code_gen_api` | GPT-4 code (cloud) |
| `memory` | Conversation storage |
| `voice_input` | Speech-to-text |
| `voice_output` | Text-to-speech |
| `avatar` | Virtual character |

> ⚠️ **Important:** Some modules conflict!
> - `image_gen_local` and `image_gen_api` can't run together
> - `code_gen_local` and `code_gen_api` can't run together

---

## 🔊 Chapter 8: Voice (`forge_ai/voice/`)

Let the AI speak and listen.

### `voice_generator.py`
Creates voice settings based on personality.

**Class:** `AIVoiceGenerator`

**Personality affects voice:**
| Trait | Voice Effect |
|-------|--------------|
| High confidence | lower pitch, slower |
| High playfulness | varied pitch, faster |
| High formality | neutral, measured |

### Other files:
| File | Purpose |
|------|---------|
| `tts_simple.py` | Basic text-to-speech |
| `listener.py` | Listen to microphone |
| `lip_sync.py` | Sync avatar mouth to speech |

---

## 📖 Quick Reference: What to Read First

**Understand the AI brain:**
1. `forge_ai/core/model.py`
2. `forge_ai/core/inference.py`

**Build a chat app:**
1. `forge_ai/gui/tabs/chat_tab.py`
2. `forge_ai/core/inference.py`

**Add a new feature:**
1. `forge_ai/modules/registry.py`
2. `forge_ai/modules/manager.py`

**Create an API:**
1. `forge_ai/comms/api_server.py`

**Make an avatar:**
1. `forge_ai/avatar/controller.py`
2. `forge_ai/avatar/autonomous.py`

---

## ⚠️ Things to Avoid

**DO NOT:**
- Load conflicting modules (both `image_gen_local` AND `image_gen_api`)
- Train with less than 1000 lines of data
- Use very high learning rates (stick with 0.0001)
- Delete model folders manually (use the Model Manager)
- Train while chatting (close chat first)
- Run multiple instances on the same model (can corrupt files)

---

**Happy coding! 🚀**
