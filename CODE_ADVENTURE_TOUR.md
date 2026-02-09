# Enigma AI Engine - Code Adventure Tour

**Your epic journey through the Enigma AI Engine codebase!**

Think of this like a "Choose Your Own Adventure" book. Each chapter corresponds
to a major part of the system. The files themselves contain detailed chapter
comments - this guide shows you the path between them.

---

## The Story So Far...

Enigma AI Engine is a kingdom of modular AI components. Everything is optional, everything
connects, and YOU decide what to enable. From a tiny Raspberry Pi to a massive
datacenter, Enigma AI Engine adapts.

---

## Starting Your Adventure

Your journey begins at: **`run.py`** (The Front Gate)

| Command | Destination | What Happens |
|---------|-------------|--------------|
| `python run.py --gui` | The Castle (Chapter 5) | Visual interface opens |
| `python run.py --train` | The Training Grounds | Teach your AI |
| `python run.py --run` | The Oracle (Chapter 2) | Chat in terminal |
| `python run.py --serve` | The Network Tower | Start REST API |

---

## Chapter Guide

### Chapter 1: The Forge (`enigma_engine/core/model.py`)
*"Where minds are born"*

The neural network itself. This is where the AI "thinking" happens - embeddings,
attention, and all the matrix math that makes intelligence emerge from numbers.

**Key concepts:** RoPE, RMSNorm, SwiGLU, GQA, Flash Attention

---

### Chapter 2: The Oracle (`enigma_engine/core/inference.py`)
*"Speaking with your AI"*

The Forge is a brain in a jar. The Oracle (EnigmaEngine) is how you TALK to it.
Takes your text, runs it through the brain, returns intelligent responses.

**Key concepts:** Tokenization, sampling, temperature, streaming

---

### Chapter 3: The Dispatcher (`enigma_engine/core/tool_router.py`)
*"Every request finds its destination"*

When you say "draw a cat" vs "write a poem", how does Enigma AI Engine know the
difference? The tool router analyzes your intent and sends you to the
right specialist.

**Routes to:** Image, Code, Video, Audio, 3D, Chat

---

### Chapter 4: The Training Grounds (`enigma_engine/core/training.py`)
*"Where AIs grow stronger"*

Teaching your AI new things. Load training data, run epochs, watch the
loss decrease as your AI learns.

---

### Chapter 5: The Castle (`enigma_engine/gui/enhanced_window.py`)
*"Your command center"*

The main GUI window. Every button, every tab, every theme change flows
through here. The largest file because it connects EVERYTHING.

---

### Chapter 6: The Art Studio (`enigma_engine/gui/tabs/image_tab.py`)
*"Painting with AI"*

Generate images from text descriptions. Supports placeholder art, Stable
Diffusion, DALL-E 3, and Replicate providers.

---

### Chapter 7: The Library (`enigma_engine/memory/manager.py`)
*"Where memories are preserved"*

Conversation storage. Your AI remembers past chats, can search by meaning
(vector search), and learns from history.

---

### Chapter 8: The Network Tower (`enigma_engine/comms/api_server.py`)
*"Connecting to the world"*

REST API server. Let other programs talk to your AI. Multi-device networking.
Remote access capabilities.

---

### The Armory (`enigma_engine/modules/`)
*"Toggle any capability"*

The module system. Load and unload features as needed. Prevents conflicts,
manages dependencies, adapts to your hardware.

---

## The Map

---

## 🖥️ Chapter 2: The Interface (`enigma_engine/gui/`)

This folder contains the visual application.

### `enhanced_window.py`
The main application window.

**Class:** `EnhancedMainWindow`

**Contains:**
- First-time setup wizard
- Tab system for all features
- Model selection
- Theme switching

### Tabs (`enigma_engine/gui/tabs/`)

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

## 💾 Chapter 3: Memory (`enigma_engine/memory/`)

Saves conversations so the AI remembers things.

### `manager.py`
Saves and loads chat history.

**Class:** `ConversationManager`

**How to use:**
```python
from enigma_engine.memory.manager import ConversationManager
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

## 🌐 Chapter 4: Networking (`enigma_engine/comms/`)

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
Connect multiple computers running Enigma AI Engine.

**Class:** `ForgeNode`

**Use cases:**
- Share models between machines
- Run the brain on a server, UI on a tablet

---

## 🔧 Chapter 5: Tools (`enigma_engine/tools/`)

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

## 🎭 Chapter 6: Avatar (`enigma_engine/avatar/`)

Create a virtual character that reacts and moves.

### `controller.py`
Main avatar control.

**Class:** `AvatarController`

**How to use:**
```python
from enigma_engine.avatar import get_avatar
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
from enigma_engine.avatar.autonomous import AutonomousAvatar
auto = AutonomousAvatar(avatar)
auto.start()   # Avatar does its own thing!
auto.stop()    # Back to manual control
```

---

## ⚙️ Chapter 7: Modules (`enigma_engine/modules/`)

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
from enigma_engine.modules import ModuleManager

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

## 🔊 Chapter 8: Voice (`enigma_engine/voice/`)

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

## 🎭 Chapter 9: Avatar Control (`enigma_engine/avatar/`)

Make your AI control a virtual character with natural body language.

### Priority System (How Control Works)

Avatar control uses a **priority hierarchy** to prevent conflicts:

```
BONE_ANIMATION (100)  ← PRIMARY: Direct bone control for rigged 3D models
     ↓ blocks
USER_MANUAL (80)      ← User dragging/clicking avatar
     ↓ blocks
AI_TOOL_CALL (70)     ← AI explicit commands via tools
     ↓ blocks
AUTONOMOUS (50)       ← Background behaviors (FALLBACK)
     ↓ blocks
IDE_ANIMATION (30)    ← Subtle idle movements
     ↓ blocks
FALLBACK (10)         ← Last resort
```

**Key Point:** Bone animation is PRIMARY - other systems are fallbacks.

### `bone_control.py`
Direct bone manipulation for rigged 3D models.

**Class:** `BoneController`

**What it does:**
- Detects bones automatically when model loads
- Controls bones with priority 100 (highest)
- Allows natural gestures (nod, wave, point, etc.)

**How to use:**
```python
from enigma_engine.avatar.bone_control import get_bone_controller

controller = get_bone_controller()
controller.move_bone("head", pitch=15, yaw=0, roll=0)  # Nod
```

### `ai_control.py`
Parses AI bone commands from responses.

**What it does:**
- Reads `<bone_control>` tags from AI output
- Executes predefined gestures
- Formats: `<bone_control>head|pitch=15,yaw=0,roll=0</bone_control>`

**Predefined gestures:**
- `nod`, `shake`, `wave`, `shrug`, `point`, `thinking`, `bow`, `stretch`

### `controller.py`
Main avatar controller with priority system.

**Class:** `AvatarController`

**Methods:**
- `request_control(requester, priority, duration)` - Request control
- `release_control(requester)` - Release control
- `current_controller` - Check who's in control

### `autonomous.py`
Fallback behaviors when bone control isn't active.

**Class:** `AutonomousAvatar`

**What it does:**
- Background idle behaviors (priority 50)
- Screen watching and reactions
- Only active when higher priorities aren't

### Avatar Tool Integration

**Tool name:** `control_avatar_bones`

**AI can call it like this:**
```json
{
  "action": "gesture",
  "gesture_name": "wave"
}
```

**Files:**
- `enigma_engine/tools/avatar_control_tool.py` - Tool definition
- `enigma_engine/tools/tool_executor.py` - Executes tool calls

---

## 📖 Quick Reference: What to Read First

**Understand the AI brain:**
1. `enigma_engine/core/model.py`
2. `enigma_engine/core/inference.py`

**Build a chat app:**
1. `enigma_engine/gui/tabs/chat_tab.py`
2. `enigma_engine/core/inference.py`

**Add a new feature:**
1. `enigma_engine/modules/registry.py`
2. `enigma_engine/modules/manager.py`

**Create an API:**
1. `enigma_engine/comms/api_server.py`

**Make an avatar:**
1. `enigma_engine/avatar/controller.py` - Priority system
2. `enigma_engine/avatar/bone_control.py` - PRIMARY control
3. `enigma_engine/avatar/ai_control.py` - AI command parsing
4. `enigma_engine/avatar/autonomous.py` - Fallback behaviors

**Train avatar AI:**
1. `data/specialized/avatar_control_training.txt` - Training examples
2. `scripts/train_avatar_control.py` - One-command training

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

**Happy coding!**

---

## Code Quality Status (Feb 2026)

The codebase has been reviewed for common issues:

**Completed:**
- All subprocess calls have timeouts (prevents hangs)
- All HTTP requests have timeouts (prevents network hangs)
- All history/cache lists have size limits (prevents memory leaks)
- Division by zero checks in calculations
- File handles properly closed with try/finally

**Module Health:**
| Category | Status |
|----------|--------|
| core/ | Clean - 24 fixes applied |
| gui/ | Clean - 8 fixes applied |
| voice/ | Clean - 15 fixes applied |
| utils/ | Clean - 4 fixes applied |
| hub/ | Clean - 2 fixes applied |

See `SUGGESTIONS.md` for full details.
