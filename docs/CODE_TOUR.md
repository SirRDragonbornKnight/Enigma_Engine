# ForgeAI - Code Tour Guide

**Your map through the ForgeAI realm.** This guide helps you navigate the codebase and find any feature's code in under 5 minutes.

> *"Every great adventure begins with understanding the terrain. This map reveals the paths through ForgeAI's architecture - from the humble entry point to the deepest neural caverns."*

---

## Quick Navigation

| I want to... | Go to... |
|-------------|----------|
| Start the application | [run.py](../run.py) |
| Understand the AI model | [forge_ai/core/model.py](../forge_ai/core/model.py) |
| See how training works | [forge_ai/core/training.py](../forge_ai/core/training.py) |
| Modify the chat interface | [forge_ai/gui/tabs/chat_tab.py](../forge_ai/gui/tabs/chat_tab.py) |
| Add a new tool | [forge_ai/tools/tool_definitions.py](../forge_ai/tools/tool_definitions.py) |
| Change configuration | [forge_ai/config/defaults.py](../forge_ai/config/defaults.py) |
| Load/unload modules | [forge_ai/modules/manager.py](../forge_ai/modules/manager.py) |

---

## The Realm Map - Project Structure

```
forge_ai/
├── run.py                  # THE GATEWAY - All journeys begin here
├── requirements.txt        # The supply manifest
├── setup.py               # Installation enchantments
│
├── forge_ai/                # === THE KINGDOM ===
│   ├── __init__.py        # Royal proclamations (exports)
│   ├── config/            # The Chamber of Configuration
│   │
│   ├── core/              # THE BRAIN - Neural architecture
│   ├── modules/           # THE ARMORY - Loadable capabilities  
│   ├── memory/            # THE VAULT - Storage & recall
│   ├── comms/             # THE MESSENGER - Networking
│   ├── gui/               # THE THRONE ROOM - User interface
│   ├── voice/             # THE HERALD - Speech
│   ├── avatar/            # THE COMPANION - Visual character
│   ├── tools/             # THE WORKSHOP - AI capabilities
│   └── utils/             # THE TOOLSHED - Helpers
│
├── data/                  # Training scrolls
├── models/                # Trained minds
├── docs/                  # Ancient wisdom
└── examples/              # Practice quests
```

---

## The Gateway: run.py

**File:** [run.py](../run.py)

Every adventure begins at the gateway. This simple sentinel routes travelers to their destination:

```bash
python run.py --train    # Enter the Training Grounds
python run.py --run      # Speak with the Oracle (CLI)
python run.py --gui      # Enter the Throne Room (GUI)
python run.py --serve    # Summon the API Server
```

**What it does:**
1. Parses command-line arguments
2. Calls the appropriate module (training, inference, GUI, or API)

---

## The Brain: forge_ai/core/

*Deep within ForgeAI's fortress lies the Brain - sacred chambers where neural networks learn to think. Here, mathematics becomes magic, and patterns become understanding.*

### model.py - The Neural Architecture
**File:** [forge_ai/core/model.py](../forge_ai/core/model.py)

Contains `Forge`, a production-grade transformer language model - the very mind of your AI.

**Key Features:**
- **RoPE (Rotary Position Embeddings)** - Better position awareness than learned embeddings
- **RMSNorm** - More stable than LayerNorm, faster training
- **SwiGLU Activation** - Better than GELU for transformers
- **GQA (Grouped Query Attention)** - Memory efficient attention
- **KV-Cache** - Fast autoregressive generation
- **Flash Attention** - Optional 2-4x speedup (auto-detected, requires CUDA + fp16)
- **Pre-norm Architecture** - More stable training
- **Weight Tying** - Embedding and output head share weights
- **Thread-Safe Model Registry** - Safe concurrent access from GUI + API
- **Cached Causal Mask** - Pre-computed attention mask for faster inference
- **Vectorized Repetition Penalty** - O(vocab) instead of O(n²) for fast generation

**Architecture Components:**
```python
Forge(
    vocab_size=32000,    # Vocabulary size
    dim=256,             # Hidden dimension
    depth=6,             # Number of transformer layers
    heads=8,             # Attention heads
    max_len=2048,        # Max sequence length
    ff_mult=4.0,         # FFN hidden multiplier
)
```

**Backwards Compatibility:** `TinyEnigma` is an alias for `Enigma`.

### tokenizer.py - Text ↔ Numbers
**File:** [forge_ai/core/tokenizer.py](../forge_ai/core/tokenizer.py)

Converts human text into numbers the AI can understand.
- Full character-level tokenizer (every character gets a number)
- Includes a dictionary for ~3000 common words
- Custom dictionary support (`forge_ai/vocab_model/dictionary.txt`)
- Handles special tokens (padding, start, end, unknown)
- **TokenizerProtocol** - Type-safe interface for any tokenizer implementation
- **Helper Functions** - `encode_text()`, `decode_tokens()`, `get_vocab_size()`, `get_special_token_ids()`

### training.py - Teaching the AI
**File:** [forge_ai/core/training.py](../forge_ai/core/training.py)

Production-ready training loop:
- **Mixed Precision (AMP)** - Faster training on GPUs
- **Gradient Accumulation** - Large effective batch sizes
- **Cosine LR Schedule** - With warmup
- **Gradient Clipping** - Prevents exploding gradients
- **Checkpointing** - Save best and periodic checkpoints

### inference.py - Getting Responses
**File:** [forge_ai/core/inference.py](../forge_ai/core/inference.py)

`ForgeEngine` class - the main way to interact with the AI:
```python
engine = ForgeEngine()
response = engine.generate("Hello!", temperature=0.8, top_p=0.9)

# Streaming generation
for token in engine.stream_generate("Tell me a story"):
    print(token, end="")

# Chat with history
response = engine.chat("What's the weather?", history=[...])
```

### model_registry.py - Managing Multiple AIs
**File:** [forge_ai/core/model_registry.py](../forge_ai/core/model_registry.py)

Create and manage named AI personalities:
```python
registry = ModelRegistry()
registry.create_model("luna", size="medium")
registry.create_model("nova", size="large")
```

### model_config.py - Size Presets
**File:** [forge_ai/core/model_config.py](../forge_ai/core/model_config.py)

Defines model sizes from mobile to server-scale:
- `tiny`: ~2M params - Raspberry Pi, mobile, testing
- `small`: ~15M params - Laptop, low-end GPU
- `medium`: ~50M params - Mid-range GPU
- `large`: ~125M params - Like GPT-2 small
- `xl`: ~350M params - Like GPT-2 medium
- `xxl`: ~770M params - Like GPT-2 large
- `xxxl`: ~1.5B params - Like GPT-2 XL

### trainer.py - The Training Grounds
**File:** [forge_ai/core/trainer.py](../forge_ai/core/trainer.py)

Full-featured `ForgeTrainer` class:
- Multi-GPU support (DataParallel)
- Mixed precision training
- Gradient accumulation
- Early stopping
- Learning rate scheduling
- Comprehensive logging

### model_scaling.py - The Growth Chamber
**File:** [forge_ai/core/model_scaling.py](../forge_ai/core/model_scaling.py)

Scale models up or down:
- `grow_model()` - Expand small → large while preserving learning
- `shrink_model()` - Compress for deployment
- `KnowledgeDistiller` - Train small model to mimic large model

### layers.py - The Building Blocks
**File:** [forge_ai/core/layers.py](../forge_ai/core/layers.py)

Additional neural network components:
- `FeedForward`, `GLU`, `GeGLU` - FFN variants
- `MultiQueryAttention` - Memory-efficient attention
- `GroupedQueryAttention` - Balance between MHA and MQA
- `SlidingWindowAttention` - For long sequences
- `LoRALayer` - Efficient fine-tuning
- `MixtureOfExperts` - Sparse scaling

### hardware.py - The Quartermaster
**File:** [forge_ai/core/hardware.py](../forge_ai/core/hardware.py)

Automatically detects your hardware and recommends optimal settings.

---

## The Vault: forge_ai/memory/

*In the depths of ForgeAI lies the Memory Vault - where conversations are preserved and wisdom is stored for future recall.*

### manager.py - The Archivist
**File:** [forge_ai/memory/manager.py](../forge_ai/memory/manager.py)

Saves conversations to JSON files:
```python
manager = ConversationManager()
manager.save_conversation("chat_with_bob", messages)
```

### memory_db.py - The Deep Archives
**File:** [forge_ai/memory/memory_db.py](../forge_ai/memory/memory_db.py)

Long-term memory storage with search capabilities.

### vector_utils.py - The Oracle's Eye
**File:** [forge_ai/memory/vector_utils.py](../forge_ai/memory/vector_utils.py)

Find similar memories using cosine similarity (vector math).

---

## The Messenger: forge_ai/comms/

*Across distant lands, the Messenger carries word between ForgeAI instances - linking PCs to Raspberry Pis, enabling AI-to-AI discourse.*

### api_server.py - The Embassy
**File:** [forge_ai/comms/api_server.py](../forge_ai/comms/api_server.py)

Flask server with endpoints:
- `POST /generate` - Get AI response
- `GET /health` - Check server status
- `GET /models` - List available models

### network.py - The Courier Network
**File:** [forge_ai/comms/network.py](../forge_ai/comms/network.py)

`ForgeNode` class for device-to-device communication:
- PC talks to Raspberry Pi
- AI-to-AI conversations
- Memory sync across devices

### discovery.py - The Scouts
**File:** [forge_ai/comms/discovery.py](../forge_ai/comms/discovery.py)

Auto-discover other ForgeAI instances on your network.

### mobile_api.py - The Swift Runners
**File:** [forge_ai/comms/mobile_api.py](../forge_ai/comms/mobile_api.py)

Optimized API for mobile apps.

---

## The Throne Room: forge_ai/gui/

*The grand Throne Room where users interact with their AI. PyQt5 provides the ornate decorations; the tabs are doors to different chambers.*

### main_window.py - The Simple Hall
**File:** [forge_ai/gui/main_window.py](../forge_ai/gui/main_window.py)

Simple PyQt5 window with:
- Chat tab
- Logbook (history) tab
- Training tab
- Avatar tab

### enhanced_window.py - The Grand Palace
**File:** [forge_ai/gui/enhanced_window.py](../forge_ai/gui/enhanced_window.py)

Advanced GUI with:
- Setup wizard for first run
- Model management
- Dark/Light themes
- Training data editor
- Vision preview
- Terminal output tab
- History per AI

### gui_state.py - The Royal Advisor
**File:** [forge_ai/gui/gui_state.py](../forge_ai/gui/gui_state.py)

Singleton manager allowing AI tools to control the GUI:
- `switch_tab()` - Navigate to different tabs
- `get_setting()` / `set_setting()` - Read/write user preferences
- `manage_conversation()` - Save/load/list conversations
- `get_help()` - Get contextual help content
- `optimize_for_hardware()` - Auto-configure settings

```python
from forge_ai.gui.gui_state import get_gui_state
gui = get_gui_state()
gui.switch_tab("image")
gui.set_setting("chat_zoom", 14)
```

---

## The Herald: forge_ai/voice/

*The Herald gives voice to thoughts - speaking aloud and listening for commands.*

### tts_simple.py - The Voice
**File:** [forge_ai/voice/tts_simple.py](../forge_ai/voice/tts_simple.py)

Makes the AI speak using pyttsx3:
```python
from forge_ai.voice import speak
speak("Hello, I am Forge!")
```

### stt_simple.py - The Ears
**File:** [forge_ai/voice/stt_simple.py](../forge_ai/voice/stt_simple.py)

Listen to user's voice:
```python
from forge_ai.voice import listen
text = listen(timeout=5)
```

---

## The Companion: forge_ai/avatar/

*A loyal companion that lives on your desktop - moving, expressing, and interacting with the world.*

### avatar_api.py - Simple Interface
**File:** [forge_ai/avatar/avatar_api.py](../forge_ai/avatar/avatar_api.py)

Simple avatar interface stub.

### controller.py - Full Avatar Control
**File:** [forge_ai/avatar/controller.py](../forge_ai/avatar/controller.py)

```python
avatar = get_avatar()
avatar.enable()
avatar.move_to(500, 300)
avatar.set_expression("happy")
avatar.speak("Hello!")
```

---

## The Workshop: forge_ai/tools/

*In the Workshop, the AI gains hands - tools to search, read, write, and interact with the world beyond its mind.*

### tool_registry.py - The Tool Rack
**File:** [forge_ai/tools/tool_registry.py](../forge_ai/tools/tool_registry.py)

Register and execute tools:
```python
result = execute_tool("web_search", query="Python tutorials")
```

### vision.py - The All-Seeing Eye
**File:** [forge_ai/tools/vision.py](../forge_ai/tools/vision.py)

AI can "see" the screen:
- Capture screenshots
- OCR (read text from images)
- Find elements on screen

### web_tools.py - The Web Weaver
**File:** [forge_ai/tools/web_tools.py](../forge_ai/tools/web_tools.py)

Search the web, fetch pages.

### file_tools.py - The Scribe's Quill
**File:** [forge_ai/tools/file_tools.py](../forge_ai/tools/file_tools.py)

Read, write, list, move files.

### document_tools.py - The Scholar's Library
**File:** [forge_ai/tools/document_tools.py](../forge_ai/tools/document_tools.py)

Read PDF, EPUB, DOCX files.

---

## The Great Circuit: How It All Flows

```
User Input
    │
    ▼
┌─────────────────┐
│   Tokenizer     │  Converts text → numbers
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Model         │  Neural network processes
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Tokenizer     │  Converts numbers → text
└────────┬────────┘
         │
         ▼
AI Response
```

---

## Quick Spells: Common Tasks

### Create a new AI
```python
from forge_ai.core.model_registry import ModelRegistry
registry = ModelRegistry()
registry.create_model("my_ai", size="small")
```

### Train your AI
```python
from forge_ai.core.training import train_model
train_model(force=True, num_epochs=10)
```

### Chat with your AI
```python
from forge_ai.core.inference import ForgeEngine
engine = ForgeEngine()
print(engine.generate("Hello!"))
```

### Run multiple AIs talking to each other
```python
from forge_ai.comms.network import ForgeNode
node = ForgeNode(name="my_node")
node.start_ai_conversation("other_ai", num_turns=5)
```

---

## The Web of Connections

```
run.py (The Gateway)
  └── forge_ai/core/training.py (--train)
  └── forge_ai/core/inference.py (--run)
  └── forge_ai/gui/enhanced_window.py (--gui)
  └── forge_ai/comms/api_server.py (--serve)

enhanced_window.py (The Throne Room)
  └── forge_ai/core/model_registry.py (model management)
  └── forge_ai/core/inference.py (chat)
  └── forge_ai/memory/manager.py (history)
  └── forge_ai/tools/vision.py (screen capture)
  └── forge_ai/avatar/controller.py (avatar)
  └── forge_ai/voice/ (TTS/STT)

api_server.py (The Embassy)
  └── forge_ai/core/inference.py (generate responses)
  └── forge_ai/core/model_registry.py (model info)
```

---

## Your Next Quest

1. **Read [HOW_TO_TRAIN.md](HOW_TO_TRAIN.md)** - Master the art of training
2. **Explore [examples/](../examples/)** - See working code in action
3. **Edit [data/training.txt](../data/training.txt)** - Add your training data
4. **Run `python run.py --gui`** - Begin your adventure!

> *"The path is now clear, traveler. May your models converge and your loss decrease!"*
