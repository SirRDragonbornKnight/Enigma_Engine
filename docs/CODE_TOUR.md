# ForgeAI - Code Tour Guide

Welcome! This guide walks you through the codebase so you understand how everything works together.

---

## 🏗️ Project Structure Overview

```
forge_ai/
├── run.py                  # 🚀 MAIN ENTRY POINT - Start here!
├── requirements.txt        # Dependencies to install
├── setup.py               # Package installation config
│
├── forge_ai/                # === THE MAIN PACKAGE ===
│   ├── __init__.py        # Package exports
│   ├── config.py          # Global configuration
│   │
│   ├── core/              # 🧠 THE BRAIN
│   ├── memory/            # 💾 STORAGE & RECALL
│   ├── comms/             # 🌐 NETWORKING
│   ├── gui/               # 🖥️ USER INTERFACE
│   ├── voice/             # 🔊 SPEECH
│   ├── avatar/            # 🤖 VISUAL CHARACTER
│   ├── tools/             # 🔧 AI CAPABILITIES
│   └── utils/             # 🛠️ HELPERS
│
├── data/                  # Training data files
├── models/                # Saved AI models
├── docs/                  # Documentation
└── examples/              # Example scripts
```

---

## 🚀 Entry Point: run.py

**File:** [run.py](../run.py)

This is where everything starts. It's a simple command-line interface:

```bash
python run.py --train    # Train the AI
python run.py --run      # Chat in terminal
python run.py --gui      # Open graphical interface
python run.py --serve    # Start API server
```

**What it does:**
1. Parses command-line arguments
2. Calls the appropriate module (training, inference, GUI, or API)

---

## 🧠 Core Package: forge_ai/core/

This is the **brain** of the AI - where all the neural network magic happens.

### model.py - The Neural Network Architecture
**File:** [forge_ai/core/model.py](../forge_ai/core/model.py)

Contains `Enigma`, a production-grade transformer language model.

**Key Features:**
- **RoPE (Rotary Position Embeddings)** - Better position awareness than learned embeddings
- **RMSNorm** - More stable than LayerNorm, faster training
- **SwiGLU Activation** - Better than GELU for transformers
- **KV-Cache** - Fast autoregressive generation
- **Pre-norm Architecture** - More stable training
- **Weight Tying** - Embedding and output head share weights

**Architecture Components:**
```python
Enigma(
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

### trainer.py - Advanced Training
**File:** [forge_ai/core/trainer.py](../forge_ai/core/trainer.py)

Full-featured `ForgeTrainer` class:
- Multi-GPU support (DataParallel)
- Mixed precision training
- Gradient accumulation
- Early stopping
- Learning rate scheduling
- Comprehensive logging

### model_scaling.py - Grow Your AI
**File:** [forge_ai/core/model_scaling.py](../forge_ai/core/model_scaling.py)

Scale models up or down:
- `grow_model()` - Expand small → large while preserving learning
- `shrink_model()` - Compress for deployment
- `KnowledgeDistiller` - Train small model to mimic large model

### layers.py - Building Blocks
**File:** [forge_ai/core/layers.py](../forge_ai/core/layers.py)

Additional neural network components:
- `FeedForward`, `GLU`, `GeGLU` - FFN variants
- `MultiQueryAttention` - Memory-efficient attention
- `GroupedQueryAttention` - Balance between MHA and MQA
- `SlidingWindowAttention` - For long sequences
- `LoRALayer` - Efficient fine-tuning
- `MixtureOfExperts` - Sparse scaling

### hardware.py - Auto-Detection
**File:** [forge_ai/core/hardware.py](../forge_ai/core/hardware.py)

Automatically detects your hardware and recommends optimal settings.

---

## 💾 Memory Package: forge_ai/memory/

Where the AI stores and retrieves information.

### manager.py - Conversation Storage
**File:** [forge_ai/memory/manager.py](../forge_ai/memory/manager.py)

Saves conversations to JSON files:
```python
manager = ConversationManager()
manager.save_conversation("chat_with_bob", messages)
```

### memory_db.py - SQLite Database
**File:** [forge_ai/memory/memory_db.py](../forge_ai/memory/memory_db.py)

Long-term memory storage with search capabilities.

### vector_utils.py - Semantic Search
**File:** [forge_ai/memory/vector_utils.py](../forge_ai/memory/vector_utils.py)

Find similar memories using cosine similarity (vector math).

---

## 🌐 Communications Package: forge_ai/comms/

Networking for multi-device setups.

### api_server.py - REST API
**File:** [forge_ai/comms/api_server.py](../forge_ai/comms/api_server.py)

Flask server with endpoints:
- `POST /generate` - Get AI response
- `GET /health` - Check server status
- `GET /models` - List available models

### network.py - Multi-Device Communication
**File:** [forge_ai/comms/network.py](../forge_ai/comms/network.py)

`EnigmaNode` class for device-to-device communication:
- PC talks to Raspberry Pi
- AI-to-AI conversations
- Memory sync across devices

### discovery.py - Find Other Devices
**File:** [forge_ai/comms/discovery.py](../forge_ai/comms/discovery.py)

Auto-discover other Enigma instances on your network.

### mobile_api.py - Phone Support
**File:** [forge_ai/comms/mobile_api.py](../forge_ai/comms/mobile_api.py)

Optimized API for mobile apps.

---

## 🖥️ GUI Package: forge_ai/gui/

Desktop graphical interfaces.

### main_window.py - Basic GUI
**File:** [forge_ai/gui/main_window.py](../forge_ai/gui/main_window.py)

Simple PyQt5 window with:
- Chat tab
- Logbook (history) tab
- Training tab
- Avatar tab

### enhanced_window.py - Full-Featured GUI
**File:** [forge_ai/gui/enhanced_window.py](../forge_ai/gui/enhanced_window.py)

Advanced GUI with:
- Setup wizard for first run
- Model management
- Dark/Light themes
- Training data editor
- Vision preview
- Terminal output tab
- History per AI

---

## 🔊 Voice Package: forge_ai/voice/

Speech capabilities.

### tts_simple.py - Text-to-Speech
**File:** [forge_ai/voice/tts_simple.py](../forge_ai/voice/tts_simple.py)

Makes the AI speak using pyttsx3:
```python
from forge_ai.voice import speak
speak("Hello, I am Enigma!")
```

### stt_simple.py - Speech-to-Text
**File:** [forge_ai/voice/stt_simple.py](../forge_ai/voice/stt_simple.py)

Listen to user's voice:
```python
from forge_ai.voice import listen
text = listen(timeout=5)
```

---

## 🤖 Avatar Package: forge_ai/avatar/

Visual character representation.

### avatar_api.py - Basic Avatar Control
**File:** [forge_ai/avatar/avatar_api.py](../forge_ai/avatar/avatar_api.py)

Simple avatar interface stub.

### controller.py - Full Avatar System
**File:** [forge_ai/avatar/controller.py](../forge_ai/avatar/controller.py)

```python
avatar = get_avatar()
avatar.enable()
avatar.move_to(500, 300)
avatar.set_expression("happy")
avatar.speak("Hello!")
```

---

## 🔧 Tools Package: forge_ai/tools/

AI capabilities to interact with the world.

### tool_registry.py - Tool Framework
**File:** [forge_ai/tools/tool_registry.py](../forge_ai/tools/tool_registry.py)

Register and execute tools:
```python
result = execute_tool("web_search", query="Python tutorials")
```

### vision.py - Screen Vision
**File:** [forge_ai/tools/vision.py](../forge_ai/tools/vision.py)

AI can "see" the screen:
- Capture screenshots
- OCR (read text from images)
- Find elements on screen

### web_tools.py - Internet Access
**File:** [forge_ai/tools/web_tools.py](../forge_ai/tools/web_tools.py)

Search the web, fetch pages.

### file_tools.py - File Operations
**File:** [forge_ai/tools/file_tools.py](../forge_ai/tools/file_tools.py)

Read, write, list, move files.

### document_tools.py - Document Reading
**File:** [forge_ai/tools/document_tools.py](../forge_ai/tools/document_tools.py)

Read PDF, EPUB, DOCX files.

---

## 📝 Data Flow: How It All Works Together

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

## 🎯 Quick Reference: Common Tasks

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
from forge_ai.comms.network import EnigmaNode
node = EnigmaNode(name="my_node")
node.start_ai_conversation("other_ai", num_turns=5)
```

---

## 🔗 File Relationships

```
run.py
  └── Uses: forge_ai/core/training.py (--train)
  └── Uses: forge_ai/core/inference.py (--run)
  └── Uses: forge_ai/gui/enhanced_window.py (--gui)
  └── Uses: forge_ai/comms/api_server.py (--serve)

enhanced_window.py
  └── Uses: forge_ai/core/model_registry.py (model management)
  └── Uses: forge_ai/core/inference.py (chat)
  └── Uses: forge_ai/memory/manager.py (history)
  └── Uses: forge_ai/tools/vision.py (screen capture)
  └── Uses: forge_ai/avatar/controller.py (avatar)
  └── Uses: forge_ai/voice/ (TTS/STT)

api_server.py
  └── Uses: forge_ai/core/inference.py (generate responses)
  └── Uses: forge_ai/core/model_registry.py (model info)
```

---

## 📚 Next Steps

1. **Read the HOW_TO_MAKE_AI.txt** for step-by-step training guide
2. **Check examples/** folder for working code samples
3. **Edit data/data.txt** to add your training data
4. **Run `python run.py --gui`** to start the interface

Happy coding! 🚀
