# AI Tester - Code Tour Guide

Welcome! This guide walks you through the codebase so you understand how everything works together.

---

## 🏗️ Project Structure Overview

```
ai_tester/
├── run.py                  # 🚀 MAIN ENTRY POINT - Start here!
├── requirements.txt        # Dependencies to install
├── setup.py               # Package installation config
│
├── enigma/                # === THE MAIN PACKAGE ===
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

## 🧠 Core Package: enigma/core/

This is the **brain** of the AI - where all the neural network magic happens.

### model.py - The Neural Network Architecture
**File:** [enigma/core/model.py](../ai_tester/core/model.py)

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
**File:** [enigma/core/tokenizer.py](../ai_tester/core/tokenizer.py)

Converts human text into numbers the AI can understand.
- Full character-level tokenizer (every character gets a number)
- Includes a dictionary for ~3000 common words
- Custom dictionary support (`enigma/vocab_model/dictionary.txt`)
- Handles special tokens (padding, start, end, unknown)

### training.py - Teaching the AI
**File:** [enigma/core/training.py](../ai_tester/core/training.py)

Production-ready training loop:
- **Mixed Precision (AMP)** - Faster training on GPUs
- **Gradient Accumulation** - Large effective batch sizes
- **Cosine LR Schedule** - With warmup
- **Gradient Clipping** - Prevents exploding gradients
- **Checkpointing** - Save best and periodic checkpoints

### inference.py - Getting Responses
**File:** [enigma/core/inference.py](../ai_tester/core/inference.py)

`AITesterEngine` class - the main way to interact with the AI:
```python
engine = AITesterEngine()
response = engine.generate("Hello!", temperature=0.8, top_p=0.9)

# Streaming generation
for token in engine.stream_generate("Tell me a story"):
    print(token, end="")

# Chat with history
response = engine.chat("What's the weather?", history=[...])
```

### model_registry.py - Managing Multiple AIs
**File:** [enigma/core/model_registry.py](../ai_tester/core/model_registry.py)

Create and manage named AI personalities:
```python
registry = ModelRegistry()
registry.create_model("luna", size="medium")
registry.create_model("nova", size="large")
```

### model_config.py - Size Presets
**File:** [enigma/core/model_config.py](../ai_tester/core/model_config.py)

Defines model sizes from mobile to server-scale:
- `tiny`: ~2M params - Raspberry Pi, mobile, testing
- `small`: ~15M params - Laptop, low-end GPU
- `medium`: ~50M params - Mid-range GPU
- `large`: ~125M params - Like GPT-2 small
- `xl`: ~350M params - Like GPT-2 medium
- `xxl`: ~770M params - Like GPT-2 large
- `xxxl`: ~1.5B params - Like GPT-2 XL

### trainer.py - Advanced Training
**File:** [enigma/core/trainer.py](../ai_tester/core/trainer.py)

Full-featured `AITesterTrainer` class:
- Multi-GPU support (DataParallel)
- Mixed precision training
- Gradient accumulation
- Early stopping
- Learning rate scheduling
- Comprehensive logging

### model_scaling.py - Grow Your AI
**File:** [enigma/core/model_scaling.py](../ai_tester/core/model_scaling.py)

Scale models up or down:
- `grow_model()` - Expand small → large while preserving learning
- `shrink_model()` - Compress for deployment
- `KnowledgeDistiller` - Train small model to mimic large model

### layers.py - Building Blocks
**File:** [enigma/core/layers.py](../ai_tester/core/layers.py)

Additional neural network components:
- `FeedForward`, `GLU`, `GeGLU` - FFN variants
- `MultiQueryAttention` - Memory-efficient attention
- `GroupedQueryAttention` - Balance between MHA and MQA
- `SlidingWindowAttention` - For long sequences
- `LoRALayer` - Efficient fine-tuning
- `MixtureOfExperts` - Sparse scaling

### hardware.py - Auto-Detection
**File:** [enigma/core/hardware.py](../ai_tester/core/hardware.py)

Automatically detects your hardware and recommends optimal settings.

---

## 💾 Memory Package: enigma/memory/

Where the AI stores and retrieves information.

### manager.py - Conversation Storage
**File:** [enigma/memory/manager.py](../ai_tester/memory/manager.py)

Saves conversations to JSON files:
```python
manager = ConversationManager()
manager.save_conversation("chat_with_bob", messages)
```

### memory_db.py - SQLite Database
**File:** [enigma/memory/memory_db.py](../ai_tester/memory/memory_db.py)

Long-term memory storage with search capabilities.

### vector_utils.py - Semantic Search
**File:** [enigma/memory/vector_utils.py](../ai_tester/memory/vector_utils.py)

Find similar memories using cosine similarity (vector math).

---

## 🌐 Communications Package: enigma/comms/

Networking for multi-device setups.

### api_server.py - REST API
**File:** [enigma/comms/api_server.py](../ai_tester/comms/api_server.py)

Flask server with endpoints:
- `POST /generate` - Get AI response
- `GET /health` - Check server status
- `GET /models` - List available models

### network.py - Multi-Device Communication
**File:** [enigma/comms/network.py](../ai_tester/comms/network.py)

`EnigmaNode` class for device-to-device communication:
- PC talks to Raspberry Pi
- AI-to-AI conversations
- Memory sync across devices

### discovery.py - Find Other Devices
**File:** [enigma/comms/discovery.py](../ai_tester/comms/discovery.py)

Auto-discover other Enigma instances on your network.

### mobile_api.py - Phone Support
**File:** [enigma/comms/mobile_api.py](../ai_tester/comms/mobile_api.py)

Optimized API for mobile apps.

---

## 🖥️ GUI Package: enigma/gui/

Desktop graphical interfaces.

### main_window.py - Basic GUI
**File:** [enigma/gui/main_window.py](../ai_tester/gui/main_window.py)

Simple PyQt5 window with:
- Chat tab
- Logbook (history) tab
- Training tab
- Avatar tab

### enhanced_window.py - Full-Featured GUI
**File:** [enigma/gui/enhanced_window.py](../ai_tester/gui/enhanced_window.py)

Advanced GUI with:
- Setup wizard for first run
- Model management
- Dark/Light themes
- Training data editor
- Vision preview
- Terminal output tab
- History per AI

---

## 🔊 Voice Package: enigma/voice/

Speech capabilities.

### tts_simple.py - Text-to-Speech
**File:** [enigma/voice/tts_simple.py](../ai_tester/voice/tts_simple.py)

Makes the AI speak using pyttsx3:
```python
from ai_tester.voice import speak
speak("Hello, I am Enigma!")
```

### stt_simple.py - Speech-to-Text
**File:** [enigma/voice/stt_simple.py](../ai_tester/voice/stt_simple.py)

Listen to user's voice:
```python
from ai_tester.voice import listen
text = listen(timeout=5)
```

---

## 🤖 Avatar Package: enigma/avatar/

Visual character representation.

### avatar_api.py - Basic Avatar Control
**File:** [enigma/avatar/avatar_api.py](../ai_tester/avatar/avatar_api.py)

Simple avatar interface stub.

### controller.py - Full Avatar System
**File:** [enigma/avatar/controller.py](../ai_tester/avatar/controller.py)

```python
avatar = get_avatar()
avatar.enable()
avatar.move_to(500, 300)
avatar.set_expression("happy")
avatar.speak("Hello!")
```

---

## 🔧 Tools Package: enigma/tools/

AI capabilities to interact with the world.

### tool_registry.py - Tool Framework
**File:** [enigma/tools/tool_registry.py](../ai_tester/tools/tool_registry.py)

Register and execute tools:
```python
result = execute_tool("web_search", query="Python tutorials")
```

### vision.py - Screen Vision
**File:** [enigma/tools/vision.py](../ai_tester/tools/vision.py)

AI can "see" the screen:
- Capture screenshots
- OCR (read text from images)
- Find elements on screen

### web_tools.py - Internet Access
**File:** [enigma/tools/web_tools.py](../ai_tester/tools/web_tools.py)

Search the web, fetch pages.

### file_tools.py - File Operations
**File:** [enigma/tools/file_tools.py](../ai_tester/tools/file_tools.py)

Read, write, list, move files.

### document_tools.py - Document Reading
**File:** [enigma/tools/document_tools.py](../ai_tester/tools/document_tools.py)

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
from ai_tester.core.model_registry import ModelRegistry
registry = ModelRegistry()
registry.create_model("my_ai", size="small")
```

### Train your AI
```python
from ai_tester.core.training import train_model
train_model(force=True, num_epochs=10)
```

### Chat with your AI
```python
from ai_tester.core.inference import AITesterEngine
engine = AITesterEngine()
print(engine.generate("Hello!"))
```

### Run multiple AIs talking to each other
```python
from ai_tester.comms.network import EnigmaNode
node = EnigmaNode(name="my_node")
node.start_ai_conversation("other_ai", num_turns=5)
```

---

## 🔗 File Relationships

```
run.py
  └── Uses: enigma/core/training.py (--train)
  └── Uses: enigma/core/inference.py (--run)
  └── Uses: enigma/gui/enhanced_window.py (--gui)
  └── Uses: enigma/comms/api_server.py (--serve)

enhanced_window.py
  └── Uses: enigma/core/model_registry.py (model management)
  └── Uses: enigma/core/inference.py (chat)
  └── Uses: enigma/memory/manager.py (history)
  └── Uses: enigma/tools/vision.py (screen capture)
  └── Uses: enigma/avatar/controller.py (avatar)
  └── Uses: enigma/voice/ (TTS/STT)

api_server.py
  └── Uses: enigma/core/inference.py (generate responses)
  └── Uses: enigma/core/model_registry.py (model info)
```

---

## 📚 Next Steps

1. **Read the HOW_TO_MAKE_AI.txt** for step-by-step training guide
2. **Check examples/** folder for working code samples
3. **Edit data/data.txt** to add your training data
4. **Run `python run.py --gui`** to start the interface

Happy coding! 🚀
