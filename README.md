# Enigma AI Engine

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: autopep8](https://img.shields.io/badge/code%20style-autopep8-000000.svg)](https://github.com/hhatto/autopep8)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg)](https://pytorch.org/)
[![Whisper STT](https://img.shields.io/badge/Whisper-STT-green.svg)](https://github.com/openai/whisper)
[![Quantization](https://img.shields.io/badge/INT8%2FINT4-Quantization-orange.svg)](https://pytorch.org/docs/stable/quantization.html)

**Create your own AI from scratch.** No cloud required. Runs on anything from a Raspberry Pi to a gaming PC.

## What is this?

Enigma AI Engine is a complete framework for building, training, and running your own AI assistant. It's designed to be:

- **Easy to use** - GUI interface, no coding required to get started
- **Educational** - See exactly how your AI learns and works
- **Private** - Everything runs locally on your machine
- **Scalable** - From tiny models on a Pi to large models on GPUs

## Quick Start (5 minutes)

### 1. Install

```bash
# Clone the repository
git clone https://github.com/SirRDragonbornKnight/enigma_engine.git
cd enigma_engine

# Option A: Use the installer (recommended)
python install.py --standard

# Option B: Quick setup script (creates venv + installs)
./setup.sh

# Option C: Manual installation
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**Installer Options:**
```bash
python install.py --check     # See what's installed
python install.py --minimal   # Core only (smallest)
python install.py --standard  # Core + GUI + Voice
python install.py --full      # Everything
python install.py             # Interactive mode
```

> **Note**: On Windows PowerShell, you may need to run:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### 2. Launch the GUI

```bash
python run.py --gui
```

### 3. Create Your AI

The setup wizard will guide you through:
1. **Name your AI** - Give it a personality!
2. **Choose model size** - Tiny for testing, larger for smarter AI
3. **Start training** - Add data and watch it learn

## Training Your AI

Your AI learns from text data. The more quality data you provide, the smarter it becomes.

### Training Data Format

Create a file with Q&A pairs like this:

```
Q: What is your name?
A: I'm Echo, your AI assistant.

Q: How are you?
A: I'm doing well, thank you for asking!

User: Tell me a joke.
AI: Why did the programmer quit? Because he didn't get arrays!
```

### Training Tips

| Amount | Quality | Result |
|--------|---------|--------|
| 50-100 pairs | Any | Basic responses, good for testing |
| 500+ pairs | Good | Reasonable conversations |
| 2,000+ pairs | Diverse | Good AI responses |
| 10,000+ pairs | High quality | Excellent AI |

### Training Settings

| Setting | Recommended | Description |
|---------|-------------|-------------|
| Epochs | 20-50 | Times through your data |
| Batch Size | 2-4 (CPU), 8-16 (GPU) | Samples per step |
| Learning Rate | 0.0001 | How fast it learns |

## Model Sizes

Choose based on your hardware:

| Size | Parameters | RAM Needed | Best For |
|------|------------|------------|----------|
| nano | ~1M | 512MB | Embedded, microcontrollers |
| micro | ~2M | 512MB | IoT devices |
| tiny | ~5M | 1GB | Raspberry Pi, edge devices |
| small | ~27M | 2GB | Entry GPU, desktop default |
| medium | ~85M | 4GB | Mid-range GPU, good balance |
| large | ~200M | 8GB | RTX 3080+, high quality |
| xl | ~600M | 12GB | RTX 4090, excellent results |
| xxl | ~1.5B | 16GB+ | Multi-GPU, near-production |

## Usage Examples

### Quick Chat with Your AI

```python
from enigma_engine.core.inference import EnigmaEngine

# Load model and chat
engine = EnigmaEngine()
engine.load_model("models/my_ai")

response = engine.generate("Hello! How are you?")
print(response)
```

### Training a New Model

```python
from enigma_engine.core.model import create_model
from enigma_engine.core.training import Trainer, TrainingConfig
from enigma_engine.core.tokenizer import get_tokenizer

# Create model and tokenizer
model = create_model("small")
tokenizer = get_tokenizer("char")

# Configure training
config = TrainingConfig(
    epochs=20,
    batch_size=4,
    learning_rate=0.0001,
    warmup_steps=100
)

# Train
trainer = Trainer(model, tokenizer, config)
trainer.train("data/training.txt")
trainer.save("models/my_trained_ai")
```

### Using the Module System

```python
from enigma_engine.modules import ModuleManager

# Initialize manager (auto-discovers all modules)
manager = ModuleManager()

# Load core modules
manager.load('model')
manager.load('tokenizer')
manager.load('inference')

# Load image generation (local Stable Diffusion)
manager.load('image_gen_local')

# Generate an image
image_mod = manager.get_module('image_gen_local')
result = image_mod.generate("a sunset over mountains")
result.save("sunset.png")

# Unload when done
manager.unload('image_gen_local')
```

### Saving and Loading Conversations

```python
from enigma_engine.memory.manager import ConversationManager
from enigma_engine.memory.export import export_conversation

# Save conversations
conv_mgr = ConversationManager()
conv_mgr.add_message("user", "What's the weather?")
conv_mgr.add_message("assistant", "I don't have weather data, but I can help you find it!")
conv_mgr.save()

# Export to different formats
export_conversation(conv_mgr.messages, "chat.md", format="markdown")
export_conversation(conv_mgr.messages, "chat.html", format="html")
```

### Using System Prompts

```python
from enigma_engine.prompts.system_prompts import get_prompt, list_prompts

# See available prompts
print(list_prompts())  # ['assistant', 'coder', 'tutor', ...]

# Get a coding-focused prompt
system_prompt = get_prompt("coder")
print(system_prompt)
```

### Benchmarking Performance

```python
from enigma_engine.utils.benchmark_inference import quick_benchmark

# Quick benchmark of current model
report = quick_benchmark("models/my_ai")
print(f"Speed: {report.tokens_per_second:.1f} tokens/s")
print(f"Memory: {report.peak_memory_mb:.0f} MB")
```

## Project Structure

```
enigma_engine/
├── run.py              # Main entry point
├── data/               # Training data goes here
│   └── training.txt    # Consolidated training data
├── models/             # Your trained AI models
├── enigma_engine/
│   ├── core/          # AI model & training
│   ├── gui/           # User interface
│   ├── voice/         # Speech features
│   ├── avatar/        # Visual avatar
│   └── tools/         # Vision, files, etc.
└── docs/              # Documentation
```

## Command Line Options

```bash
python run.py --gui     # Launch GUI (recommended)
python run.py --train   # Train from command line
python run.py --run     # CLI chat interface
python run.py --serve   # Start API server
```

## Features

### Core Features
- **Chat Interface** - Talk to your AI
- **Training Tab** - Train and monitor progress
- **Vision** - Screen capture and image analysis
- **Avatar** - Visual representation of your AI
- **Voice** - Text-to-speech responses
- **Multi-AI** - Run multiple AI models
- **Multi-Device Network** - Connect multiple Enigma AI Engine instances across devices
- **Specialized Models** - Train task-specific models (routing, vision, code)
- **Themes** - Dark, Light, Shadow, Midnight
- **🌐 Tunnel Management** - Expose your AI to the internet for remote access

### Remote Access with Tunnels 🌐

Expose your Enigma AI Engine server to the internet for remote access from anywhere:

```bash
# Quick start with ngrok
python run.py --tunnel --tunnel-token YOUR_NGROK_TOKEN

# Or use localtunnel (no account needed)
python run.py --tunnel --tunnel-provider localtunnel

# Start API server in one terminal
python run.py --serve

# Expose it in another terminal
python run.py --tunnel --tunnel-port 5000
```

Perfect for:
- Access your AI from mobile devices
- Share demos with colleagues
- Remote training monitoring
- API integrations

See `docs/TUNNEL_SETUP.md` for complete guide.

### Multi-Device Networking 🌐

Connect multiple Enigma AI Engine instances across devices for distributed AI:

- **Auto-Discovery** - Find other Enigma AI Engine nodes on your network (UDP broadcast + network scan)
- **Remote Connections** - Connect devices on same network or via manual IP entry
- **AI-to-AI Conversations** - Let different AI instances talk to each other
- **Model Sharing** - Export/import models across devices
- **Memory Sync** - Synchronize conversation history

**Quick Example:**
```bash
# Device 1: Start as server
python examples/multi_device_example.py --server --name pc_brain

# Device 2: Discover and connect
python examples/discovery_example.py  # Find device 1
python examples/multi_device_example.py --client --connect 192.168.1.100:5000

# Or use the GUI Network tab for easy point-and-click discovery
python run.py --gui  # Navigate to Network tab
```

See `docs/multi_device_guide.md` for complete guide.

### Specialized Model System 🎯

Enigma AI Engine supports training **specialized smaller models** for specific tasks:

- **Intent Router** (~1M params) - Classify user requests (nano model, instant)
- **Vision Captioning** (~5M params) - Describe images from features (tiny model)
- **Code Generation** (~27M params) - Generate and explain code (small model)

All models share the same tokenizer for seamless interoperability.

**Quick Example:**
```bash
# Train a nano router to classify intents
python scripts/train_specialized_model.py \
    --type router \
    --data data/specialized/router_training.txt \
    --model-size nano

# Use in code
from enigma_engine.core.inference import EnigmaEngine
engine = EnigmaEngine(use_routing=True)
response = engine.generate("write a sort function")  # Routes to code model
```

See `docs/SPECIALIZED_MODELS.md` for complete guide.

### Upgraded Components (Optional)
- **⚡ Flash Attention** - 2-4x faster training (auto-detected, CUDA + fp16/bf16)
- **🎤 Whisper STT** - High-accuracy speech recognition (vs basic STT)
- **🔊 Natural TTS** - Coqui/Bark for realistic voices (vs pyttsx3)
- **⚡ tiktoken** - 5-10x faster tokenization (vs BPE)
- **📦 Quantization** - INT8/INT4 for 2-4x speedup and lower memory

Install upgraded components with:
```bash
pip install flash-attn      # Flash Attention (RTX 30xx/40xx required)
pip install openai-whisper  # Whisper STT
pip install TTS             # Coqui TTS
pip install tiktoken        # Fast tokenizer
pip install bitsandbytes    # INT4 quantization
```

## Troubleshooting

### "Model is untrained"
This is normal for a new AI! Go to the Train tab and train it.

### Training is slow
- Reduce model size (use "tiny" or "small")
- Reduce batch size to 1-2
- Reduce max sequence length
- Use GPU if available

### Out of memory
- Use a smaller model size
- Reduce batch size
- Close other programs

### Bad AI responses
- Add more training data
- Train for more epochs
- Check your data quality

**More solutions**: See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for the complete troubleshooting guide.

## Privacy & Local-First Design

Enigma AI Engine is an **"open black box"** - completely transparent about what runs locally vs what requires cloud services.

### 100% Local by Default

**Enigma AI Engine runs in local-only mode by default.** This means:
- ✅ No API keys required
- ✅ No data leaves your machine
- ✅ Works completely offline
- ✅ Zero cloud dependencies

### Local vs Cloud Modules

Enigma uses a modular system where you choose what to enable:

**Local Modules** (No internet required):
- Core AI model, training, and inference
- Local image generation (Stable Diffusion)
- Local code generation (your Enigma model)
- Local text-to-speech (pyttsx3)
- Local embeddings (sentence-transformers)
- Memory, vision, avatar, GUI, and more

**Cloud Modules** (Requires API keys + internet):
- OpenAI DALL-E, GPT-4
- Replicate (SDXL, video, music generation)
- ElevenLabs (premium TTS)
- OpenAI Embeddings

Cloud modules require **explicit opt-in** and will show a warning when loaded:
```
⚠️  Warning: This module connects to external cloud services
```

### Learn More

See [docs/LOCAL_VS_CLOUD.md](docs/LOCAL_VS_CLOUD.md) for complete details on:
- Which modules are local vs cloud
- How to enable/disable cloud services
- Privacy implications
- Best practices for maximum privacy

**Remember:** By default, everything stays on your machine. You choose when to use cloud services.

## Requirements

- Python 3.8+
- PyTorch (CPU or CUDA)
- PyQt5 (for GUI)
- 2GB+ RAM (more for larger models)

## Documentation

- **[Getting Started](GETTING_STARTED.md)** - Quick start guide
- **[Code Tour](docs/CODE_TOUR.md)** - Navigate the codebase
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues & solutions
- **[Module System Guide](docs/MODULE_GUIDE.md)** - Complete guide to the module system
- **[How To Train](docs/HOW_TO_TRAIN.md)** - Training guide and tips

## Contributing

Contributions welcome! Please read the documentation in `docs/` first.

See [CREDITS.md](CREDITS.md) for attribution of libraries and research we build upon.

## License

MIT License - use freely for any purpose. See [LICENSE](LICENSE) for details.

---

**Made for AI enthusiasts who want to understand how AI really works.**
