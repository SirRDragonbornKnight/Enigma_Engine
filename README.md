# Enigma Engine

**Create your own AI from scratch.** No cloud required. Runs on anything from a Raspberry Pi to a gaming PC.

## What is this?

Enigma Engine is a complete framework for building, training, and running your own AI assistant. It's designed to be:

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

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

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
| tiny | ~2M | 1GB | Testing, Raspberry Pi |
| small | ~10M | 2GB | Pi 4, basic responses |
| medium | ~50M | 4GB | Good conversations |
| large | ~150M | 8GB | High quality |
| xl | ~300M | 12GB | Very high quality |
| xxl | ~500M | 16GB | Near-commercial quality |

## Project Structure

```
enigma_engine/
├── run.py              # Main entry point
├── data/               # Training data goes here
│   └── training_data.txt
├── models/             # Your trained AI models
├── enigma/
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

- **Chat Interface** - Talk to your AI
- **Training Tab** - Train and monitor progress
- **Vision** - Screen capture and image analysis
- **Avatar** - Visual representation of your AI
- **Voice** - Text-to-speech responses
- **Multi-AI** - Run multiple AI models
- **Themes** - Dark, Light, Shadow, Midnight

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

## Requirements

- Python 3.8+
- PyTorch (CPU or CUDA)
- PyQt5 (for GUI)
- 2GB+ RAM (more for larger models)

## Documentation

- **[Getting Started](GETTING_STARTED.md)** - Quick start guide
- **[Module System Guide](docs/MODULE_GUIDE.md)** - Complete guide to the module system
- **[Project Overview](PROJECT_OVERVIEW.txt)** - Architecture and features
- **[How To Make AI](HOW_TO_MAKE_AI.txt)** - AI creation tutorial
- **[What NOT To Do](docs/WHAT_NOT_TO_DO.txt)** - Common mistakes to avoid

## Contributing

Contributions welcome! Please read the documentation in `docs/` first.

See [CREDITS.md](CREDITS.md) for attribution of libraries and research we build upon.

## License

MIT License - use freely for any purpose. See [LICENSE](LICENSE) for details.

---

**Made for AI enthusiasts who want to understand how AI really works.**
