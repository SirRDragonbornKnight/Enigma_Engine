# Getting Started

Welcome to Enigma Engine — a fully local AI system.

---

## Quick Start

1. **Load a model** — Go to ROUTER page, select a model for the CHAT route
2. **Start chatting** — Go to CORE page, type a message, press SEND
3. **Customize** — Change generation parameters on the CONFIG page
4. **Train** — Use the FORGE page to train your own models

---

## GUI Pages

| Page | What It Does |
|------|-------------|
| CORE | Chat with the AI. History sidebar, system prompt editor |
| CMD | Terminal. Run system commands or engine commands |
| MODELS | Create and delete model files |
| ROUTER | Assign models to routes (chat, training, bricks) |
| FORGE | Train models and tokenizers, edit training data |
| CONFIG | Generation parameters and directory paths |
| DOCS | Documentation, prompts, and reference files |

---

## CLI Usage

```bash
python run.py                                         # Show system info
python run.py --gui                                    # Launch desktop GUI
python run.py --serve                                  # Start web GUI on port 8080
python run.py --serve --port 9090                      # Web GUI on custom port
python run.py --chat --model PATH                      # CLI chat with a model
python run.py --train data/training.txt --epochs 10    # Train model
python run.py --train-tokenizer data/training.txt      # Train BPE tokenizer
python run.py --help                                   # Show all options
```

---

## Directory Structure

| Folder | Contents |
|--------|----------|
| models/ | AI model files (.gguf, .pth, etc.) |
| data/ | Training data, settings, sessions |
| profiles/ | AI personality profiles (JSON) |
| bricks/ | Plugin bricks (each in its own folder) |
| information/ | Documentation and reference files |
| outputs/ | Generated content (images, code, etc.) |
| memory/ | Persistent memory storage |

---

## Tips

- **Profiles** change the AI personality and generation settings
- **System prompt** shapes how the AI responds (edit in CORE sidebar)
- **KV-cache** is cleared on NEW chat to prevent hallucinations
- **Bricks** are plugins that extend the AI (image generation, etc.)
- All text in the GUI is selectable and copyable
