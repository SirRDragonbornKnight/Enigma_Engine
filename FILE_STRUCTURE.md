# Enigma Engine - File Structure

AI engine with CLI and web GUI. Last updated 2026-02-21.

```
enigma-engine/
├── run.py                        # CLI entry point (--chat, --train, --serve)
├── pyproject.toml                # Project config (version 1.1.0)
├── setup.py                      # Package install (pip install -e .)
├── requirements.txt              # Dependencies (~45 lines)
├── forge_config.json             # Runtime config (54 keys)
├── LICENSE                       # MIT license
├── SUGGESTIONS.md                # Development plan & audit log
├── FILE_STRUCTURE.md             # This file
├── AA code maker.md              # Project rules & conventions
├── TRAINING_STATUS.md            # Training progress notes
│
├── enigma_engine/                # Main package
│   ├── __init__.py               # Package init, version, ROOT/DATA_DIR/MODELS_DIR
│   ├── py.typed                  # PEP 561 type hint marker
│   ├── router.py                 # BrickRouter (TCP server, port 9900)
│   │
│   ├── api/                      # Web GUI (FastAPI)
│   │   ├── __init__.py           # Package init
│   │   ├── server.py             # REST API + web UI server (~450 lines)
│   │   ├── static/
│   │   │   ├── style.css         # Dark theme CSS (~400 lines)
│   │   │   └── app.js            # Vanilla JS frontend (~300 lines)
│   │   └── templates/
│   │       └── index.html        # Single-page web app
│   │
│   ├── config/                   # Configuration
│   │   ├── __init__.py           # CONFIG dict, get_config(), update_config()
│   │   └── defaults.py           # Default config values (54 keys)
│   │
│   ├── core/                     # Core engine modules
│   │   ├── __init__.py           # Lazy imports for all core symbols
│   │   ├── model.py              # Enigma transformer (RoPE, RMSNorm, SwiGLU, GQA, MoE)
│   │   ├── model_presets.py      # ForgeConfig, MODEL_PRESETS, QuantizationConfig
│   │   ├── model_components.py   # RMSNorm, Attention, FeedForward, MoE, TransformerBlock
│   │   ├── model_utils.py        # apply_repetition_penalty, sample_next_token, detect_hardware
│   │   ├── inference.py          # EnigmaEngine (init, load, generate, stream)
│   │   ├── engine_generation.py  # _GenerationMixin (text gen, sampling, batch, routing)
│   │   ├── engine_chat.py        # _ChatMixin (chat, stream_chat, tools)
│   │   ├── commands.py           # CommandRegistry, [CMD] block parsing
│   │   ├── builtin_commands.py   # All registered command implementations
│   │   ├── ai_profile.py         # AIProfile, AIProfileManager (hot-swap personalities)
│   │   ├── training.py           # Trainer (SFT, best-of-N, evolutionary)
│   │   ├── lora_utils.py         # LoRA/QLoRA training with PEFT
│   │   ├── model_config.py       # Model configuration utilities
│   │   ├── model_registry.py     # ModelRegistry (list/find/cache models)
│   │   ├── weight_mapping.py     # WeightMapper (HF/GGUF/ONNX → Forge format)
│   │   │
│   │   ├── tokenizer.py          # SimpleTokenizer, TiktokenWrapper, get_tokenizer()
│   │   ├── bpe_tokenizer.py      # BPETokenizer (byte-pair encoding)
│   │   ├── char_tokenizer.py     # CharacterTokenizer (character-level + dictionary)
│   │   ├── advanced_tokenizer.py # AdvancedBPETokenizer (byte-level BPE with merges)
│   │   │
│   │   ├── kv_cache.py           # KVCache (pre-allocated, INT8 quant, sliding window)
│   │   ├── streaming.py          # TokenStreamer (SSE, WebSocket, callbacks)
│   │   ├── hardware_detection.py # GPU/VRAM/RAM detection, optimal config
│   │   ├── download_progress.py  # DownloadTracker (tqdm/rich progress bars)
│   │   │
│   │   ├── gguf.py               # GGUF format support (export + shared parsing)
│   │   ├── gguf_loader.py        # Load GGUF models (auto GPU layers, chat templates)
│   │   ├── gguf_dequant.py       # GGUF tensor parsing & dequantization (Q4_0, Q8_0)
│   │   ├── huggingface_loader.py # Load HuggingFace models (transformers)
│   │   ├── ollama_loader.py      # Load Ollama models
│   │   ├── onnx_loader.py        # Load ONNX models
│   │   └── gptq_awq_loader.py    # Load GPTQ/AWQ quantized models
│   │
│   └── vocab_model/              # Vocabulary data
│       └── char_vocab.json       # Character vocabulary
│
├── bricks/                       # TCP plugin system
│   ├── README.md                 # Brick documentation
│   ├── _template/                # Template for new bricks
│   │   ├── brick.json
│   │   ├── brick_base.py
│   │   └── main.py
│   ├── echo/                     # Test brick (echoes messages)
│   │   ├── brick.json
│   │   └── main.py
│   └── imagegen/                 # Image generation brick
│       ├── brick.json
│       └── main.py
│
├── profiles/                     # AI personality profiles
│   ├── assistant.json
│   ├── coding_helper.json
│   ├── creative_writer.json
│   └── researcher.json
│
├── models/                       # Model files
│   ├── enigma_small.pth          # Trained small model
│   ├── enigma_small_tokenizer.json
│   ├── enigma_tiny.pth           # Trained tiny model
│   ├── enigma_tiny_tokenizer.json
│   ├── registry.json             # Model registry
│   ├── checkpoints/              # Training checkpoints (epoch 5-50)
│   └── enigma/                   # Enigma model data (brain, learning, conversations)
│
├── data/                         # Runtime data
│   ├── training.txt              # Training data
│   ├── instructions.txt          # System instructions
│   ├── prompts.json              # Multi-purpose prompts
│   └── sessions/                 # Chat sessions
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_api.py               # 11 API endpoint tests (FastAPI TestClient)
│   ├── test_core.py              # 16 import/existence tests
│   └── test_functional.py        # 20 functional tests (model, tokenizer, KV-cache, commands)
│
├── outputs/                      # Generated outputs (3d, audio, code, images, etc.)
├── memory/                       # Saved conversations
├── logs/                         # Log files
└── information/                  # Documentation files
```

## Usage

```bash
# Show system info
python run.py

# CLI chat (requires model)
python run.py --chat
python run.py --chat --model path/to/model.gguf
python run.py --chat --model models/enigma_small.pth

# Train a model
python run.py --train data/training.txt --epochs 20 --model-size small

# Train a tokenizer
python run.py --train-tokenizer data/training.txt --vocab-size 8000

# Start web GUI
python run.py --serve
python run.py --serve --port 9000

# Run tests
python -m pytest tests/ -v
```
