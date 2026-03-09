# Enigma Engine - File Structure

AI engine with CLI, API server, and desktop GUI. Last updated 2026-03-04.

```
enigma-engine/
├── run.py                        # CLI entry point (--chat, --train, --serve, --gui)
├── pyproject.toml                # Project config (version 1.1.0, all deps)
├── requirements.txt              # Dependencies
├── forge_config.json             # Runtime config (54 keys)
├── LICENSE                       # MIT license
├── SUGGESTIONS.md                # Development plan & session log
├── FILE_STRUCTURE.md             # This file
├── AA code maker.md              # Project rules & conventions
├── GUI_REFERENCE.md              # Every GUI element documented
├── TRAINING_STATUS.md            # Training progress notes
│
├── enigma_engine/                # Main package
│   ├── __init__.py               # Package init, version, ROOT/DATA_DIR/MODELS_DIR
│   ├── py.typed                  # PEP 561 type hint marker
│   ├── router.py                 # ModRouter (TCP server, port 9900)
│   │
│   ├── api/                      # Local API Server (FastAPI)
│   │   ├── __init__.py           # Package init
│   │   └── server.py             # REST API server (~800 lines)
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
│   │   ├── model_config.py       # Model configuration utilities
│   │   ├── model_context.py      # Per-model persistent history and prompt storage
│   │   ├── model_registry.py     # ModelRegistry (list/find/cache models)
│   │   ├── inference.py          # EnigmaEngine (init, load, generate, stream)
│   │   ├── engine_generation.py  # _GenerationMixin (text gen, sampling, batch, routing, vision gen)
│   │   ├── engine_chat.py        # _ChatMixin (chat, stream_chat, tools, image encoding)
│   │   ├── vision_encoder.py     # VisionEncoder (ViT from scratch), preprocess, encode helpers
│   │   ├── memory.py             # PersistentMemory (fact storage, extraction, context injection)
│   │   ├── commands.py           # CommandRegistry, [CMD] block parsing
│   │   ├── builtin_commands.py   # All registered command implementations
│   │   ├── ai_profile.py         # AIProfile, AIProfileManager (hot-swap personalities)
│   │   ├── training.py           # Trainer (SFT, best-of-N, evolutionary)
│   │   ├── lora_utils.py         # LoRA/QLoRA training with PEFT
│   │   ├── weight_mapping.py     # WeightMapper (HF/GGUF/ONNX → Forge format)
│   │   ├── document_readers.py   # PDF/DOCX/text file parsing
│   │   ├── model_compare.py      # Model comparison utilities
│   │   │
│   │   ├── tokenizer.py          # SimpleTokenizer, TiktokenWrapper, get_tokenizer()
│   │   ├── bpe_tokenizer.py      # BPETokenizer (byte-pair encoding)
│   │   ├── char_tokenizer.py     # CharacterTokenizer (character-level + dictionary)
│   │   ├── advanced_tokenizer.py # AdvancedBPETokenizer (byte-level BPE with merges)
│   │   ├── plugin_loader.py      # Plugin discovery: scan plugins/, import, call register(registry)
│   │   ├── rag.py                # RAG/Document Q&A support
│   │   ├── reasoning.py          # Chain-of-thought engine, <think> tags, strip_incomplete_think
│   │   ├── web_utils.py          # Shared web utilities: DDG search, page fetch, HTML extraction
│   │   ├── safe_save.py          # Atomic torch.save — write to .tmp then os.replace()
│   │   │
│   │   ├── kv_cache.py           # KVCache (pre-allocated, INT8 quant, sliding window)
│   │   ├── streaming.py          # TokenStreamer (SSE, WebSocket, callbacks)
│   │   ├── hardware_detection.py # GPU/VRAM/RAM detection, optimal config
│   │   ├── download_progress.py  # DownloadTracker (tqdm/rich progress bars)
│   │   │
│   │   ├── gguf.py               # GGUF format support (export + shared parsing)
│   │   ├── gguf_loader.py        # Load GGUF models (auto GPU layers, chat templates, llama-server backend)
│   │   ├── gguf_dequant.py       # GGUF tensor parsing & dequantization (Q4_0, Q8_0)
│   │   ├── huggingface_loader.py # Load HuggingFace models (transformers)
│   │   ├── ollama_loader.py      # Load Ollama models
│   │   ├── onnx_loader.py        # Load ONNX models
│   │   └── gptq_awq_loader.py    # Load GPTQ/AWQ quantized models
│   │
│   ├── gui/                      # Desktop GUI (CustomTkinter, 7-mixin pattern)
│   │   ├── __init__.py           # Package init
│   │   ├── widgets.py            # Colors, fonts, widget classes, factory functions
│   │   ├── desktop.py            # Window shell, header, nav rail, status bar, entry point
│   │   ├── gui_pages.py          # Page builders: CORE, MODELS, ROUTER, FORGE, CONFIG
│   │   ├── gui_logic.py          # Chat, sessions, routes, model loading, voice I/O (TTS)
│   │   ├── gui_forge.py          # Autonomous training (solo/guided 3-phase), model management, web learn, tools
│   │   ├── gui_docs_page.py      # DOCS page: documentation browser, file editor
│   │   ├── gui_cmd_page.py       # CMD page: dual-mode terminal (SYSTEM + ENGINE)
│   │   ├── gui_mods.py           # Mod subprocess lifecycle (start/stop/auto-start)
│   │   ├── gui_mod_page.py       # Per-mod page builder from mod.json
│   │   ├── media.py              # Chat media: images, GIFs, videos, URLs
│   │   ├── themes.py             # Color theme system: 4 presets, Theme dataclass, load/save
│   │   └── scanners.py           # Filesystem scanning, config limits, ROUTE_KEYS, target_size display
│   │
│   ├── bin/                      # Bundled binaries (not in git, ~1GB)
│   │   └── llama-server/         # llama.cpp server for Blackwell GPU support
│   │       ├── llama-server.exe  # HTTP server binary
│   │       ├── ggml-cuda.dll     # CUDA backend (440MB)
│   │       └── ...               # Supporting DLLs (25 files total)
│   │
│   └── vocab_model/              # Vocabulary data
│       └── char_vocab.json       # Character vocabulary
│
├── mods/                         # TCP plugin system
│   ├── README.md                 # Mod documentation
│   ├── _template/                # Template for new mods
│   │   ├── mod.json
│   │   ├── mod_base.py
│   │   └── main.py
│   ├── echo/                     # Test mod (echoes messages)
│   │   ├── mod.json
│   │   ├── main.py
│   │   └── docs/                 # Mod documentation files
│   ├── imagegen/                 # Image generation mod
│   │   ├── mod.json
│   │   ├── main.py
│   │   ├── docs/                 # Mod documentation files
│   │   └── outputs/              # Generated images
│   ├── audiogen/                 # Audio/TTS generation mod
│   │   └── audiogen.py
│   ├── avatar/                   # 3D avatar mod (expressions, bones, lip sync)
│   │   ├── enigma_avatar/        # Avatar engine package
│   │   ├── enigma_avatar_brick.py
│   │   └── data/                 # Avatar data files
│   ├── codegen/                  # Code generation mod
│   │   └── codegen.py
│   ├── router/                   # Mod router service
│   │   └── router.py
│   ├── threed/                   # 3D generation mod
│   │   └── threed.py
│   ├── videogen/                 # Video generation mod
│   │   └── videogen.py
│   ├── vision/                   # Vision/screen analysis mod
│   │   └── vision.py
│   └── voice/                    # Voice STT/TTS mod
│       └── voice.py
│
├── profiles/                     # AI personality profiles (used by ai_profile.py, server.py API)
│   ├── assistant.json
│   ├── coding_helper.json
│   ├── creative_writer.json
│   ├── not_for_you_hahaha.json
│   ├── researcher.json
│   └── assistant/                # Profile-specific conversation data
│
├── plugins/                      # Plugin system (auto-discovered at startup)
│   ├── README.md                 # Plugin documentation
│   └── _example.py               # Example plugin template (skipped, _ prefix)
│
├── models/                       # Model files
│   ├── base.pth                  # Base native model
│   ├── registry.json             # Model registry
│   ├── checkpoints/              # Training checkpoints
│   ├── qwen3-30b-a3b/            # Qwen3 30B-A3B MoE (Q4_K_M GGUF, 17.28 GB)
│   └── qwen3-8b/                 # Qwen3 8B (HuggingFace safetensors, 15.27 GB)
│
├── data/                         # Runtime data
│   ├── training.txt              # Training data
│   ├── instructions.txt          # System instructions
│   ├── prompts.json              # Multi-purpose prompts
│   ├── gui_settings.json         # GUI settings (display names, window geometry)
│   ├── route_assignments.json    # Persisted route assignments
│   ├── avatar/                   # Avatar images and models
│   ├── model_contexts/           # Per-model history and prompts
│   ├── notes/                    # User notes
│   ├── prompts/                  # Per-route system prompts
│   │   ├── chat.md               # Default system prompt for chat
│   │   └── trainer.md            # Trainer context for FORGE
│   └── avatar/
│
├── tests/                        # Test suite (1019 tests)
│   ├── __init__.py
│   ├── test_api.py               # API endpoint tests
│   ├── test_core.py              # Import/existence tests
│   ├── test_functional.py        # Functional tests (model, tokenizer, KV-cache, commands)
│   ├── test_gui.py               # GUI mixin/method tests
│   └── test_reasoning.py         # Reasoning engine and <think> tag tests
│
├── information/                  # Documentation files
│   ├── commands_reference.md     # Engine command documentation
│   ├── external_models.md        # External model format compatibility
│   ├── getting_started.md        # Getting started guide
│   ├── how_the_ai_works.md       # How the AI works
│   ├── prompts_guide.md          # Prompts and profiles guide
│   ├── training_guide.md         # Training guide
│   └── trainer/                  # Trainer-specific guides
│       ├── data_preparation.md
│       ├── model_sizes.md
│       ├── training_methods.md
│       └── using_the_forge.md
│
├── outputs/                      # Generated outputs (3d, audio, code, images, etc.)
├── memory/                       # Saved conversations
└── logs/                         # Log files
```

## Usage

```bash
# Show system info
python run.py

# CLI chat (requires model)
python run.py --chat
python run.py --chat --model path/to/model.gguf

# Train a model
python run.py --train data/training.txt --epochs 20 --model-size small

# Train a tokenizer
python run.py --train-tokenizer data/training.txt --vocab-size 8000

# Start API server
python run.py --serve
python run.py --serve --port 9000

# Launch desktop GUI
python run.py --gui

# Run tests
python -m pytest tests/ -v
```
