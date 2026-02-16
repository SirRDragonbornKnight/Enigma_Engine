# Enigma AI Engine - Program Report

**Generated:** February 15, 2026

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [GUI Tabs Summary](#gui-tabs-summary)
4. [GUI Tabs Detailed](#gui-tabs-detailed)
5. [File Dependencies](#file-dependencies)

---

## Architecture Overview

Enigma AI Engine is a **fully modular AI framework** where everything is a toggleable module. The system supports running on hardware from Raspberry Pi to datacenter servers.

### Main Packages
```
enigma_engine/
├── core/          # AI model, inference, training, tokenizer
├── gui/           # PyQt5 interface with 28 tabs
├── modules/       # Module manager, registry, state handling
├── tools/         # Vision, web, file, document tools
├── memory/        # Conversation storage, vector DB
├── voice/         # TTS/STT wrappers
├── avatar/        # Avatar control and rendering
├── comms/         # API server, networking
├── config/        # Global configuration
└── utils/         # Helpers, security, lazy imports
```

### Key Technologies
- **Framework:** PyTorch for AI, PyQt5 for GUI
- **Model:** Custom transformer with RoPE, RMSNorm, SwiGLU, GQA, KV-cache
- **Sizes:** 15 presets from ~500K params (pi_zero) to 70B+ params (omega)

### Package List (489 Python files)

| Category | Packages | Files | Status |
|----------|----------|-------|--------|
| **Core** | core, gui/tabs, tools, utils, memory, voice, avatar, comms, modules, builtin, config, web, cli, i18n, game, plugins | ~450 | Active |
| **Features** | learning, self_improvement, marketplace, network | ~35 | Integrated |
| **Test/Internal** | security, agents, auth | ~20 | Tests only |
| **Optional** | companion, mobile | ~4 | Complete but minimal use |

---

## Core Components - How They Work

| Component | File | How It Works |
|-----------|------|-------------|
| Model | `core/model.py` | Stacks N transformer blocks. Each block: RMSNorm → self-attention (Q/K/V matrices with RoPE position encoding) → feedforward (SwiGLU activation). Grouped-Query Attention shares keys/values across heads. KV-cache stores past computations. Output: probability distribution over vocabulary. |
| Inference | `core/inference.py` | Generation loop: tokenize input → run through model → get next-token probabilities → sample (greedy/top-k/top-p/temperature) → append token → repeat until EOS or max length. Supports streaming via Python generators. |
| Training | `core/training.py` | Forward pass: compute cross-entropy loss between model output and target. Backward pass: AdamW optimizer updates weights. Handles batching, gradient accumulation, learning rate warmup + cosine decay. Checkpoints every N steps. |
| Tokenizer | `core/tokenizer.py` | BPE (byte-pair encoding): learns common subword patterns from corpus. Unknown words split into known pieces. "playing" → ["play", "ing"]. Special tokens: [PAD], [EOS], [BOS]. Vocabulary 32K-100K tokens. |
| Tool Router | `core/tool_router.py` | Scans user input for keywords ("draw" → image, "search" → web). Each tool has priority-ordered model assignments. Extracts parameters from natural language. Routes to tool_executor for execution. |
| Module Manager | `modules/manager.py` | State machine: UNLOADED → LOADING → LOADED → ACTIVE. Checks dependencies (inference needs model+tokenizer). Prevents conflicts (can't load image_gen_local AND image_gen_api). Tracks memory per module. |
| Tool Executor | `tools/tool_executor.py` | Parses `<tool_call>{JSON}</tool_call>` from AI output. Validates against schema. Runs handler with timeout (SIGALRM on Unix, threading.Timer on Windows). Blocks dangerous paths via security.py. |

---

## GUI Tabs Summary

**Total Tabs:** 28 (16 unused tabs removed Feb 15, 2026)

### Quick Reference

| Tab | Lines | Purpose |
|-----|-------|---------|
| chat_tab | 2869 | Main AI conversation interface |
| settings_tab | 4721 | Application settings & API keys |
| image_tab | 1449 | Image generation |
| training_tab | 1424 | Model training interface |
| workspace_tab | 1001 | Training data preparation |
| bundle_manager_tab | 991 | Package models for sharing |
| training_data_tab | 987 | Generate training data |
| voice_clone_tab | 923 | Voice cloning |
| persona_tab | 761 | Prompt/persona management |
| audio_tab | 747 | Text-to-speech |
| threed_tab | 718 | 3D model generation |
| network_tab | 717 | Multi-device networking |
| scheduler_tab | 694 | Task scheduling |
| analytics_tab | 671 | Usage analytics |
| base_generation_tab | 655 | Base class for generation tabs |
| embeddings_tab | 633 | Vector embeddings |
| video_tab | 623 | Video generation |
| code_tab | 607 | Code generation |
| model_router_tab | 599 | Assign models to tasks |
| federation_tab | 526 | Federated learning |
| notes_tab | 523 | Notes and bookmarks |
| gif_tab | 485 | GIF generation |
| camera_tab | 468 | Webcam capture |
| instructions_tab | 364 | Help and documentation |
| vision_tab | 212 | Image/screen analysis |
| terminal_tab | 189 | Embedded terminal |
| sessions_tab | 178 | Session management |
| avatar_tab | 46 | Avatar display |

---

## GUI Tabs Detailed - How They Work

### Core Interface Tabs

#### chat_tab.py (2869 lines)
**How it works:** QTextEdit captures user input. On send: message added to QScrollArea display → EnigmaEngine.generate() called in QThread to avoid freezing UI → response tokens streamed via Qt signal/slot as they generate → each token appended to display for typing effect. Conversation history stored via memory/manager.py as JSON. Model switcher dropdown triggers engine reload.

#### settings_tab.py (4721 lines)
**How it works:** Loads forge_config.json on init. QFormLayout with QLineEdit/QComboBox for each setting. API keys encrypted via Fernet (AES-128) before saving to ~/.enigma_engine/keys. Changes emit configChanged signal so other tabs can react. Sections collapsible via QToolButton toggles.

#### training_tab.py (1424 lines)
**How it works:** 3-step workflow: 1) File picker selects dataset → validates Q&A format 2) Hyperparameter sliders set lr, batch_size, epochs 3) Start button creates Trainer instance in QThread. Progress callback updates QProgressBar. Training loop: forward pass → loss → backward → optimizer step. Checkpoints saved every N steps.

### Generation Tabs (inherit BaseGenerationTab)

#### image_tab.py (1449 lines)
**How it works:** Provider pattern - StableDiffusionLocal wraps diffusers pipeline (UNet + VAE + scheduler), OpenAIImage calls DALL-E API, ReplicateImage uses their hosted models. User selects provider → enters prompt → Generate button starts QThread → provider.generate() runs → result path emitted → QLabel displays image with ResizableImagePreview allowing drag-to-resize.

#### code_tab.py (607 lines)
**How it works:** ForgeCode uses local model with code-specific system prompt ("You are a coding assistant..."). OpenAICode calls GPT-4 API. Output displayed in QTextEdit with QSyntaxHighlighter for color-coded keywords. Language dropdown changes syntax rules and prompt template.

#### video_tab.py (623 lines)
**How it works:** LocalVideo wraps AnimateDiff pipeline - generates sequence of latents, decodes each to frame, assembles into video. ReplicateVideo calls external API. Output saved as MP4 (via moviepy) or GIF (via PIL). Frame count and FPS sliders control output.

#### audio_tab.py (747 lines)
**How it works:** LocalTTS wraps pyttsx3 engine (system voices). ElevenLabsTTS sends text to their API with voice_id, receives audio bytes. Audio played via QMediaPlayer or sounddevice. Voice/rate/volume sliders control TTS parameters.

#### threed_tab.py (718 lines)
**How it works:** Local3DGen wraps Shap-E model - generates implicit neural representation → marching cubes extracts mesh → exports as OBJ. Cloud3DGen calls Replicate API. Preview uses QOpenGLWidget with rotation controls, or matplotlib 3D plot fallback.

### Model Management Tabs

#### modules_tab.py (1383 lines)
**How it works:** Reads MODULE_REGISTRY dict from registry.py. Each module displayed as custom widget with toggle switch, dependency list, GPU/API badges. Toggle triggers ModuleManager.load()/unload() in QThread. Dependencies grayed out if prereqs missing. Search box filters by name.

#### model_router_tab.py (599 lines)
**How it works:** Visual tool→model mapping. Each tool category has QComboBox listing available models. Selection changes saved to tool_routing.json. ToolRouter.reload_config() called to apply immediately. Color-coded categories for visual organization.

#### training_data_tab.py (987 lines)
**How it works:** AI-assisted dataset generation. User enters topic → GeneratorWorker QThread calls Claude/GPT-4 API with prompt template → API returns Q&A pairs → parsed and formatted → appended to training data file. Supports multiple output formats: Q&A, conversation, instruction.

### Utility Tabs

#### network_tab.py (717 lines)
**How it works:** NetworkScanner QThread sends UDP broadcast on local network → other Enigma instances respond with capabilities JSON → results displayed in QListWidget with device info. Connect button establishes TCP socket for distributed inference. API server toggle starts/stops Flask server.

#### camera_tab.py (468 lines)
**How it works:** CameraThread wraps OpenCV VideoCapture. Runs at 30fps in background thread, emits QPixmap via signal → QLabel displays frame. Capture button saves current frame to disk. Analyze button sends frame to vision model for description.

#### analytics_tab.py (671 lines)  
**How it works:** Reads JSON files from ~/.enigma_engine/analytics/ directory. AnalyticsRecorder logs events during usage. Tab aggregates data by day/week → displays charts via pyqtgraph line plots or matplotlib bar charts. Shows: tool usage counts, chat sessions, model performance metrics.

#### voice_clone_tab.py (923 lines)
**How it works:** Audio upload → audio_analyzer.py extracts features (pitch, rate, timbre) → voice_generator.py creates VoiceProfile dataclass → saved as JSON. Clone uses these parameters to modify TTS output. Preview plays sample with cloned voice settings.

### Container Tabs (organize other tabs)

#### avatar_tab.py (46 lines)
**How it works:** Just a QTabWidget container. Adds sub-tabs: avatar_display.py (shows avatar), game_connection.py (game controls), robot_control.py (robot commands). Routes to appropriate sub-tab based on user selection.

#### sessions_tab.py (178 lines)
**How it works:** Lists saved conversations from memory/manager.py. QListWidget sidebar shows session names by date. Clicking loads conversation JSON → displays in read-only QTextEdit. AI selector dropdown filters sessions by model.

---

## Core Package Summary - How Key Files Work

### Model & Inference

| File | How It Works |
|------|-------------|
| model.py | Transformer brain. Stacks N blocks: RMSNorm → self-attention (Q/K/V with RoPE encoding) → feedforward (SwiGLU). GQA shares K/V across heads. KV-cache stores past tokens. Output: vocabulary probability distribution. |
| inference.py | Generation loop: tokenize → run model → get probabilities → sample token (greedy/top-k/top-p) → append → repeat until EOS. Streaming yields tokens as generated. |
| training.py | Forward: compute cross-entropy loss. Backward: AdamW updates weights. Handles batching, gradient accumulation, lr warmup + cosine decay. |
| tokenizer.py | BPE encoder. Learns subword patterns. "playing" → ["play", "ing"]. Vocab 32K-100K. Special: [PAD], [EOS], [BOS]. |
| tool_router.py | Intent detector. Scans for keywords, routes to tools. Priority-ordered model assignments per tool. |

### Advanced Features

| File | How It Works |
|------|-------------|
| moe.py | Routes each token to 2 of N expert networks via gating function. Increases capacity without proportional compute. |
| ssm.py | Mamba/S4 architecture. State-space model processes sequences in O(n) vs O(n²) for attention. Good for very long sequences. |
| flash_attention.py | Memory-efficient attention. Computes in tiles, never materializes full NxN attention matrix. 2-4x faster. |
| infinite_context.py | Streaming context extension. Processes chunks, compresses old context into summary, maintains bounded memory for unbounded input. |
| paged_attention.py | KV-cache in pages. Allocates memory only as needed. Allows variable-length sequences without wasting memory on padding. |

### Quantization & Loading

| File | How It Works |
|------|-------------|
| quantization.py | Converts float32 → int8/int4. Calibrates with sample data to minimize accuracy loss. Reduces memory 4-8x. |
| awq_quantization.py | Activation-aware quantization. Identifies important weights, keeps them higher precision. Better than naive quantization. |
| gptq_quantization.py | Layer-by-layer quantization. Uses calibration data to find optimal rounding. Slower to quantize but better quality. |
| huggingface_loader.py | Downloads from HF Hub, converts weight names to Enigma format. Auto-detects architecture from config.json. |
| gguf_loader.py | Parses GGUF binary format. Reads metadata, dequantizes weights on-the-fly or keeps quantized for llama.cpp inference. |
| ollama_loader.py | Connects to local Ollama server. Translates Enigma API calls to Ollama API format. Model stays in Ollama process. |

### Training Variants

| File | How It Works |
|------|-------------|
| lora_training.py | Efficient fine-tuning. Adds small trainable A×B matrices to attention layers. Original weights frozen. 90%+ memory savings. |
| qlora.py | Quantized LoRA. Base model quantized to 4-bit, only LoRA adapters in fp16. Fine-tune 70B models on consumer GPU. |
| dpo.py | Direct Preference Optimization. Learns from preference pairs (good response vs bad response). No reward model needed. |
| rlhf.py | Reinforcement Learning from Human Feedback. Trains reward model on human rankings, then uses PPO to optimize policy. |
| distillation.py | Teacher-student training. Large teacher model generates soft labels, smaller student learns to match. Compress models. |

---

## Tools Package Summary - How Key Files Work

### Core Tool System

| File | How It Works |
|------|-------------|
| tool_executor.py | Parses `<tool_call>{JSON}</tool_call>`. Validates against schema. Runs with timeout (SIGALRM Unix/threading Windows). Blocks dangerous paths. Returns structured result. |
| tool_definitions.py | Schema registry. Each tool: name, description, parameters (name, type, required), handler function. Used for validation and AI prompt injection. |
| tool_registry.py | Central registry dict. Tools register on import. get_tool() looks up by name. list_tools() returns all available. |
| tool_manager.py | Enable/disable tools at runtime. Presets: minimal (5 tools), standard (15), full (all). Saves config to JSON. |

### Tool Categories

| Category | Files | How They Work |
|----------|-------|---------------|
| Vision | vision.py, simple_ocr.py | PIL captures screen → sent to vision model or tesseract for OCR → returns description/text |
| Web | web_tools.py, browser_tools.py | requests + BeautifulSoup parse pages. Rate limiting prevents abuse. Browser automation via playwright. |
| Files | file_tools.py, document_tools.py | Read/write with path blocking via security.py. PDF via PyPDF2, DOCX via python-docx. |
| System | system_tools.py | subprocess.run() with timeout. Captures stdout/stderr. Blocks dangerous commands. |
| Gaming | game_router.py, game_state.py | Detects game via window title → loads game config → routes through game-specific prompts |
| Robot | robot_tools.py, robot_modes.py | Abstracts hardware. GPIO for Pi, serial for Arduino. Actions: move, gripper, home. |

---

## Utils Package Summary - How Key Files Work

| File | How It Works |
|------|-------------|
| security.py | `is_path_blocked()` checks against blocklist (~/.ssh, /etc/passwd, system dirs). Called before all file operations. |
| lazy_import.py | LazyLoader replaces heavy imports with proxies. Actual import on first attribute access. 5-10x faster startup. |
| api_key_encryption.py | Fernet (AES-128) encryption. Master key from machine ID + password. Keys stored encrypted in ~/.enigma_engine/keys. |
| battery_manager.py | Monitors power state via psutil. Adjusts behavior on battery (reduce batch size, disable GPU). |
| backup.py | Copies models/data to timestamped backup folder. Compression optional. Restore copies back. |

---

## File Dependencies

### Core Module Dependencies
```
model.py ← inference.py ← chat_tab.py
         ← training.py ← training_tab.py
         ← modules/registry.py

tokenizer.py ← training.py
             ← inference.py

tool_router.py ← inference.py (with use_routing=True)
              ← model_router_tab.py
```

### GUI Tab Dependencies
```
All tabs ← enigma_engine/config/ (CONFIG)
All tabs ← gui/shared_components.py

chat_tab ← core/inference.py, memory/manager.py
settings_tab ← config/, utils/api_key_encryption.py
modules_tab ← modules/manager.py, modules/registry.py
training_tab ← core/training.py
image_tab ← gui/tabs/output_helpers.py
audio_tab ← builtin/, voice/
voice_clone_tab ← voice/*
character_trainer_tab ← tools/data_trainer.py
network_tab ← comms/discovery.py
persona_tab ← core/persona.py
dashboard_tab ← psutil (external)
```

### External Dependencies
- **PyTorch** - All AI operations
- **PyQt5** - All GUI components
- **psutil** - System monitoring (dashboard_tab)
- **pyttsx3** - Local TTS (audio_tab)
- **requests** - HTTP requests
- **transformers** - HuggingFace model loading
- **diffusers** - Stable Diffusion (image_tab)

---

## Statistics

| Category | Count |
|----------|-------|
| **GUI** | |
| Total GUI Tabs | 44 |
| GUI Lines (estimated) | ~45,000 |
| Largest Tab | settings_tab.py (4,721 lines) |
| **Core** | |
| Core Files | ~170 |
| Model Presets | 15 (nano → omega) |
| **Tools** | |
| Tool Files | ~70 |
| Tool Categories | 8 |
| **Utils** | |
| Utils Files | ~80 |
| **Total** | |
| **Python Files** | **816** |
| **Total Lines** | **458,255** |
| Dead Code Removed | 15 files (~8,000 lines) |

---

*Report generated by Enigma AI Engine Code Analysis - February 15, 2026*
