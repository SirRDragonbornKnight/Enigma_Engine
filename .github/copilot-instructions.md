# Enigma Engine - AI Coding Guidelines

## Architecture Overview
Enigma Engine is a modular AI framework with isolated components for easy replacement:
- **enigma.core**: Production-grade `Enigma` transformer model with RoPE, RMSNorm, SwiGLU, KV-cache ([enigma/core/model.py](enigma/core/model.py)), training with AMP and gradient accumulation ([enigma/core/training.py](enigma/core/training.py)), inference with streaming and chat support ([enigma/core/inference.py](enigma/core/inference.py)), and character-level tokenization with dictionary support ([enigma/core/tokenizer.py](enigma/core/tokenizer.py)).
- **enigma.memory**: Conversation storage in JSON ([enigma/memory/manager.py](enigma/memory/manager.py)) and SQLite ([enigma/memory/memory_db.py](enigma/memory/memory_db.py)), with vector search ([enigma/memory/vector_utils.py](enigma/memory/vector_utils.py)).
- **enigma.comms**: Flask API server ([enigma/comms/api_server.py](enigma/comms/api_server.py)), remote client ([enigma/comms/remote_client.py](enigma/comms/remote_client.py)), and multi-device communication ([enigma/comms/network.py](enigma/comms/network.py)).
- **enigma.gui**: PyQt5 interfaces - basic ([enigma/gui/main_window.py](enigma/gui/main_window.py)) and enhanced ([enigma/gui/enhanced_window.py](enigma/gui/enhanced_window.py)) with chat, history, training, data editor, avatar, vision, terminal, and models tabs.
- **enigma.voice**: TTS/STT wrappers ([enigma/voice/stt_simple.py](enigma/voice/stt_simple.py), [enigma/voice/tts_simple.py](enigma/voice/tts_simple.py)).
- **enigma.avatar**: Avatar control ([enigma/avatar/avatar_api.py](enigma/avatar/avatar_api.py), [enigma/avatar/controller.py](enigma/avatar/controller.py)).
- **enigma.tools**: Vision system ([enigma/tools/vision.py](enigma/tools/vision.py)), web tools, file tools, document tools, and more.

Configuration is centralized in [enigma/config.py](enigma/config.py) as a CONFIG dict, setting paths for data, models, and DB.

## Developer Workflows
- **Setup**: `python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt` (Windows).
- **Train Model**: `python run.py --train` (uses data in data/data.txt or data/user_training.txt).
- **Run Inference**: `python run.py --run` for CLI demo; API via `python run.py --serve` (Flask on 127.0.0.1:5000).
- **GUI**: `python run.py --gui` (requires PyQt5) - opens enhanced GUI with all features.
- **Examples**: See [examples/](examples/) for basic usage and multi-device setup.
- **Documentation**: See [docs/CODE_TOUR.md](docs/CODE_TOUR.md) for codebase walkthrough.

Torch is optional; install manually if needed for local model training/inference.

## Conventions and Patterns
- **Imports**: Relative imports within enigma package (e.g., `from ..config import CONFIG`).
- **Paths**: Use `pathlib.Path` for file operations; directories auto-created via CONFIG.
- **Model**: `Enigma` is the production model with RoPE, RMSNorm, SwiGLU. `TinyEnigma` is a backwards-compatible alias. Model sizes from `tiny` (~2M) to `xxxl` (~1.5B).
- **Tokenizer**: Character-level with dictionary support (~3000 common words). Add words to `enigma/vocab_model/dictionary.txt`.
- **Memory**: Conversations saved as JSON in data/conversations/; vectors use simple cosine similarity.
- **API**: Single /generate endpoint with prompt, max_gen, temperature params.
- **GUI**: Threads for STT to avoid blocking UI. Enhanced GUI has terminal tab for debugging.
- **Entry Point**: [run.py](run.py) with argparse flags; setup.py defines 'enigma' console script.

## What NOT To Do
❌ **DON'T train with very small datasets** - Need at least 1000 lines for basic results, 10,000+ for good results.

❌ **DON'T use extremely high learning rates** - Default 0.0001 is safe. Higher than 0.01 can break training.

❌ **DON'T delete model folders manually** - Use Model Manager instead. Registry needs to be updated.

❌ **DON'T train during inference** - Close chat before training, or use separate model instances.

❌ **DON'T run multiple instances on same model** - Can corrupt model files. Use different model names.

❌ **DON'T forget to save training data** - Editor doesn't auto-save. Click Save before training.

❌ **DON'T modify files in vocab_model/ directly** - Use tokenizer.add_word() or edit dictionary.txt.

❌ **DON'T hardcode IP addresses** - Use CONFIG or environment variables for network settings.

❌ **DON'T block the UI thread** - Use QThread for long operations (training, STT, network calls).

❌ **DON'T mix model sizes carelessly** - Growing/shrinking models requires proper conversion.

Replace stubs (TTS/STT, avatar) with real implementations as needed. Validate by running demos.</content>
<parameter name="filePath">c:\Users\sirkn_gbhnunq\Documents\GitHub\enigma_engine\.github\copilot-instructions.md