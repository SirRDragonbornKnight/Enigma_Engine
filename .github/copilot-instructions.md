# Enigma Engine - AI Coding Guidelines

## Architecture Overview
Enigma Engine is a modular AI framework with isolated components for easy replacement:
- **enigma.core**: Transformer-based model ([enigma/core/model.py](enigma/core/model.py)), training ([enigma/core/training.py](enigma/core/training.py)), inference ([enigma/core/inference.py](enigma/core/inference.py)), and tokenization ([enigma/core/tokenizer.py](enigma/core/tokenizer.py)).
- **enigma.memory**: Conversation storage in JSON ([enigma/memory/manager.py](enigma/memory/manager.py)) and SQLite ([enigma/memory/memory_db.py](enigma/memory/memory_db.py)), with vector search ([enigma/memory/vector_utils.py](enigma/memory/vector_utils.py)).
- **enigma.comms**: Flask API server ([enigma/comms/api_server.py](enigma/comms/api_server.py)) and remote client ([enigma/comms/remote_client.py](enigma/comms/remote_client.py)).
- **enigma.gui**: PyQt5 interface ([enigma/gui/enhanced_window.py](enigma/gui/enhanced_window.py)) with Chat, Train, Avatar, Vision, History, and Files tabs.
- **enigma.voice**: TTS/STT wrappers ([enigma/voice/stt_simple.py](enigma/voice/stt_simple.py), [enigma/voice/tts_simple.py](enigma/voice/tts_simple.py)).
- **enigma.avatar**: Avatar control stub ([enigma/avatar/avatar_api.py](enigma/avatar/avatar_api.py)).
- **enigma.tools**: System utilities ([enigma/tools/system_tools.py](enigma/tools/system_tools.py)) and vision ([enigma/tools/vision.py](enigma/tools/vision.py)).

Configuration is centralized in [enigma/config.py](enigma/config.py) as a CONFIG dict, setting paths for data, models, and DB.

## Developer Workflows
- **Setup**: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt` (Linux/Mac).
- **Train Model**: `python run.py --train` (uses data files in models/[name]/data/).
- **Run Inference**: `python run.py --run` for CLI demo; API via `python run.py --serve` (Flask on 127.0.0.1:5000).
- **GUI**: `python run.py --gui` (requires PyQt5).
- **Examples**: See [examples/](examples/) for basic usage ([examples/run_example.py](examples/run_example.py), [examples/train_example.py](examples/train_example.py)).

Torch is optional; install manually if needed for local model training/inference.

## Conventions and Patterns
- **Imports**: Relative imports within enigma package (e.g., `from ..config import CONFIG`).
- **Paths**: Use `pathlib.Path` for file operations; directories auto-created via CONFIG.
- **Model**: TinyEnigma is a toy transformer; replace with production models (Hugging Face, etc.) in [enigma/core/](enigma/core/).
- **Memory**: Conversations saved as JSON in data/conversations/; vectors use simple cosine similarity.
- **API**: Single /generate endpoint with prompt, max_gen, temperature params.
- **GUI**: Threads for STT to avoid blocking UI ([enigma/gui/enhanced_window.py](enigma/gui/enhanced_window.py)).
- **Entry Point**: [run.py](run.py) with argparse flags; setup.py defines 'enigma' console script.

Replace stubs (TTS/STT, avatar) with real implementations as needed. No formal tests; validate by running demos.</content>
<parameter name="filePath">c:\Users\sirkn_gbhnunq\Documents\GitHub\enigma_engine\.github\copilot-instructions.md