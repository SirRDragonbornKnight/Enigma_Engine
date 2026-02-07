# Enigma AI Engine Suggestions

Remaining improvements for the Enigma AI Engine codebase.

**Last Updated:** February 7, 2026

---

## Trainer AI for Generating Training Data

### Pretrained Base Model for Distribution
- [ ] **Create base model for GitHub releases**:
  - Train "small" (~27M params) model on curated dataset
  - Include in releases so users don't start from scratch
  - Document what it was trained on

---

## Router + ModuleManager Integration

### Connect Tool Router to Module System
- [ ] **Have router check ModuleManager for capability availability**:
  - Router should query ModuleManager to see which modules are loaded
  - Route to available providers based on loaded modules (e.g., `image_gen_local` vs `image_gen_api`)
  - Fall back gracefully when a module isn't loaded
  - Currently router uses `get_provider()` from tabs which works, but deeper integration would be cleaner

### Benefits
- Single source of truth for what's available
- Better error messages ("Image generation not available - enable in Modules tab")
- Dynamic routing based on loaded modules
- Cleaner architecture

---

## Completed Items (Archived)

<details>
<summary>Click to expand completed suggestions</summary>

### First-Run Setup Wizard (Done)
- [x] Fix "Unknown" CPU bug - Added fallbacks using `os.cpu_count()` and `platform.processor()`
- [x] AI Name Input Page - Added wizard page after Welcome with name suggestions

### Training Data Generator (Done)
- [x] Training Data Generator Tab - New "Data Gen" tab in MY AI section:
  - Load HuggingFace models (TinyLlama, Phi-2, Phi-3, Mistral, Qwen)
  - Input topics/domains to generate Q&A pairs
  - Support for Q&A, conversations, and instructions formats
  - Save in Enigma training format, JSONL, or JSON

### Already Implemented Features
- [x] Hardware detection with `DeviceProfiler` (device_profiles.py)
- [x] Model size recommendations based on hardware
- [x] HuggingFace model loading (huggingface_loader.py)
- [x] Model hub with download/upload (hub/model_hub.py)
- [x] Stop training functionality (enhanced_window.py)
- [x] Checkpoint saving every N epochs (training.py)
- [x] AI name through persona system
- [x] Data Trainer with CharacterTrainer and TaskTrainer
- [x] Character Trainer Tab with UI
- [x] Training data generation tools

### Code Quality (Done)
- [x] Replace QThread.terminate() with cooperative flags
- [x] Clean up temporary attachment files

### Module System (Done)
- [x] Module dependency visualization
- [x] Module conflict detection
- [x] Module profiles

</details>
