# Training & Model Status

**Updated:** March 25, 2026

---

## Models Available

| Model | Size | Type | Status |
|-------|------|------|--------|
| `base.pth` | Native | PyTorch (LLaMA-style) | Loads |
| `qwen3-30b-a3b/` | ~30.5B (3.3B active, MoE) | GGUF Q4_K_M (17.28 GB) | Works with llama-server backend |
| `qwen3-8b/` | ~8.2B | HuggingFace safetensors (15.27 GB) | Fine-tunable with FORGE QLoRA |

---

## Training Methods

All methods are fully implemented and wired to the GUI.

### Core Training (`training.py`)

| Method | Description | GUI Mode |
|--------|-------------|----------|
| SFT | Supervised fine-tuning on text/Q&A/JSONL data | Solo Training |
| DPO | Direct Preference Optimization from chosen/rejected pairs | DPO Training |
| Vision | Vision encoder + projection on image-text pairs | Vision Training |
| Audio | Audio encoder + projection on audio-text pairs | (via code) |

### RL Training (`rl_training.py`)

| Method | Description | GUI Mode |
|--------|-------------|----------|
| Reward Model | Train reward model from preference rankings | RLHF (phase 1) |
| RLHF/PPO | Policy gradient with KL penalty + reward model | RLHF (phase 2) |
| Self-Play | TRAINER scores STUDENT as reward signal | Self-Play |

### Adapter Training (`lora_utils.py`)

| Method | Description | GUI Mode |
|--------|-------------|----------|
| LoRA/QLoRA | Low-rank adapter fine-tuning (~75% VRAM savings) | LoRA Training |

### GUI Advanced Modes

| Method | Description | File |
|--------|-------------|------|
| Guided Training | AI curriculum: generate lessons, train, evaluate | `gui_forge_advanced.py` |
| Dialogue Training | TRAINER-STUDENT conversation with corrections | `gui_forge_advanced.py` |
| Adaptive Training | Autonomous pipeline (TC-C3, SA-B, SA-C), resumable | `gui_forge_adaptive.py` |
| Pre-Train | Train from scratch on large corpus | `gui_forge_new_modes.py` |
| Distillation | TRAINER generates targeted data for STUDENT | `gui_forge_new_modes.py` |

### GUI Tools

| Tool | Description | File |
|------|-------------|------|
| Generate Data | TRAINER creates Q/A pairs autonomously | `gui_forge_tools.py` |
| Evaluate | TRAINER judges STUDENT answers | `gui_forge_tools.py` |
| Web Learn | DuckDuckGo search → page fetch → Q/A generation | `gui_forge_tools.py` |
| Benchmark | Coherence scoring against test prompts | `gui_forge_tools.py` |
| Tokenizer Training | BPE tokenizer training from text data | `gui_forge_tools.py` |

### CLI Training (`run.py`)

| Command | Description |
|---------|-------------|
| `python run.py --train data/training.txt --epochs 10` | Train model on data |
| `python run.py --train-tokenizer data/training.txt` | Train BPE tokenizer |
| `python run.py --benchmark` | Run coherence benchmark |

---

## Training Code Location

```
enigma_engine/core/
├── training.py           # Trainer (SFT, DPO, vision, audio, evolutionary)
├── rl_training.py        # RewardTrainer, RLHFTrainer, SelfPlayTrainer
├── lora_utils.py         # LoraTrainer (LoRA/QLoRA via PEFT)
├── adaptive_trainer.py   # Adaptive training pipeline (TC-C3, SA-B, SA-C)
├── training_evaluation.py# Evaluation utilities
├── training_monitor.py   # Thread-safe training monitor
├── training_queue.py     # Training queue with crash recovery
├── curated_dataset.py    # Thread-safe curated dataset management
├── dataset.py            # Dataset loading and Unicode handling
└── progressive_growing.py# Net2Net model expansion (width + depth)

enigma_engine/gui/
├── gui_forge.py          # FORGE hub: shared utilities, dispatch
├── gui_forge_training.py # Solo, DPO, Vision, LoRA modes
├── gui_forge_advanced.py # Guided (3-phase), Dialogue modes
├── gui_forge_adaptive.py # Adaptive pipeline (continuous loop)
├── gui_forge_new_modes.py# Pre-Train, Distill, RLHF, Self-Play
├── gui_forge_tools.py    # Data gen, evaluate, web learn, benchmark
├── gui_forge_models.py   # Model import, create, copy, rename, delete
└── gui_forge_queue.py    # Training queue, overnight plan, curated dataset
```
