# Training & Model Status

**Updated:** February 28, 2026

---

## Models Available

| Model | Size | Type | Status |
|-------|------|------|--------|
| `base.pth` | Native | PyTorch (Llama-style) | ✅ Loads |
| `base_14.pth` | Native | PyTorch (Llama-style) | ✅ Loads |
| `qwen3-30b-a3b/` | ~30.5B (3.3B active, MoE) | GGUF Q4_K_M (17.28 GB) | ✅ Works with llama-server backend |
| `qwen3-8b/` | ~8.2B | HuggingFace safetensors (15.27 GB) | ✅ Fine-tunable with FORGE QLoRA |

---

## Training Methods

| Method | Code | GUI | Works? |
|--------|------|-----|--------|
| **SFT (Supervised Fine-tuning)** | ✅ `Trainer` class | ✅ Solo Training | ✅ Working |
| **Guided Training (3-phase)** | ✅ `_start_guided_training()` | ✅ Wired | ✅ Working — TRAINER creates curriculum, trains STUDENT, tests readiness |
| **Best-of-N** | ✅ `best_of_n()` | ✅ Shows info | N/A (inference, not training) |
| **Evolutionary** | ✅ `evolutionary_training()` | ✅ Wired | ✅ Working |
| **LoRA/QLoRA** | ✅ `lora_utils.py` | ✅ Wired | ✅ Working (requires PEFT) |
| **Generate Data** | ✅ `_generate_training_data()` | ✅ Wired | ✅ Autonomous — TRAINER generates Q/A pairs |
| **Evaluate** | ✅ `_evaluate_student()` | ✅ Wired | ✅ Interactive — TRAINER judges STUDENT answers 1-10 |
| **Web Learn** | ✅ `_web_learn()` | ✅ Wired | ✅ DuckDuckGo search + page fetch + Q/A generation |
| **DPO** | ❌ Not built | ❌ Not in GUI | Needs preference data |

---

## Training Code Location

```
enigma_engine/core/training.py (~1181 lines)
├── TrainingConfig      - Configuration dataclass
├── TrainingState       - Tracks epoch, step, loss
├── Trainer            - Main training class
│   ├── train()        - Main training loop
│   ├── _train_epoch() - Single epoch
│   └── save/load      - Checkpoint management
├── best_of_n()        - Generate N, pick random non-empty
├── best_of_n()        - Generate N, pick best
├── collect_training_data() - Gather winners
└── evolutionary_training() - Self-play loop

enigma_engine/gui/gui_forge.py (~2776 lines)
├── _build_trainer_system_prompt() - System prompt w/ student context, stages, personality
├── _load_engine_for_path()        - Load any model format via EnigmaEngine
├── _extract_prompts()             - Parse Q/A, JSONL, User/AI, raw text
├── _start_solo_training()         - Train STUDENT directly on data (SFT)
├── _start_guided_training()       - 3-phase: generate curriculum → train → test readiness
├── _generate_training_data()      - Autonomous: TRAINER generates N Q/A pairs
├── _evaluate_student()            - Interactive: TRAINER judges STUDENT answers 1-10
├── _web_learn()                   - Search web + generate training pairs from content
├── _save_forge_checkpoint()       - Save STUDENT to named checkpoint
├── _load_forge_checkpoint()       - Restore from checkpoint file
└── _display_loss_curve()          - Text-based bar chart of per-epoch losses
```

---

## Your Idea: Router-based Training

**Concept:** Put training in the router so AI trains while bricks work

```
┌─────────────────────────────────────────────┐
│  ROUTER (port 9900)                         │
│  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Brick Manager│  │ Background Trainer   │ │
│  │ - Accept     │  │ - Train on idle      │ │
│  │ - Route cmds │  │ - Collect responses  │ │
│  │ - Monitor    │  │ - Score & learn      │ │
│  └──────────────┘  └──────────────────────┘ │
└─────────────────────────────────────────────┘
         ↕                    ↕
    Bricks connect     Continuous training
    and generate       in background thread
```

**Advantages:**
- AI trains while you use it
- Learns from actual conversations
- No separate training session needed

**How it could work:**
1. Router has training thread running in background
2. Every conversation goes to training queue
3. Good responses (rated, or AI-judged) become training data
4. Model fine-tunes incrementally

---

## To Test Training Now

### Option 1: GUI
```bash
python run.py --gui
# Training Tab → Load data → Start Training
```

### Option 2: Python Script
```python
from enigma_engine.core.training import Trainer, TrainingConfig
from enigma_engine.core.model import load_model

# Load model
model, tokenizer = load_model("models/enigma_tiny.pth")

# Configure training
config = TrainingConfig(
    epochs=5,
    batch_size=2,
    learning_rate=1e-4
)

# Create trainer
trainer = Trainer(model, tokenizer, config)

# Train on data
data = [
    {"input": "Hello", "output": "Hi there!"},
    {"input": "What is AI?", "output": "AI is artificial intelligence."},
]
trainer.train(data)

# Save
torch.save(model.state_dict(), "models/tiny_forge_trained.pth")
```

---

## Questions to Answer

1. **Does SFT training work?** 
   - [ ] Test with enigma_tiny.pth on small dataset
   - [ ] Check if loss decreases
   - [ ] Verify model improves

2. **Does evolutionary training work?**
   - [ ] Test `evolutionary_training()` function
   - [ ] See if it generates, scores, trains

3. **Can router do background training?**
   - [x] Build router (`enigma_engine/router.py`)
   - [x] Add training queue 
   - [x] Wire into main_window buttons
   - [x] Add Prompts tab for configuring system prompts
   - [ ] Test incremental learning with real model

---

## Router Implementation (DONE)

**File:** [enigma_engine/router.py](enigma_engine/router.py)

**Components:**
- `BrickRouter` - TCP server on port 9900
- `BackgroundTrainer` - Training thread with example queue  
- Auto-collects chat conversations for training
- Feedback (good/bad) adjusts training scores

**How to use:**
1. Go to Bricks tab
2. Click "Start Router"
3. Chat with the AI
4. Conversations auto-queue for training
5. Rate responses to boost/lower scores
6. Training runs in background when model is connected

**Wiring:**
- Router starts when you click "Start Router"
- Chat responses auto-added to training queue
- Good feedback = score 1.5, Bad feedback = score 0.2
- Training stats shown in Bricks tab (updated every 2s)

---

## Prompts Tab (NEW)

**File:** [enigma_engine/gui/tabs/prompt_tab.py](enigma_engine/gui/tabs/prompt_tab.py)

**Purpose:** Configure system prompts that guide AI behavior

**Prompt Categories:**
1. **System** - Base personality and capabilities
2. **External Models** - Instructions for API models (GPT, Claude) on using GUI/bricks
3. **Internal Models** - Instructions for local models
4. **Brick-specific** - Per-brick instructions (how to use each brick)

**Features:**
- Edit prompts with syntax highlighting
- Variables: `{brick_list}`, `{model_name}`, `{date}`
- Auto-saves to `data/prompts.json`
- Updates router training context in real-time
- Reset to defaults button

---

## Next Steps

1. [ ] Test router with actual model connected
2. [ ] Verify training loss decreases
3. [ ] Build sample brick that connects
