# Training & Model Status

**Updated:** February 19, 2026

---

## Models Available

| Model | Size | Type | Status |
|-------|------|------|--------|
| `enigma_small.pth` | 106 MB | PyTorch (Llama-style) | ✅ Loads |
| `enigma_tiny.pth` | 20 MB | PyTorch (Llama-style) | ✅ Loads |
| `qwen2.5-32b-instruct/` | ~20 GB | GGUF | ✅ Works with templates |

---

## Training Methods

| Method | Code | GUI | Works? |
|--------|------|-----|--------|
| **SFT (Supervised Fine-tuning)** | ✅ `Trainer` class | ✅ Wired | 🔨 Untested |
| **Best-of-N** | ✅ `best_of_n()` | ✅ Shows info | N/A (inference, not training) |
| **Evolutionary** | ✅ `evolutionary_training()` | ✅ Wired | 🔨 Untested |
| **LoRA/QLoRA** | ❌ Not built | ❌ Shows "Coming Soon" | Needs PEFT library |
| **DPO** | ❌ Not built | ❌ Shows "Coming Soon" | Needs preference data |

---

## Training Code Location

```
enigma_engine/core/training.py (926 lines)
├── TrainingConfig      - Configuration dataclass
├── TrainingState       - Tracks epoch, step, loss
├── Trainer            - Main training class
│   ├── train()        - Main training loop
│   ├── _train_epoch() - Single epoch
│   └── save/load      - Checkpoint management
├── score_response()   - Score output quality
├── best_of_n()        - Generate N, pick best
├── collect_training_data() - Gather winners
└── evolutionary_training() - Self-play loop
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
