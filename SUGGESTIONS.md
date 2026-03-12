# Suggestions (Active Only)

**Date:** March 11, 2026
**Status:** FORGE supplement/stage/perplexity wiring complete and suggestions audited against current code. Next: remove remaining legacy FORGE drift, then wire tool metrics into history/model context.

## Current Reality
- FORGE user contract is 3 modes: `Basic`, `AI-Guided`, `Image`.
- Backend still contains advanced methods, but they are not part of the main FORGE UI contract.
- CORE processing indicator no longer shifts layout.
- The CONFIG gaming preset now activates a real low-overhead runtime profile: chat-learning off, router trainer off, slower UI/status timers, exact param counting skipped, and minimize unload enabled.
- Chat RAM history is capped and trimmed; full sessions still save to disk.
- AI-Guided supplement selection is now wired to `ai_supplement_var` (was incorrectly reading Basic trainer's data picker).
- Stage selection now controls pipeline start stage ("start here, then continue forward").
- Perplexity before/after is now persisted to `training_history.json` and model context; displayed in history log.
- Generate Data / Web Learn auto-train now routes output to the active mode selector (AI-Guided supplement vs Basic data).
- Adaptive start path no longer has the undefined `focus_field` runtime risk.

## What Still Needs Fixing
1. Remove remaining legacy FORGE constants and compatibility baggage in `gui_forge.py` so the code shape matches the real 3-mode UI contract.
2. Add tool success-rate tracking to evaluation/history so command-learning quality is measurable alongside perplexity.
3. Tighten FORGE copy and reference docs so they describe the current runtime behavior exactly, especially AI-Guided auto-chaining.
4. Decide whether backend-only training modes should stay as hidden implementation detail or move to a separate advanced workflow instead of lingering in the main FORGE code path.

## Confirmed Fixes (In Use)
- Replaced legacy multi-mode FORGE UI with 3 clear modes.
- Added auto-LoRA routing for larger models (> 7B params).
- Added before/after evaluation logging (perplexity improvement).
- Fixed duplicate FORGE section rendering bugs.
- Gaming preset now disables live background learning/trainer overhead instead of only changing next-launch defaults.
- Fixed CORE processing label movement at two levels:
  - Grid control column fixed width (`minsize=140`).
  - Fixed-size `SelectableLabel` no longer resizes on text updates.
- Added active memory retrieval command: `memory.search <query>`.
- **AI-Guided supplement picker now reads from `ai_supplement_var` not Basic data picker.**
- **Stage selection now slices `ALL_STAGES` from selected stage onward.**
- **Perplexity (before/after) persisted to training history and model context.**

## Memory Strategy (Chosen)
- **Now:** Option A (current hybrid)
  - RAM: capped recent chat history
  - Disk: full session archive
  - Facts: explicit `remember/search`
- **Next:** Option D
  - Add summarized long-term session memory + recent raw turns.
- **Later (optional):** Option E
  - User-selectable memory behavior modes.

## Next Priority
1. Remove remaining legacy FORGE drift in `gui_forge.py` so the code matches the visible 3-mode contract instead of carrying old 9-mode config baggage.
2. Persist tool success-rate metrics in training/evaluation history so command-learning progress can be measured, not guessed.
3. Build Discovery mode MVP for autonomous mod exploration only after the FORGE contract is fully clean.

## What I Would Do Next
1. Prune or isolate old FORGE-only legacy constants and dead compatibility paths that are no longer part of the visible `Basic` / `AI-Guided` / `Image` workflow.
2. Add a real tool-usage evaluation pass that records success-rate alongside perplexity in history and model context.
3. Tighten the FORGE page copy so every visible label describes exactly what happens at runtime, especially around AI-Guided auto-chaining.
4. Leave backend-only modes (`DPO`, `RLHF`, `SelfPlay`, `Evolutionary`) implemented but clearly out of the main FORGE contract unless they get their own intentional UI again.

---

## Training Strategy — Teaching a Native Model to Use the GUI

**The real problem:** A native (custom) model trained on plain Q&A has no idea what
`[CMD]file.write[/CMD]` means, when to use it, or when NOT to use it. It can
accidentally output `[CMD]stop[/CMD]` just because the word "stop" appeared in
conversation. That needs to be fixed by training, not just by runtime guards.

**The good news:** The infrastructure to do this already exists. You do not need a new school system.
The 4 FORGE stages (BASICS → CONVERSATION → COMMANDS → WEB) are exactly the school regimen.
The issue is the training *data* and *format*, not the training *machine*.

---

### What to do and why it will work

#### 1. Use the 4-stage curriculum as the actual regimen (works now, no code changes)

The stages are already ordered from simplest to hardest:
- **BASICS** — normal language, greetings, short answers. No commands yet.
- **CONVERSATION** — multi-turn natural dialogue. Still no commands.
- **COMMANDS** — teaches `[CMD]command[/CMD]` syntax with the right situations.
- **WEB** — adds search.web + web.fetch on top of everything learned.

Do NOT skip to COMMANDS first. The model needs BASICS + CONVERSATION behind it
or it will learn command syntax without knowing how to hold a conversation around it.
Each stage can be repeated with a different "Focus" topic as many times as needed.
Training has no hard cap — you run FORGE as many times as needed.

**Reality note (March 10, 2026):** Stage buttons now control the adaptive start stage
("start here, then continue forward"). The pipeline still intentionally auto-chains
remaining stages after the selected start point.

**How to use today:**
1. Load the native model as STUDENT, HF model (e.g. Qwen3-30b) as TRAINER.
2. Run AI-Guided mode with a clear topic/goal and review the generated data in DOCS.
3. Do not assume the selected stage button limits the run yet — verify what curriculum was actually generated.
4. For command behavior, use generated COMMANDS-style data and DPO files explicitly.
5. Use GENERATE DATA to let the HF trainer read the commands reference and produce examples.

#### 2. Use DPO mode to fix accidental command firing (works now)

DPO (Direct Preference Optimization) is already in FORGE (Preference Tuning mode).
It directly teaches: "prefer this output over that one."

Create a `.jsonl` file where:
- `prompt` = a user message where the model might accidentally fire a command
- `chosen` = the correct plain-text response
- `rejected` = the bad response that contained an accidental `[CMD]` block

Example entries:
```json
{"prompt": "Can you stop for a second?", "chosen": "Of course, I'll pause.", "rejected": "[CMD]stop[/CMD] Pausing now."}
{"prompt": "The answer is 7.", "chosen": "Yes, that's correct.", "rejected": "[CMD]7[/CMD]"}
{"prompt": "Write that down.", "chosen": "What would you like me to write?", "rejected": "[CMD]file.write note.txt Write that down.[/CMD]"}
```

50–200 of these entries will meaningfully reduce accidental command output.
This file can be generated by the HF trainer using GENERATE DATA with Focus =
"Examples of when NOT to use commands, with bad and good responses."
Then run DPO training on that file.

**This is the realistic fix for the random-command problem. Training alone tells it when to stop.**

#### 3. Auto-generate command training data from the docs (works now)

The HF trainer already generates data when you use AI-Guided mode + GENERATE DATA.
To make it generate *command-aware* examples, set the Focus field to something like:

> "The AI knows these commands: [list the 5-6 most important ones]. Generate training
> examples showing correct use of these commands when the user asks for them, and
> examples showing the AI NOT using commands when not asked."

The `information/commands_reference.md` file documents all 47 commands.
You can have the AI read it with `[CMD]file.read information/commands_reference.md[/CMD]`
in a chat session, then ask it to generate training pairs. Save those to a `.jsonl` file
and use it in FORGE. This needs no code changes, just an intentional workflow.

**Current limitation:** The dedicated old focus-field widget is intentionally removed.
Use Training Topic + Training Brief fields for domain and style control.

#### 4. Two-tier context: HF Trainer gets mechanics, Student gets persona (needs ~1 day of code)

**The problem:** When a native model is running as STUDENT, it still gets the full system
prompt including internal evaluation logic meant for the TRAINER. That's unnecessary noise.

**The solution:** When FORGE detects that the STUDENT is a native model (lives in `models/`
not in `data/model_contexts/`) and the TRAINER is an HF model (lives in `data/model_contexts/`),
load different system prompts:
- TRAINER: current full mechanics prompt (it needs to understand evaluation, scoring, etc.)
- STUDENT: a lean persona prompt only ("You are a helpful assistant..." — no internal mechanics)

**Implementation approach:**
- Check the model path at FORGE start: `if "model_contexts" in trainer_path` → full prompt
- Add a `profiles/student_default.json` with a minimal persona prompt
- Load it automatically when STUDENT is a native model

**This is not urgent.** The real learning quality comes from good training data (points 1–3 above).
The two-tier context is a polish step once training is regularly producing results.

---

### What is NOT realistic to expect

- **A native model will not "become human."** Fine-tuning shapes behavior but the model's
  core capability is fixed by its architecture and pretraining. A small native model fine-tuned
  on great data will be *consistent and useful*, not indistinguishable from a person.
- **One training run is not enough.** Budget for 3–5 cycles of generate → review → train → evaluate.
  Each cycle improves it. FORGE's before/after perplexity logging will show if it's improving.
- **Training cannot fully prevent all command accidents.** DPO reduces them,
  but the confirmation dialog (`file.write`, `file.append`) is the real safety net.
  Keep both.

---

### Recommended Order of Work

1. **(Now, no code)** — Run BASICS → CONVERSATION in AI-Guided mode. Get the native model
   conversational first.
2. **(Now, ~1 hour)** — Create a 50-entry DPO file for command accidents using the HF
   trainer. Run Preference Tuning.
3. **(Now, ~2 hours)** — Run COMMANDS stage with focus on the 10 most important commands
   (file, search, memory, model). Review/approve data in DOCS. Train.
4. **(Later, ~1 day)** — Build two-tier context loading so the STUDENT gets a clean persona
   prompt instead of the full TRAINER context.
5. **(Later, ~2 days)** — Add a "Command Policy Generator" button to FORGE that auto-reads
   `commands_reference.md` and generates DPO pairs for all registered commands in bulk.

### Recommended Order of Fixes

1. **(Now, code)** — Remove remaining legacy FORGE drift from `gui_forge.py` and related references so there is one truthful contract.
2. **(Now, code/docs)** — Persist tool success-rate metrics in training/evaluation history.
3. **(Now, docs)** — Keep docs aligned with the 3-mode FORGE contract and remove stale legacy references.
4. **(Soon, code)** — Build Discovery mode MVP for autonomous mod exploration.
5. **(After Discovery MVP)** — Evaluate whether adaptive progression needs true score-gating vs explicit observational wording.

## Implementation Guardrails
- Verify current code/tests before each behavior change.
- Keep docs aligned with actual code behavior.
- Pass lint + tests for every merge.

## Notes
- Removed stale planning sections and speculative alternatives that are not currently used.
- This file is now intentionally short and only tracks active decisions and near-term execution.
- Docs synced on March 10, 2026: `SUGGESTIONS.md`, `AA code maker.md`, and `GUI_REFERENCE.md` now reflect the same processing-indicator root fix and memory direction.

## Checked Against Code

### Already True
- Main trainer cosine scheduling already uses a non-zero floor (`eta_min = learning_rate * 0.1`).
- Main trainer weight decay is already configurable and bias/norm parameters are excluded from decay.
- Before/after perplexity is already persisted to `training_history.json` and model context.
- RMSNorm fp32 upcast in both `model_components.py` and `vision_encoder.py` (commit `6a23413`).
- AdamW betas/eps configurable in `TrainingConfig` and wired to main + vision + LoRA optimizers.
- LoRA `weight_decay` is now a configurable `__init__` param (was hardcoded `0.01`).

### Partially Done
- Tool-usage evaluation helper exists in core, but it is not yet wired into FORGE history, model context, or the history viewer.

### Real Open Backlog
1. ~~RMSNorm fp32 upcast in both text and vision paths.~~ Done.
2. ~~AdamW betas and eps configurable across trainer, LoRA, and related optimizer creation paths.~~ Done.
3. ~~Special token ID unification across tokenizer implementations.~~ Done — all tokenizers expose `think_start_id`/`think_end_id`.
4. Sequence packing to reduce padding waste.
5. ~~Validation loop with val loss and checkpoint selection by validation.~~ Done — `val_split` config, per-epoch val loop, best checkpoint by val_loss.
6. Byte-level BPE and tokenizer metrics/tooling.

### Parked Until Current FORGE Cleanup Lands
1. Discovery mode MVP.
2. Broader training roadmap items that are not part of the current 3-mode FORGE contract.
3. Whether backend-only modes (`DPO`, `RLHF`, `SelfPlay`, `Evolutionary`) stay hidden or move to a separate advanced workflow.

## Audit Result
- The duplicate roadmap dumps that followed this section were removed because they were overlapping, partially stale, and contradicted the “active only” purpose of this file.
- If a larger long-range training roadmap is still useful, keep it in a separate dedicated file instead of mixing it into the active FORGE status document.

# suggestions.txt
# Purpose: Single source of truth for what should be changed, fixed, added, and verified.
# Scope: Consolidated everything we discussed (training, optimizer, scheduler, batching, tokenizer, eval, DX).
# Updated: 2026-03-11

Legend:
- Priority: P0 (must), P1 (should), P2 (nice), P3 (later)
- Risk: low/med/high

================================================================================
0) BASELINES (DO THIS BEFORE BIG CHANGES)
================================================================================
[ ] (P0, low) Create a reproducible baseline run
    - Record seed, model preset + full config dump, tokenizer type/vocab/special IDs,
      dataset snapshot (hash/count), hyperparams (lr/betas/wd/schedule/steps).
    - Done when: rerunning twice gives similar loss curve.

[ ] (P0, low) Add a tiny fixed regression eval set
    - 200–2000 samples, never changes.
    - Track: eval_loss, perplexity, and a few golden prompts.

================================================================================
PHASE 1 — TRAINING STABILITY (NUMERICAL + OPTIMIZER + LR)
================================================================================
[x] (P0, low) RMSNorm fp32 upcast — DONE (commit 6a23413)
[x] (P0, low) AdamW betas configurable (LM defaults) — DONE (commit 6a23413)
[x] (P0, low) Cosine schedule eta_min != 0 — Already done (eta_min = lr * 0.1)
[x] (P0, low) Weight decay consistency — Already done + LoRA fixed (commit 6a23413)

================================================================================
PHASE 2 — DATA PIPELINE + BATCHING (QUALITY + SPEED)
================================================================================
[ ] (P0, med) Sequence packing
    Files:
      - enigma_engine/core/training.py
    Change:
      - pack multiple short sequences into max_seq_len chunks separated by EOS.

[ ] (P0, med) Pad masking correctness
    - Loss ignore_index uses pad_token_id.
    - Consider attention masking to avoid attending to pads.

[ ] (P1, low) Standardize chat template / formatting
    - Define canonical prompt format with BOS/EOS rules.

[ ] (P1, low) Improved filtering + stats
    - Report kept/removed, min/max/avg lengths, duplicates removed.

================================================================================
PHASE 3 — TOKENIZER (CAPACITY + CORRECTNESS)
================================================================================
[ ] (P0, low) Unify special token IDs across tokenizers — Partially done (think IDs exposed on all tokenizers; actual ID unification deferred)

[x] (P0, low) Validation loop — DONE
    - `val_split` config field, data split, per-epoch val pass, best checkpoint by val_loss.

[ ] (P0, high) True byte-level BPE
    Files:
      - enigma_engine/core/bpe_tokenizer.py

[ ] (P1, med) Increase vocab size default (32k+ recommended)

[ ] (P1, low) Tokenizer training tooling + metrics

================================================================================
PHASE 4 — EVALUATION + CHECKPOINTING
================================================================================
[x] (P0, low) Validation loop — DONE (val_split + per-epoch val + best by val_loss)

[ ] (P1, low) Golden prompt regression suite

[ ] (P1, med) Proper resume-from-checkpoint
    - Restore optimizer, scheduler, scaler, step counters.

================================================================================
PHASE 5 — PERFORMANCE / DX
================================================================================
[ ] (P1, low) Throughput telemetry
    - tokens/sec, step time, VRAM usage.

[ ] (P1, low) One canonical training CLI entrypoint

[ ] (P2, low) Fix encoding/mojibake artifacts (e.g., â€”)

================================================================================
REFERENCES / INSPIRATION
================================================================================
- Lightning-AI/litgpt, karpathy/nanoGPT: optimizer/schedule/packing best practices
- HF transformers/trl/OpenRLHF: preference training patterns