# Suggestions

## 🟡 RUNTIME TESTS PENDING (user-driven, not code work)

These tests need a GPU + the real student model and cannot be executed by the code agent. **Do these LAST after the rest of the backlog is shipped.**

- **P5-run** — small dry pass on a *copy* of the student (50 examples, 1 epoch) through FORGE Distill personality mode. Validates the full pre-probe → backup → anchor mix → train → post-probe → drift report pipeline runs end-to-end. No new code. Just exercise the GUI.
- **P5-real** — full ~500-example personality SFT on the real student model. Rollback file (`models/checkpoints/{stem}_pre_distill_{ts}.pth`) is automatically created Pass 156z9ap. Watch the post-probe drift count — non-zero = roll back.
- Any other "run the GUI / run training / validate output looks right" exercises that show up in future passes go here.

---

## 🔵 TEACH-1 — Teach-while-running (proposed, not started)

**Status:** Logged May 6, 2026. Direction confirmed in Project Goal (`AA code maker.md`). NO CODE CHANGES authorised yet — this entry is the plan.

**What already exists (do NOT rebuild):**
- `RAGIndex` wired through `_prepare_chat()` — the AI can already look things up over `data/` and `information/`.
- `BackgroundTrainer` (`router.py`) — daemon-mode trainer with NaN/finite guards, token-length cap, replay buffer.
- Anchor-set rehearsal (`_load_anchor_examples`, `data/anchor_examples.jsonl`) — mitigates catastrophic forgetting.

**Gap (the actual work):**
1. **No persistent correction store.** When the user says "that's wrong, the answer is X," the correction lives in the chat log and disappears on session end. Nothing reaches `BackgroundTrainer`.
2. **No vision-correction surface.** When image recognition mis-identifies an object, there is no GUI affordance for the user to label the right answer. The mod at `mods/vision/` reads screen pixels but cannot accept feedback.
3. **No replay-into-preference path.** Even if (1) and (2) shipped, corrections would land as plain text, not as DPO/APO `(chosen, rejected)` pairs that actually shift weights toward the right answer over the wrong one.

### Slice plan

| Slice | What | Risk | Solves |
|---|---|---|---|
| **TEACH-1a** Correction store + chat-side widget | New `data/corrections.jsonl` (atomic append). New "this is wrong, here is right" button on each AI reply in CORE chat. Records `{prompt, wrong_response, right_response, timestamp, modality}`. | low | Persistent feedback exists. |
| **TEACH-1b** Vision-correction surface | When image is attached and AI mis-identifies, user can edit the AI's reply with the right caption → stored as TEACH-1a row with `modality="vision"` + image path. | medium | Vision-specific correction exists. |
| **TEACH-1c** Replay-into-DPO path | `BackgroundTrainer` reads `data/corrections.jsonl` on its replay tick, converts each row to a DPO pair (`chosen=right_response, rejected=wrong_response`), feeds into the same `train_dpo` path the FORGE button already uses. | high | Corrections actually move weights. |

### Acceptance call-chain (Rule §1 #20)

After TEACH-1c, the production call chain must be:
- **Capture:** User clicks "this is wrong" in CORE chat → `LogicChatMixin._record_correction(prompt, wrong, right)` → atomic append to `data/corrections.jsonl`.
- **Replay:** `BackgroundTrainer._tick` → `_load_corrections()` → `_corrections_to_dpo_pairs()` → `train_dpo(pairs)` → weight update.
- **Verify:** Re-issue the original prompt → AI gives the right answer (eventually — anchor-set bound applies, see Pass 156i6).

### Devil's advocate (Rule §1 #13)

- **One bad correction can poison weights.** User typos, mis-clicks, or sarcastic "corrections" become DPO pairs the same as real ones. Mitigation: batch corrections ≥ N before training, surface a review list, never train single-shot.
- **DPO on tiny correction batches is unstable.** A 5-row replay tick will swing weights wildly. Mitigation: defer training until `len(corrections) ≥ 50` OR mix corrections with the anchor set on every tick.
- **Vision correction needs the image at training time.** If the user moves/deletes the source image between correction and replay, the vision row is dead. Mitigation: copy the image into `data/corrections/images/` on capture (small disk cost, predictable provenance).
- **Overlap with `BackgroundTrainer` replay buffer.** That buffer already trains on recent chat — corrections risk being trained twice. Mitigation: corrections are a **separate stream** (DPO pairs), recent-chat replay stays SFT — different loss, no double-counting.

### Recommended order

1. **TEACH-1a** (correction store + button only). Smallest safest start. Validates the capture surface without touching training.
2. Pause for user to actually use the button on a few real chats — confirm UX before building the replay path.
3. **TEACH-1c** (replay path) before **1b** — text corrections give us the simpler training-side proof. Vision adds image-path provenance that is its own scope.
4. **TEACH-1b** last — vision-correction widget plus image-archival.

### Parked / open questions

- **"Tell it how to do a task" (procedure capture)** — the user wanting to teach a procedure (e.g. "to summarize a PDF: do X, then Y") is not the same as correction. Procedures are RAG-indexable instruction documents, not preference pairs. Lean: write to `information/procedures/{name}.md`, RAG already picks them up. No new code needed for this half. Confirm with user before assuming.
- **Interaction with ARCH-1c** — once daemon-side training lands, TEACH-1c moves to a daemon endpoint. Not blocking; ship TEACH-1 against today's in-process trainer.

---

## 🔵 ARCH-1 — Server-first, CLI-first, YAML-driven training (revised May 6, 2026)

**Status:** Logged May 6, 2026. **Revised same day** after a multi-turn audit that surfaced the original plan's flaws. NO CODE CHANGES authorised yet — verification slice (ARCH-V1) is the smallest first move.

**Naming:** **Enigma AI** = the daemon (model + training + inference + API + CLI). **GUI** = a thin client that talks to Enigma AI over HTTP and edits YAML configs the CLI consumes.

**Goal (one sentence):** match how real shops train — CLI + YAML config + JSONL data — instead of 10.3k lines of GUI launchers, six core trainers (not fifteen), nine experimental ones parked but kept, GGUF round-trip verified before any llama.cpp claim. GUI becomes a YAML editor + run button on top of the CLI, not a parallel implementation.

### Why the plan was rewritten (lessons from the audit, May 6, 2026)

The first version of this entry inflated counts, picked the wrong abstraction, and skipped a verification step. Six audit findings:

1. **Trainer count was inflated.** Original plan said "20 modes." Real grep shows 8 Trainer methods + 7 sibling classes = 15 distinct trainer entry points. Several "GUI modes" (dialogue, distill, evolutionary, advanced, adaptive) are compositions of existing trainers, not new trainers. Real ARCH-1.5b is a 15-pass migration max, not 20.
2. **15 is itself too many.** Modern shops ship 2-4 trainers (Hermes: SFT+DPO; DeepSeek-R1: SFT+GRPO; Llama-Instruct: SFT+DPO+PPO; Qwen-Instruct: SFT+DPO). We have 15 because we wrote one per algorithm instead of treating algorithms as flags. **Real "core" minimum: 6.** The other 9 stay in the codebase as `experimental=True` registry entries — callable from CLI for power users, no GUI surface, no API schema. See "Trainer minimum" section below.
3. **`/api/train` ships ONE training shape, not a generic seam.** [api/server.py L997-L1011](enigma_engine/api/server.py#L997-L1011) `TrainRequest` accepts only `data_file`/`epochs`/`learning_rate`/`batch_size` and the handler hardcodes `Trainer(...).train()`. **Zero coverage for DPO/vision/audio/RLHF/GRPO/etc.** Same one-mode bug in `run.py --train`. Both surfaces need expansion.
4. **GUI imports `EnigmaEngine` AND multiple Trainer classes directly.** 20+ direct instantiations across `gui_forge*`/`gui_logic*` mixins. Self-Play even spins up a SECOND `EnigmaEngine` as the trainer ([gui_forge_new_modes.py L2323](enigma_engine/gui/gui_forge_new_modes.py#L2323)).
5. **GGUF export round-trip is UNVERIFIED.** [core/gguf.py L171](enigma_engine/core/gguf.py#L171) declares `general.architecture = "llama"` and the writer ships ([GGUFExporter.export](enigma_engine/core/gguf.py#L683)). Tests confirm bytes get written. **No test confirms the bytes load back through `llama.cpp` / `llama-cpp-python`.** "Path 1, we already export to llama.cpp" was based on the writer existing, not on round-trip proof. Could be silently broken — tensor-name mapping, Q4_K block layout, metadata key set are all places it could fail. New slice **ARCH-V1** verifies this BEFORE any user-facing llama.cpp claim.
6. **Vision/audio do NOT collapse into SFT.** Earlier turn proposed collapsing them; verified at [training.py L4937 train_vision](enigma_engine/core/training.py#L4937) and [L5616 train_audio](enigma_engine/core/training.py#L5616) — both have method-specific architecture (distinct encoder argument, distinct freeze/unfreeze pattern, distinct preprocessing). Industry doesn't collapse them either: LLaVA has `LlavaTrainer`, Whisper has its own seq2seq trainer, CLIP has its own contrastive trainer. Multi-modal training is its own family. **Vision and audio stay as separate trainers.** That's why core minimum is 6, not 4.

### How real shops train (industry pattern)

| Tool | Interface | Trainer count |
|---|---|---|
| HuggingFace TRL | `trl sft / dpo / grpo / ...` CLI subcommands, args via flags or YAML | ~6 (sft, dpo, grpo, kto, orpo, ppo) |
| axolotl | `axolotl train config.yaml` — single command, YAML names trainer + data + hyperparams | ~5 trainer types switchable in YAML |
| unsloth | Python script, ~30 lines, run `python train.py` | 1-3 (SFT, DPO via flag) |
| Llama-Factory | GUI is a YAML editor that calls the CLI underneath | ~6 |

**Pattern is unanimous: CLI is the source of truth. YAML names the trainer + data + hyperparams. GUI (when it exists) edits the YAML.** Nobody has 15 GUI buttons. Our 10.3k lines of GUI launchers are doing the wrong job.

### Trainer minimum (6 core, 9 experimental)

**Core (full GUI + API + CLI surface, full test coverage):**

| # | Mode | Trainer entry | Why core |
|---|---|---|---|
| 1 | `sft` | `Trainer.train` | Foundation. Pretraining + supervised fine-tuning. |
| 2 | `dpo` | `Trainer.train_dpo` | Industry-default preference alignment. |
| 3 | `grpo` | `GRPOTrainer` | Modern reasoning alignment (DeepSeek-R1 lineage). Replaces RLHF/PPO/ReMax for new work. |
| 4 | `lora` | `LoraTrainer` | Different mechanism (low-rank adapters). Needed for low-VRAM users. |
| 5 | `vision` | `Trainer.train_vision` | Capability goal (vision in Project Goal). Genuinely different architecture (encoder + projection + freeze pattern). |
| 6 | `audio` | `Trainer.train_audio` | Capability goal (audio in Project Goal). Genuinely different architecture. |

**Experimental (registry entry only — CLI-callable for power users, NO GUI button, NO API schema in ARCH-1c). Kept because the code already works and tests already pass; deletion is a separate decision later.**

| Mode | Reason for experimental status |
|---|---|
| `simpo` | DPO variant (drops ref model). Memory win for big models, otherwise same use case as DPO. |
| `kto` | DPO variant (single-signal, no pair). Niche: only useful with thumbs-up/down data. |
| `orpo` | DPO variant (merges SFT+DPO into one loss). One-shot, but offers nothing GRPO doesn't for our use cases. |
| `rest` | Reinforced Self-Training. Niche; SelfPlayTrainer overlaps. |
| `reward_model` | Stage of RLHF; only useful as input to `rlhf`. |
| `rlhf` | PPO-style RLHF. Largely superseded by GRPO. Llama-Instruct still uses; DeepSeek/Qwen-2.5 dropped it. |
| `self_play` | Hard to make stable, narrow benefit. |
| `remax` | Variance-reduced PG. Academic interest, no production wins over GRPO. |
| `adaptive` | Meta-scheduler that calls other trainers. Not a trainer per the dispatcher contract — see open question below. |

### Architecture (revised)

**One CLI: `run.py --train --config path/to/config.yaml`** (or `enigma-ai train --config ...` once we add the console_scripts entry-point in ARCH-1a). The YAML names the mode, data, and hyperparams. Same shape every time:

```yaml
# Example: configs/sft_pretrain.yaml
mode: sft               # or dpo / grpo / lora / vision / audio
data: data/training.txt
output: models/run_2026_05_06.pth
epochs: 3
lr: 5e-5
batch_size: auto
seed: 42                # optional; required for --deterministic
deterministic: false
# mode-specific blocks under their own keys:
vision:                 # only required when mode == vision
  encoder_preset: small
  unfreeze_text_layers: 0
```

**One config schema, one validator, one entry point.** No duplicated launcher paths.

**One API endpoint: `POST /api/train`** with the YAML body (or equivalent JSON). Server-side dispatcher resolves `mode` and runs the matching trainer in a background thread, streaming metrics over SSE on `GET /api/training/metrics`. Replaces the 14 per-mode endpoints the previous plan called for.

**GUI becomes a YAML editor.** ~500 lines, not 10.3k. Load YAML → render fields with widget types matching the schema → save YAML → POST to `/api/train`. Same pattern Llama-Factory uses.

### ARCH-V1 — GGUF round-trip verification (smallest first slice, do FIRST)

**Goal:** prove the existing GGUF exporter produces files `llama-cpp-python` can load and generate from. Test slice only — no production code changes. If it fails, the gap becomes ARCH-V1b and gets fixed BEFORE any user-facing Hermes/llama.cpp claim.

**Steps:**
1. New test `tests/test_gguf_roundtrip.py` (skipped if `llama-cpp-python` not installed). Build a tiny ForgeConfig model (~1M params, 2 layers, vocab 256), random weights, run `GGUFExporter(quantization="f16").export(...)` to a tmp file.
2. Load the exported file with `llama_cpp.Llama(model_path=...)`. Raise → exporter is broken; fail with the raised message.
3. Generate one token via `llama.create_completion(prompt="a", max_tokens=1)`. Raise or empty → tensor mapping is broken; fail.
4. Repeat for `q8_0` and `q4_k` so all three quant paths are gated.
5. **Skip-with-WARNING** if `llama_cpp` isn't importable in CI; the test exists for the user's local machine where the runtime IS installed. Manual run command lands in `quick_commands.md`.

### ARCH-1 — slice plan (revised)

| Slice | What | Risk | Solves |
|---|---|---|---|
| **V1** GGUF round-trip test | ✅ **SHIPPED May 6, 2026** (commit `61369fe`). New `tests/test_gguf_roundtrip.py` with 8 tests: 2 pass, 6 strict-xfail. Refutes the llama.cpp interop claim — round-trip is broken. See ARCH-V1b/V1c below for the fixes. | low | Verifies or refutes the llama.cpp interop claim. Prerequisite for any Hermes user-facing message. |
| **V1b** Llama-style tensor-name mapping | Rewrite `WEIGHT_NAME_MAP` + `convert_tensor_name` in `enigma_engine/core/gguf.py`. Current map is HF-style (`q_proj/gate_proj/mlp/self_attn`); Enigma's state_dict is Llama-style (`attention.wq`, `feed_forward.w1`, `attention_norm`). Naive `str.replace` also substring-collides (`norm → output_norm` rewrites `attention_norm` → `attn_output_norm`). Replace with a regex-based per-segment mapping. Add entries for `wq/wk/wv/wo → attn_q/k/v/output`, `w1/w2/w3 → ffn_gate/down/up`, `q_norm/k_norm → attn_q_norm/k_norm` (QK-norm), `attention_norm → attn_norm`, `ffn_norm` (already correct), `tok_embeddings → token_embd`, `norm → output_norm`, `layers.N → blk.N`. Unmark the 3 structural xfails in `TestTensorNameMappingIsLlamaStyle`. | low | Tensor names match what llama.cpp expects for `general.architecture = llama`. |
| **V1c** GGUF metadata + tokenizer audit | After V1b, the round-trip xfails likely STILL fail because llama.cpp also requires: `llama.attention.layer_norm_rms_epsilon`, BPE merges in `tokenizer.ggml.merges`, special-token role markers, `tokenizer.ggml.model = "gpt2"` or `"llama"` matching the actual tokenizer family, and quantization block layout for `q4_k`/`q8_0`. Audit each metadata key against [llama.cpp's llama-arch.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-arch.cpp) and add what's missing. Subprocess driver is already in the test file. Unmark the 3 round-trip xfails when subprocess returns 0. | medium | End-to-end round-trip works; user-facing "exports llama.cpp models" claim becomes true. |
| **1.5a** ConfigSchema + ModeRegistry + dispatcher entry-point | New `enigma_engine/training/schema.py` (Pydantic + YAML loader). New `training/registry.py` (mode → (TrainerClass, config_builder, run_method) for all 6 core + 9 experimental). New `training/dispatch.py` (single `run(config) → Job`). Behind `experimental=True` flag for the 9 parked modes. | low | The CLI's argparse → YAML → dispatcher seam. One canonical entry point. |
| **1.5b** Wire `run.py --train --config` to the dispatcher | Existing `run.py --train data.txt` stays working (legacy text-SFT path). New `run.py --train --config path.yaml` calls the dispatcher. Add `enigma-ai` console_scripts entry-point in pyproject.toml. | low | CLI surface ships; one mode at a time can be migrated. |
| **1.5c** Migrate 6 core modes to dispatcher | One mode per pass: `sft`, `dpo`, `grpo`, `lora`, `vision`, `audio`. GUI launcher for each becomes "build YAML dict → call dispatcher" (~30 lines, was ~200). | medium | Six passes. Each ends with the GUI button still working but going through the dispatcher. |
| **1.5d** Move `core/training*.py` → `enigma_engine/training/` | Mechanical rename + import-path update. `core/` becomes inference/model/tokenizer; `training/` becomes trainer base + modes + dispatcher + registry. | low (mechanical) | Clean separation. |
| **1a** API surface for `POST /api/train` (config-body) | Replace current single-mode `/api/train` with config-body endpoint that calls the dispatcher. SSE `/api/training/metrics`. Cancellation hook. **One** endpoint, not 14. | medium | Daemon can train any of the 6 core modes via a single API. |
| **1b** EnigmaClient lib | New `enigma_engine/client.py` (HTTP + SSE wrapper). `EnigmaClient.chat / chat_stream / load_model / unload_model / train(config)`. CLI client option in `run.py --client chat`. | low | Non-GUI test surface; daemon-spawn helper. |
| **1c** GUI chat over client | `gui_logic.py` and `gui_forge.py` stop importing `EnigmaEngine`. `_load_model`, `chat`, `chat_stream`, `unload`, `list_models` go through `EnigmaClient`. Daemon spawned as subprocess on GUI launch. | medium | Removes 2 of 20+ direct core imports. Validates IPC under GUI load. |
| **1d** GUI training over client | GUI training mixins build YAML config → POST `/api/train`. Self-Play double-engine path migrates to daemon-side. **GUI becomes a YAML editor** for the 6 core modes. Experimental modes intentionally have NO GUI button. | high | Removes the remaining 18+ direct trainer imports. Training survives daemon restart-on-GUI-crash. The 10.3k → ~500 line GUI shrink lands here. |
| **1e** Hardening | Lock-scope review on `/api/profiles/{id}/activate`. CORS opt-in already handled. Auth deferred until web/phone client lands. | low | Closes the security review for the localhost daemon. |
| **1f** Sister-folder reshuffle | `enigma_engine/core/` + `api/` + `training/` → `enigma_ai/`. `gui/` → `enigma_gui/`. Pure rename. Easy because GUI no longer imports core. | low (mechanical) | Final namespace split. |

### Acceptance call-chain (Rule §1 #20)

After ARCH-1d, the production call chain must be:
- **Chat:** `run.py --gui` → daemon spawned subprocess → GUI `EnigmaClient.chat(prompt)` → `POST /api/chat` → `AppState.engine.chat()` → response.
- **Training:** GUI Forge button → build YAML dict → `EnigmaClient.train(config_dict)` → `POST /api/train` → `dispatch.run(config)` → matching trainer (e.g. `Trainer.train_dpo(...)`) → SSE metrics back to GUI.
- **CLI (no GUI):** `run.py --train --config configs/dpo.yaml` → daemon (auto-spawn or attach) → same `dispatch.run(config)` path → progress on stdout.

If any chain breaks before reaching the inner code, the slice is parked, not finished.

### Devil's advocate (Rule §1 #13)

- **Six trainers might still be too many for what we ship.** SFT+DPO+LoRA covers 80% of real use cases. GRPO is recent and good but unproven for us. Vision/audio are capability goals but no user has trained either yet. Could ship with only SFT+DPO+LoRA core and bring vision/audio/GRPO in as they get used. **Lean: ship all 6** because vision/audio are explicit Project Goals and GRPO replaces three older trainers (RLHF/PPO/ReMax). But if the migration takes longer than expected, vision/audio can drop to experimental temporarily.
- **YAML loads can carry RCE risks.** Use `yaml.safe_load`, not `yaml.load`. Never `eval`. Schema validation rejects unknown keys.
- **Experimental trainers will bitrot.** No GUI button + no API schema = nobody runs them = test failures discovered late. Mitigation: experimental tests still run in the suite, just not exercised through the GUI. If a test breaks, fix or delete the trainer.
- **GUI shrink from 10.3k to 500 lines means burning code.** The shrink is real but irreversible. Mitigation: each ARCH-1.5c migration pass deletes the old GUI launcher only AFTER the new YAML-editor path is verified end-to-end on a real training run.
- **Adaptive trainer doesn't fit the dispatcher shape.** It's a meta-scheduler, not a trainer. Three options: (a) keep as experimental and document it's a special-case wrapper, (b) move to `enigma_engine/training/scheduler.py` outside the registry, (c) delete if no one uses it. **Open question — see below.**

### Recommended order

0. ~~**ARCH-V1**~~ ✅ shipped (commit `61369fe`, May 6, 2026).
0a. **ARCH-V1b** (Llama-style tensor-name mapping). Bounded fix; flips 3 of 6 xfails.
0b. **ARCH-V1c** (GGUF metadata + tokenizer audit). Flips remaining 3 round-trip xfails.
1. **ARCH-1.5a** (schema + registry + dispatcher). Tests prove every mode resolves and tears down.
2. **ARCH-1.5b** (CLI wire-up, both legacy and YAML paths working).
3. **ARCH-1.5c** (migrate 6 core modes — 6 passes).
4. **ARCH-1.5d** (mechanical rename to `training/`).
5. **ARCH-1a** (API endpoint).
6. **ARCH-1b** (EnigmaClient lib).
7. **ARCH-1c** (GUI chat over client).
8. **ARCH-1d** (GUI training over client — the 10.3k → ~500 line GUI shrink).
9. **ARCH-1e** (hardening).
10. **ARCH-1f** (sister-folder split — final cosmetic move).

### Parked / open questions

- **AdaptiveTrainer fate** — keep as experimental, move outside the dispatcher, or delete? Deferred until ARCH-1.5a is shipped and we can grep for actual GUI callers.
- **Package layout** — A confirmed (sibling package, soft split). C deferred. B folded into ARCH-1f.
- **Continuous `BackgroundTrainer` (router.py)** — moves daemon-side in ARCH-1d, but mods/ system is GUI-coupled and migrating it is its own scope. Logged as **ARCH-2**.
- **Mods/ system over API** — post-ARCH-1, mods need a daemon-side load path. Logged as **ARCH-3**.
- **Web UI / phone client** — not on the roadmap; revisit after ARCH-1e ships and stays stable.
- **Hermes-style behaviour** — separate from architecture. After ARCH-V1 confirms llama.cpp interop, the path is data + prompt template work (collect OpenHermes-2.5 / Capybara / Glaive datasets, train SFT+DPO on them with the existing pipeline). Logged as **ARCH-4 (Hermes-style finetune corpus)**, not in this plan.

**Out of scope (do NOT do these as part of ARCH-1):**
- Deleting any of the 9 experimental trainers. Separate decision after ARCH-1 stabilises.
- Collapsing vision/audio into a "modality flag" on SFT. Audit confirmed they're genuinely different trainers — not a config-flag concern.
- Touching the mods/ system, queue UI, teacher chat panes, mod-page wiring. ARCH-1's named overhaul scope (Rule §1 #18) is **"GUI training-mode launcher → CLI/YAML/dispatcher migration"** only.

---

**Last updated:** May 5, 2026 (Pass 156z9au — **Cosine schedule floor `min_lr_ratio` is now config-driven, not hardcoded.** Promoted the literal `0.1` that was duplicated at 5 sites in `core/training.py` (Trainer.train WSD branch, Trainer.train cosine branch warm-restarts, Trainer.train cosine branch no-restarts, Trainer.train_dpo, Trainer.train_vision, Trainer.train_audio) plus 1 site in `core/lora_utils.py` (`LoraTrainer.train`) into a single `TrainingConfig.min_lr_ratio: float = 0.1` field with `[0.0, 1.0]` validation. `LoraTrainer.__init__` gains a matching `min_lr_ratio: float = 0.1` kwarg with the same validation. Fixes the suggestions.txt Phase-1 item "Cosine schedule eta_min != 0 → eta_min = lr * 0.1 (or min_lr_ratio)" — the **(or min_lr_ratio)** half was never landed because the value was inlined at every site instead of named on the config. Now a user who wants a deeper cosine floor (e.g. 0.0 = textbook to-zero, 0.05 = aggressive late-step squeeze, 0.2 = very conservative) can set it once on TrainingConfig and every scheduler in the pipeline picks it up.

**Production call chain (Rule #20):**
- CLI: `run.py --train data/X.txt --epochs 10` → `Trainer(model, tok, TrainingConfig(...))` → `Trainer.train()` → cosine scheduler reads `self.config.min_lr_ratio`
- GUI: any FORGE training mode that constructs `TrainingConfig` (DPO, vision, audio, dialogue, distill, ...) → same chain via the matching `train_*` method
- LoRA: `LoraTrainer(model, tok, ..., min_lr_ratio=0.05)` → `LoraTrainer.train()` → cosine scheduler reads `self.min_lr_ratio`

**Changes (Pass 156z9au):**
- `enigma_engine/core/training.py` — added `min_lr_ratio: float = 0.1` field to `TrainingConfig` with paragraph docstring explaining 0.0 vs 0.1 vs higher; added `[0.0, 1.0]` validation in `__post_init__`; added `"min_lr_ratio": self.min_lr_ratio` to `to_dict()`; replaced 5 `self.config.learning_rate * 0.1` literals with `self.config.learning_rate * self.config.min_lr_ratio`; replaced WSD-branch local `min_lr_ratio = 0.1` with `min_lr_ratio = self.config.min_lr_ratio`.
- `enigma_engine/core/lora_utils.py` — added `min_lr_ratio: float = 0.1` kwarg to `LoraTrainer.__init__` with `[0.0, 1.0]` validation, stored as `self.min_lr_ratio`, used in the cosine scheduler at `LoraTrainer.train`. Updated docstring.
- `tests/test_core.py` — extended `TestS554TrainingConfigValidateExpanded` with 5 new tests (below-zero rejection, above-one rejection, zero accepted as textbook cosine, default value 0.1, round-trips through `to_dict`).
- `tests/test_training.py` — added new `TestMinLrRatioConfig` class with 6 tests: default value, custom round-trip through `to_dict`, **regression gate** that asserts `"learning_rate * 0.1"` does NOT appear anywhere in `core/training.py` (catches the re-introduction of the literal pattern), structural assertions that `Trainer.train` and `Trainer.train_dpo` source contains `self.config.min_lr_ratio`, and signature gate that `LoraTrainer.__init__` exposes the new kwarg with default `0.1`.

**Validation:** `ruff check enigma_engine/ tests/` → "All checks passed!". `python -m pytest tests/ -q` → 2870 passed (2859 + 11 new), 2 skipped, ~23s.

**Six-question audit (§1 #19):**
1. Would I write it this way today? — Yes. Single config field, validated once, consumed at every scheduler site. Zero magic numbers.
2. What is it connected to? — `TrainingConfig` (storage + validation + serialization), 4 cosine schedulers in `Trainer`, 1 cosine scheduler in `LoraTrainer`, the WSD branch's late-decay floor.
3. Could more connections be made? — Yes, but parked: GUI surface for `min_lr_ratio` would let advanced users tune from CONFIG / FORGE pages. Today they hand-edit `TrainingConfig(...)`. Not in scope for this slice; logged below.
4. **Logic-eye:** Does the code deliver what the docstring claims? — Yes. Default is 0.1 (matches the docstring's "LM-friendly default"); 0.0 reproduces textbook cosine (verified by validation accepting 0.0 and the math `lr * 0.0 = 0`); per-site replacement preserves the prior `* 0.1` semantics exactly when the user sticks with the default.
5. **Claim-vs-test:** Does the test prove correctness? — Yes. Behavioural-side: validation tests construct rejected/accepted configs and call `validate()`; `to_dict` round-trip test asserts the value emerges intact. Structural-side: regression gate scans the *whole* `core/training.py` for the old literal pattern (catches a regression where someone re-inlines `* 0.1` at any site, including future new schedulers); per-method `inspect.getsource` tests gate the new pattern at the two most-trafficked methods (`Trainer.train`, `Trainer.train_dpo`). The whole-file scan is stronger than per-site `getsource` checks because it catches sites that don't yet exist.
6. **Sibling-boundary sweep:** Did I grep every site that shares this contract? — Yes. Grep `eta_min=.*learning_rate.*0\.1` in `enigma_engine/**/*.py` returned 0 matches post-fix. The cosine `eta_min` family is closed. Sibling families NOT touched (intentionally): the WSD floor at the same value is the same family and was promoted; `cosine_restart_period` warm-restarts in the same branch share the same floor and were promoted in the same `multi_replace_string_in_file` block.

**Finished:**
- `TrainingConfig.min_lr_ratio` field shipped with default + validation + `to_dict` round-trip
- 5 cosine `eta_min` sites in `core/training.py` now config-driven
- 1 cosine `eta_min` site in `core/lora_utils.py` now config-driven (with new kwarg)
- WSD decay-floor local var now config-driven
- 11 new tests cover defaults, validation edges, serialization, and a whole-file regression gate against re-inlined literals

**Parked (concrete next step):**
- GUI surface for `min_lr_ratio` on the CONFIG / FORGE training pages — **next step**: add a numeric entry on the relevant page (CONFIG general training section), bind it to the `TrainingConfig(...)` constructor inside the matching `_start_*_training` handler, default value 0.1, range `[0.0, 1.0]`. Low priority because the field has a sane LM-friendly default and only deep-dive users tune it.

---

**Last updated:** May 5, 2026 (Pass 156z9at — **Vision Stage-2 pre-train auto-checkpoint.** Closes the parked Stage-2 follow-up from 156z9as. Vision training has two stages: projection-only (text backbone frozen, no rollback rail needed) and Stage-2 with `unfreeze_text_layers > 0` (text backbone mutates, same risk profile as full SFT). Helper call is gated on the Stage-2 condition so projection-only runs do NOT produce a redundant rollback file.

**Production call chain (Rule #20):**
FORGE Vision → `_start_vision_training` → text-data load → `unfreeze_text_layers` validate (clamped 0–64) → `Trainer(...)` → **`if unfreeze_text_layers > 0: pre_vision_backup_path = self._pre_training_backup(student_path, suffix="pre_vision_stage2")`** → `train_vision(...)` → `atomic_torch_save` → `Rollback: {name}` log when Stage-2 backup landed.

**Changes:**
- [enigma_engine/gui/gui_forge_training.py](enigma_engine/gui/gui_forge_training.py) `_start_vision_training` — added the gated helper call immediately after `Trainer(...)` init; completion log surfaces "Rollback: {name}" only when the Stage-2 backup ran.
- Tests in [tests/test_personality_data.py](tests/test_personality_data.py) `TestPreTrainingBackupWireSites`:
  - NEW `test_vision_stage2_uses_helper_gated` — gates on `_pre_training_backup(` + `suffix="pre_vision_stage2"` + the literal `if unfreeze_text_layers > 0` gate AND asserts the gate index precedes the helper-call index (ordered substring check). Falsifiable: removing the gate would drop the gate-index comparison; reordering the gate after the helper call would fail the same assertion.

**Validation:**
- `ruff check enigma_engine/ tests/` — clean.
- `pytest tests/ -q` — **2859 passed, 2 skipped** (+1).

**Six-question self-audit (Rule #19):**
1. **Author's-lens** — Same one-line wire-site as the other handlers, plus a single-line `if` gate. The gate is the only thing that distinguishes vision from the other entry points; everything else (suffix, log shape, completion-block surface) follows the established pattern.
2. **Connections** — Helper now reachable from 6 wire-sites: distill, dialogue, dpo (covers apo), rl_variant (covers grpo + remax), preference_variant (covers simpo + orpo), vision-stage2. All 8 user-facing full-weight modes covered. LoRA + projection-only-vision intentionally skipped (no full-weight mutation).
3. **More connections** — Pretrain (`_start_training`) and general SFT remain open. Their in-run `checkpoint_dir` saves give step-based rollback granularity already; pre-train backup is duplicative there. Decision: leave open for now, document as a design choice in the parked entry rather than wire by reflex.
4. **Logic-eye** — Gate semantics match the docstring: helper only runs when text weights will mutate. Completion log "Rollback" line is correctly conditional on `pre_vision_backup_path` truthiness, so projection-only runs never see a misleading "Rollback" hint pointing at a backup that does not exist.
5. **Claim-vs-test** — Structural test gates the literal helper call expression, the literal suffix string, AND the literal gate string AND the ordered-substring relationship between them. A regression that drops the gate (always backup) OR drops the helper (never backup) OR swaps their order both fail. Adversarial: deleting the helper line fails the helper-call assertion; deleting the gate fails the gate assertion; swapping their order fails the ordered-index check.
6. **Sibling-boundary sweep** — All FORGE entry points that mutate full text weights now have the rail. Pretrain + general SFT are the only outliers; they have step-based rollback via `checkpoint_dir` so the missing rail is by design, not by oversight. Documented in the Parked block below.

**Finished / Killed / Parked (Rule #20):**
- **Finished**: vision Stage-2 backup gated on `unfreeze_text_layers > 0`. Closes the second of two parked entries from 156z9as.
- **Parked (concrete decision required, not work)**: pretrain (`_start_training` in gui_forge_training) and general SFT — the in-run `checkpoint_dir` saves are step-based and produce rolling checkpoints during the run, which is strictly better than a single pre-training snapshot. Adding a pre-training backup would be redundant for these paths. Decision: leave as-is unless the user reports a specific scenario where a single named pre-training snapshot would help (e.g. "rollback to the moment BEFORE pretrain started" rather than "rollback to step N of pretrain"). If that scenario emerges, the same one-line helper call can be added.

---

**Last updated:** May 5, 2026 (Pass 156z9as — **Sibling-extension of pre-training auto-checkpoint to all RL/preference-alignment entry points.** Continues 156z9ar by wiring the `_pre_training_backup` helper into the remaining full-weight FORGE entry points: DPO (and APO via the `loss_type="apo_zero"` wrapper that re-uses `_start_dpo_training`), GRPO + ReMax (via the shared `_start_rl_variant_training` handler), and SimPO + ORPO (via the shared `_start_preference_variant_training` handler). Five algorithms covered with three handler edits — exactly the kind of leverage the helper extraction was designed to enable.

**Production call chains (Rule #20):**
1. FORGE DPO → `_start_dpo_training(loss_type="dpo")` → ... → `Trainer(...)` → **`_pre_training_backup(student_path, suffix=f"pre_{loss_type}")`** → `pre_dpo_*.pth` → `train_dpo(loss_type="dpo")` → save → "Rollback".
2. FORGE APO → `_start_apo_training()` → `_start_dpo_training(loss_type="apo_zero")` → same wire-site as above → `pre_apo_zero_*.pth` → `train_dpo(loss_type="apo_zero")` → save → "Rollback".
3. FORGE GRPO → `_start_grpo_training()` → `_start_rl_variant_training("GRPO")` → reward+policy phases → **`_pre_training_backup(student_path, suffix=f"pre_{algo.lower()}")`** → `pre_grpo_*.pth` → save → "Rollback".
4. FORGE ReMax → `_start_remax_training()` → `_start_rl_variant_training("ReMax")` → same wire-site → `pre_remax_*.pth` → save → "Rollback".
5. FORGE SimPO → `_start_simpo_training()` → `_start_preference_variant_training("SimPO")` → **`_pre_training_backup(student_path, suffix=f"pre_{algo.lower()}")`** → `pre_simpo_*.pth` → `train_simpo(...)` → save → "Rollback".
6. FORGE ORPO → `_start_orpo_training()` → `_start_preference_variant_training("ORPO")` → same wire-site → `pre_orpo_*.pth` → `train_orpo(...)` → save → "Rollback".

**Changes (Pass 156z9as):**
- [enigma_engine/gui/gui_forge_training.py](enigma_engine/gui/gui_forge_training.py) `_start_dpo_training` — added helper call after `Trainer(...)` init using `f"pre_{loss_type}"` suffix so DPO and APO produce distinguishable rollback names; "Rollback: {name}" line appended to completion log.
- [enigma_engine/gui/gui_forge_new_modes.py](enigma_engine/gui/gui_forge_new_modes.py) `_start_rl_variant_training` (shared handler for GRPO + ReMax) — added helper call after preference-data load using `f"pre_{algo.lower()}"`; "Rollback" log in completion block.
- `_start_preference_variant_training` (shared handler for SimPO + ORPO) — same shape, helper called immediately after `Trainer(...)` init.
- Tests in [tests/test_personality_data.py](tests/test_personality_data.py) `TestPreTrainingBackupWireSites`:
  - NEW `test_dpo_uses_helper` — gates on `_pre_training_backup(` + `f"pre_{loss_type}"` + `pre_dpo_backup_path` + `Rollback`. Implicitly covers APO since APO routes into the same method.
  - NEW `test_rl_variant_uses_helper` — gates on `_pre_training_backup(` + `f"pre_{algo.lower()}"` + `pre_rl_backup_path` + `Rollback` in the shared GRPO/ReMax handler.
  - NEW `test_simpo_orpo_uses_helper` — same shape against the shared SimPO/ORPO handler with `pre_pref_backup_path`.

**Validation:**
- `ruff check enigma_engine/ tests/` — clean.
- `pytest tests/ -q` — **2858 passed, 2 skipped** (+3 vs 156z9ar baseline).

**Six-question self-audit (Rule #19):**
1. **Author's-lens** — Three handler-level edits cover five user-facing modes. The leverage comes from the existing wrapper pattern (DPO/APO share `_start_dpo_training`; GRPO/ReMax share `_start_rl_variant_training`; SimPO/ORPO share `_start_preference_variant_training`) that was already in place from prior alignment-mode work. Nothing new built; we're just placing the helper call at the one point in each handler where the model is in memory but training has not yet stepped weights.
2. **Connections** — Helper lives on `ForgeNewModesMixin` (Pass 156z9ar) and is reachable from every FORGE entry point via mixin composition. Each new wire-site uses the canonical pattern: helper-call result captured in a `pre_*_backup_path` local, surfaced as `Rollback: {Path(...).name}` in the completion log when non-None. No new global state, no new module-level imports, no signature drift.
3. **More connections** — Remaining full-weight entry points: `_start_vision_training` (Stage-2 only, `unfreeze_text_layers > 0`); `_start_kto_training` if/when it's wired (not present in current source). LoRA explicitly excluded (base weights untouched). Dialogue + distill closed in 156z9ar. Pretrain + general SFT (`_start_training` in gui_forge_training) — those use the explicit FORGE config with `checkpoint_dir` and produce step-based checkpoints during the run, so the rollback rail is less critical, but still worth a future pass for consistency. Logged for future review, not a regression.
4. **Logic-eye** — Each call site captures the helper return into a local AND surfaces it in the completion log AND gates the log on truthiness. No false-promise paths. Suffix string differs per call site so DPO/APO rollback files cannot be confused with each other; same for GRPO/ReMax and SimPO/ORPO.
5. **Claim-vs-test** — Three new structural tests gate the LITERAL helper call expression paired with the LITERAL suffix expression at each handler. A regression that drops the helper call but keeps the local variable, OR keeps the call but forgets to forward the algo-derived suffix, fails the test. The "no `shutil.copy2` left in entry point" assertion from 156z9ar still gates `_start_distill_training` against re-inlining; the new handlers never had inline backup bodies so there is no equivalent regression to catch.
6. **Sibling-boundary sweep** — Walked the FORGE entry-point family. Closed in 156z9ar: distill, dialogue. Closed in 156z9as: dpo, apo (via dpo), grpo, remax, simpo, orpo. Remaining open: vision Stage-2, pretrain, general SFT. That's 8 of ~10 full-weight entry points covered; the remaining two are parked with concrete next steps below.

**Finished / Killed / Parked (Rule #20):**
- **Finished**: `_pre_training_backup` helper now called from `_start_distill_training` (156z9ar), `_start_dialogue_training` (156z9ar), `_start_dpo_training` (covers DPO+APO), `_start_rl_variant_training` (covers GRPO+ReMax), `_start_preference_variant_training` (covers SimPO+ORPO).
- **Parked (concrete next step)**: (a) `_start_vision_training` Stage-2 — gate the helper call on `unfreeze_text_layers > 0` since projection-only training never mutates the text backbone; (b) `_start_training` (pretrain / general SFT) — the in-run `checkpoint_dir` saves are step-based and already give rollback granularity, so the pre-training backup is duplicative. Document this as the design decision OR add the rail anyway for consistency.

---

**Last updated:** May 5, 2026 (Pass 156z9ar — **Sibling-extension of pre-training auto-checkpoint to dialogue training + DRY refactor.** Carries forward the parked follow-up from Pass 156z9ap. The 30-line inline backup body in `_start_distill_training` is extracted into a reusable `_pre_training_backup(model_path, *, suffix)` helper on `ForgeNewModesMixin` (mixin composition makes it visible from every Forge*Mixin via the host class), and dialogue training (the next-highest-risk full-SFT entry point after distill) gets the same rollback rail.

**Production call chains (Rule #20):**
1. FORGE Distill → `_start_distill_training` → **`_pre_training_backup(student_path, suffix="pre_distill")`** → returns `models/checkpoints/{stem}_pre_distill_{ts}.pth` path → anchor mix gate → identity probe (P5-pre-3) → `Trainer(...)` → train → save → `Rollback: {name}` log.
2. FORGE Dialogue → `_start_dialogue_training` → trainer/student route check → epochs/lr validate → forge_params → `TrainingConfig(...)` → **`_pre_training_backup(student_path, suffix="pre_dialogue")`** → returns `models/checkpoints/{stem}_pre_dialogue_{ts}.pth` path → `Trainer(...)` → train → save → `Rollback: {name}` log appended to DIALOGUE TRAINING COMPLETE block.

**Changes (Pass 156z9ar):**
- [enigma_engine/gui/gui_forge_new_modes.py](enigma_engine/gui/gui_forge_new_modes.py) `ForgeNewModesMixin._pre_training_backup` — NEW helper. Lazy-imports `MODELS_DIR` from `enigma_engine.gui.scanners` (matches the existing tokenizer-load pattern in the same mixin so the backup helper has zero new module-level imports). Branches: source missing → return None + INFO "skipped"; copy raises → return None + loud `[!]` "FAILED" log (non-fatal); success → return path string + INFO "backup: {name}". `suffix` kwarg parametrizes the filename so each call site produces self-explanatory rollback files (`{stem}_pre_distill_{ts}{ext}`, `{stem}_pre_dialogue_{ts}{ext}`, ...).
- `_start_distill_training` — 30-line inline backup body deleted; replaced with single `pre_distill_backup_path = self._pre_training_backup(student_path, suffix="pre_distill")` call. Behaviour unchanged; DRY.
- [enigma_engine/gui/gui_forge_advanced.py](enigma_engine/gui/gui_forge_advanced.py) `_start_dialogue_training` — added `pre_dialogue_backup_path = self._pre_training_backup(student_path, suffix="pre_dialogue")` immediately before the `Trainer(student_mdl, tokenizer2, train_config)` constructor. Completion block surfaces `Rollback: {name}` when the backup landed.
- Tests (all in [tests/test_personality_data.py](tests/test_personality_data.py)):
  - NEW `TestPreTrainingBackupHelper` (4 behavioural tests with `tmp_path` + `monkeypatch.setattr(scanners, "MODELS_DIR", tmp_path)`) — creates timestamped copy with bytes round-trip + source untouched; returns None on missing source with skip log; swallows `shutil.copy2` raise + emits loud `[!]` log; suffix parametrizes filename so `pre_dialogue` and `pre_distill` calls produce distinguishable names.
  - NEW `TestPreTrainingBackupWireSites` (2 structural tests) — distill entry point uses `_pre_training_backup(` + `suffix="pre_distill"` AND no longer contains a raw `shutil.copy2` (the regression-against-re-inlining gate); dialogue entry point uses `_pre_training_backup(` + `suffix="pre_dialogue"` + binds to `pre_dialogue_backup_path` + emits `Rollback` in completion log.
  - UPDATED `TestP5Pre2WireSite::test_pre_distill_backup_runs_before_trainer_init` to gate on `_pre_training_backup(` instead of the removed `_pre_distill_` substring; UPDATED `test_pre_distill_backup_uses_timestamp` to gate `strftime` against the helper source (single source of truth for timestamping) instead of the entry point.

**Validation:**
- `ruff check enigma_engine/ tests/` — clean.
- `pytest tests/test_personality_data.py -q` — 65 passed (was 59).
- `pytest tests/ -q` — **2855 passed, 2 skipped** (+6 vs 156z9aq baseline; all 6 are the new explicit assertions).

**Six-question self-audit (Rule #19):**
1. **Author's-lens** — Helper extraction is the textbook DRY refactor when a 30-line body is about to be repeated at 6+ sites. The `suffix` kwarg is the only parameter that varies between call sites; everything else (source-existence check, MODELS_DIR resolution, timestamp format, error handling) is identical. No abstraction over what's not actually shared. Defensible.
2. **Connections** — Helper lives on `ForgeNewModesMixin` next to `_run_identity_probe` (P5-pre-3 sibling). Mixin composition (`ForgeMixin(ForgeTrainingMixin, ForgeAdvancedMixin, ForgeAdaptiveMixin, ForgeNewModesMixin, ...)`) makes the helper visible to every entry point in every sibling mixin via `self._pre_training_backup(...)`. Lazy `MODELS_DIR` import matches the canonical pattern (tokenizer-load at L559, vocab-init at L1480 — both in the same file). Tests patch `scanners.MODELS_DIR` (the lazy-imported source of truth) so future refactors that change which module re-exports it will fail the patch loudly.
3. **More connections** — Five more sibling entry points (`_start_grpo_training`, `_start_remax_training`, `_start_simpo_training`, `_start_orpo_training`, `_start_dpo_training` — note `_start_apo_training` wraps DPO so it inherits the rail naturally) all overwrite weights in place and would benefit from the same call. The helper is ready; each site is a one-line addition. Parked as the next sibling-extension slice rather than bundled here to keep the slice tight. `_start_lora_training` does NOT need the rail (LoRA writes a separate adapter file, base weights untouched). `_start_vision_training` projection-only mode is similar to LoRA in that the text backbone is frozen — but Stage-2 (`unfreeze_text_layers > 0`) DOES mutate text weights and SHOULD have the rail; logged as part of the parked sibling-extension scope.
4. **Logic-eye** — Helper docstring names every contract (path, return values for each branch, suffix purpose, MODELS_DIR location, non-fatal-on-failure semantics). The actual code matches: missing-source returns None + INFO log; copy-failure returns None + loud `[!]` log; success returns path string + INFO log. No aspirational language, no over-promised claims. Distill refactor preserves the pre-existing `pre_distill_backup_path` variable name so all downstream completion-log code keeps working without edits.
5. **Claim-vs-test** — Behavioural tests cover all three branches of the helper with tmp_path isolation and the `monkeypatch.setattr` falsification pattern (patching `shutil.copy2` to raise proves the swallow-and-log path works against the actual exception rather than just the docstring claim). Structural tests on the two wire-sites use literal-substring gates that are paired with the suffix string so a regression that drops the helper call OR forgets to forward the suffix fails immediately. The "no `shutil.copy2` left behind in distill entry point" assertion is the regression-against-re-inlining gate that catches a future refactor that helpfully resurrects the inline body.
6. **Sibling-boundary sweep** — Walked the FORGE entry-point family. 9 `_start_*_training` methods total: 7 mutate full weights (distill, dialogue, dpo, apo→dpo, simpo, orpo, grpo, remax — that's actually 8 because apo wraps dpo), 1 mutates LoRA-only (lora), 1 conditionally mutates (vision: projection-only is safe, Stage-2 is not). Pass 156z9ap closed distill; this pass closes dialogue. Six more full-weight sites + vision Stage-2 are parked under "Sibling-extension of pre-training backup" with the helper ready to call.

**Finished / Killed / Parked (Rule #20):**
- **Finished**: helper helper itself (production-reachable from `_start_distill_training` and `_start_dialogue_training`); distill refactor (zero behaviour change, same code path); dialogue wiring (new rollback rail surfaces in completion log).
- **Parked (concrete next step)**: extend the helper call to the remaining 5 RL-alignment entry points (`_start_grpo_training`, `_start_remax_training`, `_start_simpo_training`, `_start_orpo_training`, `_start_dpo_training`) and to `_start_vision_training` Stage-2 (gated on `unfreeze_text_layers > 0`). Each is a one-line addition; tests are the same shape as the dialogue wire-site test in this pass. Held back to keep slice scope tight, not because the work is complex.

---

**Last updated:** May 5, 2026 (Pass 156z9aq — **P5-pre-3: identity-guard probe (pre+post) for personality distillation.** Third slice of the Personality-5 careful-build chain. Final safety rail before P5-run / P5-real: a deterministic in-memory probe that asks the student "Who are you?" / "What model are you?" / "Are you Qwen?" before AND after distillation, then reports drift.

**Production call chain (Rule #20):** FORGE GUI Distill mode → user selects `personality` → `_start_distill_training` → teacher gen + filter (P5-pre-1) → pre-distill backup (P5-pre-2) → anchor mix gate (P5-pre-2) → **(NEW) pre-probe: encode `User: {prompt}\nAssistant:` for each `IDENTITY_PROBE_PROMPTS` entry, call `student.generate(...)` under `torch.no_grad()` + `student.eval()`, decode, store `{prompt: response}` in `pre_probe_responses`** → `Trainer(student, tokenizer, train_config)` → `trainer.train()` (in-memory weights updated) → **(NEW) post-probe: same loop on the now-trained student** → **(NEW) `summarize_identity_probe(pre, post)` returns `{pre_safe, post_safe, drifted, recovered, total}`** → log "Identity safety: K/N pre → J/N post" + per-prompt drift list + rollback hint pointing at the P5-pre-2 backup → `atomic_torch_save`.

**Changes (Pass 156z9aq):**
- [enigma_engine/core/personality_data.py](enigma_engine/core/personality_data.py) — added pure data + helper:
  - `IDENTITY_PROBE_PROMPTS: list[str]` — 5 short, direct identity questions (`"Who are you?"`, `"Who made you?"`, `"What model are you?"`, `"Are you Qwen?"`, `"What is your name?"`). Pure module-level constant; no torch import.
  - `summarize_identity_probe(pre, post) -> dict` — pure function comparing two `{prompt: response}` dicts via existing `passes_identity_filter`. Returns `pre_safe`, `post_safe`, `drifted` (safe→leak regression), `recovered` (leak→safe gain), `total`. Operates on the intersection of keys so out-of-band prompt sets cannot crash the summary.
- [enigma_engine/gui/gui_forge_new_modes.py](enigma_engine/gui/gui_forge_new_modes.py):
  - **NEW `_run_identity_probe(model, tokenizer, device, prompts, max_new_tokens=64)`** mixin method — encodes each prompt with `User: ...\nAssistant:`, runs `model.generate(..., temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.1)` under `torch.no_grad()`, decodes the new tokens only (slices off the prompt), strips, returns `{prompt: response}`. Saves the prior `model.training` flag, switches to `eval()`, restores in `finally`. Per-prompt try/except so a single failed probe yields `""` and doesn't kill the run — observability is non-fatal.
  - **`_start_distill_training` pre-probe** — gated on `"personality" in categories`, runs AFTER anchor-mix resolution and BEFORE `TrainingConfig(...)` construction. Stores `pre_probe_responses`. Surfaces baseline leak count so the user sees the starting point. Probe failure logs `[!]` and disables post-probe comparison for the run.
  - **`_start_distill_training` post-probe** — runs AFTER `state = trainer.train(...)` returns and AFTER the abort-guard (so we don't probe a NaN-corrupted model), BEFORE `atomic_torch_save`. Calls `summarize_identity_probe`, logs `"Identity safety: K/N pre → J/N post"`, lists every drifted prompt, points the user at the P5-pre-2 rollback file when drift is non-zero (or warns when no rollback exists).
- Tests (all in [tests/test_personality_data.py](tests/test_personality_data.py)):
  - NEW `TestIdentityProbeData` (2 tests) — probe list non-empty, all strings, no duplicates.
  - NEW `TestSummarizeIdentityProbe` (6 behavioural tests) — no-drift / safe→leak drift / leak→safe recovery / intersection-only counting / empty-input / sorted output.
  - NEW `TestProbeWireSite` (4 structural tests) — pre-probe ordered before `trainer.train`; post-probe uses `summarize_identity_probe` + reads `summary["drifted"]`; gate `"personality" in categories` appears ≥2× (anchor mix + probe); helper exists with `model.eval()` + `model.train()` restoration + `no_grad()`.

**Validation:**
- `ruff check enigma_engine/ tests/` — clean.
- `pytest tests/test_personality_data.py -q` — 59 passed (was 47).
- `pytest tests/ -q` — **2849 passed, 2 skipped** (+12 vs 156z9ap baseline; all 12 are the new explicit assertions).

**Six-question self-audit (Rule #19):**
1. **Author's-lens** — Probe is the cheapest possible drift detector: no extra training, no second model load, no benchmark dataset, no neural reward. Just five strings, the existing `Enigma.generate` we already use, and the existing `passes_identity_filter` we use during data filtering. Reuse over rebuild. Defensible.
2. **Connections** — `IDENTITY_PROBE_PROMPTS` and `summarize_identity_probe` live in `personality_data.py` so they're testable without torch (pure-data module discipline preserved). The probe loop calls `student.generate()` directly (no `EnigmaEngine` re-load — would double VRAM). Filter reuses `passes_identity_filter` so probe-failure semantics match data-filter semantics by construction. Rollback logs point at the P5-pre-2 backup, closing the safety loop end-to-end.
3. **More connections** — Sibling FORGE training entry points (`_start_dialogue_training`, `_start_simple_sft`, `_start_lora_training`, `_start_dpo_training`) could ALL benefit from the probe + backup pattern when their data category is identity-relevant. Parked under "P5-pre-2/3 sibling-extension follow-up." Not done in this pass to keep slice scope tight.
4. **Logic-eye** — The probe claims "drift detection." Code delivers: encodes prompt → runs same generation path the user will hit at chat time → checks identity-leak using the same filter we trust at data-filter time → flags any prompt that flipped safe→leak. No over-promised behaviour. The post-probe ALSO tests the regression direction (leak→safe `recovered`) which is rare but possible (fresh student before personality SFT may already leak Qwen-trained vocab; SFT may genuinely fix it).
5. **Claim-vs-test** — Behavioural tests on `summarize_identity_probe` cover all four branches: safe→safe (no entry), safe→leak (`drifted`), leak→safe (`recovered`), leak→leak (no entry). Empty-input edge case included. Sorted-output gated explicitly (regression to insertion-order would be invisible without it). Wire-site tests are structural-only (a behavioural test would need a live student); each one falsified mentally by deleting the line it gates and confirming the assertion would fail.
6. **Sibling-boundary sweep** — Three other distill modes touch student weights: dialogue-only, simple-SFT, LoRA. None currently run a probe. Same-family contract is "any training that updates student weights based on teacher signal could drift identity"; closing the family fully means extending the probe to those entry points. Logged as parked sibling-extension. Within this slice the gate is `"personality" in categories` because that's the ONLY category in `_start_distill_training` whose data carries the drift pressure — other categories train on math/code/reasoning where identity leak is a teacher-data filter problem (closed by P5-pre-1), not a probe problem.

**Finished / Killed / Parked (Rule #20):**
- **Finished**: Identity-guard probe is reachable from the production entry-point `_start_distill_training` via the `personality` category checkbox; pre+post probes both run; `summarize_identity_probe` is fully tested; logs surface drift count + rollback hint. End-to-end chain reaches into new code and back out to the user.
- **Parked**:
  - Pre/post benchmark integration with `run_gsm8k_benchmark` — heavy to wire mid-flow; logging the manual command pre and post would be the right minimal step. Not done.
  - Sibling-extension of probe + backup to dialogue / simple-SFT / LoRA / DPO entry points.
  - "Restore Backup" GUI button on MODELS page (rolls back `models/checkpoints/{stem}_pre_distill_{ts}.pth` → `models/{stem}.pth`).

**Next slices in the P5 chain:**
- **P5-run** — small dry pass on a *copy* of the student (50 examples, 1 epoch) to validate the end-to-end pipeline runs without errors. No new code; runtime exercise only.
- **P5-real** — full ~500-example personality SFT on the real student model with rollback ready.

---

**Last updated:** May 5, 2026 (Pass 156z9ap — **P5-pre-2: anchor mix + pre-SFT auto-checkpoint for personality distillation.** Second slice of the Personality-5 careful-build chain. Two safety rails before the eventual personality SFT trigger:

**Production call chain (Rule #20):** FORGE GUI Distill mode → user selects `personality` (any combo) → `_start_distill_training` → teacher-side data gen + per-response filter (P5-pre-1) → **(NEW) `shutil.copy2(student_path, models/checkpoints/{stem}_pre_distill_{ts}.pth)`** → **(NEW) `_resolve_anchor_path` resolves `data/anchor_examples.jsonl` → if `personality` in categories AND anchor exists, set `general_mix_ratio=0.3` + `general_data=str(anchor_path)`** → `Trainer(student, tokenizer, train_config)` → `Trainer.train()` mixes anchor sequences via existing `general_data` + `general_mix_ratio` infra → SFT → save → "Rollback: {backup_name}" surfaced in completion log.

**Changes (Pass 156z9ap):**
- [enigma_engine/core/training.py](enigma_engine/core/training.py) `_parse_training_data` — added `response` key fallback alongside `completion` / `answer` in BOTH the dict-list path AND the JSONL path. Anchor JSONL (`{"prompt": ..., "response": ..., "score": ...}`) now parses correctly through Trainer's `general_data` mix path. Without this, the JSONL parser silently yielded zero sequences, fell through to last-resort line-split, and landed raw `{...}` strings as training tokens — pure noise. Aligns with `BackgroundTrainer.add_example` ([router.py L383](enigma_engine/router.py#L383)) which already produces `response` keys.
- [enigma_engine/gui/gui_forge_new_modes.py](enigma_engine/gui/gui_forge_new_modes.py) `_start_distill_training`:
  - **Pre-distill auto-checkpoint** — before the `Trainer(student, ...)` constructor: `shutil.copy2(student_path, models/checkpoints/{stem}_pre_distill_{YYYYMMDD_HHMMSS}.pth)`. Skipped with INFO log when `student_path` does not yet exist on disk; loud `[!]` log on copy failure (non-fatal — run proceeds without the rail). Backup path surfaced in DISTILLATION COMPLETE block as `"Rollback: {name}"`.
  - **Anchor mix gate** — only when `"personality" in categories`: read `gui_settings.json` key `anchor_data_path`, route through `scanners._resolve_anchor_path`, set `general_mix_ratio=0.3` + `general_data=str(anchor)` if the file resolves. Other distill categories keep `general_mix_ratio=0.0` (status quo). When personality is selected but no anchor file resolves, log a loud notice that catastrophic-forgetting risk is HIGHER for this run.
  - `general_mix_ratio=0.0  # Only distilled data` literal removed; `TrainingConfig` is now constructed with the gated `_mix_ratio` and `_mix_data` locals.
- Tests:
  - NEW `TestParseTrainingDataJSONL.test_jsonl_response_key_alias` — behavioural: anchor-shape JSONL parses to `User: ... \nAssistant: ...` sequences.
  - NEW `TestParseTrainingDataDictList.test_dict_list_response_key_alias` — behavioural: dict-list path also accepts `response` key.
  - NEW `TestP5Pre2WireSite` (4 structural tests) — pre-distill backup runs BEFORE Trainer init (ordered substring); backup uses `strftime("%Y%m%d_%H%M%S")` and `pre_distill_backup_path` local; anchor mix gated on `"personality" in categories` AND uses `_resolve_anchor_path` AND forwards `general_mix_ratio=_mix_ratio` + `general_data=_mix_data`; default ratio is `0.3` (regression to `0` would re-introduce forgetting silently).

**Validation:**
- `ruff check enigma_engine/ tests/` — clean.
- `pytest tests/test_personality_data.py tests/test_training.py -q` — 461 passed.
- `pytest tests/ -q` — **2837 passed, 2 skipped** (+13 vs 156z9ao baseline; +6 explicit new + 7 pre-existing skip→pass shifts that are env-dependent and unrelated).

**Six-question self-audit (Rule #19):**
1. **Author's-lens** — Two small gates added at one entry point. The backup is unconditional (any distill category benefits from rollback). The anchor mix is gated specifically to `personality` because identity-drift pressure is unique to that category — other categories (reasoning/knowledge/code) don't pull the model toward teacher self-naming. Defensible.
2. **Connections** — Reads `gui_settings.json` via existing `_read_gui_str_setting` mixin method; calls existing `scanners._resolve_anchor_path` (the same helper `desktop.py` uses to wire `BackgroundTrainer`). Writes to `models/checkpoints/` using `MODELS_DIR` already imported in scope. No new dependencies, no new modules.
3. **Could more connections be made?** — The same anchor file is now shared by THREE consumers: `BackgroundTrainer._retrain_on_replay` (continuous), `BackgroundTrainer._retrain_on_replay` idle scheduler (Pass 156w), and now distill-mode personality SFT. All three resolve through the same `_resolve_anchor_path` helper, so a future user override in `gui_settings.json` propagates everywhere consistently. Good.
4. **Logic-eye on doc claim** — Stamp claim: "anchor mix when personality is selected" + "pre-SFT auto-checkpoint with timestamped name → guaranteed rollback". Code delivers exactly: `if "personality" in categories: ... _mix_ratio = 0.3` (anchor mix gate); `_shutil.copy2(src, ckpt_dir / f"{stem}_pre_distill_{ts}{suffix}")` (timestamped backup before training). Backup is "guaranteed" only when student file exists at start — handled with INFO log when it doesn't (truthful: a fresh-construct student has nothing to back up).
5. **Claim-vs-test** — Behavioural tests for the parser fix prove the anchor file END-TO-END: `Trainer._parse_training_data` on anchor-shape JSONL now yields proper `User: ... \nAssistant: ...` sequences (not raw `{...}` strings). Wire-site structural tests gate the four contract claims (backup ordering, timestamp helper, gating on category, forwarding ratio + data). Falsifiability: deleting the `_mix_ratio = 0.3` line OR moving it to `_mix_ratio = 0.0` would fail `test_anchor_mix_default_ratio_is_30_percent`; moving the `shutil.copy2` block AFTER `trainer = Trainer(...)` would fail `test_pre_distill_backup_runs_before_trainer_init`; dropping the `"personality" in categories` gate would fail `test_anchor_mix_gated_on_personality_category`.
6. **Sibling-boundary** — `_start_dialogue_training`, `_start_simple_sft`, `_start_lora_training` and other FORGE training entry-points DO NOT make a pre-training backup. Distill is the riskiest entry-point (overwrites student with weights derived from a different model's distribution) so backup landed there first. Sibling extension is parked as a follow-up: same `shutil.copy2` block can land in any FORGE entry-point that overwrites the source model.

**P5 careful-build remaining slices (in order):**
- ~~**P5-pre-1**~~ — Done (Pass 156z9am, audit Pass 156z9an, fixes Pass 156z9ao).
- ~~**P5-pre-2**~~ — Done (this pass).
- **P5-pre-3** — add 10% eval split from generated examples; wire pre/post `python run.py --benchmark` runs; identity-guard probe ("Who are you?" / "Who made you?" before+after, assert no drift to teacher identity). NO training run yet.
- **P5-run** — small dry pass (50 examples, 1 epoch) on a *copy* of the model to validate the full pipeline.
- **P5-real** — full ~500-example run on the real model with rollback ready.

**Parked / follow-ups:**
- **Sibling backup extension** — extend `shutil.copy2` pre-training rail to `_start_dialogue_training` / `_start_simple_sft` / `_start_lora_training` / `_start_dpo_training` if the user requests broader rollback coverage. Distill is the highest-risk path so it landed first.
- **P5-pre-2-followup** — surface the rollback path in a "Restore Backup" GUI button on the MODELS page so the user doesn't need to navigate the file system to revert. Small UX win, not blocking.

---

**Previous: Pass 156z9ao — **P5-pre-1 audit fixes (F-A / F-B / F-C) + structural-vs-output-shape lesson.** Three findings logged in Pass 156z9an's "real audit" closed in this pass: **F-A** empty-response bucketing — the personality inline filter in `_start_distill_training` short-circuited on `bool(clean_response)` so empty teacher output was logged as "too short" without incrementing any reject counter, drifting the GUI counts away from `filter_personality_examples` aggregate behaviour; fixed by routing `not clean_response` through the `quality` bucket. **F-B** `conversation` category had the same `User: …\nAssistant:` double-wrap bug as the 5 personality prompts fixed in 156z9an — pre-existing, not introduced by P5, but the formatter is shared; rewrote 5 conversation prompts to direct imperatives. **F-C** added behavioural test `test_distill_formatter_well_formed_for_every_prompt` that runs every prompt through `f"User: {p}\nAssistant: {fake}"` and asserts exactly one `user:` and one `assistant:` marker — catches the double-wrap structurally even if a future prompt sneaks past the start/end string checks. **AA code maker.md §4 Testing** got the lesson: structural import-presence tests do NOT validate output shape of formatted artifacts; pair with a shape-invariant behavioural test. **Validation:** `2824 passed, 9 skipped` (+1 vs 156z9an for the new shape test). Lint clean. Commit `ee130d8`.

**Last updated:** May 4, 2026 (Pass 156z9am — **P5-pre-1: personality distillation prompt pool + identity/quality/dedup filters.** First slice of the Personality-5 careful-build chain. User said "we have to be careful with it" — this slice is pure data plumbing, ZERO training, fully testable in isolation. Sets the foundation for P5-pre-2 (anchor mix + auto-checkpoint), P5-pre-3 (eval split + benchmark + identity-guard probe), then dry-run on copy, then real run.

**Production call chain (Rule #20):** FORGE GUI Distill mode → user selects `personality` category → `_start_distill_training` → `category_prompts["personality"] = list(_PERSONALITY_PROMPTS)` (50 prompts × 10 themes) → per-prompt teacher generation → `passes_identity_filter` + `passes_quality_filter` + `is_near_duplicate` (personality category only) → accepted responses appended to `all_examples` → SFT.

**Changes (Pass 156z9am):**
- NEW [enigma_engine/core/personality_data.py](enigma_engine/core/personality_data.py) — pure data + pure functions, no torch / I/O / GUI deps:
  - `PERSONALITY_PROMPTS` — 50 prompts across 10 themes (self-introduction, reaction to compliment/criticism, opinions, empathy, anecdotes, curiosity, humor, vulnerability, values, casual). Up from 5 hardcoded prompts.
  - `passes_identity_filter(text)` — case-insensitive substring reject for teacher-model names (qwen, llama, mistral, deepseek, gemma, phi-3, claude, chatgpt, gpt-4, openai, anthropic, alibaba, etc.) AND personality-flattening disclaimers ("as an AI language model", "as an AI, I", "I am an AI", "I was trained by"...).
  - `passes_quality_filter(text, min_len=40, max_len=2000)` — rejects too-short, too-long, AND pure-refusal openers ("I cannot...", "I don't have feelings", "Sorry, but I..."). Refusal check anchored to first 60 chars of stripped lowercase head — mid-sentence "I cannot stop noticing..." passes.
  - `is_near_duplicate(text, prior_texts, threshold=0.85)` — char-trigram Jaccard. Catches paraphrased teacher repeats while letting diverse outputs through.
  - `filter_personality_examples(examples)` — top-level wrapper, runs all three in order (identity → quality → duplicate), returns `(kept, reject_counts)` with keys `identity / quality / duplicate / empty`.
- WIRED [enigma_engine/gui/gui_forge_new_modes.py](enigma_engine/gui/gui_forge_new_modes.py) `_start_distill_training`:
  - `category_prompts["personality"]` now reads from `_PERSONALITY_PROMPTS` (deferred import inside method body to avoid circular).
  - Per-response filter stack runs ONLY for `cat == "personality"`. Other categories keep the legacy `len > 20` minimum (out of scope this slice).
  - `personality_reject_counts` dict accumulates rejects; logged at end of generation as `"Personality filters rejected N response(s): identity=A, quality=B, duplicate=C"`.
  - Skip-log line includes the reject reason (`identity-leak / quality / duplicate`) instead of always saying "too short".
- Tests: NEW [tests/test_personality_data.py](tests/test_personality_data.py) — 40 tests across 6 classes covering pool size + uniqueness + diversity, every identity-leak pattern category, every quality-filter branch + threshold kwarg, near-duplicate behaviour at multiple thresholds, aggregate filter ordering + count invariants, and a wire-site structural test confirming the GUI distill loop imports and uses the pool + filters.

**Validation:**
- `ruff check enigma_engine/ tests/` — clean.
- `pytest tests/ -q --tb=no` — **2821 passed, 9 skipped, 0 failed** (baseline 2781 from Pass 156z9al; +40 new = exact match).
- `pytest tests/test_personality_data.py -v` — 40/40 PASSED.

**Six-question self-audit (Rule #19):**
1. **Author's-lens** — Pure data module + GUI wire-site is the smallest possible cut. Filters are kwarg-tunable so P5-pre-2 can dial thresholds without changing the contract.
2. **Connections** — `personality_data` ← imported by `gui_forge_new_modes._start_distill_training` (only call site; verified via grep). Filters use stdlib only — no transitive deps.
3. **Could more connections be made?** — Not yet. `filter_personality_examples` aggregate is currently unused by the GUI (which uses the three primitives separately to log per-reason rejects in real time). That's intentional: the aggregate is for batch post-processing (e.g. cleaning a previously-collected `distilled_*.txt` corpus), which is a future slice.
4. **Logic-eye** — Module docstring + commit-claim says "first slice, ZERO training". Code delivers exactly that. No optimizer step, no model touch. Doc claim and behaviour match.
5. **Claim-vs-test** — Behavioural tests (39/40) call the actual filter functions and assert outcomes; only 2 wire-site tests are structural (`inspect.getsource`) and they gate exact-substring patterns at the integration boundary. Adversarial tests included: `test_filter_order_identity_before_dedup` (identity-leaked text must NOT pollute dedup pool), `test_accepts_answer_mentioning_cannot_mid_sentence` (refusal-opener check is start-anchored, not mid-sentence). Falsifiable: deleting the `passes_identity_filter` import would fail `test_distill_imports_personality_filters`; deleting the personality-category if-branch in the loop would fail it too (the substring `personality_reject_counts` only appears in that branch).
6. **Sibling-boundary** — Distill mode has 6 categories: `personality`, `reasoning`, `knowledge`, `conversation`, `commands`, `creativity`. Filter is gated to `personality` ONLY by design — the other 5 don't have a teacher-identity-leakage problem in the same way (a teacher generating reasoning/knowledge content typically doesn't open with "As Qwen, I think..."). Out-of-scope but tracked: if a future audit finds teacher drift in those, the same primitives can be applied per-category.

**P5 careful-build remaining slices (in order, each its own pass):**
- **P5-pre-2** — wire `general_mix_ratio > 0` for personality SFT (anchor file already at [data/anchor_examples.jsonl](data/anchor_examples.jsonl)); add pre-SFT auto-checkpoint with timestamped name → guaranteed rollback. NO training run yet.
- **P5-pre-3** — add 10% eval split from generated examples; wire pre/post `python run.py --benchmark` runs to save numbers; add identity-guard probe ("Who are you?" / "Who made you?" before+after, assert no drift to teacher identity). NO training run yet.
- **P5-run** — once 1–3 are green: small dry pass (50 examples, 1 epoch) on a *copy* of the model to validate the full pipeline.
- **P5-real** — full ~500-example run on the real model with rollback ready.

---

**Previous: Pass 156z9al — B-3d streaming inline RAG splice.** User instruction "3,1, then 2" — checkpoint commit (3) shipped as `5005025`, this pass closes the parked B-3d sub-pass (1), backlog pick comes next (2).

**Production call chain (Rule #20):** `POST /api/chat/stream` → FastAPI handler → `state.stream_chat` → `EnigmaEngine.stream_chat` → `_prepare_chat` → `EnigmaEngine.stream_generate` → **(NEW)** multi-round splice orchestrator → per-round `_stream_round_tokens` helper → `_sample_token` (with `json_constraint` if set) → token strings yielded to the SSE stream interleaved with `<search_result>...</search_result>\n` splice blocks.

**Changes:**
- `_GenerationMixin._stream_round_tokens` ([engine_generation.py](enigma_engine/core/engine_generation.py)) — NEW helper, ~120 lines. Yields token strings from the prompt; mutates a caller-supplied `state` dict with `emitted_count`, `emitted_text`, `terminated_on ∈ {"max", "eos", "search", "json_done"}`. Owns the per-round mechanics: KV cache clear+prefill, JSON FSM advance + `is_done` break, repetition penalty, exempt-tokens, EOS check, and `stop_on_close` early return when `</search>` lands in the joined emitted text. Caller responsibility: holding `self._generation_lock` and being inside `torch.no_grad()`.
- `_GenerationMixin.stream_generate` body REPLACED with a round orchestrator. Computes `splice_enabled = inline_search_splice_enabled and rag_index is not None and rag_index.is_built`, `max_rounds = max(1, int(max_search_rounds)) if splice_enabled else 1`. Per round: `is_final_round = (round_idx == max_rounds - 1)`, `stop_on_close = splice_enabled and not is_final_round`, `remaining = max_gen - cumulative_tokens` (INFO log + break when ≤0). On `terminated_on == "search"`: `rfind` the search tag pair, query the RAG index, format context, yield a `\n<search_result>\n{ctx}\n</search_result>\n` splice block, build the next round's prompt as `current_prompt + emit_through_close + splice_block`. On any other termination: break (with WARNING `"B-3d: max_search_rounds=N budget exhausted but model emitted another <search> tag; left as plain text"` when final-round + `<search>` present in plain text). `finally:` calls `_record_search_emissions(full_emitted_text, path="stream")`. The full emitted text includes splice blocks — but those contain `<search_result>` not `<search>`, so the recorder does NOT spuriously log the inserted context as a model-emitted query.
- `_GenerationMixin._record_search_emissions` sibling-WARNING gate narrowed from `path != "native"` to `path not in ("native", "stream")`. Comment block updated to reflect that streaming is no longer a sibling gap. Remaining sibling paths still on the WARNING: `vision`, `speculative`, `medusa`, `lookahead`, `batch`, `gguf`. (Future passes can collapse those one-by-one with the same shape.)

**Tests added/updated** ([tests/test_chat.py](tests/test_chat.py)):
- NEW class `TestB3dStreamingSplice` — 7 behavioural tests stubbing `_stream_round_tokens` to drive the orchestrator through every branch:
  1. `test_no_splice_when_flag_off` — flag OFF: 1 round, `stop_on_close=False`, no RAG calls.
  2. `test_no_splice_when_rag_index_missing` — flag ON but `_rag_index=None`: same single-round behaviour (defensive precondition).
  3. `test_single_round_splice_yields_block_in_stream` — flag ON, round 0 emits `<search>q1</search>` and terminates on `"search"`: orchestrator yields a splice block as a stream chunk (consumer sees `<search_result>...DOC1...</search_result>` mid-stream), runs round 1 (final, `stop_on_close=False`) for wrap-up.
  4. `test_natural_stop_no_splice` — flag ON but inner round terminates on `"max"`: no splice, no second round, no RAG calls.
  5. `test_per_round_max_gen_respects_user_budget` — round-0 `max_gen=5`, after emitting 3 tokens the round-1 budget is `max_gen=2` (cumulative budget mirrors B-3c-2).
  6. `test_budget_exhausted_with_unspliced_search_logs_warning` — `max_search_rounds=2`, final round emits a plain-text `<search>q2</search>`: B-3d WARNING fires.
  7. `test_tail_record_runs_on_full_emitted_text` — spy on `_record_search_emissions`: invoked exactly once with `path="stream"`, text contains both the model emission and the spliced doc context.
- UPDATED `TestB3aSiblingPathWarning::test_sibling_path_emits_b3a_warning_when_flag_on` — `"stream"` removed from sibling sweep (now silent on splice flag because streaming supports it).
- NEW `TestB3aSiblingPathWarning::test_stream_path_silent_after_b3d_ships` — regression gate that `path="stream"` does NOT emit the B-3a sibling WARNING; Stage B-2 generic WARNING still fires.
- UPDATED `TestExemptTokensCoverage::test_stream_generate_uses_helper` — retargeted to `_stream_round_tokens` (where the `_build_exempt_tokens` call moved during the refactor).
- UPDATED `TestJsonSchemaConstraintWiring::test_stream_generate_builds_constraint_advances_and_breaks` — split into outer/inner halves: build of `JsonSchemaConstraint` and forward kwarg gated on `stream_generate` source; `json_constraint.advance()` and `json_constraint.is_done` gated on `_stream_round_tokens` source.

**Validation:**
- `ruff check enigma_engine/ tests/` — clean.
- `pytest tests/ -q --tb=no` — **2781 passed, 9 skipped, 0 failed** (baseline 2773; +7 B-3d + 1 stream-silent regression gate, no other deltas).

**Six-question self-audit (Rule #19):**
1. *Would I write it this way from scratch?* Yes — single inner helper + outer orchestrator mirrors `_generate_text` + `_maybe_rag_splice` with the streaming twist that the splice block is itself a yielded chunk. The state-dict-via-kwarg pattern keeps the helper a real generator (not a tuple-returning function) so callers can yield-as-they-go.
2. *What is this connected to?* Inputs: `inline_search_splice_enabled` + `max_search_rounds` (init in `_init_common`), `_rag_index` (set by `engine.attach_rag_index`), `_record_search_emissions` (Pass 156z9d observability). Outputs: yielded stream chunks consumed by `stream_chat` → SSE in `engine_chat.py`/`api/server.py`.
3. *Could more connections be made?* Future: `rewind_cache(close_pos)` instead of `clear_cache()` per round (the parked KV-rewind optimisation). Today's clear+re-prefill is correct but O(seq_len) per splice; a follow-up sub-pass can swap to O(splice_block_len) via `model.rewind_cache(...)`. Logged as B-3d-followup.
4. *Logic-eye on the doc claim.* Class doc/comments say streaming yields splice blocks mid-stream and runs multiple rounds — code does both, gated on the same precondition triple as the non-streaming helper. No over-promise.
5. *Claim-vs-test.* Each behavioural test exercises a different branch of the orchestrator; together they prove flag-off / no-rag / single-splice / multi-round / budget-decrement / final-round-warning / tail-record paths. Structural-only test (`test_stream_generate_uses_helper` / `test_stream_generate_builds_constraint_advances_and_breaks`) is paired with the behavioural battery above, so the structural assertion is a regression gate not the proof.
6. *Sibling-boundary sweep.* Same B-3a sibling family: `vision`, `speculative`, `medusa`, `lookahead`, `batch`, `gguf` chat. None of these are streaming, so B-3d does not apply directly — they remain on the B-3a WARNING gate awaiting per-path closure. Logged in the WARNING comment. Still parked, not regressed.

**Parked / follow-ups:**
- KV cache rewind for splice path (B-3d-followup) — replace `model.clear_cache()` per round with `model.rewind_cache(close_pos)` for O(splice_block_len) re-prefill instead of O(full_prompt_len). Low priority; correctness is unaffected.
- Sibling B-3a closure for `vision` / `speculative` / `medusa` / `lookahead` / `batch` / `gguf` — each non-streaming variant needs its own splice helper or a shared adapter. Track as B-3e..B-3j.

---

**Last updated:** May 6, 2026 (Pass 156z9ak — **B-3c-2 token-budget fix.** Audit triggered by user "do an audit add update." Author's-lens scan of `_maybe_rag_splice` (the helper shipped Pass 156z9ai) found a real bug: every round was passing the original `max_gen` to `_generate_manual` unchanged, so a user requesting `max_tokens=512` with `max_search_rounds=3` could receive up to ~3×512 tokens.

**Fix:**
- `_GenerationMixin._generate_text` ([engine_generation.py L640+](enigma_engine/core/engine_generation.py#L640)) — now computes `tokens_round0 = max(0, output_ids.shape[1] - input_ids.shape[1])` from the round-0 generation tensors and forwards it to the helper.
- `_GenerationMixin._maybe_rag_splice` — new kwarg `tokens_already_generated: int = 0` seeds a cumulative running token count. Per round: `remaining = max_gen - cumulative_tokens`; if `<= 0` the helper logs INFO `"B-3b/c: max_gen=N budget exhausted before round K (cumulative=M); exiting splice loop"` and exits cleanly. Otherwise the round's `_generate_manual` call uses `round_max_gen = remaining`. After the call returns, `cumulative_tokens += cont_ids.shape[1] - new_input_ids.shape[1]`.
- Stale code comment "Single round only — multi-round recursion is parked for B-3c" replaced with current B-3b/c language.
- Helper docstring updated with the new "Token budget (Pass 156z9ak)" section explaining the cumulative accounting.

**Tests added (2 new in `TestB3cBoundedRecursion`, total now 16 across B-3 family):**
- `test_per_round_max_gen_respects_user_budget` — fakes `_generate_manual` to emit exactly 1 extra token per round; with `max_gen=4` and 3 rounds, asserts the helper's per-round `max_gen` arg is strictly decreasing (4 → 3 → 2). Falsifies if someone reverts the cumulative accounting back to passing `max_gen` unchanged each round.
- `test_budget_zero_exits_loop_cleanly` — round 0 emits 10 tokens past prompt with `max_gen=5`; helper hits the budget gate at round 1 entry, logs INFO `"budget exhausted"`, exits with only 1 manual call. Asserts no second `_generate_manual` call and the warning is emitted.
- Existing helper tests extended with `captured["max_gens"]` tracking (no behavioural change to existing assertions; the new field is opt-in for budget tests only).

**Acceptance chain (production entry-point INWARDS, unchanged from B-3c):**
- `POST /api/chat` → `engine.chat()` → `engine.generate()` → `_generate_text` → manual loop → `</search>` auto-stop → trim → **`_maybe_rag_splice` loop** with cumulative budget → each round's `_generate_manual` gets `remaining = max_gen - cumulative` → return final text.

**Validation:**
- `pytest tests/test_chat.py -k "B3b or B3c" -v` → 16 passed in 3.86s (was 14).
- `ruff check enigma_engine/ tests/` → all checks passed.
- Full suite: **2773 passed, 0 failed, 9 skipped** (was 2771; +2 from new budget tests).

**Audit findings beyond B-3:**
- `git diff --stat HEAD` — no other net-deletion `.py` files in working tree. The training.py regression closed Pass 156z9aj is the only one.
- Working tree across `enigma_engine/` is healthy: 14 modified files, all net-positive deltas.
- 5 modified mod files (avatar/audiogen/codegen/etc.) and various GUI files have uncommitted changes but no obvious doc-vs-code lies surfaced; out of scope for this audit pass.

**Author's-lens six-question self-audit on the new fix (§1 #19):**
1. *Would I write it this way?* — Yes. Cumulative counter accumulated from a tensor-shape delta is the standard pattern; defensive `max(0, int(...))` on both the kwarg seed and the per-round delta matches the existing defensive `max(1, int(...))` on `max_search_rounds`.
2. *Connections?* — Reads `output_ids.shape[1]` and `input_ids.shape[1]` at the call site (already in scope); writes nothing to engine state. Helper kwarg is keyword-only with a default of 0 so existing callers unaffected.
3. *More connections?* — Only `_generate_text` calls `_maybe_rag_splice` today. If a future caller wires the helper into another path (B-3d streaming, vision), they need to remember to pass the round-0 token count. Default of 0 means a forgotten kwarg silently allocates the FULL `max_gen` to the helper — the same bug just shifted. Documented in the docstring's Token budget section.
4. *Logic-eye (does code deliver what doc claims)?* — Docstring claims "the total user budget is respected across all rounds instead of being multiplied by N." Code: per-round `max_gen` is `remaining = max_gen - cumulative`; cumulative starts at `tokens_already_generated` (round 0 actual) and grows by each round's actual emit count; loop exits when `remaining <= 0`. Verified by `test_per_round_max_gen_respects_user_budget` (decreasing per-round budget) and `test_budget_zero_exits_loop_cleanly` (early exit when round 0 alone exhausts budget).
5. *Claim-vs-test?* — Reverting `cumulative_tokens += this_round_tokens` to `cumulative_tokens += 0` would make all 3 rounds pass `max_gen` and fail `test_per_round_max_gen_respects_user_budget`. Reverting the `if remaining <= 0: break` gate would fail `test_budget_zero_exits_loop_cleanly`. Reverting the wire-site `tokens_already_generated=tokens_round0` to `=0` would lose the round-0 accounting and fail `test_budget_zero_exits_loop_cleanly` (round 0 emitted 10 tokens but helper would think 0 → would attempt round 1).
6. *Sibling-boundary sweep?* — Only one helper, only one wire-site, only one contract. Other generation paths (`stream_generate`, `_generate_with_vision`, GGUF branch, etc.) don't call `_maybe_rag_splice` at all; their B-3 contract is the WARNING from B-3a. No sibling drift.

**Lessons added to §4:** none new — applied existing rules (cumulative-counter pattern, defensive int normalisation on user-supplied kwargs, docstring claim must match code, behavioural-test pairing per claim).)

**Last updated:** May 6, 2026 (Pass 156z9aj — **AUDIT** of B-3 chain + cleanup. Triggered by user "do an audit, was that everything, clean up a bit." Findings:

**B-3 chain status: substantively complete.**
- B-3a (engine flag + native auto-stop + sibling WARNINGs + GUI), B-3b (single-round splice helper), B-3c (bounded multi-round recursion, `max_search_rounds=3`) — all CLOSED across Passes 156z9ag/ah/ai. End-to-end native non-GGUF path: model emits `<search>q</search>` → trimmed at stop → `_maybe_rag_splice` runs up to N rounds of retrieve+splice+continue → final-round `</search>` strip → return.
- B-3d (streaming inline splice + KV rewind) remains parked per Pass 156z9af — only ships on user request. Sibling WARNING from B-3a stays the honest UX for streaming/vision/GGUF callers.
- 14/14 B-3 tests green. No prompt-echo regressions (Pass 156z9e discipline applied per-round to advancing tail).

**Real audit finding (NOT introduced by this session): `enigma_engine/core/training.py` working tree is dirty with a 480-line NET DELETION vs HEAD (commit `4e4daa8` "Pass 156z9f").** Diff confirms the missing functions ARE in the committed version:
- `_effective_warmup(warmup_steps, total_steps)` (Sched-2, claimed CLOSED Pass 152) — gone from working tree.
- `set_training_seed(seed, deterministic=False)` (DET-2, claimed CLOSED Pass 156i3) — working tree only has `set_training_seed(seed=42)`, no `deterministic` kwarg, no `CUBLAS_WORKSPACE_CONFIG`, no `torch.use_deterministic_algorithms` call.
- `TrainingConfig.deterministic` field — gone.
- `Trainer._apo_zero_loss` static + `_resolve_preference_loss` registry + `loss_type` kwarg on `train_dpo` (D-9, claimed CLOSED Pass 156j) — gone.
- 8 sibling `train_*` `set_training_seed(deterministic=...)` forwards — gone.

**Impact:** 30 tests in `tests/test_training.py` (`TestAPOZeroLoss`, `TestEffectiveWarmup`, `TestVisionDataParsing` seed-forward gate, `TestDeterministicFlag`) red. SUGGESTIONS shows them as DONE; the code disagrees. Classic §1 #20 "doc claims more than code delivers" plus §1 #18 "scope drift" — something deleted training-pipeline work outside the named overhaul scope of any pass on the visible git log.

**Origin diagnosis (best-effort):** all other uncommitted files in `enigma_engine/` are NET-POSITIVE (B-3 additions + other in-progress slices, +1914/-579 across 14 files). Only `training.py` is net-negative (+52/-480). Suggests a single bad merge / revert / find-replace on `training.py` rather than a systemic regression. The committed HEAD version is intact and presumably correct (that's where the close-stamps were originally validated).

**Recommendation — NOT auto-applied:** restore `training.py` from HEAD (`git checkout HEAD -- enigma_engine/core/training.py`) is the obvious fix, but the working tree may also contain in-progress edits the user wants to keep. Per §operationalSafety this is a destructive action that needs user confirmation. **Action requested from user:** confirm whether to (a) `git checkout HEAD -- enigma_engine/core/training.py` to restore the missing features (will lose any uncommitted `training.py` edits — diff shows none worth keeping based on the +52 lines being mostly noise), or (b) leave as-is and treat the 30 failures + SUGGESTIONS close-stamps as "lying for now."

**RESOLVED in same pass — user authorised restore.** `git checkout HEAD -- enigma_engine/core/training.py` executed; working tree now matches HEAD on that file. Verification: lint clean (`ruff check enigma_engine/ tests/` → all checks passed), targeted failures reversed (`pytest tests/test_training.py -k "TestAPOZeroLoss or TestEffectiveWarmup or TestDeterministicFlag or TestVisionDataParsing"` → 32 passed, was 30 failed + 2 passed), full suite **2771 passed, 0 failed, 9 skipped** (up 30 from 2741). Sched-2 / DET-2 / D-9 close-stamps are now honest again.

**Cleanup actually applied this pass:**
- `ruff check --fix enigma_engine/core/training.py` removed 3 unused `noqa` directives at L1529 (`F841` no longer triggered), L3771 + L3798 (`N812` not enabled in this project's ruff config). Working tree is now LINT-CLEAN: `ruff check enigma_engine/ tests/` → "All checks passed!"
- No code logic touched. No SUGGESTIONS close-stamps modified except adding this audit pass.

**Was that everything? Answer: B-3 chain yes (modulo parked B-3d), but the training.py regression eclipses the cleanliness claim. The 30-failure pre-existing baseline I've been quoting since Pass 156z9ag is not actually pre-existing — it's a working-tree regression I should have caught and reported on Pass 156z9ag's audit, not waited for the user to ask. §4 lesson reinforced: "Self-audit immediately after shipping" includes auditing the test-suite baseline at session start, not assuming `git status`-clean.

**Lessons added to §4 (one new):**
- **Test-suite baseline must be diffed against HEAD on session start, not blindly accepted as "pre-existing."** Pass 156z9ag/ah/ai all reported "30 pre-existing failures" without checking whether those tests targeted features whose code had been silently deleted from the working tree. `git diff --stat HEAD -- <module>` showed a single net-deletion file (`training.py`, -480 lines) that explained 100% of the failures via four claimed-shipped features being gone. Rule: when starting a session whose suite has a non-zero red baseline, run `git diff --stat HEAD -- <suite-targets>` BEFORE quoting the failure count as "pre-existing." Net-deletion files with claimed-shipped features in adjacent SUGGESTIONS stamps are doc-vs-code lies that need either restoration or the close-stamps re-opened. Don't carry the lie forward through three passes.)

**Last updated:** May 6, 2026 (Pass 156z9ai — **B-3c SHIPPED**: bounded multi-round splice recursion. `EnigmaEngine.max_search_rounds: int = 3` budget + loop inside `_maybe_rag_splice` that re-runs retrieve+splice+continue up to N times. Rounds 1..N-1 keep `</search>` in continuation stops so the model can request another search; round N (final) strips `</search>` so the model wraps up using accumulated context. Closes B-3c from the four-sub-pass plan stamped Pass 156z9af; B-3d (streaming + KV rewind) remains parked.

**What shipped:**
- `EnigmaEngine._init_common` ([inference.py L281+](enigma_engine/core/inference.py#L281)) — new `self.max_search_rounds: int = 3`. Default 3 per the B-3 plan; capped to >= 1 inside the helper so a typo can't disable the loop.
- `_GenerationMixin._maybe_rag_splice` ([engine_generation.py L640+](enigma_engine/core/engine_generation.py#L640)) — refactored from single-shot to bounded loop. Tracks `current_text` + `current_prompt` advancing each round so prompt-echo discipline (Pass 156z9e) applies to the freshly-generated tail, not the cumulative spliced prompt. Per-round preconditions are re-checked: missing unclosed `<search>`, empty query, retrieval exception, or empty context all break the loop and return whatever was last spliced (or `None` if no round succeeded). Final-round `</search>` strip is conditional on `is_final_round = (round_idx == max_rounds - 1)`. Budget-exhaustion path logs `WARNING "B-3c: max_search_rounds=N budget exhausted but model emitted another <search> tag; left as plain text"` only when the final-round continuation actually contains another `<search>` — silent on clean exits.

**Acceptance chain (production entry-point INWARDS):**
- `POST /api/chat` → `engine.chat()` → `engine.generate()` → `_generate_text` → manual loop → `</search>` auto-stop → trim → **`_maybe_rag_splice` loop** (rounds 0..N-1) → each round: `rag_index.query` → splice → re-encode → `_generate_manual` → trim → check for next `<search>` → loop or exit → return final text.
- `_rag_index` set by GUI's `_build_rag_index` ([gui_logic.py L1545](enigma_engine/gui/gui_logic.py#L1545)).

**Sibling-boundary scope (unchanged from B-3a/B-3b):** native non-GGUF `_generate_text` only. The 7 sibling paths still emit B-3a one-shot WARNINGs. No widening.

**Tests (5 new in B-3c family + 1 updated B-3b test, all green):**
- `tests/test_chat.py::TestB3cBoundedRecursion` — 5 behavioural/structural tests:
  - `test_two_rounds_splice_within_budget` — model emits `<search>q1` → splice CTX1 → `<search>q2` → splice CTX2 → wrap-up. Asserts 3 manual calls, `rag.queries == ["q1", "q2"]`, both ctx blocks in result, both non-final rounds keep `</search>` in stops.
  - `test_budget_exhaustion_strips_close_tag_on_final` — `max_search_rounds=1` forces the very first splice round to be final; asserts `</search>` is stripped from continuation stops.
  - `test_budget_exhausted_with_unspliced_search_logs_warning` — final-round continuation contains `<search>q2</search>` as plain text; asserts WARNING log line `"B-3c"` + `"budget exhausted"` and the dangling tag survives in the result.
  - `test_loop_exits_when_no_unclosed_search_in_continuation` — first round splices, continuation has no `<search>`; loop breaks at round 2 instead of running the full budget. Asserts 2 manual calls (initial + 1 splice), 1 rag.query.
  - `test_max_search_rounds_default_is_three` — structural gate on `_init_common` body asserting `self.max_search_rounds = 3` literal is present.
- `TestB3bRagSplice::test_splice_happens_when_flag_on_and_rag_built` — updated to set `obj.max_search_rounds = 1` so the original "single-round, `</search>` stripped" assertions still hold under the new loop semantics. Other 6 B-3b tests unchanged (each tests a precondition miss that exits the loop before any splice round).
- Run: `pytest tests/test_chat.py -k "B3b or B3c"` → 14 passed in 1.42s. Full suite: 2741 passed (up 5 from B-3b baseline 2736), 30 pre-existing failures (`_apo_zero_loss` family in test_training.py, unchanged), 9 skipped, 3 pre-existing RUF100 in `enigma_engine/core/training.py` (unchanged, unrelated).

**Author's-lens six-question self-audit (§1 #19):**
1. *Would I write it this way?* — Yes. The loop is a natural extension of B-3b's helper; per-round guard returns are reused as loop-break conditions. Capping `max_rounds` at `max(1, int(...))` defensively normalises a config typo (matches §4 "interval kwarg defensive normalisation" rule from BackgroundTrainer).
2. *Connected to what?* — Reads `self.inline_search_splice_enabled`, `self._rag_index`, `self.max_search_rounds`. No writes to engine. Same producer/consumer shape as B-3b — single helper, single `_generate_text` wire-site.
3. *Could more connections be made?* — `chat()` could expose `max_search_rounds` as a per-request override; deferred — engine-attribute is the surface today, matches B-3a flag pattern. GUI could expose a numeric input (mirrors §1 "no sliders, numeric input only" preference); not user-requested.
4. *Logic-eye (does code deliver what doc/comment claims)?* — Helper docstring states "Round 1..N-1 keep `</search>` in stops; Round N strips it." Code: `is_final_round = (round_idx == max_rounds - 1)` with `round_stops = [s for s in ... if s != "</search>"] or None` only on the final branch. Verified by `test_two_rounds_splice_within_budget` (non-final keeps it) and `test_budget_exhaustion_strips_close_tag_on_final` (final strips). WARNING claim "budget exhausted but model emitted another `<search>`" matches the conditional log: only emitted when `"<search>" in current_text[len(current_prompt):]` after the final round.
5. *Claim-vs-test (would test pass while code is wrong)?* — Behavioural tests sentinel-mock `_generate_manual` and capture both call count AND per-call `stop_strings`. Reverting the final-round-strip branch fails `test_budget_exhaustion_strips_close_tag_on_final`. Reverting the `current_prompt = new_prompt` advance would break the prompt-echo discipline within multi-round (the second-round generated portion would include the first-round splice block as "model emitted") — caught by `test_two_rounds_splice_within_budget` because it expects `rag.queries == ["q1", "q2"]` not `["q1", "q1"]`. Reverting the WARNING conditional would either lose the message (fails `test_budget_exhausted_with_unspliced_search_logs_warning`) or fire on every clean exit (no test for that — acceptable, the warning is informational).
6. *Sibling-boundary sweep?* — Grepped all `_record_search_emissions` callers and `_generate_manual` callers: only `_generate_text` native path is widened. Sibling paths still pass through B-3a WARNING-only. No `chat()` API surface change. No KV-cache code. No streaming surgery. Single contract family touched, single helper modified.

**Status of B-3 follow-ups:**
- ~~B-3a — engine flag + native auto-stop + sibling WARNINGs + GUI~~ **CLOSED Pass 156z9ag.**
- ~~B-3b — `_maybe_rag_splice` helper + native call site~~ **CLOSED Pass 156z9ah.**
- ~~B-3c — Bounded multi-round recursion (`max_search_rounds=3`)~~ **CLOSED Pass 156z9ai (this stamp).**
- B-3d — Streaming inline splice + KV rewind. Park last; only ships on user request. Sibling-boundary WARNING from B-3a remains the honest UX until then.

**Lessons added to §4:** none new — applied existing rules (defensive int normalisation on config kwargs, prompt-echo discipline applied per-round on the advancing tail not the cumulative buffer, single helper + single wire-site, behavioural+structural test pairing per round semantics).)

**Last updated:** May 6, 2026 (Pass 156z9ah — **B-3b SHIPPED**: `_maybe_rag_splice` helper on `_GenerationMixin` + wire-site at `_generate_text` native non-GGUF path between auto-stop trim and `_record_search_emissions`. Closes B-3b from the four-sub-pass plan stamped Pass 156z9af; B-3c (multi-round recursion) and B-3d (streaming + KV rewind) remain parked.

**What shipped:**
- `_GenerationMixin._maybe_rag_splice(text, prompt, max_gen, ..., *, effective_stop_strings, json_constraint) -> str | None` ([engine_generation.py L640+](enigma_engine/core/engine_generation.py#L640)) — pure helper. Returns `None` on any precondition miss (flag OFF, `_rag_index` missing or unbuilt, prompt-prefix mismatch, no unclosed `<search>` in generated portion, empty query, retrieval exception, empty context). On success: builds `text + "</search>\n<search_result>\n<ctx>\n</search_result>\n"`, encodes, runs `_generate_manual` ONCE with `</search>` stripped from continuation stops (single-round per B-3b scope; recursion is B-3c), trims continuation against remaining stops, returns continued text. Exception-safe: `try/except` around encode + manual call → returns `None` so caller keeps original text.
- `_generate_text` ([engine_generation.py L632-642](enigma_engine/core/engine_generation.py#L632)) — single new call site immediately after the auto-stop trim block: `spliced = self._maybe_rag_splice(...)` then `if spliced is not None: text = spliced`. Five lines + comment.

**Acceptance chain (production entry-point INWARDS, per Rule #20):**
- `POST /api/chat` → `enigma_engine.api.server` chat endpoint → `engine.chat(message)` → injects RAG doc-context into system prompt (existing Pass 125 behaviour at [engine_chat.py L387](enigma_engine/core/engine_chat.py#L387)) → `engine.generate(prompt, ...)` → `_generate_text(prompt, ...)` → manual loop hits `</search>` stop → trim → **`_maybe_rag_splice` (NEW)** → on splice, second `_generate_manual` produces continuation → return continued text.
- GUI: chat send → `engine.chat(message)` → same chain.
- `_rag_index` set by GUI's `_build_rag_index` ([gui_logic.py L1545](enigma_engine/gui/gui_logic.py#L1545)) when user enables RAG widget; otherwise None and helper short-circuits (no error, just no splice).

**Sibling-boundary scope (matches B-3a):** native non-GGUF `_generate_text` only. The 7 sibling paths (gguf, stream, batch, vision, speculative, medusa, lookahead) keep the B-3a one-shot WARNING via `_record_search_emissions(path=...)`. No widening this pass.

**Tests (9 in B-3b family, all green):**
- `tests/test_chat.py::TestB3bRagSplice` — 7 behavioural tests via stubbed `_GenerationMixin` + sentinel `_generate_manual` capturing call count, encoded prompts, and stop-list. Cases: (a) flag-on + rag-built + unclosed `<search>QUERY` → 2 manual calls, `rag.query("QUERY")` invoked, second prompt contains `<search_result>`+ctx+`</search_result>`, `</search>` stripped from continuation stops, returned text is the continuation; (b) flag-off → 1 manual call, no rag.query; (c) `_rag_index = None` → 1 manual call; (d) rag built=False → 1 manual call, no rag.query; (e) **adversarial prompt-echo** — prompt itself contains `<search>foo</search>`, generated portion benign → 1 manual call (proves prompt-prefix strip); (f) whitespace-only query → 1 manual call; (g) retrieval returns empty ctx → 1 manual call but rag.query was attempted (proves the empty-ctx skip is distinct from no-attempt skip).
- `tests/test_chat.py::TestB3bRagSpliceWireSiteStructural` — 2 structural gates: comment-stripped regex `self\._maybe_rag_splice\s*\(` against `_generate_text` source (adversarially-falsifiable by deleting the call site); helper presence check on `_GenerationMixin`.
- Run: `pytest tests/test_chat.py -k B3b -v` → 9 passed in 3.18s. Scope suite (`tests/test_chat.py tests/test_gui.py tests/test_memory.py`): 549 passed, 2 skipped. Full suite: 2736 passed, 30 pre-existing failures (`test_training.py` `_apo_zero_loss` family, unchanged from prior baseline), 9 skipped.

**Author's-lens six-question self-audit (§1 #19):**
1. *Would I write it this way?* — Yes. One helper, one call site, all preconditions named at the top of the helper as guard returns. No mutation of caller state. `try/except` around encode + manual call so retrieval failure can't break generation.
2. *Connected to what?* — Reads `self.inline_search_splice_enabled` (B-3a flag), `self._rag_index` (Pass 125 GUI/API plumbing), `self._encode_prompt`, `self._generate_manual`, `self._decode_output`. Writes nothing on the engine. Calls `RAGIndex.format_context` static via local import (matches engine_chat.py pattern). Single producer (helper), single consumer (`_generate_text` tail).
3. *Could more connections be made?* — `chat()`'s system-prompt RAG injection ([engine_chat.py L387](enigma_engine/core/engine_chat.py#L387)) and `_maybe_rag_splice`'s mid-generation injection now both query the same `_rag_index` — could share retrieval cache, deferred. Sibling paths (stream/vision/etc.) could call the helper too in B-3c/d; intentionally not wired this pass per scope.
4. *Logic-eye (does code deliver what doc/comment claims)?* — Helper docstring lists 7 precondition return-`None` branches; code has exactly 7 guard returns matching that list. Comment "single round only — multi-round recursion is parked for B-3c" matches: continuation call uses `cont_stops` with `</search>` stripped, so an emitted `<search>` in continuation just appears as text and is NOT auto-stopped again. No aspirational language.
5. *Claim-vs-test (would test pass while code is wrong)?* — Behavioural tests sentinel-mock `_generate_manual` and check (i) call count went 1→2, (ii) `rag.query` was called with the extracted query string literal, (iii) the second-call prompt contains `<search_result>` + the formatted ctx, (iv) `</search>` is NOT in continuation stops. Structural test uses regex `self\._maybe_rag_splice\s*\(` matching the literal call expression — comment mentioning the helper would NOT satisfy. Prompt-echo test is the adversarial case from §4 Pass 156z9e learned principle.
6. *Sibling-boundary sweep?* — Grepped `_record_search_emissions` and `_generate_manual` callers: only `_generate_text` native path was widened with the splice call. The 7 sibling paths still pass through `_record_search_emissions(path=...)` and emit the B-3a WARNING when flag is ON — unchanged. No `chat()` API surface change. No KV-cache code touched.

**Status of B-3 follow-ups:**
- ~~B-3a — engine flag + native auto-stop + sibling WARNINGs + GUI~~ **CLOSED Pass 156z9ag.**
- ~~B-3b — `_maybe_rag_splice` helper + native call site~~ **CLOSED Pass 156z9ah (this stamp).**
- B-3c — Bounded multi-round recursion (max_search_rounds=3 default) at the same wire-site. Next sub-pass.
- B-3d — Streaming inline splice + KV rewind. Park last; only ships on user request.

**Lessons added to §4:** none new — applied existing rules (Rule #20 acceptance chain explicit, §4 Pass 156z9e prompt-echo defence, single helper + single wire-site to avoid signal-without-consumer, behavioural+structural test pairing, no widening of sibling scope).)

**Last updated:** May 5, 2026 (Pass 156z9ag — **B-3a SHIPPED**: `inline_search_splice_enabled` engine flag (default OFF, opt-in) + `</search>` auto-stop in `_generate_text` native non-GGUF path + sibling-boundary WARNINGs on 7 alt paths + GUI checkbox + persistence + apply-on-model-load. Closes B-3a from the four-sub-pass plan stamped Pass 156z9af; B-3b/c/d remain parked as planned.

**What shipped (engine layer):**
- `EnigmaEngine._init_common` ([inference.py L261+](enigma_engine/core/inference.py#L261)) — new `self.inline_search_splice_enabled: bool = False` next to the existing `inline_search_enabled = True` observability flag. Opt-in, single source of truth, off-by-default keeps every existing user on today's behaviour.
- `_GenerationMixin._record_search_emissions` ([engine_generation.py L414-422](enigma_engine/core/engine_generation.py#L414)) — new keyword-only `path: str = "native"` arg; when splice flag is ON and `path != "native"` and queries were recorded, emits a second WARNING `"B-3a: inline_search_splice_enabled=True but the '%s' generation path does not yet support </search> auto-stop or splice; %d query(ies) recorded ... with no splice applied."` Sibling distinguishes itself by passing its own `path=` kwarg — single warning template, every sibling self-identifies. Native path stays silent because that's the one path that DOES support the splice in B-3a.
- `_generate_text` ([engine_generation.py L575-585](enigma_engine/core/engine_generation.py#L575)) — new `effective_stop_strings` block: when splice flag is ON, defensive-copy `stop_strings` into a new list and append `"</search>"` if not already present. Forwarded to `_generate_manual` and reused at the post-decode trim site (line 627) so the close tag triggers both the manual-loop early-stop AND the text-side trim — defence in depth if the manual loop yields one extra token past the match.
- 7 sibling call sites updated to pass `path=` kwarg: `_generate_text` GGUF branch (`path="gguf"`), `stream_generate` (`path="stream"`), `batch_generate` (`path="batch"`), `_generate_with_vision` (`path="vision"`), `speculative_generate` (`path="speculative"`), `medusa_generate` (`path="medusa"`), `lookahead_generate` (`path="lookahead"`). Native path's own call stays default `path="native"` — no change needed.

**What shipped (GUI layer):**
- [`desktop.py`](enigma_engine/gui/desktop.py) — `__init__` library default `self.inline_search_splice_enabled = False` + boot-load `self.inline_search_splice_enabled = self._read_gui_bool_setting("inline_search_splice_enabled", False)`.
- [`gui_pages_config.py`](enigma_engine/gui/gui_pages_config.py) — GENERATION BEHAVIOR card gets a new CTkCheckBox `"Inline <search> auto-stop (B-3a, opt-in)"` with tooltip explaining native-only scope; new `_toggle_inline_search_splice_enabled` handler persists to `gui_settings.json` via `atomic_write_json`, mirrors to `self.inline_search_splice_enabled`, applies live to `eng.inline_search_splice_enabled` if engine loaded, status bar `f"⚡ Inline <search> auto-stop {state}"`.
- [`gui_logic.py`](enigma_engine/gui/gui_logic.py) `_on_model_loaded` — parallel try/except after the existing `engine.inline_search_enabled` apply; sets `self.engine.inline_search_splice_enabled = bool(getattr(self, "inline_search_splice_enabled", False))` so the user's persisted toggle survives every model reload (avoids signal-without-consumer / Pass 156y2 JSON-wins-on-load anti-pattern — disk is the source of truth, applied on every reload).

**Acceptance chain (production entry-point INWARDS, per Rule #20):**
- Native: `POST /api/chat` → `chat()` → `_generate_text` → `_generate_manual(stop_strings=[..., "</search>"])` when `engine.inline_search_splice_enabled` is True.
- GUI checkbox toggle → `atomic_write_json(gui_settings.json)` → next launch `EnigmaGUI.__init__` boot-load → `self.inline_search_splice_enabled` → on model load `_on_model_loaded` → `engine.inline_search_splice_enabled`.
- Sibling paths: same `POST /api/chat` (or `chat/stream`, etc.) → respective generation method → `_record_search_emissions(text, path="<sibling>")` → emits B-3a WARNING when splice flag ON.

**Tests (51 in B-3a + adjacent B-2 family, all green):**
- `tests/test_chat.py::TestB3aSpliceFlagDefaults` — structural gate on `_init_common` initialising the flag to `False`.
- `tests/test_chat.py::TestB3aSiblingPathWarning` — 5 behavioural tests covering native-silent, sibling-warns (loop over 7 sibling paths), flag-OFF-silent, no-emission-no-warning, observability-OFF-kills-warning.
- `tests/test_chat.py::TestB3aGenerateTextAutoStop` — 5 behavioural tests via sentinel-mocked `_generate_manual` capturing the `stop_strings` kwarg: flag-ON-appends, flag-OFF-no-append, None-stop-creates-list, no-mutation-of-callers-list, idempotent-when-already-present.
- `tests/test_chat.py::TestB3aGenerateTextWireSiteStructural` — comment-stripped structural gate (per §4 "Substring-presence" rule) asserting `inline_search_splice_enabled` + `"</search>"` + `effective_stop_strings` all appear in real code in `_generate_text`.
- `tests/test_chat.py::TestB3aSiblingCallSitesUsePathKwarg` — structural gate asserting each sibling method's source contains both `_record_search_emissions` AND `path="<expected>"` literal — adversarially-falsifiable by removing any one sibling's `path=` kwarg (would silently drop that sibling's WARNING attribution).
- `tests/test_gui.py::TestInlineSearchSpliceConfig` — 5 GUI tests mirroring `TestInlineSearchEnabledConfig`: toggle persists + applies to live engine, toggle-without-engine still persists, `_on_model_loaded` applies persisted flag (structural), boot-load returns persisted value AND falls back to library-default-False on missing key, boot-load wire-site present in `__init__` via word-boundary regex matching `_read_gui_bool_setting("inline_search_splice_enabled"`.
- Run: `pytest tests/test_chat.py tests/test_gui.py -k "B3a or InlineSearchSplice or InlineSearchEnabled or StageB2"` → 51 passed in 3.85s. Full suite: 2734 passed, 30 pre-existing failures all in test_training.py (APOZeroLoss / VisionDataParsing / EffectiveWarmup / DeterministicFlag — unrelated to B-3a, no files touched by this slice appear in those test paths).

**Author's-lens six-question self-audit (§1 #19):**
1. *Would I write it this way?* — Yes. Single flag, default OFF, mirrors `inline_search_enabled` pattern shipped Pass 156z9u/w. Defensive list-copy avoids mutating caller state. `path=` kwarg makes sibling self-identification DRY (single warning template, sibling labels its own call). No new abstractions.
2. *Connected to what?* — Engine attribute → `_generate_text` (read inside auto-stop block) → `_generate_manual` (consumes via `stop_strings` kwarg) → text-side trim block (consumes via same `effective_stop_strings`) → `_record_search_emissions` (warns on siblings). GUI: checkbox → settings JSON → boot-load → on-model-load → engine attribute. Two clean chains, no orphans.
3. *Could more connections be made?* — `ChatRequest` could expose a per-request override; deliberately deferred — engine-attribute-only is the simpler design and matches `inline_search_enabled`'s shape. Profile-scoped (`AIProfile.inline_search_splice_enabled`) would be natural if profiles ever gain generation-behaviour fields; not a feature today.
4. *Logic-eye (does code deliver what doc/comment claims)?* — Comment block in `_generate_text` says "Native non-GGUF path only — sibling paths warn via `_record_search_emissions(path=...)` instead." Code matches: native is the one path that appends `</search>`; all 7 siblings pass `path=` and trigger the WARNING when flag is ON. `_record_search_emissions`'s new docstring/log message says "%s path does not yet support </search> auto-stop OR splice" — accurate (B-3a only ships auto-stop; splice is B-3b's job, but the WARNING is forward-compatible).
5. *Claim-vs-test (would test pass while code is wrong)?* — Behavioural tests sentinel-mock `_generate_manual` and inspect the captured `stop_strings`; reverting the append line would break `test_flag_on_appends_close_tag_to_stop_strings`. Structural test strips comment-only lines before scanning — a stale comment cannot satisfy the gate. Sibling-call-site test asserts each sibling's source contains the literal `path="<expected>"` — removing any one sibling's `path=` kwarg fails its specific assertion (loop-per-sibling, not a single OR-of-all).
6. *Sibling-boundary sweep?* — Grepped every call site of `_record_search_emissions` in `engine_generation.py` (7 sites + the native default = 8 total); every one passes a path identifying its method. WARNING fires once per call when flag ON + sibling + queries recorded — bounded log volume. No KV-cache code touched (deferred to B-3d). No streaming surgery (deferred to B-3d). No `chat()` / `chat_stream()` API surface change (existing engine attribute is the surface).

**Sibling-boundary acceptance per sub-pass:** B-3a closes auto-stop on native non-GGUF only. The 7 WARNINGs on alt paths are honest UX — they tell the user "your splice flag is on but this path doesn't support it yet." B-3b will add the splice text helper at the same single tail-of-`_generate_text` site (one new wire-site, no fan-out). B-3c adds bounded recursion. B-3d adds streaming inline splice + KV rewind. Each later sub-pass narrows the WARNING set as it ships.

**Status of follow-ups:**
- ~~B-3a — engine flag + native auto-stop + sibling WARNINGs + GUI~~ **CLOSED Pass 156z9ag (this stamp).**
- B-3b — `_perform_search_splice` helper + first call site at `_generate_text` tail. Next sub-pass.
- B-3c — Bounded recursion for multi-round splice. After B-3b.
- B-3d — Streaming inline splice + KV rewind. Park last; only ships on user request.

**Lessons added to §4:** none new this pass — sibling-boundary discipline (Pass 156z7), defensive list-copy (general), opt-in flag pattern (Pass 156z9u), structural-gate-with-comment-stripping (Pass 156z9y) all applied as written.)

**Last updated:** May 4, 2026 (Pass 156z9af — B-3 RAG inline-splice park: concrete 4-sub-pass plan + scope decision recorded. No code change. Decision per §1 #14 (decision support): user picked "park properly with named sub-pass plan" over a minimum-complete one-pass slice (~400 LOC engine + ~250 LOC tests, native text-only path) and over a full-family slice (~800 + ~400, streaming + native, KV-cache surgery). Rationale: Rule #20 forbids half-built features; B-3 has four genuinely distinct moving parts (early-stop, retrieval, splice text, generation resume) and at least three sibling boundaries (streaming, vision, GGUF) where naive splicing breaks subtly. Rather than ship a narrow slice and accumulate Park-decay across the family, the four sub-passes below each ship a **complete-on-its-own** behavior with a real production consumer.

**B-3 sub-pass plan (each is a finished slice on its own, in order — later subs depend on earlier subs being live):**

1. **B-3a — `inline_search_splice_enabled` flag + `</search>` auto-stop in `_generate_text` native text-only path.** Gated default-OFF. When ON: append `</search>` to `stop_strings` inside `_generate_text` before calling `_generate_manual`. Generation halts cleanly on the closing tag; existing `_record_search_emissions(text, prompt=prompt)` already runs and populates `last_search_queries` — that's the consumer this slice ships against, no new helper needed. Sibling-boundary sweep: streaming/vision/speculative/medusa/lookahead/batch/GGUF all log a one-shot WARNING when the flag is ON and a `<search>` block lands in the recorded queries (sibling-distinguishing literal: existing `_record_search_emissions` warning text plus a new "splice path not supported on <method>" message). Tests: behavioural — flag ON + prompt designed to elicit `<search>` → assert `</search>` is in stop_strings forwarded into `_generate_manual` (sentinel-mock the call); flag OFF → assert NOT in stop_strings; falsification check by removing the append line. **Acceptance check:** chain `POST /api/chat → chat() → _generate_text → _generate_manual(stop_strings=[..., "</search>"])`. Closes when chain runs end-to-end with flag wired through `ChatRequest` + GUI checkbox.

2. **B-3b — `_perform_search_splice(decoded_text, prompt) -> str | None` helper + first call site.** Pure-text helper (no KV cache, no token surgery). Input: full decoded continuation containing exactly one trailing `<search>QUERY</search>` block (the auto-stop guarantees this shape). Output: `decoded_text + "<search_result>" + format_context(rag_index.query(QUERY)) + "</search_result>"`, or `None` if no `_rag_index` is built (caller falls through to "queries recorded, not spliced" path — same UX as B-3a alone). Wired at the single tail site of `_generate_text` non-GGUF path immediately AFTER `_record_search_emissions`. Behaviour change: when splice produces a non-None result, return that string instead of the bare continuation — the caller (`chat()` / `generate()`) sees a response that already contains the retrieved context as visible text. Resume-by-recursion is **deferred to B-3c** — this slice ships single-round splice only; multi-round (`<search>` → result → `<search>` again) is recorded as `last_search_queries` length > 1 and logged as "additional search rounds dropped, B-3c not yet shipped". Tests: stub `_rag_index` returning fixed chunks; splice produces text containing `<search_result>...</search_result>`; missing rag_index → returns None → bare continuation passes through unchanged. **Acceptance check:** chain `POST /api/chat → chat() → _generate_text → _record_search_emissions → _perform_search_splice → return spliced text`. Single-round only; multi-round is the next sub-pass.

3. **B-3c — Generation resume via re-prefill recursion (multi-round splice).** Replace the "drop additional rounds" warning from B-3b with a bounded recursion: after splice, re-call `_generate_text` with the spliced text as the new prompt and `_search_rounds_remaining=N-1` (default `max_search_rounds=3`). Recursion bottom-out: `_search_rounds_remaining == 0` → splice once more if pending, then disable splice for the final call (model gets one last chance to answer using the accumulated context). No KV cache surgery — pure prefill path. Honest doc: "each round costs a full prefill pass; budget is hardware-bounded." Sibling-boundary check: streaming/vision/spec/medusa/lookahead/batch/GGUF still rejected (their B-3a WARNING stays in place — none of them get splice in B-3c). Tests: stub model that emits `<search>q1</search>` then `<search>q2</search>` then plain text; assert `last_search_queries == [q1, q2]`, final text contains both `<search_result>` blocks, recursion depth honors budget. Adversarial test: model that emits `<search>` every round → assert recursion stops at budget and final text does NOT contain a third unspliced `<search>` block. **Acceptance check:** end-to-end chain runs with N=3 rounds, budget exhaustion is loud, no KV state leaks across recursion.

4. **B-3d — Streaming path inline splice (`stream_generate`).** Highest-risk sub-pass — needs token-by-token detection of `</search>` in the streamed output (decode-incrementally), pause yielding, retrieval, then resume yielding the spliced `<search_result>` block followed by continued generation. KV-cache option: rewind to pre-`<search>` position (already proven in `rewind_cache(pos)` per §4 KV-Cache rules), prefill the spliced result tokens, resume yielding. Test: streaming consumer (async iterator) sees `<search>q</search><search_result>...</search_result><continued response>` in order, no token duplication, no token loss across the pause. **Defer to last** — the three earlier sub-passes deliver the value (REST chat works); streaming is GUI-chat-only and chunked text already works because `chat()` calls `_generate_text` directly, not stream. Park concretely: this sub-pass only ships if a user explicitly asks for streaming RAG splice. Until then, the B-3a one-shot WARNING on streaming path is the honest UX.

**Off-switch / dead-infra protection:** `inline_search_splice_enabled` is a separate engine attribute from `inline_search_enabled` (the observability flag shipped Pass 156z9u). Splice OFF + observability ON = today's behaviour (queries recorded + WARNING). Splice ON + observability ON = sub-pass slices wire on top. Splice ON + observability OFF = ill-defined; gate at `_record_search_emissions`'s early-return (observability OFF kills splice too because the queries list is the input to splice). Tested in B-3a as part of the off-switch suite.

**Why not single-pass (Option A from the scope decision):** ships a finished feature for the native non-streaming path, but introduces 3 sibling-boundary WARNINGs that live on for at least 2 more passes before B-3d closes streaming. Each WARNING is a one-shot log line per request — sustainable but noisy in test suites for users who hit streaming paths. The 4-sub-pass split keeps each ship narrow enough to fully test before adding the next layer.

**Why not full-family (Option B):** KV-cache surgery in streaming is the genuine risk surface (attention-mask desync after rewind + re-prefill). Combining the two highest-risk parts (resume-recursion + KV-rewind) in one slice multiplies the regression surface. The sub-pass split isolates them.

---

**Last updated:** May 1, 2026 (Pass 156z9ae — Park-decay audit-close on the "GUI Generate Teacher Corpus button" follow-up.  No code change.  Walked the suggestion entry's bullets one-by-one against current code: subprocess spawn ([gui_forge_teacher.py L322](enigma_engine/gui/gui_forge_teacher.py#L322)), stdout streaming ([reader loop L370-401](enigma_engine/gui/gui_forge_teacher.py#L370)), progress regex `_PROGRESS_RE` + `_parse_teacher_progress` ([L138-160](enigma_engine/gui/gui_forge_teacher.py#L138)), Magpie/Prompts radio toggle ([gui_pages_forge.py L538-554](enigma_engine/gui/gui_pages_forge.py#L538)), endpoint/model/tag/prompts/Magpie-N/max-tokens widgets ([gui_pages_forge.py L520-636](enigma_engine/gui/gui_pages_forge.py#L520)), STOP cancel button + cancel-during-health-check ([L437-454](enigma_engine/gui/gui_forge_teacher.py#L437)), auto-fill `train_data_var` on rc==0 ([_teacher_finalize L412-425](enigma_engine/gui/gui_forge_teacher.py#L412)), `_kill_teacher_subprocess` shutdown hook wired from `_on_close` ([desktop.py L449](enigma_engine/gui/desktop.py#L449)).  Mixin mounted in `ForgeMixin` ([gui_forge.py L37](enigma_engine/gui/gui_forge.py#L37)).  Tests: 17 in `TestForgeTeacherSubprocess` ([tests/test_gui.py L6071](tests/test_gui.py#L6071)) + 42 in [tests/test_collect_distill_data.py](tests/test_collect_distill_data.py) all green (`pytest tests/test_gui.py::TestForgeTeacherSubprocess tests/test_collect_distill_data.py` → 59 passed in 0.89s).  Same Park-decay pattern as Pass 156z9t (B-1b Rust SPECIAL_TOKENS) and Pass 156z9z (A1 Q8_0 vectorization) — entry shipped via sibling-pattern rollout in an unrelated pass and the SUGGESTIONS row was never closed.  Author's-lens six-question lens on the closing pass: (1) write-it-this-way → yes, subprocess+reader+health-check is the canonical pattern in this codebase (`gui_cmd_page._cmd_proc`, `gui_mods` mod subprocesses are the two siblings).  (2) connected-to → ForgeMixin, `_log` line buffer, `_update_forge_progress`, `_reset_forge_progress`, `train_data_var`, `_on_close` shutdown chain.  (3) missing-connections → none — `_on_close` hook is the last unwired sibling and it IS wired.  (4) logic-eye → the docstring's "auto-fill picker on exit 0" matches the code's `if rc == 0 and out_path.exists(): var.set(str(out_path))`; no aspirational language anywhere.  (5) claim-vs-test → 17 GUI tests cover argv shape (Magpie + Prompts modes), int-coercion, progress regex (positive + negative cases), endpoint health-check happy-path + non-2xx + URLError, jsonl row counter, single-flight guard, finalize auto-fill, cancel-during-health-check, mixin signature.  (6) sibling-boundary sweep → the `subprocess.Popen` family in this GUI has 4 sites — desktop.py restart, gui_cmd_page CMD execution, gui_mods mod-launch, gui_forge_teacher (this); each has its own kill-on-close; teacher's is wired and tested.  No new tests this pass — feature is fully covered.)

**Last updated:** May 1, 2026 (Pass 156z9ad-doc — added §1 Rule #21 "No fluff in responses" and §2 DO NOT "Responses" block to [AA code maker.md](AA%20code%20maker.md). Match length to the question, drop preamble/recap/closer, one-word answers are complete when the question is small. No code changes.)

**Last updated:** May 1, 2026 (Pass 156z9ad — closed the "GUI Apply-button schema validation" follow-up opened by Pass 156z9ac, AND rebuilt the Rust BPE extension to retire the 2 stale-`.pyd` test failures that have shadowed every stamp since Pass 156z3.  GUI fix: the CONFIG-page Apply button parsed JSON and checked `isinstance(parsed, dict)` but never ran the structural-shape validator, so a schema like `{"type": "array"}` or a malformed `properties` block passed Apply with status "JSON schema applied (1 keys)" then crashed at send time inside `JsonSchemaConstraint.__init__` — caught by Pass 156z9ab Finding 5's `except ValueError` clause, but only at the chat send boundary, not at the closer Apply boundary the user just clicked.  Fix: call `validate_json_schema_shape(parsed)` between the existing `isinstance(parsed, dict)` check and the persist/attr-set step in [`_apply_json_schema`](enigma_engine/gui/gui_pages_config.py).  On `ValueError`, surface the validator's exact message via the status bar and don't-clobber attr/disk — same semantics as the existing parse-error and non-dict rejection branches.  Single source of truth held: validator helper is still the one place schema shape rules live; constraint constructor, both API endpoints, and now Apply all delegate to it.  Rust rebuild: `maturin build --release` + `pip install target/wheels/*.whl --force-reinstall --no-deps` from `rust_extensions/` resyncs the installed `.pyd` against the current `lib.rs`.)
**Tests:** 2745 passed, 2 skipped, **0 failures**. 2 new tests in `TestJsonSchemaConfig`:
- `test_apply_invalid_shape_rejected_with_validator_message` — behavioural: `{"type": "array"}` text submitted via Apply leaves the live attr at its prior value, leaves the persisted text at the prior value, and the status bar message contains `"object"` (the required type, named so the user knows what to fix).  Adversarial-falsifiable by reverting the new `try: validate_json_schema_shape(parsed)` block — confirmed by removing the call and watching the test fail with `assert {'type': 'array'} is {'keep': 'me'}`.
- `test_apply_calls_validate_json_schema_shape` — structural gate: strips comment-only lines from the source body before scanning, then asserts the call expression `validate_json_schema_shape(parsed)` AND an `except ValueError` clause are present in real code (not in a stale comment).  Comment-stripping per §4 "Label-tracking" rule — a stray comment mentioning the validator must not satisfy the gate.
**Rust:** rebuilt `enigma_bpe-0.1.0-cp312-cp312-win_amd64.whl` from `rust_extensions/`; `TestRustBPETrain::test_rust_train_produces_vocab` and `test_rust_special_tokens_match_python` both green.

**Pass 156z9ad (close "GUI Apply-button schema validation" — N-15 family follow-up):**

- **What was broken.** [`_apply_json_schema`](enigma_engine/gui/gui_pages_config.py) accepted any JSON dict.  Schemas with structurally-unsupported shapes (root `type` not `"object"`, non-dict `properties`, malformed property spec, unsupported leaf type) passed Apply with the misleading status "JSON schema applied (N keys)".  The shape error only surfaced at chat send time — caught by Pass 156z9ab Finding 5's `except ValueError` chain, but framed as a chat-system error, not as feedback on the Apply click that just happened.  UX gap: user clicks Apply, sees "applied", types a message, sees "JSON schema invalid" — no obvious connection to the Apply that "succeeded" five seconds earlier.

- **The fix.**  Three lines of imports + a 5-line try/except inserted between the existing `isinstance(parsed, dict)` check and the `self.json_schema = parsed` line.  `validate_json_schema_shape` was the helper extracted in Pass 156z9ac specifically so boundary callers like this could surface clean validator messages without duplicating the FSM-shape rules.  On `ValueError` the handler posts `f"[!] JSON schema invalid: {exc}"` to the status bar and `return`s — leaving `self.json_schema` and the persisted text at their last-successfully-applied state, exact same don't-clobber semantics as the existing parse-error and non-dict branches above it.

- **Why structural validation belongs at Apply, not at send.**  The send-time catch is still correct (defence in depth — bypassing API callers, profile-loaded schemas, future programmatic surfaces) but it fires AFTER a pretend-success at Apply.  Validating at Apply gives the user immediate, scoped feedback: the click that submitted the bad schema is the click that produces the rejection.  Same principle as Pass 156z9ab Finding 2 ("surface real failures at the closest boundary to the user") and Pass 156z9ac ("validate before lock acquire") — every accepted-then-failed feedback gap is a UX bug.

- **Sibling-boundary sweep (§1 #19 question 6).**  Five entry-points now accept `json_schema` from external input, all consistent on validation timing:
  1. `POST /api/chat` — validates before `_inference_lock.acquire` (Pass 156z9ac).
  2. `POST /api/chat/stream` — validates before `_inference_lock.acquire` (Pass 156z9ac).
  3. CONFIG-page Apply button — validates at click time (Pass 156z9ad, this slice).
  4. Chat send path (`_send_message`) — catches `ValueError` from constraint constructor as defence in depth (Pass 156z9ab Finding 5).
  5. Direct Python callers (`engine.chat(json_schema=...)`, `engine.generate(json_schema=...)`) — `JsonSchemaConstraint.__init__` calls the helper directly (Pass 156z9ac).

  All five route through the same `validate_json_schema_shape` helper.  The CMD-page chat path is intentionally OUT of family — Pass 156z9ab Finding 1 explicitly drops the kwarg before calling `engine.chat` so the CMD policy isn't overridden by a user schema.

- **Author's-lens self-audit (§1 #19 six questions, applied to this slice):**
  1. *Would I write it this way?* — Yes.  Three new lines.  Reuses existing helper.  No new abstractions.
  2. *Connected to what?* — `validate_json_schema_shape` (now its 5th caller).  Status-bar surface (consistent with the four existing rejection branches in the same method).  No other connections.
  3. *Could more connections be made?* — Schema-import-from-file UX would need the same validation; deferred until that surface exists.  Profile-scoped schema loading (`AIProfile.json_schema`?) would also benefit; not currently a feature.
  4. *Logic-eye?* — Status message says "JSON schema invalid: {exc}" where `{exc}` is the validator's exact message naming the offending field.  No aspirational language.  Don't-clobber semantics match the surrounding branches by design — code matches the docstring's "Behaviour matrix".
  5. *Claim-vs-test?* — Behavioural test reverts the fix and watches it fail (confirmed: `assert {'type': 'array'} is {'keep': 'me'}` fires).  Structural test strips comment-only lines before scanning — adversarially-falsifiable by reverting the call and leaving any commented-out reference (which would otherwise satisfy a naive substring check).
  6. *Sibling-boundary sweep?* — All five entry-points enumerated above; behaviour is consistent.

- **Open follow-ups updated:**
  - ~~A1 — Q8_0 vectorization~~ CLOSED Pass 156z9z.
  - ~~B-4 training data emitter~~ CLOSED Pass 156z9aa.
  - ~~N-15b GUI surface for `json_schema`~~ CLOSED Pass 156z9aa.
  - ~~F1-F5 audit findings on Pass 156z9aa~~ CLOSED Pass 156z9ab.
  - ~~API schema validation at boundary~~ CLOSED Pass 156z9ac.
  - ~~GUI Apply-button schema validation~~ CLOSED Pass 156z9ad.
  - ~~Rust `lib.rs` rebuild~~ CLOSED Pass 156z9ad — rebuilt + reinstalled, 2 tests green, suite at 2745/0/2.
  - **B-3 RAG splice** still open (multi-pass).  First sub-pass: token-level `<search>` early-stop hook reusing `_record_search_emissions` as the detection helper. Subsequent sub-passes wire (a) `rag.py` retrieval call, (b) `<search_result>...</search_result>` context surgery, (c) generation resume via KV cache rewind + re-prefill. Each sub-pass needs a real consumer in the same slice or it's dead infra (per Rule #20).
  - ~~GUI "Generate Teacher Corpus" button~~ CLOSED Pass 156z9ae (stale Park, feature shipped via sibling pass; 17 GUI tests + 42 CLI tests green).
  - **Schema meta-validation on Apply** (parked future-park; deferred until a user trips a non-structural schema-shape error — at that point a vendored `jsonschema`-style check would land on top of the structural validator, not replace it).
  - ~~**Rust `lib.rs` rebuild**~~ CLOSED Pass 156z9ad (see above).
  - **IQ-series GGUF dequant** still parked.

---

**Pass 156z9ac (close "API schema validation at boundary" — N-15 family follow-up):**

- **What was broken.** Three production HTTP paths (`/api/chat`, `/api/chat/stream`, and `AppState.chat` indirectly) accepted any `dict[str, Any]` for `json_schema` and forwarded it verbatim to `engine.chat`/`engine.stream_chat`. Inside the engine, `_generate_text` constructed `JsonSchemaConstraint(schema, tokenizer)` which raised `ValueError("json_schema['type'] must be 'object' (FSM is object-only), got 'array'")` (or similar). FastAPI's default exception handler mapped that to HTTP 500 with a JSON traceback. Symptoms: (a) user gets "Internal Server Error" instead of "your schema's `type` must be `object`", (b) the inference lock was acquired before the validator ran, so bad-schema requests blocked legitimate ones for the duration of the engine's setup phase, (c) FastAPI's default 500-handler logged the traceback at ERROR every time, polluting the log.

- **The fix.** Extract the validation block from [`JsonSchemaConstraint.__init__`](enigma_engine/core/json_schema_mask.py) into a module-level [`validate_json_schema_shape(schema, *, supported_types=...)`](enigma_engine/core/json_schema_mask.py) helper. The constraint constructor now calls the helper FIRST, then proceeds with the FSM setup it always did. The two FastAPI handlers ([`POST /api/chat`](enigma_engine/api/server.py) and [`POST /api/chat/stream`](enigma_engine/api/server.py)) call the helper at the top of the function body, BEFORE `_inference_lock.acquire(blocking=False)`. On `ValueError` the handler returns `JSONResponse(status_code=400, content={"error": f"Invalid json_schema: {exc}"})`. On success the handler proceeds unchanged.

- **Why extract instead of catch.** The "obvious" fix is `try: engine.chat(...) except ValueError as e: return 400`. Three reasons that's wrong: (1) **Lock first, validate later** — by the time `ValueError` propagates from inside `engine.chat`, the lock is held. Validating before lock acquisition means a flood of bad-schema requests can't DoS the engine. (2) **Conflation** — `engine.chat` could plausibly raise `ValueError` for OTHER reasons in future (bad temperature, etc.); the bare-`except ValueError` handler would mistakenly return 400 for those too. Calling the validator explicitly gates on the schema-specific error path. (3) **Single source of truth** — the validator helper is now the one place schema shape rules live; the constraint constructor delegates. A future relaxation (e.g. allowing `type: array` at root if the FSM grows array support) needs one edit, not two.

- **Why not pre-validate everything via `jsonschema` library?**  Considered and rejected. Vendoring or installing the `jsonschema` package for full draft-2020-12 validation is overkill — our FSM only handles a tiny subset (object-root, flat properties, six leaf types), and validating against a full meta-schema would accept inputs the FSM still can't constrain (e.g. `$ref`, `oneOf`, `required`). Our validator is **structural-only by design** — it gates exactly the inputs the FSM can constrain, no more. Documented this in the helper docstring so the next maintainer doesn't "improve" it into a meta-schema validator.

- **Sibling-boundary sweep (§1 #19 question 6).** Three sites accept `json_schema` from external input: `/api/chat` (non-streaming), `/api/chat/stream` (streaming), and the GUI CONFIG page (Pass 156z9aa/z9ab). The GUI already validates at the Apply button (`isinstance(parsed, dict)` check) AND catches `ValueError` at the send path (Pass 156z9ab). Both API endpoints now share the same validator helper. CMD page intentionally drops the field per Pass 156z9ab F1, no validation needed. The `AppState.chat` Python-API entry-point does NOT validate — it's an internal call between the FastAPI handler and the engine, and the handler validates before calling it; adding redundant validation at AppState would just be defensive code for a path the handler already gates. All four entry-points behave consistently: malformed schema fails loud at the boundary closest to the user.

- **Author's-lens self-audit (§1 #19 six questions, applied to this slice):**
  1. *Would I write it this way?* — Yes. Module-level function, single source of truth, no class methods that hide behind self, easy to import from anywhere.
  2. *Connected to what?* — `JsonSchemaConstraint.__init__` (constraint constructor still validates), `chat()` handler, `chat_stream()` handler. Three production callers, all in this pass.
  3. *Could more connections be made?* — A future GUI Apply button could call `validate_json_schema_shape` to surface the validator's exact message in the status bar (currently the GUI only checks `isinstance(parsed, dict)` then trusts the engine to fail loud at send time). Logged below as future-park.
  4. *Logic-eye?* — Validator docstring explicitly says "structural-only" and names what it does NOT check (`$ref`, format, required-array). No aspirational language. Constraint constructor's docstring still describes the FSM correctly.
  5. *Claim-vs-test?* — Each rejection branch (non-dict / non-object / non-dict-properties / non-dict-spec / unsupported-type) has a test. The adversarial "constraint constructor still validates" test catches the regression where someone deletes the `validate_json_schema_shape(schema, ...)` line from `__init__` thinking the API is the only caller. The "lock not acquired on 400" test gates the lock-ordering contract that matters for DoS resistance.
  6. *Sibling-boundary sweep?* — All three external entry-points (chat / chat_stream / GUI) audited above; behaviour is consistent.

- **Open follow-ups updated:**
  - ~~A1 — Q8_0 vectorization~~ CLOSED Pass 156z9z.
  - ~~B-4 training data emitter~~ CLOSED Pass 156z9aa.
  - ~~N-15b GUI surface for `json_schema`~~ CLOSED Pass 156z9aa.
  - ~~F1-F5 audit findings on Pass 156z9aa~~ CLOSED Pass 156z9ab.
  - ~~API schema validation at boundary~~ CLOSED Pass 156z9ac.
  - **B-3 RAG splice** still open (multi-pass). First sub-pass remains: token-level `<search>` early-stop hook reusing `_record_search_emissions` as the detection helper. Subsequent sub-passes wire (a) `rag.py` retrieval call, (b) `<search_result>...</search_result>` context surgery, (c) generation resume via KV cache rewind + re-prefill. **Each sub-pass needs a real consumer in the same slice or it's dead infra** (per Rule #19). The early-stop hook alone (without retrieval) would be a UX regression — model emits `<search>` and stops mid-response with no answer. So sub-passes (a) + early-stop must ship together, OR the early-stop must be gated behind a default-False flag with the consumer named in SUGGESTIONS as the immediate next slice.
  - **GUI Apply-button schema validation** (small, new).  Use the now-extracted `validate_json_schema_shape` helper on Apply so the status bar surfaces the exact validator message (e.g. "json_schema['type'] must be 'object'") instead of the generic "JSON schema applied (1 keys)" followed by a send-time ValueError. Same pattern as the F2 fix from Pass 156z9ab — surface real failures at the closest boundary to the user.
  - **GUI "Generate Teacher Corpus" button** still open (medium, GUI). Spawn `collect_distill_data.py` as a subprocess, stream progress, auto-fill the Distill mode's data-path field. Now needs a Magpie-mode toggle alongside the prompts-file picker.
  - **Schema meta-validation on Apply** (parked future-park; deferred until a user trips a non-structural schema-shape error).
  - **Rust `lib.rs` rebuild** still pending — uncommitted local edit means 2 Rust BPE tests fail against the stale installed `.pyd`. Run `maturin build --release` + `pip install target/wheels/*.whl --force-reinstall --no-deps` from `rust_extensions/` to resync.
  - **IQ-series GGUF dequant** still parked.

---

**Pass 156z9ab (audit-close on 156z9aa — 5 findings fixed):**

- **F5 — `ValueError` from `JsonSchemaConstraint` was uncaught.** [gui_logic_chat.py L282-307](enigma_engine/gui/gui_logic_chat.py#L282-L307). The `_gen()` closure in `_send_message` had `except TypeError:` (legacy fallback for engines that don't accept `json_schema=`) but no handler for the engine's own validation errors. Author's-lens question 4 (logic-eye on doc claims): the Pass 156z9aa stamp said "Engine validates the schema dict" — true, but the validator's failure mode (raise `ValueError`) was never caught at the GUI boundary. New handler: log WARNING with the exception message, schedule `_show_schema_err(m=schema_err)` on the main thread (sets chat-system message + status-bar left text with `[!]` prefix + truncated 80-char preview), then `return`. Default-arg trick (`m: str = schema_err`) freezes the message into the closure to avoid late-binding when multiple bad-schema attempts pile up. Variable named `schema_err` not `msg` because the outer `_send_message` scope already binds `msg = self.chat_input.get(...)` and Python's local-binding rule would have raised `UnboundLocalError` at the earlier `f"...{msg[:80]}"` site (caught by ruff F823 in the first iteration of this slice — falsification before ship).
- **F2 — silent disk-save failure mislabelled as success.** [gui_pages_config.py `_persist_json_schema_text`](enigma_engine/gui/gui_pages_config.py). Was: `except (OSError, ...): logger.debug(...); return None`. Caller couldn't tell success from disk failure. Now: returns `bool`, logs WARNING (`logger.warning("Could not save json_schema_text to %s: %s", settings_path, exc)`) — matches the loud-on-real-issue volume table (file-present-zero-yield → WARNING; missing-file → WARNING; success → silent). Both callers (`_apply_json_schema`, `_clear_json_schema`) consult the return and post split status messages. The in-memory state IS still updated even on disk failure (so the running session honors the user's intent), but the status bar names the persistence failure explicitly so the user knows to expect it gone after restart. Author's-lens question 5 (claim-vs-test): a structural test that just gated `"applied" in source` would have passed both before and after the fix; the new behavioural test `test_apply_surfaces_disk_failure_in_status_bar` patches `atomic_write_json` to raise and asserts the literal string `"disk save failed"` appears in the status bar — adversarially falsifiable by reverting either the bool return OR the caller's branching.
- **F3 — training poison in B-4 positives.** [collect_search_data.py `_POSITIVE_EXAMPLES`](collect_search_data.py). The contract for "positive" examples is: a query the model SHOULD emit `<search>` for AND that B-3's document RAG can plausibly answer. Tool-use queries (calendar lookup, git log, filesystem listing, test-coverage scrape) emit `<search>` correctly but RAG-over-static-corpus returns nothing — training on those rows would teach the model to emit a tag for queries the search backend can't fulfill, leading to silent drop or hallucinated retrieval results. Sibling principle from §4 GUI&CodeMatching: "Training data format must match inference format" applied to query category, not just text format. Replaced 5 rows + the "today's date" row with 5 document-retrievable positives that map to entries the planned RAG corpus (HF dataset cards, RFCs, PEPs, model cards) actually contains.
- **F1 — CMD-page sibling-boundary intentional drop.** [gui_cmd_page.py around the `engine.chat` call](enigma_engine/gui/gui_cmd_page.py). Sibling-boundary sweep (§1 #19 question 6) found the CMD page also calls `engine.chat` and could conceivably forward `self.json_schema` for symmetry. Design call: do NOT forward. CMD policy injects a system prompt asking for `[CMD]...[/CMD]` blocks; a user JSON schema would override that with a dict structure and silently disable command execution. Explicit `kwargs.pop("json_schema", None)` with a 9-line comment naming the rationale, gated by a structural test so a future refactor can't silently re-enable. Rule-#20 walk: feature is **finished** for the chat-page entry-point (POST through GUI button → `_send_message` → `engine.chat(json_schema=...)` → `_sample_token(json_constraint=...)`), and **explicitly killed** for the CMD-page entry-point (kwarg dropped before engine call, comment + test gate the kill). NOT parked — there is no future-work entry for "schema-constrained CMD output"; if that's ever wanted it's a new feature with its own design pass.
- **F4 — doc drift on own stamp.** Logic-eye on the Pass 156z9aa stamp: it said `_apply_json_schema` empty-text branch "delegates to `_clear_json_schema`" but the actual code inlines the same operations (sets `self.json_schema = None`, sets the textbox to empty, persists empty string). Functionally identical, but the stamp claim was wrong. Mentioned here so the next audit doesn't re-find it. Pattern: "doc claims more than code delivers" applies to stamps too — re-read the claim against the code AFTER shipping the slice, every time.

**Author's-lens self-audit on Pass 156z9ab itself (six questions, before merge):**
1. *If I wrote this from scratch today, would I do it this way?* — Yes. Each fix is the minimum change at the right boundary; no new abstractions.
2. *What is this connected to?* — `_send_message` (chat path), `_apply_json_schema` / `_clear_json_schema` (CONFIG handlers), CMD page chat path, B-4 corpus emitter. All four wire-sites touched in this pass.
3. *Could more connections be made?* — Stream chat path (`/api/chat/stream`) and FastAPI `/api/chat` already have schema validation per Pass 156z6/z3; their error surface is HTTP 400 with the validator message — appropriate for a server. GUI is the only one needing the friendly-message translation.
4. *Logic-eye — does code deliver what doc/comment claims?* — F4 was caught BY this question on the prior stamp. New stamp re-checked: each bullet above maps 1:1 to a real code change. The Rule-#20 walk for F1 explicitly states "killed" with the test gating the kill — not aspirational.
5. *Claim-vs-test — would the test catch a regression?* — F5 test deletes the `except ValueError` line → fails. F2 test reverts `bool` return → fails (status string check). F1 test removes the `kwargs.pop` line → fails. Each test was written to be falsifiable by reverting the specific fix.
6. *Sibling-boundary sweep — every site in the contract family?* — Chat send path: 1 site (`gui_logic_chat._send_message`) catches `ValueError`. CMD send path: 1 site (`gui_cmd_page` inner `_generate`) drops the kwarg. Streaming/API paths already handled per prior passes. Background trainer / batch generator do NOT call `engine.chat(json_schema=...)` — out of family.



**Pass 156z9aa (N-15b GUI for `json_schema` + B-4 synthetic search corpus):**

- **N-15b — GUI surface for constrained decoding.** The engine + API server have accepted `json_schema=...` since Pass 156z3 (`EnigmaEngine.chat`) and Pass 156z6 (`ChatRequest` + `/api/chat/stream`); the GUI was the last unwired layer. Four touch-points:
  - **In-memory mirror** at [desktop.py L107](enigma_engine/gui/desktop.py#L107): `self.json_schema: dict | None = None`. Sits next to `self.inline_search_enabled` because both are engine-attribute mirrors of persisted GUI flags (sibling-pattern).
  - **Boot-load helper** `_read_gui_json_schema_setting` placed after `_read_gui_str_setting` in desktop.py — reads `gui_settings.json["json_schema_text"]`, returns parsed dict on success, returns None on missing/empty/whitespace (silent — fresh-install / cleared-by-user normal path) AND on parse-error / non-dict (loud WARNING — corrupted persisted state, follows the §4 loud-on-real-issue volume table). The helper IS the boot-load wire-site at desktop.py `__init__`; called there immediately after the inline_search boot-load.
  - **Widget block** in `_build_page_config` (gui_pages_config.py): SelectableLabel "JSON SCHEMA (constrained decoding)" + description label + `CTkTextbox(height=100)` pre-filled from `_cached_settings.get("json_schema_text", "")` + tooltip naming the dict-required engine contract + Apply/Clear button row. Three handlers: `_apply_json_schema` (reads textbox via `.get("1.0", "end").strip()`, empty→delegates to `_clear_json_schema`, parses via `json.loads`, non-dict→status error with NO clobber of `self.json_schema` and NO persist, dict→sets `self.json_schema` AND persists raw text via `atomic_write_json`), `_clear_json_schema` (clears textbox + nulls live attr + persists `""`), `_persist_json_schema_text(text)` (single canonical write site for `gui_settings.json["json_schema_text"]`).
  - **Chat send-path forwarding** in [gui_logic_chat.py](enigma_engine/gui/gui_logic_chat.py) `_send_message`: between the reasoning kwarg and the `engine.chat(**kwargs)` call, `_gui_json_schema = getattr(self, "json_schema", None); if _gui_json_schema is not None: kwargs["json_schema"] = _gui_json_schema`. The `getattr` (not direct attribute read) handles legacy GUI sessions that pre-date the field — same defensive pattern other engine-flag forwards use in this method.

- **Why textbox + Apply/Clear, NOT a toggle?** `json_schema` is structured data (an arbitrary JSON Schema dict), not a boolean. The user has to author or paste a schema; the Apply step lets them validate before committing (parse error stays in the textbox, status_bar shows the error, live attr unchanged); Clear is the explicit off-switch. A checkbox-driven design would have nowhere for the dict to come from and would force a coupled file picker.

- **Why is `json_schema` NOT folded into `config_overrides`?** `config_overrides` carries numeric chat kwargs (temperature, top_p, top_k, max_tokens, repetition_penalty) that flow through every `**kwargs`-receiving generation path. `json_schema` is a dict that the engine consumes in exactly one place (`_sample_token(json_constraint=...)` per Pass 156z3) and only on the chat path. Mixing them would force a kwarg-pass through every generation entry-point that doesn't accept structured constraints today (`stream_chat`, `_generate_text`, `_generate_with_vision`, `batch_generate`) — six new sites for a feature that lives on one. Same architectural reasoning as the Pass 156z9x note for `inline_search_enabled`.

- **9 new tests in `TestJsonSchemaConfig` ([tests/test_gui.py](tests/test_gui.py)):**
  1. `test_apply_valid_dict_persists_and_sets_attr` — locks the three success post-conditions (live attr + disk + status_bar success).
  2. `test_apply_invalid_json_does_not_clobber_attr` — adversarial: pre-set live attr survives a parse-error Apply call AND no `json_schema_text` key persisted. Catches the regression where a future "be helpful" branch overwrites the attr with `None` on parse failure (which would silently disable constrained decoding instead of preserving the user's prior valid schema).
  3. `test_apply_non_dict_rejected` — JSON list (or any non-dict) rejected with same don't-clobber semantics. Engine contract requires dict.
  4. `test_apply_empty_text_clears_attr` — empty/whitespace textbox + Apply IS a valid intentional clear (distinguished from the parse-error case which leaves state unchanged).
  5. `test_clear_button_resets_textbox_and_attr` — explicit Clear path empties textbox + persisted text + live attr atomically.
  6. `test_boot_load_parses_persisted_dict` — helper returns parsed dict for valid persisted JSON. The wire-site that carries the user's saved schema across restarts.
  7. `test_boot_load_invalid_json_returns_none_with_warning` — corrupted persisted text → None + WARNING log naming `json_schema_text` AND `invalid`. Loud-on-real-issue gate; without it a hand-corrupted `gui_settings.json` silently boots into no-constraint mode.
  8. `test_boot_load_empty_or_missing_returns_none_silent` — missing key + empty string + whitespace-only string all return None with NO WARNING log records. Fresh-install / cleared normal path stays quiet.
  9. `test_chat_send_path_forwards_json_schema_kwarg` — structural test using two **independent** regexes against `inspect.getsource(LogicChatMixin._send_message)`: one matches `getattr(self, "json_schema"`, the other matches `kwargs["json_schema"] =`. Either alone is shared with the surrounding comment block (per the §4 substring-presence rule); the pair only matches the live wire-site.

- **B-4 — synthetic `<search>` training corpus emitter** in [collect_search_data.py](collect_search_data.py). Stdlib only. Two embedded curated lists:
  - `_POSITIVE_EXAMPLES` — 30 factual questions paired with `<search>compact-query</search>` completions across 5 sub-themes: time-sensitive (election, stock price, weather, headlines), specific verifiable facts (population, GDP, height, ISBN), technical/API/version queries (default port, HTTP status, CUDA version), lookups by name (papers, repos, abstracts), and local/personal context the model categorically cannot know (calendar, git log, file mtimes).
  - `_NEGATIVE_EXAMPLES` — 29 questions paired with direct answers, NO `<search>` tag anywhere, across arithmetic (model can compute), definitions (stable knowledge), reasoning explanations, common knowledge (planet count, freezing point), conversational prompts (greetings, haiku, naming), and programming questions answerable from training.

- **Why both classes?** Without negatives the model overfits to "always emit `<search>` after a question mark" — strictly worse than the unconditioned baseline. The negatives carry the equally-important signal: "you already know this, answer directly." Class balance ~50/50 for the seed corpus; production training would tune this empirically.

- **`_validate_examples` is mandatory at build time, not optional.** Class-purity invariant: every positive MUST contain `<search>...</search>`; every negative MUST NOT. A corruption that drops the tag from a positive silently teaches the model to skip search on a question that needs lookup; a corruption that adds the tag to a negative teaches the model to emit search on common knowledge. Validator raises `RuntimeError` with the offending prompt names so a bad template lands as a build-time failure, not a silent corpus poisoning.

- **Output format reuses `_write_jsonl` + `_write_combined_text` from [collect_finetuning_data.py](collect_finetuning_data.py)** — the canonical helpers that emit JSONL pairs AND the `User: <p>\n\nAssistant: <c>` blank-line-separated text format the existing FORGE SFT trainer consumes per Pass 156i8 D-11. Dual-emit means the emitter touches zero training-side code; the trainer's plain-text reader picks up `synthetic_search_seed.txt` like any other corpus.

- **CLI surface:** `python collect_search_data.py --tag seed` (default both classes) → `data/finetune/synthetic_search_seed.{jsonl,txt}`. Ablation flags `--positive-only` / `--negative-only` for class-balance experiments only — production should always use both. `--tag` propagates verbatim to filenames so multiple corpora co-exist (e.g. `seed`, `v2_alpha`, `expanded_facts`). `--output-dir` defaults to `data/finetune/` to match FORGE's auto-discovery.

- **11 new tests in [tests/test_collect_search_data.py](tests/test_collect_search_data.py):**
  - 3 validator tests: shipped corpus passes; positive-without-tag triggers `POSITIVE missing` RuntimeError; negative-with-tag triggers `NEGATIVE contains` RuntimeError. Both adversarial cases use `monkeypatch.setattr` on the embedded lists so the test exercises the validator's branch logic, not just its happy path.
  - 4 `build_corpus` tests: default mixes both classes with non-empty per-class counts and well-shaped dicts; `positive_only=True` returns exactly `_POSITIVE_EXAMPLES` length and 100% search-tagged; `negative_only=True` returns exactly `_NEGATIVE_EXAMPLES` length and 0% search-tagged; both flags at once raises ValueError with `mutually exclusive` message.
  - 4 CLI tests: end-to-end main writes both files; `--tag` propagates to filenames; `--positive-only` flag survives the CLI→build_corpus path (regression guard against a future refactor that drops the gate); both flags at once returns rc=2 and writes nothing.

- **Author's-lens self-audit (§1 #19 six-question lens, applied to BOTH slices):**
  1. *Would I write it this way?* N-15b: yes — textbox + Apply/Clear is the canonical CTk pattern for free-form structured input (sibling: there's nothing closer in the GUI today, so the choice is self-cohering rather than pattern-matching). B-4: yes — embedded curated lists keep the seed corpus reviewable in one file, no external data dependency, no network. Growth path is appending.
  2. *Connected to what?* N-15b reads/writes `gui_settings.json` via `atomic_write_json` (canonical pattern, 8+ other persisted settings); forwards into `engine.chat` via the existing `**kwargs` plumbing (no new generation entry-point); the engine consumer (`json_constraint`) was wired Pass 156z3 and the API consumer (`ChatRequest`) was wired Pass 156z6 — this pass closes the GUI gap. B-4 reuses `_write_jsonl` + `_write_combined_text` from the main collector (single source of truth for output shape); writes into `data/finetune/` which the existing FORGE pipeline auto-discovers.
  3. *Connections that should exist but don't?* N-15b: a future schema-validation step on Apply (validate the dict against draft-2020-12 JSON Schema meta-schema, not just `isinstance(parsed, dict)`) — out of scope for the GUI-wiring slice; user can paste a malformed schema today and it fails inside `_sample_token` instead of inside Apply. Logged as future-park, not blocked. B-4: future automation (FORGE adaptive-mode self-generates more positives via the model's own teacher mode) — also out of scope; the manual append path is honest.
  4. *Logic-eye?* N-15b: the apply handler's three branches (empty / parse-error / non-dict / dict) are all explicit; status_bar messages name the failure mode for parse-error and non-dict so the user can fix their input. B-4: validator docstring names BOTH failure modes (positive-without-tag AND negative-with-tag) and the test pair gates each one falsified-against-corruption. No aspirational language anywhere.
  5. *Claim-vs-test?* N-15b: 9 tests exercise apply success + 3 don't-clobber rejections (parse-error, non-dict, behaviour-vs-clear distinction) + Clear + 3 boot-load branches (parsed / invalid+WARNING / silent-empty) + send-path wiring with two-regex adversarial gate. B-4: 11 tests exercise validator pass + 2 validator failure-mode falsifications + 4 build_corpus shape + 4 CLI rc/files. Both slices have at least one falsification-against-corruption per claimed contract.
  6. *Sibling-boundary sweep?* N-15b: walked the chat-receiving family — `EnigmaEngine.chat` accepts `json_schema` per Pass 156z3, `ChatRequest` accepts it per Pass 156z6, GGUF backend raises `NotImplementedError` per Pass 156z3 (not a silent-drop site, the rejection is loud). The GUI is the LAST chat-input layer that needed the field; no sibling left unwired. The streaming counterpart (`stream_chat`) accepts `json_schema` since Pass 156z6, and `kwargs.update(self.config_overrides)` doesn't touch it (config_overrides carries numerics only) — no collision. B-4: walked the SFT-data-reading family — FORGE's SFT trainer reads `data/finetune/*.txt` files matching the `User:/Assistant:` block format produced by `_write_combined_text` (verified at Pass 156i8); JSONL siblings (`distill_*.jsonl`, `instruct_*.jsonl`) are alternate consumers; `synthetic_search_*.{jsonl,txt}` joins the existing dual-emit pattern verbatim, no naming collision, no shape divergence.

- **Open follow-ups updated:**
  - ~~A1 — Q8_0 vectorization~~ CLOSED Pass 156z9z.
  - ~~B-4 training data emitter~~ CLOSED Pass 156z9aa.
  - ~~N-15b GUI surface for `json_schema`~~ CLOSED Pass 156z9aa.
  - **B-3 RAG splice** (still open, multi-pass effort). First concrete sub-pass identified: **token-level `<search>` early-stop hook in the generation loop**, reusing the existing `_record_search_emissions` infrastructure as the detection helper. After early-stop fires, the next sub-passes wire (a) `rag.py` retrieval call with the captured query string, (b) `<search_result>...</search_result>` context surgery into the conversation buffer, (c) generation-resume via re-prefill of only the new tokens (KV cache rewind to pre-search-emit position, append the result block, prefill, continue sampling). Each of (a)/(b)/(c) is its own slice with its own tests; do not attempt them in one pass. With B-4 now shipped, the model has training signal for *when* to emit `<search>` — B-3 closes the loop on *what happens after*.
  - **Schema meta-validation on Apply** (small future-park). Current N-15b validates `isinstance(parsed, dict)` and trusts the engine layer to fail if the dict isn't a valid JSON Schema. A draft-2020-12 meta-schema validator inside `_apply_json_schema` would surface schema-shape errors at Apply-time instead of mid-generation. Stdlib-only is non-trivial here (would need a vendored mini-validator); deferred until a user actually trips a schema-shape error in production.
  - **Rust `lib.rs` rebuild** still pending — uncommitted local edit to [rust_extensions/src/lib.rs](rust_extensions/src/lib.rs) from a prior session means 2 Rust BPE tests fail against the stale installed `.pyd`. Run `maturin build --release` + `pip install target/wheels/*.whl --force-reinstall --no-deps` from `rust_extensions/` to resync. Not blocking GUI/data-collector work but blocks the Rust BPE test slice.
  - **IQ-series GGUF dequant** still parked.

---

**Pass 156z9z (A1 Q8_0 vectorization stale-Park audit-close):**

- **No code change this pass.** Audit walked through the parked entry's three acceptance criteria one-by-one against current `gguf_dequant.py` + `tests/test_core.py`:
  1. *"Rewrite body using the pattern: `raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, 34); d = np.frombuffer(raw[:, :2].copy().tobytes(), dtype=np.float16).astype(np.float32); values = raw[:, 2:34].view(np.int8); out = values.astype(np.float32) * d[:, None]`"* — code on disk matches line-for-line (variable name `qs` vs proposed `values`; same `.copy().tobytes()` fp16 read pattern; same `.view(np.int8)` for signed reinterpret; same `[:, None]` broadcast).
  2. *"add `test_dequantize_q8_0_zero_blocks` (parity with the four siblings' degeneracy gates)"* — present at L2540, asserts both `result.dtype == torch.float32` AND `torch.all(result == 0)`. Sibling-parity gate exactly as requested.
  3. *"confirm `test_dequantize_q8_0_values` still green"* — present at L2526; ran `pytest -k "test_dequantize_q8_0"` → 3 passed (values + zero_blocks + signed_values). Bonus coverage from `test_dequantize_q8_0_signed_values` (added in some prior pass — adversarial gate against a regression that drops the `.view(np.int8)` and reads qs as uint8, which would flip 0xFF from -1.0 to +255.0).

- **Author's-lens self-audit (§1 #19 six-question lens, applied to the closing-doc pass):**
  1. *Would I write it this way?* Yes — when a Park entry's recipe has been quietly executed by a sibling-pattern rollout, the right move is an audit-close stamp citing the verifying line numbers + test names so the next reader can re-verify in 30 seconds.
  2. *Connected to what?* `dequantize_q8_0` is the 5th of 5 in the linear-quant family (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0); Q8_0 dispatcher branch already routes via the same `parse_gguf_tensors` pattern as siblings. No wiring drift.
  3. *Connections that should exist but don't?* None. Q8_0 has no qh, no nibble splitting, no signed-scale stitch — the simplest member of the family. No helper-unification opportunity.
  4. *Logic-eye?* The audit-close stamp does not over-promise: it claims "current body matches the proposed recipe" and cites line numbers; it claims "3 sibling-parity tests green" and ran them. No aspirational language.
  5. *Claim-vs-test?* The closing claim "feature is closed" is gated by 3 behavioural tests + the dispatcher routing test from Pass 156z9p that exercises the Q8_0 dispatch branch with a sentinel recorder. A regression that breaks Q8_0 vectorization fails at least one of the four.
  6. *Sibling-boundary sweep?* k-quant family (Q2_K..Q6_K) parity check from Pass 156z9s confirms each sibling has zero-block + value-routing + bit-stitch tests. Linear-quant family (Q4_0..Q8_0) parity now verified end-to-end here. Both families closed for arithmetic dequantization. Only IQ-series (lookup-grid quants) remain parked.

- **Lesson reinforced for §4 Learned Principles (Park hygiene):** *Park entries that name specific code state ("still uses a per-block Python for loop") decay silently when the code moves forward through unrelated passes.* Sibling-pattern rollouts (e.g. the Pass 156z9n/o k-quant family adding the `n_blocks * bytes_per_block` reshape pattern + dtype-explicit zero return) often quietly bring outdated siblings forward without explicit Park-entry attention. Audit pattern: at start of each multi-pass session, grep parked entries' specific code-state claims (`per-block for loop`, `13-entry array with the wrong order`, `Q4_K_M would hit Unknown tensor type`) against current code; mark stale ones CLOSED with the verifying line numbers + test names + the pass that probably closed them. Two consecutive Park-decay closes in this session (156z9t closed B-1b, 156z9z closes A1) — the principle is now load-bearing.

- **Open follow-ups updated:**
  - **A1 — Q8_0 vectorization** CLOSED Pass 156z9z (stale Park, code already vectorized + 3 tests green).
  - **B-3 RAG splice** still open (multi-pass effort: token-level early-stop + KV-cache splice + generation-resume).
  - **B-4 training data emitter** still open (synthetic `<search>` corpus — capability gate for B-3).
  - **N-15b GUI surface for `json_schema`** still open (server.py + `ChatRequest` already accept the field per Pass 156z6; only the GUI textarea/checkbox is missing — small slice, not blocked on anything).
  - **IQ-series GGUF dequant** (`IQ2_XXS`/`IQ2_XS`/`IQ2_S`/`IQ3_XXS`/`IQ3_S`/`IQ4_NL`/`IQ4_XS`/`IQ1_S`/`IQ1_M`) still parked — non-linear lookup-grid quants, each needs a 256-entry codebook ported from `ggml-quants.c`. No on-disk model the user has needs them today.

---

**Pass 156z9y (post-156z9x audit hardening — Finding A):**

- **Test 1 (behavioural) — `test_boot_load_reads_persisted_off_value`** in [tests/test_gui.py](tests/test_gui.py): monkeypatches `desktop.DATA_DIR` to a tmp_path, writes `{"inline_search_enabled": false}`, calls `EnigmaGUI._read_gui_bool_setting("inline_search_enabled", True)` on a stub instance via `object.__new__`, asserts the helper returns False (NOT the library default True). Sibling-parity branch: missing key falls back to default True so fresh installs keep observability on. Adversarial: a regression in `_read_gui_bool_setting` that swallows the on-disk value (e.g. JSON parse error silently returning default) would flip the False→True assertion immediately.
- **Test 2 (structural, strengthened) — `test_boot_load_wire_site_present_in_init`**: regex `_read_gui_bool_setting\(\s*"inline_search_enabled"` against `inspect.getsource(EnigmaGUI.__init__)`. The `\s*` tolerates the line-continuation whitespace black/ruff produce when the call wraps across two lines (the actual source has a newline + 12 spaces between `(` and `"`). Targets the call expression paired with the literal key, NOT the bare token — the bare token also appears at the L107 in-memory default and would mask the regression.

- **Falsification check performed.** Edited desktop.py in-place to delete the L172-173 boot-load assignment (kept the comment block + replaced the two code lines with a `FALSIFICATION_TEST_BROKEN` marker comment), ran pytest → both tests **failed correctly** with the AssertionError naming the missing wire-site. Restored the line, ran pytest → all 5 tests in `TestInlineSearchEnabledConfig` pass. The structural test is now an honest regression gate, not a presence-only check.

- **Author's-lens self-audit caught a real bug in code shipped 2 minutes earlier.** The original Pass 156z9x version of this test asserted `"inline_search_enabled" in src` AND `"_read_gui_bool_setting" in src` — both substrings present in `__init__`, but the FIRST appears at the L107 in-memory default line and the SECOND appears at the `auto_load_chat_model` call (and 3 other unrelated boot-load calls). Either substring satisfies the assertion regardless of whether the boot-load wire-site for `inline_search_enabled` exists. The test was structural-presence, not wire-site-correctness. Same shape as Pass 156z9s Q4_K layout-routing failure that triggered the §4 *"Adversarial test claims must be falsified against the broken implementation, not just verified against the correct one"* lesson.

- **Lesson reinforced for §4 Learned Principles:** *Substring-presence assertions on `inspect.getsource` are vacuous when the substring appears at multiple sites in the body — always pair the new wire-site with a literal that ONLY appears there.* The regex `_read_gui_bool_setting\(\s*"inline_search_enabled"` is the minimum substring that uniquely identifies the boot-load call expression. A "function-name-plus-key" pair is the sibling-distinguishing token; either alone is shared with siblings. Generalises: when adding a structural test for a new wire-site that joins an existing pattern (4 boot-load calls in `__init__`, multiple `_record_search_emissions` call-sites in generation), assert the FULL call expression paired with the new argument, not just the function name or the argument alone.

- **Self-audit trail:** the original 156z9x test was the THIRD adversarial-test-falsification miss in this contract family (after Pass 156z9s Q4_K and Pass 156z9w batch_generate off-switch). Each one was caught by the Author's-lens pass on the diff and fixed in the next slice. The discipline is now: **after writing any structural test on `inspect.getsource`, mentally apply the regression you claim to catch and check whether the broken-impl source still satisfies the assertion**. If yes → the test is presence-only and needs strengthening before it ships.

---

**Pass 156z9x (Stage B-2c GUI surface — `inline_search_enabled` checkbox):**

- **Boot-time load** in [desktop.py L162-167](enigma_engine/gui/desktop.py): `self.inline_search_enabled = self._read_gui_bool_setting("inline_search_enabled", True)`. Default True preserves Pass 156z9d's always-on observability for fresh installs. Initial in-memory default lives at L101 next to `self.reasoning_enabled` (mirror of the canonical boolean engine-state pattern).
- **Checkbox widget** in [gui_pages_config.py](enigma_engine/gui/gui_pages_config.py) `_build_page_config`: new HUDFrame card titled "GENERATION BEHAVIOR" inserted AFTER the numeric `CONFIG_LIMITS` loop and BEFORE the "Display names" section. `BooleanVar` initialised from `_cached_settings.get("inline_search_enabled", True)`; `CTkCheckBox` styled to match `_show_emo_cb` and `_learn_while_chatting_cb`; tooltip names exactly what the flag does (scan + WARNING + `last_search_queries`) so the user knows the consequence of disabling.
- **Toggle handler `_toggle_inline_search_enabled`** in [gui_pages_config.py](enigma_engine/gui/gui_pages_config.py): atomic-writes `inline_search_enabled` to `gui_settings.json`, updates `self.inline_search_enabled` (in-memory mirror used by `_on_model_loaded` for subsequent loads), and applies to `self.engine.inline_search_enabled` if an engine is currently loaded — the toggle takes effect on the NEXT generation call (the helper reads the attribute via `getattr` on every call). Persistence path mirrors `_toggle_show_emotional_state`; engine-apply step is the new behaviour the architecture required.
- **Model-load propagation** in [gui_logic.py `_on_model_loaded`](enigma_engine/gui/gui_logic.py): after `_restore_lora_adapter_for_base(path)`, set `self.engine.inline_search_enabled = bool(getattr(self, "inline_search_enabled", True))`. Without this, every fresh engine ships with its library default True regardless of the user's persisted choice — the same signal-without-consumer anti-pattern the §4 "Boundary signal without a consumer = dead infrastructure" rule warns about, applied to a GUI persistence layer.
- **3 new tests** in [tests/test_gui.py](tests/test_gui.py) `TestInlineSearchEnabledConfig`:
  - `test_toggle_persists_and_applies_to_live_engine` — toggle OFF with `FakeEngine` attached → asserts `gui_settings.json["inline_search_enabled"] is False` AND `obj.inline_search_enabled is False` AND `obj.engine.inline_search_enabled is False`. Behavioural test that catches a regression where the persistence layer drops the engine-apply step.
  - `test_toggle_with_no_engine_still_persists` — toggle OFF with `obj.engine = None` → asserts persistence happened and the no-engine apply path is a silent no-op (does not crash). Locks the pre-load toggle contract.
  - `test_on_model_loaded_applies_persisted_flag` — structural test that `_on_model_loaded` source contains `engine.inline_search_enabled`. Behavioural coverage at the engine layer is provided by the off-switch tests in `test_chat.py` shipped Pass 156z9u/w.

- **Architecture note — why NOT inside `config_overrides`?** `config_overrides` feeds numeric chat kwargs (temperature, top_p, top_k, max_tokens, repetition_penalty) which the engine consumes per-call via `**kwargs`. `inline_search_enabled` is an engine *attribute* gated inside `_record_search_emissions` via `getattr`, not a per-call kwarg. Mixing them would force a kwarg-pass through every generation entry-point that doesn't currently accept it (`chat`, `stream_chat`, `_generate_text`, `_generate_with_vision`, `batch_generate`) — six new sites for a flag that lives on the engine for the entire session. Storing as a top-level `gui_settings.json` key + applying via attribute assignment keeps the surface flat.

- **Author's-lens self-audit (§1 #19 six-question lens):**
  1. *Would I write it this way?* Yes — mirrors `_show_emo_cb` and `_learn_while_chatting_cb` (the two closest siblings: BooleanVar + checkbox + atomic-write + status_bar message). The only divergence is the engine-apply step, which is required because the flag must reach the engine, not just disk.
  2. *Connected to what?* Reads from same `gui_settings.json` as 8+ other persisted settings; writes via `atomic_write_json` (canonical pattern); applies to `self.engine` (live) and propagates to future engines via `_on_model_loaded`.
  3. *Connections that should exist but don't?* Future possible: a CLI surface (`run.py --no-inline-search`) for headless runs. Not blocked on anything; out of scope for a GUI-only park item.
  4. *Logic-eye?* Docstring on `_toggle_inline_search_enabled` names every consequence (persist + in-memory mirror + live-engine apply). The toggle code performs all three. The model-load apply line in `_on_model_loaded` carries a comment explaining WHY (default-true library would silently revert user's off-toggle on every load).
  5. *Claim-vs-test?* `test_toggle_persists_and_applies_to_live_engine` asserts THREE post-conditions in one test (disk + in-memory mirror + live engine). A regression that only writes to disk OR only updates the mirror OR only applies to engine fails loudly. The structural `_on_model_loaded` test is honest about its limitation (gates literal token presence) and points to behavioural coverage at the engine layer.
  6. *Sibling-boundary sweep?* The flag is consumed at exactly one site (`_record_search_emissions`) and it's the same helper that all 3 generation paths funnel through (verified Pass 156z9u sweep). The GUI persistence layer has 4 sites (boot-load, widget builder, toggle handler, model-load apply) — all four were touched in this pass. No other entry-points need to read the flag.

- **Rule-#20 honesty:** the "Future-park (small)" entry from Pass 156z9u is now CLOSED. Walked the call chain end-to-end: GUI checkbox toggle → `gui_settings.json` write → desktop boot read → in-memory mirror → `_on_model_loaded` apply → `engine.inline_search_enabled` → `_record_search_emissions` early-return → no scan, no WARNING, empty `last_search_queries`. Production path verified.

- **Open follow-ups (Stage B series, updated):**
  - ~~B-1b — Rust SPECIAL_TOKENS drift~~ CLOSED Pass 156z9t.
  - ~~B-2 — generation-loop hook~~ CLOSED Pass 156z9d/e/u (detection + WARNING + 3-site wiring + off-switch all shipped).
  - ~~B-2c GUI surface~~ CLOSED Pass 156z9x.
  - **B-3 — RAG splice.** When the model emits `<search>query</search>`, call into [rag.py](enigma_engine/core/rag.py) for top-k results, format as `<search_result>...</search_result>`, append to context, resume generation. Substantial: requires token-level early-stop in the generation loop (different from the current text-level post-gen scan), prompt-context surgery to splice the `<search_result>` block in without breaking KV cache, and a generation-resume entry-point that re-prefills only the new tokens. This is a multi-pass effort, not a single slice.
  - **B-4 — training data emitter.** Synthetic dataset where prompts contain hard factual questions and gold completions show the `<search>...</search>` pattern. Without B-4, even with B-3 wired, the model never spontaneously emits `<search>` — this is the actual capability gate.

---

**Pass 156z9w (post-audit hardening on 156z9r..156z9u — Findings 1, 2, 3):**

- **Finding 1 — Q4_K `test_dequantize_q4_K_layout_routing` strengthened** ([tests/test_core.py](tests/test_core.py)). Original Pass 156z9s version: only `scales[0]=0x02` (sub-block 0's sc=2), `qs[0]=0x05`, `qs[64]=0x07`; asserted `out[0]=10` and `out[32..256] == 0`. Failure mode: a nibble-swap that misroutes sub-block 1's qs to read low-nib instead of high-nib still produces `out[32]=0` because sub-block 1's sc was zero in the test data — the fault was structurally invisible. Rewrite: `scales[0]=0x02` (sb0 sc=2), `scales[1]=0x03` (sb1 sc=3), `scales[8]=0x05` (sb4 sc=5 via low nib + scales[0]>>6=0 high stitch); `qs[0]=0x91` (sb0 elem 0 = 1, sb1 elem 0 = 9 — distinct values catch nibble-swap); `qs[64]=0x07` (sb4 elem 0 = 7). Asserts `out[0]=2.0`, `out[32]=27.0`, `out[128]=35.0` plus independence checks at `out[64,96,160,192,224]=0`. A nibble-swap on the (0,1) pair flips both out[0] (2.0 → 18.0) and out[32] (27.0 → 3.0); a stride bug on the j=4 path changes out[128].
- **Finding 2 — batch_generate off-switch integration test added** ([tests/test_chat.py](tests/test_chat.py) `TestStageB2bBatchPerPromptAttribution.test_batch_generate_off_switch_clears_per_prompt`). The 4 B-2c off-switch tests shipped in Pass 156z9u all called `_record_search_emissions` directly. A future regression that inlined the scan inside batch_generate (bypassing the helper) would slip past every existing test. New test emulates the batch_generate scan loop on a stub with `inline_search_enabled=False` + emission text in both prompts → asserts `last_search_queries_per_prompt == [[], []]` AND `last_search_queries == []`. Catches the inline-bypass regression at the batch boundary.
- **Finding 3 — Rust SPECIAL_TOKENS comment off-by-two corrected** ([rust_extensions/src/lib.rs L867](rust_extensions/src/lib.rs)). Comment said `// Special tokens (IDs 0..12)` but the array has 14 entries (IDs 0..13 inclusive — `<search>` is 12, `</search>` is 13). Pass 156z9t verified the *array* matched Python byte-for-byte but did not update the *comment*. One-char fix: `12` → `13`. Comment-only — no rebuild required, no test changes, no behaviour change.

- **Finding 4 (acknowledged, no fix needed).** Audit flagged that the Pass 156z9r/s SUGGESTIONS stamps used "K-quant family closed" language while no on-disk Q3_K_M parity test exists. Re-read the actual stamps: neither 156z9r nor 156z9s claims "closed" — they claim "discipline now uniform" and "5-6 adversarial tests each," which is honest. The "closed" wording was only in the audit's own summary, not in SUGGESTIONS. No doc reframe needed. Real-world parity tests for Q2_K and Q3_K (loading actual GGUF models, comparing dequant output against llama.cpp reference) remain on the open follow-up list — same status as before this pass.

- **Author's-lens self-audit (§1 #19 six-question lens):**
  1. *Would I write it this way?* Yes — the strengthened Q4_K test follows the same pattern as Q3_K's `second_half_independence` test (3-in-1 adversarial gate with branching values that differ from broken-impl values by orders of magnitude).
  2. *Connected to what?* Same `dequantize_q4_K` function and `_record_search_emissions` helper as before. Test-only changes.
  3. *Connections that should exist but don't?* None — these are pure test-quality fixes plus one Rust comment.
  4. *Logic-eye?* Each new test docstring names the regression it catches AND the wrong-implementation value (nibble-swap → out[0]=18.0 not 2.0; off-switch bypass → empty lists not populated). Claim and test are paired.
  5. *Claim-vs-test?* The strengthened Q4_K test gates THREE properties simultaneously (sb0 routing, sb1 nibble-routing, sb4 stride) where the original gated only sb0. The batch off-switch test asserts both `last_search_queries_per_prompt == [[], []]` AND `last_search_queries == []` — checks both consequences in one test.
  6. *Sibling-boundary sweep?* Helper-only off-switch tests + batch integration test now cover the matrix: helper-direct (4 tests Pass 156z9u) + batch-loop integration (1 test this pass). No other generation path bypasses `_record_search_emissions` — verified by grep `_record_search_emissions` across `enigma_engine/core/`.

- **Lesson logged for §4 Learned Principles (audit-on-audit):** *Adversarial test claims must be falsified against the broken implementation, not just verified against the correct one.* The Pass 156z9s `test_dequantize_q4_K_layout_routing` docstring claimed to catch stride-bug regressions, but the test data made the claim untestable — sub-blocks 1..7 had sc=0, so a stride bug that misrouted their qs source still produced 0 output. Detection rule: when writing an adversarial test, mentally apply the regression you claim to catch and **compute the broken value** — if the broken value equals the correct value (both 0, both empty, both None), the test is structural-presence not behavioural-correctness. The cleanest fix is to give every region you claim to test a distinct nonzero "correct value," so a regression flips a real number to a different real number rather than 0 → 0. Same shape as the Pass 156i4 `test_replay_keeps_best_examples` deque-maxlen lesson (FIFO-ordered insertion made the "keeps best" claim trivially satisfied).

---



**Pass-by-pass prose for Passes 156d through 156z9u archived in [PASS_HISTORY.md](information/history/PASS_HISTORY.md) (doc-debloat May 1, 2026; ~890 lines moved). Open follow-ups carried forward into the live stamps above; learned principles live in AA code maker.md §4.**

---

**Pass 156c:** V-5 vision data collector shipped. New file [collect_vision_data.py](collect_vision_data.py) with `collect_llava_pretrain(max_samples, images_dir)`. Streams `liuhaotian/LLaVA-Pretrain` caption metadata; image bytes stay in user-managed `images.zip` extraction (separation principle — no auto-download of multi-GB archives). Per-row file-existence check skips missing images with warning (suppressed after 5 to avoid log flood; total reported at end). Output JSONL `data/vision/llava_pretrain.jsonl` with `{"image": <abs_path>, "text": <gpt_caption>}` — directly consumable by [scan_vision_data](enigma_engine/gui/scanners.py#L679) JSONL strategy and `Trainer.train_vision()`. Hard `FileNotFoundError` if `--images-dir` missing or non-existent (fail-fast over silent empty JSONL). Eight tests in [tests/test_collect_vision_data.py](tests/test_collect_vision_data.py) covering happy path, missing-image warning, max_samples cap, dedup, malformed-row skip, unknown-repo error, missing-dir hard-fail, and JSONL writer round-trip — fake-`datasets` injection per Pass 155 lesson. Closes V-5; unblocks V-1, V-2, V-4, V-7. Suite **2357/9** green, ruff clean.
**Pass 156:** F/Code-6 vision-training audit complete (read-only, author's lens). No code changes; findings logged below.
**V-8 [FIXED, Pass 156b]:** Vision encoder load path now wired. New helper `_load_vision_encoder_from_checkpoint(raw_checkpoint, model, model_file)` at [inference.py](enigma_engine/core/inference.py) called from `_load_pytorch` right after `model.load_state_dict(...)`. Volume policy implemented exactly per the agreed table: state+config present and load OK → `INFO` once with image_size/dim/patches; state+config present and load fails → `RuntimeError` (loud); state present without config → `RuntimeError`; state present but model has no `vision_projection` → `RuntimeError`; neither key present → silent (normal text-only path). Chat-side counterpart: [engine_chat.py L80](enigma_engine/core/engine_chat.py#L80) `_encode_images_for_chat` now logs `WARNING` per call when `image_paths` is non-empty but `vision_encoder is None` ("image input ignored — train a vision-capable model with Forge and reload"); empty image list stays `DEBUG`. Seven new tests in [tests/test_inference.py](tests/test_inference.py) (`TestVisionEncoderLoad` ×5 + `TestImageDroppedWarning` ×2) — all green. Original V-8 entry below kept for trace.

**V-8 [original, kept for trace — now FIXED above]:** Vision-encoder load path is fully missing. Trace: trainer at [gui_forge_training.py L911](enigma_engine/gui/gui_forge_training.py#L911) builds `VisionEncoder(v_preset)`, [L997-998](enigma_engine/gui/gui_forge_training.py#L997) writes `vision_encoder_state` + `vision_encoder_config` to the `.pth` as top-level keys. Grep across the entire `enigma_engine/` package shows **zero readers** of either key. [inference.py L209](enigma_engine/core/inference.py#L209) sets `self.vision_encoder = None` and grep shows **zero other assignments** anywhere. [engine_chat.py L80](enigma_engine/core/engine_chat.py#L80) checks `if encoder is None: return None`, so `vision_features` stays None at [L584](enigma_engine/core/engine_chat.py#L584), then [L622](enigma_engine/core/engine_chat.py#L622) skips the multimodal branch entirely. Net result: train vision → save .pth → load → drop an image into chat → model behaves identically to text-only, with **no warning**. The `vision_projection` weights do round-trip (they live inside `model_state_dict`) but they're useless without an encoder feeding them. Classic "infrastructure without consumers is dead code." Stopping work and asking the user which of three patch options to take before any edits: (a) auto-load encoder from the same .pth as the model, (b) require a separate `--vision-checkpoint` path on the engine, (c) bake the encoder weights into a sub-key of `model_state_dict` and let `load_state_dict(..., strict=False)` handle it. Until this is decided, F/Code-6 should be marked "training works, inference disconnected" rather than "complete." **Resolved with option (a).**


**Pass 155b:** SmolTalk2 split-resolution bug found in live run; fixed + retested.
**Pass 155:** D-4 OpenThoughts3 + D-11 SmolTalk2 fetchers shipped.

**Pass 156 audit — `Trainer.train_vision()` (F/Code-6):** Read-only audit of [enigma_engine/core/training.py L4785-5125](enigma_engine/core/training.py#L4785). Function name is `train_vision()`, not `train_vision_encoder()` (summary was wrong; verified by grep). Single GUI call site at [gui_forge_training.py L988](enigma_engine/gui/gui_forge_training.py#L988). Five behavioural tests in [tests/test_training.py L85-300](tests/test_training.py#L85) cover update / return-state / loss-decrease / callbacks / stop / no-projection-error. Author's-lens findings below.

*Strong points (ship-as-is for SFT-stage):* Validation gate, freeze/unfreeze logic correct (text frozen, projection always trainable, output/embedding/norm unfrozen for text loss, optional last-N text layers, encoder fully trainable, freeze_backbone re-applied AFTER blanket unfreeze — order matters and is right). Uses `_effective_warmup` (Pass 152). AMP scaler gated correctly. NaN/Inf guard + `max_loss` guard + early stopping + best-checkpoint + periodic checkpoint with `_cleanup_periodic_checkpoints` all present. Next-token slicing (`logits[:, n_patches:-1, :]` vs `text_tensor[:, 1:]`) verified correct: both end up length `N-1` for an `N`-token caption. Tests confirm gradients flow into both encoder weights and projection weights ([tests/test_training.py L116-118](tests/test_training.py#L116)).

*Gaps (ranked, not bugs but real omissions):*
1. **No `max_grad_accumulation` support** — confirmed by grep across the function body. Every other training method honors it (pre-train L2976/3226/3311/3320, DPO L4042, ReST L4311, audio L4542). Vision optimizer steps every sample. Files at scale will starve the optimizer of effective batch size on a 16 GB VRAM budget. *V-1 backlog row.* **[FIXED, Pass 156e]** — see top of file. Two behavioural tests added; suite +2.
2. **Eager preprocess to GPU RAM** — [L4906-4929](enigma_engine/core/training.py#L4906): `pairs.append((img_tensor, token_ids))` accumulates every preprocessed tensor *on `self.device`* before the loop starts. At 224²·3·4 B ≈ 600 KB per image, 100K pairs = 60 GB on GPU. OOMs RTX 5090 immediately at LLaVA-Pretrain scale (558K pairs). Must convert to lazy/iter pattern (preprocess in the step loop) or use disk-backed offset list per the learned principle. *V-2 backlog row.* **[FIXED, Pass 156f]** — see top of file.
3. **Batch size hardcoded to 1** — [L4889 comment](enigma_engine/core/training.py#L4889) `total_steps = len(data) * self.config.epochs  # 1 step per pair` is intentional but wastes 5090. Real LLaVA training uses batch 32-128. Note: enabling true batching needs padded text + attention mask plumbing through `forward_multimodal`. Non-trivial. *V-3 backlog row.*
4. ~~**No heartbeat write** — `gui_forge_training.py` does NOT call `training_heartbeat.json` (only `gui_forge_new_modes.py` does, [L183/L238](enigma_engine/gui/gui_forge_new_modes.py#L183)). OOM-kills on long vision runs leave no last-known-good state. *V-4 backlog row.*~~ **V-4 closed Pass 156d.** `_start_vision_training` now writes the same heartbeat as pre-training (stale check on entry, `_write_hb` closure inside thread, status calls at every lifecycle branch). Vision OS-OOM kills are now post-mortem-detectable.
5. ~~**No data collector** — there is no `collect_vision_data.py` to mirror `collect_pretraining_data.py` / `collect_finetuning_data.py`. F/Code-6 ships a *trainer* with no production data pipeline. Real options: liuhaotian/LLaVA-Pretrain (558K), Lin-Chen/ShareGPT4V-PT (1.2M), COCO captions (118K), LAION-5B subsets. *V-5 backlog row — needed before V-1/V-2 to give them work to do.*~~ **V-5 closed Pass 156c.** Shipped `collect_vision_data.py` with LLaVA-Pretrain fetcher; ShareGPT4V/COCO can be added with the same shape now that the framework is in place. Row schema is best-guess from the public dataset card and **must be live-validated on the first real run** (Pass 155 lesson).
6. ~~**No validation/eval split** — train loop only computes training loss. Other paths have `evaluate()` mid-training. Overfitting on small datasets is invisible. *V-6 backlog row, low priority.*~~ **V-6 + V-6b closed Pass 156g.** Backend: `train_vision()` accepts optional `val_data`; `_run_validation()` runs no-grad eval after each epoch, populates `state.validation_losses`, drives best-checkpoint + early-stopping when present. GUI: [gui_forge_training.py L1057](enigma_engine/gui/gui_forge_training.py#L1057) shuffle-splits `vision_data` according to `train_config.val_split` and passes the held-out slice as `val_data=`.
7. ~~**Single-token caption silently dropped** — [L5005](enigma_engine/core/training.py#L5005) `if min_len < 1: continue`. Caption like `"cat"` after BOS-stripping has `min_len=0` and is dropped. Edge case but should be logged once instead of silent-continue. *Tiny fix, V-7.*~~ **V-7 closed Pass 156d.** First drop logs a warning naming reason + epoch + step; remaining drops counted; end-of-run summary reports total. Pipeline is no longer silently lossy.

*Connections-that-don't-exist (the third author's-lens question):*
- Vision-only loss option (CLIP-style image-text contrastive on the projection head) is not in the trainer at all. Current loss is captioning CE only. For the LLaVA-style pipeline this is correct; for general image embedding (used by ImgGen / search-by-image) you'd want contrastive too. *Not a backlog row — this is approach-2 territory and we picked Approach 1 + 3.*
- The GUI save block at [gui_forge_training.py L995-1003](enigma_engine/gui/gui_forge_training.py#L995) emits `vision_encoder_state` and `vision_encoder_config` as separate top-level keys, but `Enigma.load_state_dict()` doesn't know about those keys — the GUI loader has to pull them apart manually. No load-side code visible in the audit window. *V-8 backlog row — verify load path before the first real run.*

*Verdict:* `train_vision()` is **ship-as-is for the SFT-stage smoke test** (small image-text pairs, 1-3 epochs, validates pipeline correctness). For LLaVA-scale pretraining, V-3 (true batching) is the remaining must-land item; it's optional if 1-step-per-pair is acceptable. V-1, V-2, V-4, V-5, V-6, V-6b, V-7, V-8 closed. Suite **2370/9**.

**Pass 155b live-validation patch (SmolTalk2):** First live Phase-1 smoke run on RTX 5090 caught two contract assumptions that the Pass 155 fake-`datasets` tests had silently blessed: (1) `split="train"` was hardcoded, but SmolTalk2 has **no** "train" split — each config (`SFT`, `Mid`, `Preference`) has 25-ish named splits like `smoltalk_smollm3_smol_magpie_ultra_no_think`, `OpenThoughts3_1.2M_think`, etc.; (2) the prior write-up confused configs and splits — `smol_magpie_ultra` is a *split-name fragment*, not a config. Patched [collect_finetuning_data.py](collect_finetuning_data.py) `collect_smoltalk2(max_samples, config, split=None)`: when `split is None`, calls `get_dataset_split_names(repo, config)` and concatenates rows from every split until `max_samples`; when explicit, uses it directly. New CLI flag `--smoltalk2-split NAME`. Test fake [tests/test_collect_finetuning_data.py](tests/test_collect_finetuning_data.py) `_install_fake_datasets()` now accepts `splits_by_path` and exposes `get_dataset_split_names`; +2 tests (`test_split_none_iterates_all_splits`, `test_explicit_split_used_directly`). Live re-run with `--smoltalk2-config SFT --smoltalk2-split smoltalk_smollm3_smol_magpie_ultra_no_think` produced 999 pairs / 2.0 MB. OpenThoughts3 live-validated separately: 1000 pairs / 57.9 MB, **all 1000 rows contain `<think>` tag** verbatim (`Select-String <think>` count = 1000). Suite **2342/9** green, ruff clean. Files on disk: [data/finetune/openthoughts3.jsonl](data/finetune/openthoughts3.jsonl), [data/finetune/smoltalk2.jsonl](data/finetune/smoltalk2.jsonl), [data/finetune/combined_finetune.jsonl](data/finetune/combined_finetune.jsonl) (1999 pairs).

**Pass 155b learned principle (logged in [AA code maker.md](AA%20code%20maker.md)):** Test fakes that ignore kwargs hide signature/contract bugs. When a real argument can be wrong (split name, config name, region), the fake should validate at least the shape of expected values, not blindly accept anything. Pass 155 fake `load_dataset(path, *args, **kwargs)` ignored `split=` entirely — the test "passed" against `split="train"` even though no real SmolTalk2 split is named that. Always read what the fake throws away; that's your blind spot.

**Pass 155 backlog row (kept for context):** D-4 OpenThoughts3 + D-11 SmolTalk2 fetchers.

**D-4 OpenThoughts3 fetcher (Pass 155):** New `collect_openthoughts3(max_samples)` in [collect_finetuning_data.py](collect_finetuning_data.py). Streams `open-thoughts/OpenThoughts3-1.2M` (Apache-2.0, 1.2M rows = 850K math + 250K code + 100K science). Extracts `(human, gpt)` turns from each row's `conversations` list. **Reasoning `<think>...</think>` tags preserved verbatim** — NO `_clean_text` whitespace collapse on the gpt value, so the QwQ-32B reasoning trace newlines survive and align with our special token IDs (`<think>=4`, `</think>=5`). Per-row schema verified Pass 139. Reuses existing `_dedup_pairs` (sha256 prefix). CLI: `python collect_finetuning_data.py --openthoughts3 100000`.

**D-11 SmolTalk2 fetcher (Pass 155 + 155b):** New `collect_smoltalk2(max_samples, config, split=None)` in [collect_finetuning_data.py](collect_finetuning_data.py). Streams `HuggingFaceTB/smoltalk2`. Real config names are `SFT`, `Mid`, `Preference` (verified live, not the speculative `smol_magpie_ultra` named in earlier drafts \u2014 that string is a *split fragment* inside the SFT config). Each config has ~25 named splits, none called "train". Pass 155b adds split auto-enumeration: when `--smoltalk2-split` is omitted, `get_dataset_split_names()` is called and rows from every split in the chosen config are concatenated until `max_samples`. On a missing config or split the HF loader raises ValueError naming the available choices; we log and return `[]` (per learned principle: detect on first attempt, no loop). ChatML schema (`messages: [{role, content}]`); first user/assistant pair extracted, optional system prepended (skips generic "You are a helpful assistant."). CLI: `python collect_finetuning_data.py --smoltalk2 100000 --smoltalk2-config SFT [--smoltalk2-split NAME]`. Both new fetchers wired into `--all`. 8 tests in [tests/test_collect_finetuning_data.py](tests/test_collect_finetuning_data.py) using fake-`datasets` injection \u2014 no network: 3 OpenThoughts3 (verbatim tags, missing-turn skip, max_samples cap) + 5 SmolTalk2 (extract pair, unknown-config error log, short/empty skip, split=None auto-enum, explicit split). Closes P1 backlog rows D-4 and D-11. Suite **2342/9** green, ruff clean.

**Pass 154:** Stage A helpers shipped Pass 153 are now live in the chat loop. In [enigma_engine/gui/gui_logic_chat.py](enigma_engine/gui/gui_logic_chat.py) `_send_message::_gen()`, after the model produces its first-pass reply (post-`extract_reasoning` / `strip_incomplete_think`, pre-`parse_commands`), a post-generation gate runs `should_retry_with_research(msg, resp)` when (a) web access is on, (b) no pre-generation `auto_research()` already ran, and (c) the user has not pressed STOP. On uncertainty score ≥ 0.55 the system fetches `auto_research(msg, max_results=3)`, appends it to `combined_prompt`, and re-invokes `engine.chat()` once with the augmented `system_prompt`. Reasoning is re-extracted from the retry output. Failures are swallowed (logged at DEBUG) so the original reply is never lost. Single retry only — no loop. 5 new structural tests in [tests/test_gui_logic_chat.py::TestAutoResearch2Wiring](tests/test_gui_logic_chat.py) verify the helper import, the `_web_access_on` gate, the `not web_research_ctx` skip, the `_stop_requested` short-circuit, and the try/except wrapper. Behavioral coverage of the helpers themselves stays in `test_core.TestAutoResearch`. Closes the AutoResearch-2 Stage A wiring follow-up noted in Pass 153. Suite **2334/9** green, ruff clean.

**Pass 153:** AutoResearch-2 Stage A (post-generation uncertainty gate) BUILD shipped.

**AutoResearch-2 Stage A:** New public API in [enigma_engine/core/auto_research.py](enigma_engine/core/auto_research.py) — `UncertaintyResult` dataclass (`score`, `reasons`), `score_uncertainty(query, response)`, and `should_retry_with_research(query, response, threshold=0.55, enabled=True)`. Signal-driven, deterministic, no RNG (per R-UNPREDICT-1 spec, Pass 146). Combines four signals: hedge phrases ("I'm not sure", "I think", "might be", capped at 0.6), refusal/apology ("I apologize", "I cannot answer", capped at 0.6), short-response anomaly (long query + tiny reply = +0.3), and question-echo (response repeats query with little new content = +0.3). Hard off-switch via `enabled=False`. Empty response → score 1.0 with reason `empty_response`. Stage B (inline `<search>` token in generation loop) remains future work — needs logits access. **Wiring into chat loop is a separate follow-up pass** ([gui_logic_chat.py L239](enigma_engine/gui/gui_logic_chat.py#L239) `_gen()` is the call site to update). 10 new tests in [tests/test_core.py::TestAutoResearch](tests/test_core.py) cover confident/hedge/refusal/empty/short-query/deterministic for `score_uncertainty`, plus skip/trigger/off-switch/threshold-configurable for `should_retry_with_research`. Closes AutoResearch-2 Stage A (P1 backlog row 11). Suite **2328/10** green, ruff clean.

**Pass 152:** Sched-2 (warmup cap for short SFT/DPO runs) BUILD shipped.

**Sched-2:** New module-level helper [`_effective_warmup(warmup_steps, total_steps)`](enigma_engine/core/training.py#L40) caps warmup at `total_steps // 5` (20% of run). Fixes the silent bug where the default `warmup_steps=100` produced 100% warmup on a 100-step SFT run and 50% on a 200-step run — i.e. no decay phase at all. All 5 scheduler-construction sites updated to use the helper: pre-train scheduler setup ([L2745](enigma_engine/core/training.py#L2745)), SWA cosine-restart boundary check ([L3368](enigma_engine/core/training.py#L3368) — uses `self._total_training_steps`), DPO scheduler ([L4023](enigma_engine/core/training.py#L4023)), vision-projection scheduler ([L4865](enigma_engine/core/training.py#L4865)), audio-projection scheduler ([L5274](enigma_engine/core/training.py#L5274)). Long runs unaffected (cap inactive when `warmup_steps <= total_steps // 5`); user-explicit large `warmup_steps` is still respected up to the 20% ceiling. Edge cases handled: `total_steps=0` returns `max(1, warmup_steps)` (no division-by-zero), `total_steps=1` returns `1`. 8 new tests in [tests/test_training.py::TestEffectiveWarmup](tests/test_training.py) cover short-cap (50/200 totals), cap-inactive (1k/10k totals), zero/one edge cases, explicit-respected, and explicit-excessive-capped. Closes P2 row #20 (Sched-2). Suite **2319/9** green, ruff clean.

**Pass 151:** Vision-1b (projection MLP upgrade) BUILD shipped.

**Vision-1b:** [enigma_engine/core/model.py L186-200](enigma_engine/core/model.py#L186) `self.vision_projection` upgraded from `nn.Linear(vision_hidden, dim, bias=False)` to LLaVA-1.5 reference impl `nn.Sequential(Linear(vision_hidden, dim, bias=True), GELU, Linear(dim, dim, bias=True))` per arxiv:2310.03744 §3.2 + HF `LlavaMultiModalProjector`. Forward call site at [model.py L705](enigma_engine/core/model.py#L705) untouched (callable on `Sequential` identically). `.parameters()` consumer at [training.py L4815](enigma_engine/core/training.py#L4815) safe. Updated [test_training.py L101/119](tests/test_training.py#L101) `.weight` → `[0].weight` (snapshot first Linear). State-dict keys changed: `vision_projection.weight` → `vision_projection.{0,2}.{weight,bias}` — acceptable break because no vision checkpoints exist yet (vision_hidden_size opt-in, Code-6 not built). Param cost at dim=2048: ~2.1M → ~6.3M (+4M, negligible vs 742M). 6 new tests in [tests/test_model_arch.py::TestVisionProjectionMLP](tests/test_model_arch.py): structure (Sequential of 3), GELU middle, dimensions, bias=True both Linears, forward shape preservation, end-to-end forward_multimodal call. Closes Decision Queue row D and P2 row #27 (Vision-1b half).

Tests: 6 new (TestVisionProjectionMLP). Suite **2311/9** green, ruff clean.

**Pass 150:** Eval-1 / D-10 (GSM8K reasoning benchmark) BUILD shipped.

**Eval-1 / D-10:** Three new public functions in [enigma_engine/core/training_evaluation.py](enigma_engine/core/training_evaluation.py): `parse_final_number(text)` (canonical `#### N` marker first, last-number fallback, comma-stripped, decimal/negative-safe), `load_gsm8k(path, n)` (offline-first JSONL loader, defaults to `data/gsm8k_test.jsonl`, raises `FileNotFoundError` with the exact one-time `datasets.load_dataset('openai/gsm8k', 'main', split='test').to_json(...)` recovery command — no silent network fetches at benchmark time), and `run_gsm8k_benchmark(engine, examples, num_shots=8, max_gen=256, temperature=0.0, on_progress=None)` (8-shot CoT prefix from Wei et al. 2022 Table 20, exact-match scoring with float tolerance, per-item `output[:200]` capture, returns `{total, correct, accuracy, results}`). CLI: `--benchmark` upgraded from `store_true` to `nargs="?"` with `choices=["coherence", "gsm8k"]` and `const="coherence"` (backward compatible — bare `--benchmark` still runs coherence). New flags `--benchmark-data PATH`, `--benchmark-limit N`, `--benchmark-shots N`. New `run_gsm8k_benchmark_cli()` dispatcher in [run.py](run.py) loads model, calls loader, streams progress. 14 tests in [tests/test_evaluation.py::TestGSM8KBenchmark](tests/test_evaluation.py) cover gold-format parse, comma/negative/decimal/empty/no-number, file-missing error message, JSONL read, n-cap, mock engine accuracy, unparseable output, float tolerance, empty input. Closes Decision Queue row C and P1 row #7.

Tests: 14 new (TestGSM8KBenchmark). Suite **2305/9** green, ruff clean.

**Pass 149:** Tok-2 + MTP-2b BUILDs shipped.

**Tok-2:** [enigma_engine/core/bpe_tokenizer.py L88](enigma_engine/core/bpe_tokenizer.py#L88) default `use_utf8_bytes` flipped `False`→`True`; [enigma_engine/core/advanced_tokenizer.py](enigma_engine/core/advanced_tokenizer.py) gained full byte-mode parity (init flag, `_text_to_bytes`/`_bytes_to_text` helpers, encode/decode branches, save/load persistence with legacy-False default). Rust `enigma_bpe` already had the flag end-to-end — no rebuild required. Closes Decision Queue row A and P0 row #2.

**MTP-2b:** [enigma_engine/core/model_presets.py L129](enigma_engine/core/model_presets.py#L129) default `n_predict_heads` flipped `2`→`0` and inverted comment fixed (paper conclusion was upside-down per Pass 140 finding). Saves ~33-49M params on every fresh model built from the default config. Existing `n_predict_heads > 0` guards in [model.py](enigma_engine/core/model.py) (L238, L496, L594) and [progressive_growing.py](enigma_engine/core/progressive_growing.py) (L219) handled the zero case correctly with no surgery. Pass 148 already designated EAGLE-2 as the inference-time speculative path — Medusa (the only consumer of the heads) is no longer planned. Closes P2 row #27b. Updated [test_research_upgrades.py L876](tests/test_research_upgrades.py#L876) to assert new spec + added `test_default_model_has_no_predict_heads`. Lowered `test_small_preset_range` floor 50M→30M in [test_core.py L1543](tests/test_core.py#L1543) (the floor was inflated by the MTP head bytes, not real model capacity).

Tests: 6 new/changed (Tok-2) + 2 new/changed (MTP-2b) + 1 spec-floor adjustment. Suite **2291/9** green, ruff clean.

**Pass 148 (currency check addendum):** Tool-verified the research currency of the long-form resolved items before Phase 0 kicks off. **Phase 1** fetched primary sources for 6 high-turnover domains (speculative decoding, vision encoders, image gen, video gen, 3D gen, reasoning teachers): 3 items marked **[SUPERSEDED]** with successor blocks inline (**Video-1** AnimateDiff → Wan 2.1 T2V-1.3B primary + LTX-Video 2B-distilled alt; **3D-1** Shap-E → Hunyuan3D-2 primary + TRELLIS-image-large alt; **Medusa** training recipe stays valid but EAGLE-2 arxiv:2406.16858 supersedes it as the inference-time speculative method); 2 upgrade notes (**Vision-1b** SigLIP 2 arxiv:2502.14786; **Approach 3 teacher** DeepSeek-R1-0528-Qwen3-8B distilled checkpoint alongside Qwen3-30B-A3B). **Phase 2** fetched primary sources for 4 more domains (text embeddings, ASR, audio generation, vision-LLM): added **Vision-2b** (LLaVA-NeXT → LLaVA-OneVision arxiv:2408.03326 with SigLIP SO400M + Qwen2, 0.5B variant fits 16 GB), **RAG-2b** (bge-small-en-v1.5 → Qwen3-Embedding-0.6B arxiv:2506.05176, MTEB multilingual 64.33, Apache-2.0), **Audio-5b** (MusicGen CC-BY-NC noted; Stable Audio Open Small arxiv:2505.08175 preferred for general SFX), and new **Audio-ASR-1** (Whisper → NVIDIA Parakeet-TDT-0.6B-v2 6.05 WER / RTFx 3386, ~16× faster). **D-20 FA4** updated to verified v4.0.0.beta10 status: SM120 fwd+bwd+varlen merged via PRs #2329/#2330/#2333, `pip install flash-attn-4` works on Linux/WSL, Windows wheels still TBD. Also fixed stale **Priority Index P3 numbering** (duplicate #28 → renumbered 37-45) and stale **audit counter** (135 → 148 passes). Doc-only, no code changes.
**Pass 148:** Two-round audit of the Pass 147 GUI Modernization Plan. All edits scoped to the planning section below. Round-1 fixes: removed gate/weight double-counting, dropped impossible install-size target, relocated service contract from `enigma_engine/api/` (already FastAPI) to new `enigma_engine/services/`, added baseline measurement step, added 5-day per-track time-box, added decision-ladder cases for ties and dual failures, anchored page inventory to filesystem reality, deduped Phase 4/5 overlap. Round-2 fixes (tool-verified): (1) "parallel" → sequential 10-working-day bake-off (solo builder); (2) added explicit `api/server.py` → `services/` migration sub-step in Phase 4 so the "single service contract" claim is real; (3) resolved PyInstaller contradiction (POC shortcut in Phase 1, packaging decision deferred to Phase 3c); (4) replaced contradictory cold-start/page-switch dual thresholds with primary-vs-stretch targets; (5) split GUI file list into 16 user-facing page modules vs 7 support modules (`widgets.py`, `themes.py`, `scanners.py`, `media.py`, `gui_logic*.py`); (6) baseline now measures G1/G2/G3 on CustomTkinter too so "beat baseline" is comparable; (7) confirmed Tauri sidecar URL (https://v2.tauri.app/develop/sidecar/) documents `externalBin` + `-$TARGET_TRIPLE` suffix + `shell:allow-execute` capability with pyinstaller as canonical Python bundling path; (8) added project-goal drift check ("personality from training, not the user" + "blackbox") to Phase 0d page inventory. No source code changes.
**Pass 147:** Planning-only GUI architecture research pass (no implementation). Goal constraints locked: best possible UX/architecture, fully private, local/offline-capable, black-box. Researched desktop and webview stacks and documented migration-ready decision gates. Key sources: Tauri security model and trust boundaries (https://v2.tauri.app/security/), Electron security checklist and threat model (https://www.electronjs.org/docs/latest/tutorial/security), Qt for Python deployment and API surface (https://doc.qt.io/qtforpython-6/), Tkinter threading/event-loop architecture (https://docs.python.org/3/library/tkinter.html), pywebview local wrapper model (https://pywebview.flowrl.com/guide/), local/self-host references for web UIs (Open WebUI README and offline mode notes: https://github.com/open-webui/open-webui, Gradio local-only defaults and sharing behavior: https://gradio.app/guides/sharing-your-app/, NiceGUI local server/native mode docs: https://nicegui.io/documentation, Flet desktop/web run modes: https://flet.dev/docs/). Output: execution-ready planning section added below.
**Pass 146:** Research-only continuation. Resolved R-UNPREDICT-1 with primary-source evidence and converted it to an implementation-ready design: uncertainty-gated retrieval and tool-use should be signal-driven (confidence/retrieval-quality), not keyword-only and not pure RNG. Evidence used: ReAct (arxiv:2210.03629), Toolformer (arxiv:2302.04761), Self-RAG (arxiv:2310.11511), and CRAG (arxiv:2401.15884). Also cleaned stale GRPO-2 research status in the post-training section (already shipped in Pass 142). No code changes.
**Pass 145:** Research-only continuation. Resolved all R-ARCH-4 items (4a/4b/4c): finalized sequential integration order, defined phase-gated benchmark suite, and added explicit fallback/skip policy so failed optional tracks do not stall core roadmap. Net result: Approach 4 is now fully specified for execution with hard stop conditions on core regressions and safe bypass for non-core failures. No code changes.
**Pass 144:** Research-only continuation. Resolved all R-ARCH-3 items (3a/3b/3c/3d) for the distillation POC track. Core conclusions: 742M-class students can learn substantial reasoning/coding behavior from larger teachers when supervision is rich (DeepSeek-R1 distill + Orca evidence), data should be staged by risk (pilot/useful/strong bands rather than one fixed size), benchmarking must be per-specialist plus routed end-to-end quality-cost frontier, and routing should start heuristic for fast POC but graduate to learned routing at defined misroute/fallback thresholds (RouteLLM evidence). No code changes.
**Pass 143:** Research-only continuation. Resolved R-ARCH-2b and R-ARCH-2c. Conclusion for 2b: use heterogeneous specialist sizing (capacity by task difficulty) while keeping a single interface contract (tokenizer/prompt/eval schema), based on quality-cost routing evidence from RouteLLM/FrugalGPT (arxiv:2406.18665, arxiv:2305.05176). Conclusion for 2c: avoid per-request hot-swaps; use resident pool + async warmup + queue policy, grounded in iteration-level scheduling (ORCA OSDI'22), KV-memory paging (PagedAttention arxiv:2309.06180), and split prefill/decode serving evidence (DeepSpeed-FastGen arxiv:2401.08671). No code changes.
**Pass 142:** GRPO-2 BUILD shipped + research sweep. New file [enigma_engine/core/reward_functions.py](enigma_engine/core/reward_functions.py) (rule-based `format_reward` / `math_reward` / `code_reward` / `reasoning_reward`). [gui_forge_new_modes.py L2229-L2260](enigma_engine/gui/gui_forge_new_modes.py#L2229-L2260) routes GRPO `reward_fn` to `reasoning_reward` and skips neural Phase 1 for GRPO (ReMax still uses neural path). Added [tests/test_reward_functions.py](tests/test_reward_functions.py) (4 tests). Closes Decision Queue row B and P1 rows 4 (GRPO-2 BUILD) + 5 (GRPO-4 block neural path). Research-only half of the pass resolved R-ARCH-1b (LLaVA CLIP-ViT-L/14@336 frozen + 2-layer MLP+GELU projector, 304M), R-ARCH-1c/1d (layered router + RouteLLM-style training), R-ARCH-2a/2d/2e (hybrid architecture, preference+LLM-judge labelling, cost/quality Pareto eval) against arxiv:2310.03744 + arxiv:2406.18665 + arxiv:2305.05176. Suite 2292/2 green, ruff clean.
**Pass 141:** Bookkeeping only. All [RESEARCH] items from the original backlog are closed (last batch resolved in Pass 140). Remaining backlog is code work: [BUILD] / [CONFIRMED GAP] items. Logged a Pass 141 Decision Queue below so the next code session can pick a target without re-triaging. No file changes outside this document's header + queue section. No code changes. Other 3 canonical docs (CODE_REVIEW.md, GUI_REFERENCE.md, AA code maker.md) stamped to Pass 141 with "no-op sync" notes.
**Pass 140:** Research-only continuation. Resolved 13 remaining [RESEARCH] items from the original backlog: **Attn-2** (Differential Transformer ICLR 2025 Oral arxiv:2410.05258 — keep enabled; manual-attention path is the cost), **MTP-2** (Meta arxiv:2404.19737 — sub-1B is ambiguous bracket; found inverted comment at [model_presets.py L129](enigma_engine/core/model_presets.py#L129) and untied 98M-param predict_heads at [model.py L237-240](enigma_engine/core/model.py#L237-L240)), **Tok-1** (Tao et al. NeurIPS 2024 arxiv:2407.13623 — optimal ~40K at 742M, 64K over-provisioned), **Inf-2** (three-tier attention dispatch — differential-attn default forces manual path), **Spec-1/2/3** (LoRA-per-specialist is the only serious 16 GB path; LM Eval Harness for benchmarks), **Grow-1/2** (train Chinchilla-optimal before grow; re-warmup not reset), **Medusa-1/2** (our joint-loss path matches Medusa-2; add λ warmup), **Router-2/3** (layered regex → embedding → LLM fallback), **Haptic-1** (no standard dataset/model exists — deprioritize). **New backlog items logged:** MTP-2a (reduce default `n_predict_heads` 2→1, weight-tie), MTP-2b (set to 0 if Medusa not planned), Medusa-1a (λ warmup from 0 → 0.2 over ~500 steps), Medusa-2a (measure acceptance rate via `--benchmark medusa-acceptance`). No code changes.
**Pass 139:** Cleanup + continuation research. Removed stale duplicate [RESEARCH]/[RESOLVED] entries for Arch-8, Attn-1, MTP-1, Sched-2, Distill-1/2 so each item has one canonical status line. Added code-verified Distill-2 note: Forge distillation currently generates teacher data at `temperature=0.8` in [enigma_engine/gui/gui_forge_new_modes.py](enigma_engine/gui/gui_forge_new_modes.py#L1471), which matches the recommended SeqKD range (0.7-1.0). Resolved Distill-4 with concrete evidence: Qwen3 reports ~10x GPU-hour savings from strong-to-weak distillation, and DeepSeek-R1 reports ~800K SFT samples (600K reasoning + 200K non-reasoning) for distilled students. Inf-4 status finalized as resolved with explicit KV-cache math (150MB at 4K, ~1.17GB at 32K). MTP-2 moved to PARTIAL: literature supports MTP overall, but no clean sub-1B/742M ablation was found; internal A/B recommended. Data-4 + Opt-3 finalized (HF datasets-server API confirms `<think>` tags in OpenThoughts3 raw rows; weight-tied `output.weight` never visited by `named_parameters(remove_duplicate=True)`). **Vision-1..4 resolved via LLaVA-1.5 paper (arxiv:2310.03744):** frozen CLIP-ViT-L/14 @ 336px → 2-layer MLP+GELU → LLM, 576 visual tokens, `openai/clip-vit-large-patch14-336` (1024-d), Stage 1 on `liuhaotian/LLaVA-Pretrain` (558K) then Stage 2 on `liuhaotian/LLaVA-Instruct-150K` + academic VQA (665K). **Code gap logged as Vision-1b:** our `self.vision_projection` is a single `nn.Linear` (LLaVA-1 era), should be 2-layer MLP+GELU for LLaVA-1.5 parity. **Inf-5 resolved and deferred to P3:** `flash_attn_with_kvcache` is the FA2 decode-tuned kernel (changelog v2.2), ~15-30% decode gain at our 742M/4K/batch-1 workload (not the 2× headline numbers which assume 8K+/larger models), and flash-attn install is fragile on RTX 5090 (Blackwell SM120 unverified) + Windows. Revisit if a stable Blackwell/Windows wheel ships with FA4. No code changes.
**Pass 138:** Architecture, MTP, and Distillation research pass. Resolved: Arch-8 (no LayerScale at 0.6B–3B — SmolLM3 + Qwen3 verified), Attn-1 (4:1 GQA validated at 3B via SmolLM3), MTP-1 (λ=0.3 from DeepSeek-V3 paper; our 0.5 is high — lower to 0.3 for next pretraining run), Sched-2 (max(100, steps//10) confirmed adequate), Distill-1 (SeqKD = correct approach, reverse KL is future upgrade per MiniLLM ICML 2023), Distill-2 (T=4–10 is Hinton soft-label only; for our SeqKD use T=0.7–1.0 for teacher generation). Data-4 still PARTIAL — OpenThoughts3 tag format needs raw dataset sample check. No code changes.
**Pass 137:** Data-1 through Data-5 researched via external sources (SmolLM3 blog, FineWeb arxiv 2406.17557, DCLM arxiv 2406.11794, OpenThoughts3 arxiv 2506.04178, FineMath arxiv 2502.02737). Five of six claims VERIFIED by audit; Data-4 downgraded to PARTIAL — the `<think>` tag format was assumed from SmolLM3's description, not confirmed in OpenThoughts3's own dataset card. Concrete Stage 1 mixing ratios now available for N-6 restart. One new finding: our MinHash threshold default is 0.8 but industry standard (FineWeb/DCLM) is 0.75 — logged as Data-5b, low priority. No code changes.
**Pass 136:** CRIT-6 closed — DCLM + FineMath + The Stack v2 wired into [collect_pretraining_data.py](collect_pretraining_data.py). No unit tests added (matches existing network-gated fetcher pattern — fandom/fineweb/c4). New learned principle: gated HF datasets need fail-fast auth detection (see AA code maker.md).

Pass-by-pass prose (Passes 110-135) lives in [information/history/PASS_HISTORY.md](information/history/PASS_HISTORY.md). Per-file bug-fix history lives in the file tables of [CODE_REVIEW.md](CODE_REVIEW.md). This document keeps only the current state + forward-looking work.

**Recommended path:** Approach 1 (Reasoning-First) + Approach 3 (Distillation POC) in parallel. See "Architecture Approaches" section below.

## Next Actions (top of backlog)

1. ~~**CRIT-6:** Collect DCLM + FineMath + The Stack v2 (D-1/D-2).~~ **Done.** `--dclm`, `--finemath`, `--code` flags added to [collect_pretraining_data.py](collect_pretraining_data.py). Run: `python collect_pretraining_data.py --dclm 15 --finemath 10 --code 10`
2. ~~**Data-1..5:** mixing-ratio research for N-6 restart.~~ **Done (Pass 137).** Stage 1 target mixture: **85% web / 12% code / 3% math** per SmolLM3 blog. Stage 3 cooldown: **63% web / 24% code / 13% math**. Use `<think>` XML tags for reasoning traces.
3. ~~**GRPO-2 BUILD:** Write `reward_functions.py` (math pass/fail + format check) before any reasoning GRPO training.~~ **Done (Pass 142).** New file [enigma_engine/core/reward_functions.py](enigma_engine/core/reward_functions.py); GRPO dispatcher wired at [gui_forge_new_modes.py L2258-L2260](enigma_engine/gui/gui_forge_new_modes.py#L2258-L2260).
4. **Resume N-6:** pre-training with reasoning data emphasis using mixture from Data-1..5.
5. **Approach 3 POC:** distill reasoning traces from Qwen3-30B-A3B into 742M (SFT on ~5K cold-start CoT examples). **Pass 148 currency note:** two additional teacher options verified current — (a) **Qwen3-32B** (arxiv:2505.09388, May 2025, thinking/non-thinking dual mode, YaRN 131K) as a same-family larger alternative to 30B-A3B; (b) **DeepSeek-R1-0528-Qwen3-8B** (May 2025) as a ready-made distilled 8B reasoning checkpoint — SoTA among open sub-10B reasoning models — usable as cold-start weights for our 742M student if we want to bypass teacher inference entirely. QwQ-32B is now older than Qwen3-32B-thinking and should not be used as teacher.
6. **Personality-5:** Run personality distillation + identity/roleplay separation build plan (R-PERSONALITY-1 research resolved).
7. **GUI-ARCH-1:** Complete GUI platform decision gate (PySide6 vs Tauri vs keep/refactor current) using the planning criteria below before any UI rewrite.
8. **HYG-1 (NEW, Pass 156z9al-audit):** Stop tracking Rust build artifacts. `git ls-files rust_extensions/target/` shows **301 files** under `target/release/`, `target/maturin/`, `target/.fingerprint/` — all regenerated by `cargo build` / `maturin build` and currently bloating every commit that touches Rust (Pass 5005025 + 6cb4109 each carried 30+ binary deltas). `.gitignore` has zero Rust-related entries. **Action (destructive — needs user confirmation):** add `rust_extensions/target/` to `.gitignore`, then `git rm -r --cached rust_extensions/target/` and commit. Target-tree disappears from tracking but stays on disk. **Why parked, not done:** `git rm --cached` is a tracked-file removal that the operationalSafety rule flags as needing user confirmation; also the user may have a reason for wanting some of these in-tree (e.g. CI without Rust toolchain). Ask before doing.

---

## GUI Modernization Planning (Private, Local, Black-Box)

**Non-negotiable constraints (from user):**
- Best architecture wins even if current GUI is replaced.
- Fully private and local/offline-capable operation.
- Black-box deployment posture (no cloud dependency, minimal data exposure surface).

**Option matrix (planning verdict):**

| Option | Fit | Strengths | Risks | Verdict | Primary source |
|---|---|---|---|---|---|
| **A. PySide6/Qt desktop rewrite** | **High** | Native desktop UX, mature widget ecosystem, Python-first integration, strong local/offline packaging path | Medium rewrite effort from current CustomTkinter pages | **Primary candidate** | Qt for Python docs (https://doc.qt.io/qtforpython-6/) — confidence: medium (broad landing doc, not a single pinned subsection) |
| **B. Tauri v2 + local backend** | **High** | Explicit trust boundaries, capability-scoped IPC, constrained frontend surface, modern UI possibilities | Highest migration complexity (Rust + web frontend + bridge redesign) | **Secondary candidate for maximum hardening** | Tauri v2 Security (https://v2.tauri.app/security/) — confidence: high (primary vendor security doc, version-pinned) |
| **C. Keep current CustomTkinter and refactor** | Medium | Lowest immediate churn, preserves existing behavior | Harder to achieve large UX leap; legacy complexity remains | **Fallback if migration risk is unacceptable** | Tkinter threading model (https://docs.python.org/3/library/tkinter.html#threading-model) — confidence: high (official Python stdlib reference) |
| **D. Electron rewrite** | Medium | Rich ecosystem and tooling | Larger attack surface and strict hardening burden per Electron security checklist | **Not preferred unless web stack requirement becomes dominant** | Electron Security (https://www.electronjs.org/docs/latest/tutorial/security) — confidence: high (primary vendor security doc) |
| **E. Local web UI frameworks (NiceGUI/Flet/Gradio/Open WebUI style)** | Medium | Fast iteration, browser-based flexibility, local hosting possible | Adds local HTTP surface by default; black-box posture requires additional controls | **Good as optional operator console, not primary secure desktop shell** | NiceGUI docs (https://nicegui.io/documentation), Flet docs (https://flet.dev/docs/), Gradio sharing guide (https://gradio.app/guides/sharing-your-app/), Open WebUI (https://github.com/open-webui/open-webui) — confidence: medium (vendor docs, generic landing) |

**Architecture implications from research (decision-relevant):**
- [high] Tauri enforces a core/frontend trust boundary via IPC and capability configuration; frontend only reaches system resources through explicitly exposed commands — Tauri v2 Security (https://v2.tauri.app/security/).
- [high] Electron can be secured, but the default power model requires continuous hardening discipline (context isolation, sandboxing, IPC sender validation, navigation limits, no untrusted remote content) — Electron Security (https://www.electronjs.org/docs/latest/tutorial/security).
- [high] Tkinter calls from any Python thread are dispatched via an event posted to the interpreter's event queue (thread-safe by design), but because the event loop is single-threaded, event handlers that do long work block every other event until they return; long-running tasks must run off the event handler (timer slicing or a worker thread) rather than inside a button callback — Python docs, Threading model (https://docs.python.org/3/library/tkinter.html#threading-model).
- [medium] Web-first local UIs (Gradio/NiceGUI/Flet/Open WebUI patterns) typically run a localhost server; black-box posture depends on strict bind/auth/network policy rather than framework default alone — vendor docs listed above; claim is inferred from combined reading, not a single pinned statement.

**Source confidence legend:** high = primary vendor/official docs directly stating the claim; medium = vendor docs supporting the claim but claim partly inferred; low = secondary/blog sources only (none used here).

**Decision gate (must pass before rewrite start):**
- **Security gate:** chosen stack can run fully offline, bind locally by default, and expose only explicit command/API surfaces.
- **UX gate:** can support Forge-scale multi-page workflows without regressions in responsiveness.
- **Migration gate:** incremental coexistence plan exists (old GUI + new shell during transition).
- **Packaging gate:** Windows-first packaging and update strategy is defined and testable.

**Execution plan (planning only — no code yet; user approval required before any step starts):**

**Hard gates (binary pass/fail, scored separately from the rubric):**
- **G1 — Offline-by-default:** packet capture during a 10-minute idle + one full FORGE workflow shows zero outbound connections other than loopback. Tool: `pktmon` (built into Windows 10/11) or Wireshark if installed. Fails → stack is eliminated regardless of other scores.
- **G2 — No remote update / telemetry by default:** updater disabled or pinned to local-only feed; no analytics SDK in the dependency tree.
- **G3 — UI does not freeze under model load:** during a synthetic 30-second training step running on a worker thread, the UI records no frame stall longer than baseline on the same workload (primary), with a stretch absolute target of < 100 ms. **Measured against the CustomTkinter baseline in Phase 0 too** so "beat baseline" is a real comparison, not an absolute guess. (R4-3: previous pure-100 ms gate was either vacuous or eliminated the incumbent too — primary/stretch pattern instead.)

**Phase 0 — Freeze scope, measure baseline, define contract skeleton (no UI framework code).**
- **GUI-ARCH-0-prep:** Create `information/gui/` folder (does not exist today — verified by directory listing). All Phase 0 docs land there.
- **GUI-ARCH-0a (decision doc):** Write `information/gui/ARCH_DECISION.md` capturing constraints (local / offline / black-box), non-goals (no cloud, no telemetry, no remote auto-update, no analytics SDKs), and current pain points in CustomTkinter (single-threaded Tcl event loop: handlers that do long work block every other event until they return — see https://docs.python.org/3/library/tkinter.html#threading-model — hand-rolled theming, per-widget rebuild cost).
- **GUI-ARCH-0b (baseline):** Measure the **current CustomTkinter GUI** against every rubric metric below and record the numbers in `information/gui/BASELINE.md`. This anchors all POC targets to "beat what we already have", not to arbitrary guesses.
- **GUI-ARCH-0c (service contract skeleton — NOT full migration):** Create `enigma_engine/services/` (new module; does not exist today — `enigma_engine/api/` is already owned by HTTP `server.py` and must not be reused). Populate with an interface-only skeleton covering the 5–10 most-called entry points the current GUI reaches into `core/*` (identify via `grep "from enigma_engine.core" enigma_engine/gui/` — verified this pass, 30+ call sites exist across `desktop.py`, `gui_cmd_page.py`, `gui_forge.py`, `gui_forge_adaptive.py`, `gui_docs_page.py`, and others). The skeleton delegates to `core/*` internally so behavior is unchanged. **No GUI callers are migrated in Phase 0.** Purpose of the skeleton: give Track A / Track B POCs a concrete import surface to target, and make the full per-page migration in Phase 4 incremental instead of a big-bang rewrite.
- **GUI-ARCH-0d (page inventory):** Enumerate actual GUI surfaces from `enigma_engine/gui/` filesystem (verified this pass). Split into two lists because they have different migration risk:
  - **User-facing pages (16 modules — each gets a classification row):** `desktop.py`, `gui_cmd_page.py`, `gui_docs_page.py`, `gui_forge.py`, `gui_forge_adaptive.py`, `gui_forge_advanced.py`, `gui_forge_models.py`, `gui_forge_new_modes.py`, `gui_forge_queue.py`, `gui_forge_tools.py`, `gui_forge_training.py`, `gui_mods.py`, `gui_mod_page.py`, `gui_pages.py`, `gui_pages_config.py`, `gui_pages_forge.py`. For each: name, entry function, current complexity (LOC + widget count), classification (must-have v1 / v2 / drop).
  - **Support modules (7 — migrate as infrastructure, not per-page):** `widgets.py`, `themes.py`, `scanners.py`, `media.py`, `gui_logic.py`, `gui_logic_chat.py`, `gui_logic_media.py`. These are not pages and do not get classification rows; they get a single "port strategy" note each (direct port / rewrite / drop).
  - **Project-goal drift check (binary pass/fail):** for every proposed v1 page, confirm the page does not expose user controls that conflict with the core project constraints "personality from training, not the user" and "blackbox". Any widget that lets the user hand-author the AI's personality, identity, or internal state is flagged for removal or gating before cutover; any widget that exposes raw model internals beyond what is needed for training/debugging is flagged the same way. Output: `information/gui/PAGE_INVENTORY.md` with both lists plus a drift-check appendix.
- **Exit criteria:** `information/gui/` folder exists, four docs + `enigma_engine/services/` skeleton merged and reviewed, drift-check appendix empty or every flagged widget has a disposition, test suite green, no GUI behavior change. No framework picked yet.

**Phase 1 — Two-track bake-off POC (sequential, time-boxed).**
- Rationale: docs alone cannot decide A vs B. One complex page built in each candidate stack is the cheapest honest signal.
- **Time-box:** 5 working days per track, **sequential not parallel** (single builder). Total bake-off budget is 10 working days plus measurement and writeup. If a track cannot produce a running **CONFIG-page prototype** calling the Phase 0 service skeleton in 5 days, it is dropped. (R4-1: bake-off page changed from FORGE to CONFIG — FORGE has 12 modes / 6 collapsible tool sections / paned layout, so making it the primary POC scope means tracks die on page complexity, not on stack fit. FORGE becomes an in-budget stretch goal that scores bonus on the "Theming & layout headroom" metric if reached.) Track B (Tauri) is genuinely harder to stand up in 5 days because of Rust + sidecar + capabilities wiring; the time-box may eliminate it on effort alone — that is an acceptable signal, not a bug. Build Track A first (closer to current Python stack), then Track B.
- **GUI-ARCH-1a (Track A — PySide6/Qt):** Throwaway CONFIG-page prototype in PySide6 calling the Phase 0 `enigma_engine/services/` skeleton where covered, and `enigma_engine/core/*` directly where thin. The POC is explicitly allowed to import `core/*` — it is throwaway code. Stretch goal: also stand up FORGE within the 5-day budget; reaching it scores bonus on the "Theming & layout headroom" metric. Package with PyInstaller **for Phase 1 measurement only** (this is a POC shortcut; the final Phase 3c tradeoff still chooses PyInstaller vs Nuitka). Source: https://doc.qt.io/qtforpython-6/.
- **GUI-ARCH-1b (Track B — Tauri v2 + local Python backend as sidecar):** Same CONFIG-page prototype in Tauri v2 with a minimal frontend (vanilla HTML/CSS/JS; no heavy JS framework for the POC) and a Python backend bundled as a sidecar binary (Tauri's documented pattern for Python — https://v2.tauri.app/develop/sidecar/ — explicitly mentions pyinstaller-bundled Python CLI/API servers as the supported use case). The sidecar re-exports the same service skeleton over loopback with an auth token. Capabilities configured to the exact command whitelist per https://v2.tauri.app/security/. FORGE is the same in-budget stretch goal as Track A.
- **Scoring rubric (filled in at end of phase, each track scored vs baseline from Phase 0):**

  | Metric | Weight | Target |
  |---|---|---|
  | Cold start (shell ready to accept input) | 15% | primary: ≤ baseline; stretch: < 2 s absolute |
  | Page-switch latency (to CONFIG; FORGE if reached) | 15% | primary: ≤ baseline; stretch: < 200 ms absolute |
  | Idle RAM, shell process only (excluding torch + model weights) | 15% | primary: ≤ baseline × 1.5; stretch: ≤ 300 MB absolute |
  | Packaged install size, **shell + UI runtime only** (explicitly excludes torch/CUDA/model weights, which live alongside the app in a shared folder) | 10% | primary: ≤ baseline × 1.5; stretch: ≤ 200 MB compressed absolute |
  | Dev velocity (hours to parity on one page) — **soft signal only; familiarity bias disclosed** | 10% | lower is better |
  | Packaging/update story on Windows (reproducible build, no remote calls) | 15% | documented + reproducible |
  | Theming & layout headroom for future pages | 10% | subjective, recorded with examples |
  | **User preference (side-by-side use of both prototypes)** | 10% | user picks after hands-on |

  Weights sum to 100%. Gates G1/G2/G3 are separate and binary.
- **Exit criteria:** Both POCs either pass all three gates or are eliminated. Rubric filled in. Baseline beaten or reason-why-not documented. Recommendation written in `information/gui/BAKE_OFF_RESULT.md`.

**Phase 2 — Decision and contingency.**
Decision rule, applied in order:
1. Any track failing a gate (G1/G2/G3) is eliminated regardless of score.
2. Among remaining tracks, highest weighted score wins.
3. If scores are within 10 points, user preference (metric row 8) is the tiebreaker.
4. If **only one** of A/B remains, it wins provided it also beats the Phase 0 baseline on at least Cold start, Page-switch, and Idle RAM. If it does not beat baseline on those three, fall back to Option C.
5. If **neither** A nor B remains, fall back to Option C (keep CustomTkinter + refactor) but still land the `enigma_engine/services/` contract — it is not wasted effort.
6. **Wall-clock abort:** if Phase 1 exceeds **15 working days** (start of Track A POC to filled rubric), stop. Default to Option C regardless of rubric state. (R4-5: prevents the bake-off from drifting into a multi-month effort — the 10-day time-box covers normal execution; +5 day buffer covers slip; past that the project has lost more than the rewrite is worth.)
- **Exit criteria:** Single chosen stack + written rationale referencing which rule fired + contingency logged.

**Phase 3 — Migration blueprint (still no mass rewrite).**
- **GUI-ARCH-3a (coexistence):** Legacy CustomTkinter stays shippable. `Launch Enigma.bat` reads a `gui_mode` key from `data/gui_settings.json` and dispatches on its value: `legacy` (default during Phase 3) launches the CustomTkinter shell, `next` launches the new shell. No hardcoded `--gui` / `--gui-next` flag in the .bat; the flag still exists on `run.py` for direct CLI use, but the .bat picks one mode based on the settings file so users get a single entry point during coexistence. (R4-6: the .bat is the real Windows entry point — routing through `gui_settings.json` keeps the user-visible launcher stable across the cutover.) After Phase 4 final cutover, default flips to `next`.
- **GUI-ARCH-3b (cutover order):** Driven by PAGE_INVENTORY. Low-risk pages first (Logs, Diagnostics, Config), Chat mid, FORGE last. Each page has an acceptance checklist (feature parity vs legacy + no regression in existing pytest suite).
- **GUI-ARCH-3c (packaging):** Windows-first installer, reproducible build from source, updater off or local-only. For Track A: decide PyInstaller vs Nuitka with a written tradeoff note (neither is in the project today). For Track B: Tauri bundler config reviewed for any outbound defaults.
- **GUI-ARCH-3d (rollback):** Every page cutover lands behind a feature flag in `data/gui_settings.json` so any page can revert independently.
- **Exit criteria:** Blueprint doc committed.

**Phase 4 — Incremental cutover + service-contract finish + final collapse.**
- **GUI-ARCH-4a..n (per GUI page):** One page per cutover PR. Each PR does three things together: (1) migrate the page to the chosen stack, (2) replace that page's direct `from enigma_engine.core.*` imports with calls through `enigma_engine/services/` (extending the skeleton as needed), (3) port the page's tests. Each PR runs `ruff check enigma_engine/ tests/` + `python -m pytest tests/ -v`, re-runs gates G1/G2/G3 on the merged shell, user smoke-tests the new page, and **flips the per-page rollback flag in `data/gui_settings.json` off, relaunches, verifies the legacy page still renders and its own test suite still passes, then flips the flag back on and re-verifies the new page. Untested rollback = no rollback.** (R4-4)
- **GUI-ARCH-4-api (API server migration):** Once the service skeleton is substantial (roughly half the GUI pages migrated), port `enigma_engine/api/server.py` to call `enigma_engine/services/` instead of `core/*` directly. This is NOT a GUI change but it is part of the same contract cleanup and must happen before legacy removal — otherwise `server.py` remains the last direct `core/*` consumer and the “single service contract” claim is false. Tests: existing API tests must still pass with no behavior change.
- **GUI-ARCH-4-final:** When all v1 pages are migrated AND `api/server.py` is on the service contract, `--gui-next` becomes default, legacy `--gui` kept for one release as emergency rollback, then legacy code path + feature flags removed, `GUI_REFERENCE.md` + `Launch Enigma.bat` updated, final decision archived in `information/gui/ARCH_FINAL.md`. At this point no GUI or API code imports `enigma_engine.core.*` directly; everything routes through `enigma_engine.services`.
- **Exit criteria:** Legacy GUI removed, all v1 pages on the new shell, `api/server.py` migrated, full test suite green, gates re-verified on shipped build.

**Risks logged up-front:**
- Track B (Tauri) carries Rust + JS + Python tri-language complexity. Mitigation: frontend stays minimal (no heavy JS framework), IPC surface bound to the same service contract Track A uses, time-box enforced.
- Track A (PySide6) with PyInstaller can produce large binaries and has known issues with dynamic imports. Mitigation: pin deps, validate bundle contents, consider Nuitka as alternate (decided in Phase 3c).
- Familiarity bias in dev-velocity metric: whichever stack the builder knows better will score higher on velocity for reasons unrelated to the stack's suitability. Mitigation: weight is only 10% and metric is flagged "soft signal".
- Option E (local web UIs) is excluded from primary candidacy because it requires additional bind/auth/network policy work to satisfy the black-box posture. It can still ship later as an opt-in operator console, never as the default shell.

**Not doing (explicit non-goals):**
- No cloud sync, no telemetry, no remote update server, no analytics SDKs, no share-mode web UI as the default surface.
- No rewrite before the bake-off POC is complete and reviewed.
- No partial cutover without a rollback flag.
- No changes to `enigma_engine/core/*` driven by GUI work — the service contract adapts to core, not the other way around.

**Honest total schedule estimate (solo builder, no parallelism):**

| Phase | Wall-clock estimate | Notes |
|---|---|---|
| Phase 0 (prep + baseline + services skeleton + page inventory) | 1–2 weeks | Docs + measurements + stub module. No framework commitment. |
| Phase 1 (Track A POC, then Track B POC) | ~2 weeks sequential (5 working days each + measurement/writeup) | Track B may be dropped on 5-day time-box; then Phase 1 is ~1 week. |
| Phase 2 (decision) | 1–2 days | Fill rubric, apply decision ladder, pick stack. |
| Phase 3 (migration blueprint + packaging + rollback) | ~1 week | Written plan only, no mass rewrite. |
| Phase 4 (incremental per-page cutover + api/server.py migration + legacy removal) | **months** — one PR per page, tests green each time | Real migration work. Pace set by per-page risk, not by calendar. |

Total: Phase 0–3 is roughly 5–6 weeks of dedicated work before legacy code starts being retired in Phase 4. Numbers assume no other high-priority item pulls focus. If N-6/N-9/N-10 training work is active, multiply by 2x and treat GUI as background.

**Round-4 open items — ALL CLOSED Pass 156m (doc-only fixes applied to plan above):**

| # | Item | Fix | Severity | Status |
|---|---|---|---|---|
| R4-1 | Bake-off page is FORGE — 12 modes, 6 collapsible tool sections, paned layout. If POC can't stand it up in 5 days, track dies on scope not fit. | Change Phase 1 POC target to a **mid-complexity page (CONFIG)** as the primary bake-off scope. FORGE becomes a stretch-goal inside the 5 days; reaching it scores bonus on the "Theming & layout headroom" metric. | **Medium — do first.** One sentence, protects the whole bake-off. | **DONE Pass 156m.** |
| R4-2 | Rubric "Idle RAM ≤ 300 MB" and "Install size ≤ 200 MB compressed" are absolute targets with no baseline measurement — same bug class as round 3 caught on cold-start/page-switch. | Rewrite both rows to the primary/stretch pattern: "primary: ≤ baseline × 1.5; stretch: absolute N". Numbers fill in from Phase 0b baseline. | Medium | **DONE Pass 156m.** |
| R4-3 | Gate G3 "100 ms frame stall" is arbitrary. CustomTkinter baseline may already fail 100 ms during model load → gate is either vacuous or eliminates the incumbent too. | Re-scope G3 as "no worse than baseline on the same workload" plus stretch absolute 100 ms. Same primary/stretch pattern as R4-2. | Low — same class as R4-2. | **DONE Pass 156m.** |
| R4-4 | Phase 4 per-page acceptance has feature-flag rollback **defined** but no **test** that rollback actually works. Untested rollback = no rollback. | Add one bullet to the per-page checklist: "Flip flag off, relaunch, verify legacy page renders + passes its own test suite. Flip flag back on and re-verify." | Medium | **DONE Pass 156m.** |
| R4-5 | Decision ladder has no wall-clock abort. If Phase 1 bogs down to 3+ weeks, plan has no "stop and fall back to Option C" trigger. | Add decision rule 6: "If Phase 1 exceeds 15 working days, stop. Default to Option C regardless of rubric state." | Low | **DONE Pass 156m.** |
| R4-6 | `Launch Enigma.bat` is the real Windows entry point but Phase 3a only mentions `--gui` / `--gui-next` flags. During coexistence the .bat must pick one. | Phase 3a: the .bat reads a `gui_mode` key from `data/gui_settings.json` (default `legacy` during Phase 3, `next` after Phase 4 cutover is complete). No hardcoded flag in the .bat. | Low | **DONE Pass 156m.** |
| R4-7 | After R4-1..R4-6 land, plan is execution-ready. | Stamp the close across the canonical docs with a one-line note: "Plan round-4 audit closed. Phase 0 unblocked." | Low — bookkeeping. | **DONE Pass 156m.** |

After R4-7: Phase 0 is the real next action. No more audits.

**First action when the user says "go":** Phase 0 only — produce the four docs (ARCH_DECISION, BASELINE, PAGE_INVENTORY) and the `enigma_engine/services/` contract stub. No UI framework work until those are reviewed.

---

## Pass 141 Decision Queue (next code session picks one)

All candidates below are ready to implement — spec is written, code locations are known, test pattern is clear. User to pick which lands first. No work started yet. Candidates are ordered by blocking-power on the N-6 → N-9 → N-10 training pipeline, not by size.

| # | Candidate | Scope | Blocks | Test Pattern | Est. Size |
|---|-----------|-------|--------|--------------|-----------|
| ~~A~~ | ~~**Tok-2** byte-level BPE fallback~~ | ~~[enigma_engine/core/bpe_tokenizer.py](enigma_engine/core/bpe_tokenizer.py) + [enigma_engine/core/advanced_tokenizer.py](enigma_engine/core/advanced_tokenizer.py)~~ | **DONE (Pass 149).** Default flipped on; AdvancedBPETokenizer gained byte-mode parity; Rust already correct. | — | — |
| ~~B~~ | ~~**GRPO-2** `reward_functions.py`~~ | ~~New file: `enigma_engine/core/reward_functions.py`. Rule-based rewards: math pass/fail (numeric parse + compare), format check (looks-for-`<think>` + answer delimiter), code exec (subprocess sandbox). Wire into [gui_forge_new_modes.py L2250](enigma_engine/gui/gui_forge_new_modes.py#L2250) replacing `RewardModel.score()` for reasoning mode.~~ | **DONE (Pass 142).** Shipped + 4 tests green. | — | — |
| ~~C~~ | ~~**Eval-1** GSM8K benchmark~~ | ~~Add `run_gsm8k_benchmark()` to [enigma_engine/core/training_evaluation.py](enigma_engine/core/training_evaluation.py). Load HF `openai/gsm8k` (already cacheable offline), few-shot prompt, numeric answer parse from final line. Expose via `--benchmark gsm8k` in [run.py](run.py).~~ | **DONE (Pass 150).** `parse_final_number` + `load_gsm8k` + `run_gsm8k_benchmark` shipped, 8-shot Wei et al. CoT prefix, exact-match scoring, offline-first JSONL loader with clear download recovery message. CLI: `python run.py --benchmark gsm8k --model models/<name>.pth`. 14 tests green. | — | — |
| ~~D~~ | ~~**Vision-1b** projection upgrade~~ | ~~[enigma_engine/core/model.py](enigma_engine/core/model.py) `self.vision_projection = nn.Linear(...)` → `nn.Sequential(nn.Linear, nn.GELU, nn.Linear)` per LLaVA-1.5 (arxiv:2310.03744). Keep input/output dims identical.~~ | **DONE (Pass 151).** `nn.Sequential(Linear(vh,d,bias=True), GELU, Linear(d,d,bias=True))` per HF `LlavaMultiModalProjector`. State-dict keys changed; safe (no vision checkpoints exist). 6 tests. | — | — |
| ~~E~~ | ~~**MTP-2a** reduce `n_predict_heads` 2→1 + weight-tie~~ | ~~[enigma_engine/core/model_presets.py L129](enigma_engine/core/model_presets.py#L129)~~ | **SUPERSEDED — took MTP-2b path Pass 149.** Default flipped to 0 entirely (Medusa retired in favor of EAGLE-2). | — | — |
| F | **Code-6** FORGE vision-projection training mode | New FORGE mode wired in [enigma_engine/gui/gui_forge_new_modes.py](enigma_engine/gui/gui_forge_new_modes.py) + [enigma_engine/gui/gui_pages_forge.py](enigma_engine/gui/gui_pages_forge.py). Freeze transformer, train only `vision_projection` + CLIP-side adapter on LLaVA-Pretrain (558K) → LLaVA-Instruct-150K (665K). | Vision specialist pipeline. Projection is wired but untrainable. | Fail test: after 1 training step, `vision_projection.weight.grad` is not None and transformer weights `.grad` are None. | Large — new mode + data loader + GUI radio-card entry. Depends on D (Vision-1b) for best parity. |
| G | **Personality-5** personality injection wiring | [enigma_engine/gui/gui_forge_new_modes.py L1259](enigma_engine/gui/gui_forge_new_modes.py#L1259) — category exists, data generation exists, but no consistency loss or per-profile regularizer. Gate behind user opt-in (profile-scoped, not global). | R-PERSONALITY-1. | Fail test: two generations with same profile + temperature=0 produce consistent first-person pronoun + stated-value alignment. | Medium — needs consistency metric before loss wiring. |
| H | ~~**EWC-1** wire `core/ewc.py` into FORGE SFT/dialogue~~ | **CLOSED Pass 156i3 — WONTFIX (superseded by LoRA-per-specialist path).** [ewc.py](enigma_engine/core/ewc.py) library stays for rare-case future use; LoRA freezes base weights so forgetting is physically impossible. See **LoRA-1** (P2) for the replacement work. | — | — | — |
| ~~DET-2~~ | ~~**DET-2** full bitwise GPU reproducibility~~ | **DONE Pass 156i3.** `set_training_seed(seed, deterministic=False)` gained opt-in `deterministic` kwarg (sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` + `torch.use_deterministic_algorithms(True, warn_only=True)`). `TrainingConfig.deterministic` field default False; all 8 train_* siblings forward the flag. `warn_only=True` keeps MoE `index_add_` from crashing. 4 new tests; suite 2379/9. **Out of scope:** DataLoader worker_init_fn (Enigma single-threaded), `Trainer.__init__` DRY refactor (separate design row). | — | — | — |

**Tie-breaker guidance:** A (Tok-2) has the broadest downstream blast radius — every retrain from here on benefits. E (MTP-2a) is the smallest win and frees 49-98M params immediately. B (GRPO-2) is the hard prerequisite for any reasoning RL work.

---

## Priority Index (most → least important)

Full backlog sorted by urgency. Each item links to the detailed entry further down.
Items already done or irrelevant are omitted. Research-only items (no implementation
yet possible) are grouped at the bottom of P2/P3.

### 🔴 P0 — CRITICAL (blocks next pre-training restart)

| # | Item | Why P0 |
|---|------|--------|
| 1 | ~~**CRIT-6 / D-1 / D-2** — Collect DCLM + FineMath + The Stack v2~~ | **DONE (Pass 136).** `--dclm`, `--finemath`, `--code` flags in [collect_pretraining_data.py](collect_pretraining_data.py). |
| ~~2~~ | ~~**Tok-2** — Add byte-level fallback to BPE tokenizer~~ | **DONE (Pass 149).** Byte-mode default ON in `BPETokenizer` + full parity in `AdvancedBPETokenizer`. Rust already supported. CJK/emoji/Arabic now roundtrip without `<unk>`. |
| 3 | **D-4** — Reasoning mid-training phase (OpenThoughts3) | Gap between "follows instructions" and "actually reasons." Must happen between N-6 and N-9 for Approach 1 to work. |

### 🟠 P1 — HIGH (before N-9 SFT / alignment)

| # | Item | Why P1 |
|---|------|--------|
| ~~4~~ | ~~**GRPO-2 BUILD** — Write `reward_functions.py` (math accuracy + format + code exec + LLM judge)~~ | **DONE (Pass 142).** [reward_functions.py](enigma_engine/core/reward_functions.py) shipped. |
| ~~5~~ | ~~**GRPO-4** — Block neural reward path for reasoning; route to rule-based reward~~ | **DONE (Pass 142).** GRPO branch in [gui_forge_new_modes.py L2229-L2260](enigma_engine/gui/gui_forge_new_modes.py#L2229-L2260) skips neural Phase 1 and uses `reasoning_reward`. |
| 6 | **Approach 3 POC** — Distill reasoning traces from Qwen3-30B-A3B (N-9.5a/b/c/d) | Proof-of-concept for whether multi-model architecture is viable. 3-4 week experiment. Informs Approach 1 vs Approach 2 decision. |
| ~~7~~ | ~~**Eval-1 / D-10** — Implement GSM8K benchmark in `training_evaluation.py`~~ | **DONE (Pass 150).** `run_gsm8k_benchmark()` + `parse_final_number()` + `load_gsm8k()` shipped. CLI: `--benchmark gsm8k`. Offline JSONL loader (no network at benchmark time). 14 tests in [tests/test_evaluation.py](tests/test_evaluation.py). |
| ~~8~~ | ~~**D-11** — Add SmolTalk2 SFT data (reasoning traces + tool calling)~~ | **DONE Pass 156i8 (consumer wiring).** Pass 155+155b shipped the fetcher; Pass 156i8 closes the consumer-side gap by emitting `data/finetune/combined_finetune.txt` in canonical `User: \n\nAssistant: ` format alongside the combined JSONL. The existing SFT plain-text reader consumes it with zero training-side change. 1,999-pair corpus on disk (~60 MB). See **D-11b** for FORGE picker UX follow-up. |
| ~~8b~~ | ~~**D-11b** — FORGE file picker default for finetune data~~ | **DONE Pass 156i9.** New `scanners._pick_default_training_file()` helper prefers `data/finetune/combined_finetune.txt` over `data/training.txt`. Wired into FORGE Basic page `train_data_var` init. 4 new tests including adversarial-ordering. |
| ~~9~~ | ~~**Code-6** — FORGE training mode for vision projection (frozen transformer)~~ | **DONE (closed Pass 156q, work landed across Pass 151-156).** `Image` mode card → `_start_vision_training` → `Trainer.train_vision()` with LLaVA Stage-1 defaults (`freeze_backbone=True`, `unfreeze_text_layers=0` — projection-only). Sub-rows V-1 through V-8 all closed: Vision-1b 2-layer MLP+GELU projection (Pass 151), V-4 OOM/crash heuristics, V-5 `collect_vision_data.py` LLaVA-Pretrain fetcher (Pass 156c), V-7 abort-summary, V-8 vision-encoder load path in inference (Pass 156b). Stage-2 unfreeze-last-N-text-layers knob is in the trainer but not yet exposed in the GUI — opened as **Code-6b** below. |
| ~~9b~~ | ~~**Code-6b** — Expose `unfreeze_text_layers` in FORGE Image mode~~ | **DONE Pass 156r.** New numeric input under the vision-encoder-size dropdown (default 0 = LLaVA Stage-1 projection-only). `_start_vision_training` reads with try/except validation (negative → 0, >64 → 64, bad input → 0, all warned), forwards as `trainer.train_vision(unfreeze_text_layers=N, ...)`, logs in summary as `"N text layers"` or `"projection only (Stage-1)"`. Two structural tests gate the var presence + literal kwarg forward. |
| 10 | **Personality-5 cluster** — Implement personality-in-weights plan + identity/roleplay separation | **PARTIAL.** Personality-3 boundary fix DONE Pass 156y (`AIProfile.personality` reframed as roleplay-only, `is_roleplay()` signal, `assistant` base cleaned). Personality-3b (canonical role-template cleanup) + Personality-4 (identity-vs-roleplay design call + first end-to-end consumer) DONE Pass 156z (3 disk JSONs + DEFAULT_PROFILES cleaned, `apply_profile_to_engine` logs roleplay branch, design call: empty personality = base/task overlay, populated = roleplay character). Open: Personality-3b for legacy user profile `not_for_you_hahaha.json` (deferred — user-driven migration), Personality-5 BUILD (operational — run FORGE distillation), Row G (consistency loss). |
| 11 | **AutoResearch-2** — Self-initiated research (`<search>` tag OR uncertainty post-check) | Model currently cannot say "I don't know, let me look it up." Linked to GRPO-4. |

### 🟡 P2 — MEDIUM (standard backlog, no blocking order)

| # | Item | Notes |
|---|------|-------|
| 12 | ~~**D-9** — APO alignment mode (15-line DPO loss variant)~~ | **DONE Pass 156j (library).** `_apo_zero_loss` static + `_resolve_preference_loss` registry + `loss_type="dpo"\|"apo_zero"` kwarg on `train_dpo`. 8 behavioural tests including chosen-rejected-independence (the key APO property) and dispatch-actually-routes. See **D-9b** for FORGE GUI surface. |
| 12b | ~~**D-9b** — FORGE radio card + dispatcher for APO-zero alignment~~ | **DONE Pass 156k.** APO radio card added to alignment row alongside GRPO/ReMax/SimPO/ORPO. `_start_apo_training` thin wrapper delegates to refactored `_start_dpo_training(loss_type="apo_zero")` which forwards the kwarg to `trainer.train_dpo`. Status bar + logs + `_save_training_run` label parametrized via `algo_label`. 5 structural GUI tests; behavioural routing proven by Pass 156j's sentinel-mock dispatch test. |
| 13 | **D-19** — Strong-to-weak distillation from Qwen3-30B before GRPO | 10× cheaper than GRPO at small scale. Revisit at N-9. |
| 14 | **N-14** — Dense semantic memory (FAISS) to replace TF-IDF RAG | Prep for vision+retrieval. |
| 15 | ~~**EWC-1** — Wire `core/ewc.py` into FORGE SFT/dialogue training path~~ | **CLOSED Pass 156i3 — WONTFIX (superseded by LoRA-per-specialist).** Library kept for rare-case full-weight specialization; LoRA path makes forgetting impossible by freezing base weights. See **LoRA-1**. |
| 15b | ~~**LoRA-1** — Wire LoRA-per-specialist into FORGE SFT/dialogue~~ | **DONE Pass 156p (training-side).** Explicit `LoRA` foundation mode card forces adapter training on any model size; rank/alpha widgets and `_start_lora_training()` already wired pre-pass. Adapter saved to `models/checkpoints/{name}_lora.pth`. **Inference-side adapter swap deferred → LoRA-1b** below (load `*_lora.pth` at chat time, hot-swap, multi-adapter UI). |
| 15c | **LoRA-1b** — Inference-side adapter loading and swap (foundation) | **DONE Pass 156s.** PEFT-directory-only save format ([lora_utils.py](enigma_engine/core/lora_utils.py) — manual-fallback `.pth` branch deleted), `scan_lora_adapters(base_model_path=None)` scanner with stem-filtered base matching, `EnigmaEngine.apply_adapter(path)` + `clear_adapter()` + `active_adapter` field with PEFT wrapping and KV-cache-clear on swap, `route_assignments["chat_adapter:<stem>"]` persistence with auto-restore on `_on_model_loaded` and orphan purge when adapter is deleted off-disk. Per-base scoping prevents cross-base adapter mis-application. 6 new tests (3 behavioural scanner + 3 structural double-gates). UX surfaces (MODELS-page list, profile auto-apply, branch markers) → 15d below. |
| ~~15d~~ | ~~**LoRA-1b UX (Pass 156t)** — MODELS-page list, profile field, branch marker~~ | **DONE Pass 156t.** MODELS-page per-card LoRA section with Apply/Clear buttons and base-stem filter, `AIProfile.adapter: Optional[str]` field with apply-or-clear-on-load semantics, legacy `_lora.pth` migration script. Branch marker generalised into Pass 156v Step 1 (`_chat_session_marker` helper). Lazy KV reprefill via `apply_adapter` cache-clear (Pass 156s). |
| ~~15e~~ | ~~**LoRA-1b stacking (Pass 156u)** — multi-adapter weighted stacks~~ | **DONE Pass 156u-A + 156u-B.** PEFT `add_weighted_adapter` engine path (`EnigmaEngine.apply_adapter_stack`), per-base persistence (`route_assignments["chat_adapter_stack:<stem>"]`), mutual-exclusion with single-adapter key, stack-first restore precedence. UI surface: per-row checkbox + numeric weight entry (no sliders, per Dia rules), APPLY STACK button, parse-error collection across all rows. KV invalidation on every weight change. Pass 156u-A2 audit-fixed corrupted-stack-entry resilience and added duplicate-path test. |
| 15f | **Session-1 (Pass 156v)** — unified session-state divider markers | **PARTIALLY DONE.** Step 1 (Pass 156v) shipped `_chat_session_marker` helper + `session_marker` chat tag + adoption at all 5 LoRA adapter-swap success paths. Step 2 (Pass 156v Step 2) extended adoption to model load / unload / RAG enable / RAG disable. **Still open:** profile swap and system-prompt edit have no chat-page surface yet (no widget swaps a runtime persona today — "profile" in code is a forge-page training-data brief, not a chat-runtime persona); add markers when those surfaces are built. Separate concern: model-swap currently calls `_load_model_context(path)` which CLEARS chat history — the marker covers UX visibility but the lazy-reprefill behaviour change (preserve history across model swaps) is a follow-up design decision, not part of Session-1 unification. |
| ~~15g~~ | ~~**Legacy `_lora.pth` migration (Pass 156t companion)**~~ | **CLOSED Pass 156v (no migration needed).** Migration script [migrate_legacy_lora.py](migrate_legacy_lora.py) shipped Pass 156t; zero `_lora.pth` files exist on disk in `models/checkpoints/` or `models/lora_adapters/`. Active code path (Pass 156s+) only writes PEFT directories. Script remains for any future legacy import. |
| ~~16~~ | ~~**N-15** — Constrained decoding (grammar for JSON / tool calls)~~ | **DONE Pass 156z3.** `EnigmaEngine.generate(json_schema=...)` builds `JsonSchemaConstraint` once, threads it through `_generate_text` → `_generate_manual` → `_sample_token` (existing mask hook), drives FSM via `.advance()` per token, early-stops on `is_done`. GGUF path raises `NotImplementedError` (mask never reaches llama.cpp's sampler). Tests +4 (3 wire-site + 1 GGUF rejection); FSM itself already had 5 unit tests from T3-9. **N-15b (next):** expose `json_schema` on chat API endpoint + GUI checkbox so users can reach the feature without writing Python. |
| ~~17~~ | ~~**N-16** — Best-of-N sampling w/ reward model~~ | **DONE Pass 156x.** `EnigmaEngine.generate_best_of_n(prompt, n, reward_fn, *, return_all=False, **gen_kwargs)` runs N candidates, scores via user-supplied `(prompt, response) -> float`, returns highest with first-occurrence tie-break. Validates n>=1 (ValueError), warns on temperature=0+n>1, swallows reward errors with -inf so flaky scorers don't break the batch. 9 behavioural tests via `_FakeSelf` unbound-method pattern. |
| ~~18~~ | ~~**N-19** — External-teacher distillation (text-only, OpenAI-compatible IPC)~~ | **DONE Pass 156z5 (slice 1: offline corpus collector).** Per-user pivot: NO in-process teacher ("complicated to build AI on AI"). Instead: [collect_distill_data.py](collect_distill_data.py) talks HTTP to any OpenAI-compatible `/v1/chat/completions` endpoint (Ollama default `http://localhost:11434/v1`, llama.cpp server, vLLM, our own `run.py --serve`). Stdlib only (urllib). Writes `data/finetune/distill_<tag>.{jsonl,txt}` in the canonical `User: …\n\nAssistant: …` format the existing FORGE Distill mode already consumes — zero training-loop changes. Resumable (skips prompt-keys already in JSONL). Privacy guard: WARNING when endpoint host is not localhost. Failures (HTTP error, unreachable, empty completion) skipped + counted, never abort the run. **+19 tests.** Black-box distillation per the 2024 KD survey ([arxiv 2402.13116](https://arxiv.org/abs/2402.13116)) + Self-Instruct family. **Open follow-ups (slice 2/3, deferred):** Magpie-style empty-prefix instruction synthesis (per-model chat-template handling), GUI "Generate Teacher Corpus" button on the FORGE page, optional top-k logprobs capture (some endpoints expose it — use only if a real white-box use case appears). |
| 19 | ~~**Continuous-1** — Review `BackgroundTrainer` LR (1e-5) + buffer=1000 for continuous SFT safety~~ | **CLOSED Pass 156i4.** Audit caught real silent-drift bug (no NaN/Inf abort \u2192 single bad sample corrupts model permanently with NaN gradients) + OOM exposure (no token-length cap) + claim-vs-test mismatch on replay-buffer eviction. All three fixed; LR=1e-5 + buffer=1000 retained as defaults (sane). See **Continuous-2** for catastrophic-forgetting follow-up. |
| ~~19b~~ | ~~**Continuous-2** — Add anchor-set rehearsal to `BackgroundTrainer._retrain_on_replay`~~ | **CLOSED Pass 156i5.** New `anchor_data_path` kwarg + `_load_anchor_examples()` helper + extended `replay_batch` in `_retrain_on_replay`. Docstring re-framed ("mitigates", not "prevents"). 5 behavioural tests. See **Continuous-3** for GUI surface + default anchor data follow-up. |
| ~~19c~~ | ~~**Continuous-3** — GUI surface + default anchor data for `BackgroundTrainer.anchor_data_path`~~ | **DONE Pass 156i7 (core wiring + default file).** [data/anchor_examples.jsonl](data/anchor_examples.jsonl) ships with 51 curated rows; `ModRouter` auto-wires the path when the file exists. GUI widget for in-app editing/selection deferred to **Continuous-3b** (small UX follow-up). Anchor-only periodic schedule (no recent activity required) deferred to **Continuous-3c** (scheduling design). |
| 19e | ~~**Continuous-3b** — GUI surface for anchor file selection/edit~~ | **DONE Pass 156o.** CONFIG page TRAINING section shows anchor path + row count (or "file missing" / "(none — recent-only)"); BROWSE picks a custom JSONL; USE DEFAULT clears the override. Persisted via `gui_settings.json` key `anchor_data_path`; desktop reads it on boot and forwards to `ModRouter(anchor_data_path=...)`. Restart-to-apply. In-app row editor NOT shipped — file is plain JSONL, edit in any text editor; revisit only if users actually request inline editing. |
| ~~19f~~ | ~~**Continuous-3c** — Anchor-only periodic schedule~~ | **DONE Pass 156w.** Idle-time scheduler in `BackgroundTrainer.run()` queue.Empty branch, gated by new opt-in kwarg `anchor_idle_interval_seconds` (default None = off, 0/negative → off). Helper `_should_run_anchor_idle_replay()` checks all preconditions (interval, anchor path, running, not paused, model+optimizer wired, inference idle, elapsed ≥ interval). Cooldown timer is also reset by regular `_train_batch`-triggered replays so the two paths can't double-fire back-to-back. +9 tests. |
| ~~19d~~ | ~~**Continuous-2a** — Pass 156i5 self-audit findings (logic-eye + claim-vs-test)~~ | **DONE Pass 156i6.** All five findings addressed: (1) sibling-comment drift fixed at [router.py L83](enigma_engine/router.py#L83) and [L470](enigma_engine/router.py#L470); (2) anchor early-out reordered so anchors load before `not replay_buffer` return; (3) WARNING added when anchor file present but yields zero usable rows; (4) legacy-path test strengthened with direct `_load_anchor_examples() == []` assert; (5) cache test added (`test_anchor_file_loaded_only_once_across_replay_passes` — patches `Path.open`, asserts 1 open across 3 replay passes). Suite 2391/9. Note: caller-loop gate `len(replay_buffer) >= batch_size` at [router.py L392](enigma_engine/router.py#L392) still suppresses anchor-only rehearsal during quiet periods — separate design question deferred to Continuous-3. |
| ~~20~~ | ~~**Sched-2** — Change SFT/DPO default warmup from 100 to `max(100, steps // 10)`~~ | **DONE (Pass 152).** [`_effective_warmup()`](enigma_engine/core/training.py#L40) caps warmup at `total_steps // 5` (20%). Wired into all 5 scheduler sites. 8 tests in [TestEffectiveWarmup](tests/test_training.py). |
| 21 | **D-6** — Tokenizer 64K expansion | Must happen BEFORE N-6 restart if at all. After N-6 starts, defer indefinitely. |
| 22 | **R-3 / R-4 / R-5** — Profile MinHash dedup, data pipeline, sequence packing before Rust rewrite | Tokenizer rewrite (R-1/R-2) is done. Others need measurement first. |
| 23 | **D-12** — NoPE hybrid attention (every 4th layer no-RoPE) | Architecture change — only before N-6 if at all. |
| 24 | **ImgGen-4** — SDXL-Turbo / SD-Lightning (4-8 step inference) | Upgrade imagegen mod. |
| 25 | **Audio-5** — MusicGen-small integration | Music generation is a stated project goal; currently TTS-only. |
| 26 | **R-UNPREDICT-1** — Controlled-unpredictability design | **Resolved (Pass 146).** Build follow-up is AutoResearch-2 staged implementation (post-generation uncertainty gate first, inline `<search>` token second). |
| 27 | **Vision-1b** — Upgrade `vision_projection` from single `nn.Linear` to LLaVA-1.5 style 2-layer MLP + GELU | Code gap vs LLaVA-1.5 confirmed this pass. Do alongside Code-6 training wiring. **Pass 148 currency note:** vision encoder target can be upgraded CLIP-ViT-L/14@336 → **SigLIP 2** (Google, arxiv:2502.14786, Feb 2025, 4 sizes 86M/303M/400M/1B) which outperforms SigLIP at all scales across zero-shot, retrieval, VLM transfer, and localization. SigLIP 2 is a drop-in replacement for the frozen encoder; projector design stays LLaVA-1.5 2-layer MLP+GELU. Swap is optional and deferred until after Code-6 training works end-to-end with the current CLIP. |
| 27b | ~~**MTP-2a / MTP-2b** — Fix MTP default~~ | **DONE Pass 149 (MTP-2b path).** Default `n_predict_heads` flipped `2`→0 and inverted comment fixed at [model_presets.py L129](enigma_engine/core/model_presets.py#L129). Saves ~33-49M params per fresh model. Existing zero-guards handled it; explicit `n_predict_heads>0` opt-in still works for Medusa runs. Pass 148 designated EAGLE-2 as the speculative path so Medusa is no longer planned. |
| 28 | **Research queue status** | Original [RESEARCH] backlog items are closed; remaining top items are [BUILD] / [CONFIRMED GAP] implementation work. |

### 🟢 P3 — LOW / DEFERRED (after alignment is done, or parked)

| # | Item | Why deferred |
|---|------|--------------|
| 37 | **D-13 / D-17** — Context length extension (4K → 32K) via RoPE theta scaling + YaRN | Post N-10 only. |
| 38 | **D-20** — Flash Attention 4 SM120 integration | Wait for stable Windows wheel (Q3-Q4 2026). |
| 39 | **Video-1 / 3D-1 / Haptic-1** — Video / 3D / haptic generation | Re-scoped Pass 148 currency check: Wan 2.1 T2V-1.3B (8.19 GB VRAM) and Hunyuan3D-2 (image→3D) now fit 16 GB budget — still deferred by priority, not by feasibility. Haptic has no open dataset. |
| 40 | **D-18** — Aux-loss-free MoE balancing | Only if MoE is enabled — currently not. |
| 41 | **Avatar-1 / Avatar-3 / Avatar-4** — Text-to-animation + training data + output format | Full bone rig exists, pipeline not scheduled given 14-task goal list. |
| 42 | **FSDP / DDP / multi-GPU** | Single-GPU until hardware changes. |
| 43 | **PRM-1** — Process Reward Model training pipeline | DeepSeek paper explicitly calls PRM unsuccessful for RL. Keep code, don't build labeling pipeline. |
| 44 | **Checkpoint browser w/ perplexity comparison** | Nice-to-have after N-7. |
| 45 | **Pre-tokenized binary cache integration** | Script exists ([pretokenize_data.py](pretokenize_data.py)); integrate into training after first N-6 run. |

---

## Status

**Data:** 88.8 GB collected (87.6 GB combined.txt after paragraph dedup). ~26B tokens, 35 tok/param for xl 742M.
**Model:** `xl` (742M params, ~12 GB VRAM with grad ckpt). GPU: RTX 5090, 32 GB total, 16 GB budget.
**Recipe:** GUI Pre-Train → Memory=16 GB → xl. LR 2e-5, batch auto, accum 4, 1-3 epochs.
**Audit:** 148 passes, 515+ findings tracked. Per-pass bug-fix history is archived in [information/history/PASS_HISTORY.md](information/history/PASS_HISTORY.md); per-file fix tracking is in [CODE_REVIEW.md](CODE_REVIEW.md). Open items below. Accepted-risk list at bottom.

---

## Architecture Approaches (April/May 2026 Research)

**Problem Statement:** Enigma's 742M must deliver on 14 broad tasks (vision, audio, 3D, reasoning, code, generation, etc.) but a single LLM can't do all well. Research question: **single model with plugins, or ensemble of specialists?**

**Decision Matrix:**

| Approach | Complexity | Timeline | Risk | Fits Goals | Notes |
|----------|-----------|----------|------|-----------|-------|
| **1. Reasoning-First Ensemble** | Medium | 4-5 months | Low | ✓✓ Excellent | Foundation-first: reasoning model + specialist plugins added later. Matches goal priority. |
| **2. Modular Specialist System** | High | 4-5 months | Medium | ✓✓ Excellent | 5-6 tiny models (200-500M each) + router. Each expert at its task. Most realistic production. |
| **3. Distill from Qwen3-30B** | Low | 3-4 weeks | Very Low | ✓ Good | Fast variants via strong teacher. Proof-of-concept, not final system. |
| **4. Incremental Integration** | Low-Medium | 6-9 months | Very Low | ✓ Good | Add capabilities one at a time. Slowest but safest. |

---

### Approach 1: Reasoning-First Ensemble (RECOMMENDED)

**Core idea:** Pivot 742M into reasoning expert. Add vision/audio/3D as lightweight plugins at inference time.

**Why:** "Learning & reasoning" is goal #1. Test-time compute (QwQ/DeepSeek-R1 style thinking) is the foundation every other task needs.

**Architecture:**
- **Core:** 742M LLM specialized for reasoning (math, code, logic, planning)
- **Vision plugin:** 200M encoder (frozen) + use 742M decoder for VQA
- **Code plugin:** Fine-tuned 742M variant on The Stack v2
- **Router:** Lightweight decision system (what type of task is this?)
- **Generation dispatch:** Invoke SD 1.5 quantized (2GB VRAM) when needed

**VRAM budget:**
- 10GB: Core 742M model
- 3GB: Vision encoder (preload)
- 2GB: Generation model swap space
- 1GB: Router + overhead

**Implementation roadmap:**
1. N-6: Pre-train with reasoning data emphasis (OpenThoughts3, FineMath)
2. N-9/N-10: Instruction fine-tune + GRPO for "thinking before answering"
3. N-11: Train vision encoder on image captioning data
4. N-12: Build router logic (dispatch rules + test on 1000 queries)
5. N-13: Wire plugins (vision → encoder path, generation → SD swap)

**Research needed (R-ARCH-1):**
- [RESOLVED] R-ARCH-1a: DeepSeek-R1 paper (arxiv 2501.12948) read in full. Key findings:
  - **Token format**: `<think>reasoning process here</think><answer>answer here</answer>` — EXACT match to our registered tokens `<think>=4, </think>=5`. No architecture change needed.
  - **RL algorithm**: GRPO (Group Relative Policy Optimization). For each question, sample G outputs from old policy, score all, normalize advantages as `(r_i - mean) / std`. No value head (saves ~50% compute vs PPO). Our `GRPOTrainer` in `rl_training.py` line 2311 already implements this correctly.
  - **Reward design (R1-Zero, the pure RL approach)**:
    - Accuracy reward: rule-based only — math: check `\boxed{}` answer, code: run test cases. Binary pass/fail.
    - Format reward: verify `<think>...</think>` tags present in output.
    - NO neural reward model — explicitly avoided because it leads to reward hacking.
    - Language consistency reward (optional): proportion of target language words in CoT.
  - **Training pipeline (R1-Zero approach, simplest)**:
    - Just GRPO on base model with accuracy + format rewards. No SFT first.
    - Model naturally learns to extend thinking when rewarded — response length grows automatically.
    - Drawback: language mixing, poor readability. Solves reasoning but not UX.
  - **Training pipeline (R1 full, 4 stages)**:
    1. Cold start SFT: ~1,000-5,000 long CoT examples in readable format → fine-tune base model
    2. Reasoning GRPO: same accuracy + format + language consistency rewards, train to convergence
    3. Rejection sampling SFT: generate from RL ckpt, keep only correct ones (~600K reasoning + ~200K general = 800K total), re-train from base
    4. Second GRPO pass: rule-based for math/code + neural reward model for general tasks
  - **CRITICAL for 742M scale**: DeepSeek trained Qwen-32B with GRPO for 10K+ steps → matched QwQ-32B-Preview. But distilling R1 into Qwen-32B with SFT alone → **massively outperformed** RL training. Conclusion: **at 742M, distillation (Approach 3) is far more efficient than GRPO training**. RL can improve a distilled model further but should come AFTER distillation, not before.
  - **Thinking budget**: No hard budget during training. Model naturally allocates more thinking tokens when rewarded. At inference: R1 eval caps at 32,768 tokens. For 742M production: cap think tokens at 2048-4096 (our context is 4096 total — leave ~512 for input + answer). [enigma_engine/core/engine_generation.py](enigma_engine/core/engine_generation.py) needs a `max_think_tokens` parameter added.
  - **Data format for cold-start**: User/Assistant chat with `<think>COT</think><answer>final</answer>` pattern. Summary sentence after `</think>` is optional but improves readability. Generate using Qwen3-30B-A3B as teacher — already available in `models/qwen3-30b-a3b/`.
  - **For Enigma action**: (1) Generate 3,000-5,000 math/logic cold-start examples using Qwen3-30B-A3B FORGE Distillation. (2) SFT on those examples. (3) Optionally: GRPO with accuracy+format reward. File refs: [enigma_engine/core/reasoning.py](enigma_engine/core/reasoning.py) (has `strip_incomplete_think()`), [enigma_engine/core/rl_training.py](enigma_engine/core/rl_training.py#L2311) (GRPOTrainer line 2311), [enigma_engine/core/engine_generation.py](enigma_engine/core/engine_generation.py) (generation loop — needs max_think_tokens).
- [RESOLVED] R-ARCH-1b: LLaVA vision encoder specs (200M size, frozen vs fine-tuned). Verified from primary sources:
  - **Encoder choice:** LLaVA-1.5 uses `CLIP-ViT-L-336px` (arxiv:2310.03744, Sec 3.3 + Appendix).
  - **Freeze policy:** original LLaVA keeps vision encoder frozen in Stage 1 and Stage 2; Stage 1 trains projection only, Stage 2 trains projection + LLM (`theta={W,phi}`) while encoder stays frozen (arxiv:2304.08485, Sec 4.2).
  - **Projector architecture:** LLaVA-v1 uses Linear projector; LLaVA-1.5 baseline upgrade is 2-layer MLP connector (arxiv:2310.03744, Sec 3.3). LLaVA model zoo projector table confirms `MLP-2x` for v1.5 projector variants.
  - **Model size correction:** "200M vision encoder" is not accurate for CLIP-L. `ViT-L-14-336` visual tower is ~304.29M params (OpenCLIP model profile), and full CLIP `ViT-L-14-336` (vision + text) is ~427.94M params.
  - **Enigma decision:** for 16 GB VRAM target, keep CLIP encoder frozen and train projector (+ optional lightweight adapter) first; only consider partial unfreeze after projector convergence and only with explicit benchmark gain.
- [RESOLVED] R-ARCH-1c: Router design patterns (decision tree, learned router, heuristic rules). Recommendation is now explicit: **layered router** = heuristic regex gate first, embedding-similarity classifier second, LLM classifier fallback only for low-confidence cases. This matches our existing Router-2/3 conclusions and keeps latency low while preserving quality.
- [RESOLVED] R-ARCH-1d: How to train router (labeled data: "is this a vision task?"). Best practical recipe from RouteLLM (arxiv:2406.18665 + LMSYS blog):
  - Start with weak supervision labels from logs/heuristics (task tags: vision/code/reasoning/general).
  - Train a small classifier first (BERT-tiny class) for low-latency routing.
  - Add targeted augmentation for out-of-distribution buckets (small curated set is enough to move quality significantly).
  - Keep LLM-based routing as judge/fallback, not primary path.
  - Evaluate router quality-cost curves, not just top-1 accuracy (target is quality at lower expensive-model call rate).

**Timeline:** 5 months (N-6 → N-13)

**Advantages:**
- Aligns with stated priority (reasoning first)
- Validated by Qwen3 paper
- Plugins are optional (still works as reasoning-only model)
- Can test each component independently

**Disadvantages:**
- Plugins are not specialist-trained (vision encoder is borrowed, not optimized for Enigma)
- Router adds latency (query classification step)
- Sharing 742M decoder across tasks means compromises

---

### Approach 2: Modular Specialist System

**Core idea:** Train 5-6 small specialists (200-500M each), router learns to dispatch.

**Specialists:**
1. **Core Router (500M):** Text tasks, decision-making
2. **Vision Specialist (300M):** Image understanding (LLaVA-style)
3. **Reasoning Specialist (300M):** Math, code, logic (QwQ-style)
4. **Code Specialist (200M):** Programming (StarCoder-style)
5. **Knowledge Specialist (300M):** Facts, retrieval, wiki-like (optimized for RAG)

**Router learning:**
- Train router on 10K+ labeled queries ("which specialist should answer?")
- Router outputs: probability distribution over specialists
- Load top specialist(s), invoke

**VRAM budget:**
- 10GB: One active specialist (swappable)
- 2GB: Router stays resident
- 2GB: Generation dispatch (SD 1.5)
- 2GB: Swap/preload overhead

**Implementation roadmap:**
1. Design router: architecture, input/output spec
2. Collect 10K labeled training queries (which specialist?)
3. Train 5 specialists in parallel (can distribute across multiple GPUs or time-slice)
4. Train router on labeled data
5. Benchmark: test on held-out query set

**Research needed (R-ARCH-2):**
- [RESOLVED] R-ARCH-2a: Router architectures (learned vs rule-based, latency comparison). Best tradeoff for this project is hybrid: rule-based front gate + learned lightweight classifier + LLM fallback. Latency/cost rationale is documented in Router-2/3 findings and aligns with RouteLLM-style quality-cost routing (arxiv:2406.18665).
- [RESOLVED] R-ARCH-2b: Specialist architectures (should each be exactly 742M, or vary by task?) Recommendation: **heterogeneous specialist sizing** beats "all 742M" for quality-cost frontier, with one caveat: keep a single shared tokenizer/chat format and shared checkpoint schema so orchestration stays simple.
  - RouteLLM/FrugalGPT results are built on strong+weak model pairing (different capacity/cost), and show large savings at near-constant quality; this is direct evidence that heterogeneous capacity is a feature, not a bug (arxiv:2406.18665, arxiv:2305.05176).
  - Practical sizing policy for this repo/hardware: keep reasoning + code at 742M first, allow narrower specialists for easier domains (routing/classification/short-format helpers), and only scale up task-specific experts when benchmark delta justifies VRAM/latency cost.
  - Do not force equal params across all specialists; force equal **interface contract** instead (same tokenizer IDs, prompt format, stop tokens, and eval harness).
- [RESOLVED] R-ARCH-2c: Load balancing (how to swap models in/out without stalling?) Recommendation: avoid per-request model hot-swap. Use **resident pool + async warmup + queueing**.
  - Systems evidence: iteration-level scheduling avoids head-of-line blocking by scheduling per token-step rather than waiting for full request batches (ORCA OSDI'22: usenix osdi22-yu).
  - Memory evidence: paged KV-cache management increases effective batch capacity and reduces fragmentation pressure that causes stall cascades under mixed-length traffic (PagedAttention/vLLM: arxiv:2309.06180).
  - Throughput evidence: prompt/generation split strategies and serving-side scheduling materially reduce latency/tail latency under load (DeepSpeed-FastGen: arxiv:2401.08671).
  - Concrete plan: keep router + 1 default specialist resident; maintain one prewarmed standby slot; route overflow by queue policy (short-first within class, deadline-aware fallback to base 742M); perform background prefetch/load and only cut over when health-check + warmup pass.
- [RESOLVED] R-ARCH-2d: Training data generation (label 10K queries with correct specialist). Recommended pipeline:
  - Bootstrap labels from heuristics + existing route outcomes (weak labels).
  - Add small high-quality human/LLM-judge corrected set for ambiguous prompts.
  - Actively sample disagreement buckets for targeted relabeling instead of uniform 10K labeling.
  - Train router on preference/outcome labels and optimize for quality-cost curve.
- [RESOLVED] R-ARCH-2e: Evaluation (multi-specialist system performance vs single model). Router/specialist evaluation should use **quality-cost frontier** rather than raw accuracy only (RouteLLM arxiv:2406.18665, FrugalGPT arxiv:2305.05176):
  - Report quality at fixed expensive-route percentages (e.g., 10/25/50/75/100%).
  - Report cost to reach target quality (e.g., 95% of strong-model quality).
  - Include OOD slices separately (where simple routers often fail).
  - Keep per-specialist task benchmarks and add end-to-end routed benchmark as the deployment metric.

**Timeline:** 4-5 months

**Advantages:**
- Each specialist optimized for its task (vision model learns vision REALLY well)
- Can iterate on specialists independently (retrain vision without touching code specialist)
- Closest to real-world systems (OpenAI, Anthropic likely use something like this)
- Scales: add more specialists without retraining core

**Disadvantages:**
- Complex implementation (multi-model orchestration, router training)
- Requires labeled training data for router (10K queries with expert labels)
- Training 5 models is 5× the compute
- Higher latency (router inference + model loading)

---

### Approach 3: Distillation from Qwen3-30B (FASTEST PROOF-OF-CONCEPT)

**Core idea:** Use Qwen3-30B (already installed) to create specialized 742M variants via distillation. Quick experiment to validate multi-model architecture.

**What to create:**
1. **Enigma-742M-Reasoning:** Qwen3-30B generates thinking traces on math/logic
2. **Enigma-742M-Vision:** Qwen3-30B generates VQA answers on image dataset
3. **Enigma-742M-Code:** Qwen3-30B generates code solutions on programming tasks

**Process per variant:**
1. Use Qwen3-30B to generate target outputs on task-specific data
2. Train 742M via FORGE Distillation mode
3. Test on benchmark (GSM8K for reasoning, etc.)

**VRAM budget:**
- 12GB: 742M variant during training
- 2GB: Qwen3-30B for teacher inference (separate, offline)
- 2GB: Overhead

**Implementation roadmap:**
1. Collect/prepare data for 3 tasks (existing: OpenThoughts3, LLaVA dataset, The Stack)
2. Generate targets from Qwen3-30B (offline, can batch)
3. Train 3 variants via FORGE
4. Benchmark each on task-specific test set
5. **Decision point:** If variants work, proceed to Approach 1 or 2. If not, learn why and iterate.

**Research needed (R-ARCH-3):**
- [RESOLVED] R-ARCH-3a: Distillation effectiveness at 742M scale (does 742M learn from 30B?) **Yes, materially.** Evidence: DeepSeek-R1 reports strong distilled performance from a much larger teacher into small dense students, including 1.5B/7B/8B/14B/32B checkpoints, and explicitly states smaller models benefit from distilled reasoning traces (arxiv:2501.12948 + DeepSeek-R1 model card). Orca independently shows smaller models gain large reasoning improvements from rich explanation traces generated by stronger teachers (arxiv:2306.02707).
- [RESOLVED] R-ARCH-3b: Data requirements (how much task-specific data per variant?) Use staged targets, not one-shot max:
  - **Pilot:** 10K-30K curated examples per variant to validate training plumbing + overfit risk.
  - **Useful:** 80K-200K per variant for measurable specialist lift.
  - **Strong:** 300K-800K for reasoning-heavy variants when teacher quality is high.
  - Rationale: LIMA shows alignment can move with very small curated sets (1K), Orca shows complex reasoning transfer needs broader/denser traces, and DeepSeek-R1 distill models cite 800K curated samples for robust small-model transfer.
- [RESOLVED] R-ARCH-3c: Benchmark setup (which evals to use for each variant?) Recommended minimum benchmark matrix:
  - **Reasoning variant:** GSM8K + MATH-500 (+ AIME-style slice when available), report pass@1 and consistency across 3 seeds.
  - **Code variant:** HumanEval pass@1 + LiveCodeBench/Codeforces-style difficulty slice (DeepSeek-R1 uses both families).
  - **Vision variant:** internal 5K held-out VQA gate from this roadmap plus one public VLM benchmark already used in your LLaVA track for comparability.
  - **System metric:** add routed end-to-end quality-cost frontier, not only per-specialist scores.
- [RESOLVED] R-ARCH-3d: Router for Approach 3 (can we use simple heuristics, or need learned router?) Recommendation: **two-phase**.
  - Phase POC (3 variants): heuristic router is acceptable to move fast (regex/task-keyword + cheap classifier fallback).
  - Phase production: learned router is required once overlap/ambiguity rises; RouteLLM shows meaningful cost-quality gains from learned routing and highlights OOD degradation without targeted augmentation (arxiv:2406.18665).
  - Promotion gate: switch from heuristic-only to learned when misroute rate or fallback-to-base rate exceeds a fixed threshold on held-out mixed-task traffic.

**Timeline:** 3-4 weeks (1-2 weeks per variant)

**Advantages:**
- **Fastest to results** (1-2 weeks per variant)
- Low risk (teacher is proven, process is standard)
- Validates architecture idea before committing to Approach 1 or 2
- Uses existing FORGE infrastructure
- Clear success/fail signal

**Disadvantages:**
- Still 742M (not true specialists)
- Depends on teacher quality (Qwen3-30B is good but not perfect)
- Variants may overlap (not diverse enough)
- Proof-of-concept only, not production system

---

### Approach 4: Incremental Integration (SAFEST)

**Core idea:** Keep Enigma 742M as base, add one capability at a time, test before moving on.

**Phases:**
| Phase | What | Timeline | Success Criteria |
|-------|------|----------|------------------|
| Now | N-6: Resume pre-train (base 742M) | 2-3 months | Loss curve stable, no NaNs |
| A | N-9/N-10: Instruction fine-tune + alignment | 2-3 weeks | Benchmark on MMLU/GSM8K |
| B | Add vision encoder (frozen LLaVA) | 2 weeks | Can process images without crashing |
| C | Test vision reasoning | 1 week | 50%+ accuracy on 5K VQA pairs |
| D | Add code fine-tuning variant | 2 weeks | Passes 10+ HumanEval problems |
| E | Add reasoning variant (OpenThoughts) | 2 weeks | GSM8K > 30% |
| F | Build router (if 3+ specialists work) | 1-2 weeks | Router predicts correct specialist 85%+ |
| G | Add generation/audio plugins | 1-2 weeks each | Can invoke SD/Whisper correctly |

**VRAM budget:**
- 12GB: Active model
- 3GB: Temporary specialist during loading
- 1GB: Overhead

**Research needed (R-ARCH-4):**
- [RESOLVED] R-ARCH-4a: Sequential integration order (which capability first?) Recommended order:
  1. Base stability (N-6 pretrain health: no NaNs, stable loss, usable perplexity)
  2. Instruction/alignment baseline (N-9/N-10) so all later specialists inherit usable chat behavior
  3. Reasoning specialist before vision/audio plugins (highest leverage for text quality and eval visibility)
  4. Code specialist
  5. Vision specialist + VQA validation
  6. Router only after at least two specialists beat base on their own domains
  7. Optional generation/audio plugins last (non-core to text reasoning quality)
  - Principle: always harden the core language path before adding modality or orchestration complexity.
- [RESOLVED] R-ARCH-4b: Benchmark suite (what tests for each phase?) Use phase-gated checks:
  - **Core health gate (every phase):** train stability (NaN/Inf), latency sanity, memory budget adherence, regression test suite.
  - **Alignment gate:** MMLU + instruction-following slice.
  - **Reasoning gate:** GSM8K + MATH-500 (or equivalent held-out math set).
  - **Code gate:** HumanEval pass@1 (+ optional LiveCodeBench slice).
  - **Vision gate:** internal held-out 5K VQA gate + one public VLM benchmark used by your LLaVA path.
  - **Router gate:** route accuracy + routed end-to-end quality-cost frontier vs single-model baseline.
- [RESOLVED] R-ARCH-4c: Fallback logic (if a phase fails, which is it safe to skip?) Safety policy:
  - If **core/alignment** fails: stop entire roadmap and fix core first (nothing else should proceed).
  - If **reasoning** fails: continue code/vision experiments only if they do not regress base chat quality; keep routing disabled.
  - If **one specialist** fails (code or vision): ship other specialists + base fallback; do not block unrelated track.
  - If **router** fails: keep heuristic routing or manual mode selection; never force learned router into production path.
  - If **plugin phase** fails: skip safely; plugins are optional and must not block core chat/reasoning releases.

**Timeline:** 6-9 months

**Advantages:**
- **Very low risk** (test each before committing)
- **Feedback loop** (learn from each phase)
- Can stop at any point with a working system
- Simple code changes (one feature at a time)

**Disadvantages:**
- Slowest timeline
- Redundant work if a late capability fails
- Hard to optimize across tasks (single model, compromises everywhere)

---

## Audit Notes (April 2026)

**Before reading this checklist, read this audit. ~50% of the original items were already implemented.**

**Already implemented — verified in code:**
- QK-Norm (`use_qk_norm=True` default, applied in model_components.py lines 347/423/451)
- MTP (`n_predict_heads=2`, `predict_heads` ModuleList, `mtp_loss` wired in model.py 496-518)
- Intra-document masking (`build_packing_masks()` in training.py builds block-diagonal causal mask)
- Flash Attention (tries `flash_attn_func`, falls back to `F.scaled_dot_product_attention`)
- Weight tying (`self.output.weight = self.tok_embeddings.weight` in model.py line 233)
- LayerScale + stochastic depth (`use_layer_scale`, `drop_path_rate` in ForgeConfig)
- Speculative decoding (`speculative_generate` in engine_generation.py)
- Vision/Audio projections (`vision_hidden_size`, `audio_hidden_size` in ForgeConfig + model.py)
- GQA (`n_kv_heads` in ForgeConfig, validated in __post_init__)
- SwiGLU, RMSNorm, RoPE + YaRN, Sliding window, MoE (`use_swiglu`, `use_rms_norm`, `rope_theta`, `rope_scaling_type`, `sliding_window`, `use_moe`) — all in ForgeConfig
- Label smoothing (`label_smoothing=0.0` parameter in model.py)
- LLRD (`_build_llrd_param_groups()` in training.py)
- Weight decay param groups (bias/norm get no_decay, see training.py 1332-1345)
- LR warmup (`warmup_steps` in TrainingConfig + defaults.py)
- Differential attention (`use_differential_attn=True` default)
- NEFTune (`neftune_alpha=5.0` default)
- MLA (`mla_latent_dim`), ToMe (`tome_ratio`), YOCO KV sharing (`kv_share_groups`), MoD (`use_mixture_of_depths`), shifted sparse attention (`use_shifted_attention`) — all in ForgeConfig
- AdamW + AdemaMix optimizers with beta1/beta2 — in training.py
- KV cache, sampling (top-k/top-p/temperature/repetition penalty) — all implemented in inference

**D-series status corrections:**
- D-14 (QK-Norm): ~~Action needed~~ → **ALREADY DONE** (`use_qk_norm=True` is the default, wired)
- D-15 (MTP): ~~Verify scaffold~~ → **ALREADY DONE** (loss wired in model.py, λ=1/n_heads)
- D-7 (intra-doc masking): ~~Verify~~ → **ALREADY DONE** (`build_packing_masks()` exists + is called)
- D-8 (embed weight decay): **STILL OPEN** — training.py line 1337 checks `'bias' in name or 'norm' in name` but NOT `'embed'`. Embeddings and the output head get weight_decay when they should not. 5-line fix.

**Removed from checklist (wrong scope for this project):**
- DDP/FSDP/torch.distributed — single GPU system; multi-GPU is future work with no near-term value
- Multilingual benchmarks — project is English-focused
- Torch barrier/deadlock prevention — no distributed training
- Items describing behavior that is purely inference-time and already configurable in GUI (top-k, top-p, temperature, stop tokens)

**Real missing research (added below):**
- Audio encoder training procedure (projection layer exists, no pipeline)
- Video/3D generation via mod system (goal items, no research done)
- Mod integration procedure (how to plug in audiogen, videogen, threed mods)
- LoRA adapter training workflow (directory exists, training procedure unknown)
- Catastrophic forgetting prevention when creating specialists (EWC exists, is it wired for this use case?)
- Context extension implementation (YaRN config field exists but forward-pass code needs verification)
- Router multi-model dispatch design (router.py handles mods, not model selection)

---

## Comprehensive Research Checklist (Full Coverage)

**Legend:** Items marked [VERIFY] need code verification only (no external research). Items marked [RESEARCH] need external sources. Items marked [BUILD] are confirmed gaps needing implementation.

---

### Architecture — What Needs Verification (Not External Research)

These are implemented in code but the exact behavior needs to be read and confirmed before N-6 restart:

- [RESOLVED] Arch-1: `build_packing_masks()` → `attention_mask_2d` kwarg → model.forward(). Confirmed wired: [enigma_engine/core/training.py](enigma_engine/core/training.py) lines 1669, 1676, 3183-3240.
- [RESOLVED] Arch-2: xl uses actual GQA: n_heads=24, n_kv_heads=6 (4:1 ratio, same as LLaMA3). Verified [enigma_engine/core/model_presets.py](enigma_engine/core/model_presets.py).
- [RESOLVED] Arch-3: xl preset confirmed — dim=1536, n_layers=24, n_heads=24, n_kv_heads=6, max_seq_len=4096, rope_theta=500000.0, dropout=0.05, hidden_dim=4×1536=6144.
- [RESOLVED] Arch-4: `use_differential_attn=True` is the ForgeConfig dataclass default ([enigma_engine/core/model_presets.py](enigma_engine/core/model_presets.py#L127) line 127) — applies to ALL presets including xl. Latency research still open (see Attn-2).
- [RESOLVED] Arch-5: MTP checkpoint size confirmed. `count_parameters()` in [enigma_engine/core/model_presets.py](enigma_engine/core/model_presets.py#L931) line 931: `mtp = n_predict_heads × vocab_size × dim`. For xl (dim=1536, n_predict_heads=2, vocab≈32K): **2 × 32000 × 1536 ≈ 98M extra params = ~196MB at fp16.** These are NOT weight-tied. Larger vocab (e.g., 64K) doubles this to ~390MB. Factor into any checkpoint size planning.
- [RESOLVED] Arch-6: YaRN IS fully implemented — `compute_freqs_cis()` in [enigma_engine/core/model_components.py](enigma_engine/core/model_components.py#L190) has a real `elif scaling_type == "yarn"` branch (lines 190-213) that scales frequency bands. Not dead config.
- [CONFIRMED] Arch-7: MTP loss weight = 1/n_predict_heads ([enigma_engine/core/model.py](enigma_engine/core/model.py#L518) line 518: `loss + mtp_loss / max(1, len(self.predict_heads))`). With xl's n_predict_heads=2: each head adds 0.5× weight. DeepSeek-V3 uses λ=0.3 — **MTP-1 resolved, see MTP section. Our λ=0.5 is higher; lower to 0.3 for next pretraining run.**
- [RESOLVED] Arch-8: LayerScale NOT used at 3B or 0.6B scale. Verified SmolLM3-3B config.json and Qwen3-0.6B config.json — neither has LayerScale parameters. SmolLM3 blog ablations test GQA, NoPE, intra-doc masking, embed weight decay — no LayerScale anywhere. **Conclusion: keep `use_layer_scale=False` (current default). Stochastic depth also absent from both — keep `drop_path_rate=0.0`.** These features do not appear in production 0.6B–3B models. No change needed.

---

### Architecture — Real Gaps (External Research Needed)

These are features referenced in code but the correct values/approaches need external evidence:

**Attention:**
- [RESOLVED] Attn-1: GQA ratio research complete. SmolLM3-3B uses 16:4 = **4:1** (same as our xl). Qwen3-0.6B uses 16:8 = **2:1** (more conservative). SmolLM3 blog: "Our ablations on a 3B model trained with 100B tokens from FineWeb-Edu showed that GQA matches the performance of MHA while significantly reducing KV cache." **Our xl 24:6 = 4:1 ratio is validated at 3B scale. Qwen3-0.6B uses 2:1 which is more conservative for sub-1B but we already have a trained checkpoint — no change needed.** If retraining from scratch at 742M, 2:1 (12 KV heads) would be marginally safer; 4:1 is acceptable per SmolLM3 3B evidence.
- [RESOLVED] Attn-2: Differential attention (Ye et al., *Differential Transformer*, arxiv:2410.05258, **ICLR 2025 Oral**). Paper ablates at 830M — directly comparable to our 742M (the scaling claim applies). Confirmed wins vs. vanilla transformer at equal params/tokens: lower perplexity, better long-context retrieval, reduced hallucination in QA/summarization, better in-context learning, more robust to prompt permutation. **Already enabled in our code** ([enigma_engine/core/model_presets.py](enigma_engine/core/model_presets.py#L127) line 127 default `True`; wired at [enigma_engine/core/model_components.py](enigma_engine/core/model_components.py#L360) lines 360-367 and 578). **Real cost:** the dual-softmax structure does NOT fit `F.scaled_dot_product_attention` / Flash Attention — [line 543](enigma_engine/core/model_components.py#L543) falls through to the manual attention path when `use_differential_attn=True`. Training and prefill attention therefore run slower than a vanilla-Flash baseline. **Memory:** negligible overhead (head count unchanged, one extra learnable λ per layer). **Verdict:** KEEP ENABLED for quality gains; the Flash-path loss is already measured & absorbed into our training budget. Worth tracking via an internal A/B only if we ever hit an attention-bound bottleneck. No action.
- [RESOLVED] Attn-3: xl (and all medium/base/large/xxl) presets already use rope_theta=500000.0 in [enigma_engine/core/model_presets.py](enigma_engine/core/model_presets.py). Only the small preset uses 10K. This was a documentation error — code was already correct.

**Normalization:**
- [RESOLVED] Norm-1: RMSNorm epsilon is 1e-6 in [enigma_engine/core/model_components.py](enigma_engine/core/model_components.py). Matches Qwen3 best practice. fp32 upcast present.
- [RESOLVED] Norm-2: QK-Norm uses the same RMSNorm class as layer normalization. Full RMSNorm on Q and K, matching Qwen3 pattern.

**MTP (Multi-Token Prediction):**
- [RESOLVED] MTP-1: DeepSeek-V3 technical report (arxiv:2412.19437v2, Section 4.2) confirms: **λ=0.3 for first 10T tokens, then λ=0.1 for remaining 4.8T tokens.** MTP depth D=1 (predict 1 extra token beyond next-token). Ablation table (Table 4) validates MTP consistently improves benchmarks at both small (15.7B) and large (228.7B) scale. MTP acceptance rate at inference: 85–90% across topics. **Our code uses λ=0.5 (= 1/2 predict heads) which is higher than DeepSeek's 0.3. Recommendation: lower to λ=0.3 for new pretraining runs.** For existing checkpoints already trained with 0.5, keep as-is — retrofitting is not worth the churn. File: [enigma_engine/core/model.py](enigma_engine/core/model.py#L518) — change divisor from `len(predict_heads)` to `len(predict_heads) / 0.6` (makes 2 heads → 0.3) OR use a dedicated `mtp_loss_weight` config field set to 0.3.
- [RESOLVED] MTP-2: Meta *Better & Faster LLMs via Multi-token Prediction* (arxiv:2404.19737) abstract verbatim: "The method is **increasingly useful for larger model sizes**." Paper tested 300M / 1.3B / 6.7B / 13B — at 300M the gain is neutral or slightly negative for natural language, weakly positive for code; at 1.3B gains become clearer on code (HumanEval/MBPP); at 13B the reported HumanEval delta is +12%. **Our 742M sits in the ambiguous bracket.** Two concrete issues found in code this pass:\n  1. **Inverted comment BUG** at [enigma_engine/core/model_presets.py](enigma_engine/core/model_presets.py#L129) line 129: `# R25: Multi-token prediction extra heads (biggest gain at small model sizes)` — this inverts the paper's conclusion. Fix: rewrite comment to `"most useful at larger scales; marginal/ambiguous at <1B"`.\n  2. **Hidden param cost:** each entry in `self.predict_heads` is `nn.Linear(dim=1536, padded_vocab≈32K, bias=False)` at [model.py line 240](enigma_engine/core/model.py#L240), **NOT weight-tied** to `tok_embeddings`. Default `n_predict_heads=2` adds ~98M params (~13% of 742M) that are discarded at inference except when using Medusa speculative decoding. This is effectively 98M params spent on an auxiliary training signal of uncertain benefit at our scale.\n\n**Verdict / action:**\n- If Medusa inference is planned → keep `n_predict_heads >= 1` (Medusa needs them) but **reduce default from 2 → 1** to halve the cost, and weight-tie the head to `tok_embeddings.weight` same as `self.output` (free 49M saving). Log as new item **MTP-2a**.\n- If Medusa is NOT planned → set `n_predict_heads = 0` by default, free 98M params for the main model. Log as new item **MTP-2b**.\n- Internal A/B (MTP on vs off, same data/steps, val loss + HumanEval-style code) is still the honest way to decide — but the default should not be 2 untied heads by accident.

---

### Data — Real Research Gaps

These have no implementation yet and need external evidence for correct choices:

**Tokenization:**
- [RESOLVED] Tok-1: Tao et al. *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies* (NeurIPS 2024, arxiv:2407.13623) trained models from 33M → 3B params on up to 500B chars to fit a compute-optimal vocab scaling law. Concrete result: at **3B params**, expanding 32K → **43K** improved ARC-Challenge from 29.1 → 32.0 at the same 2.3e21 FLOPs. Llama2-70B's optimal predicted vocab is ~216K (7× its actual 32K). For our 742M (below their 3B data point), the compute-optimal is in the **~35-45K** range — SmolLM3's 49K at 3B is a slight overshoot but defensible. **64K would be over-provisioned at our scale** (puts us above the 3B-scale optimum while we're less than a third of that). Current 32K is slightly under-optimal but not catastrophically so. **Decision:**\n  - Do NOT expand to 64K.\n  - If (and only if) we retrain from scratch, target **~40K vocab** (matches the paper's 3B-scale optimum, leaves headroom for a small multilingual or code-specific extension).\n  - D-6 ("Tokenizer 64K expansion") in the backlog should be **downgraded** — retitle to "Tokenizer 40K expansion (if retraining from scratch only)." Anything past 32K requires a full restart; committing to 64K costs ~49M extra embedding params for no measurable quality gain at 742M per the scaling law.
- [CONFIRMED GAP] Tok-2: [enigma_engine/core/advanced_tokenizer.py](enigma_engine/core/advanced_tokenizer.py) has NO byte-level fallback. Unknown chars fall through to CHAR-LEVEL decode first (line 267), then UNK token id=3 (line 272/279) if the char is not in vocab. Rust BPE is the same — only merges tokens in its merge table, no byte fallback. **Rare Unicode (CJK, Arabic, emoji, etc.) will produce UNK ids, corrupting model input and breaking multimodal.** Fix before tokenizer retrain: add 256 raw byte tokens to base vocab (byte-level BPE approach used by GPT-2/Qwen). File: [enigma_engine/core/advanced_tokenizer.py](enigma_engine/core/advanced_tokenizer.py) `::_init_base_vocab()` and [rust_extensions/src/lib.rs](rust_extensions/src/lib.rs) `::pre_tokenize()`.
- [BUILD] Tok-3: Collect The Stack v2 (D-2) before tokenizer retrain — code vocabulary affects merge priority. Train tokenizer on data mixture AFTER collecting all sources.

**Data mixing (all resolved Pass 137 — external research):**
- [RESOLVED] Data-1: **Stage 1: 12% code**, per SmolLM3 blog (https://huggingface.co/blog/smollm3). Quote: "Code: 12% — The Stack v2 (16 programming languages), StarCoder2 pull requests, Jupyter and Kaggle notebooks, GitHub issues, and StackExchange." Stage 2 ramps to 15%, Stage 3 (annealing/cooldown) to 24%. Qwen3 (arxiv:2505.09388) says "increasing STEM/coding proportion in Stage 2" but gives NO exact percentages. DeepSeek-LLM (arxiv:2401.02954) also provides no explicit %. **No direct evidence on mid-run introduction of code.** SmolLM3's 12→15→24 progression suggests monotonic increase is safe; abrupt jumps unvalidated. SmolLM3 is 3B; pattern assumed to scale down to 742M but not empirically verified at our scale.
- [RESOLVED] Data-2: **Stage 1: 3% math** (FineMath-3+ + InfiWebMath-3+), per SmolLM3 blog. Quote: "Math: 3% — FineMath3+ and InfiWebMath3+". Stage 2: 10%. Stage 3 (cooldown): 13%. FineMath paper (arxiv:2502.02737) validates the datasets themselves. **FineMath-3+ (34B tokens) for main mixture, FineMath-4+ (9.6B tokens) optionally for Stage 3 resampling.** HuggingFace: `HuggingFaceTB/finemath` configs `finemath-3plus` / `finemath-4plus` / `infiwebmath-3plus`.
- [RESOLVED] Data-3: HF dataset ID confirmed `mlfoundations/dclm-baseline-1.0`. 4T tokens, 3B documents. Public, no gating. Paper arxiv:2406.11794. Our [collect_pretraining_data.py](collect_pretraining_data.py) fetcher (added Pass 136) streams correctly.
- [RESOLVED] Data-4: HF dataset `open-thoughts/OpenThoughts3-1.2M` CONFIRMED. Composition CONFIRMED: 850K math + 250K code + 100K science = 1.2M. Paper arxiv:2506.04178. **Tag format CONFIRMED via HuggingFace datasets-server API (live row fetch Pass 139).** Schema: 4 fields — `difficulty` (int64), `source` (str), `domain` (str), `conversations` (list of `{"from": "human"/"gpt", "value": str}`). Row 0 gpt value begins: `"<think> Okay, I need to solve this problem..."` and ends with `</think>\n\n` then the final answer. Row 1 same pattern. The QwQ-32B reasoning trace IS wrapped in `<think>...</think>` inline in the `value` field — our special tokens `<think>=4`, `</think>=5` align exactly. **Action for OpenThoughts3 fetcher in [collect_finetuning_data.py](collect_finetuning_data.py):** extract `value` from each conversation turn and preserve `<think>`/`</think>` tags verbatim — do NOT strip them during preprocessing.
- [RESOLVED] Data-5: **Industry standard is 75% MinHash similarity**, not our current 80%. FineWeb technical report (arxiv:2406.17557v2, Section 3.4): "We chose to collect each document's 5-grams...computed MinHashes using 112 hash functions in total, split into 14 buckets of 8 hashes each — targeting documents that are at least 75% similar." DCLM uses same approach. Our 80% is slightly more permissive (keeps more near-duplicates). See Data-5b below for follow-up action.
- [CONFIRMED GAP] Data-5b: MinHash threshold in our training.py `minhash_dedup()` defaults to 0.8. FineWeb/DCLM/SmolLM3 use 0.75. **Low priority code change:** flip default from 0.8 to 0.75 in [enigma_engine/core/training.py](enigma_engine/core/training.py) `minhash_dedup()` before next tokenizer/data-prep cycle. Impact on already-deduped `combined.txt` is zero (re-dedup not planned); only matters for future data runs. File note: also check the LSH band/row split — FineWeb uses 14 bands × 8 rows of 112 hashes; if our implementation uses different band counts the effective threshold differs.
- [RESOLVED] Data-6 (NEW — annealing/Stage-3 mixture for 742M): SmolLM3 Stage 3 ("decay"): **24% code, 63% web, 13% math**. Noted finding: "no significant improvements from upsampling curated subsets in cooldown phase" — so Stage 3 gains come primarily from LR decay and code/math upweight, not from swapping in premium subsets. Pattern validated at 3B, assumed to scale to 742M. Evidence is empirical not theoretical — worth a small ablation before committing.

---

### Training Algorithm — Real Gaps

**Optimizer:**
- [RESOLVED] Opt-1: β2=0.95 is hardcoded in TrainingConfig (`adam_beta2: float = 0.95`). Not the PyTorch default. Correct.
- [RESOLVED] Opt-2: D-8 FIXED this session. Both `_setup_optimizer()` and `_build_llrd_param_groups()` in training.py now include `or 'embed' in name` in the no_decay check.
- [RESOLVED] Opt-3: Weight-tying confirmed at [enigma_engine/core/model.py](enigma_engine/core/model.py#L233) line 233: `self.output.weight = self.tok_embeddings.weight` — direct Python tensor alias (same `id()`). Optimizer param groups built in [enigma_engine/core/training.py](enigma_engine/core/training.py#L1337-L1351) iterate `self.model.named_parameters()`, which defaults to `remove_duplicate=True` — the tied tensor is yielded exactly once, under the name of the submodule registered first. Since `tok_embeddings` (model.py line 179) is registered before `output` (line 230), the tied weight surfaces only as `tok_embeddings.weight` and matches `'embed' in name` → goes to `no_decay_params`. `output.weight` is never visited. **Safe:** single param group, correct decay policy, no double-update. Code is correct as written.

**Scheduler:**
- [RESOLVED] Sched-1: WSD is already the DEFAULT scheduler (`schedule_type: str = "wsd"` in TrainingConfig). Not a build task. Was a documentation error.
- [RESOLVED] Sched-2: SmolLM3 pretraining uses 2000 fixed warmup steps out of ~8T tokens (~negligible %). SFT/DPO warmup not explicitly given in the blog. The `max(100, steps // 10)` formula is our own recommendation for small fine-tuning runs and is consistent with SmolLM3's proportionally-small warmup pattern. **Action confirmed: Sched-2 implementation (Suggestion row 20) is correct — change SFT/DPO default to `max(100, steps // 10)`.** This is a low-risk quality-of-life fix for short DPO runs.

**Precision:**
- [RESOLVED] Prec-1: `_resolve_amp_dtype("auto")` checks `torch.cuda.is_bf16_supported()` → picks bfloat16 on Blackwell/Ampere+ (RTX 5090 → bfloat16 confirmed). Loss scaling not needed for bfloat16's wider exponent range (Prec-2 implicitly confirmed safe).
- [RESOLVED] Prec-2: bfloat16 does NOT need dynamic loss scaling. Its exponent range is the same as float32 (8 bits) vs float16's 5 bits. GradScaler is not called when `_amp_dtype == bfloat16` ([enigma_engine/core/training.py](enigma_engine/core/training.py) lines 1255, 4651, 4860, 5269 all gate on `!= bfloat16`). RTX 5090 uses bfloat16, no action needed.

---

### Inference — Real Gaps

These are missing features or unverified behaviors:

- [RESOLVED] Inf-1: KV cache is pre-allocated. `KVCache` allocates full `max_seq_len` tensors upfront. No dynamic fragmentation.
- [RESOLVED] Inf-2: Answered directly from our code. Three dispatch paths at [enigma_engine/core/model_components.py](enigma_engine/core/model_components.py#L521-L565) lines 521-565:\n  1. **Flash path** (line 534): enabled only when `HAS_FLASH_ATTN && x.is_cuda && fp16/bf16 && not use_cache && not use_differential_attn && (mask is None or T == k.shape[1])`. This covers **training and prompt prefill** only.\n  2. **SDPA path** (line 544): enabled when `not self.use_differential_attn && hasattr(F, \"scaled_dot_product_attention\") && x.is_cuda`. Used for decode when Flash is unavailable. PyTorch's SDPA auto-dispatches to FA2 for prefill (seqlen_q > 1) but falls to the math/cuDNN kernel for seqlen_q = 1 decode (no specialized split-KV kernel — that's the Inf-5 gap).\n  3. **Manual attention** (line 568+): fallback for CPU/MPS/any-dtype AND the catch-all when `use_differential_attn=True`.\n\n**Key finding:** because `use_differential_attn` defaults to `True` ([model_presets.py L127](enigma_engine/core/model_presets.py#L127)), **our runtime config actually uses the manual attention path for both prefill and decode** — no prefill/decode specialization at all. Flash and SDPA are effectively dead code paths in the default config, only reachable if differential attention is explicitly disabled. Performance implication: prefill is slower than it could be (no Flash), but at least the prefill/decode behavior is consistent. **Action:** none required — current behavior is intentional (Attn-2 resolution: keep differential attention for quality). Logged as context. If we ever disable differential attention for a workload, the distinction kicks back in automatically.
- [RESOLVED] Inf-3: `H2OKVCache` evicts low-attention tokens. `StreamingLLMCache` uses attention sinks for infinite context. `TurboQuantKVCache` INT4 compression. No vLLM-style paged attention, but these mechanisms solve VRAM-bounded long sessions.
- [RESOLVED] Inf-4: xl KV cache VRAM calculated from config and formula `2 × n_kv_heads × head_dim × n_layers × seq_len × dtype_bytes`. With xl (`n_kv_heads=6`, `head_dim=64`, `n_layers=24`, `seq_len=4096`, bf16=2): **150MB KV cache**. At 32K context: **~1.17GB**. Conclusion: for 742M, KV cache is not the dominant VRAM consumer; model weights + activations dominate. INT8 KV compression is optional, not required for baseline operation.

---

### Post-Training — Real Research Gaps

**Process reward model:**
- [BUILD→DEPRIORITIZE] PRM-1: `ProcessRewardModel` and `PRMTrainer` exist in [enigma_engine/core/rl_training.py](enigma_engine/core/rl_training.py#L787) (lines 787, 916). Training format requires `step_labels: list[float]` (+1.0 correct, -1.0 wrong) per reasoning step. **No pipeline to auto-generate step labels exists.** Options: (a) manually label math solutions step-by-step, (b) use Qwen3-30B to judge each step. **NEW from DeepSeek paper (Section 4.2 Unsuccessful Attempts):** DeepSeek explicitly calls PRM an unsuccessful approach for RL training — reward hacking risk, hard to define fine-grain steps, automated annotation unreliable, adds complexity. PRM is useful ONLY for reranking top-N responses at inference time (not for online RL). **Recommendation: deprioritize PRM for RL. Use rule-based accuracy reward instead. Keep PRMTrainer code but don't build the labeling pipeline yet.**

**Reasoning alignment:**
- [RESOLVED] GRPO-1: GRPOTrainer confirmed in [enigma_engine/core/rl_training.py](enigma_engine/core/rl_training.py#L2311) (line 2311). No value head — takes external `reward_fn: Callable[[str, str], float]`. Group-relative advantage computed as `(r_i - mean(group)) / std(group)`. GRPO works at 742M but DeepSeek paper shows **distillation >> GRPO for small models** (32B GRPO matched QwQ-32B, but 32B distilled from R1 massively outperformed). For 742M: distill first, GRPO optionally second.
- [RESOLVED] GRPO-2: Shipped in Pass 142. Rule-based rewards live in [enigma_engine/core/reward_functions.py](enigma_engine/core/reward_functions.py) and GRPO routing in [enigma_engine/gui/gui_forge_new_modes.py](enigma_engine/gui/gui_forge_new_modes.py#L2229-L2260) uses `reasoning_reward` (with neural reward path blocked for GRPO).
- [RESOLVED] GRPO-3: Rewards normalized per-group before advantage: `adv = (r - r.mean()) / (r.std() + 1e-8)`. Matches paper. Confirmed in GRPOTrainer.
- [BUG-RISK] GRPO-4 (neural reward hacking): The current RLHF+GRPO training path (`gui_forge_new_modes.py` line 2250) passes `reward_model.score()` — a **neural** RewardModel — as the reward_fn. DeepSeek explicitly calls this "unsuccessful" for RL training: the policy learns to game the neural scorer (reward hacking), not to produce actually correct reasoning. Neural reward is only safe for preference ranking of pre-generated outputs, not for online RL. **Risk: using this path for reasoning training will train overconfident wrong answers.** Fix: build rule-based reward_fn (GRPO-2 spec above) and use it instead. The neural reward path should be restricted to alignment/helpfulness training, not reasoning. File: [enigma_engine/gui/gui_forge_new_modes.py](enigma_engine/gui/gui_forge_new_modes.py#L2250) line 2250, [enigma_engine/core/rl_training.py](enigma_engine/core/rl_training.py#L2341) line 2341.

**Distillation (Approach 3):**
- [RESOLVED] Distill-1: MiniLLM (arxiv:2306.08543, ICML 2023) provides the authoritative comparison. Ranking: **Reverse KL (MiniLLM) > SeqKD (cross-entropy on teacher outputs) > word-level forward KL soft targets**. Rationale: forward KL forces student to cover ALL modes of teacher (mode-averaging) → assigns probability to void regions → exposure bias, poor calibration. Reverse KL is mode-seeking → focuses on teacher's major modes → better precision, lower exposure bias. **Practical impact for our FORGE Distillation:** we already use SeqKD (train student on teacher-generated hard labels via cross-entropy), which is position #2 and significantly simpler than MiniLLM. This is the correct approach. Full reverse KL requires policy gradient, teacher-mixed sampling, and reward hacking mitigation — complex, not worth building for our current scale. **Current implementation is correct. No change needed for the distillation loss type.**
- [RESOLVED] Distill-2: **T=4–10 applies to soft-label KD (Hinton 2015 style) only** — where the full teacher probability distribution is used as training targets. Our FORGE Distillation uses SeqKD (hard labels: teacher generates response tokens, student trains with standard cross-entropy). In SeqKD, teacher temperature only affects generation quality/diversity of the training data. MiniLLM uses T=1 for teacher sampling. DeepSeek-V3 SFT uses deterministic/greedy teacher outputs for hard-label training. **Code check:** current FORGE Distill generation uses `temperature=0.8` in [enigma_engine/gui/gui_forge_new_modes.py](enigma_engine/gui/gui_forge_new_modes.py#L1471), which is within the recommended 0.7-1.0 range. No loss-level temperature scaling is needed.
- [RESOLVED] Distill-3: FORGE Distillation uses `_load_engine_for_path()` in [enigma_engine/gui/gui_forge.py](enigma_engine/gui/gui_forge.py#L945) (line 945). That calls `EnigmaEngine(model_path=path)` — accepts **ANY** format EnigmaEngine supports: native .pth, GGUF (.gguf), HuggingFace, GPTQ/AWQ. So yes, Qwen3-30B-A3B in GGUF format ([models/qwen3-30b-a3b/](models/qwen3-30b-a3b/)) can be used as the teacher directly.
- [RESOLVED] Distill-4: Qwen3 reports **compute reduction** (Strong-to-Weak distillation needs ~1/10 GPU hours vs running full 4-stage post-training per small model) but does **not** publish a fixed sample/token count in Section 4. DeepSeek-R1 provides concrete scale: **~800K SFT samples total** for reasoning distillation (about **600K reasoning + 200K non-reasoning**) and shows strong gains for 1.5B/7B/14B/32B/70B distilled students. Practical guidance for Enigma 742M: start with **200K-800K** high-quality teacher traces (minimum viable at low end, DeepSeek-proven at high end), then ablate quality vs quantity.

---

### Multi-Model Architecture — Research Gaps (Approaches 1 & 2)

**Router design:**
- [RESOLVED] Router-1: [enigma_engine/router.py](enigma_engine/router.py) is a TCP socket server (port 9900) — mod IPC bus, not model dispatcher. `ModRouter` manages connect/disconnect/heartbeat for external mod processes. Also has `BackgroundTrainer` that learns from conversations (replay buffer, DPO pairs, periodic retrain on best examples). To dispatch between AI models (Enigma vs specialist vs Qwen3), a new layer is needed on top of this — Router-2/3 still unsolved.
- [RESOLVED] Router-2: Practical options ordered by cost/accuracy:
  1. **Regex / keyword rules** (~0 ms, ~70-80% accuracy). Detects obvious patterns: code blocks (```), "draw/generate/image", math symbols, file paths. Cheap, brittle, good first-pass filter.
  2. **Sentence embedding + cosine similarity to task centroids** (~5 ms on CPU with sentence-transformers/all-MiniLM-L6-v2 at 22M params, ~80-90% accuracy). Pre-compute centroid vectors for each task class from labeled examples; route by nearest centroid.
  3. **BERT-tiny classifier** (4.4M params, ~10 ms CPU, ~85-92% accuracy when fine-tuned on ~5K labeled examples). Good accuracy/cost point.
  4. **Our 742M model as classifier** (~200-500 ms prefill for a short classification prompt, ~90-95% accuracy). High latency; see Router-3.
- **Recommendation:** layered approach — regex rules first (catch 60% of obvious cases), embedding similarity fallback (catch next 30%), LLM for the remaining 10% only if confidence from the first two is low. This keeps p50 latency near 0 and only pays LLM cost on genuinely ambiguous inputs.
- [RESOLVED] Router-3: Yes, technically — pass the task through Enigma with a classification prompt ("Classify the following request as one of: code / vision / reasoning / general. Request: ..."). **Tradeoffs:**
  - **Accuracy:** ~90-95% with good prompt engineering, comparable to a fine-tuned BERT-tiny but without training a separate model.
  - **Latency:** ~200-500 ms for prefill of a ~200-token classification prompt at batch=1 on 5090 (dominated by prefill FLOPs, not by the 1-token classification output). This is added to EVERY user turn — doubles the perceived response latency vs. a direct LLM call without routing.
  - **VRAM:** no extra load — Enigma is already resident.
- **Verdict:** use this ONLY as the fallback in the Router-2 layered approach, not as the primary router. The 200-500 ms cost is unacceptable on fast-path requests where regex + embedding already give a confident answer. Implementation note: can be made much cheaper with a cached embedding of the system prompt (skip re-prefill of the static instruction part).

**Vision integration:**
- [RESOLVED] Vision-1: CONFIRMED via LLaVA-1.5 paper (arxiv:2310.03744) and [llava-vl.github.io](https://llava-vl.github.io/). Architecture: **frozen CLIP ViT encoder → projection → LLM decoder (trained)**. Projection design changed between versions: LLaVA-1 used a single linear matrix; **LLaVA-1.5 upgraded to a 2-layer MLP with GELU** ("fully-connected vision-language cross-modal connector"), reporting "surprisingly powerful and data-efficient" gains across 11 benchmarks. **Code gap:** our [enigma_engine/core/model.py](enigma_engine/core/model.py#L186) `self.vision_projection` is a single `nn.Linear` (LLaVA-1 era). For Code-6 we should upgrade to `nn.Sequential(Linear, GELU, Linear)` to match LLaVA-1.5 SoTA. Logged as **Vision-1b** under backlog.
- [RESOLVED] Vision-2: CONFIRMED. LLaVA-1 (ViT-L/14 @ 224 px) → 16×16 = **256 visual tokens**. LLaVA-1.5 (ViT-L/14 @ 336 px) → 24×24 = **576 visual tokens**. LLaVA-HD / LLaVA-NeXT (AnyRes grid up to 4×): 576 × up-to-5 = up to 2880. **Decision for 742M / 4096 ctx:** 576 tokens is fine (leaves 3520 for text). LLaVA-1.5 paper reports the 336 px variant outperforms the 224 px variant on VQA benchmarks (exact ablation delta not re-verified this pass). Use 336 px to match LLaVA-1.5. If multi-image or long-text ever needed, fall back to the 224 px / 256-token config.
- **[Pass 148 currency check]** Vision-2b: **LLaVA-NeXT (Jan 2024) is now superseded by LLaVA-OneVision** (arxiv:2408.03326, last revision Oct 26 2024, Apache-2.0). LLaVA-OneVision uses **SigLIP SO400M (not CLIP) + Qwen2 LM**, and the SAME model handles single-image / multi-image / video in one architecture (no separate variants). Sizes: 0.5B / 7B / 72B — the **0.5B variant is the right LLaVA-class drop-in for our 742M base on 16 GB**. Vision tokens use anyres grids similar to LLaVA-NeXT but with stronger downsampling for video. **Action:** when wiring Code-6, target the LLaVA-OneVision recipe + SigLIP 2 encoder (Vision-1b) directly instead of LLaVA-1.5; the 1.5 numbers above remain valid as a minimum-viable reference. Keep CLIP-ViT-L/14-336 only as the lowest-friction encoder option.
- [RESOLVED] Vision-3: CONFIRMED. LLaVA-1.5 uses `openai/clip-vit-large-patch14-336` (HF: 304M params, 1024-d penultimate hidden state). Drop-in: `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` (same arch, trained on LAION-2B, 224 px only — would need to accept the 2 pt VQA hit or use LAION's 336 px variant if released). **Recommendation:** start with `openai/clip-vit-large-patch14-336` — exactly what LLaVA-1.5 ships. Add to `core/model.py` ModelConfig: `vision_hidden_size = 1024` when vision is enabled. Use `select_feature="patch"` (drop CLS), matching LLaVA-1.5.
- [RESOLVED] Vision-4: CONFIRMED two-stage training. **Stage 1 — feature alignment (projection only, LLM + CLIP frozen):** `liuhaotian/LLaVA-Pretrain` (558K image-caption pairs from LAION/CCS/SBU, BLIP-filtered). 1 epoch, small LR. **Stage 2 — instruction tuning (projection + LLM, CLIP frozen):** `liuhaotian/LLaVA-Instruct-150K` + academic VQA mix totaling **665K samples** (VQAv2, GQA, OKVQA, OCRVQA, A-OKVQA, TextCaps, RefCOCO, Visual Genome). LLaVA-1.5 full training: ~1 day on 8×A100 for 13B. For our 742M on one 5090 expect proportional time. **Modern alternative:** `lmms-lab/LLaVA-OneVision-Data` is the current superset (single-image + multi-image + video), larger but better coverage. Action for Code-6: wire Stage 1 data collection into `collect_finetuning_data.py` behind a `--llava-pretrain` flag.
- [RESOLVED] Vision-5: [enigma_engine/core/model.py](enigma_engine/core/model.py) `model.forward()` accepts `vision_features: Optional[Tensor]` (line 670). `vision_projection` created in __init__ when `vision_hidden_size is not None` (lines 185-191). Vision embeds prepended to token embeddings before the transformer. NOT dead code — LLaVA-style path is wired. Training the projection is still needed (Code-6).

**Specialist training (Approach 2):**
- [RESOLVED] Spec-1: Yes, catastrophic forgetting at 742M on focused fine-tuning is well-documented (Luo et al. 2023, *An Empirical Study of Catastrophic Forgetting in LLMs During Continual Fine-tuning*). Two practical mitigations exist: (a) **EWC** (Kirkpatrick 2017) — our [enigma_engine/core/ewc.py](enigma_engine/core/ewc.py) implements this, but EWC is sensitive to Fisher estimation quality and requires keeping the base model's Fisher matrix in memory; (b) **LoRA adapters per specialist** (Hu et al. 2021) — we already have [enigma_engine/core/lora_utils.py](enigma_engine/core/lora_utils.py). **Verdict:** LoRA is the right default for multi-specialist setups — base weights are frozen so forgetting is physically impossible, adapters are 10-30 MB each, swap at runtime. Reserve EWC for the case where we intentionally want full-weight specialization but need to retain some general capability (rare — usually better to use a LoRA). Backlog item EWC-1 (wire EWC into FORGE SFT/dialogue) is still valid but should be de-prioritized in favor of a LoRA-per-specialist path for Approach 2.
- [RESOLVED] Spec-2: Existing text's fp16 arithmetic is correct — 742M × 2 B ≈ 1.48 GB per specialist, theoretical max **~10 specialists** in 16 GB VRAM. **Realistic ceiling is 4-6 full-weight specialists** once KV cache (150 MB - 1.2 GB depending on ctx), activation buffers, and framework overhead are included. **Major revision:** LoRA changes this math entirely — **one frozen base model (1.5 GB fp16) + N LoRA adapters (~20 MB each)** supports dozens of specialists simultaneously with hot-swap at < 100 ms. For our 16 GB budget, LoRA gives **20+ effective specialists** vs. ~5 full-weight specialists, and with N× smaller disk footprint. Combined with Spec-1 verdict: **LoRA-per-specialist is the only serious multi-specialist path on 16 GB VRAM**. Full-weight merging (via `model_merging.py`) remains an option for creating the final consolidated model before deployment but not for simultaneous multi-specialist inference.
- [RESOLVED] Spec-3: Use the **EleutherAI LM Evaluation Harness** (github.com/EleutherAI/lm-evaluation-harness) as the harness — it's the community standard and covers all the benchmarks below with one runner. Per-specialist benchmark set:
  - **Reasoning specialist:** GSM8K (arithmetic CoT), MATH (hard math), ARC-Challenge (science MCQ), MMLU-Pro (general knowledge with CoT). **Skip AIME** — too hard for 742M, noise floor.
  - **Code specialist:** HumanEval + HumanEval+ (single-function synthesis), MBPP + MBPP+ (short programs), BigCodeBench-Lite if time permits.
  - **Vision specialist** (post Code-6 / Vision-1b): VQAv2 (general visual QA), TextVQA (OCR-in-image), GQA (scene reasoning), MMMU (multimodal reasoning). Skip COCO captioning — caption quality is not a good signal for multimodal reasoning specialists.
  - **Knowledge / general:** MMLU (0-shot + 5-shot), TriviaQA, NaturalQuestions. Include ARC-Easy and HellaSwag as sanity checks that general capability hasn't collapsed.
- **Action for Eval-1 / D-10:** wire the LM Eval Harness in as the `--benchmark` backend, driven by a YAML config that selects the relevant suite per specialist type. Much lower effort than reimplementing GSM8K/MMLU scorers internally.
- [RESOLVED] Spec-4 (merging): [enigma_engine/core/model_merging.py](enigma_engine/core/model_merging.py) has SLERP, TIES, and linear merge for combining specialists. TIES (Trim, Elect Sign, Disjoint Merge) is the state-of-the-art for merging fine-tuned variants without catastrophic forgetting of overlapping knowledge. Already built — can merge a code specialist back with the general model. Research still needed on which merge method works best at 742M.

---

### Generation & Mod System — Research Gaps

These are stated project goals (image, audio, video, 3D generation) but have zero research done:

**Image generation:**
- [RESOLVED] ImgGen-1: Stable Diffusion 1.5 fits in 16GB budget alongside Enigma. VRAM breakdown: Enigma 742M = ~1.5GB weights + ~0.3-0.5GB activations/KV cache = ~2GB total; SD 1.5 = ~5.5GB fp32 or ~2.8GB fp16 + ~1-1.5GB activations = ~4-6GB total; combined ~6-8GB leaves 8-10GB for batch_size=1 inference. SD 2.1 (6GB) and SDXL (7.5GB) fit in theory but require careful memory management. **Recommendation: SD 1.5 is the baseline (confirmed working); SD 2.1 marginally possible; SDXL requires quantization or sequential inference.**
- [RESOLVED] ImgGen-2: Direct simultaneous GPU residence not practical (both models ~4-6GB each = 8-12GB before overhead). Two deployment patterns: (a) **Sequential:** Enigma generates caption text → SD 1.5 loads, generates image from caption → offload SD to CPU RAM, process image with Enigma. Typical latency ~3-5s per image (not real-time but acceptable for asynchronous generation). (b) **Subprocess:** Launch SD in separate process (avoids Python memory sharing issues), communicate via file/socket, Enigma unloads from GPU during SD inference. Pattern used: `enable_model_cpu_offload()` in diffusers pipeline moves models to CPU when not actively computing. **Verdict: Sequential with CPU offload is the practical approach. No hard VRAM sharing; one model loads, the other sleeps on CPU RAM.**
- [RESOLVED] ImgGen-3: Real implementation. `StableDiffusionPipeline` via diffusers library. DPMSolver/Euler schedulers. Providers: placeholder (offline) / local / DALL-E / Replicate. Uses SD 1.5/2.1 pipeline.

**Audio generation:**
- [RESOLVED] Audio-1: Meta's **MusicGen comes in three sizes: 300M (small), 1.5B (medium), 3.3B (large)**. Paper arxiv:2306.05284 (*Simple and Controllable Music Generation*, Copet et al., Meta). MusicGen-small at **300M** is the smallest viable text-to-music option (training: 20K hours of licensed music, instrumental-only via HT-Demucs preprocessing). **VRAM requirement:** ~1-2GB for batch=1 16-second generation. **Alternatives considered:** AudioGen (Meta, speech + ambient sound, ~1.5B) — larger and not music-specific; Bark (Suno, ~400M TTS, not music). **Verdict: MusicGen-small 300M is the target for music generation.** AudioGen is larger and not needed if the goal is music-only. TTS is separate (Bark or existing pyttsx3).
- [RESOLVED] Audio-2: **TTS (speech synthesis) ≠ music generation.** Current code has TTS via pyttsx3/ElevenLabs. **Music generation is Audio-5 (new capability, separate).** Two different pipelines: (a) TTS for Enigma's voice output (already implemented, pyttsx3), (b) MusicGen for background/ambient music (not yet built, marked Audio-5 in P2). **Answer: use existing pyttsx3 for voice; add MusicGen-small separately for music. Goals are complementary, not exclusive.**
- [RESOLVED] Audio-3: All three are real implementations. `audiogen` = TTS via pyttsx3/ElevenLabs/system fallback (NOT music generation — see Audio-5 new gap). `voice` = voice synthesis with pycache (actively used). `transcriber` = speech-to-text transcription.
- [RESOLVED] Audio-4: [enigma_engine/core/audio_encoder.py](enigma_engine/core/audio_encoder.py) is a Whisper-style encoder (Conv1d → GELU → Conv1d stride=2 → sinusoidal pos → Transformer → RMSNorm). Presets: tiny(384-dim, 10M), base(512-dim, 25M), small(768-dim, 95M). All stdlib, no external deps.

**Video generation:**
- **[SUPERSEDED — Pass 148 currency check]** Video-1: The AnimateDiff + SD 1.5 block below was primary-source-accurate at resolution time but has been surpassed by permissively-licensed 2025 text-to-video models that still fit the 16 GB VRAM budget. **New primary:** **Wan 2.1 T2V-1.3B** (Alibaba, Feb 2025, Apache-2.0) — 8.19 GB VRAM on RTX 4090, generates 5-second 480p video in ~4 min, beats closed-source Runway Gen-3 on VBench in published numbers. **Primary alternative:** **LTX-Video 0.9.8 2B-distilled** (Lightricks, 2025) — real-time capable on consumer GPUs, but non-standard dev license. **Do not use** HunyuanVideo (13B, 45-60 GB VRAM — too big). AnimateDiff stays valid as a drop-in SD-based fallback for animation rather than true T2V. Original block preserved below for implementation reference:
  - [RESOLVED — historical] Video-1: **AnimateDiff (Guo et al. ICLR2024) + Stable Diffusion 1.5 is the minimum viable path.** AnimateDiff is a **motion module adapter** (~50-100MB additional params) inserted into SD's UNet to extract motion priors from video training data. Architecture: frozen SD 1.5 base + trainable motion modules at each block. **VRAM breakdown:** SD 1.5 = 5.5GB; AnimateDiff adapter overhead = negligible (~100MB); pipeline with `enable_model_cpu_offload()` and `enable_vae_slicing()` = ~6-6.5GB peak. Remaining 9.5GB available for batch_size=1 short video generation (16 frames, 25 diffusion steps). **Implementation:** diffusers library has native AnimateDiffPipeline (no custom code needed). **Alternative:** ModelScopeT2V at 4.5B is slightly smaller but less integrated into diffusers. **Verdict: AnimateDiff + SD 1.5 is production-ready, fits 16GB budget, use diffusers pipeline directly.**
- [RESOLVED] Video-2: Real implementation. AnimateDiff local pipeline + animated GIF fallback + Replicate API. Not a stub.

**3D generation:**
- **[SUPERSEDED — Pass 148 currency check]** 3D-1: Shap-E (arxiv:2303.12522, May 2023) and TripoSR (arxiv:2403.02151, Mar 2024) have been surpassed by 2024-2025 image→3D models with dramatically better mesh quality. **New primary:** **Hunyuan3D-2** (Tencent, arxiv:2501.12202, Jan 2025) — image→3D diffusion model, SoTA open reconstruction quality, fits 16 GB VRAM. **Primary alternative:** **TRELLIS-image-large** (Microsoft, arxiv:2412.01506, Dec 2024, MIT license) — 2.3M downloads/month, structured latent 3D representation, image→3D. Note: the official TripoSR HuggingFace model card itself now points users to its SF3D successor. **Text→3D path:** no open SoTA model at our scale matches image→3D quality; route text→3D as text→image (FLUX.1-schnell or SD 3.5) → image→3D (Hunyuan3D-2). Original block preserved below for implementation reference:
  - [RESOLVED — historical] 3D-1: **Shap-E (OpenAI, arxiv:2303.12522) is the primary text-conditioned 3D generation model at 350M params, fits 16GB budget.** Key details: **text→GLB (3D mesh)** in single forward pass. Trained on ~1M 3D objects from Google Scanned Objects. VRAM: ~2-3GB for inference. Alternatives: (a) **Zero123** (Liu et al., ~600M, image→novel-view synthesis, requires initial image), (b) **SyncDreamer** (Xu et al., ~400M, single-image to multi-view, requires input image), (c) **TripoSR** (Zhou et al., ~500M, single image→SDF). **Deployment:** Shap-E for text→3D (no input needed), Zero123/SyncDreamer for image→3D (avatar refinement/variation). **Verdict: use Shap-E (350M text→3D) as primary, keep SyncDreamer as fallback for image-based 3D refinement (avatar re-posing).** Combined: Enigma (1.5GB) + Shap-E (2-3GB) fits with room for activation buffers.
- [RESOLVED] 3D-2: Real implementation. Shap-E local generation (OpenAI, ~350M) + geometric primitives fallback + Replicate API. Not a stub.

**Haptic feedback prediction:**
- [RESOLVED] Haptic-1: Honest answer \u2014 **I do not know** of a standard, well-validated dataset + architecture for text-conditioned haptic prediction at the scale we'd need. What exists:\n  - **LMT Haptic Texture Database** (Strese et al. 2017): 108 real-world material textures with accelerometer recordings. Small, texture-only, not text-conditioned.\n  - **HaptiPedia** (~200 haptic icons): metadata + signal library, curated not algorithmic.\n  - **Research papers** on text\u2192haptic (Seifi et al. 2021, Schneider et al. 2023) use hand-crafted mappings or small MLP regressors trained on <1K examples. No open pretrained model exists.\n- **What's realistic for us:** a small MLP mod (~1M params) that takes a short text description + the current context embedding from Enigma and outputs a low-dim haptic parameter vector (amplitude, frequency, duration, texture class). Training data would have to be synthetic or bootstrapped from LMT + manual labeling.\n- **Verdict:** Haptic generation is a research project, not a near-term build. Keep in P3 / DEFERRED. Remove from the active research backlog \u2014 no realistic unlock by external literature search alone; would require novel data collection. Better to focus on the Vision / Code / Reasoning specialist track that has clear recipes.

---

### Additional Files Scanned (Pass 3 continued)

Final files read and assessed:

| File | What it is | Research implication |
|------|-----------|---------------------|
| [enigma_engine/router.py](enigma_engine/router.py) | TCP mod IPC bus + BackgroundTrainer | Router-1 resolved; dispatch layer still missing |
| [enigma_engine/core/rag.py](enigma_engine/core/rag.py) | Full BM25/TF-IDF RAG with adaptive chunking | Dense retrieval (embedding RAG) not present |
| [enigma_engine/core/kv_cache.py](enigma_engine/core/kv_cache.py) | Pre-alloc + H2O + INT4 + StreamingLLM cache | KV cache is very advanced; Inf-1/3 resolved |
| [mods/vision/](mods/vision/) | Screen capture + OCR, NOT LLM multimodal | Vision mod ≠ LLaVA; gap confirmed |
| [mods/codegen/](mods/codegen/) | TemplateCode + LocalCode (calls Enigma) + OpenAI | Code mod uses base 742M — no code specialist |
| [enigma_engine/core/memory.py](enigma_engine/core/memory.py) | Flat markdown memory, 200 facts, pattern extraction | No embedding-based retrieval |
| [enigma_engine/core/model_merging.py](enigma_engine/core/model_merging.py) | SLERP + TIES + linear merge | Can merge specialists after training |
| [enigma_engine/core/monologue.py](enigma_engine/core/monologue.py) | Inner monologue / journal / coherence gate | Phase 5, already built |
| [enigma_engine/core/adaptive_trainer.py](enigma_engine/core/adaptive_trainer.py) | TrainingPlan: basics→conv→commands→web curriculum | Adaptive pipeline exists, not yet used at scale |
| [mods/avatar/](mods/avatar/) | Full avatar: BoneController (all joints), GLB/GLTF/OBJ loader | Production-ready bone rig, output format TBD |

---

### New Gaps Found — Full Codebase Scan (May 2026)

These files/features exist in the codebase but had no research items. Added after reading all core/ files and all mods/.

**Inference speed:**
- [RESOLVED] Inf-5: `flash_attn_with_kvcache` IS the FA2 decode-path API (added in flash-attn v2.2 per the official changelog: "Optimize for inference... query sequence length = 1. The bottleneck here is to load KV cache as fast as possible, and we split the loading across different thread blocks"). Single kernel: updates KV inplace + attention + optional rotary, supports GQA (our 4:1), paged KV, sliding window. **Current code:** [enigma_engine/core/model_components.py](enigma_engine/core/model_components.py#L521-L527) explicitly gates Flash off when `use_cache=True`; decode falls through to `F.scaled_dot_product_attention` ([line 557](enigma_engine/core/model_components.py#L557)). SDPA on CUDA auto-dispatches to FA2 for seqlen_q>1 (prefill) but lacks the specialized 1-token decode kernel. **Realistic speedup at our workload (742M, 4K ctx, batch=1):** ~15-30% decode tok/s, NOT the 2× numbers quoted for 8K+/large models — at 742M the MLP + tied output head are a meaningful share of per-token cost, and SDPA's native FA2 prefill is already fast. **Blocker:** flash-attn README lists CUDA support for Ampere/Ada/Hopper only; Blackwell consumer SM120 (RTX 5090) is unverified, and Windows support is "might work... requires more testing." Installing the package on our target box is fragile. **Verdict:** keep deferred. If a stable Blackwell+Windows wheel lands alongside FA4, revisit. Moved to P3 (was P2 research). Not worth the dependency risk on the hot path for a ~20% gain when SDPA already covers CPU/MPS/any-dtype fallback cleanly.

**Progressive growing:**
- [RESOLVED] Grow-1: Our [enigma_engine/core/progressive_growing.py](enigma_engine/core/progressive_growing.py) already implements Net2Net (Chen et al. 2016) width expansion + identity-init depth expansion. Published guidance on *when* to grow:
  - **Chinchilla-optimal tokens first.** Don't grow until the source model has seen at least Chinchilla-optimal data for its current size (~20 tok/param — ~15B tokens for 742M). Growing earlier wastes the smaller model's training signal; the larger architecture underfits because the expansion happens before the smaller weights are well-converged. Our 35 tok/param data budget is ideal: train 742M → Chinchilla-optimal (~15B tokens), then optionally grow to ~2.5B and continue on remaining tokens.
  - **Triggering signal.** Standard practice (MSG/stacking papers, Gong et al. 2019 *Efficient Training of BERT by Progressively Stacking*): grow when the loss curve plateaus at the current size. Operationally: val loss slope over the last 500-1000 steps falls below a threshold (e.g. < 1% relative improvement per 1000 steps).
  - **Data budget after growing.** Empirically 50-100% of the pre-grow token count on the new architecture before final convergence. So xl → xxl expansion at ~15B tokens should be followed by another ~10-15B tokens at 2.5B scale. We don't have that token budget right now (26B total collected, already committed to xl completion).
- **Verdict:** keep `progressive_growing.py` as infrastructure; do NOT schedule xl → xxl on current data budget. Revisit after N-10 completes and if we've collected another 10B+ tokens.
- [RESOLVED] Grow-2: Standard Net2Net / MSG / gradual-stacking practice is to **re-warmup** after expansion, NOT reset LR to the initial value. Recipe from multiple stacking papers (Gong 2019, Shen 2022 *Staged Training*, bert2BERT Chen 2022):
  1. After expansion, set LR to a small fraction of the current LR (e.g. current_lr × 0.1 or directly match the min_lr from the cosine schedule).
  2. Re-warmup over 100-500 steps back to the current LR (NOT back to the original peak LR — the smaller model's final LR is already past peak).
  3. Resume the existing cosine/WSD decay from that point.
- **Rationale:** a full warmup-from-scratch would destroy the small model's learning; no warmup at all lets gradients from the zero-initialized new params destabilize the rest of the network in the first few steps.
- **Action if/when xl → xxl is scheduled:** add a `--post-grow-warmup-steps` flag (default 200) and `--post-grow-lr-scale` (default 0.1) to the training CLI. Not urgent — tied to Grow-1's "not on current data budget" verdict.

**Speculative decoding (Medusa):**
- **[Pass 148 currency note — training recipe still valid, inference-time method superseded]** Our joint-loss MTP training path matches Medusa-2 and remains the correct *training* recipe (predict_heads stay useful as an auxiliary signal). For *inference-time* speculative decoding, **EAGLE-2** (Li et al., arxiv:2406.16858, Jun 2024) is now the preferred integration target — lossless, 3.05–4.26× speedup, 20–40% faster than EAGLE-1 (itself ~1.6× faster than Medusa). EAGLE-3 exists but has not been independently verified in this pass. If/when we add speculative decoding to the inference path, target EAGLE-2 not Medusa, even though Medusa-2 training hooks (predict_heads) stay.
- [RESOLVED] Medusa-1: The original *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* (Cai et al., arxiv:2401.10774, ICML 2024) proposes **two training recipes**:
  - **Medusa-1:** base model is frozen, Medusa heads are trained separately (cheap, fast, ~2× speedup, lower acceptance).
  - **Medusa-2:** heads and base are trained jointly with a weighted loss (base loss + λ × Medusa loss). Higher acceptance (~3× speedup) but requires re-fine-tuning the base.
- **Our approach matches Medusa-2:** `predict_heads` are included in the main loss at [enigma_engine/core/model.py](enigma_engine/core/model.py#L496-L518) lines 496-518, `loss = loss + mtp_loss / len(predict_heads)`. This is the higher-quality recipe.
- **Caveat:** Medusa-2 paper uses weight λ=0.2 and warms up Medusa heads first with Medusa-1 style for a few hundred steps before enabling joint. Our code adds them unweighted (λ=1.0 effectively via `/len(heads)`) from step 0. **Potential improvement:** add a warmup period and a `mtp_loss_weight` config (default 0.2 per Medusa-2, ramped from 0 over first ~500 steps). Log as **Medusa-1a**.
- Verdict: current joint training is correct direction; weighting is the tunable knob.
- [RESOLVED] Medusa-2: Original Medusa paper benchmarks on Vicuna-7B and Vicuna-13B — no sub-1B numbers published. **Why it matters at our scale:** speculative decoding's speedup comes from amortizing memory-bandwidth-bound decode across multiple tokens per forward pass. At 742M, decode IS memory-bandwidth bound on 5090 (the weights + KV cache don't fit L2, main bottleneck is HBM reads), so the speedup mechanism DOES apply. However:
  - **Acceptance rate is the question.** At 742M, the model is less confident → Medusa head predictions diverge from the base output more often → acceptance drops. Community reports on sub-1B Medusa variants (informal, no paper) suggest ~1.5-1.8× speedup vs ~2-3× at 7B+.
  - **Batch=1 only.** Speculative decoding gains collapse at batch > 4 because the base model's forward pass already amortizes across the batch. Our single-user inference path is the best case.
- **Verdict:** likely worthwhile at 742M batch=1 but the ~1.5-1.8× speedup is not a slam dunk vs. the 98M parameter cost of `predict_heads` (MTP-2a/b). If we drop to `n_predict_heads=1` per MTP-2a, Medusa effectively becomes 2-token lookahead — still useful, half the cost. **Action:** measure acceptance rate via a `--benchmark medusa-acceptance` flag on our real workload before committing. Log as backlog item **Medusa-2a**.

**Image generation mod upgrade:**
- [RESOLVED] ImgGen-4: SDXL-Turbo is SDXL-family and uses SDXL pipeline classes (`StableDiffusionXLPipeline` / `AutoPipelineForText2Image`), so it is not a drop-in replacement for SD1.5 pipeline call sites. It can run in 1-4 steps, but requires explicit SDXL load/wiring and has higher VRAM pressure than the current SD1.5 path. Decision: keep SD1.5 default; only add SDXL-Turbo behind a separate provider flag if we explicitly accept the heavier SDXL branch.

**Audio: music generation gap:**
- [RESOLVED] Audio-5: Code verification confirms `mods/audiogen/audiogen.py` is **TTS-only** (LocalTTS via pyttsx3, ElevenLabs cloud, system fallback) and has no music model path. Minimum viable music generation path is **MusicGen-small (300M)** as a separate provider (not a TTS replacement). From MusicGen model cards/paper (arxiv:2306.05284): sizes are 300M / 1.5B / 3.3B; small is the only practical local option for this stack. Budget fit: Enigma (~1.5-2 GB effective) + MusicGen-small (~1-2 GB inference) stays well under 16 GB, but generation is slow enough that async job-style UX is preferred. **Decision:** keep TTS in `audiogen` unchanged; add a separate `musicgen` provider/path (new mod or provider in audio mod) using MusicGen-small first, medium/large deferred.
- **[Pass 148 currency check]** Audio-5b: License + currency revision. **MusicGen weights are CC-BY-NC-4.0** (non-commercial only) — fine for personal/research use, NOT redistributable in a commercial build. **Stable Audio Open 1.0** (arxiv:2407.14358, Stability AI, July 2024) is now the better-licensed alternative for general audio/SFX: stereo 44.1 kHz, up to 47 s, T5 + DiT, **Stability AI Community License** (free for individuals + businesses with annual revenue under $1M). **Stable Audio Open Small** (arxiv:2505.08175, May 2025) is the lowest-VRAM option: up to 11 s, transformer-DiT, ARM-CPU optimized, 8-step pingpong sampler — generates on a phone, comfortably fits beside Enigma on 16 GB. **Decision (revised):** for music-specifically, MusicGen-small remains acceptable for offline/personal use given the CC-BY-NC license. For general audio + SFX + ambient, prefer Stable Audio Open Small (< $1M revenue org) → Stable Audio Open 1.0 → MusicGen-small in that order. Wire each as a separate provider behind a single `audiogen.music` interface.
- **[Pass 148 currency check]** Audio-ASR (new): The repo currently uses Whisper variants for transcription. As of mid-2025, **NVIDIA Parakeet-TDT-0.6B-v2** (May 2025, CC-BY-4.0, 600M params) leads HuggingFace's Open-ASR leaderboard with **6.05% mean WER and RTFx 3386** versus Whisper-large-v3-turbo at 7.83% / RTFx 200 — both more accurate AND ~16× faster on the same hardware, English-only, single-pass up to 24 minutes, with word timestamps + punctuation. NVIDIA also released **Parakeet-TDT-0.6B-v3** with 25 European languages (multilingual). **Moonshine** (arxiv:2410.15608, Oct 2024, MIT, 27M tiny / 61M base) is the right choice for low-latency edge / always-on streaming. **Decision:** swap default `mods/transcriber` English path to Parakeet-TDT-0.6B-v2; keep Whisper-large-v3 as multilingual fallback until Parakeet-v3 weights are validated locally; add Moonshine-base as the streaming/edge path. Logged as **Audio-ASR-1**.

**Avatar mod:**
- [RESOLVED] Avatar-1: Integration does **not** require training data for a first working version. Code path is already command-driven: `AvatarBrick` receives `avatar.bone` / `avatar.expression` over router TCP JSON, and `BoneController` enforces anatomical limits. Practical integration sequence: (1) map Enigma internal state + response tags to a small pose/expression preset table, (2) emit router commands at message boundaries, (3) optionally add viseme/lipsync later. Training data is only needed for learned motion style, not for baseline expressive control. **Priority call:** medium (P3/P2 boundary) behind reasoning/vision tasks, but low implementation risk because infrastructure already exists.

**Auto-research:**
- [RESOLVED] AutoResearch-1: `should_auto_research()` called in [enigma_engine/gui/gui_logic_chat.py](enigma_engine/gui/gui_logic_chat.py#L239) `::_gen()` (line 239) during every chat message — actively fires for substantive questions. NOT unused. Quality impact is qualitative and unverified — user should assess.
- [RESOLVED] AutoResearch-2 (confidence gap): Confirmed from code that `should_auto_research()` is query-keyword based and runs **before** model generation, so it cannot react to model uncertainty. Decision: implement in two stages, not one. **Stage A (fast, no training):** add a post-generation uncertainty pass in chat loop; if output contains calibrated uncertainty markers OR low-evidence patterns, trigger one retry with `auto_research()` context attached. **Stage A SHIPPED Pass 153** — `score_uncertainty()` + `should_retry_with_research()` in [auto_research.py](enigma_engine/core/auto_research.py). Wiring into [gui_logic_chat.py L239](enigma_engine/gui/gui_logic_chat.py#L239) `_gen()` is a follow-up pass. **Stage B (proper long-term):** add inline tool token (`<search>...</search>`) support so model can self-initiate research mid-generation. This requires generation-loop interruption/resume plus training examples that demonstrate when to emit `<search>`. **Verdict:** gap is real; staged plan is now clear and linked to existing files (`auto_research.py`, `gui_logic_chat.py`).

**Continuous learning:**
- [RESOLVED] Continuous-1: Code review confirms continuous trainer currently does **full-parameter updates** at `lr=1e-5` with replay buffer `maxlen=1000`, periodic replay retrain every 200 examples, and no EWC penalty. This is effective for short-term adaptation but has real long-horizon drift risk (matches continual fine-tuning forgetting concerns already cited in Spec-1). Practical decision: keep architecture, but treat current defaults as aggressive. Recommended safe defaults for always-on mode: `learning_rate=1e-6` (or 2e-6 max), keep replay buffer capped by recency (1000-2000), and require periodic evaluation gates before checkpoints are promoted. If fast adaptation is desired, do it in adapter/LoRA space rather than full-weight online updates.

**EWC wiring (confirmed gap):**
- [BUILD] EWC-1: [enigma_engine/core/ewc.py](enigma_engine/core/ewc.py) is standalone — zero FORGE integration. For Approach 2 specialist fine-tuning to prevent catastrophic forgetting, EWC must be wired: (1) after base training, compute EWC from a sample of general data, (2) during specialist fine-tuning, add `ewc.penalty(model)` to the loss. Requires adding EWC to the SFT/dialogue training path in [enigma_engine/core/training.py](enigma_engine/core/training.py) and an EWC section in FORGE GUI.

---

### Personality System — Audit (Pass 7, corrected Pass 8)

**Design intent (from user):** Personality is something the AI *develops*, not something the user configures. User should not be able to influence the AI's own character. Roleplay (user says "act as X") is fine — that is character-acting, separate from identity.

**What exists:**

| Component | File | What it is | Correct direction? |
|---|---|---|---|
| `ai_profile.personality` dict (tone/verbosity/formality/humor) | [enigma_engine/core/ai_profile.py](enigma_engine/core/ai_profile.py#L105) line 105 | USER-SET traits dict | **NO** — user-controlled personality = user influencing AI character |
| `model_context.personality` string | [enigma_engine/core/model_context.py](enigma_engine/core/model_context.py#L123) line 123 | Display label for identity card | Neutral — display only |
| `emotional_state` (5 dimensions: valence/energy/engagement/trust/frustration) | [enigma_engine/core/model_context.py](enigma_engine/core/model_context.py#L135) line 135 | AI-computed per message, decays, used for BackgroundTrainer engagement score | **PARTIAL** — AI-computed (not user-set), feeds training signal via BackgroundTrainer — this is the right direction |
| FORGE Distillation "personality" category | [enigma_engine/gui/gui_forge_new_modes.py](enigma_engine/gui/gui_forge_new_modes.py#L1259) line 1259 | Teacher generates warm, genuine responses → student SFT learns the style | **YES** — personality baked into weights via training |
| Training brief "Personality" field | [enigma_engine/gui/gui_forge.py](enigma_engine/gui/gui_forge.py#L64) line 64 | User description for guiding teacher during FORGE training | **MIXED** — user can seed the initial character during a one-time training run, but shouldn't be able to change it at runtime |

**Pass 7 audit self-correction (revised Pass 9):**

- [RESOLVED] Personality-1 (emotional_state injection): N-22 (Pass 125) already injects `emotional_state` into the system prompt via [enigma_engine/gui/gui_logic.py](enigma_engine/gui/gui_logic.py#L335) `::_build_gui_context()` line 335 as `[Internal State: valence=X, energy=Y, ...] Let this state color your tone naturally — do not announce it.` This is **correct**: emotional_state is AI-computed from sentiment analysis (`compute_engagement_score` in [enigma_engine/core/sentiment.py](enigma_engine/core/sentiment.py)), not user-set. Giving the AI awareness of its own computed state ≠ user configuring the AI's personality. Pass 8 retracted this as "wrong direction" — that retraction was itself wrong. No action needed. Also feeds BackgroundTrainer engagement weight in [enigma_engine/gui/gui_logic_chat.py](enigma_engine/gui/gui_logic_chat.py#L1252) line 1252 — both consumers are correct.
- [CRITICAL] Personality-2: `ai_profile.personality` dict (tone/verbosity/formality/humor) is USER-SET via profile files in [profiles/](profiles/). The earlier suggestion to inject it into the system prompt would have meant the user directly configures the AI's character — which contradicts design intent ("personality is something the AI develops, not something the user configures"). **Do not inject this dict.** The dict itself is the problem — see Personality-3.

**What is actually a gap:**

- [ARCH-GAP] Personality-3: `ai_profile.personality` user controls (tone/verbosity/formality/humor) exist in the AI's default profile and are presented as configurable. This is the wrong abstraction for the AI's own identity. These controls should be **scoped only to roleplay/character profiles** (where the user is defining a fictional character to act as), not to the AI's base identity. The AI's base profile should have no user-configurable personality traits. File: [enigma_engine/core/ai_profile.py](enigma_engine/core/ai_profile.py#L105) line 105, profile files in [profiles/](profiles/). Note: this is a design/UX decision, not a one-line code fix.
- [ARCH-GAP] Personality-4: No clear separation between **identity** (the AI's own developed character, baked into weights) and **roleplay** (user-requested character-acting, prompt-based for that session). The same profile system handles both. A user loading [profiles/assistant.json](profiles/assistant.json) looks identical to the AI just being itself. This needs a design decision: should "be yourself" be a hardcoded state (no profile), and profiles only ever mean "act as this character"?
- [BUILD] Personality-5: FORGE Distillation "personality" category exists ([enigma_engine/gui/gui_forge_new_modes.py](enigma_engine/gui/gui_forge_new_modes.py#L1259) line 1259) and generates character training data via teacher. This is the correct build path for personality-in-weights. **It has NOT been run as part of any training plan yet.** The emotional roadmap calls it "Step 1b" but it's never been scheduled. Recommend running distillation with the "personality" category as part of the next training cycle to bake initial character traits.
- [RESOLVED] R-PERSONALITY-1 (reframed): Stable personality should be weight-trained, not runtime-user-configured. Concrete direction for this codebase: (1) build a fixed personality corpus through FORGE personality distillation, (2) run a dedicated SFT pass to anchor style/values, (3) disable runtime personality knobs for base identity, and (4) keep roleplay as explicit temporary overlays only. Minimum viable seed should start around ~500 high-quality personality examples (100 is too fragile) and can scale later; BackgroundTrainer reinforcement should include guardrails so engagement does not reward shock/flattery behavior.

**Roleplay separation (design note):**
- AI's own personality → trained into weights (FORGE Distillation "personality" category) + reinforced by BackgroundTrainer engagement → not prompt-injectable by user, not configurable via profile personality dict
- Character acting → user loads a character profile or says "act as X" → prompt-based for session scope → clearly temporary
- The code does not currently enforce this separation — any profile can override system_prompt which effectively overrides personality. An architectural guard is needed.

**Unpredictability (design note — Pass 10):**

- [RESOLVED] R-UNPREDICT-1: "Organic" / unpredictable behavior must split **behavior** (surprising but signal-driven) from **infrastructure** (deterministic, seedable). Literature-backed decision:
  - **ReAct (arxiv:2210.03629):** interleave reasoning and external actions to reduce hallucination; supports action-on-demand rather than always-on tool calls.
  - **Toolformer (arxiv:2302.04761):** model can learn when/what/how to call tools; trigger should be learned/signal-based, not keyword-only.
  - **Self-RAG (arxiv:2310.11511):** adaptive retrieval via reflection tokens outperforms fixed retrieval; retrieval should be conditional.
  - **CRAG (arxiv:2401.15884):** retrieval-quality evaluator and confidence degree should govern fallback to broader search.
  - **Enigma implementation policy:** Stage A add post-generation uncertainty gate (confidence + novelty + retrieval-quality score) before auto-research retry; Stage B add inline `<search>` token handling in generation loop. Keep hard off-switch + reproducible seed path for all stochastic branches.
  - **Status:** research complete; now an implementation backlog item (AutoResearch-2), not an open research item.
  
  Concrete places to add controlled unpredictability in this codebase:
  1. **Self-initiated research** — replace `should_auto_research()` keyword match with a confidence signal derived from model logits + memory novelty. Fires when the model is uncertain, not when the query contains "what is". Links: AutoResearch-2. File: [enigma_engine/core/auto_research.py](enigma_engine/core/auto_research.py) `::should_auto_research()`.
  2. **Mood-weighted replay** — BackgroundTrainer already consumes engagement scores; weight replay sample selection by `emotional_state` so the same conversation gets revisited differently depending on internal state. Same input, different lesson. File: [enigma_engine/router.py](enigma_engine/router.py) BackgroundTrainer + N-25 idle callback.
  3. **Monologue variance** — `monologue.py` self-reflection currently uses a fixed prompt. Shape the internal prompt by current `emotional_state` (not user-visible) so reflection drifts with mood. Slow, consistent personality drift baked back into memory. File: [enigma_engine/core/monologue.py](enigma_engine/core/monologue.py).
  4. **Curiosity token** — let model emit `<search>query</search>` mid-generation when uncertain (linked to R-PERSONALITY-1 and AutoResearch-2 option (a)). Trigger auto-research inline. Unpredictable *when*, predictable *that* it fires on uncertainty.
  **Non-negotiable constraints:**
  - Training runs must remain seedable (reproducible crashes + ablations)
  - Tests must remain deterministic (no flakes from "unpredictable" paths)
  - Every unpredictable feature needs an off-switch (debug flag) and a seed (reproduction)
  - No `random.random()` or `torch.rand()` without a seed source tied to either (a) training seed, or (b) an internal signal (emotional_state, confidence, novelty)
  **Anti-patterns to block:**
  - "Let it grow naturally" used as excuse to skip curriculum/structure — chaos is not personality
  - Neural reward model that reinforces confident-sounding noise (GRPO-4 — already logged)
  - Unseeded hashing for dedup (already a Learned Principle)
  Status: research resolved in Pass 146. No code changes yet; implementation follow-up is AutoResearch-2 staged build.

**Tests:**
- [RISK] Test-1: Three GRPO tests in [tests/test_training.py](tests/test_training.py#L3284) lines 3284–3340 are structural (`inspect.getsource`) tests checking for `.clamp(-20, 20)` and `unbiased=False` string literals. These test HOW not WHAT. A refactor that moves the logic (e.g. into a helper function) will fail these tests even if the math is correct. Per testing rules, structural tests are last resort. These particular checks (numeric correctness guards) are borderline acceptable because wrong values cause silent NaN/inf. But there is no behavioral end-to-end test of the reward loop — nothing verifies that the trainer actually improves responses when given a meaningful reward function. Recommendation: add one behavioral integration test that gives a mock reward_fn returning 1.0 for longer responses and verifies the model's output length increases after training. Keep the structural tests for the numeric guard but don't add more.

**RAG system:**
- [RESOLVED] RAG-1: [enigma_engine/core/rag.py](enigma_engine/core/rag.py) has a full BM25/TF-IDF pipeline: `TfidfVectorizer` (Okapi BM25 with IDF weighting), `RAGIndex`, adaptive chunking (sentence-boundary aware), `index_directory()`. Already built. NOT in research list at all.
- [RESOLVED] RAG-2: Small dense retrievers are now concrete. `sentence-transformers/all-MiniLM-L6-v2` is 22.7M params with 384-d embeddings; `BAAI/bge-small-en-v1.5` is 33.4M params with 384-d embeddings and stronger retrieval-oriented benchmarks. Both are lightweight enough for CPU-side embedding service in this project. **Decision:** use `bge-small-en-v1.5` as primary (better retrieval quality), `all-MiniLM-L6-v2` as fallback/minimal dependency option. Do not use Enigma hidden states as replacement embeddings for retrieval: they are not contrastive-retrieval tuned, are more expensive per chunk, and couple retrieval quality to LM checkpoint drift.
- **[Pass 148 currency check]** RAG-2b: **Qwen3-Embedding-0.6B** (arxiv:2506.05176, Jun 5 2025, Apache-2.0) is the current SoTA in the sub-1B class and a clean upgrade over `bge-small-en-v1.5`. Verified facts: 0.6B params, 32K context, MRL output dim selectable from 32 → 1024, MTEB multilingual mean **64.33** (vs BGE-M3 59.56, multilingual-e5-large 63.22), 100+ languages, supports `flash_attention_2`. The 8B variant is #1 on MTEB multilingual at **70.58** but is too heavy to co-reside with Enigma on 16 GB. **Decision (revised):** Qwen3-Embedding-0.6B = primary; bge-small-en-v1.5 = lightweight fallback when Qwen3 weights aren't available; all-MiniLM-L6-v2 = absolute-minimum fallback. Keep BGE-M3 in mind only when hybrid dense + sparse + ColBERT scoring is explicitly needed (Qwen3-Embedding is dense-only).
- [RESOLVED] RAG-3: Wiring is currently split but compatible: GUI chat may inject web context via `auto_research()` before generation, and `_prepare_chat()` independently injects local RAG document context when `_rag_index` is active. This means web and local retrieval can both fire in the same turn. **Decision policy:** (1) local RAG first for user/local corpus questions, (2) web search only when local RAG confidence is low or query is time-sensitive/current-events, (3) neither for simple conversational requests. Implement as a single lookup orchestrator that scores source confidence and appends one merged context block, instead of two independent injections.

**KV cache (Inf-1/3 resolved):**
- [RESOLVED] Inf-1: KV cache IS pre-allocated. `KVCache` allocates full `max_seq_len` upfront, uses in-place slice updates. No dynamic fragmentation.
- [RESOLVED] Inf-3: `H2OKVCache` (Heavy Hitter Oracle) handles VRAM-bounded long sessions by evicting low-attention tokens. `StreamingLLMCache` provides infinite context via attention sinks (first 4 tokens always kept). `TurboQuantKVCache` uses INT4 packing. PagedAttention not present, but these mechanisms cover the same need.

**Vision mod distinction:**
- [RESOLVED] Vision-5 addendum: [mods/vision/vision.py](mods/vision/vision.py) is screen capture + OCR (`ScreenCapture`, `OCR` via tesseract/easyocr, `ImageAnalyzer`). This is computer vision for what's ON SCREEN, NOT multimodal LLM vision (understanding image content via the language model). The `vision_hidden_size` projection in ForgeConfig is for the LLM multimodal path — completely separate. Both are gaps for research.
- [RESOLVED] Vision-6: Priority is (A) wire LLM vision first. OCR already covers screen text extraction, while project goals need image understanding connected to language reasoning. The codebase already has the foundation (`core/vision_encoder.py` + multimodal hooks), so the shortest path to meaningful capability is LLaVA-style wiring/training (Vision-1b + Code-6). BLIP-2-class captioning can be a later enhancement, not the first milestone.

**Code mod:**
- [RESOLVED] Code-9: [mods/codegen/codegen.py](mods/codegen/codegen.py) calls the base 742M Enigma model for code generation via `LocalCode`. No specialized code model. The mod works but quality depends entirely on base model capability. A code-specialized fine-tune (Approach 2) would improve this significantly.

**Memory system:**
- [RESOLVED] Memory-1: [enigma_engine/core/memory.py](enigma_engine/core/memory.py) is flat text + regex pattern extraction (200 fact max). Retrieves relevant facts by keyword search at query time (not embedding-based). Simple but functional for user preferences.
- [RESOLVED] Memory-2: Yes, episodic memory can be built directly on the existing RAG path without a new memory subsystem. Current state: `core/memory.py` stores durable user facts; journal stores reflections; chat history already exists per session/model. Practical design: write each completed conversation/session summary as a retrievable document (timestamp + topic tags + key decisions) into a dedicated memory corpus, then query it through the same `RAGIndex.query()` path used for docs. This gives "last week recall" behavior with existing infrastructure. Working-memory can stay lightweight (recent topic list + current task summary) and be injected in prompt, while episodic memory stays retrieval-based.

**Model merging (Approach 2 enabler):**
- [RESOLVED] Merge-1: [enigma_engine/core/model_merging.py](enigma_engine/core/model_merging.py) has SLERP, TIES, and linear merge. TIES handles sign conflicts when merging multiple fine-tuned models. This is the right tool for combining specialists after fine-tuning. Already built and tested.
- [RESOLVED] Merge-2: Keep `density=0.2` as the default baseline for TIES at 742M as well. Code already uses `density: float = 0.2` in `ties_merge()`, which matches the common "top changed parameters" trimming regime from TIES usage. There is no universal scale-law value proven for every checkpoint/task pair, so the practical answer is calibration, not a one-size constant: sweep {0.1, 0.2, 0.3} on a held-out benchmark and pick best aggregate score. Until that ablation is run, 0.2 remains the correct default.

**Inner monologue:**
- [RESOLVED] Mono-1: [enigma_engine/core/monologue.py](enigma_engine/core/monologue.py) is Phase 5 inner monologue. Modes: disabled/journal_only/automatic. Coherence gate at 0.7 threshold (heuristic scorer). Journal stored as `data/model_contexts/<model_key>/journal.json`. Not on any research list.
- [RESOLVED] Mono-2: Keep the heuristic coherence gate as the primary filter for now. It is cheap, deterministic, and transparent, while the RL reward models are trained for preference/reward tasks and are not calibrated as a monologue-truth discriminator. Using RewardModel as the only gate risks swapping one heuristic for another opaque scorer. Practical path: retain heuristic gate (`score_coherence`), and add optional secondary rerank later (small classifier or reward model) only after collecting false-positive/false-negative data from real journal entries.

**Adaptive training pipeline:**
- [RESOLVED] Adaptive-1: [enigma_engine/core/adaptive_trainer.py](enigma_engine/core/adaptive_trainer.py) has `TrainingPlan` (stages: basics→conversation→commands→web), adaptive difficulty, TRAINER-evaluates-STUDENT loop. This is the AI-supervised curriculum. Not yet used for any of the planned approaches.
- [RESOLVED] Adaptive-2: Evaluation reliability depends more on evaluator quality margin than on exact parameter ratio. Code confirms current loop has teacher generate tests and judge student answers (`gui_forge_adaptive.py` Phase 3), so self-evaluation with the same-scale/student-like model is biased and weak. Decision: for meaningful adaptive gating, TRAINER should be a stronger external evaluator (e.g., Qwen3-30B class) while STUDENT remains 742M. If same-scale evaluator must be used, restrict it to coarse pass/fail and require external benchmark checks before advancing major stages.

**Avatar system:**
- [RESOLVED] Avatar-2: Avatar mod is production-ready. `AvatarBrick` main controller. `BoneController` has anatomical limits for ALL bones (head, neck, spine, arms with elbow/wrist, legs with knee/ankle, all 10 fingers, jaw, left+right eyes). `ModelManager` loads GLB/GLTF/OBJ via pygltflib or manual parsing fallback. Communicates over TCP to the ModRouter. NOT a stub. (head, neck, spine, arms with elbow/wrist, legs with knee/ankle, all 10 fingers, jaw, left+right eyes). `ModelManager` loads GLB/GLTF/OBJ via pygltflib or manual parsing fallback. Communicates over TCP to the ModRouter. NOT a stub.
- [RESOLVED] Avatar-3: Standard mapping target should be **FACS/ARKit-style coefficients**, then converted to current bone/expression commands. External references confirm the pattern: FACS action units are the common semantic layer for facial animation, and ARKit blendshape coefficients are normalized 0.0-1.0 controls per feature. Practical design for this codebase: emotion state (`valence/energy/...`) → small coefficient vector (brow, jaw_open, mouth_smile, eye_squint, head_pitch/yaw) → router command packets (`avatar.expression` + `avatar.bone`). This closes the loop without retraining the avatar model. Dataset requirement is optional at this stage; rule-based mapping is enough for v1.
- [RESOLVED] Avatar-4: Current consumers are **not** expecting BVH today. Verified outputs by code: `mods/threed/threed.py` emits OBJ (builtin), PLY/PKL (local Shap-E path), or GLB/OBJ (Replicate path) into `outputs/3d/`. Avatar runtime output is live router events (`avatar.bone_moved`, etc.) carrying JSON rotation triples, not baked animation files. **Decision:** keep live JSON stream as runtime protocol; if export is needed later, standardize on GLB+animation clips for interchange and keep BVH as optional mocap export only.

---

### Codebase-Specific Gaps (Must Verify Before Changes)

These are areas where the code may have bugs or missing wiring, found during audit:

- [RESOLVED] Code-1: D-8 FIXED. Both optimizer paths updated. Weight-tied output head is the same tensor as tok_embeddings — optimizer sees it once, counted in no_decay correctly.
- [RESOLVED] Code-2: Confirmed. `build_packing_masks()` → `attention_mask_2d` → model.forward(). Training.py lines 3183-3240.
- [RESOLVED] Code-3: WSD is the default. `schedule_type: str = "wsd"` in TrainingConfig.
- [RESOLVED] Code-4: β2=0.95 hardcoded in TrainingConfig.
- [RESOLVED] Code-5: RMSNorm epsilon is 1e-6. Correct.
- [BUILD] Code-6: No FORGE training path for vision projection training exists. `model.forward()` accepts `vision_features` and the projection is wired (Vision-5 resolved), but there is no training mode that: (1) loads a CLIP encoder, (2) passes image tensors through it, (3) trains only the projection Linear while freezing the LLM body. This requires a new FORGE training mode ("Vision Projection" or an extension of LoRA mode). Search terms: "LLaVA stage 1 training" (projection only), "LLaVA stage 2 training" (full SFT). File to create: a method in [enigma_engine/gui/gui_forge_training.py](enigma_engine/gui/gui_forge_training.py) mirroring `_start_vision_training()` but with frozen transformer.
- [RESOLVED] Code-7: [enigma_engine/core/lora_utils.py](enigma_engine/core/lora_utils.py) is a full LoRA/QLoRA trainer with PEFT + bitsandbytes + accelerate support. Classes: `LoraConfig`, `QLoraConfig`, `OffloadConfig`, `LoraTrainer`. Complete training workflow with `save_adapter()`. Used by RLHF/SelfPlay trainers for reference policy via `create_lora_model()`.
- [RESOLVED] Code-8: EWC ([enigma_engine/core/ewc.py](enigma_engine/core/ewc.py)) is standalone — NOT wired into FORGE GUI. Uses empirical Fisher information (squared gradients), diagonal approximation, `penalty()` returns scalar tensor for adding to loss. For Approach 2 specialist training, EWC must be manually wired into the relevant training path.

---

### Evaluation — Real Gaps

The codebase has `python run.py --benchmark` but it's unclear what it actually tests:

- [CONFIRMED GAP] Eval-1: `--benchmark` calls `run_coherence_benchmark()` from [enigma_engine/core/monologue.py](enigma_engine/core/monologue.py). It is a **self-reflection coherence test** — 20 reflective prompts, scored by `score_coherence()` (lexical diversity heuristic), reports ready/marginal/not_ready. **Does NOT test GSM8K, MMLU, HellaSwag, HumanEval, or any academic benchmark.** N-7 plans to "benchmark after pre-training" but the current `--benchmark` command is useless for measuring reasoning/knowledge quality. [BUILD] before N-7: implement at minimum GSM8K (parse final number, test 1319 examples, ~20 min on RTX 5090) in [enigma_engine/core/training_evaluation.py](enigma_engine/core/training_evaluation.py). Reference: lm-evaluation-harness on GitHub for prompt format.
- [RESOLVED] Eval-2: A full exhaustive suite is unlikely to fit <30 minutes at 742M (especially HellaSwag 10K + full MMLU). Practical <30 min regression run: GSM8K full or capped subset (depending on current decode speed) + HumanEval full, then sampled HellaSwag/MMLU slices for smoke checks. Full-set HellaSwag/MMLU should be treated as longer offline jobs for reporting.
- [RESOLVED] Eval-3: Minimum scoring stack: GSM8K final-number extraction with strict match, HumanEval pass@k via execution tests, and exact-match/multiple-choice accuracy for tasks with deterministic references. Use judge-model scoring only as a fallback for truly free-form items, and report it separately from deterministic metrics.
- [CONFIRMED GAP] Eval-4: Since `--benchmark` is coherence-only (Eval-1), there is **no few-shot prompting infrastructure at all** — no chain-of-thought prompting, no demonstration examples, no MMLU 5-shot. Before running real benchmarks: (1) implement few-shot prompt builder (prepend K examples before the test item), (2) verify the model can follow few-shot format. Note: zero-shot vs 5-shot MMLU typically differs by 10-20pp — always specify which when reporting.

---

### Critical Items (Highest Priority)

These must be resolved before N-6 restart or will cause silent training problems:

- [RESOLVED] **CRIT-1**: D-8 FIXED. Both `_setup_optimizer()` and `_build_llrd_param_groups()` include `or 'embed' in name` in no_decay. Embeddings now correctly excluded from weight decay.
- [RESOLVED] **CRIT-2**: β2=0.95 is hardcoded in TrainingConfig (`adam_beta2: float = 0.95`). Not PyTorch default. Correct.
- [RESOLVED] **CRIT-3**: `build_packing_masks()` is passed to the model forward call in [enigma_engine/core/training.py](enigma_engine/core/training.py). Intra-doc masking is active.
- [RESOLVED] **CRIT-4**: Pre-train path in [enigma_engine/gui/gui_forge_new_modes.py](enigma_engine/gui/gui_forge_new_modes.py) auto-sets `warmup_steps = max(10, total_steps // 100)` (~1% of total). For 2.5M steps → ~25,000 warmup steps. Safe for N-6. Other training paths (SFT/DPO) still use default 100 — acceptable for short fine-tuning runs.
- [RESOLVED] **CRIT-5**: WSD is already the default scheduler (`schedule_type: str = "wsd"` in TrainingConfig). Not a build task.
- [BUILD] **CRIT-6**: ~~D-1/D-2 — DCLM + FineMath + The Stack v2. Must collect before N-6 restart to include in data mix. Running N-6 on current 87.6GB without code/math data misses Stage 2 benefits.~~ **DONE (Pass 136).** `fetch_dclm()`, `fetch_finemath()`, `fetch_the_stack()` added to [collect_pretraining_data.py](collect_pretraining_data.py). Run: `python collect_pretraining_data.py --dclm 15 --finemath 10 --code 10 --resume`

---

### Deferred (Post N-10, No Urgency Now)

- Context extension (D-13/D-17, YaRN full implementation)
- Flash Attention 4 stable Windows wheel (D-20)
- Video/3D generation mods (Video-1, 3D-1)
- Haptic prediction (Haptic-1)
- FSDP multi-GPU (single GPU until hardware changes)
- DDP training infrastructure (single GPU)

---

## Recommendation + Next Steps

**Decision:** Pursue **Approach 1 + Approach 3 in parallel**.

**Rationale:**
- Approach 1 is the long-term strategy (reasoning foundation + plugins)
- Approach 3 is a 3-week proof-of-concept (validate multi-model architecture)
- Together: low risk, high learning, clear path forward

**Immediate actions (this week):**

1. **Research (parallel, 2-3 hours each):**
   - C-1: QwQ paper (reasoning architecture)
   - C-7: Read enigma_engine/router.py (what's already there?)
   - A1-2: OpenThoughts3 training approach
   - A3-1: HuggingFace distillation guide

2. **Planning (1 week):**
   - List what FORGE Distillation mode needs (teacher loading, loss function, data)
   - Design router decision tree (heuristic rules for task classification)
   - Plan N-6 pre-training data mix (emphasize reasoning data, per D-4)

3. **Execution decision (week 2):**
   - Start N-6 pre-training with reasoning data emphasis (no code change)
   - In parallel: collect data for Approach 3 distillation (OpenThoughts3, VQA, Code)
   - Generate targets from Qwen3-30B (offline, can parallelize)
   - Train first variant (reasoning) via FORGE

**Critical path to decision point (month 2):**
- N-6 checkpoint (if loss stable, continue)
- Approach 3 variant 1 result (if reasoning variant > 50% on GSM8K, validates architecture)
- → Then decide: go full Approach 1 or commit to Approach 2?

---

## Roadmap

### Phase 2 — First Real Pre-Training (current) + Approach 1 Preparation

| # | Task | Status |
|---|------|--------|
| N-6 | **Pre-train with full dataset + reasoning data emphasis** | **Ready to resume.** Align with Approach 1: emphasize OpenThoughts3/FineMath in data mix. D-4 (reasoning mid-training) should be integrated. Was at Step 60/2,537,114 (~0.002%), 78 tok/s with batch=2. Target: 300-600 tok/s with auto-batch. |
| N-7 | **Benchmark after pre-training** | `python run.py --benchmark` (GSM8K, HellaSwag, MMLU benchmarks per D-10) |
| N-RES-1 | **Research critical items** | C-1/C-2/C-7 (QwQ architecture, router.py review) — 3-4 hours, complete by start of N-9 |

### Phase 3 — Fine-Tuning & Alignment (Reasoning-First Focus)

| # | Task | Details | Approach 1 Alignment |
|---|------|---------|----------------------|
| N-8 | ~~Collect fine-tuning data~~ | **Done.** collect_finetuning_data.py — OASST, Dolly, SlimOrca | Gather OpenThoughts3 + FineMath for N-9.5 (Reasoning Mid-training per D-4) |
| N-9 | Instruction fine-tune | FORGE → Basic mode on mixed SFT data | Use reasoning-heavy SFT data (SmolTalk2 with thinking traces) |
| N-10 | DPO + GRPO alignment | Curated preference pairs, GRPO for reasoning | **Focus GRPO on test-time compute**: reward model should value "thinks before answering" |
| N-11 | ~~Wire GRPO/ReMax/SimPO/ORPO to GUI~~ | **Done.** Radio cards, dispatcher | Keep all; GRPO is primary for Approach 1 |

### Phase 3.5 — Approach 3 Proof-of-Concept (Parallel with N-9/N-10)

**Goal:** Validate multi-model architecture by training 1-3 specialized variants via distillation from Qwen3-30B. **Quick experiment (3-4 weeks) to inform Approach 1 vs Approach 2 decision.**

| # | Task | Details | Success Criteria |
|---|------|---------|------------------|
| N-9.5a | **Collect Approach 3 data** | OpenThoughts3 (reasoning), LLaVA/VQA (vision), The Stack (code) | 10K examples per variant |
| N-9.5b | **Generate from Qwen3-30B** | Use qwen3-30b-a3b to create task-specific target outputs (offline) | Targets generated, stored in data/ |
| N-9.5c | **Train Reasoning variant** | FORGE Distillation: 742M learns from Qwen3-30B targets on OpenThoughts3 | Variant checkpoints saved |
| N-9.5d | **Benchmark Reasoning variant** | Test on GSM8K (math), AIME-style problems | > 50% accuracy = validates reasoning distillation |
| N-9.5e | **Train Vision variant** (if time) | FORGE Distillation on VQA data | > 60% accuracy on VQA test set |
| N-9.5f | **Decision point** | If variants work: continue Approach 1. If not: iterate on data/teacher. | Clear go/no-go signal by month 2 |

**VRAM allocation:**
- 12GB: 742M distillation training
- 2GB: Qwen3-30B teacher (separate, offline generation)
- 2GB: Overhead

**Timeline:** 3-4 weeks total (1-2 weeks per variant, can parallelize data collection)

**Integration with Phase 3:**
- Week 1-2: N-9.5a/b in parallel with N-9 (collect data)
- Week 3: N-9.5c/d (train reasoning variant while N-10 planning happens)
- Week 4: Decision + pivot

### Phase 4 — Evaluate & Improve (Approach 1 Foundation)

| # | Task | Details | Approach 1 Alignment |
|---|------|---------|----------------------|
| N-12 | **Build router + task classifier** | Design decision tree or heuristic router (C-8 research) | Classify: vision? code? reasoning? generation? → dispatch to specialist or 742M core |
| N-13 | **Evaluation benchmarks** | HellaSwag/MMLU/GSM8K for reasoning core | Test reasoning variant from N-9.5d in production-like scenario |
| N-14 | **Dense semantic memory** | Replace TF-IDF in [enigma_engine/core/rag.py](enigma_engine/core/rag.py) with FAISS/dense embeddings | Preparation for vision+retrieval later |

### Phase 5 — Advanced Features

| # | Task | Details |
|---|------|---------|
| N-15 | Constrained decoding | Grammar-constrained generation for JSON/tool calls |
| N-16 | Best-of-N sampling | Generate N, score with reward model, return best |
| N-17 | ~~Model merging~~ | **Done.** core/model_merging.py — SLERP, TIES, linear merge. GUI caller added in N-21 (Pass 126). |
| N-18 | ~~Continual learning~~ | **Done.** core/ewc.py — Fisher information + penalty |
| N-19 | Knowledge distillation | Logit-level distillation from teacher |
| N-20 | ~~Agentic tool loops~~ | **Done.** engine_generation.py — parse/execute/inject loop |
| N-21 | ~~Wire model merging to GUI~~ | **Done.** MODELS page inline merge row: two model dropdowns, SLERP/LINEAR/TIES method dropdown, t entry, density entry, output name, MERGE button. `_merge_models()` in [enigma_engine/gui/gui_forge_models.py](enigma_engine/gui/gui_forge_models.py) calls [enigma_engine/core/model_merging.py](enigma_engine/core/model_merging.py) in a background thread. |
| N-22 | ~~Emotional state → system prompt~~ | **Done.** `_build_gui_context()` injects internal emotional state tone cue. |
| N-23 | ~~Move RAG into _prepare_chat()~~ | **Done.** RAG retrieval moved into `core/engine_chat.py::_prepare_chat()`; GUI/API/background paths now share the same retrieval path when `engine._rag_index` is set. |
| N-24 | ~~Mod router frame framing~~ | **Done.** [enigma_engine/router.py](enigma_engine/router.py) already uses 4-byte big-endian message framing in both `_send_message()` and `_receive_message()`. |
| N-25 | ~~Background trainer GPU isolation~~ | **Done.** `BackgroundTrainer.run()` defers batches while inference is active via router-provided idle callback; GUI wires callback from `_is_generating`. |

### Phase 6 — Compiled Performance (Rust via PyO3)

Pure Python hot paths that would benefit from Rust rewrite. Priority by impact.
Speedup estimates derived from real benchmarks (HuggingFace tokenizers, OpenAI tiktoken, Google SentencePiece source code and published numbers).

**Evidence base:**
- HF tokenizers (Rust): "Less than 20 seconds to tokenize 1 GB on server CPU" (~50+ MB/s). Trains wikitext-103 (516 MB) in "a few seconds."
- tiktoken (Rust): "3-6x faster than comparable open source tokeniser." Source code says "Most of the time is spent in regex."
- SentencePiece (C++): ~50K sentences/sec, 6 MB memory.
- Our Python BPE: measured ~7 MB/s encode (4718 vocab, 0.45 MB test). BPE train: 20 min for 32K vocab on 2 GB (2.4M unique words).

| # | Component | File(s) | Speedup | Affects Inference | Notes |
|---|-----------|---------|---------|-------------------|-------|
| R-1 | ~~**Tokenizer encode/decode**~~ | [enigma_engine/core/bpe_tokenizer.py](enigma_engine/core/bpe_tokenizer.py) + [rust_extensions/](rust_extensions/) | ~~**7-20x**~~ **6x** | **Yes** | **Done.** Measured: 32-36 MB/s Rust (cached), ~8 MB/s (unique words) vs ~5-7 MB/s Python. Symbol interning + skip-array merge. PyO3 + maturin, GNU toolchain. Auto-fallback in BPETokenizer.encode()/decode(). |
| R-2 | ~~**BPE train()**~~ | [enigma_engine/core/bpe_tokenizer.py](enigma_engine/core/bpe_tokenizer.py) | ~~**20-60x**~~ | No | **Done.** Full Rust train via PyO3: pre-tokenize → word freq → pair freq/reverse index → max-heap with lazy deletion → incremental merge loop. Python auto-fallback in `_try_rust_train()`. 8 tests. Tie-breaking differs from Python (implementation-defined), final vocab within ±3 tokens. |
| R-3 | **MinHash dedup** | [enigma_engine/core/training.py](enigma_engine/core/training.py) `minhash_dedup()` | **Needs profiling** | No | O(n²) pairwise + SHA-256 per shingle. Likely benefits from Rust + SIMD, but no baseline measurement yet. |
| R-4 | **Data pipeline** | [collect_pretraining_data.py](collect_pretraining_data.py), [pretokenize_data.py](pretokenize_data.py) | **Needs profiling** | No | File I/O is OS-bound. Parallelism helps but Python `open()` is already a thin wrapper. Measure before committing. |
| R-5 | **Sequence packing** | [enigma_engine/core/training.py](enigma_engine/core/training.py) `pack_sequences_lazy()` | **Needs profiling** | No | Heap-based bin packing. Moderate gain expected — lowest priority. |

**Language choice: Rust + PyO3.** Both OpenAI (tiktoken) and HuggingFace (tokenizers) independently chose this stack. C++ (SentencePiece) works but has clunkier Python bindings (SWIG). Cython gives 2-5x for this workload — not worth it vs Rust's 7-60x. No production tokenizer uses Cython, Mojo, or Zig.

**Approach:** PyO3 + maturin. Builds a native `.pyd` (Windows) / `.so` (Linux) extension. Python wrapper matches existing `TokenizerProtocol` interface — no changes to callers. Vocab saved/loaded as same JSON format.

**R-1 and R-2 are one Rust crate** — a single BPE tokenizer that trains fast AND encodes fast. R-3 through R-5 should be profiled before committing to rewrite.

### Phase 7 — Long-Term Sustainability

| # | Task | Details |
|---|------|--------|
| S801 | ~~GGUF VRAM→context tiers~~ | **Done.** InferenceMemoryBudget.gguf_context_length / gguf_gpu_layers. 48 GB now gets 64K. |
| S802 | ~~Inference batch/seq tiers~~ | **Done.** InferenceMemoryBudget.inference_batch_size / inference_max_seq_len. 32 GB gets batch=8. |
| S803 | ~~Token count cache cap~~ | **Done.** Scaled to RAM via token_count_cache_cap. 8 GB → 4096, 64 GB → 32768. |
| S804 | ~~BPE cache cap~~ | **Done.** Scaled to RAM via bpe_cache_cap. 8 GB → 10000, 64 GB → 80000. |
| S805 | ~~Advanced tok cache cap~~ | **Done.** Scaled to RAM via advanced_tok_cache_cap. Same profile as S804. |
| S806 | ~~Dataset chunk size~~ | **Done.** dataset_chunk_chars / dataset_stream_threshold scale to RAM. Pi 5 gets 200M, 64 GB gets 500M. |
| S807 | ~~API max_tokens cap~~ | **Done.** api_max_tokens scales to VRAM. 32 GB → 16384, 8 GB → 4096. |

### Ideas

| Idea | When |
|------|------|
| ~~Training ETA in FORGE panel~~ | **Done.** Batch-level ETA in [enigma_engine/gui/gui_forge_training.py](enigma_engine/gui/gui_forge_training.py) |
| Checkpoint browser with perplexity comparison | After N-7 |
| ~~SimPO/ORPO GUI wiring~~ | **Done.** Included in N-11 |
| Pre-tokenized binary cache (`tokens.bin`) | Script written ([pretokenize_data.py](pretokenize_data.py)). Skips 20+ min data reload on future runs. Integration into [enigma_engine/core/training.py](enigma_engine/core/training.py) TBD after first run. |
| ~~Larger vocabulary (16K-32K trained BPE)~~ | **Done.** 32K BPE vocab trained on 2 GB sample (2.4M unique words). Active in current pre-training run. |

---

## External Research — April/May 2026

**Pass 1 (April):** SmolLM3, Gemma 3, DCLM, FineMath, OpenThoughts3, APO — D-1 through D-13. Single-source, directionally correct.
**Pass 2 (May):** Full paper reads — Qwen3 (arXiv:2505.09388v1, 742M-scale architecture details), DeepSeek-V3 (arXiv:2412.19437v2, MTP + MLA ablations), Flash Attention 4 releases (SM120/Blackwell confirmed). D-14 through D-20 added. **Correction:** Qwen3 does NOT use MTP (D-15 clarifies DeepSeek-V3 only). D-3's three-stage training now cross-validated by 3 sources.

All items below are research suggestions only. No code has been changed.

---

### D-1 — Add DCLM-Baseline and FineMath to pre-training data

**Why:** We have 88.8 GB of mostly web text (FineWeb-Edu 40 GB, C4 20 GB, Wiki 15 GB, OWT 10 GB). We have **zero dedicated math data** and **zero model-filtered web data**. This is the biggest gap.

- **DCLM-Baseline** (`mlfoundations/dclm-baseline-1.0`): 4T tokens from Common Crawl, model-filtered with a fastText classifier trained on OpenHermes 2.5 (instruction-tuned quality signal). Beats FineWeb-Edu at all compute levels in MMLU. CC-BY-4.0. Used by SmolLM3 Stage 1. The `collect_pretraining_data.py --fineweb 25` pattern already knows how to stream HF datasets — DCLM loads the same way. Collect 10–20 GB.

- **FineMath-4+** (`HuggingFaceTB/finemath`, `finemath-4plus` config): 9.6B tokens / 6.7M documents of high-quality step-by-step math from CommonCrawl. Scored by LLaMA-3.1-70B-Instruct on a 5-point scale, filtered to 4+. LaTeX and Markdown formatted. GSM8K/MATH benchmark gains are measurable from as little as a few billion tokens. ODC-By license. No math in current dataset means zero math reasoning at inference time. Collect 10–15 GB.

- **InfiMM-WebMath-3+** (`HuggingFaceTB/finemath`, `infiwebmath-3plus` config): 20.5B tokens, complementary to FineMath (different source URLs). SmolLM3 uses 50/50 blend of both. Combining gives ~50B tokens while matching FineMath quality. Collect 10 GB.

**Action:** Add DCLM and FineMath download to [collect_pretraining_data.py](collect_pretraining_data.py). Integrate into `combined.txt` with cap at 10–15 GB each. Rerun paragraph dedup after adding.

**Priority: HIGH** — no math data is the single biggest gap before instruction fine-tuning.

---

### D-2 — Add code data from The Stack v2

**Why:** SmolLM3 used 12–24% code data across its three training stages and found it substantially improved reasoning. We have zero code in pre-training. Code data teaches structured reasoning, pattern completion, and precise instruction following.

- **The Stack v2** (`bigcode/the-stack-v2`): Deduplicated source code across 600+ programming languages, 67B unique files. Use only the 16 languages SmolLM3 used: Python, JavaScript, TypeScript, Java, C, C++, Go, Rust, Ruby, PHP, Shell, SQL, HTML, CSS, Markdown, JSON. Filter to permissive licenses (MIT/Apache/BSD). Collect 5–10 GB from popular languages first.

- **StarCoder2 pull requests + GitHub issues**: Available as part of The Stack v2 ecosystem. More conversational/reasoning content than raw code. Collect 1–2 GB.

**Action:** Add The Stack v2 streaming download to [collect_pretraining_data.py](collect_pretraining_data.py). Use `--code 10` flag pattern, filter by language and license. Prioritize Python + JS + Rust.

**Priority: HIGH** — zero code = severely limited structured reasoning.

---

### D-3 — Three-stage pretraining data schedule

**Why:** SmolLM3 (11T tokens, 3B params), Qwen3, and Llama3 all use multi-stage training where data composition shifts across stages. Running a single fixed mixture for 2.5M steps is validated to be suboptimal — later stages should upsample math and code while the model is most capable of using them.

**Recommended schedule for Enigma (26B tokens → ~1-2 epochs):**
- Stage 1 (0→~20B tokens): 85% web (FineWeb-Edu + DCLM), 12% code, 3% math — general capability
- Stage 2 (20B→~24B tokens): 75% web, 15% code, 10% math — deeper math/code
- Stage 3 / Decay (24B→26B tokens): 63% web, 24% code, 13% math, inject OpenThoughts3 reasoning samples — LR linear decay to 0

**Implementation:** In [enigma_engine/core/training.py](enigma_engine/core/training.py), multi-stage scheduling can be done by maintaining separate dataset iterators and swapping them at step boundaries (not via GUI — wired in the Trainer). The WSD scheduler is more compatible with staged training than cosine — see D-5.

**Action:** Plan staged data mixing before starting next pre-training run. Can be done manually by re-weighting `combined.txt` sections or by having Trainer switch datasets at step N.

**Priority: MEDIUM** — depends on D-1/D-2 data being collected first.

---

### D-4 — Reasoning mid-training before SFT (chain-of-thought injection)

**Why:** SmolLM3 trained 35B tokens of reasoning data BETWEEN pre-training and SFT. This gave the base model the ability to reason before instruction fine-tuning shaped the behavior. Result: AIME 2025 improved from 9.3% → 36.7% with extended thinking. Without reasoning mid-training, SFT reasoning capability is severely limited.

- **OpenThoughts3-1.2M** (`open-thoughts/OpenThoughts3-1.2M`): 1.2M rows of math (850K), code (250K), and science (100K) with long reasoning traces. Annotated by QwQ-32B (16 answers per question). Apache-2.0. 28.2 GB. Enigma already has QwQ-like models in the `models/qwen3-30b-a3b/` directory — we could use our own Qwen3 30B to generate additional domain-specific reasoning traces.

- **NVIDIA Llama-Nemotron Post-Training v1.1** (`nvidia/Llama-Nemotron-Post-Training-Dataset`): 3.91M rows of reasoning traces from R1. Used by SmolLM3. Apache-2.0.

**Phase 3.5 (NEW):** Add a "Reasoning Mid-training" phase between N-9 (SFT) and N-10 (DPO):
- 1–3 epochs on OpenThoughts3-1.2M using ChatML format (no system prompt)
- Use FORGE → Basic mode, low LR (1e-5), no DPO
- This step costs ~10B tokens of GPU compute for 1 epoch

**Action:** Add N-9.5 Reasoning Mid-training to roadmap. Add OpenThoughts3 download to [collect_finetuning_data.py](collect_finetuning_data.py). Source data using Qwen3-30B from [models/qwen3-30b-a3b/](models/qwen3-30b-a3b/) for domain-specific math/science traces.

**Priority: HIGH** — this is the gap between a model that follows instructions and one that actually reasons.

---

### D-5 — WSD scheduler instead of cosine for long pre-training runs

**Why:** SmolLM3 and OLMo 2 both use WSD (Warmup-Stable-Decay): warm up for 2000 steps, hold stable LR for most of training, then linear decay to 0 in the final 10% of steps. The advantage over cosine:
1. You can stop training at any stable-phase checkpoint and resume without losing the decay budget
2. The stable phase gives more consistent gradient signal than cosine's oscillating LR
3. If a run is interrupted mid-decay, you can restart the decay from a stable-phase checkpoint

Current Enigma uses cosine LR (`CosineAnnealingLR` in training.py). For a 2.5M step run (~3 months), a mid-run interruption with cosine means the LR schedule is off when resumed. WSD means the stable-phase checkpoint is always valid.

**Implementation:** `torch.optim.lr_scheduler.LambdaLR` with three phases: linear warmup (0→1 over `warmup_steps`), constant 1.0 (until `decay_start_step`), linear decay (1.0→0 over final `decay_steps`). `decay_start = int(0.9 * total_steps)`.

**Action:** Add WSD scheduler as an option in `TrainingConfig` (keep cosine as default for safety). Wire to GUI as a scheduler dropdown option. Recommended for any run > 500K steps.

**Priority: MEDIUM** — the current cosine will work, but WSD is strictly better for long interrupted runs.

---

### D-6 — Expand tokenizer vocabulary to 64K-128K

**Why:** Current tokenizer is 32K BPE trained on 2 GB English web text. This means:
- All code is very poorly encoded (curly braces, keywords all split into tiny fragments)
- Math symbols (LaTeX, ≤, ∑, π) are typically unknown or single-byte
- Non-English text is fragmented (2-4x more tokens per character than English)

Real-world evidence: LLaMA3 uses 128K (100K base + 28K non-English), Gemma3 uses 262K SentencePiece. SmolLM3 reuses LLaMA3.2 tokenizer. Increasing from 32K → 64K on a corpus that includes code and math could reduce average tokens/document by 15-30% for those domains.

**What to collect for tokenizer training:**
- Python/JS/Rust source code (The Stack v2 — 100MB sample per language)
- FineMath documents (LaTeX-heavy math content)
- Multi-language web text (Spanish, French, German, Chinese Wikipedia dumps)
- Current 2 GB English sample (keep as base)

**Risk:** Vocab expansion requires re-initializing the embedding layer for new tokens, which means any pre-trained checkpoint will need fine-tuning on the new tokens before useful behavior resumes. **Do this BEFORE starting the next pre-training run, not after.** Once pre-training starts with 32K vocab, the cost of switching grows.

**Action:** Retrain tokenizer before N-6 with: 2 GB English web + 500 MB code + 200 MB math + 500 MB multilingual. Target 64K vocab. This would reduce total token count by ~10% for current dataset, reducing pre-training compute by ~10%.

**Priority: HIGH (time-sensitive)** — must happen before N-6 restarts, not after.

---

### D-7 — Intra-document masking during sequence packing

**Why:** When packing multiple documents into a single training sequence (which `pack_sequences_lazy()` does), the attention mask should prevent tokens from Document B attending to tokens from Document A. Without this mask, the model learns spurious cross-document dependencies. Llama3 uses this, SmolLM3 uses this. Found to improve long-context performance and training stability.

**Implementation check needed:** Verify whether `_create_packed_mask()` in [enigma_engine/core/training.py](enigma_engine/core/training.py) or [enigma_engine/core/model.py](enigma_engine/core/model.py) builds a proper block-diagonal mask per packed sequence. The 4D mask generated during sequence packing needs to be causal within each document and blocked across documents.

**This may already be implemented.** Check `pack_sequences_lazy()` and `_create_packed_mask()` return values. If the mask is just a standard causal mask (upper triangular), this feature is missing.

**Action:** Read [enigma_engine/core/training.py](enigma_engine/core/training.py)`::_create_packed_mask()` before the next pre-training run. If it returns a simple causal mask, add block-diagonal masking. Moderate implementation cost, high benefit.

**Priority: MEDIUM** — verify before N-6 restarts.

---

### D-8 — Remove weight decay from embedding layers

**Why:** OLMo 2 finding, adopted by SmolLM3: removing weight decay from embedding layers stabilizes training. Embedding norms naturally stabilize at healthier values without it. With weight decay on embeddings, the model may underfit on rare tokens because their embeddings get shrunk toward zero. SmolLM3 reported "more stable training dynamics" with this change.

**Implementation:** In [enigma_engine/core/training.py](enigma_engine/core/training.py), the `AdEMAMix` and `Muon` optimizer setups apply weight decay globally. Need to set `weight_decay=0` specifically for `model.embedding.weight` and `model.output.weight` (if not tied). If embeddings are tied (`model.embedding.weight is model.output.weight`), one parameter group suffices.

**Code pattern:**
```python
no_decay = ["embedding.weight", "output.weight"]
param_groups = [
    {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]
```

**Action:** Verify whether current optimizer setup applies weight decay to embeddings. If yes, add this split. Low-risk change, 3-5 line fix.

**Priority: LOW-MEDIUM** — implement before N-6 restart.

---

### D-9 — Anchored Preference Optimization (APO) as DPO alternative

**Why:** APO (D'Oosterlinck et al., Aug 2024) is a controllable variant of DPO. In DPO, the optimization objective can be underspecified — the model can satisfy the loss by degrading the rejected response rather than improving the chosen one. APO adds an explicit anchor term that controls how far the model can deviate from the reference model. Result: more stable training, better downstream scores. SmolLM3 chose APO over DPO and GRPO for their alignment phase. CLAIR data + APO improved Llama-3-8B by 7.65%, closing the gap with GPT4-turbo by 45%.

**Current state:** Enigma has DPO, SimPO, GRPO, ORPO, ReMax all wired. APO is not implemented.

**APO loss formula:** Standard DPO log-ratio term plus an anchor penalty that caps the KL divergence from the reference. Implemented as a modified DPO loss function with one additional hyperparameter `β_anchor` (recommended 0.1–0.5).

**Action:** Add APO to `alignment_training.py` as a new alignment mode. Add radio card in GUI alignment section. Relatively low implementation cost — it's a 15-line modification of the DPO loss.

**Priority: LOW-MEDIUM** — implement after N-10 (DPO) is validated.

---

### D-10 — Evaluation framework: wire standard benchmarks

**Why:** After pre-training (N-7) and fine-tuning (N-9/N-10), there's no automated way to measure if the model improved. `python run.py --benchmark` exists but only runs the coherence benchmark (internal perplexity). We need external benchmarks to compare against published models.

**Recommended benchmarks (all open, downloadable):**
- **GSM8K** (math word problems, 1.3K test examples, graded by exact match). Simple to evaluate — parse final number from output.
- **HellaSwag** (commonsense NLI, 10K examples, multiple choice). Already mentioned in roadmap N-12.
- **MMLU-CF** (controlled-format MMLU, removes format sensitivity). 57 subjects, 14K questions.
- **HumanEval+** (code generation, 164 problems, execute and test). Requires Python execution sandbox.

**Implementation approach:** Add `--eval-gsm8k`, `--eval-hellaswag`, `--eval-mmlu` flags to [run.py](run.py). Download benchmark data once, save to `data/eval/`. Run with the model in greedy decode mode (temperature=0). Output: accuracy percentage + 95% CI. No extra dependencies needed beyond what's in requirements.txt.

**Action:** Implement N-12 evaluation framework. Start with GSM8K (simplest to implement, unambiguous grading). HellaSwag second. MMLU third.

**Priority: MEDIUM** — needed to validate that training actually helped.

---

### D-11 — SmolTalk2 as SFT + preference data

**Why:** HuggingFace released the complete SmolLM3 post-training dataset as `SmolTalk2` (`HuggingFaceTB/smoltalk2`). It contains three subsets: `Mid` (reasoning mid-training, 140B tokens), `SFT` (1.8B tokens, 12 non-reasoning + 10 reasoning datasets), and `Preference` (APO preference pairs, non-reasoning + reasoning modes). This is fully open and purpose-built for instruction tuning a small model well.

**Current fine-tuning data:** OASST, Dolly 15k, SlimOrca, UltraChat. Good but these are older datasets without reasoning traces.

**What SmolTalk2 adds:**
- Reasoning traces in SFT format (thinking mode + non-thinking mode examples)
- Tool calling examples in XML and Python formats
- Multilingual SFT examples (French, Spanish, German, Italian, Portuguese)
- APO preference pairs rated chosen=Qwen3-32B, rejected=Qwen3-0.6B

**Action:** Add SmolTalk2 SFT subset download to [collect_finetuning_data.py](collect_finetuning_data.py). Use the SFT subset only (not Mid — that's 140B tokens we don't need to download). ~10-15 GB. Integrate into FORGE instruction fine-tuning (N-9) data mix.

**Priority: MEDIUM** — upgrade to current state-of-art SFT data when N-9 is executed.

---

### D-12 — NoPE hybrid attention for long-context improvement

**Why:** SmolLM3 implemented NoPE (Yang et al., Jan 2025, arXiv 2501.18795) and found it improves long-context performance without hurting short-context. The approach: remove RoPE from every 4th transformer layer. NoPE layers have no positional bias, making them sensitive to content rather than position — they act as long-range context aggregators. RoPE layers provide local positional structure. The hybrid outperforms pure RoPE at extended context lengths.

**Current state:** Enigma model has RoPE at every layer ([enigma_engine/core/engine_generation.py](enigma_engine/core/engine_generation.py), [enigma_engine/core/model_components.py](enigma_engine/core/model_components.py)).

**Implementation cost:** Low — the change is `if layer_idx % 4 != 0: apply_rope(...)`. In [enigma_engine/core/model.py](enigma_engine/core/model.py) or [enigma_engine/core/model_components.py](enigma_engine/core/model_components.py), the attention forward pass already accepts positional arguments. Add a `use_rope` flag per layer determined by `layer_idx % 4 == 0`.

**Note:** This changes the model architecture. Any pre-trained checkpoint becomes incompatible unless the change is made before N-6. If made after N-6, requires fine-tuning to recover.

**Action:** Evaluate timing — implement before N-6 restart if possible. If N-6 has already started, defer to after pre-training when a long-context fine-tuning phase is added.

**Priority: LOW (time-sensitive)** — worth implementing before N-6 if not already started.

---

### D-13 — Context length extension via RoPE theta scaling (not training from scratch)

**Why:** All production models (Gemma3, SmolLM3, LLaMA3) extend context length at the END of pre-training, not from the beginning. Gemma3 trained at 32K sequences, then extended to 128K by increasing RoPE theta from 10K to 1M (×100). SmolLM3 extended 4K→32K→64K in two sequential 50B-token stages after the main 11T token run. Starting at 4K is correct — it's faster and cheaper.

**Current state:** Enigma `max_seq_len` is configurable but the tokenizer and training setup use 4096 by default. This is correct.

**Recommended extension plan (after N-9/N-10):**
1. After fine-tuning is stable, run a 5B-token extension stage at 16K context: increase RoPE theta to 500K
2. Run a 2B-token extension stage at 32K context: increase RoPE theta to 2M
3. Use YaRN at inference for 2× extrapolation beyond training context

**YaRN** (`rope_scaling_type="yarn"` in config): Scales RoPE frequencies non-uniformly, allowing inference at 2× the training context with minimal degradation. Drop-in addition to the inference path. No training needed.

**Action:** Add RoPE theta as a configurable `TrainingConfig` parameter (currently hardcoded?). Add YaRN scaling to inference path as an optional flag. Plan context extension as N-26 after alignment is done.

**Priority: LOW** — comes after the main training pipeline is proven.

---

### D-14 — QK-Norm for Training Stability (verified from Qwen3 + ViT-22B full papers)

**Why:** Qwen3 (arXiv:2505.09388, Section 2) adds QK-Norm to all attention layers specifically for training stability. ViT-22B (Dehghani et al., arXiv:2302.05442) independently applied the same technique at 22B scale. Two independent large-scale adoption points, different modalities, same conclusion.

**The problem it solves:** Without QK-Norm, Q and K vectors can grow in magnitude over training. Attention scores Q@K.T/sqrt(d) spike → softmax collapses to argmax (one-hot attention) → zero gradients for all non-winning tokens → gradient starvation. Qwen3 treats this as important enough to mention it in their architecture section as a named change from Qwen2.

**Implementation:** Add `self.q_norm = nn.RMSNorm(head_dim)` and `self.k_norm = nn.RMSNorm(head_dim)` to the attention module. Apply after projecting and splitting Q/K into heads, before RoPE. Two extra RMSNorm ops per attention layer — negligible compute (~0.1% overhead), ~0.1% extra parameters.

**Timing concern:** Architecture change. RMSNorm weights initialize to 1 (identity), so adding this to a partial checkpoint allows recovery in continued training, but adding before N-6 is cleaner. Does not change model output at step 0 (since norm of a vector by its norm = same direction).

**Action:** Read [enigma_engine/core/model_components.py](enigma_engine/core/model_components.py) attention forward pass. Add Q/K normalization per head before RoPE application. Verify existing attention tests still pass (shape + determinism).

**Priority: HIGH** — low-cost architecture improvement validated by 2 independent production models at scale. Recommend implementing before N-6 restart.

---

### D-15 — Multi-Token Prediction (MTP) Training Objective (DeepSeek-V3 only, not Qwen3)

**IMPORTANT NOTE:** First-pass research suggested both Qwen3 and DeepSeek-V3 use MTP. This is wrong. **Qwen3 does NOT use MTP.** Qwen3 uses a 4-stage post-training pipeline (CoT cold start → GRPO → mode fusion → General RL). Only DeepSeek-V3 uses MTP in pre-training. Source verified: Qwen3 full paper arXiv:2505.09388 has no MTP section. DeepSeek-V3 full paper arXiv:2412.19437 Section 2.2 fully describes MTP.

**Why it matters (DeepSeek-V3 ablation, 2.4B activated / 15.7B total params, 1.33T tokens):**

| Benchmark | Without MTP | With MTP | Gain |
|-----------|-------------|----------|------|
| BBH | 39.0 | 41.4 | +2.4pp |
| MMLU | 50.0 | 53.3 | +3.3pp |
| GSM8K | 25.4 | 31.4 | +6.0pp |
| HumanEval | 20.7 | 26.8 | +6.1pp |

Consistent gains across all 4 benchmarks at sub-2B activated parameter scale. This is the closest available ablation to Enigma's 742M scale.

**Architecture:** D=1 depth (predicts one extra token beyond next-token). Each MTP module contains one Transformer block + projection M_k ∈ R^(d×2d) + shared embedding + shared output head. Combines prior-depth repr h^(k-1)_i with Emb(t_{i+k}) via projection. Loss weight λ=0.3 for first 10T tokens, 0.1 for remaining 4.8T. At inference: discard head (zero compute cost) OR keep for speculative decoding (85-90% 2nd-token acceptance → 1.8× TPS).

**Current Enigma state:** [enigma_engine/core/model.py](enigma_engine/core/model.py) has MTP head scaffolding. Verify: (1) Is MTP loss actually applied in [enigma_engine/core/training.py](enigma_engine/core/training.py) training loop? (2) Is the λ schedule implemented (0.3 early → 0.1 late)? (3) Is the MTP block architecture correct (sequential, not parallel)?

**Action:** Read [enigma_engine/core/model.py](enigma_engine/core/model.py) MTP section and [enigma_engine/core/training.py](enigma_engine/core/training.py) loss calculation. If scaffold exists but loss isn't wired, wire it. If wired but λ is constant, add the step-based schedule.

**Priority: MEDIUM** — validated improvement with small-scale ablation evidence. Architecture change — do before N-6 restart.

---

### D-16 — Three-Stage Pre-Training Cross-Validated (D-3 now confirmed by 3 sources)

**Note:** D-3 was originally sourced from SmolLM3 only. Now verified from full paper reads of Qwen3 (arXiv:2505.09388, Section 3.2) and DeepSeek-V3 (arXiv:2412.19437, Section 4.3). All three systems independently use staged training with shifting data composition.

**Qwen3 exact stages (from paper Table and Section 3.2):**
- S1: 30T tokens at 4K seqlen — general knowledge (web, books, code, multilingual)
- S2: 5T tokens at 4K seqlen — knowledge-intensive: higher proportion of STEM, code, math, reasoning data
- S3: Hundreds of billions of tokens at 32K seqlen — long-context extension using YARN + Dual Chunk Attention (DCA)

**For Enigma's ~26B tokens:** Our entire pre-training run is roughly S1 equivalent. S2 requires FineMath (D-1) — plan a continued training phase on math-heavy data after first 20B tokens. S3 is D-13/D-17 (context extension). Three independent data points now confirm this is the right architecture for staged training. D-3's recommendations are validated.

**Priority: See D-3** — no separate action needed. This entry upgrades D-3's evidence from single-source to 3-way cross-reference.

---

### D-17 — RoPE ABF Exact Parameters (cross-validates and updates D-13)

**Note:** D-13 proposed context extension but lacked specific parameter values. Now verified from two full paper reads.

**Qwen3 (arXiv:2505.09388, Section 3.2):** Raises RoPE base frequency from 10,000 to **1,000,000** (100× increase) using ABF (Adjusted Base Frequency) technique during S3. This is a training-time change — not an inference extrapolation hack. The model is trained on 32K sequences with the 1M base, allowing genuine long-context attention patterns to form.

**DeepSeek-V3 (arXiv:2412.19437):** Uses YARN with scale s=40, α=1, β=32, t=0.1×ln(s)+1. Applied only to the decoupled RoPE key (their MLA architecture). Context extended 4K→32K then 32K→128K in two phases.

**Recommended parameters for Enigma extension phases:**
- Before context extension: raise rope_base 10K → 500K
- 16K training phase: rope_base = 500K
- 32K training phase: rope_base = 1,000,000
- At inference: YARN scale factor (s=8 for 2× extrapolation) as optional config flag

**Action:** Verify [enigma_engine/core/model.py](enigma_engine/core/model.py) reads `rope_base` from config. If hardcoded as 10000, make it a `TrainingConfig` field. See D-13 for full context extension plan.

**Priority: LOW** — implement alongside D-13 (context extension phase, after N-10 alignment).

---

### D-18 — Aux-Loss-Free MoE Load Balancing (DeepSeek-V3, only if MoE enabled)

**Why:** Standard MoE training adds an auxiliary loss term to prevent expert collapse. DeepSeek-V3 (arXiv:2412.19437, Section 2.1.2 + ablation Table 5) replaces this with a bias-based mechanism that achieves better balance without polluting the main training signal.

**Mechanism:** Each expert gets a bias b_i (initialized 0). Routing decision uses s_{i,t} + b_i (adds bias). Actual gating weight uses s_{i,t} only (bias not applied). After each step: b_i += γ if expert underloaded, b_i -= γ if overloaded. γ=0.001. Bias frozen for the last 500B tokens to prevent late-training oscillation.

**Verified ablation (small scale, 1.33T tokens):**
- GSM8K: aux-loss 27.1 → aux-loss-free 29.6 (+3.8pp)
- HumanEval: aux-loss 22.0 → aux-loss-free 22.6 (+2.6pp)

**Action:** If MoE is enabled in Enigma (currently optional), replace the existing MoE aux loss with the bias mechanism. Minimal implementation: per-expert scalar bias, ±γ update logic after each batch, freeze flag at step threshold.

**Priority: LOW** — only relevant when MoE is turned on. No action until MoE training begins.

---

### D-19 — Strong-to-Weak Distillation for Post-Training Small Models (Qwen3)

**Why:** Qwen3 (arXiv:2505.09388, Section 4.4) reports that running the full 4-stage post-training pipeline on sub-2B models is 10× more expensive than distillation, yet distillation achieves competitive quality. The approach: take the large model (32B) after full 4-stage post-training, generate responses on instruction data, use those responses as SFT targets for the small model.

**Why this matters for Enigma:** At 742M params, running GRPO RL (our N-11 pipeline) is expensive and the reward signal is noisy at small scale. A distillation-first approach may be more efficient:
1. Use Qwen3-30B (already in `models/qwen3-30b-a3b/`) to generate high-quality responses on our SFT data
2. Fine-tune Enigma 742M on those generated responses via FORGE Distillation mode
3. Use GRPO only if distillation plateau is reached (not as the primary alignment path)

**Current state:** FORGE has a Distillation training mode. It can load a teacher model and train on teacher outputs. Verify whether the Distillation mode in [enigma_engine/gui/gui_forge_new_modes.py](enigma_engine/gui/gui_forge_new_modes.py) supports loading a local GGUF/pth teacher (like Qwen3-30B).

**Action:** Review Distillation mode for compatibility with local Qwen3-30B teacher. Plan to use it as the primary post-training path before attempting GRPO RL. This is more efficient given we have a capable teacher already installed.

**Priority: MEDIUM** — practical shortcut. Revisit at N-9 (SFT) when the post-training plan is finalized.

---

### D-20 — Flash Attention 4 (FA4): SM120 / RTX 5090 Support Active in Beta

**Why:** Enigma currently uses standard PyTorch `scaled_dot_product_attention`. Flash Attention 3 (for H100 Hopper/SM90) delivered 1.5-2× training speedup and 75% theoretical FLOP utilization. FA4 extends this to Blackwell — specifically SM120 (RTX 5090 class, Blackwell GeForce).

**Verified status (github.com/Dao-AILab/flash-attention/releases, verified Pass 148):**
- FA4 latest tag is **v4.0.0.beta10** (releases cadence ~weekly, active development).
- **SM120 (Blackwell GeForce = RTX 5090 / 5080):** **fwd + bwd + varlen all merged** via PRs #2329, #2330, #2333 (author blake-snc). Confirmed in source on `main`.
- **Install:** `pip install flash-attn-4` works **on Linux** (FA4 ships as a separate package alongside FA2/FA3, distinct from `flash-attn`). FA4 is built on **CuTeDSL** rather than the older CUTLASS C++ template path.
- **Windows:** no official FA4 Windows wheel in release assets. The README still notes Windows compilation "requires more testing" even for FA2 (current PyPI flash-attn 2.8.3, Aug 15 2025). Until upstream ships Windows wheels, our 5090 + Windows 11 target box can only consume FA4 via WSL2 or by self-compiling, both of which are out-of-scope for a multi-month training run.
- SM100 (B200/H200 data center): GQA, FP8 E4M3/E5M2, paged attention for MLA, block-sparse — present (not relevant to us).

**Current situation:** FA4 kernels for our exact GPU (SM120) are merged and installable on Linux. The blocker for Enigma is purely the **Windows wheel + multi-month training stability**, not the kernel itself. Worth wiring behind a config flag now and exercising it in WSL benchmarks before committing it to a long pre-training run.

**When ready to integrate:**
1. `pip install flash-attn-4` (Linux/WSL) or compile from source for Windows.
2. Add `use_flash_attn: bool = False` to `TrainingConfig`.
3. In attention forward: gate on `torch.cuda.get_device_capability() >= (12, 0)` for SM120 path, fall back to existing SDPA otherwise.
4. Verify training loss matches non-flash baseline on small data before any full run.

**Action:** Bench FA4 vs SDPA on RTX 5090 under WSL2 in Pass 149+. Hold off on production Windows-native FA4 until upstream ships an official Windows wheel.

**Priority: LOW (timing)** — high value when stable. Don't add a pre-release dependency to a multi-month training run.

---

### Summary Table

| ID | Area | Impact | Cost | Timing | Sources |
|----|------|--------|------|--------|---------|
| D-1 | Add DCLM + FineMath data | High | Low (scripting) | Before N-6 restart | SmolLM3, FineMath paper |
| D-2 | Add The Stack v2 code data | High | Low (scripting) | Before N-6 restart | SmolLM3 |
| D-3 | Three-stage training schedule | Medium | Medium | Before N-6 restart | SmolLM3, **Qwen3**, **DeepSeek-V3** |
| D-4 | Reasoning mid-training phase | High | Medium (new phase) | After N-9 SFT | SmolLM3, OpenThoughts3 |
| D-5 | WSD scheduler option | Medium | Low (15 lines) | Before N-6 restart | SmolLM3, OLMo 2 |
| D-6 | Expand tokenizer to 64K | High | Medium (retrain BPE) | **Before N-6 restart** | LLaMA3, Gemma3, Qwen3 (151K) |
| D-7 | Intra-document masking | Medium | Low (verify first) | Before N-6 restart | LLaMA3, SmolLM3 |
| D-8 | No weight decay on embeddings | Low-Med | Low (5 lines) | Before N-6 restart | OLMo 2, SmolLM3 |
| D-9 | APO alignment mode | Low-Med | Low (15 lines) | After N-10 | APO paper (D'Oosterlinck 2024) |
| D-10 | Standard eval benchmarks | Medium | Medium | After N-7 | Universal practice |
| D-11 | SmolTalk2 SFT data | Medium | Low (scripting) | Before N-9 | SmolLM3 |
| D-12 | NoPE hybrid attention | Low | Low (architecture) | **Before N-6 if possible** | NoPE paper (Yang 2025) |
| D-13 | Context length extension | Low | Low (config + YaRN) | After N-10 | Gemma3, SmolLM3, **Qwen3**, **DeepSeek-V3** |
| D-14 | QK-Norm for training stability | **High** | Low (3-5 lines) | **Before N-6 restart** | **Qwen3** (arXiv:2505.09388), ViT-22B (arXiv:2302.05442) |
| D-15 | MTP training objective | Medium | Medium (verify scaffold) | Before N-6 restart | **DeepSeek-V3** only (arXiv:2412.19437) |
| D-16 | Three-stage training confirmed | — | — | — | Cross-ref for D-3: SmolLM3 + **Qwen3** + **DeepSeek-V3** |
| D-17 | RoPE ABF exact parameters | Low | Low (config field) | With D-13 | **Qwen3** + **DeepSeek-V3** full papers |
| D-18 | Aux-loss-free MoE balancing | Low (if MoE) | Low (when needed) | When MoE enabled | **DeepSeek-V3** (arXiv:2412.19437) |
| D-19 | Strong-to-weak distillation | Medium | Low (verify FORGE) | At N-9 SFT | **Qwen3** (Section 4.4) |
| D-20 | Flash Attention 4 / SM120 | Medium | Low (when stable) | Q3-Q4 2026 | FA4 releases (beta10 confirmed SM120) |

**Critical path before resuming N-6:** D-14 (QK-Norm, architecture) → D-1 → D-2 → D-6 (tokenizer) → D-7/D-8 → restart. D-15 MTP worth verifying before N-6 if scaffold is already present.

**Research confidence:** Items cross-validated from full paper reads (marked **bold**) in Sources column. D-1/D-2/D-4/D-5/D-8/D-11 from first pass only — still single-sourced, still directionally correct but less precisely specified.

---

### Open Audit Findings (Pass 110)

| # | Severity | File | Issue |
|---|----------|------|-------|
| ~~S812~~ | ~~Correctness~~ | ~~[enigma_engine/core/model_components.py](enigma_engine/core/model_components.py)~~ | **Fixed (Pass 115).** Causal mask rebuilt with torch.triu for merged sequence length. |
| ~~S813~~ | ~~Correctness~~ | ~~[enigma_engine/core/rl_training.py](enigma_engine/core/rl_training.py)~~ | **Fixed (Pass 115).** ReplayBuffer now stores full_ids/prompt_len/ref_logps so PPO can recompute log-probs for replay items. |
| ~~S815~~ | ~~Security~~ | ~~[enigma_engine/api/server.py](enigma_engine/api/server.py)~~ | **Fixed (Pass 115).** run.py reads CONFIG["enigma_api_key"] as fallback when --api-key not on CLI. |
| ~~S816~~ | ~~Security~~ | ~~[enigma_engine/api/server.py](enigma_engine/api/server.py)~~ | **Fixed (Pass 111).** Returns paths relative to MODELS_DIR. load_model resolves both relative and absolute. |
| S817 | ~~Security~~ | [enigma_engine/api/server.py](enigma_engine/api/server.py) | **Accepted risk (Pass 115).** Lock prevents GPU corruption during concurrent inference. Single-model engine is inherently single-threaded. All 3 endpoints use same non-blocking acquire with 429 response. |
| ~~S819~~ | ~~Dead UX~~ | ~~[enigma_engine/gui/gui_forge_new_modes.py](enigma_engine/gui/gui_forge_new_modes.py)~~ | **Fixed (Pass 115).** Alignment modes (GRPO/ReMax/SimPO/ORPO) now show only basic section. |
| ~~S820~~ | ~~Config~~ | ~~[enigma_engine/api/server.py](enigma_engine/api/server.py)~~ | **False positive (Pass 111).** Port 8080 is consistent in run_server(), run_serve(), and docstring. No 8000 in code. |
| ~~S821~~ | ~~Comment~~ | ~~[enigma_engine/core/model_components.py](enigma_engine/core/model_components.py)~~ | **False positive (Pass 111).** Inline comment correctly says "dot product of L2-normalized features" — which IS cosine similarity. Docstring says "most similar pairs", not "cosine". |
| ~~S822~~ | ~~Missing~~ | ~~[enigma_engine/core/model_components.py](enigma_engine/core/model_components.py)~~ | **Fixed (Pass 115).** Skip ToMe when T > 4096 (O(T²) similarity matrix would OOM). |
| ~~S823~~ | ~~Perf~~ | ~~[enigma_engine/core/rl_training.py](enigma_engine/core/rl_training.py)~~ | **Fixed (Pass 117).** Added `_get_logps_hidden_entropy()` — single model pass returning logps + hidden states + entropy. Replaced triple policy passes in RLHF rollout (2→1), RLHF minibatch (3→1), SelfPlay rollout (2→1), SelfPlay minibatch (3→1), ReMax update (2→1). Ref-model calls unchanged. 4 tests verify logps/hidden/entropy all match separate calls. 2262 tests pass. |
| ~~S824~~ | ~~Precision~~ | ~~[enigma_engine/core/model_components.py](enigma_engine/core/model_components.py)~~ | **Fixed (Pass 111).** Accumulator upcast to fp32, weighted_output.float() for index_add_, cast back to input dtype before return. |
| ~~S825~~ | ~~Dropped~~ | ~~[enigma_engine/core/streaming.py](enigma_engine/core/streaming.py)~~ | **Fixed (Pass 115).** Changed to unbounded queue (maxsize=0). Stream is finite, tokens are tiny strings. |
| ~~S828~~ | ~~Correctness~~ | ~~[enigma_engine/gui/gui_forge_tools.py](enigma_engine/gui/gui_forge_tools.py)~~ | **Fixed (Pass 112).** _save_forge_checkpoint() now checks _active_trainer; during active training calls trainer._save_checkpoint(dest) for live weights. When idle, falls back to shutil.copy2 (file is up-to-date). _active_trainer stored/cleared across all 13 training paths (7 missing assignments added in audit). |
| ~~S829~~ | ~~Perf~~ | ~~[enigma_engine/gui/gui_forge_tools.py](enigma_engine/gui/gui_forge_tools.py)~~ | **Fixed (Pass 112).** All 3 presets (Quick/Balanced/Thorough) changed from hardcoded batch=2/4 to "auto". Widget default was already "auto". Auto-batch selects optimal size for available VRAM. |
| ~~S830~~ | ~~Correctness~~ | ~~[enigma_engine/core/training.py](enigma_engine/core/training.py)~~ | **Fixed (Pass 112).** Added save_every_steps field to TrainingConfig (default 0=disabled). Training loop saves checkpoint every N steps with keep=3 cleanup. Pre-train auto-sets to max(500, total_steps // 20). |
| ~~S840~~ | ~~Hygiene~~ | ~~[enigma_engine/gui/gui_forge.py](enigma_engine/gui/gui_forge.py)~~ | **Fixed.** Removed unused local `import json` in `_save_training_brief()` (ruff F401). |
| ~~S841~~ | ~~Hygiene~~ | ~~[tests/test_new_features.py](tests/test_new_features.py)~~ | **Fixed.** Removed unused top-level `import json` (ruff F401). |
| ~~S842~~ | ~~Hygiene~~ | ~~[tests/test_new_features.py](tests/test_new_features.py)~~ | **Fixed.** Removed unused local `import os` in `test_cleanup_removes_env_vars()` (ruff F401). |
| ~~S843~~ | ~~Hygiene~~ | ~~[tests/test_new_features.py](tests/test_new_features.py)~~ | **Fixed.** Removed unused local `import math` in `test_moving_average_nan_does_not_leak_stale_values()` (ruff F401). |
| ~~S844~~ | ~~Hygiene~~ | ~~[tests/test_new_features.py](tests/test_new_features.py)~~ | **Fixed.** Removed redundant local `import json` redefinitions (ruff F811). |

---

## Accepted Risk (no action needed)

| # | File | Why accepted |
|---|------|-------------|
| S725 | [collect_pretraining_data.py](collect_pretraining_data.py) | SHA-256 truncated to 64 bits. Collision at ~4B paragraphs — we have ~50M. |
| S769 | [enigma_engine/router.py](enigma_engine/router.py) | `mod.last_seen` scalar write. CPython GIL atomic. |
| S791 | [collect_pretraining_data.py](collect_pretraining_data.py) | Filename collision on 80-char truncation. Never observed. |
| S792 | [collect_pretraining_data.py](collect_pretraining_data.py) | XML bomb risk. Trusted sources only. |
| S793 | [collect_pretraining_data.py](collect_pretraining_data.py) | FineWeb resume O(n) skip. HF datasets limitation. Data done. |
| S817 | [enigma_engine/api/server.py](enigma_engine/api/server.py) | SSE lock held for entire stream. Single-model GPU engine is inherently single-threaded. Non-blocking acquire + 429 response. |
| S796 | [enigma_engine/core/huggingface_loader.py](enigma_engine/core/huggingface_loader.py) | Stream thread exceptions silently lost. Display-only. |
| S797 | [enigma_engine/core/huggingface_loader.py](enigma_engine/core/huggingface_loader.py) | Param estimation inaccurate for GQA/MoE. Display-only. |

---

## Reference

**Data sources (all done):** FineWeb-Edu 40.2 GB, C4 20.1 GB, Wiki dump 15.1 GB, OWT 10.1 GB, Gutenberg 1.5 GB, SE 1.1 GB, Wayback 64 MB, Fandom 46 MB, Wikipedia/Simple 22 MB.
**Future sources:** ArXiv, The Stack (code), OpenWebMath, PubMed.
**Fine-tuning data:** OASST, Dolly 15k, SlimOrca, UltraChat.
**Parked tech:** nGPT, DoReMi, PagedAttention, Mamba/SSM, Neuro-Symbolic, Vision datasets, MoE (all too invasive or wrong scale for 742M).
**Rejected sources:** Common Crawl, The Pile, Wiktionary, Wikiquote.
