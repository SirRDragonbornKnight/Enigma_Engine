# Pass History Archive

Chronological archive of per-pass notes moved out of the active tracking docs
(`SUGGESTIONS.md`, `CODE_REVIEW.md`, `GUI_REFERENCE.md`) during the doc-hygiene
pass of April 22, 2026. The active docs retain only the current state; anything
older lives here. Per-file bug-fix history is still tracked inline in the
`CODE_REVIEW.md` file tables — this archive only holds the top-of-file pass
prose.

**Convention:** Most recent at top. Each entry is the original prose, verbatim.
Do not delete entries — they document why decisions were made.

**Note on pass numbering:** The main audit counter runs Pass 1..135 and is used
throughout `CODE_REVIEW.md` and most `GUI_REFERENCE.md` notes. Separately, a few
SUGGESTIONS.md entries below use small numbers like "Pass 9" / "Pass 10" —
these are sub-session tags within the Pass 135 work day and do NOT belong to
the main counter.

---

## SUGGESTIONS.md — Top-of-file pass prose (archived Pass 135)

**Updated:** April 24, 2026 (Pass 148 — Currency-check addendum, two phases, doc-only.
**Phase 1** verified primary sources for 6 high-turnover domains (speculative
decoding, vision encoders, image gen, video gen, 3D gen, reasoning teachers).
Outcome: 3 [SUPERSEDED] swaps with successor blocks inline — Video-1 (AnimateDiff
→ Wan 2.1 T2V-1.3B + LTX-Video 0.9.8 alt), 3D-1 (Shap-E → Hunyuan3D-2 + TRELLIS
alt), Medusa training recipe stays valid but EAGLE-2 (arxiv:2406.16858) supersedes
it as the inference-time speculative method. 2 upgrade notes — Vision-1b adds
SigLIP 2 (arxiv:2502.14786) as future encoder upgrade; Approach 3 teacher list
adds Qwen3-32B and DeepSeek-R1-0528-Qwen3-8B alongside Qwen3-30B-A3B (QwQ-32B
deprecated). **Phase 2** verified 4 more domains: added Vision-2b (LLaVA-NeXT →
LLaVA-OneVision arxiv:2408.03326, 0.5B variant fits 16 GB), RAG-2b (bge-small
→ Qwen3-Embedding-0.6B arxiv:2506.05176, MTEB multilingual 64.33, Apache-2.0),
Audio-5b (MusicGen CC-BY-NC noted; Stable Audio Open Small arxiv:2505.08175
preferred for general SFX), and new Audio-ASR-1 (Whisper → NVIDIA Parakeet-TDT-
0.6B-v2, 6.05 WER / RTFx 3386, ~16× faster). D-20 FA4 updated to verified
v4.0.0.beta10 status: SM120 fwd+bwd+varlen merged via PRs #2329/#2330/#2333,
`pip install flash-attn-4` works on Linux/WSL, Windows wheels still TBD. Also
fixed stale Priority Index P3 numbering (duplicate #28 → renumbered 37-45) and
stale audit counter (135 → 148 passes). No code, no tests run — doc-only.)

**Updated:** April 22, 2026 (Pass 135 — Doc hygiene only. Converted all inline
backticked file-path references in this document to proper markdown hyperlinks
(138 links, all verified to point to existing files on disk). Fixed one escaped
`\n` artifact in the Accepted Risk table (S793/S817 row). Remaining 9 backticked
refs are intentional non-links: non-existent files (`reward_functions.py`,
`alignment_training.py`), generated artifacts (`combined.txt`,
`logs/training_heartbeat.json`), shell-command invocations, and method names.
No code, GUI, or behavior changes. No tests run — no code changed.)

**Updated:** April 22, 2026 (Pass 10 — Added R-UNPREDICT-1 research/design item
on controlled unpredictability. Behavior unpredictable (sampling, self-initiated
research, mood-weighted replay, emergent personality drift), infrastructure
deterministic (seeded training, seeded tests, atomic saves). Links to existing
AutoResearch-2, GRPO-4, Personality-5, N-25. Pass 9 personality findings stand
unchanged.)

**Pass 9:** Corrected Pass 8 over-retraction on emotional_state.
`emotional_state` is AI-computed from sentiment analysis (not user-set), and
N-22 (Pass 125) already injects it as "Internal State: … color your tone
naturally" — AI self-awareness of its own state, NOT user-configurable
personality. Personality-1 retraction REVERSED. Personality-2 retraction
STANDS: `ai_profile.personality` dict IS user-set (tone/verbosity/formality/
humor), injecting would mean user configures AI character. Personality-3/4/5
and R-PERSONALITY-1 stand. GRPO-4, AutoResearch-2, Test-1 retained.

**Pass 128 (research-only):** No code changes. Full codebase scan pass 4: read
router.py (BackgroundTrainer confirmed, ModRouter = mod IPC not model dispatch),
rag.py (BM25 RAG confirmed), kv_cache.py (pre-allocated + H2O/INT4/StreamingLLM
confirmed), mods/vision (screen OCR, NOT LLM vision), mods/codegen (uses base
742M), mods/avatar (full bone rig, GLB/GLTF/OBJ loader), monologue.py (Phase 5,
journal + coherence gate), adaptive_trainer.py (curriculum pipeline),
advanced_tokenizer.py (Tok-1 vocab expansion gap confirmed),
training_evaluation.py (perplexity only — Eval-1 gap confirmed).
SUGGESTIONS.md updated: 15 items resolved, 18 new research items added
(RAG-2/3, Memory-2, Merge-2, Mono-2, Adaptive-2, Avatar-3/4, Vision-6,
Router-2/3, RAG-2/3, etc.).

**Pass 127:** Lint cleanup — 0 issues in tests/enigma_engine.

### Bug-fix batches (SUGGESTIONS.md "Status" section)

**Bug fixes (prior):** S798 pack_sequences RAM OOM, S799 _pack_sequences CUDA
OOM, S800 TrainingMemoryBudget, S801-S807 InferenceMemoryBudget, S808-S810
BPE/GIL/phases.

**Bug fixes (Pass 110):** S811 SimPO/ORPO KeyError (4 missing forge_params
keys → inline defaults). S814 _ensure_initialized race (work ran outside lock).
S818 Self-Play missing except KeyboardInterrupt. S826 _pack_sequences OOM (list
accumulated ~3 TB of 4D masks — converted to generator). S827
_ensure_initialized deadlock (_LazyConfig proxy re-entered non-reentrant
_init_lock — bypassed with dict.update/dict.__setitem__/dict.items).

**Bug fixes (Pass 111):** S816 /api/models leaked absolute paths (→ relative
to MODELS_DIR, load_model resolves both). S824 MoE fp16 accumulator (→ fp32
upcast + cast back). S820 false positive (port 8080 consistent everywhere).
S821 false positive (dot product of L2-normalized features IS cosine similarity).

**Bug fixes (Pass 115):** S812 ToMe causal mask (rebuilt proper triu mask after
merge). S813 PPO replay zero gradient (ReplayBuffer now stores
full_ids/prompt_len/ref_logps — replay items can recompute log-probs). S815 API
key config wired (run.py reads CONFIG fallback). S819 alignment mode visibility
(GRPO/ReMax/SimPO/ORPO show only basic section). S822 ToMe size guard (skip
merge for T>4096). S825 async queue unbounded (was maxsize=1000, dropped tokens).

**Audit (Pass 113):** S828 incomplete — 7 of 13 training paths had finally
clear but no assignment before train(). Fixed: RLHF, Self-Play, RL variant,
SimPO/ORPO, Dialogue, Evolutionary, Adaptive now all assign _active_trainer
before train().

**Pass 114:** S831 stop button slow (GUI set training_active=False but never
called trainer.request_stop() — Trainer's _should_stop() at batch boundary
never triggered; stop only worked through callback after batch completed).
Fixed: _stop_training() now calls _active_trainer.request_stop(), changes
button to "STOPPING...", resets to "STOP" in all 14 finally blocks. S832
resume UX confusing ("Resume" checkbox too small/vague). Fixed: renamed to
"Resume from checkpoint", tooltip rewritten, TRAIN button shows "RESUMING..."
when checkpoint found, Trainer emits progress with step/epoch info. Cleanup:
deleted 7 junk files, 10 orphaned model contexts, 10 .bak files; moved 19
curriculum files to data/curriculum/, other_git_repos.md to information/.

**Pass 116 (principles audit):** S833 rl_training.py — `advantages.std()`
missing `unbiased=False` in RLHF PPO (line 1468) and SelfPlay PPO (line 2118).
Bessel's N-1 correction biased for small RL batches. Fixed: explicit
`unbiased=False`, matching GRPO which was already correct. S834 model.py —
`config_path.write_text()` non-atomic (line 1417). Fixed: uses
`atomic_write_text()` from safe_save. S835 huggingface_loader.py — error
handler missing traceback (line 176). Fixed: `logger.warning` now includes
`traceback.format_exc()`. S836 mod_tools.py — error handler missing traceback
(line 93). Fixed: added `logger.error` with `traceback.format_exc()`. S837
training.py — `TrainingConfig.to_dict()` missing 7 fields: `amp_dtype`,
`run_evaluation`, `eval_test_prompts`, `z_loss_weight`, `use_lisa`,
`lisa_activated_layers`, `golden_eval_path`. These fields were lost on
checkpoint save/load roundtrip. Fixed: all 7 added to to_dict(). Structural
test added to prevent regression. S838 builtin_commands.py — code_run sandbox
missing `os.exec*`, `os.spawn*`, `subprocess.getoutput/getstatusoutput`.
Fixed: added to forbidden list. Temp file leaked on non-TimeoutExpired
exceptions — added cleanup to general except handler. S839 plugin_loader.py —
_DANGEROUS_ATTRS missing `subprocess.getoutput/getstatusoutput`,
`ctypes.cdll/windll/CDLL/WinDLL`. Import flagging missing `ctypes`. Fixed:
all added.

**Pass 117 (principles audit continuation):** KL penalty consistency (all 4 RL
trainers — RLHF PPO, SelfPlay PPO, GRPO, ReMax) — all use
`(policy_logps - ref_logps).mean()` consistently. ✓ Clean, no changes.
Streaming stop-string logic — `stopped` flag present in stream_chat()
(`if pending and not stopped`); stream_generate() uses `break` then eof
cleanup only at chat layer. ✓ Clean. CORS defaults — opt-in only via
`--cors-origins` flag, no default wildcard. ✓ Clean. Loss weighting non-pad
tokens — main CE loop (`epoch_loss += batch_loss * non_pad`) correctly
weighted; alignment trainers (DPO/KTO/ORPO/SimPO) use pair-level loss, not
token CE, weighting N/A. ✓ Clean. Dead imports — `tool_executor` guarded by
`if enable_tools:` (never set True), `universal_router` guarded by
`try/except ImportError`. ✓ Clean. LambdaLR warmup zero-LR —
`(step + 1) / warmup` already correct at both sites (lines 2738, 4016).
✓ Clean. **All principles audit checks complete for this pass — 0 new bugs
found.** S823 PPO triple forward pass: added `_get_logps_hidden_entropy()` —
single model pass for logps + hidden + entropy. RLHF rollout 2→1 passes, RLHF
minibatch 3→1, SelfPlay rollout 2→1, SelfPlay minibatch 3→1, ReMax update
2→1. 4 tests added. 2262 tests pass.

**Pass 118 (small-things audit):** S823 hardening follow-up in
[enigma_engine/core/rl_training.py](../../enigma_engine/core/rl_training.py):
`_get_logps_hidden_entropy()` now mirrors NEFTune train-mode behavior and
raises if output head missing; dead `_get_hidden_states()` methods removed
from RLHF/SelfPlay trainers. Targeted tests pass (5/5). New lint findings
logged (S840-S844) from GUI/tests hygiene sweep. GUI runtime check: multiple
`run.py --gui` processes remained alive; no crash observed.

**Pass 119 (resilience):** _LOG_MAX_FILES 10→100 in
[enigma_engine/gui/gui_forge.py](../../enigma_engine/gui/gui_forge.py) (log
rotation was deleting old run logs). Training heartbeat file added
(`logs/training_heartbeat.json`) — written to disk every 30s during
pre-training with {pid, status, model, phase, step, loss, timestamp};
survives OS OOM kills. Stale heartbeat detection added to
`_start_pretrain_training()` — checks previous session's heartbeat on launch,
logs warning if prior session didn't exit cleanly. RAM warning at 80% during
data load phases (Phases 1+4). NaN/Inf loss detection with early return in
`on_loss` callback. Lint: `ruff check gui_forge_new_modes.py` clean after
removing dead `import os as _os_stale`. Pre-existing S840 (unused import json
in gui_forge.py) NOT from this session.

**Pass 120 (test hardening):** Replaced structural source-text assertions
with 3 behavioral mock-based tests in `TestForgeDistillRuntime`
(tests/test_training.py). Tests: (1) vocab mismatch logs error + Trainer.train
not called, (2) stop flag after generation prevents training phase, (3)
accepted example logs full User+Assistant text. Key pattern:
`enigma_engine.core` lazy `__getattr__` breaks `patch("module.submodule.Class")`
— fixed by injecting fake sub-modules via `patch.dict(sys.modules, {...})`.
All 6 distill tests pass. No GUI or core changes.

**Pass 121 (tests + GUI persistence):** Rust BPE wheel was stale — `train()`
added to lib.rs but old .pyd in venv had no train method. Rebuilt with
maturin + reinstalled; 4 TestRustBPETrain failures resolved (all 8 Rust BPE
tests pass). GUI Forge settings persistence had two bugs: (1) `_on_close` in
desktop.py never called `_save_training_brief()` — any changes typed without
triggering a mode-switch were lost on close. Fixed: added safe
`getattr(self, '_save_training_brief', None)` call before
`_save_window_geometry`. (2) `_save_training_brief` / `_load_training_brief`
in gui_forge.py missing 16 widget vars (distill examples/tokens, reasoning/
evolutionary/auto-train/resume checkboxes, general mix, training stage,
replay capacity/ratio, vision dir/preset, AI supplement path, quantize mode,
GGUF export mode, guided pairs, web learn pages, vocab size). All now saved
and restored.

---

## CODE_REVIEW.md — Top-of-file pass prose (archived Pass 135)

**Pass 135 (April 22, 2026) — Doc hygiene only.** Converted 138 inline
backticked file-path references in SUGGESTIONS.md to proper markdown
hyperlinks; all link targets verified present on disk (including ambiguous
`core/bones.py` / `core/model.py` in the avatar research item, which resolve
to `mods/avatar/enigma_avatar/core/...` not `enigma_engine/core/...`). Fixed
one escaped `\n` literal that was breaking a table row in the Accepted Risk
section. Nine backticked refs intentionally left unlinked (non-existent files,
generated artifacts, shell commands, method-name mentions already linked
elsewhere in the same bullet). No code, tests, or GUI changes.

**Pass 134 (April 22, 2026):** Design discussion on controlled unpredictability.
No code changes. New research/design item R-UNPREDICT-1 logged in
SUGGESTIONS.md: split unpredictability into **behavior** (driven by internal
signals — confidence, emotional_state, novelty — must surprise) vs
**infrastructure** (seeded training, deterministic tests, atomic saves —
must reproduce). Four concrete build targets identified: (1) self-initiated
research from model confidence, (2) mood-weighted replay in BackgroundTrainer,
(3) monologue prompt variance shaped by emotional_state, (4) `<search>`
curiosity token mid-generation. Links to AutoResearch-2, GRPO-4,
Personality-5, N-25. Non-negotiables: seedable training, deterministic tests,
every unpredictable feature has debug off-switch + reproduction seed. No GUI
or core edits in this pass.

**Pass 133:** Corrected Pass 132 over-retraction. Pass 132 (audit-on-audit)
correctly flagged Personality-2 (ai_profile.personality dict is user-set, do
not inject) but over-retracted on Personality-1. `emotional_state` is
AI-computed from sentiment analysis in `core/sentiment.py`, not user-set.
N-22 (Pass 125) in `gui_logic.py` line 335 injects it as "Internal State:
... color your tone naturally" — this is AI self-awareness of its own
computed state, NOT user configuring personality. That closure is consistent
with design intent. Personality-1 retraction REVERSED. Personality-2
retraction STANDS. Personality-3/4/5 (ai_profile.personality dict scoping,
identity-vs-roleplay separation, FORGE Distillation personality category
never scheduled) all stand. R-PERSONALITY-1 research item stands.

---

## GUI_REFERENCE.md — Top-of-file pass notes (archived Pass 135)

**Pass 135 note:** No GUI layout, widget, or behavior changes. Doc hygiene
only — converted inline backticked file-path references in SUGGESTIONS.md to
proper markdown hyperlinks (138 links verified against disk). GUI layer
unaffected.

**Pass 134 note:** No GUI layout, widget, or behavior changes. Design
discussion only — R-UNPREDICT-1 logged in SUGGESTIONS.md covering controlled
unpredictability (behavior driven by internal signals; infrastructure stays
deterministic and seedable). Four candidate build targets identified but
none implemented: confidence-driven auto-research, mood-weighted replay,
monologue prompt variance, `<search>` curiosity token. GUI layer unaffected
in this pass.

**Pass 133 note:** No GUI layout, widget, or behavior changes. Audit
clarification only: reviewed N-22 (Pass 125) `[Internal State: ...]` injection
in `gui_logic.py::_build_gui_context()` line 335. Confirmed **consistent with
design intent** — `emotional_state` is AI-computed from sentiment analysis
(`core/sentiment.py::compute_engagement_score`), not user-configurable.
Injecting AI-computed internal state ≠ user configuring AI personality.
N-22 stays closed. Separately confirmed that `ai_profile.personality` dict
(tone/verbosity/formality/humor in `core/ai_profile.py` line 105) is USER-SET
and must NOT be injected into the system prompt — that would be user
configuring the AI's character, contradicting the design intent.

**Pass 128 note:** Research-only pass. No GUI layout, widget, or behavior
changes. Full codebase scan completed: all core/ files, all mods/ files
(vision, codegen, avatar, audiogen, imagegen, videogen, threed, voice,
transcriber), router.py. Findings written to SUGGESTIONS.md. Key GUI-relevant
discoveries: (1) `mods/vision/vision.py` is screen capture + OCR, NOT LLM
multimodal vision — Vision-6 research gap logged. (2)
`mods/avatar/enigma_avatar_brick.py` is production-ready with full bone rig
but has no text-to-animation pipeline — Avatar-3 research gap logged.
(3) `mods/codegen/codegen.py` calls base 742M as LocalCode — no specialist
model. (4) `core/training_evaluation.py` is perplexity-only — no benchmark
suite — Eval-1 gap confirmed.

**Pass 127 note:** Lint cleanup only. No GUI layout, widget, or behavior
changes. `ruff check enigma_engine/ tests/` now passes with zero issues.

**Pass 126 note:** N-21 implemented. MODELS page now has an inline merge row
below the HuggingFace download row.
- Two model dropdowns (populated from the live model registry, auto-updated
  when models are added/removed)
- Method dropdown: SLERP / LINEAR / TIES
- t value numeric entry (0.0–1.0; used by SLERP and LINEAR)
- density numeric entry (0.0–1.0; used by TIES)
- Output name entry (auto-generated as `modelA_mode_modelB` if left blank)
- MERGE button — runs in a background thread via `core/model_merging.py`;
  shows inline status, refreshes model list on success
- Handler: `ForgeModelsMixin._merge_models()` in `gui_forge_models.py`
- Dropdowns sync live when models are created/deleted (via
  `_refresh_page_models()`)
- 2 structural tests: `TestModelsPageMerging` in `tests/test_gui.py`

**Pass 125 note:** N-22 and N-23 implemented. No GUI layout or widget changes.
- **N-22 closed:** `_build_gui_context()` in `gui_logic.py` now injects a
  `[Internal State: valence=X, arousal=Y, ...]` tone cue just before
  `[END SYSTEM CONTEXT]`, read from
  `model_context._snapshot_emotional_state()`. The AI's generation now
  reflects the 5-dimensional emotional state naturally. 5 behavioral tests
  cover all label thresholds and exception safety.
- **N-23 closed:** The 25-line RAG widget-peek block removed from
  `_build_gui_context()`. Retrieval now lives in
  `engine_chat.py::_prepare_chat()` — any code path that calls
  `_prepare_chat()` (GUI, API server, BackgroundTrainer) automatically gets
  document context when `engine._rag_index` is set. `gui_logic.py` now wires
  `engine._rag_index` on every RAG toggle on/off and build/fail path.

**Pass 124 note:** N-25 implemented. `BackgroundTrainer` in `router.py` now
defers training batches while inference is active via an idle callback wired
from `_is_generating`. No GUI layout or widget changes.

**Pass 123 note:** NaN/Inf guards added to SimPO and KTO training loops in
`core/training.py`. Both modes were missing the abort-on-NaN check that every
other training mode had (main loop, DPO, vision, audio). Now: when
`batch_loss` is NaN or Inf, training sets `state.abort_reason`, calls
`model.eval()`, and returns early — matching the DPO abort pattern exactly.
No GUI layout or widget changes. Design review identified 5 architectural
gaps logged as N-21 through N-25 in SUGGESTIONS.md.

**Pass 122 note:** Tokenizer selection bug fixed in `gui_forge_training.py`.
All 4 training paths (Basic/Solo, DPO, Vision, LoRA) previously called
`get_tokenizer("auto")` which picked tiktoken (100,283 vocab) when installed,
causing the vocab guard to reject locally-trained models (4,713 vocab). All
paths now try `MODELS_DIR/tokenizer.json` as BPETokenizer first, fall back to
`get_tokenizer("auto")` only if that file doesn't exist. No GUI layout or
widget changes.

**Pass 121 note:** GUI Forge settings persistence fixed. `desktop.py`
`_on_close` now calls `_save_training_brief()` before destroying widgets —
previously, settings changed without a mode-switch were lost on close.
`gui_forge.py` save/load expanded to cover 16 previously-missing widget vars.
No layout or visual changes.

**Pass 120 note:** Test hardening only — 3 new behavioral mock-based tests
in `TestForgeDistillRuntime`. No GUI layout, widget, or behavior changes.

**Pass 119 note:** Resilience improvements to pre-training. No layout or
widget changes visible to the user; behavior change is new log output in
FORGE panel (RAM warnings, NaN alerts, heartbeat status messages on next
launch if prior session crashed).

---

## How to use this archive

- When a new pass completes, move its prose block here and leave a one-line
  "Pass NNN: short summary" in the active doc.
- Never delete entries — they are the only record of *why* we reversed a
  prior decision.
- If a pattern recurs across 3+ passes, promote it to a Learned Principle in
  `AA code maker.md` and cross-reference back here.
