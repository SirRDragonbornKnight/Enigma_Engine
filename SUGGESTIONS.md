# Suggestions

## STRATEGY RESET — May 26, 2026 (ACTIVE — supersedes all prior active sections)

**Trigger.** Review session confirmed two strategic decisions that change execution priority going forward.

### Decision 1 — Primary brain = Qwen3 fine-tuning, not from-scratch training

The custom `Enigma` transformer is architecturally well-built, but training it from scratch on consumer hardware (16 GB VRAM) cannot close the gap with Qwen3's pretrained knowledge for the breadth of tasks Enigma AI targets. Qwen3-8B (or 30B-A3B on capable hardware) fine-tuned with LoRA is the primary brain path. The from-scratch `Enigma` model stays in the codebase as an experimental / research track only — it does not drive the roadmap.

**What this means in practice:**
- Load and serve Qwen3 via the existing HuggingFace loader or llama-server GGUF path
- Personality + task specialisation come from LoRA SFT runs on curated data via `training/training.py`
- The custom `Enigma` architecture (ForgeConfig presets, scratch training) is not deleted — it's just not the production brain

### Decision 2 — UI = Gradio, replacing both Svelte and tkinter

The Svelte frontend (`enigma_engine/web/`) cannot build as-is (five missing pages: `Training.svelte`, `Files.svelte`, `Models.svelte`, `Config.svelte`, `Terminal.svelte`) and requires TypeScript context-switching away from Python. The tkinter GUI (`enigma_engine/gui/`) is already marked for deletion in `CLEANUP_TRACKER.md`. Both are replaced by a single `enigma_engine/ui.py` built on Gradio.

**Why Gradio:**
- Pure Python — same language as the entire engine, no context switching
- Browser-rendered — no Electron, no tkinter, works on any machine
- Built for AI projects — streaming chat, image/audio upload, sliders for sampling params, all built-in
- `enigma_engine/client.py` already exists and can be reused directly

---

### What stays unchanged
- `enigma_engine/core/` — full brain stack (model, inference, chat, training, RAG, memory, reasoning, vision/audio encoders, LoRA utils). No changes.
- `enigma_engine/training/` — training pipeline. Kept. Focus shifts to LoRA SFT on Qwen3.
- `enigma_engine/api/server.py` — FastAPI server. Kept. Gradio UI talks to this.
- `mods/` — mod system kept. Work on individual mods resumes after brain is solid (Block 5).
- All tests, lint discipline, and audit practices. No changes.

### What is parked / scheduled for removal
- `enigma_engine/web/` — Svelte frontend. **Scheduled for deletion after Gradio UI is verified.** See CLEANUP_TRACKER.md.
- `enigma_engine/gui/` — tkinter GUI. Already marked for deletion in CLEANUP_TRACKER.md. Unchanged.
- From-scratch `Enigma` scratch training as a primary goal. **Demoted to experimental only.**

---

### Execution sequence

#### Block 0 — Pre-flight bug fixes (do these first, ~1 hour of work)

These are small targeted fixes. Do them before any training run so the model stack is clean.

- [ ] **BUG-1 (CRITICAL):** `enigma_engine/core/model_components.py` ~L426 — guard `_kv_share_source._kv_cache` before calling `.get()` when `use_cache=True` on first forward. Add test `test_kv_share_first_forward_no_crash`.
- [ ] **BUG-2:** `enigma_engine/core/model.py` `_apply_static_int8_quantization()` — fix log message to say "dynamic" when falling back.
- [ ] **BUG-3:** `enigma_engine/core/inference.py` `generate()` — add `ValueError` when more than one of `max_tokens`/`max_new_tokens`/`max_length` is passed, matching `stream_generate()`.
- [ ] Run `ruff check enigma_engine/ tests/` and `python -m pytest tests/ -q` — confirm still green after fixes.

#### Block 1 — Get Qwen3 loading and chatting

This is the real starting gate. Everything else builds on a working Qwen3 base.

- [x] **Download Qwen3:** Already on disk.
  - `models/qwen3-8b/` — full HF safetensors (5 shards + config.json). **Use this for LoRA fine-tuning.**
  - `models/qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf` — 30B MoE GGUF. **Use this for heavy inference/chat via llama-server.**
- [ ] **Verify load:** Start the API server (`python run.py --serve --port 8081`), then `POST /api/models/load` with the path. Confirm `/api/models/status` returns `loaded`.
- [ ] **Verify chat:** `POST /api/chat` with a simple message. Confirm a real text response comes back — not empty, not an error.
- [ ] **Verify stream:** `POST /api/chat/stream` (SSE). Confirm tokens arrive incrementally.
- [ ] **TRAIN stability check:** `POST /api/train` with `smoke_test_basic.txt`. Confirm lifecycle: `active → complete`, `abort_reason: ""`.

#### Block 2 — Gradio UI

- [ ] Build `enigma_engine/ui.py` — Gradio app with: streaming chat tab, model load/unload controls, sampling parameter sliders (temperature, top-p, top-k), memory/RAG status display, training trigger tab
- [ ] Reuse `enigma_engine/client.py` as the API bridge — no new HTTP layer needed
- [ ] Smoke-test: load Qwen3, send a chat message, verify streaming response appears in UI
- [ ] Delete `enigma_engine/web/` (Svelte scaffold) once UI is verified
- [ ] Delete `enigma_engine/gui/` tkinter files per CLEANUP_TRACKER.md list

#### Block 3 — AutoResearch inline search (Stage B-2 / B-3)

The token registry is done (`reasoning.py` Stage B-1). B-3 RAG injection is done in `engine_generation.py`. The missing piece is the B-2 generation hook.

- [ ] **B-2:** The `_generate_text()` function already injects `</search>` as a stop token (Stage B-3a). What's missing: after the model emits `<search>query</search>` and stops, the caller needs to dispatch the query to RAG/web and inject results back before resuming. Wire this loop in `_generate_text()` or `stream_generate()`.
- [ ] Tests: hook fires on `<search>` token emission, RAG results injected into context, generation resumes correctly

#### Block 4 — Personality corpus + Qwen3 LoRA fine-tune

This is the most impactful block. The training infrastructure already exists — what's needed is the data.

**Step 4.1 — Corpus design**
- [ ] Define the voice: pick 5–10 adjectives that describe how Enigma speaks (tone, formality, quirks). Write these down in a `data/enigma_voice.md` reference doc.
- [ ] Define task coverage: list the topic areas Enigma should handle well (coding help, general Q&A, creative writing, reasoning, casual chat). Aim for ~equal distribution across categories.
- [ ] Corpus format: each example is a JSON object `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`, one per line (JSONL). Match the ChatML format Qwen3 uses.

**Step 4.2 — Corpus collection (~1 000 examples minimum)**
- [ ] Generate an initial batch using a strong base model (GPT-4o, Claude, base Qwen3) — prompt it to respond in Enigma's documented voice. Filter every output through `personality_data.py` (`passes_quality_filter`, `is_near_duplicate`, identity-leak check).
- [ ] Write ~50–100 examples by hand for the categories where voice matters most (casual chat, self-introduction, opinion questions). These hand-written examples anchor the fine-tune.
- [ ] Save to `data/personality_corpus.jsonl`.

**Step 4.3 — LoRA SFT run**
- [ ] Config: `learning_rate=2e-4`, `lora_r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `epochs=3`, target modules `q_proj,v_proj` (standard for Qwen3). These are safe defaults — adjust after first eval.
- [ ] Launch: `POST /api/train` with `mode: "sft"`, point at `data/personality_corpus.jsonl`, student model = `models/Qwen3-8B`.
- [ ] Monitor: watch `loss` curve via `/api/training/status`. Expect loss to drop from ~2.5 → ~1.0 over 3 epochs on 1 000 examples.
- [ ] Save adapter to `models/enigma_lora_v1/`.

**Step 4.4 — Evaluate**
- [ ] Load Qwen3 + apply adapter: `POST /api/models/load`, then `POST /api/models/adapter` with `models/enigma_lora_v1/`.
- [ ] Run coherence benchmark: `python -c "from enigma_engine.core.monologue import run_coherence_benchmark; ..."` — target score ≥ 0.6 ("ready").
- [ ] Side-by-side: chat with base Qwen3 vs fine-tuned Enigma on 5 test prompts. Does it sound like Enigma?
- [ ] If score < 0.6 or voice is wrong: add more hand-written examples in the weak areas, re-run.

**Step 4.5 — Corrections loop (ongoing after first eval)**
- [ ] Use the TEACH-1 FIX button (or `data/corrections.jsonl` directly) to log wrong answers during chat.
- [ ] `BackgroundTrainer` picks these up automatically on the next replay cycle.
- [ ] After 50+ corrections accumulate, a DPO pass runs automatically (`_maybe_train_dpo_pairs()` in `router.py`).

#### Block 5 — Mods (one at a time, only after Blocks 1–4 are green)

- [ ] Vision quality: end-to-end test with a real vision-capable model
- [ ] imagegen: flip `default_provider` from `"placeholder"` to `"local"` with weights-present gate (REALIGN-1.2 slice 2.1)
- [ ] audiogen / voice duplicate resolution (REALIGN-1.2 slice 2.1)
- [ ] Avatar rebuild: new `mods/avatar/main.py` at correct launcher depth, real `mod_base.py` protocol, renderer pick, load `bone_limits.json`
- [ ] Remaining mods in priority order

---

### Definition of done for "brain ready"
1. Qwen3-8B or 30B-A3B loads, chats, and streams correctly end-to-end
2. LoRA personality fine-tune applied and coherence-scored above `DEFAULT_COHERENCE_THRESHOLD` baseline
3. AutoResearch B-2/B-3 wired — model can look things up mid-generation
4. Gradio UI running, functional, and smoke-tested
5. All tests green, lint clean

---

### Pre-Block-1 bug fixes (from May 26 deep audit — fix before next feature work)

Three bugs found in the deep audit of `enigma_engine/core/`. All are small, targeted fixes. Full details in `CODE_REVIEW.md` under **Deep Audit Pass — May 26, 2026**.

- [ ] **BUG-1 (CRITICAL):** `model_components.py` ~L426 — cross-layer KV sharing crashes with `AttributeError` on first forward pass when `use_cache=True` because `_kv_share_source._kv_cache` is `None`. Guard needed before calling `.get()`.
- [ ] **BUG-2 (low):** `model.py` `_apply_static_int8_quantization()` — logs `"Applied static INT8 quantization"` when it actually ran dynamic INT8. Fix the log message.
- [ ] **BUG-3 (medium):** `inference.py` `generate()` — silently accepts multiple conflicting aliases (`max_tokens`/`max_new_tokens`/`max_length`) last-wins, while `stream_generate()` raises `ValueError`. Align `generate()` to match.

---

## Archive

Pre-May-26-2026 passes (pre-Strategy-Reset) live in [history/SUGGESTIONS-archive-pre-reset.md](history/SUGGESTIONS-archive-pre-reset.md).
