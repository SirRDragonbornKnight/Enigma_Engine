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

- [x] **BUG-1 (CRITICAL):** `enigma_engine/core/model_components.py` ~L426 — guard `_kv_share_source._kv_cache` before calling `.get()` when `use_cache=True` on first forward. Add test `test_kv_share_first_forward_no_crash`. **Closed `da65525` (May 27 2026). Fix uses fall-through-to-local-K/V when source layer hasn't warmed yet; 2 regression tests (`test_kv_share_first_forward_no_crash`, `test_kv_share_after_clear_cache_no_crash`) with falsification verified.**
- [x] **BUG-2:** `enigma_engine/core/model.py` `_apply_static_int8_quantization()` — fix log message to say "dynamic" when falling back. **Closed `da65525`. Success log moved inside helper to reflect actual path; INT4 fallback also corrected. 2 caplog tests.**
- [x] **BUG-3:** `enigma_engine/core/inference.py` `generate()` — add `ValueError` when more than one of `max_tokens`/`max_new_tokens`/`max_length` is passed, matching `stream_generate()`. **Closed `da65525`. Sibling-boundary closure on `_prepare_chat` followed May 27 2026 audit; chat path now raises same `ValueError`.**
- [x] Run `ruff check enigma_engine/ tests/` and `python -m pytest tests/ -q` — confirm still green after fixes. **3310 passed, 3 skipped, ruff clean (May 27 2026).**

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

#### Block 4.5 — Layered personality (PERSONA-2)

**Trigger.** May 27 2026 review surfaced that the current "Personality from training, not the user" constraint is too tight — it conflates **core identity** (correct to lock) with **surface style** (better adjusted per-context). The strongest assistants (Claude, ChatGPT) combine both. Black-box constraint preserved: model artifact stays untouched, style preferences are user-side runtime config that follows the user, not the model.

**Design — layered personality:**

| Layer | What | Where stored | Who sets it | How applied |
|---|---|---|---|---|
| 1. Core identity | Voice, humor, values, reasoning style — what makes Enigma *Enigma* | LoRA adapter weights | Training corpus + TEACH-1 corrections | Active LoRA at inference |
| 2. Style preferences | Verbosity, formality, default length, output format | `data/style_preferences.json` (atomic save) | User via GUI settings | Injected as `[USER STYLE PREFERENCES]` block in system prompt |
| 3. Per-conversation overrides | "Be terse for this chat" via natural language or `/style` command | Conversation state (in-memory) | User mid-conversation | Layered on top of (2) for current conversation only; resets on new conversation |

**Constraint to change in AA code maker.md:**
- FROM: *"Personality from training, not the user — the AI's voice, mood, and style are learned, not configured per-session."*
- TO: *"Core identity from training; surface style from the user. The AI's voice, humor, values, and reasoning patterns are LoRA-trained and locked — users don't edit weights, and asking the AI to 'be a pirate' doesn't override its core. Surface preferences (verbosity, formality, length, output format) are user-adjustable per-conversation via the profile or natural-language requests. Identity corrections feed training; style corrections feed the profile."*

**Implementation slices (in order; each is shippable):**

- [ ] **Slice 0 — Update constraint wording.** Edit AA code maker.md §Project Goal Constraints. Document the layered model in the same constraint block. Smallest change; gates everything else.
- [ ] **Slice 1 — `StylePreferences` schema + storage.** New dataclass in `enigma_engine/core/style_preferences.py` with: `verbosity` (terse/normal/verbose), `formality` (casual/neutral/formal), `default_response_length` (short/medium/long), `prefer_code_examples` (bool), `prefer_bullet_points` (bool). Atomic load/save to `data/style_preferences.json`. Defaults preserve current behavior — no regression for existing users. Tests: schema round-trip, atomic save, defaults match current behavior.
- [ ] **Slice 2 — Inject into system prompt.** In `_build_gui_context()` (gui_logic.py:196) and the equivalent API-side path, assemble a `[USER STYLE PREFERENCES]` block when ANY preference is non-default. Skip injection entirely when all defaults (zero overhead for users who don't customize). Block format is machine-readable enough for the LLM to use AND human-readable enough that the user can see what's being applied. Tests: caplog/captured-prompt assertion that the block appears when non-default, absent at defaults.
- [ ] **Slice 3 — GUI controls.** One settings panel in Gradio UI (and CONFIG page in tkinter while it lives) with 5 controls — 3 dropdowns + 2 checkboxes. Save button persists to disk. "Reset to defaults" button. Tests: round-trip save/load through the GUI handler.
- [ ] **Slice 4 — Per-conversation override commands.** Chat-side: user can say `/style terse`, `/style normal`, `/style reset` (chat commands) to override Slice 2 settings for the current conversation only. Resets on new conversation. Also: natural-language detection ("be more concise" → infer `verbosity=terse` for this chat). Tests: override persists per-conversation, doesn't leak to next conversation.
- [ ] **Slice 5 — Corrections-loop categorization. DEFERRED — blocked on Block 2 (Gradio UI).** When user clicks FIX (TEACH-1), they pick: (a) "What it said" → feeds DPO replay (current behavior), or (b) "How it said it" → updates style preferences. Default if unspecified: identity (preserves current behavior). Tests: identity corrections still flow to existing DPO pipeline (no regression); style corrections update preferences without touching weights. **Why deferred (May 27 2026):** the categorization UX needs an actual UI affordance — the FIX button has to surface a "What's wrong?" picker for the user to choose between identity and style. Without the Gradio UI (Block 2) shipped, building the categorization plumbing now would be backend code with no user-facing path to exercise it (dead-infra anti-pattern). Resume this slice immediately after Block 2 lands: add the picker to the Gradio FIX-button flow, then wire the categorized correction through `BackgroundTrainer.ingest_corrections_file()` (identity branch) vs `PUT /api/style-preferences` (style branch).

**Tests to write FIRST (per Test Loop):**
- Default behavior unchanged when style preferences are at defaults — no regression for existing users
- Style preferences round-trip through JSON
- Style preferences inject as a clearly-marked block, never silently
- Per-conversation overrides don't leak across conversations
- Identity corrections still feed DPO (no behavior change to existing TEACH-1 path)
- Black-box invariant: model file hash unchanged when style preferences are written

**Risks / open questions:**
- **Schema bloat.** 5 preferences is enough. Don't add more until a real user need surfaces. YAGNI.
- **Contradictory preferences** (e.g. "terse" + "always include code examples"). Document precedence: more-specific overrides more-general. Test the edge case.
- **Style override of identity** (e.g. user types `/style "be a pirate"`). REFUSE — that's identity, not style. The chat command takes only known enum values; refuse free-text.
- **Slice 5 is the hardest part.** Could rathole. Defer until Slices 0-4 are green and we see what corrections users actually make.
- **Multi-persona** (e.g. "work me" / "creative me" with different preferences). Out of scope for now. Design Slice 1 schema so it could extend to multiple profiles later.
- **Per-tool style** (terse in code, chatty in casual). Out of scope. Could later key style off the active tool context.

**Definition of done:**
1. Constraint updated in AA code maker.md
2. `StylePreferences` schema + storage works
3. Injection into system prompt is observable in tests
4. GUI controls round-trip
5. Per-conversation overrides work and reset cleanly
6. Corrections-loop categorization wired (basic version)
7. Black-box invariant: no model file written by this feature; all preferences are user-side runtime config
8. All tests green, ruff clean
9. Existing TEACH-1 / DPO path unchanged for identity corrections

**Honest scope estimate:** Slices 0-4 are ~1-2 days solo. Slice 5 is a separate slice that might take 1-2 more days because the user-categorization UX is the trickiest part. Total: 3-4 working days for a complete layered-personality feature.

**What this does NOT change:**
- The LoRA training pipeline — unchanged
- The TEACH-1 corrections loop for identity — unchanged
- The `AIProfile` task-overlay system_prompt — unchanged
- The black-box constraint — preserved (model artifact never touched by this feature)

---

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

### Recent closures (May 27 2026)

**Backward-compat shim sweep** — per AA code maker §2 ("Do not add backward-compatible shims"). Cleaned up actual rule violations in code:

- `enigma_engine/core/model_config.py` — deleted (whole-file re-export shim for `MODEL_PRESETS`)
- `enigma_engine/core/tokenizer.py::load_tokenizer()` — deleted (alias for `get_tokenizer()`); single caller migrated
- `enigma_engine/core/gguf_loader.py` re-exports — deleted (15 dequant/parse symbols re-exported "for backward compatibility"; zero callers)
- `enigma_engine/api/server.py::TrainRequest` legacy SFT-only shape — deleted (`data_file`+`epochs`+`learning_rate`+`batch_size` parallel to dispatcher shape); `mode` field is now required; 6 shim tests deleted, 2 new tests added for new shape + legacy rejection

**Mislabeled comments fixed:**
- `inference.py` / `engine_generation.py` — `max_tokens`/`max_new_tokens`/`max_length` were tagged "backward compatibility"; relabeled as industry-standard SDK aliases (HF/OpenAI/Anthropic)
- `rag.py::TfidfVectorizer` — "kept for backward compatibility" replaced with honest "rename has high cost (30+ test imports + on-disk schema)"
- `/api/history` GET/DELETE routes — "legacy alias" / "legacy nuke route" replaced with honest "convenience over per-conv route" / "complements per-conv DELETE"

**F1-F4 audit closures** (earlier May 27 session):
- F1: BUG-1/2/3 status flipped to `[x]` in SUGGESTIONS.md (had been `[ ]` open in two duplicate sections)
- F2/F3: CLEANUP_TRACKER stale claims about already-closed bugs in shell_cmd and engine_generation refreshed
- F4: `_prepare_chat` silent alias-pop closed — same `ValueError` gate as `generate()` and `stream_generate()` now applied at the chat layer too (sibling-boundary miss from the original BUG-3 fix)

Final suite after sweep: **3306 passed, 3 skipped**, ruff clean.

---

## Archive

Pre-May-26-2026 passes (pre-Strategy-Reset) live in [history/SUGGESTIONS-archive-pre-reset.md](history/SUGGESTIONS-archive-pre-reset.md).
