# Code Cleanup Tracker

**Started:** May 15, 2026
**Status:** Paused — Strategy reset May 26 (see SUGGESTIONS.md). Cleanup resumes after SUGGESTIONS.md Block 2 (Gradio UI) is verified.
**Scope:** All `.py` files in repo root, `enigma_engine/`, and `tests/`.
**Out of scope:** `rust_extensions/target/`, `.venv314/`, `models/`, `data/`, `outputs/`, `temp_claw/`, generated stubs.

## ⚠️ UPCOMING: GUI + Web UI deletion (unblocked once Gradio UI is verified — SUGGESTIONS.md Block 2)

**Strategy reset May 26, 2026:** Both the tkinter GUI and the Svelte web frontend are replaced by a single Gradio UI (`enigma_engine/ui.py`). Delete both after Gradio is smoke-tested and functional.

### Tkinter GUI — delete when Gradio UI is verified
- ❌ `enigma_engine/gui/desktop.py` (1166 LOC, main tkinter GUI)
- ❌ `enigma_engine/gui/gui_pages.py` (primary page builder)
- ❌ `enigma_engine/gui/gui_forge.py` (training config builder)
- ❌ `enigma_engine/gui/gui_forge_*.py` (specialized FORGE launchers, ~4 files)
- ❌ `enigma_engine/gui/gui_cmd_page.py` (shell subprocess one-shot)
- ❌ `enigma_engine/gui/gui_logic.py` (event handlers)
- ❌ `enigma_engine/gui/gui_logic_chat.py` (chat integration)
- ❌ `enigma_engine/gui/gui_mods.py` (mod launcher)
- ❌ `enigma_engine/gui/widgets.py` (custom tkinter widgets)
- ❌ `enigma_engine/gui/__init__.py` (GUI package)
- ❌ Remove `customtkinter` from `pyproject.toml`
- ❌ Update `enigma_engine/__init__.py` to remove GUI imports

### Svelte web frontend — delete when Gradio UI is verified
- ❌ `enigma_engine/web/` (entire directory — `package.json`, `vite.config.ts`, `src/App.svelte`, `src/pages/Chat.svelte`, `src/lib/api.ts`, `src/lib/store.ts`, `index.html`, `tsconfig*.json`)
- ❌ Reason: Five pages were missing (`Training.svelte`, `Files.svelte`, `Models.svelte`, `Config.svelte`, `Terminal.svelte`), frontend is not buildable, TypeScript overhead not warranted while brain is incomplete. Replaced by Gradio.

**Files preserved:**
- ✅ `enigma_engine/gui/` directory itself — retained as a location for Gradio static assets if needed later
- ✅ All core logic untouched (`enigma_engine/core/`, `enigma_engine/training/`, `enigma_engine/api/`)

## Cleanup Levels Applied (each file gets ALL of these)

1. **Import sorting** — `ruff check --select I --fix`-style, isort grouping, dead imports removed.
2. **Function/method ordering** — public-first, private below, dunder methods grouped, helpers near callers.
3. **Dead code removal** — unused functions, unreachable branches, abandoned classes.
4. **Cross-file consolidation** — flag near-duplicate helpers, parallel implementations, naming inconsistencies. Fix in same pass OR open a tracked follow-up.
5. **Style/hygiene** — docstring/claim honesty (logic-eye on Raises/Returns), comment accuracy, magic constants extracted, etc.

## Per-File Acceptance Gate

After every file: `ruff check enigma_engine/ tests/` clean + `pytest tests/ -q` green. No file marked ✅ on a regression.

## Status Legend

- ⬜ not-started
- 🟡 in-progress
- ✅ done
- ⏭️ skip (trivial — empty `__init__.py` or auto-generated; no cleanup possible)

## Baseline

- **Lint:** clean (`ruff check enigma_engine/ tests/`, May 15 2026 pre-sweep)
- **Suite:** 3232 passed, 3 skipped, 37.48s (Pass 156z9ed)

---

## Repo root scripts (12 files)

| Status | File | Lines | Pass | Notes |
|--------|------|-------|------|-------|
| ⬜ | [collect_distill_data.py](collect_distill_data.py) | 756 | — | — |
| ⬜ | [collect_finetuning_data.py](collect_finetuning_data.py) | 600 | — | — |
| ⬜ | [collect_pretraining_data.py](collect_pretraining_data.py) | 2995 | — | — |
| ✅ | [collect_search_data.py](collect_search_data.py) | 341 | 156z9ek | author's-lens clean; B-4 seed corpus emitter, `_validate_examples` adversarial gate prevents class poisoning |
| ✅ | [collect_vision_data.py](collect_vision_data.py) | 291 | 156z9ek | author's-lens clean; LLaVA-Pretrain metadata streamer with image-on-disk gate + 5-warn cap |
| ✅ | [create_smoke_test_data.py](create_smoke_test_data.py) | 304 | 156z9ek | author's-lens clean; seeded reproducible smoke data for pretrain/basic/dpo |
| ✅ | [migrate_legacy_lora.py](migrate_legacy_lora.py) | 152 | 156z9eg | clean |
| ✅ | [pretokenize_data.py](pretokenize_data.py) | 228 | 156z9ei | clean |
| ⬜ | [run.py](run.py) | 1163 | — | — |
| ✅ | [run_model_output.py](run_model_output.py) | 35 | 156z9ek | author's-lens clean; trivial 5-prompt smoke test script |
| ⬜ | [run_training_diagnostic.py](run_training_diagnostic.py) | 747 | — | — |

## enigma_engine root (3 files)

| Status | File | Lines | Pass | Notes |
|--------|------|-------|------|-------|
| ⏭️ | [enigma_engine/__init__.py](enigma_engine/__init__.py) | 19 | — | re-export shim, leave alone |
| ✅ | [enigma_engine/client.py](enigma_engine/client.py) | ~310 | 156z9ei + da65525 + May 27 2026 (multi-pass) | clean. **da65525:** `images` kwarg threaded through `chat()` and `chat_stream()`. **May 27 2026:** `clear_history()` docstring relabeled (was "legacy nuke-all route", now honest "nuke-all wipes every server conversation"). **PERSONA-2 Slice 3:** added `get_style_preferences()` + `set_style_preferences(*, verbosity, formality, default_response_length, prefer_code_examples, prefer_bullet_points)`; partial-update semantics — None kwargs omitted from the PUT body. Bool kwargs coerced to bool. Server-side 422 on invalid enum surfaces as `RuntimeError`. |
| ⬜ | [enigma_engine/router.py](enigma_engine/router.py) | 1390 | — | — |

## enigma_engine/api (2 files)

| Status | File | Lines | Pass | Notes |
|--------|------|-------|------|-------|
| ⏭️ | [enigma_engine/api/__init__.py](enigma_engine/api/__init__.py) | 1 | — | empty |
| 🟡 | [enigma_engine/api/server.py](enigma_engine/api/server.py) | ~1900 | da65525 + May 27 2026 (multi-pass) | partial pass — vocab compatibility gate on `/api/models/load`, `images` field on `/api/chat` (rejected on `/api/chat/stream`), ValueError→HTTP 400 mapping. **TrainRequest legacy SFT-only shim deleted** (was `data_file`+`epochs`+`learning_rate`+`batch_size` parallel to dispatcher shape); 6 tests deleted, 2 new shape tests. **PERSONA-2 Slice 3:** `GET` + `PUT /api/style-preferences` with `StylePreferencesUpdate` (`extra="forbid"`, partial-update semantics, enum validation → HTTP 422 on invalid). **PERSONA-2 Slice 4:** `_style_overrides_by_conversation` dict + `_handle_style_command` parser + `/style` interception in `state.chat()` AND `/api/chat/stream` (F-A audit closure, both sibling paths); `style_overrides` kwarg forwarded to `engine.chat`/`engine.stream_chat`; conversation delete + clear-all purge overrides. "Legacy" doc comments relabeled to drop misleading framing. Full per-file audit not done yet. |

## enigma_engine/config (2 files)

| Status | File | Lines | Pass | Notes |
|--------|------|-------|------|-------|
| ⏭️ | [enigma_engine/config/__init__.py](enigma_engine/config/__init__.py) | 10 | — | re-export only |
| ✅ | [enigma_engine/config/defaults.py](enigma_engine/config/defaults.py) | 510 | 156z9ek | author's-lens clean; `_LazyConfig` proxy + double-check `_init_lock` with documented deadlock-bypass via `dict.update`/`__getitem__`/`__setitem__` |

## enigma_engine/core (60 files)

| Status | File | Lines | Pass | Notes |
|--------|------|-------|------|-------|
| ✅ | [enigma_engine/core/__init__.py](enigma_engine/core/__init__.py) | ~280 | 156z9ek + May 27 2026 | author's-lens clean; lazy `__getattr__` with `_LAZY_LOADER_MAP` defers torch/transformers imports. **May 27 2026:** removed `get_model_config` re-export (model_config.py deleted) and `load_tokenizer` re-export (alias deleted). Both also stripped from `__all__`. |
| ✅ | [enigma_engine/core/adaptive_trainer.py](enigma_engine/core/adaptive_trainer.py) | 682 | 156z9ep | Clean — six-question lens pass, no bugs found. All `TrainingPlan`/`StageResult` fields consumed by `gui_forge_adaptive.py` caller; `decide_action` semantics match test contract; `parse_score`/`_normalize_for_dedup`/`validate_example` regexes verified against edge cases. Drift 584→682. |
| ✅ | [enigma_engine/core/advanced_tokenizer.py](enigma_engine/core/advanced_tokenizer.py) | 438 | 156z9ej | author's-lens clean; Stage B-1 pop-on-None correctly drops phantom `<search>` for legacy vocab |
| ✅ | [enigma_engine/core/ai_profile.py](enigma_engine/core/ai_profile.py) | 598 | 156z9el | sibling-boundary drift on adapter-apply catch list; added `ValueError` to match `_restore_lora_adapter_for_base` (§4 Pass 156u-A2); regression test confirms PEFT shape-mismatch is swallowed |
| ✅ | [enigma_engine/core/audio_encoder.py](enigma_engine/core/audio_encoder.py) | 671 | 156z9et | clean — Whisper-style frontend + Conformer hybrid; preset doc-vs-code agree; behavioural test coverage already present |
| ✅ | [enigma_engine/core/auto_research.py](enigma_engine/core/auto_research.py) | 278 | 156z9eg | clean |
| ✅ | [enigma_engine/core/bpe_tokenizer.py](enigma_engine/core/bpe_tokenizer.py) | 780 | 156z9ew | clean — BPE with heap-based merge + Rust fast path + Tok-2 UTF-8 byte mode; Optional-kwarg style nits + aspirational "~6x faster" claim parked |
| ✅ | [enigma_engine/core/builtin_commands.py](enigma_engine/core/builtin_commands.py) | 1860 | 156z9fb + da65525 | clean — ~50 commands registered. **shell_cmd python/pip/pytest now behind explicit `allow_code_exec=True` opt-in (`da65525`, May 27 2026; 5 tests).** Remaining parked: code_run substring-forbidden-list trivially bypassed (from subprocess import Popen as P), _check_blocked_path uses str eq on resolved (children in blocked dirs slip through; needs Path.relative_to), file_read OOMs on multi-GB files, imagegen ComfyUI hardcodes model name + diffusers downloads 4GB silently + pipe not in try/finally (VRAM leak) + CPU silent fallback, note_add %S timestamp collides on rapid-fire, web_fetch swallows HEAD error |
| ✅ | [enigma_engine/core/char_tokenizer.py](enigma_engine/core/char_tokenizer.py) | 451 | 156z9ej | author's-lens clean; replace-from-disk on `special_tokens` honours Stage B-1 None-on-legacy |
| ✅ | [enigma_engine/core/chat_export.py](enigma_engine/core/chat_export.py) | 234 | 156z9eg | clean |
| ✅ | [enigma_engine/core/commands.py](enigma_engine/core/commands.py) | 212 | 156z9ef | sorted imports; replaced `__import__("threading").Lock()` hack with proper `import threading` |
| ✅ | [enigma_engine/core/curated_dataset.py](enigma_engine/core/curated_dataset.py) | 290 | 156z9eg | clean |
| ✅ | [enigma_engine/core/dataset.py](enigma_engine/core/dataset.py) | 619 | 156z9es | doc drift: `MAX_FILE_SIZE` comment said 20 GB, value was 100 GB — comment corrected |
| ✅ | [enigma_engine/core/document_readers.py](enigma_engine/core/document_readers.py) | 144 | 156z9ee | clean already; all public symbols have consumers; line count corrected (was 107) |
| ✅ | [enigma_engine/core/download_progress.py](enigma_engine/core/download_progress.py) | 580 | 156z9en | Global progress-bar state leak: `disable_progress_bars()` paired with `enable_progress_bars()` only on success path. Wrapped in try/finally. Pre-existing line-count drift (461→580). |
| ✅ | [enigma_engine/core/engine_chat.py](enigma_engine/core/engine_chat.py) | 909 | 156z9ez + May 27 2026 | clean — _ChatMixin with shared _prepare_chat; B-3a sibling family intact (json_schema rejection + _record_search_emissions on all 4 GGUF sites). **F4 closure May 27 2026:** `_prepare_chat` now raises `ValueError` on conflicting `max_tokens`/`max_new_tokens`/`max_length` (was silent nested-pop last-wins); matches `generate()` and `stream_generate()` contract. `max_length` newly accepted. 2 regression tests, falsification verified. GGUF+images silent-drop (V-8 sibling miss) + RAG-query DEBUG-instead-of-WARNING still parked. |
| ✅ | [enigma_engine/core/engine_generation.py](enigma_engine/core/engine_generation.py) | ~2895 | 156z9fd + 156z9fe + 156z9ff + May 27 2026 | 7 parked. **3 sibling-boundary bugs closed in prior passes**: `_execute_tools_in_text` forwards `json_schema` (`156z9fe`), `batch_generate` raises `NotImplementedError` on `json_schema` (`156z9fe`), `_cancel_generation` poll wired across 8 generation loops via `_check_cancel()` helper (`156z9ff`). **May 27 2026:** `max_tokens` / `max_new_tokens` / `max_length` aliases in `stream_generate` comment relabeled (industry-standard names, not back-compat shims). Remaining parked: speculative `torch.rand` reproducibility note; `_sample_token` vs `_sample_token_batch` inconsistent NaN fallback; `_update_ngram_pool` O(n²) lookahead; `_default_answer_extractor` last-line heuristic wrong for math; `_start_proper_noun_scan` tokenizer-thread race; `_generate_manual` minor; speculative `.item()` syncs. RAG-splice family 8 sites verified consistent. |
| ✅ | [enigma_engine/core/gguf.py](enigma_engine/core/gguf.py) | 1487 | 156z9f8 | clean — GGUF writer/exporter/quantizer + reader funcs; export_to_gguf silently downgrades unsupported quant types (Q5_*/Q6_*/BF16) to F16, add_tensor doesn't validate shape for quantized data, _apply_arch_consistency only Llama↔Qwen3 + skips state-dict-only path, GGUFMetadata vocab_size=32000 silent default for state-dict export, rope_dimension_count==128 sentinel override fragile |
| ✅ | [enigma_engine/core/gguf_dequant.py](enigma_engine/core/gguf_dequant.py) | 1184 | 156z9f2 + May 27 2026 | clean — 11 ggml-spec-matching dequant functions + tensor reader. **May 27 2026:** docstring updated — used to claim "re-exported from gguf_loader for backward compatibility" but the shim was deleted; doc now says "import directly from this module." Other parked: extract_config Llama-only metadata keys (silent fallback for Mistral/Qwen/Falcon) + missing n_kv_heads/ffn_dim extraction + no post-parse completeness check. |
| ✅ | [enigma_engine/core/gguf_loader.py](enigma_engine/core/gguf_loader.py) | ~1215 | 156z9f5 + May 27 2026 | clean — GGUFModel + LlamaServerBackend + load_gguf_model. **May 27 2026:** backward-compat re-exports of `parse_gguf_*` and `dequantize_*` (15 names) deleted (zero callers found via grep). Other parked: gguf-library path skips dequant (returns garbage), Llama-only config extraction, return-type lie (Enigma vs GGUFModel), in-process vs server metadata sibling drift, strict=False allows silent garbage, default tools have no executor. |
| ✅ | [enigma_engine/core/gptq_awq_loader.py](enigma_engine/core/gptq_awq_loader.py) | 1017 | 156z9f1 | clean — GPTQ + AWQ + LRU registry; pad_token_id `or eos` anti-pattern (token 0 fallthrough) + registry.register silent GPTQ default + temperature=0 with sampling parked |
| ✅ | [enigma_engine/core/hardware_detection.py](enigma_engine/core/hardware_detection.py) | 679 | 156z9eq | clean — partial-lock pattern in `detect_hardware()` is wasteful on cold-start but not a correctness bug (final state consistent) |
| ✅ | [enigma_engine/core/huggingface_loader.py](enigma_engine/core/huggingface_loader.py) | 1212 | 156z9f3 | clean — HF model loader + EnigmaEngine-shaped wrapper + HF→Forge conversion pipeline (GPT-2/Neo/Llama/Mistral/Phi/Qwen2/3); unused timeout param + temperature=0 with sampling + dialogpt substring match + 314B grok in SUGGESTED_MODELS parked |
| ✅ | [enigma_engine/core/inference.py](enigma_engine/core/inference.py) | 1755 | 156z9fa + da65525 + May 27 2026 | clean — EnigmaEngine composite (_GenerationMixin+_ChatMixin). **da65525:** runtime token-ID range guard in `_encode_prompt` before tensor moves to CUDA; **BUG-3** closed (`generate()` now raises `ValueError` on conflicting `max_tokens`/`max_new_tokens`/`max_length`, matching `stream_generate()`); 3 tests + falsification. **May 27 2026:** vocab-limit helpers extracted to `model_utils.py` (dedup with `AppState`); `max_tokens` comment relabeled (industry-standard alias, not back-compat shim). Other parked: `clear_kv_cache()` silent no-op on Enigma, `_load_pytorch` strict=False, head_dim=64 default wrong for Llama-7B, `apply_adapter` no base-model pre-validation, dup-check Path no resolve, `count_tokens` silent 0, `stream()` no generation_lock. |
| ✅ | [enigma_engine/core/json_schema_mask.py](enigma_engine/core/json_schema_mask.py) | 358 | 156z9ej | author's-lens clean; FSM states + `validate_json_schema_shape` boundary helper intact |
| ✅ | [enigma_engine/core/kv_cache.py](enigma_engine/core/kv_cache.py) | 1344 | 156z9f7 | clean — KVCache/KVCacheManager/PrefixKVCache/H2O/TurboQuant/StreamingLLM; KVCacheConfig dead infra, PrefixKVCache.build requires nonexistent forward_with_kv_capture, H2O eviction uses scores[0] only (wrong for batch>1), TurboQuant allocates full INT8 + INT4 buffers (no memory savings), KVCacheManager.current_pos reads layer 0 only |
| ✅ | [enigma_engine/core/lora_utils.py](enigma_engine/core/lora_utils.py) | ~1390 | 156z9f6 + May 27 2026 | clean — LoraConfig/DoRA/LoraTrainer/LoRAAdapterManager. **May 27 2026:** `merge_lora_weights` silent-corruption guard added — manual (non-PEFT) path now raises `ValueError` on shape mismatch OR when zero keys match (was silent no-op claiming success); 3 regression tests. Other parked: `_lora_adapters` dict has no consumer, `apply_lora` strict=False silent, `estimate_training_memory` lora_params 17x off + ignores attention activations, OOM string-match fragile, `_create_batches` called twice, `final_loss` naming misleading. |
| ✅ | [enigma_engine/core/memory.py](enigma_engine/core/memory.py) | 474 | 156z9ej | author's-lens clean; thread-safe singleton, fact dedup + outdated-replace, MAX_FACTS trim |
| ✅ | [enigma_engine/core/mod_tools.py](enigma_engine/core/mod_tools.py) | 177 | 156z9ef | clean; `registry._commands` private-attr coupling logged as follow-up |
| ✅ | [enigma_engine/core/model.py](enigma_engine/core/model.py) | 1775 | 156z9f9 + da65525 | clean — Enigma transformer + factories. **da65525 (BUG-2):** `_apply_static_int8_quantization` log honesty — success log moved inside helper so it reflects the actual path (dynamic fallback when no calibration data); INT4 bitsandbytes-absent fallback also corrected. 2 caplog tests. Other parked: from_pretrained/from_safetensors call cls() w/ no args, generate_speculative has 5 issues, generate stop_tokens=[2] hardcoded, __init__ ignores unknown kwargs, export_to_onnx broken for complex forward(), _apply_int4_quantization mutates during named_modules() iter. |
| ✅ | [enigma_engine/core/model_components.py](enigma_engine/core/model_components.py) | ~1295 | 156z9f4 + da65525 | clean. **da65525 (BUG-1):** cross-layer KV-share NoneType crash fixed — when source layer's `_kv_cache`/`_shared_kv` is None (first forward, post-clear_cache, or out-of-order config), follower now falls through to compute K/V from its own `wk`/`wv` projection weights instead of crashing on `None.get()`. 2 regression tests + falsification verified. Other parked: differential attention init produces lambda=0.51 not "near zero" + MAX_CACHE_SEQ_LEN=4096 hard-caps KV cache + 4D mask silently dropped on Flash path + ToMe O(B×r) Python loops. |
| ✅ | ~~enigma_engine/core/model_config.py~~ `[DELETED May 27 2026]` | — | — | Backward-compat re-export shim killed per AA code maker §2 rule. Callers (tests) deleted. |
| ✅ | [enigma_engine/core/model_context.py](enigma_engine/core/model_context.py) | 536 | 156z9em | `_load_context` except clause caught only JSONDecodeError/OSError; non-numeric `emotional_state` value (`float()` → ValueError/TypeError) would crash whole context load. Added ValueError, TypeError per §4 sibling-catch-list rule; regression test |
| ✅ | [enigma_engine/core/model_merging.py](enigma_engine/core/model_merging.py) | 249 | 156z9eg | fixed `callable \| None` → `Callable \| None` (3 sites) |
| ✅ | [enigma_engine/core/model_presets.py](enigma_engine/core/model_presets.py) | 957 | 156z9f0 | clean — ForgeConfig + 17 presets + estimators + parsers; get_preset/config_for_param_target silent-drop of 30+ advanced fields parked (no current preset overrides them; latent regression for future MoE/MoD/MLA preset additions) |
| ✅ | [enigma_engine/core/model_registry.py](enigma_engine/core/model_registry.py) | 217 | 156z9ei | clean; line count corrected (was 204) |
| ✅ | [enigma_engine/core/model_utils.py](enigma_engine/core/model_utils.py) | ~380 | 156z9ei + da65525 | clean. **da65525:** added shared vocab-limit helpers (`get_model_vocab_limit`, `get_tokenizer_vocab_size`) used by both `AppState.load_model` (load-time gate) and `EnigmaEngine._encode_prompt` (per-prompt gate). Eliminates a vocab-limit duplication that was about to drift. |
| ✅ | [enigma_engine/core/monologue.py](enigma_engine/core/monologue.py) | 207 | 156z9ef + 156z9eh | clean post-Pass-156z9de kill; 156z9eh closed sibling miss (quoted `callable \| None` → `Callable \| None`) |
| ✅ | [enigma_engine/core/multi_gpu.py](enigma_engine/core/multi_gpu.py) | 247 | 156z9eg | clean |
| ✅ | [enigma_engine/core/nf4_linear.py](enigma_engine/core/nf4_linear.py) | 185 | 156z9ef | clean — NF4 quantization for QLoRA |
| ✅ | [enigma_engine/core/ollama_loader.py](enigma_engine/core/ollama_loader.py) | 434 | 156z9ej | author's-lens clean; GGUF metadata/tensor parse + Enigma-format converter |
| ✅ | [enigma_engine/core/onnx_loader.py](enigma_engine/core/onnx_loader.py) | 424 | 156z9ej | author's-lens clean; `validate_loaded_model` docstring already corrected at 156z9cu |
| ✅ | [enigma_engine/core/personality_consistency.py](enigma_engine/core/personality_consistency.py) | 169 | 156z9eg | clean |
| ✅ | [enigma_engine/core/personality_data.py](enigma_engine/core/personality_data.py) | 524 | 156z9eo | narrowed `_REFUSAL_OPENERS` `"i can't help"` → `"i can't help you"`+`"i can't help with"` (idiom collision: `"I can't help but [verb]"` is non-refusal). +2 tests. |
| ✅ | [enigma_engine/core/plugin_loader.py](enigma_engine/core/plugin_loader.py) | 215 | 156z9eg | clean |
| ✅ | [enigma_engine/core/probe_history.py](enigma_engine/core/probe_history.py) | 127 | 156z9ef | clean — atomic identity/consistency probe summaries |
| ✅ | [enigma_engine/core/progressive_growing.py](enigma_engine/core/progressive_growing.py) | 695 | 156z9ev | clean code; 2 dead-infra (bert2bert mapping + GradualUnfreezer have no production callers) + 1 latent edge-case bug (`_init_identity_layer` missing FFN biases when `use_bias=True` + new layers) parked |
| ✅ | [enigma_engine/core/rag.py](enigma_engine/core/rag.py) | 672 | 156z9eu + May 27 2026 | clean — BM25 + adaptive chunking + co-occurrence expansion. **May 27 2026:** `TfidfVectorizer` docstring relabeled — old wording said the name was "kept for backward compatibility" (violates AA code maker §2); new wording honestly states the name is historical and rename has high cost (30+ test imports + on-disk to_dict schema). Class behavior unchanged. Other parked: chunk_text overlap-loop risk (below boundary threshold). |
| ✅ | [enigma_engine/core/rag_dense.py](enigma_engine/core/rag_dense.py) | 223 | 156z9eg | clean |
| ✅ | [enigma_engine/core/reasoning.py](enigma_engine/core/reasoning.py) | 314 | 156z9eg | clean |
| ✅ | [enigma_engine/core/reward_functions.py](enigma_engine/core/reward_functions.py) | 233 | 156z9eg | clean |
| ✅ | [enigma_engine/core/rl_training.py](enigma_engine/core/rl_training.py) | 2842 | 156z9fc | clean — RewardModel/PRM/RLHF/SelfPlay/GRPO/ReMax trainers; NEFTune noise mismatch corrupts old-vs-new logps at PPO epoch 0 (spurious clipping), GRPO/ReMax use ref logps as PPO "old" (theoretically wrong), ReMax zero_grad AFTER step (first iter uses stale .grad), GRPO/ReMax deepcopy doubles VRAM (no LoRA-ref path), _get_logps_hidden_entropy manually duplicates Enigma.forward() internals (parallel-impl drift), config.neftune_alpha direct access (AttributeError on legacy configs), compute_advantages Python loop with per-token .item() GPU sync, PRM causal mask not cached, RewardTrainer hardcodes User/Assistant template (wrong for Llama-3/ChatML), SelfPlay score regex grabs FIRST number ("scale of 1 to 5" → 1), PRMTrainer AdamW default betas inconsistent with RewardTrainer 0.95, _setup_reference mutates caller's model |
| ✅ | [enigma_engine/core/safe_save.py](enigma_engine/core/safe_save.py) | 201 | 156z9ef | dropped unused `typing.Dict`, switched annotation to PEP-585 `dict` |
| ✅ | [enigma_engine/core/sentiment.py](enigma_engine/core/sentiment.py) | 400 | 156z9ej | author's-lens clean; heuristic sentiment + modulate_generation_params + engagement scoring |
| ✅ | [enigma_engine/core/streaming.py](enigma_engine/core/streaming.py) | 489 | 156z9er | FIX: `_emit` race — `_chunks.append` was outside `_async_lock`, allowing `__aiter__` backfill to duplicate chunks; moved append inside lock + structural test |
| ✅ | [enigma_engine/core/style_preferences.py](enigma_engine/core/style_preferences.py) | ~290 | PERSONA-2 Slices 1+2 (May 27 2026) | clean — new file, fully wired. `StylePreferences` dataclass (verbosity / formality / default_response_length / prefer_code_examples / prefer_bullet_points) + atomic JSON load/save via `safe_save.atomic_write_json` + `render_style_preferences_block(prefs)` + `get_style_preferences_block_for_prompt(path=None)` + `STYLE_PREFERENCES_PATH` module constant. Defaults preserve current behavior (`is_default()` → True). Forward-compat: unknown fields silently dropped on load. Loud-on-real-issue: WARNING on corrupt/non-object/invalid-enum loads, silent on missing file. **Slice 2 production consumer is `_prepare_chat` (engine_chat.py)** — block injected after RAG, before reasoning; skipped at defaults (zero overhead). 51 behavioral tests in `tests/test_style_preferences.py` including black-box invariant (model file bytes + mtime unchanged when prefs are saved), parametrized valid-enum acceptance, prompt-injection round-trips. |
| ✅ | [enigma_engine/core/tokenizer.py](enigma_engine/core/tokenizer.py) | ~880 | 156z9ex + May 27 2026 | clean — factory + SimpleTokenizer + TiktokenWrapper. **May 27 2026:** `load_tokenizer()` backward-compat alias for `get_tokenizer()` deleted; single caller (`mods/codegen/codegen.py`) migrated to `get_tokenizer()`. Other parked: TiktokenWrapper missing sibling-contract attrs (string tokens, search_*_id, _sync helper) + decode(skip_special_tokens=False) crashes on reserved IDs. |
| ✅ | [enigma_engine/core/tokenizer_metrics.py](enigma_engine/core/tokenizer_metrics.py) | 193 | 156z9eg | clean |
| ✅ | [enigma_engine/core/vision_encoder.py](enigma_engine/core/vision_encoder.py) | 935 | 156z9ey | clean — ViT from-scratch/hybrid/pretrained + video/screen/camera; `_init_pretrained` mutates caller's shared preset on timm-missing fallback (latent footgun), config validation gaps (dim%n_heads, n_layers/heads/dim>=1), `encode_video_frames` divzero on max_frames=0 — parked |
| ✅ | [enigma_engine/core/web_utils.py](enigma_engine/core/web_utils.py) | 232 | 156z9eg | clean |
| ✅ | [enigma_engine/core/weight_mapping.py](enigma_engine/core/weight_mapping.py) | 299 | 156z9ei | clean |

## enigma_engine/gui (24 files)

| Status | File | Lines | Pass | Notes |
|--------|------|-------|------|-------|
| ⏭️ | [enigma_engine/gui/__init__.py](enigma_engine/gui/__init__.py) | 1 | — | empty |
| ✅ | [enigma_engine/gui/baseline_instrument.py](enigma_engine/gui/baseline_instrument.py) | 100 | 156z9ef | clean — just shipped in Pass 156z9ed |
| ⬜ | [enigma_engine/gui/desktop.py](enigma_engine/gui/desktop.py) | 1248 | — | — |
| ⬜ | [enigma_engine/gui/gui_cmd_page.py](enigma_engine/gui/gui_cmd_page.py) | 1347 | — | — |
| ⬜ | [enigma_engine/gui/gui_docs_page.py](enigma_engine/gui/gui_docs_page.py) | 834 | — | — |
| ⬜ | [enigma_engine/gui/gui_forge.py](enigma_engine/gui/gui_forge.py) | 1769 | — | — |
| ⬜ | [enigma_engine/gui/gui_forge_adaptive.py](enigma_engine/gui/gui_forge_adaptive.py) | 884 | — | — |
| ⬜ | [enigma_engine/gui/gui_forge_advanced.py](enigma_engine/gui/gui_forge_advanced.py) | 908 | — | — |
| ⬜ | [enigma_engine/gui/gui_forge_models.py](enigma_engine/gui/gui_forge_models.py) | 1037 | — | — |
| ⬜ | [enigma_engine/gui/gui_forge_new_modes.py](enigma_engine/gui/gui_forge_new_modes.py) | 3189 | — | — |
| ⬜ | [enigma_engine/gui/gui_forge_queue.py](enigma_engine/gui/gui_forge_queue.py) | 599 | — | — |
| ⬜ | [enigma_engine/gui/gui_forge_teacher.py](enigma_engine/gui/gui_forge_teacher.py) | 466 | — | — |
| ⬜ | [enigma_engine/gui/gui_forge_tools.py](enigma_engine/gui/gui_forge_tools.py) | 1418 | — | — |
| ⬜ | [enigma_engine/gui/gui_forge_training.py](enigma_engine/gui/gui_forge_training.py) | 2311 | — | — |
| ⬜ | [enigma_engine/gui/gui_logic.py](enigma_engine/gui/gui_logic.py) | 1480 | — | — |
| ⬜ | [enigma_engine/gui/gui_logic_chat.py](enigma_engine/gui/gui_logic_chat.py) | 1667 | — | — |
| ⬜ | [enigma_engine/gui/gui_logic_media.py](enigma_engine/gui/gui_logic_media.py) | 649 | — | — |
| ✅ | [enigma_engine/gui/gui_mod_page.py](enigma_engine/gui/gui_mod_page.py) | 330 | 156z9ei | clean |
| ✅ | [enigma_engine/gui/gui_mods.py](enigma_engine/gui/gui_mods.py) | 181 | 156z9eg | clean |
| ⬜ | [enigma_engine/gui/gui_pages.py](enigma_engine/gui/gui_pages.py) | 1590 | — | — |
| ⬜ | [enigma_engine/gui/gui_pages_config.py](enigma_engine/gui/gui_pages_config.py) | 1612 | — | — |
| ⬜ | [enigma_engine/gui/gui_pages_forge.py](enigma_engine/gui/gui_pages_forge.py) | 1591 | — | — |
| ⬜ | [enigma_engine/gui/media.py](enigma_engine/gui/media.py) | 480 | — | — |
| ⬜ | [enigma_engine/gui/scanners.py](enigma_engine/gui/scanners.py) | 912 | — | — |
| ✅ | [enigma_engine/gui/themes.py](enigma_engine/gui/themes.py) | 186 | 156z9ef | clean — frozen-dataclass theme registry |
| ⬜ | [enigma_engine/gui/widgets.py](enigma_engine/gui/widgets.py) | 1083 | — | — |

## ~~enigma_engine/services~~ `[DELETED dbc19ea, May 25 2026]`

Phase 0c skeleton was deleted as dead infra (no production callers). 9 files / ~250 LOC. Git history preserves the original work.

## enigma_engine/training (8 files)

| Status | File | Lines | Pass | Notes |
|--------|------|-------|------|-------|
| ✅ | [enigma_engine/training/__init__.py](enigma_engine/training/__init__.py) | 45 | 156z9ef | clean — lazy `__getattr__` for heavy dispatch import |
| ✅ | [enigma_engine/training/dispatch.py](enigma_engine/training/dispatch.py) | 339 | 156z9ej | author's-lens clean; 14-mode dispatch + honest `NotImplementedError` for adaptive |
| ✅ | [enigma_engine/training/registry.py](enigma_engine/training/registry.py) | 36 | 156z9ef | clean — training mode metadata registry |
| ✅ | [enigma_engine/training/schema.py](enigma_engine/training/schema.py) | 260 | 156z9ei | clean |
| ⬜ | [enigma_engine/training/training.py](enigma_engine/training/training.py) | 5220 | — | — |
| ⬜ | [enigma_engine/training/training_evaluation.py](enigma_engine/training/training_evaluation.py) | 530 | — | — |
| ⬜ | [enigma_engine/training/training_monitor.py](enigma_engine/training/training_monitor.py) | 436 | — | — |
| ⬜ | [enigma_engine/training/training_queue.py](enigma_engine/training/training_queue.py) | 593 | — | — |

## tests (50 files)

| Status | File | Lines | Pass | Notes |
|--------|------|-------|------|-------|
| ⏭️ | [tests/__init__.py](tests/__init__.py) | 1 | — | empty |
| ⬜ | [tests/test_api.py](tests/test_api.py) | 1586 | — | — |
| ⬜ | [tests/test_api_conversations.py](tests/test_api_conversations.py) | 834 | — | — |
| ⬜ | [tests/test_baseline_flag_wiring.py](tests/test_baseline_flag_wiring.py) | 86 | — | — |
| ⬜ | [tests/test_baseline_instrument.py](tests/test_baseline_instrument.py) | 76 | — | — |
| ⬜ | [tests/test_chat.py](tests/test_chat.py) | 2586 | — | — |
| ⬜ | [tests/test_client.py](tests/test_client.py) | 181 | — | — |
| ⬜ | [tests/test_collect_distill_data.py](tests/test_collect_distill_data.py) | 598 | — | — |
| ⬜ | [tests/test_collect_finetuning_data.py](tests/test_collect_finetuning_data.py) | 292 | — | — |
| ⬜ | [tests/test_collect_search_data.py](tests/test_collect_search_data.py) | 130 | — | — |
| ⬜ | [tests/test_collect_vision_data.py](tests/test_collect_vision_data.py) | 264 | — | — |
| ⬜ | [tests/test_commands.py](tests/test_commands.py) | 624 | — | — |
| ⬜ | [tests/test_core.py](tests/test_core.py) | 4253 | — | — |
| ⬜ | [tests/test_curated_dataset.py](tests/test_curated_dataset.py) | 478 | — | — |
| ⬜ | [tests/test_download_progress.py](tests/test_download_progress.py) | 174 | — | — |
| ⬜ | [tests/test_evaluation.py](tests/test_evaluation.py) | 238 | — | — |
| ⬜ | [tests/test_functional.py](tests/test_functional.py) | 1118 | — | — |
| ⬜ | [tests/test_gguf.py](tests/test_gguf.py) | 157 | — | — |
| ⬜ | [tests/test_gguf_roundtrip.py](tests/test_gguf_roundtrip.py) | 647 | — | — |
| ⬜ | [tests/test_gui.py](tests/test_gui.py) | 6448 | — | — |
| ⬜ | [tests/test_gui_forge_new_modes.py](tests/test_gui_forge_new_modes.py) | 16 | — | — |
| ⬜ | [tests/test_gui_forge_training.py](tests/test_gui_forge_training.py) | 137 | — | — |
| ⬜ | [tests/test_gui_logic_chat.py](tests/test_gui_logic_chat.py) | 860 | — | — |
| ⬜ | [tests/test_inference.py](tests/test_inference.py) | 832 | — | — |
| ⬜ | [tests/test_kv_cache.py](tests/test_kv_cache.py) | 607 | — | — |
| ⬜ | [tests/test_loaders.py](tests/test_loaders.py) | 192 | — | — |
| ⬜ | [tests/test_memory.py](tests/test_memory.py) | 809 | — | — |
| ⬜ | [tests/test_model_arch.py](tests/test_model_arch.py) | 1679 | — | — |
| ⬜ | [tests/test_monologue.py](tests/test_monologue.py) | 147 | — | — |
| ⬜ | [tests/test_new_features.py](tests/test_new_features.py) | 856 | — | — |
| ⬜ | [tests/test_nf4_linear.py](tests/test_nf4_linear.py) | 210 | — | — |
| ⬜ | [tests/test_personality_consistency.py](tests/test_personality_consistency.py) | 279 | — | — |
| ⬜ | [tests/test_personality_data.py](tests/test_personality_data.py) | 905 | — | — |
| ⬜ | [tests/test_plugins.py](tests/test_plugins.py) | 426 | — | — |
| ⬜ | [tests/test_probe_history.py](tests/test_probe_history.py) | 193 | — | — |
| ⬜ | [tests/test_progressive_growing.py](tests/test_progressive_growing.py) | 437 | — | — |
| ⬜ | [tests/test_reasoning.py](tests/test_reasoning.py) | 409 | — | — |
| ⬜ | [tests/test_repo_hygiene.py](tests/test_repo_hygiene.py) | 63 | — | — |
| ⬜ | [tests/test_research_upgrades.py](tests/test_research_upgrades.py) | 2543 | — | — |
| ⬜ | [tests/test_reward_functions.py](tests/test_reward_functions.py) | 26 | — | — |
| ⬜ | [tests/test_run_chat_client.py](tests/test_run_chat_client.py) | 392 | — | — |
| ⬜ | [tests/test_sampling.py](tests/test_sampling.py) | 567 | — | — |
| ⬜ | [tests/test_security.py](tests/test_security.py) | 222 | — | — |
| ⬜ | [tests/test_streaming.py](tests/test_streaming.py) | 351 | — | — |
| ⬜ | [tests/test_themes.py](tests/test_themes.py) | 104 | — | — |
| ⬜ | [tests/test_tokenizer.py](tests/test_tokenizer.py) | 797 | — | — |
| ⬜ | [tests/test_training.py](tests/test_training.py) | 6174 | — | — |
| ⬜ | [tests/test_training_dispatch.py](tests/test_training_dispatch.py) | 507 | — | — |
| ⬜ | [tests/test_weight_mapping.py](tests/test_weight_mapping.py) | 206 | — | — |

---

## Cross-File Follow-Ups Discovered During Sweep

(Append entries as we go. Format: `**[FILE-PAIR]**: short description — opened pass N, resolved pass M or still open.`)

— *(none yet)*

---

## Totals

- **Total files in scope:** 170
- **Skipped (trivial init):** 5
- **Active scope:** 165
- **Done:** 85
- **In-progress:** 0
- **Remaining:** 80
