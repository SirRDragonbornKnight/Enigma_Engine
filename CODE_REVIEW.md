# Code Review Tracker

**Started:** March 24, 2026
**Previous pass:** March 25, 2026 (Pass 9) — Cleanup + test fix

**Current pass:** March 26, 2026 (Pass 10) — Full re-review
**Status:** 7 findings (#70-#76), 6 fixed, 1 false positive (#76 — Dict used in annotations). Lint clean, 2396 passed, 0 failed, 3 skipped.

**⚠ GUI growth since review:** All 23 GUI files grew +7,481 lines total since Pass 10. Largest: gui_pages.py (+850), gui_cmd_page.py (+811), scanners.py (+819), gui_forge_new_modes.py (+740), gui_forge_models.py (+725). Pass 11 should re-review all GUI files.

---

## How This Works

- Go through each file, read it fully, note anything worth changing
- Mark each file: DONE, SKIP (no logic), or the date reviewed
- Actionable items get added to SUGGESTIONS.md with item numbers
- Pick up where we left off next session

---

## config/

| File | Status | Notes |
|------|--------|-------|
| config/defaults.py | DONE 3/25p8 | Clean |
| config/__init__.py | DONE 3/25p8 | Clean |

## core/ — Model & Architecture

| File | Status | Notes |
|------|--------|-------|
| core/model_presets.py | DONE 3/25p8 | Clean |
| core/model_config.py | DONE 3/25p8 | Clean — thin re-export |
| core/model_components.py | DONE 3/25p8 | Clean |
| core/model.py | DONE 3/26p10 | #73 FIXED: Causal mask now created on model device |
| core/kv_cache.py | DONE 3/25p8 | Clean |
| core/model_utils.py | DONE 3/25p8 | Clean |
| core/model_registry.py | DONE 3/25p8 | Clean — `_load_registry` read outside lock is benign (only from __init__) |
| core/model_context.py | DONE 3/25p8 | Clean — history lock is caller coordination (design choice) |

## core/ — Tokenizers

| File | Status | Notes |
|------|--------|-------|
| core/tokenizer.py | DONE 3/25p8 | Clean |
| core/bpe_tokenizer.py | DONE 3/25p8 | Clean |
| core/char_tokenizer.py | DONE 3/25p8 | #64: add_word() now respects max_vocab_size cap |
| core/advanced_tokenizer.py | DONE 3/25p8 | Clean |

## core/ — Inference & Generation

| File | Status | Notes |
|------|--------|-------|
| core/inference.py | DONE 3/25p8 | Clean — PATH env mutation in GGUF load (informational) |
| core/engine_generation.py | DONE 3/25p8 | Clean |
| core/engine_chat.py | DONE 3/25p8 | Added `_cap_history_summary()` — caps unbounded summary growth at 4096 chars (#68) |
| core/streaming.py | DONE 3/25p8 | #48 fixed. #50 dead `get_token_streamer`/`_streamer` removed |
| core/reasoning.py | DONE 3/25p8 | #60 fixed — extract_all_reasoning double-counting |

## core/ — Training

| File | Status | Notes |
|------|--------|-------|
| core/training.py | DONE 3/25p8 | Clean |
| core/rl_training.py | DONE 3/26p10 | #70 FIXED: ref model GPU transfer wrapped in try/finally |
| core/training_evaluation.py | DONE 3/25p8 | Clean |
| core/training_monitor.py | DONE 3/25p8 | Clean |
| core/training_queue.py | DONE 3/26p10 | #72 FIXED: Save error logged at WARNING |
| core/dataset.py | DONE 3/25p8 | Clean |
| core/adaptive_trainer.py | DONE 3/25p8 | Clean |
| core/curated_dataset.py | DONE 3/25p8 | Clean |
| core/lora_utils.py | DONE 3/25p8 | Clean |
| core/progressive_growing.py | DONE 3/25p8 | Clean |

## core/ — External Loaders

| File | Status | Notes |
|------|--------|-------|
| core/gguf.py | DONE 3/26p10 | #74-75 FIXED: Array/string allocation bounds added |
| core/gguf_dequant.py | DONE 3/25p8 | Clean |
| core/gguf_loader.py | DONE 3/25p8 | #52 stderr pipe closed after success + in stop() |
| core/gptq_awq_loader.py | DONE 3/25p8 | Clean |
| core/huggingface_loader.py | DONE 3/25p8 | Clean — #49 fixed |
| core/ollama_loader.py | DONE 3/25p8 | #61 fixed — dead OllamaBlob dataclass removed |
| core/onnx_loader.py | DONE 3/25p8 | Clean |
| core/weight_mapping.py | DONE 3/25p8 | Clean |

## core/ — Utilities

| File | Status | Notes |
|------|--------|-------|
| core/safe_save.py | DONE 3/25p8 | Clean |
| core/memory.py | DONE 3/25p8 | Clean |
| core/commands.py | DONE 3/25p8 | Clean |
| core/builtin_commands.py | DONE 3/26p10 | #76: False positive — Dict used in 20+ annotation strings |
| core/sentiment.py | DONE 3/25p8 | Clean |
| core/monologue.py | DONE 3/25p8 | Clean |
| core/ai_profile.py | DONE 3/25p8 | Clean |
| core/chat_export.py | DONE 3/25p8 | PDF write non-atomic (fpdf2 API limitation) |
| core/document_readers.py | DONE 3/25p8 | Eager heavy lib imports (fitz, docx) — informational |
| core/download_progress.py | DONE 3/25p8 | Clean |
| core/hardware_detection.py | DONE 3/25p8 | Clean (minor cache race, harmless) |
| core/web_utils.py | DONE 3/25p8 | Clean |
| core/rag.py | DONE 3/25p8 | Clean |
| core/auto_research.py | DONE 3/25p8 | Clean |
| core/multi_gpu.py | DONE 3/25p8 | Clean |
| core/plugin_loader.py | DONE 3/25p8 | Clean |
| core/mod_tools.py | DONE 3/25p8 | Clean |
| core/vision_encoder.py | DONE 3/26p10 | #71 FIXED: VideoCapture wrapped in try/finally |
| core/audio_encoder.py | DONE 3/25p8 | Clean |

## gui/ — ⚠ All files grew significantly since review (re-review in Pass 11)

| File | Status | Notes |
|------|--------|-------|
| gui/widgets.py | DONE 3/25p8 (814→1121) | #55 Tooltip child-boundary fix, #56 SelectableLabel height auto-compute |
| gui/themes.py | DONE 3/25p8 (143→161) | Clean |
| gui/desktop.py | DONE 3/25p8 (1080→1553) | #65: Escape binding leak fixed — _dismiss now calls _bind_escape_stop |
| gui/gui_pages.py | DONE 3/25p8 (1296→2146) | #54 CTkEntry transparent fix, #57 emotional state label width fix |
| gui/gui_pages_config.py | DONE 3/25p8 (1039→1367) | #58 "Monologue Mode" label widened 140→160px |
| gui/gui_pages_forge.py | DONE 3/25p8 (1186→1602) | #59 "Val split" label widened 80→90px |
| gui/gui_logic.py | DONE 3/25p8 (985→1003) | Clean |
| gui/gui_logic_chat.py | DONE 3/25p8 (1277→1385) | Clean |
| gui/gui_logic_media.py | DONE 3/25p8 (617→763) | Clean |
| gui/gui_forge.py | DONE 3/25p8 (1224→1259) | Clean |
| gui/gui_forge_training.py | DONE 3/25p8 (954→1030) | Clean |
| gui/gui_forge_advanced.py | DONE 3/25p8 (1033→1350) | Clean |
| gui/gui_forge_adaptive.py | DONE 3/25p8 (819→1399) | Clean |
| gui/gui_forge_new_modes.py | DONE 3/25p8 (1011→1751) | Clean |
| gui/gui_forge_tools.py | DONE 3/25p8 (1268→1699) | Clean |
| gui/gui_forge_models.py | DONE 3/25p8 (773→1498) | #53 ModelConfig→ForgeConfig import fix (3 sites) |
| gui/gui_forge_queue.py | DONE 3/25p8 (405→599) | Clean |
| gui/gui_mods.py | DONE 3/25p8 (174→251) | Clean — stderr handled on both crash + success paths |
| gui/gui_mod_page.py | DONE 3/25p8 (300→809) | Clean (private CTk — known) |
| gui/gui_cmd_page.py | DONE 3/25p8 (1487→2298) | Added `_cmd_is_non_latin()` — routes non-Latin input to AI in ENGINE mode (#67) |
| gui/gui_docs_page.py | DONE 3/25p8 (803→1042) | #66: Auto-save unified with manual save (both use .strip() + newline) |
| gui/media.py | DONE 3/25p8 (446→830) | Clean |
| gui/scanners.py | DONE 3/25p8 (630→1449) | Clean |

## api/

| File | Status | Notes |
|------|--------|-------|
| api/server.py | DONE 3/25p8 | Clean — #46 fixed |
| api/__init__.py | DONE 3/25p8 | Clean |

## Top-level

| File | Status | Notes |
|------|--------|-------|
| router.py | DONE 3/25p8 | Clean |
| run.py | DONE 3/25p8 | Clean |
| __init__.py | DONE 3/25p8 | Clean |

## Tests

| File | Status | Notes |
|------|--------|-------|
| tests/test_core.py | DONE 3/25p8 | Clean |
| tests/test_gui.py | DONE 3/25p9 | Font offset tests fixed — now reset state before asserting |
| tests/test_api.py | DONE 3/25p8 | Clean |
| tests/test_functional.py | DONE 3/25p8 | Clean |
| tests/test_new_features.py | DONE 3/25p8 | Added answer-content assertions for #60 |
| tests/test_monologue.py | DONE 3/25p8 | Clean |
| tests/test_reasoning.py | DONE 3/25p8 | Clean |
| tests/test_evaluation.py | DONE 3/25p8 | Clean |
| tests/test_benchmark.py | DONE 3/25p8 | Clean |
| tests/test_progressive_growing.py | DONE 3/25p8 | Clean |

