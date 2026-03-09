# AA Code Maker - AI Instructions

Read this file before writing code. Update after each session.

---

## 1. RULES (Non-Negotiable)

1. Read this file and SUGGESTIONS.md before doing anything
2. Read files before editing — verify imports, attributes, and APIs exist
3. Write tests first, then build features
4. Check if it already exists before building anything new
5. No cloud, no external dependencies unless necessary
6. Challenge bad ideas — do not be a yes man
7. We do not need multiple ways to do the same thing
8. Do not leave any issues, warnings, or broken code behind
9. Keep it simple — do not over-engineer
10. Make sure it is realistic and practical

---

## 2. DO NOT

- Do not delete or rewrite existing code without being asked
- Do not introduce new dependencies without approval
- Do not change file structure or architecture unprompted
- Do not add symbols or special characters in names
- Do not add untested code
- Do not add duplicate code — check if it exists first
- Do not add invisible/hidden characters
- Do not leave debug prints (`print`, `console.log`)
- Do not add redundant comments that just repeat the code
- Do not hardcode file paths
- Do not hardcode code for the AI
- Do not add unused imports
- Do not make random formatting changes
- Do not use sliders — use numeric input only
- Do not add backward-compatible shims
- Do not add windows/popups unless specified
- Do not add global state changes without a thread lock

---

## 3. KEY LESSONS

### What Went Wrong (Never Do Again)
- `torch.save()` corrupted files on crash — always use `atomic_torch_save()` from `safe_save.py`
- Bare `except: pass` hid bugs for weeks — always log at minimum with `logger.debug()`
- `.view()` crashed on AMP/MoE tensors — use `.reshape()` everywhere
- Hardcoded `ignore_index=0` broke tiktoken — always read `pad_token_id` from tokenizer
- Hardcoded `max_length=512` broke long-context models — read `model.config.max_seq_len`
- Importing gui modules from core created circular deps — core must never import gui
- Popups broke the UX flow — use inline confirmation bars, never popups or `askyesnocancel`
- `shell=True` caused injection bugs — always `shell=False` + `shlex.split()`
- `PIPE` without readers deadlocked on Windows — use `subprocess.DEVNULL` when not reading output
- Lambda captured exception var that Python cleared after except — capture to `str(exc)` first
- Forgot `encoding='utf-8'` on Windows — defaults to cp1252, broke everything
- `except BaseException` caught `SystemExit` and prevented clean shutdown — use `except Exception`
- `torch.load` on large files just for param counting was wasteful — use zipfile peek + file-size heuristic
- `from x import Y` inside a function where `Y` was used before the import — Python treats it as local, crashes
- `0.0.0.0` exposed the server to network — default to `127.0.0.1`
- Loop vars in lambdas captured by reference, not value — bind with default args `lambda p, _gen=gen: f(p, _gen)`
- GIF animations held frame refs forever — must cap and release old ones
- `CTkSegmentedButton` does not support `.bind()` — no tooltips or right-click on it
- Used `score / penalty` for both positive and negative scores — must use `torch.where(scores > 0, scores / penalty, scores * penalty)`
- `exp(20)` ≈ 485M overflowed perplexity cap — cap with `exp(min(loss, 20.0))` capped at 1e6
- TrainingConfig dicts loaded as model config broke everything — check for `"epochs" in cfg_dict` to reject
- CNNStem spatial reduction is /8 not /16 — trace stride-2 conv × pool1 × pool2, verify with forward pass
- Non-atomic JSON/text writes across 8 modules — always use atomic write (write `.tmp` → `os.replace()`) for JSON/text too, not just torch saves
- Blanket `except Exception` in engine_chat.py hid GGUF errors for silent degradation — catch specific exceptions, never mask real failures
- `stream_chat()` missing reasoning support while `chat()` has it — when parallel code paths exist, keep them in sync or extract shared logic
- `copy.deepcopy(model)` for RLHF reference doubled VRAM — use LoRA-based reference (disable_adapter_layers for ref logprobs) or CPU offload fallback
- Dead code (unused fields/methods) left in dataclasses misled readers — remove dead code instead of leaving aspirational stubs
- Concurrent dataset access without Lock corrupted entries — always add threading.Lock when GUI + background threads share a mutable object
- KV cache `get()` returned views — caller mutations silently corrupted the cache. Non-quantized path must `.clone()` like the quantized path does
- Top-level `import torch` in run.py loaded 500MB for `--help` — keep heavy imports inside the functions that need them
- Hardcoded port default in argparse conflicted with CONFIG — use `default=None` and resolve from CONFIG at runtime, CLI overrides
- Adding middleware to a FastAPI app after `TestClient` started it raises RuntimeError — create a fresh app for middleware tests
- `from_model()` skipped `__init__` — use `_init_common()` so both paths share attributes
- `len(text) // 4` token estimation was wrong for CJK/code — require real tokenizer, never guess
- `except Exception` in tokenizer chain masked misconfigs as "try next backend" — catch ImportError only
- Dense TF-IDF matrix used 300MB for data that was 95% zeros — use scipy sparse
- `run_command` in default AI tools = prompt injection to shell — never give AI system commands by default
- CORS `*` on localhost API let any website talk to local AI — no CORS middleware unless explicitly enabled
- Window resize lag with nested grids — debounce Configure events (150ms timer) to batch geometry updates instead of recalculating per-pixel
- CTkTextbox doesn't enable undo by default — must configure underlying `_textbox` widget with `undo=True`
- CMD status strip polled heavy APIs every 5s even when page hidden — check `_current_page` before scheduling next update to avoid wasting CPU

### What Worked Well (Keep Doing)
- Mixin pattern for splitting large classes — keeps files under 800 lines, easy to maintain
- Widget factories (`themed_entry`, `themed_dropdown`) — eliminated boilerplate and bugs
- Inline confirmation bars instead of popups — better UX, no focus-stealing
- `_model_op_busy()` guard — prevented every concurrent model op bug
- Atomic save (write `.tmp` → `os.replace()`) — survived crashes and power loss
- Deferred boot scanning — GUI feels instant, scanning happens in background
- `validate_training_data()` before training — catches bad data early
- `_ensure_*_imports()` deferred imports — saved 540 MB idle RAM
- `_is_generating` flag in both button handler AND Enter key — stopped double-sends
- `dict.fromkeys()` dedup before batching — removed duplicates while preserving order
- `getattr()` guard before accessing optional widget attrs — no more AttributeError crashes
- Background threads + `self.after(0, callback)` — UI never freezes
- `python check.py` before every commit — catches lint + test failures instantly
- Per-model context auto-save/load — each model keeps own history and prompt
- `strip_incomplete_think()` — cleanly handles truncated `<think>` blocks
- Structural guard test on `EnigmaGUI._on_close()` to forbid `except Exception: pass` — keeps shutdown failures observable and prevents silent regressions
- Keep comments aligned with actual behavior (especially in security paths) — avoids false confidence during audits and review
- Full codebase audit with verification passes — spot-checking all 30 claims against real code caught 3 false positives before they became bad fixes
- Standalone options over sequential steps — each fix independently implementable, no hidden dependencies between choices
- Centralizing `safe_load_weights()` as single entry point — replaced 16 scattered `torch.load` calls, added guard test to prevent regression
- Guard tests that scan source code — `test_no_direct_torch_load_in_codebase` catches regressions automatically
- VRAM-tier batch sizing over formula estimation — simple lookup table that's never wrong, vs memory formulas that always are
- LoRA-based RLHF reference — frozen base weights *are* the reference, zero extra VRAM, with CPU-offload fallback for non-PEFT setups
- Extracting `_train_one_batch()` from training loop — OOM recovery becomes a simple try/retry wrapper around it
- `pytest.mark.structural` for source-inspection tests — keeps them runnable but deselectable with `-m "not structural"`
- Ruff-based F401 check replacing string-matching dead import tests — one reliable test replaces 8 fragile ones
- Fresh FastAPI app per test class — avoids middleware conflicts from shared module-level app
- LRU cache with thread-safe Lock for web research — avoids duplicate searches and DuckDuckGo rate limits
- Cosine similarity dedup for video frames — drops redundant tokens from static scenes with zero model changes
- `default=None` + runtime CONFIG resolution — single source of truth for ports, CLI override still works
- `_prepare_chat()` eliminated chat/stream_chat divergence — single source of truth for prompt building
- 3-layer plugin security (allowlist → register scan → AST scan) — each layer is cheap, catches different things
- `recommend_training_batch_size()` from VRAM tiers — safe defaults, user can override
- OOM → auto gradient checkpointing → retry once → clean abort — recovers the easy case, fails clearly on the hard case
- Resize debounce on Configure events — prevents geometry recalc lag in complex nested grid layouts
- Enabling undo on CTkTextbox via `_textbox.configure(undo=True)` — clean single-line fix for missing editor feature
- Page-aware status updates — check `_current_page` before scheduling resource-heavy updates (CMD status strip: only update when visible)

---

## 4. TEST LOOP

```
Write Test → Build Feature → Run Test → Pass? → Yes: Refactor & Merge Tests → No: Fix & Repeat
```

Command: `python -m pytest tests/ -v`

### Testing Rules

- One test file per module (e.g., `test_training.py` for `training.py`)
- Merge related small test files — no single-test files
- Tests must be independent — no shared mutable state between tests
- Mock heavy dependencies (torch, file I/O) — tests should run in ~11s total
- Delete test files that test removed features

---

## 5. ARCHITECTURE

### Project Stats
- **54,500+ lines** of Python across 79 modules
- **1,558 tests** passing in ~14s
- **8-mixin GUI** pattern with CustomTkinter (14 sub-mixin files)
- **100% local** — no cloud, no npm, no build step

### GUI Mixin Inheritance
```
EnigmaGUI(DocsPageMixin, ForgeMixin, ModMixin, ModPageMixin, CMDPageMixin, LogicMixin, PagesMixin, ctk.CTk)
```

### GUI Files (enigma_engine/gui/)
| File | Responsibility |
|------|---------------|
| themes.py | Theme dataclass (20 fields), 4 presets, load/save preference |
| widgets.py | C_* color constants, fonts, widget classes, SelectableLabel, CollapsiblePanel, SelectableTextbox, Tooltip, factories |
| desktop.py | Window shell, header, nav rail, status bar, TTS lifecycle, geometry persistence, auto-restart, shortcuts overlay, entry point |
| gui_pages.py | Page builder hub: CORE, MODELS, ROUTER (inherits ForgePageMixin + ConfigPageMixin) |
| gui_pages_forge.py | FORGE page layout (ForgePageMixin) |
| gui_pages_config.py | CONFIG page layout, backup/restore (ConfigPageMixin) |
| gui_docs_page.py | DOCS page: browser, editor, search filter, inline rename, unsaved detection |
| gui_logic.py | Logic hub: config, model loading, routes, display names, toggles (inherits LogicChatMixin + LogicMediaMixin) |
| gui_logic_chat.py | Chat messaging, session management, typewriter, file attachment, history, token counter (LogicChatMixin) |
| gui_logic_media.py | Media rendering, voice I/O, TTS (LogicMediaMixin) |
| gui_forge.py | Forge hub: training setup, shared utils, dispatch, _build_generation_prompt(), _format_training_pair() (inherits ForgeTrainingMixin + ForgeAdvancedMixin + ForgeAdaptiveMixin + ForgeNewModesMixin + ForgeToolsMixin + ForgeModelsMixin + ForgeQueueMixin) |
| gui_forge_training.py | Basic training modes: solo, DPO, vision, LoRA (ForgeTrainingMixin) |
| gui_forge_advanced.py | Advanced training: evolutionary, guided, dialogue, curriculum saving to data/ (ForgeAdvancedMixin) |
| gui_forge_adaptive.py | Adaptive pipeline: TC-C3 continuous adaptive loop, SA-B auto-chain stages, SA-C saveable/resumable JSON plan (ForgeAdaptiveMixin) |
| gui_forge_tools.py | Forge tools: data gen, evaluate, web learn, checkpoints, cards, loss chart canvas (ForgeToolsMixin) |
| gui_forge_models.py | Model ops: import, create, copy, rename, delete, quantize, GGUF export, HuggingFace download (ForgeModelsMixin) |
| gui_forge_queue.py | Queue, overnight plan, curated dataset GUI callbacks: add/show/run queue, save/load plan, review/approve dataset (ForgeQueueMixin) |
| gui_forge_new_modes.py | New training modes: RLHF (2-phase: reward model → PPO), Self-Play (TRAINER as reward) (ForgeNewModesMixin) |
| gui_mods.py | Mod subprocess lifecycle (start/stop/auto-start) |
| gui_mod_page.py | Per-mod page builder from mod.json |
| gui_cmd_page.py | Dual-mode terminal (SYSTEM/ENGINE), AI ACCESS, status strip, info commands |
| media.py | Image/GIF/video detection, inline rendering, URL links, MAX_CHAT_IMAGES cap |
| scanners.py | Filesystem scanning, ROUTE_KEYS, PATH_SETTINGS, param counting |

### Core Files (enigma_engine/core/) — Key Modules
| File | Responsibility |
|------|---------------|
| model.py | Transformer model definition, forward() accepts pad_token_id kwarg |
| inference.py | `_init_common()` shared init, `_load_gguf()`/`_load_pytorch()` loaders, generation loop, sampling, batch generation, GGUF GPU auto-detection, `_token_count_cache` for repeated token counting |
| engine_chat.py | Chat completion, `ChatContext` dataclass + `_prepare_chat()` shared logic, auto-truncation, GGUF + native paths, reasoning, image encoding |
| engine_generation.py | Token sampling, vectorized batch sampling, multimodal generation |
| vision_encoder.py | VisionEncoder (ViT from scratch), CNNStem (hybrid CNN+ViT), preprocess_image, encode helpers, frame dedup (cosine similarity), `max_visual_tokens` truncation |
| training.py | SFT, evolutionary, LoRA/QLoRA, DPO, vision training, reasoning tag support, User/AI dialogue parsing, dedup, dynamic pad token, model max_seq_len batching, DataValidationResult + validate_training_data, `_train_one_batch()` + `_handle_oom()` OOM recovery |
| gguf_loader.py | GGUF model loading + llama-server backend, tool-calling (run_command removed from defaults for security) |
| huggingface_loader.py | HuggingFace model loading |
| web_utils.py | Shared web utilities — DDG search, `_validate_url()` SSRF protection, streaming page fetch (1 MB cap), HTML text extraction |
| builtin_commands.py | [CMD] implementations, memory commands, image gen, model compare |
| tokenizer.py | Main tokenizer interface, `ImportError`-only catching (no silent fallback), `<think>`/`</think>` special tokens |
| commands.py | Command parsing, registry, plugin auto-loading, `sanitize_args()` strips shell metacharacters from AI-generated args |
| reasoning.py | Chain-of-thought engine, `<think>` tags, strip_incomplete_think |
| memory.py | Persistent AI memory — fact storage, extraction, context injection |
| plugin_loader.py | Plugin discovery with 3-layer security: trusted allowlist, `def register` pre-scan, AST danger scan (flags exec/eval/os.system/subprocess) before exec_module |
| model_presets.py | Model size presets, config_for_param_target() |
| safe_save.py | Atomic saves (torch + safetensors + text + JSON) — write to .tmp then os.replace(), fsync for durability, .bak backup rotation |
| hardware_detection.py | GPU/CPU/RAM auto-detection, `recommend_training_batch_size()` VRAM-tier batch sizing |
| streaming.py | Streaming generation with token-by-token output |\n| kv_cache.py | KV cache for transformer generation, quantized (int8) and non-quantized paths, `get()` returns cloned tensors for safety |
| ai_profile.py | AI personality profiles with generation/memory configs |
| model_context.py | Per-model persistent history, prompt, identity, and session path storage |
| model_registry.py | Model file registry, `safe_load_weights()` centralized loader, metadata tracking |
| gguf.py | GGUF binary format parser |
| lora_utils.py | LoRA/QLoRA adapter utilities + LoRAAdapterManager (per-task create/save/switch/merge) |
| rag.py | RAG with BM25 scoring, stop-word filtering, optional scipy sparse matrices, document chunking |
| document_readers.py | Text extraction from PDF, DOCX, and other file formats |
| onnx_loader.py | ONNX model loading |
| gptq_awq_loader.py | GPTQ and AWQ quantized model loading |
| mod_tools.py | Mod discovery, command registration, structured AI tool prompts |
| auto_research.py | Proactive web research — heuristic trigger, DDG search, LRU cache (100 entries), rate limiting (5s), parallel page fetching (ThreadPoolExecutor), context injection |
| adaptive_trainer.py | Adaptive training plan, linear stage progression, difficulty-aware prompts (12 stage×difficulty templates), atomic save |
| training_queue.py | Training queue (FIFO, background thread, pause/resume, save/load JSON, crash recovery), overnight plan (save/load/resume) |
| curated_dataset.py | JSONL curated dataset with approve/reject workflow, queries by source/stage, training data export, thread-safe (Lock + copy-on-read) |
| rl_training.py | Reinforcement learning: RewardModel, RewardTrainer, RLHFTrainer (PPO + KL penalty, LoRA-based reference), SelfPlayTrainer (TRAINER as reward, LoRA-based reference) |
| training_monitor.py | Training monitoring: TrainingMonitor (per-step loss, moving average, epoch perplexity tracking, training history JSON, loss chart data) |

### Key Patterns

| Area | Pattern | Details |
|------|---------|--------|
| **Architecture** | Mixin pattern | Split large classes into focused mixins, inherit in one main class |
| | Re-exports | When splitting files, re-export from original module for backward compat |
| | Lazy package init | `__getattr__` + `_LAZY_LOADER_MAP` in `core/__init__.py` |
| | Deferred imports | `_ensure_*_imports()` for heavy libs — saves 540 MB idle RAM |
| | Optional imports | `try/except ImportError` with graceful fallbacks |
| | Constants in one place | Colors, fonts, route keys defined once, imported everywhere |
| | Background threads | All heavy ops (model load, training, chat) run in threads |
| **GUI** | Widget factories | `themed_entry()`, `themed_dropdown()`, `themed_scroll()` |
| | SelectableLabel | All display text uses tk.Entry readonly, not plain CTkLabel |
| | Inline confirmation bars | Red/yellow/orange bars with YES/NO — never popups |
| | Right-click context menus | `<Button-3>` on cards and widgets |
| | Deferred action callbacks | `_pending_action` + `_confirm_*` / `_cancel_*` pairs |
| | Inline name editing | Entry `state="normal"` + orange border + SAVE/CANCEL + Enter/Escape |
| | STOP/SEND swap | Same grid slot, swap via grid/grid_forget |
| | Status bar | Cross-page feedback visible from any page |
| | `_model_op_busy()` guard | Prevents concurrent copy/import/create/delete |
| | Deferred boot scanning | scan_models/scan_mods in background after window shows |
| | Window geometry | Saved to gui_settings.json, restored on launch, clamped on-screen |
| | Theme system | Theme dataclass presets → C_* constants, `_restart_gui()` for changes |
| **Engine** | Dual engine path | GGUF + PyTorch both work in engine_chat.py |
| | llama-server backend | For unsupported GPU arches (Blackwell) |
| | Plugin loader | Discover .py in plugins/, 3-layer security (allowlist → pre-scan → AST) before exec_module |
| | Per-model context | Each model gets own history, prompt, session path |
| | Per-route prompts | data/prompts/chat.md and trainer.md |
| **Security** | Command arg sanitization | `sanitize_args()` strips `;\|&$\`\\!{}` from all command args before handler execution |
| | Plugin allowlist | `trusted_plugins` config — empty = allow all, non-empty = only listed filenames loaded |
| | Plugin AST scan | `_ast_scan_dangers()` flags exec/eval/os.system/subprocess/shutil.rmtree before loading |
| | No default system tools | `run_command` removed from GGUF `_get_default_tools()` — AI has no system command access by default |
| | CORS off by default | No CORS middleware unless `--cors-origins` is explicitly passed to `run_server()` |
| | Input size validation | Pydantic `Field` constraints: 32K message, 50 batch prompts, 256 path length |
| | Inference concurrency lock | `_inference_lock` — chat/stream/batch return 429 when engine is busy |
| | Path traversal protection | Model load + train endpoints verify paths resolve inside MODELS_DIR/DATA_DIR |
| **Training** | Data parsing | 5 formats: Q/A, User/AI, Human/Assistant, JSONL, raw text |
| | Stage-aware data gen | `_build_generation_prompt()` + `_format_training_pair()` per stage |
| | Data validation | `validate_training_data()` with warnings/errors/stats before training |
| | NaN detection + early stopping | Training guardrails |
| | OOM recovery | `_handle_oom()` clears cache + enables gradient checkpointing, retry once |
| | VRAM-aware batch sizing | `recommend_training_batch_size()` — tier-based defaults by GPU VRAM |
| | CPU-first student loading | Load to CPU, free trainer, move to GPU (prevents OOM) |
| | Rolling checkpoints | `rolling_best_k` keeps N best by loss, auto-deletes worst + periodic cleanup |
| | Adaptive training | TrainingPlan JSON save/load, phase 1/2/3, 12 stage×difficulty templates |
| | Training queue | Singleton lazy-init, executor bridges to Trainer, start/pause/resume |
| | Curated dataset | Auto-accumulate from data gen/web learn/guided, approve/reject workflow, thread-safe |
| | RL training | RewardModel + RLHFTrainer (PPO, LoRA ref) + SelfPlayTrainer (TRAINER as reward, LoRA ref) |
| | Training monitor | Per-step loss, moving average, epoch perplexity, history JSON |
| | LoRA manager | Per-task adapters in models/lora_adapters/{task}/, switch/merge |
| | Loss chart | Canvas line chart, thread-safe via self.after() |
| **Testing** | `structural` marker | Source-inspection tests tagged `@pytest.mark.structural`, deselect with `-m "not structural"` |
| | Ruff-based lint tests | `TestDeadImports` uses `ruff check --select F401` for reliable unused import detection |
| **Reasoning** | Chain-of-thought | `<think>`…`</think>` tags, `strip_incomplete_think()` |
| | Display | `🧠 Reasoning:` label + dim text inline, TTS speaks only the answer |
| | Special tokens | `<think>`/`</think>` above enc.n_vocab, no ID collision |
| **Vision** | Hybrid CNN+ViT | CNNStem (/8 reduction) + ViT, hybrid_small/hybrid_medium presets |
| **Voice** | Persistent TTS worker | pyttsx3 init once, Queue feeds, poison pill shutdown |
| **Memory** | PersistentMemory | data/notes/memory.md, topic-aware dedup, MAX_FACTS cap (50) |
| **Saves** | Atomic saves | `atomic_torch_save()` everywhere, zero direct `torch.save` or `torch.load` |
| | Safe loading | `safe_load_weights()` is the single entry point for all model/checkpoint loading — supports .pth, .safetensors |
| | Atomic text/JSON | `atomic_write_text()` / `atomic_write_json()` with fsync + .bak backup for all 8 data modules |
| **Config** | Config overrides | Persisted to gui_settings.json on close, restored on launch |
| **Mods** | ModClient base class | Sync sockets + threading, mods only implement cmd_* handlers |

### Color Palette (themes.py → widgets.py)
Colors defined in `themes.py` as `Theme` dataclass presets, loaded into `C_*` constants in `widgets.py` at import. 4 themes: dark (default), midnight, carbon, solarized. Saved in `data/gui_settings.json["theme"]`.
- Backgrounds: `#080808` (bg), `#0e0e0e` (panel), `#181818` (surface), `#1c1c1c` (input)
- Text: `#e8e8e8` (bright), `#b0b0b0` (normal), `#555555` (dim)
- Accent: `#8B95A5` (silver), `#a855f7` (purple/user), `#22d3ee` (cyan/CMD)
- Status: `#22c55e` (green), `#ef4444` (red), `#f97316` (orange)
- Borders: `#1f1f1f` (default), `#2e2e2e` (accent), `#2a2a2a` (dim), `#3d3d3d` (hover)

### GUI Pages
- **CORE** — Chat with sidebar (history/prompt), fullscreen, web toggle, media, STOP/EDIT, reasoning display
- **CMD** — Dual terminal (SYSTEM/ENGINE), AI ACCESS toggle, live status strip, 10 info commands
- **DOCS** — File browser with search, editor with inline rename, unsaved detection, notes category
- **MODELS** — Create/import/download/copy/rename/delete models, param count, NATIVE/EXTERNAL tags
- **ROUTER** — Assign models to routes (CHAT, TRAINER, each mod)
- **FORGE** — Train (9 modes + Train with AI + Include reasoning toggles), evaluate, generate data, web learn, auto-train, checkpoints, progress bar, loss chart canvas, focus field, training queue, overnight plan, curated dataset review
- **CONFIG** — Generation parameters, directory paths, display names, theme picker, AI profile switcher, backup/restore
- **MOD pages** — Dynamic per-mod pages from mod.json

---

## 6. WHEN STUCK

1. Stop — do not guess or hallucinate
2. State what you know and what you don't
3. Ask the user for clarification
4. If a fix breaks something else, revert and rethink — do not patch on top of patches
5. If a task is ambiguous, ask for clarification before writing code
6. Explain *why* when pushing back on an approach

---

## 7. WORKFLOW

### Before Coding
1. Read this file
2. Read SUGGESTIONS.md
3. Check what files exist (search before building)
4. Write test first

### After Coding
1. Run `python check.py` (lint + tests in one command)
2. Run `python run.py` to verify
3. Update KEY LESSONS if something went wrong or worked well
4. Update SUGGESTIONS.md session log

### Priority Order (when rules conflict)
**Correctness > Simplicity > Performance > Style**

---

## 8. CODE QUALITY

- Consider edge cases and error paths — do not ignore warnings or problems
- Do not remove comments or code unless there is a reason
- Add clear comments explaining non-obvious logic
- Propose 2-3 approaches before implementing — let the user choose
- Update correlating files when changes are made (tests, docs, imports)
- When adding something new, think about what the user can do with it
- Everything should work — nothing is placeholder
- Cross-platform: must work on both Linux and Windows
- If changing core/, run full test suite
- Ignore the installed mods and non-trained base model — not relevant
- evaluate what you have done
- do something about code if you find something worth noting.

---

## 9. QUICK COMMANDS

```
python check.py                                     # Lint + tests (run before every commit)
python check.py --fix                                # Auto-fix safe lint issues, then test
python check.py --lint                               # Lint only (fast)
python check.py --test                               # Tests only
python -m pytest tests/ -v                           # Run all tests verbose
ruff check enigma_engine/                            # Run linter standalone
python run.py                                        # Show system info
python run.py --gui                                  # Launch desktop GUI
python run.py --serve                                # Start API server (port from CONFIG)
python run.py --serve --port 8080                     # Start API server on specific port
python run.py --train data/training.txt --epochs 10  # Train model
python run.py --train-tokenizer data/training.txt    # Train BPE tokenizer
python run.py --help                                 # Show all CLI options
```