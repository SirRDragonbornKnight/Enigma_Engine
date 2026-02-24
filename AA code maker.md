# AA Code Maker - AI Instructions

Read this file before writing code. Update after each session.

---

## 1. RULES

- Check SUGGESTIONS.md before adding anything new
- Read files before editing them
- Verify imports and attributes exist before using them
- Write tests first, then build features
- if you can make it local
- work smarter, not harder
- Keep it simple, don't over-engineer
- make sure it is realistic

---

## 2. TEST LOOP

```
Write Test → Build Feature → Run Test → Pass? → Yes: Refactor & Merge Tests → No: Fix & Repeat
```

Command: `python -m pytest tests/ -v`

---

## 3. ARCHITECTURE

### Project Stats
- **26,500 lines** of Python across 55 modules
- **89 tests** passing in ~3s
- **7-mixin GUI** pattern with CustomTkinter
- **100% local** — no cloud, no npm, no build step

### GUI Mixin Inheritance
```
EnigmaGUI(DocsPageMixin, ForgeMixin, BrickMixin, BrickPageMixin, CMDPageMixin, LogicMixin, PagesMixin, ctk.CTk)
```

### GUI Files (enigma_engine/gui/)
| File | Lines | Responsibility |
|------|-------|---------------|
| widgets.py | 495 | Colors, fonts, widget classes, CollapsiblePanel, SelectableTextbox, Tooltip, factory functions |
| desktop.py | 398 | Window shell, header, nav rail (collapsible), status bar, label copy, display name loading, entry point |
| gui_pages.py | 1084 | Page builders: CORE (fullscreen + web search), MODELS, ROUTER, FORGE, CONFIG (paths + display names) |
| gui_docs_page.py | 401 | DOCS page: documentation browser, file editor, profile creation, CRUD operations |
| gui_logic.py | 1043 | Chat, sessions, profiles, route assignment, model loading, per-model context, voice input, path settings, display names, web search |
| gui_forge.py | 429 | Training, tokenizer training, model create/delete, data file editing |
| gui_bricks.py | 171 | Brick subprocess lifecycle (start/stop/auto-start) |
| gui_brick_page.py | 321 | Per-brick page builder from brick.json |
| gui_cmd_page.py | 577 | Dual-mode terminal (SYSTEM shell + ENGINE commands + AI ACCESS) |
| scanners.py | 296 | Filesystem scanning, config limits, ROUTE_KEYS, PATH_SETTINGS, scan_docs |

### Core Files (enigma_engine/core/) — Largest
| File | Lines | Responsibility |
|------|-------|---------------|
| model.py | 1541 | Transformer model definition |
| training.py | 1175 | SFT, evolutionary, LoRA/QLoRA training |
| inference.py | 1157 | Generation loop, sampling, batch generation |
| huggingface_loader.py | 1129 | HuggingFace model loading |
| builtin_commands.py | 1104 | All [CMD] command implementations |
| gptq_awq_loader.py | 976 | GPTQ/AWQ quantized model loading |
| lora_utils.py | 936 | LoRA adapter utilities |
| gguf.py | 867 | GGUF format parsing (shared by loaders) |
| tokenizer.py | 827 | Main tokenizer interface |
| gguf_loader.py | 798 | GGUF model loading |
| model_context.py | 265 | Per-model persistent history and prompt storage |

### Key Patterns
- **Mixin pattern** — split large classes into focused mixins, inherit in one main class
- **Re-exports** — when splitting files, re-export from original module for backward compatibility
- **Widget factories** — `themed_entry()`, `themed_dropdown()`, `themed_scroll()` reduce boilerplate
- **Static helpers** — extract duplicated dicts/configs into `@staticmethod` methods
- **Optional imports** — `try/except ImportError` with graceful fallbacks
- **Background threads** — all heavy ops (model load, training, chat) run in threads
- **Constants in one place** — colors, fonts, route keys defined once, imported everywhere

### Color Palette (widgets.py)
- Backgrounds: `#080808` (bg), `#0e0e0e` (panel), `#181818` (surface), `#1c1c1c` (input)
- Text: `#e8e8e8` (bright), `#b0b0b0` (normal), `#555555` (dim)
- Accent: `#8B95A5` (silver/gray), `#a855f7` (purple for user messages), `#22d3ee` (cyan for CMD)
- Status: `#22c55e` (green), `#ef4444` (red), `#f97316` (orange)
- Borders: `#1f1f1f` (default), `#2e2e2e` (accent), `#2a2a2a` (dim), `#3d3d3d` (hover)

### GUI Pages
- **CORE** — Chat interface with collapsible history/prompt sidebar, per-model context, fullscreen mode, web search
- **CMD** — Dual-mode terminal: SYSTEM (real PowerShell) + ENGINE (AI commands) with AI ACCESS toggle
- **DOCS** — Documentation browser, file editor, profile management, brick docs
- **MODELS** — Create and delete model files
- **ROUTER** — Assign models to routes (CHAT, TRAINER, each brick)
- **FORGE** — Train models and tokenizers with real-time log
- **CONFIG** — Generation parameters with friendly names, directory path settings, display names
- **BRICK pages** — Dynamic per-brick pages built from brick.json

---

## 4. LESSONS LEARNED

### Mistakes to Avoid
| Mistake | Prevention |
|---------|-----------|
| Recreated existing feature | Search codebase BEFORE building anything |
| Used nonexistent import or API | Verify imports and attrs exist before using |
| Claimed done without verification | Run tests, check code exists before documenting |
| Wrong function order or missing params | Define functions before calling, verify param flow |
| File encoding issues on Windows | Use `Out-File -Encoding utf8` |
| Python cache showed old code | Clear `__pycache__` when debugging |
| Large files became unmaintainable | Keep under 800 lines, split into mixins early |
| Dead config or deprecated API | Regenerate config after cleanup, check current API names |
| Duplicated logic across modules | Put shared code in ONE module, import elsewhere |
| `shell=True` with subprocess | Always use `shell=False` with `shlex.split()` |
| Mixin attribute warnings in Pylance | Expected — attrs set in `__init__` resolve via MRO at runtime |
| Cleared loaded session history | Use `_reset_display()` for display-only, `_new_chat()` for full reset |
| Nav collapse did nothing | Use grid_columnconfigure on parent, not frame width (sticky overrides it) |
| Mic button could not be stopped | Use listen_in_background() with stopper callable, not blocking listen() |
| Wired buttons to non-existent methods | Implement method bodies BEFORE wiring command= callbacks |

### Patterns That Work
| Category | Pattern |
|----------|---------|
| Architecture | Mixin pattern for splitting large classes — keeps tests green |
| Architecture | Re-export from original module after splits — no import breakage |
| Architecture | Constants in one file (widgets.py) — change once, propagates everywhere |
| Architecture | ROUTE_KEYS constant — consistent route identification across modules |
| GUI | Widget factories with `setdefault` — reduce 5-7 line calls to 1 |
| GUI | Background threads for all heavy ops — UI never freezes |
| GUI | CTkTextbox with word wrap — fixes text-offscreen bug |
| GUI | Typewriter chat with `after()` chain — 3 chars per 8ms |
| GUI | Dynamic UI from brick.json — 6 widget types rendered at runtime |
| GUI | CollapsiblePanel widget — clickable header with chevron toggles content visibility |
| GUI | Per-model context auto-save/load — each model gets own history and prompt |
| GUI | Tooltip class — dark popup on hover for button descriptions |
| GUI | SelectableTextbox — read-only CTkTextbox that allows select/copy |
| GUI | _enable_label_copy tree-walk — right-click copy on all CTkLabels |
| GUI | Toolbar row pattern — separate utility buttons from action buttons |
| GUI | DOCS page with scan_docs() — auto-discovers guides, profiles, brick docs |
| GUI | Category-based file browser — groups files by source with color coding |
| GUI | Fullscreen chat toggle — hide header/nav/status, restore on exit or Escape |
| GUI | Per-model display names via model_info.json — priority chain for AI name |
| GUI | Web search dialog with background thread — non-blocking DuckDuckGo queries |
| Engine | GGUF + PyTorch both work with proper chat templates |
| Engine | KV-cache clearing prevents hallucinations |
| Engine | Optional imports with `try/except` graceful fallbacks |
| Engine | Auto-detect hardware (VRAM, GPU layers, context size) |
| Training | NaN detection, early stopping, config validation guardrails |
| Training | `_model_config_dict` static helper — one source for model config |
| Training | LoRA/QLoRA via PEFT, evolutionary self-play — all working |
| Bricks | TCP plugin system — bricks connect TO main app, not the other way |
| Bricks | Auto-start on launch, `_launch_brick` shared helper for Popen |
| Bricks | `scan_bricks()` returns commands, UI spec, and prompt for pages |
| Bricks | Brick docs in bricks/<id>/docs/ auto-discovered by scan_docs() |
| Testing | Test-first workflow — write test, build feature, verify |
| Testing | Functional tests — forward pass, tokenizer round-trip, KV-cache, commands |

---

## 5. DO NOT ADD

- No symbols or special characters in names
- No untested code
- No duplicate code — check if it exists first
- No invisible/hidden characters
- No leftover debug prints (`print`, `console.log`)
- No redundant comments that just repeat the code
- No hardcoded file paths
- No unused imports
- No random formatting changes
- No sliders — use numeric input only
- No backward-compatible shims

---

## 6. WORKFLOW

### Before Coding
1. Read this file
2. Read SUGGESTIONS.md
3. Check what files exist (search before building)
4. Write test first

### After Coding
1. Run `python -m pytest tests/ -v`
2. Run `python run.py` to verify
3. Update LESSONS LEARNED above
4. Update SUGGESTIONS.md session log

---

## 7. QUICK COMMANDS

```
python run.py                    # Show system info
python -m pytest tests/ -v       # Run all tests
python -c "from enigma_engine.core import EnigmaEngine; print('OK')"  # Check imports
python run.py --train data/training.txt --epochs 10   # Train model
python run.py --train-tokenizer data/training.txt     # Train BPE tokenizer
python run.py --serve                                # Start web GUI on port 8080
python run.py --gui                                  # Launch desktop GUI
python run.py --help                                  # Show all CLI options

```

## 8. CODE QUALITY

- Handle errors properly — do not ignore warnings, errors, or problems
- Consider edge cases
- Do not remove comments or code unless there is a reason
- Add clear comments explaining logic
- Find the best approach before writing code
- Verify nothing similar already exists before adding
- Double check before making changes
- This code is still in development — everything should work, nothing is placeholder
- Make sure additions are realistic and reasonable
- make the entire gui selectable

---