# Enigma Engine - Development Plan

---

## Session Log

### 2026-02-24 Session (Documentation Update + Missing Methods Fix)
- Discovered _web_search_dialog, _save_display_names, _reset_display_names were wired to buttons but never implemented
- Implemented all 3 missing methods in gui_logic.py + _load_display_names for startup loading
- _load_display_names() called in desktop.py __init__ after _load_config_defaults()
- _save_display_names persists user_name/ai_name to data/gui_settings.json
- _reset_display_names restores defaults ("YOU"/"ENIGMA") and clears entries
- _web_search_dialog opens CTkInputDialog, runs search.web in background thread, inserts results into chat
- Updated GUI_REFERENCE.md: chat input 56px, thinking indicator fixed width, web search button, display names section, file map line counts, 7-mixin inheritance
- Updated AA code maker.md: GUI file line counts, core file line counts, new patterns, new lesson learned
- gui_pages.py at 1084 lines and gui_logic.py at 1043 lines — both over 800-line target, candidates for splitting
- 89 tests pass in ~3s

### 2026-02-24 Session (Display Names + Chat Input)
- Chat input height reduced from 100px to 56px (SEND button matches)
- Fixed processing indicator layout shift - label now has fixed width (140px), dots no longer push GUI around
- AI name now changes per-model via model_info.json in the model folder
- models/qwen2.5-14b-instruct/model_info.json created with display_name "Qwen"
- New methods: _load_model_display_name, _active_ai_name in LogicMixin
- Priority: model_info.json display_name > self.ai_name (CONFIG setting) > "ENIGMA" default
- Name cleared on model unload, falls back to CONFIG ai_name
- Chat messages use self.user_name and _active_ai_name() - no more hardcoded "YOU"/"ENIGMA"
- CONFIG page has DISPLAY NAMES section with Your Name / AI Name entries + SAVE/RESET
- user_name and ai_name attributes on EnigmaGUI (defaults: "YOU" / "ENIGMA")
- Web search button added to CORE toolbar with cyan accent
- 89 tests pass (83 existing + 6 new: name attributes, chat names, history names, model display name, model_info.json, config entries)

### 2026-02-24 Session (Chat Fullscreen Mode)
- Added fullscreen chat toggle — CORE page covers entire GUI (header, nav, status bar hidden)
- Fullscreen button (\u26f6) in CORE top bar, next to sidebar toggle
- Enter fullscreen: hides header, nav rail, status bar; sidebar auto-hidden for max space
- EXIT button (red) appears in top bar when fullscreen is active
- Press Escape key or click EXIT to return to normal layout
- Restores previous nav and sidebar state on exit
- New methods: _toggle_chat_fullscreen, _exit_chat_fullscreen, _on_escape_fullscreen in PagesMixin
- _chat_fullscreen state attribute on EnigmaGUI, _header ref stored in desktop.py
- 83 tests pass (78 existing + 5 new: fullscreen toggle, button, state, header ref, escape key)

### 2026-02-24 Session (DOCS Page + Documentation)
- Created DOCS page — full documentation browser with file editor and CRUD operations
- New DocsPageMixin (gui_docs_page.py, 401 lines) with category-based file browser and text editor
- scan_docs() in scanners.py discovers files from information/, profiles/, and bricks/*/docs/
- Three doc categories: guides (cyan), profiles (purple), brick:<id> (silver)
- File browser shows categorized sections with clickable file entries
- Full editor with SAVE, DELETE, RELOAD buttons and filename indicator
- + NEW button creates documentation files (.md) in information/
- + PROFILE button creates AI profile JSON files in profiles/
- Created 5 default guide files in information/: how_the_ai_works, training_guide, commands_reference, getting_started, prompts_guide
- Created brick docs: bricks/echo/docs/about.md, bricks/imagegen/docs/about.md
- All doc files verified against actual codebase for accuracy (model sizes, commands, profiles, CLI flags)
- Removed PROFILE dropdown from CORE page — replaced with small purple indicator label showing active profile
- Profile management now lives on DOCS page (create, edit, delete profiles)
- Added DOCS to nav rail between CMD and MODELS
- 7-mixin GUI pattern: DocsPageMixin + ForgeMixin + BrickMixin + BrickPageMixin + CMDPageMixin + LogicMixin + PagesMixin
- INFO_DIR constant and scan_docs() re-exported from desktop.py for backward compatibility
- 118 tests pass (105 existing + 13 new: docs mixin, scan_docs, nav integration, file readability)

### 2026-02-24 Session (GUI Usability + HF Model)
- Fixed nav collapse: switched from frame width to grid_columnconfigure — now actually resizes
- GUI minimum size reduced from 1100x720 to 800x500 — sidebar uses weight-based columns
- Fixed mic button: replaced blocking listen() with listen_in_background() + stopper callable — click again to cancel instantly
- Rearranged CORE button layout: SEND/NEW stacked right of input, voice/mic/attach in toolbar row below
- Added Tooltip widget class in widgets.py — dark popup on hover after 400ms delay
- Tooltips on SEND, voice, mic, attach, sidebar toggle, nav toggle buttons
- All CTkLabel text now right-click copyable via _enable_label_copy tree-walk in desktop.py
- SelectableTextbox class (read-only but allows select/copy) used for chat, history, logs
- Deleted all old models (qwen2.5-32b, enigma_small, enigma_tiny, empty dirs)
- Downloaded Qwen2.5-14B-Instruct-Q4_K_M.gguf (8.4 GB) from bartowski on HuggingFace
- Registered new model in models/registry.json
- 104 tests pass (86 existing + 18 new: tooltip, button layout, mic stop, label copy, nav grid, selectable text)

### 2026-02-24 Session (FORGE Data Editor)
- FORGE page now has a 3-column layout: Controls | Data Editor | Output Log
- Data Editor panel lets users view and edit training data files directly from the GUI
- Selecting a data source in the dropdown loads its content into the editor
- SAVE button writes editor content back to the file
- NEW FILE button creates a new .txt file in data/ (prompts for name)
- REFRESH button re-scans data files and updates the dropdown
- First data file auto-loads into editor when FORGE page builds
- New methods on ForgeMixin: _load_data_into_editor, _save_data_file, _new_data_file, _refresh_data_files
- CORE page sidebar (history + prompt) now has a toggle button (\u25e8) in the top bar
- Click the toggle to hide the sidebar — chat expands to full width
- Click again to bring the sidebar back with panels intact
- Button dims when sidebar is hidden, accented when visible
- 77 tests pass (72 existing + 5 new: data editor + sidebar toggle)

### 2026-02-24 Session (Voice Input Button)
- Added voice input (microphone) button to CORE page input area
- Click the mic button to start recording from the microphone
- Transcribed text is inserted into the chat input field
- Button turns red while recording, dims when idle
- Click again or wait for silence to stop recording
- Uses speech_recognition library (optional import, graceful fallback)
- Uses Google speech-to-text (works offline with vosk if installed)
- Recording runs in background thread — UI never freezes
- 79 tests pass (77 existing + 2 new: mic button, voice input methods)

### 2026-02-24 Session (Collapsible Nav + Directory Paths)
- Nav rail is now collapsible via hamburger button (☰) in the header
- Collapsed nav shrinks to 46px showing only accent bars (no text)
- Expanded nav restores to 170px with full labels
- Hamburger icon dims when collapsed, accented when expanded
- CONFIG page now has DIRECTORY PATHS section at the bottom
- Seven editable paths: Models, Training Data, Outputs, Profiles, Sessions, Memory, Bricks
- Each path has a text entry + browse (...) button for directory picker
- SAVE PATHS persists overrides to data/path_settings.json
- RESET restores all paths to defaults
- Saved paths auto-load into entries on startup
- New in scanners.py: PATH_SETTINGS, OUTPUTS_DIR, load_path_settings, save_path_settings, get_path
- 86 tests pass (79 existing + 7 new: nav collapse, path settings)

### 2026-02-23 Session (Collapsible Sidebar Panels)
- HISTORY and SYSTEM PROMPT sections on CORE page are now collapsible
- New CollapsiblePanel widget in widgets.py — clickable header with chevron (▼/▶) toggles content
- Click header to collapse a panel, click again to expand
- When one panel is collapsed, the other expands to fill the space
- When both collapsed, only the two thin header rows are visible
- _rebalance_sidebar() dynamically adjusts grid row weights based on panel states
- Both panels start expanded (sharing space equally) by default
- GUI_REFERENCE.md updated with collapsible behavior docs
- 72 tests pass (69 existing + 3 new: CollapsiblePanel class, methods, and usage in CORE page)

### 2026-02-23 Session (Per-Model Context Storage)
- Added per-model context system — each model gets its own persistent history and system prompt
- New module: enigma_engine/core/model_context.py (ModelContext class, ~230 lines)
- Storage layout: data/model_contexts/<model_stem>/context.json + history.json
- context.json stores: system prompt, config overrides, profile ID, last-used timestamp
- history.json stores: chat messages array with validation on load
- model_key_from_path() derives clean key from model path (handles generic stems like "model.safetensors")
- Auto-load: when a model loads, its saved history and prompt are restored into the GUI
- Auto-save: after each chat exchange, on model unload, on session save, on prompt apply, on new chat
- _restore_history_display() replays loaded history into chat display widget
- GUI wiring: LogicMixin gains _save_model_context, _load_model_context, _restore_history_display
- desktop.py gains model_context attribute on EnigmaGUI
- Exported from core/__init__.py: ModelContext, model_key_from_path, load_model_context, list_model_contexts
- 69 tests pass (58 existing + 8 new functional + 3 new GUI integration)

### 2026-02-23 Session (Dual-Mode CMD + Status Bar Fix)
- Rewrote CMD page as dual-mode terminal: SYSTEM (real PowerShell) + ENGINE (AI commands)
- SYSTEM mode runs real shell commands via subprocess.Popen, tracks CWD changes
- ENGINE mode runs AI command registry, supports "ask <question>" to query the AI
- AI ACCESS toggle — when ON (green), AI-generated commands that aren't engine commands get forwarded to real PowerShell
- Mode toggle via CTkSegmentedButton (SYSTEM/ENGINE) at top of page
- Dynamic prompt label: "PS dir>" for system, "ENG>" for engine
- New color tag "ai_output" (orange) for AI-generated text responses
- Removed reference panel — single full-width terminal layout
- Status bar center now shows real CPU name (platform.processor) instead of just "CUDA"
- desktop.py updated with `import platform` for CPU detection
- gui_cmd_page.py went from 373 to 578 lines with dual-mode architecture
- 58 tests pass (56 existing + 2 new: test_cmd_mode_constants, test_cmd_engine_registry)

### 2026-02-23 Session (GUI Overhaul + CMD Page)
- Added CMD page — terminal-style command interface connected to AI command registry
- CMD page features: command execution, output log, command history (up/down arrows), AI ask mode, reference panel
- CMD page supports: help, clear, history, ask <question>, and all engine commands (config.*, model.*, file.*, etc.)
- AI responses in CMD page auto-detect [CMD] blocks and execute them
- NavButton redesign — left-edge accent bar indicator instead of background highlight
- Header redesign — split "ENIGMA ENGINE" title with contrast, dot+status on right
- Nav rail — removed "NAV" label, cleaner separator for bricks section, zero-padding buttons
- New colors: C_CYAN (#22d3ee) for CMD page accent, C_BORDER_ACCENT (#2e2e2e) for subtle borders
- Improved panel colors: C_PANEL darker (#0e0e0e), C_SURFACE (#181818), C_TEXT brighter (#b0b0b0)
- Chat messages now have left margins and spacing for readability
- File attachment tag uses cyan instead of silver
- 6-mixin architecture: ForgeMixin + BrickMixin + BrickPageMixin + CMDPageMixin + LogicMixin + PagesMixin
- 9 GUI files now (added gui_cmd_page.py), all under 800 lines
- 57 tests pass (51 existing + 6 new CMD page tests)

### 2026-02-23 Session (Condensation + File Splits)
- Added 3 widget factory functions to widgets.py (themed_entry, themed_dropdown, themed_scroll)
- Replaced ~15 verbose widget constructions in gui_pages.py with factory calls
- Merged font aliases (FONT_CHAT = FONT_MONO = FONT_BODY)
- Extracted _model_config_dict helper (removed duplicate 8-line dict)
- Merged _clear_chat/_new_chat into _reset_display + _new_chat (fixed latent session load bug)
- Extracted _launch_brick helper from _start_brick/_auto_start_brick
- Consolidated scan_models duplicate extension loop
- Converted history buttons (SAVE/LOAD/EXPORT) to loop
- Split gui_pages.py: brick page → gui_brick_page.py (BrickPageMixin, 321 lines)
- Split gui_logic.py: training → gui_forge.py (ForgeMixin, 353 lines), bricks → gui_bricks.py (BrickMixin, 171 lines)
- EnigmaGUI now inherits 5 mixins: ForgeMixin + BrickMixin + BrickPageMixin + LogicMixin + PagesMixin
- All 8 GUI files now under 800 lines (largest: gui_pages.py at 700)
- 51 tests pass in 2.87s

### 2026-02-23 Session (Models Page + Color Fix)
- Split ROUTER into MODELS page (create/delete) and ROUTER page (route assignments only)
- MODELS page: CREATE form (name + size) and DELETE button per model card
- ROUTER TRAINER route description updated to "Target model for training in the Forge"
- Color palette purged of all blue tints — pure black/gray backgrounds and borders
- Route dropdowns auto-refresh when models are created or deleted
- 51 tests pass in 2.77s

### 2026-02-23 Session (Test Trim + Dark Theme)
- Trimmed test suite from 86 to 51 tests — removed redundant import/scanner tests, consolidated classes
- Changed color theme from cerulean (#2E86C1) to silver/gray (#8B95A5) in widgets.py
- All 51 tests pass in 2.82s

### 2026-02-23 Session (Router + Brick Template)
- Added per-route model assignment (CHAT, TRAINER, each brick gets own model dropdown)
- scan_bricks() now returns prompt field, added ROUTE_KEYS constant
- 86 tests pass, each route can have its own model

### 2026-02-23 Session (Bricks as Pages + Template Alignment)
- Bricks are full navigable pages with info, commands, dynamic UI widgets, output log
- Template cleaned — removed PyQt5, added 6 widget types (text_input, text_area, number, button, dropdown, checkbox)
- 79 tests pass

### 2026-02-22 Session (GUI Performance Overhaul)
- Stripped 6 visual effect classes (PulseFrame, ScanlineOverlay, GlitchLabel, BootScreen, ActivityBar, CornerFrame)
- Cerulean color scheme, all fonts +5pt, ROUTER page replaces MODELS, CONFIG uses friendly names
- 76 tests pass

### 2026-02-21 Sessions (Web GUI + Desktop GUI + Splits)
- FastAPI web GUI (server.py + vanilla JS frontend, 11 API tests)
- CustomTkinter desktop GUI with mixin pattern (desktop.py + gui_pages.py + gui_logic.py + scanners.py + widgets.py)
- Split model.py (3097→1541), inference.py (2284→1158), commands.py (1243→164), gguf_loader.py (1105→799)
- Consolidated GGUF parsing into gguf.py, hardened router + training
- 76 tests pass

### 2026-02-21 Session (Codebase Cleanup)
- Created weight_mapping.py, removed dead GUI code from 6 files
- Fixed bugs in bpe_tokenizer, char_tokenizer, tokenizer, model, commands
- 16 tests pass, all imports clean

### 2026-02-20 Sessions (Training Pipeline + GUI Delete)
- Fixed BPE encode(), deprecated torch.cuda.amp, added model.eval() after training
- Cleaned CONFIG (83→54 keys), requirements (158→45 deps), added training CLI
- Deleted old PyQt5 GUI entirely
- 16 tests pass, engine trains and runs end-to-end

---

## Audit Summary

Full codebase audit completed 2026-02-20 through 2026-02-21. **30 of 30 cleanup items resolved.** Model architecture is solid (RoPE, RMSNorm, SwiGLU, GQA, flash attention, gradient checkpointing all correct). All critical/high bugs fixed, all file splits done, all dead code removed.

### Still Open (Architecture Improvements)

| # | Item | Effort | Impact |
|---|------|--------|--------|
| 1 | CONFIG dict → `@dataclass` config classes (kills duplicate keys, adds validation) | Medium | High |
| 2 | Side effects at import time — `defaults.py` runs `_load_user_config()` and `mkdir` on import | Low | Medium |
| 3 | Sub-packages — move loaders, tokenizers, training out of `core/` into own packages | Medium | Medium |
| 4 | `ModelBackend` Protocol — loaders have no shared interface | Medium | High |
| 5 | Delete `setup.py` — consolidate into `pyproject.toml` only | Low | Low |
| 6 | FILE_STRUCTURE.md is wrong — says GUI removed, lists deleted files as removed | Low | Low |
| 7 | Mixed async/sync bricks — `BrickClient` is async, `BrickRouter` is threading | High | Medium |
| 8 | Global `_LOADED_MODELS` dict — hidden state, hard to test | Medium | Medium |
| 9 | Move ASCII art to ARCHITECTURE.md, trim inline docstrings | Low | Low |
| 10 | Add `Protocol`/ABC interfaces for tokenizers and loaders | Low | Medium |
| 11 | Vectorize batch sampling in `batch_generate()` | Medium | Low |

---

## Current State (2026-02-24)

**Full engine with CLI, Web GUI, and Desktop GUI. 89 tests pass.**

### What Works
- CLI chat: `python run.py --chat --model path/to/model.gguf`
- Web GUI: `python run.py --serve` (FastAPI on port 8080)
- Desktop GUI: `python run.py --gui` (CustomTkinter, silver/gray dark theme)
- Train model: `python run.py --train data/training.txt --epochs 20 --model-size small`
- Train tokenizer: `python run.py --train-tokenizer data/training.txt --vocab-size 8000`
- System info: `python run.py`
- 89 tests pass: `python -m pytest tests/ -v`
- GGUF + PyTorch models both work with proper chat templates
- BPE tokenizer correctly applies merge rules during encoding
- RTX 5090 detected: 100 GPU layers, 16K context window
- Brick plugin system (echo + imagegen bricks)
- Per-route model assignment (CHAT, TRAINER, each brick gets own model)
- Tooltips on all action buttons, selectable/copyable text throughout GUI
- DOCS page with documentation browser, file editor, profile management
- Qwen2.5-14B-Instruct GGUF (8.4 GB) installed in models/
- 100% local — no cloud dependencies

---

## Features To Add

### Quick Wins
| Feature | Description | Notes |
|---------|-------------|-------|
| PDF Parsing | Read PDF documents, extract text | Add pypdf2 |
| DOCX Parsing | Read Word documents | Add python-docx |
| Streaming to CLI | Show tokens as they generate | Wire streaming.py to run_chat() |

### Medium Features
| Feature | Description | Notes |
|---------|-------------|-------|
| RAG/Document Q&A | Ask questions about your files | Add sentence-transformers |
| Knowledge Base | Persistent memory with vector store | Profiles have fields for it |
| Code Sandbox | Execute Python safely | Add subprocess limits |
| DPO Training | Direct Preference Optimization | Needs preference data format |

---

## What Already Exists (Reference)

### Command System
AI outputs `[CMD]command[/CMD]` blocks that get parsed and executed.

**Commands:** config.get/set/list, model.info/list/load/switch/download, train.start/stop/status, train.data.add/list, file.read/write/append/list, search.files/content, search.web, web.fetch, note.add/list, memory.save/load/list, brick.list/status/start/stop/send, shell, system.info/clear, history, help, stop

### Training
- SFT (supervised fine-tuning), Best-of-N sampling, Evolutionary self-play, LoRA/QLoRA via PEFT — all working
- DPO — not yet implemented

### Brick Architecture
TCP plugin system on port 9900. Bricks register with name + commands, AI sends commands via `brick.send`.
- **echo** — test brick (working)
- **imagegen** — SD WebUI, ComfyUI, diffusers (working)

### Hardware Detection
Auto-detects GPU/VRAM/RAM. Sets GPU layers and context size automatically (16K for 24GB+, 8K for 12GB+, 4K otherwise).

---

## NEXT STEPS

1. Test training with own models on custom data
2. Wire streaming to CLI chat
3. Address open architecture items (CONFIG dataclass, ModelBackend Protocol)

---

## Evolutionary Training (Reference)

**Loop:** Run N instances on same task → score outputs → keep top-1 → fine-tune → repeat.

**Scoring:** Rule-based (tests pass?), self-eval, consistency, length/format, perplexity.

**Format:** `{"prompt": "...", "completion": "<winner>", "score": 91}`

---

## Brick Protocol (Reference)

TCP + JSON (newline-delimited) on localhost. Router on port 9900, bricks on 9901+.

**Message:** `{"id": "msg-123", "type": "command", "data": {"command": "generate", "args": {}}}`

---

## DO NOT ADD (Reference)

- No symbols or special characters in names
- No untested code
- No duplicate code — check if it exists first
- No invisible/hidden characters
- No leftover debug prints
- No redundant comments that just repeat the code
- No hardcoded file paths
- No unused imports
- No random formatting changes
- No sliders