# GUI Reference - Every Element Explained

This document maps every visible element in the Enigma Engine desktop GUI,
what it is, where it lives in code, and what it does.

Use this to decide what to change, move, remove, or redesign.

**Last synced:** April 28, 2026 (Pass 156k). **GUI change:** APO added as a 13th alignment-row radio card on FORGE — fifth alignment mode alongside GRPO/ReMax/SimPO/ORPO. Selecting APO routes through the same DPO training body with `loss_type="apo_zero"` so chosen and rejected are anchored independently to the reference (avoids the DPO "degrade-rejected" failure mode). Status bar / log lines / training-history label show "APO-ZERO" when the mode is active. **Earlier sync:** Pass 156i3 — DET-2 CLI-only, no widgets touched. Pass 148 — no GUI changes in Passes 137-148.

Per-pass GUI notes (Passes 119-135) are archived in [information/history/PASS_HISTORY.md](information/history/PASS_HISTORY.md). New GUI work gets a one-line entry below and a full entry in the archive.

**Recent closed design gaps (N-21..N-25, all closed by Pass 126):** Model merging row on MODELS page (N-21); AI-computed emotional state injected into generation via `_build_gui_context()` (N-22); RAG moved into `_prepare_chat()` so GUI/API/BackgroundTrainer all share retrieval (N-23); mod-router length-prefix framing (N-24); BackgroundTrainer defers batches while inference active (N-25).

**Known dead ends:** FORGE tool buttons auto-disable when prerequisites are missing. CORE toggles (TTS, Web, RAG) now surface errors and auto-reset. Mic per-phrase recognition failures are intentionally silent (keeps listening). Only remaining dead end: individual `recognize_google()` failures during continuous listening (by design).

**Text selectability:** Almost all display text uses `SelectableLabel` (tk.Entry in readonly state) — supports click-drag selection, Ctrl+C copy, and right-click copy menu with no blinking cursor. Multi-line labels that use `wraplength` remain as `CTkLabel` with right-click copy via `_enable_label_copy`.

**Universal hotkeys:** All text input widgets (CTkEntry and CTkTextbox) support Ctrl+Z (undo), Ctrl+Y (redo), and Ctrl+A (select all) via `wire_hotkeys()` in `widgets.py`. CTkTextbox widgets use the native tk.Text undo stack. CTkEntry widgets use a lightweight `_EntryUndoStack` (since tk.Entry has no native undo). All entries created via `themed_entry()` get hotkeys automatically. Other inputs are wired explicitly at creation time.

---

## FILE MAP

| File | What It Controls |
|------|-----------------|
| widgets.py | Theme-driven C_* color constants, fonts, font size offset system (get/set/load_font_size_offset), reload_theme() for live switching, wire_hotkeys() (unified Ctrl+Z/Y/A for all text inputs), _EntryUndoStack, widget classes (HUDFrame, GlowFrame, StatusDot, NavButton, SectionLabel, SelectableLabel, ToggleButton, StatusBar (3-zone left/center/right), CollapsiblePanel (chevron expand/collapse with on_toggle callback), SelectableTextbox, Tooltip (hover popup with boundary-aware dismiss)), factory functions (themed_entry, themed_dropdown, themed_scroll), _resolve_parent_bg() |
| desktop.py | Window shell: header (pin toggle, shortcuts overlay dropdown (Ctrl+1-7/Ctrl+N/Escape/etc)), nav rail (collapsible via grid_columnconfigure), page nav shortcuts (Ctrl+1-7, Ctrl+N), status bar, label copy, auto-start mods, status ticker (with hardware detection — CPU/GPU/RAM), display name loading, Escape-to-stop binding, TTS lifecycle (init/shutdown), deferred boot scanning, window geometry persistence (save/restore), **Forge training brief saved on close** (_on_close calls _save_training_brief before widgets destroyed), window state handlers (minimize/map for model suspension), live theme switching (_apply_theme_live, _retheme_tree), parent watchdog, reflection/monologue system (idle detection → _run_reflection → coherence scoring → journal storage), recent topics extraction, memory facts retrieval, coherence-gated journal greeting, performance/gaming mode (auto-derived UI throttles), auto-unload on minimize (suspend/resume model), GUI restart for font changes, resize debounce + completion tracking, router training sync, background model param counting, register GUI commands. Inherits 7 mixins. |
| gui_pages.py | Page builders: CORE (fullscreen toggle + voice I/O (mic + TTS toggle) + web toggle + RAG toggle + reasoning toggle + token counter + resizable sidebar + clickable history + media tags + STOP/EDIT buttons + auto-expanding input + file indicator + thinking indicator + chat input context menu + emotional state panel (5 dimensions: valence/energy/engagement/trust/frustration with progress bars + tooltips) + journal panel + collapsible sidebar panels), MODELS (identity cards with display name/personality/stats/tags + param count + tooltips + EDIT/EXPORT/COPY/GROW/DELETE + IMPORT/DOWNLOAD + NATIVE/EXTERNAL tags + inline delete confirmation + right-click context menu + HF repo inline entry + **merge row: model A dropdown + model B dropdown + SLERP/LINEAR/TIES dropdown + t entry + density entry + output name entry + MERGE button**), ROUTER (SUSPEND/UNLOAD buttons + route cards). Inherits ForgePageMixin + ConfigPageMixin |
| gui_pages_forge.py | FORGE page layout: paned resizable layout (controls left + log right with draggable sash) + trainer/student status cards + 13-mode radio-card selector (Foundation: `Pre-Train`, `Distill`, `Basic`, `Image` | Advanced: `AI-Guided`, `Dialogue`, `RLHF`, `Self-Play` | Alignment: `GRPO`, `ReMax`, `SimPO`, `ORPO`, `APO`) + reasoning checkbox + evolutionary selection checkbox + knowledge preservation (general mix + data file) + Pre-Train section (data browse, vocab size, retrain-tokenizer, byte-level BPE) + Distill section (6 category checkboxes, examples count, max tokens) + training brief editor (quick profile fields + custom instructions) + training stages section (BASICS/CONVERSATION/COMMANDS/WEB) + mode-specific sections + Auto-train checkbox + stage buttons with tooltips + Resume from checkpoint checkbox + TRAIN/STOP buttons + CollapsiblePanel tools (6 section separators: TRAINING/WEB/TOKENIZER/MODEL EXPORT/QUEUE/DATASET + generate data, evaluate, benchmark, history, web learn, tokenizer, quantize, export GGUF, command policy, queue/plan/dataset) + vision browse + vision encoder preset dropdown (tiny/small/medium) + LoRA advanced subsection + student param count label + hyperparameter presets + progress bar + loss chart canvas panel + loss chart info label + rolling best K + validation split + ADVANCED SETTINGS panel (replay capacity/ratio) (ForgePageMixin) |
| gui_pages_config.py | CONFIG page layout: generation parameters, paths, display names, live theme picker (no restart), font size control, learn-while-chatting toggle, history cap, memory mode, monologue mode, emotional state visibility toggle, file operation confirmation toggle, performance section (auto-load chat model, auto-start mods, auto-unload on minimize), gaming mode preset button, backup/restore with inline import confirmation, tooltips, debounced wraplength resize (100ms + same-value skip) (ConfigPageMixin) |
| gui_docs_page.py | DOCS page: documentation browser with search filter, path tooltips, file editor with path label and stats footer (live lines/words/chars), inline file rename, blank doc creation, unsaved change detection with inline bar (save/discard/cancel), Ctrl+S shortcut, Ctrl+F find bar with prev/next navigation + match counter, Ctrl+Z/Y undo/redo, inline delete confirmation, right-click context menu, auto-save (30s timer), notes category, CRUD operations |
| gui_logic.py | Logic hub: config, model loading, routes, display names, toggles, path settings, web access toggle, GUI context builder (`_build_gui_context()` — injects model name/routes/tools/web/RAG status/memory commands/emotional state tone cue), CMD activity pipeline, GGUF param estimation, RAG index build/toggle (wires `engine._rag_index` on every set/clear). Inherits LogicChatMixin + LogicMediaMixin |
| gui_logic_chat.py | Chat messaging, session management, AI session naming, session rename/delete (inline confirmation), duplicate save prevention, typewriter, file attachment, chat input history (Up/Down recall), history, send guard, stop generation, message editing, auto fact extraction, reasoning display, token counter, background trainer feeding (learn-while-chatting with engagement scoring), emotional state injection/decay, proactive web research (auto_research — runs in background thread), thinking animation (tracked after() with cancel-on-restart), chat export (5 formats: MD/HTML/PDF/JSON/TXT), history restoration display, deferred output rendering, router training state sync (LogicChatMixin) |
| gui_logic_media.py | Media rendering, voice I/O (TTS queue worker with persistent thread + STT continuous listening), TTS via pyttsx3 (chunk-based, non-blocking, safe stop via word callback), text cleaning for TTS (markdown/code stripping), inline media rendering with image cap, GIF animation (frame cycling with duration tracking), video thumbnails (play button overlay), clickable URLs (unique tag IDs), markdown image syntax support, chat input auto-resize (LogicMediaMixin) |
| gui_forge.py | Forge hub: training setup, 13-mode dispatch (Pre-Train/Distill/Basic/Image/AI-Guided/Dialogue/RLHF/Self-Play/GRPO/ReMax/SimPO/ORPO/APO), quick profile fields, training brief persistence (build/save/load with auto-save + **save on app close**; covers all 30+ Forge widget vars including distill examples/tokens, reasoning/evolutionary/auto-train/resume checkboxes, general mix, training stage, replay capacity/ratio, vision dir/preset, AI supplement path, quantize mode, GGUF export mode, guided pairs, web learn pages, vocab size), knowledge preservation (general mix), validation split, RL params, trainer/student system prompt builders (size-aware, stage-specific curriculum, training brief injection), mode-based section visibility, stage button dynamic coloring, format training pair helper (stage-appropriate formats), build generation prompt with CoT reasoning support, extract prompts from multiple formats (PDF/DOCX/JSONL/Q&A/User-AI), tokenizer training, model param counting (multi-format), button state management (stop signals Trainer.request_stop() with "STOPPING..." feedback, all 13 finally blocks + 1 watchdog reset to "STOP"), unified training dispatcher, log file rotation keeps last 100 forge logs (logs/forge_YYYYMMDD_HHMMSS.log). Inherits ForgeTrainingMixin + ForgeAdvancedMixin + ForgeAdaptiveMixin + ForgeNewModesMixin + ForgeToolsMixin + ForgeModelsMixin + ForgeQueueMixin |
| gui_forge_training.py | Basic training modes: solo, DPO (JSONL preference pairs), vision (encoder presets small/medium/large with state saving), LoRA (adapter saving, partial layer freezing fallback, auto-detect large models for LoRA), CPU-first student loading, HuggingFace model/tokenizer loading with weight mapping, before/after perplexity evaluation, general knowledge mix, loss curves, batch-level ETA (Step X/total | loss | lr | tok/s | VRAM | ETA Xh XXm), epoch-level ETA (ForgeTrainingMixin) |
| gui_forge_advanced.py | Advanced training: 3-phase guided training (curriculum generation → student training → readiness assessment with 1-10 scoring + stage advancement recommendations), curriculum file saving to data/ for DOCS review, dialogue training (TRAINER↔STUDENT multi-turn conversation with history tracking + conversation improvement tracking (first/second half comparison) + AI reinforcement on high scores ≥8/10 + corrections + transcript saving), engagement scoring for background trainer (ForgeAdvancedMixin) |
| gui_forge_adaptive.py | Adaptive pipeline: TrainingPlan JSON persistence (save/load with resume-on-crash), 3-phase adaptive loop (generate → train → test), accumulated curriculum per stage, adaptive difficulty adjustment, test score parsing (robust multi-format), difficulty probing with thresholds, stage advancement (retry/advance/complete), deduplication across retries, quality filtering (validate + clean examples), training report card (letter grades A-F with verdict + loss trend analysis + per-stage breakdown + next-step recommendations), plan status tracking (running/paused/completed/failed), loss trend detection (ForgeAdaptiveMixin) |
| gui_forge_new_modes.py | Training modes: Pre-Train (from-scratch training on existing model, BPE tokenizer retraining on 500K chunks with vocab validation 256-100000, corpus processing, perplexity tracking, elapsed time per phase, RAM feedback, load progress callback, RAM warning at 80% during data load phases, NaN/Inf loss detection with early return, training heartbeat file (logs/training_heartbeat.json written every 30s with pid/status/model/phase/step/loss/timestamp — survives OOM kills), stale session detection on launch (warns if previous run didn't exit cleanly), phase announcements ("=== Phase X/5: Name ===" with expected quirks/duration/behavior before each of 5 phases: data loading, tokenizer training, model loading, sequence streaming, training)), Distill (6-category teacher→student knowledge distillation with category-specific seed prompts + prompt variation for diversity + multi-retry generation + configurable examples/tokens + reasoning chain support), RLHF (2-phase: reward model → PPO), Self-Play (TRAINER as reward with SelfPlayConfig), GRPO (delegates to _start_rl_variant_training — reward model + GRPOTrainer), ReMax (delegates to _start_rl_variant_training — reward model + ReMaxTrainer), SimPO (delegates to _start_preference_variant_training — trainer.train_simpo), ORPO (delegates to _start_preference_variant_training — trainer.train_orpo), APO (delegates to refactored _start_dpo_training with loss_type="apo_zero" — trainer.train_dpo with anchored-zero loss), teacher system prompt builder, data accumulation to curated dataset, stage-aware training, data validation (min 100 chars) (ForgeNewModesMixin) |
| gui_forge_tools.py | Forge tools: data gen (mode-aware routing + curated dataset auto-accumulate + reasoning flag), evaluate, coherence benchmark (0.7 threshold for monologue quality gate), web learn (DuckDuckGo search → fetch → 800-char chunking → TRAINER generates Q/A pairs + topic sanitization + curated dataset auto-accumulate), checkpoints (organized models/checkpoints/ structure, save uses _active_trainer for live weights during training or shutil.copy2 when idle), tokenizer training, cards (model identity card with training history), auto-train (data selection + start), forge param count display, loss chart (ASCII bar chart + canvas line chart with grid lines + moving average overlay + step labels), command policy DPO generator (parse commands_reference.md → chosen/rejected JSONL pairs), training history JSON (persist/display past runs with perplexity), hyperparameter presets (Quick: 3 epochs/lr=0.0001/batch=auto, Balanced: 10 epochs/lr=0.00005/batch=auto, Thorough: 30 epochs/lr=0.00002/batch=auto), HuggingFace download progress, progress percentage label (ForgeToolsMixin) |
| gui_forge_models.py | Model ops: import (file dialog + directory detection), create (with preset_name persistence to ModelContext), copy (directory-based for sharded models), rename (with model context dir rename + route assignment update + case-only handling on Windows), delete, progressive model growing (Net2Net-style zero-init with compatible preset estimation + grow dialog + weight transfer by min dimensions + preset_name persistence), quantize (dynamic/int8/int4 modes), GGUF export (F16/Q8_0/Q4_0 types), HuggingFace download (progress callback + directory vs file detection), operation busy guard (_model_op_busy), background thread safety (ForgeModelsMixin) |
| gui_forge_queue.py | Queue with pause/resume + job progress callbacks, overnight plan save/load with resume capability + plan-to-queue conversion, curated dataset GUI (review with pending entry summaries + approve all pending + dataset source tagging), dataset auto-accumulate helper (ForgeQueueMixin) |
| gui_mods.py | Mod subprocess lifecycle (start/stop/auto-start, _launch_mod), crash detection (1s timeout), stderr capture (UTF-8 safe), auto-launch on command send (lazy start), UI widget value gathering for command args, router mod command sending with args marshalling, mod page status updates (running/ready indicators) |
| gui_mod_page.py | Per-mod page builder from mod.json: dynamic UI rendering (text_input, text_area, number, dropdown, checkbox, button widgets with state persistence), rules rendering (up to 6 from mod.json), AI prompt display, dependencies list, settings summary, status indicators (StatusDot + RUNNING/READY labels), command arguments with type hints + required markers, clear output log button |
| gui_cmd_page.py | Dual-mode terminal: SYSTEM shell + ENGINE commands + AI ACCESS + activity monitor (_cmd_activity for thread-safe logging) + live status strip (5 labels: device/RAM%/VRAM%/model/uptime, 5s refresh) + 15 info commands (status/sysinfo/gpu/memory/models/routes/sessions/mods/data/uptime/profiles/emotions.show/emotions.reset + config/file/memory commands) + AI command execution ([CMD] block parsing with file operation confirmation) + command history (up/down) + command cancellation + CWD tracking in prompt + input context menu (cut/copy/paste/select-all) + tooltips + welcome screen (OS/CPU/RAM/GPU/CUDA/routes/counts) + multilingual routing (non-Latin input auto-routes to AI) |
| media.py | Chat media support: image/GIF/video detection, Pillow loading, URL image download (10MB cap), GIF download + frame extraction (20MB cap), GIF animation (frame cycling with duration), video thumbnails (OpenCV + play button overlay), URL detection, clickable links (unique tag IDs), markdown image syntax parsing, media segmentation (split_text_and_media), URL sanitization, path resolution with PROJECT_ROOT fallback, MAX_CHAT_IMAGES cap, MAX_CHAT_HISTORY cap (500), active GIF cap (5) |
| scanners.py | Filesystem scanning, config limits, ROUTE_KEYS, PATH_SETTINGS, editable path overrides (save/load path_settings.json), route assignment persistence (save/load route_assignments.json), path persistence, scan_docs, trainer docs, param counting (_peek_target_size fast path + SafeUnpickler for security + zipfile peek + _estimate_params_from_size file-size heuristic for >2GB models + sharded safetensors merging), target_size display, _format_param_count, vision data scanning (paired images+text, JSONL, video frame extraction via OpenCV) |
| themes.py | Color theme system: Theme frozen dataclass (21 fields), 4 presets (dark/midnight/carbon/solarized), load/save preference, theme API |

### Mixin Inheritance Order
```
EnigmaGUI(DocsPageMixin, ForgeMixin, ModMixin, ModPageMixin, CMDPageMixin, LogicMixin, PagesMixin, ctk.CTk)
```

ForgeMixin inheritance:
```
ForgeMixin(ForgeTrainingMixin, ForgeAdvancedMixin, ForgeAdaptiveMixin, ForgeNewModesMixin, ForgeToolsMixin, ForgeModelsMixin, ForgeQueueMixin)
```

---

## WINDOW

| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Window title | "ENIGMA ENGINE" in OS title bar | Identifies the app | desktop.py |
| Window size | 1440x900 default, 800x500 minimum | Sets the app dimensions. Resizable down to 800x500. Position and size saved to gui_settings.json on close, restored on launch (clamped to on-screen bounds) | desktop.py |
| Background color | Very dark black (#080808) | Base color behind everything | desktop.py |

---

## HEADER BAR (top strip across the window)

| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Header bar | Dark panel (#0e0e0e), 56px tall, 1px border bottom | Contains title and model status | desktop.py |
| "ENIGMA" title | Large bold bright text (#e8e8e8) | First half of app branding | desktop.py |
| " ENGINE" title | Large bold silver text (#8B95A5) | Second half of app branding | desktop.py |
| "1.1.0" | Tiny dim text (no "v" prefix) | Version number | desktop.py |
| Pin button | Small button (📌) in header, after nav toggle | Toggles always-on-top window mode. Visual feedback when pinned | desktop.py |
| Shortcuts button | Small button (?) next to pin button | Opens an inline dropdown overlay listing all keyboard shortcuts. Tooltip: "Keyboard shortcuts" | desktop.py |
| Status dot | Colored circle on LEFT of status text, with tooltip | Gray = no model, orange = loading, green = loaded, red = error. Tooltip explains colors | desktop.py |
| Model status label | Text like "NO MODEL" or "model_name // RTX 5090" | Shows current model state with actual GPU name (no brackets) | desktop.py |

---

## NAV RAIL (left sidebar, 170px wide)

| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Nav rail | Dark panel (#0e0e0e) with right border only | Contains page buttons and mod launchers. Collapsible via header toggle | desktop.py |
| Nav toggle | Arrow button (◀/▶) in header | Collapses nav to 0px (fully hidden) or expands to 170px with labels. Silver when expanded, dim when collapsed | desktop.py |
| CORE button | NavButton with left-edge accent bar | Switches to CORE page (chat) | desktop.py |
| CMD button | NavButton with left-edge accent bar | Switches to CMD page (command terminal) | desktop.py |
| DOCS button | NavButton with left-edge accent bar | Switches to DOCS page (documentation) | desktop.py |
| MODELS button | NavButton with left-edge accent bar | Switches to MODELS page (model files) | desktop.py |
| ROUTER button | NavButton with left-edge accent bar | Switches to ROUTER page (route assignments) | desktop.py |
| FORGE button | NavButton with left-edge accent bar | Switches to FORGE page (training) | desktop.py |
| CONFIG button | NavButton with left-edge accent bar | Switches to CONFIG page (settings) | desktop.py |
| Separator line | 1px horizontal rule | Divides nav pages from mods section | desktop.py |
| "MODS" label | Tiny dim text | Labels the mod section | desktop.py |
| Mod NavButtons | NavButton, one per mod | Switches to that mod's dedicated page | desktop.py |
| "(none)" | Tiny dim text | Shown if no mods found in mods/ folder | desktop.py |

**Nav button behavior:** Active button shows a 3px left-edge accent bar (silver) and bright text on surface background. Inactive buttons show dim text with no bar. Only one active at a time. No symbols or icons, no "NAV" label.

**Shortcuts overlay:** The ? button opens an inline `CTkFrame` dropdown (via `place()` inside the main window — not a Toplevel, so it's never hidden behind an always-on-top window) listing all keyboard shortcuts as key-combo / description rows. Dismissible via Close button or Escape key. Shortcuts listed:

| Key | Action |
|-----|--------|
| Ctrl + 1–7 | Switch pages (CORE → CONFIG) |
| Ctrl + N | New chat session |
| Escape | Stop generation |
| Shift + Return | Newline in chat input |
| Ctrl + Z | Undo (all text inputs) |
| Ctrl + Y | Redo (all text inputs) |
| Ctrl + A | Select all (all text inputs) |
| Ctrl + S | Save (docs editor) |
| Ctrl + F | Find (docs editor) |
| Up / Down | Command history (CMD page) |

**Nav collapse:** Click the ◀ arrow button in the header to collapse the nav rail to 0px (fully hidden via grid_remove). Content expands to fill the full width. Click ▶ to restore full 170px with labels.

**Mod behavior:** Each mod is a full page. All mods auto-start when the app launches. Mod pages show info, commands, UI widgets from mod.json, and an output log.

---

## PAGE: CMD (Dual-Mode Terminal)

Dual-mode terminal with SYSTEM (real PowerShell) and ENGINE (AI command registry) modes.
AI ACCESS toggle lets the AI execute real system commands when enabled.
Also serves as an activity monitor — all AI operations (chat [CMD] blocks, command execution) are logged here via `_cmd_activity()`. When the AI uses chain-of-thought reasoning, the full `<think>` content is logged here (with `🧠 Reasoning:` header) followed by the answer, so users can monitor the AI's thought process without cluttering the chat page.
Rich welcome screen shows system info, loaded model, routes, and asset counts on open.
Live status strip auto-refreshes every 5 seconds with device, RAM, VRAM, model, and uptime.

### Top Bar
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| "Terminal" header | SectionLabel with cyan accent | Page title | gui_cmd_page.py |
| SYSTEM/ENGINE toggle | CTkSegmentedButton | Switches between real shell and engine command modes | gui_cmd_page.py |
| CLEAR button | Small dark button | Clears the terminal output | gui_cmd_page.py |
| AI ACCESS label | Tiny text, dim or green | Labels the AI ACCESS toggle | gui_cmd_page.py |
| AI ACCESS button | Toggle button (ON/OFF) | When ON (green), AI can run real system commands. When OFF (dim), AI restricted to engine commands only | gui_cmd_page.py |

### Terminal Output
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Output display | CTkTextbox in HUDFrame, green text on dark bg | Shows command output with color-coded tags | gui_cmd_page.py |

### Status Strip
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Device label | Tiny text (CPU/GPU name) | Shows compute device (actual GPU name like RTX 5090) | gui_cmd_page.py |
| RAM label | Tiny text (used/total) | Live RAM usage, updates every 5s | gui_cmd_page.py |
| GPU label | Tiny text (used/total) | Live VRAM usage if GPU present, updates every 5s | gui_cmd_page.py |
| Model label | Tiny text, green when loaded | Shows loaded model name or "No model" | gui_cmd_page.py |
| Uptime label | Tiny text | Session uptime in hours/minutes | gui_cmd_page.py |

### Input Line
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Prompt label | Dynamic cyan text ("PS dir>" or "ENG>") | Changes based on active mode and current directory | gui_cmd_page.py |
| Command input | CTkEntry with dynamic placeholder | Placeholder changes per mode. Type commands, Enter to execute | gui_cmd_page.py |
| RUN button | Green button | Executes the command in the input field | gui_cmd_page.py |

**Modes:**
| Mode | Prompt | What It Does |
|------|--------|--------------|
| SYSTEM | PS dir> | Runs real PowerShell commands. Tracks CWD changes. Supports internet, programs, everything the OS can do |
| ENGINE | ENG> | Runs AI engine commands (config.*, file.*, model.*, etc.). Supports "ask \<question\>" to query the AI |

**AI ACCESS toggle:**
- **OFF** (default): AI can only run engine commands. Unknown commands show an error with hint to enable AI ACCESS.
- **ON** (green): AI-generated commands that aren't recognized by the engine registry get forwarded to real PowerShell. The AI can install packages, run scripts, access the internet, etc.

**Terminal color tags:**
| Tag | Color | Used For |
|-----|-------|----------|
| prompt | Cyan (#22d3ee) | The PS> or ENG> prompt |
| command | Bright text (#e8e8e8) | User's typed command |
| output | Green (#22c55e) | Successful command output |
| error | Red (#ef4444) | Error messages |
| info | Dim text (#555555) | System messages, mode switches, help text |
| ai_output | Orange (#f97316) | AI-generated text responses |
| activity | Dim text (#555555) | AI activity logged from other pages (chat, training) |
| divider | Border accent (#2e2e2e) | Spacing dividers |

**ENGINE mode commands:**
| Command | Description |
|---------|-------------|
| help | Show all engine commands and terminal info |
| clear / cls | Clear the terminal output (works in both modes) |
| history | Show command history |
| ask \<question\> | Send question to AI, auto-execute any [CMD] blocks in response |
| status | Show loaded model, architecture, routes, generation state, uptime |
| sysinfo | Show OS, Python, CPU, RAM, hardware type, PyTorch version |
| gpu | Show per-GPU VRAM usage with visual bar chart |
| memory | Show RAM + VRAM usage with visual bar charts |
| models | List all models grouped by native/external with sizes |
| routes | Show route assignments with missing-file detection + unassigned routes |
| sessions | List saved chat sessions with message counts |
| mods | List installed mods with running status, ports, commands |
| data | List training data files with sizes |
| uptime | Show session uptime |
| profiles | List AI profiles |
| config.get/set/list | Get, set, or list config values |
| model.info/list | Model information and listing |
| file.read/write/list | File operations |
| memory.remember \<fact\> | Save a fact to persistent AI memory |
| memory.forget \<keyword\> | Remove a fact from persistent memory |
| memory.notes | Show all remembered facts |
| memory.clear_notes | Clear all persistent memories |
| memory.search \<query\> | Search remembered facts by keyword/topic |
| emotions.show | Show the AI's current emotional state (valence, energy, engagement, trust, frustration) with visual bars |
| emotions.reset | Reset emotional state to neutral baseline |
| system.info | System information |
| (all engine commands) | Full command registry available |

**AI command execution flow:**
1. AI responds to "ask" with text and optional [CMD] blocks
2. Each [CMD] block is tried as an engine command first
3. If unknown and AI ACCESS is ON, the command runs as a real system command
4. If unknown and AI ACCESS is OFF, an error is shown

**Keyboard:** Up/Down arrows navigate command history. Enter executes. Ctrl+Z/Y undo/redo, Ctrl+A select all in command input.

---

## MOD PAGES (one per mod, built from mod.json)

Each mod gets its own page, dynamically built from the mod's `mod.json` config.

### Top Bar
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Mod name header | Section label | Page title (mod name) | gui_mod_page.py |
| START button | Silver accent button | Starts the mod subprocess | gui_mod_page.py |
| STOP button | Dark button, disabled until running | Stops the mod subprocess | gui_mod_page.py |
| Status dot | Colored circle | Gray = stopped, green = running | gui_mod_page.py |
| Status label | Text "RUNNING" or "STOPPED" | Shows mod process state | gui_mod_page.py |

### Left Column: Info + Commands + Interface
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Info card | HUDFrame | Shows mod name, description, version, port. Also shows dependencies, up to 4 settings keys, and AI usage prompt if defined | gui_mod_page.py |
| Commands list | One row per command | Shows command name (silver) and description (dim) from mod.json. Also shows argument details (type, required, description) per command | gui_mod_page.py |
| Interface card | HUDFrame with accent border | Renders UI widgets from mod.json "ui" section | gui_mod_page.py |
| text_input widgets | CTkEntry | Text input fields defined in mod.json | gui_mod_page.py |
| text_area widgets | CTkTextbox | Multi-line text areas defined in mod.json | gui_mod_page.py |
| number widgets | CTkEntry (numeric) | Number inputs with defaults from mod.json | gui_mod_page.py |
| dropdown widgets | themed_dropdown | Dropdown menus with options and default value from mod.json | gui_mod_page.py |
| checkbox widgets | CTkCheckBox | Boolean toggle inputs from mod.json | gui_mod_page.py |
| button widgets | CTkButton (silver accent) | Sends the mapped command to the mod | gui_mod_page.py |

### Right Column: Output Log
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Output header | Green section label | Labels the log | gui_mod_page.py |
| Output log | CTkTextbox, green text | Shows mod start/stop events, command sends, responses | gui_mod_page.py |

---

## STATUS BAR (bottom strip across the window)

| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Status bar | 30px tall strip at very bottom | Three-section info bar | desktop.py |
| Left section | Text like "READY" or "MODEL_NAME LOADED" | Shows current state | desktop.py |
| Center section | Text like "CPU" or "RTX 5090 // 31.8 GB VRAM" | Shows compute device with actual GPU name | desktop.py |
| Right section | Text like "UPTIME 00:05:23" | Shows app uptime, updates every second | desktop.py |

---

## PAGE: CORE (Chat Interface)

### Top Bar
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| "NEURAL INTERFACE" header | Section label with accent line | Page title | gui_pages.py |

| Fullscreen toggle | Small button (\u26f6 icon) | Enters fullscreen chat — hides header, nav, status bar. CORE page covers the entire GUI. Dim when normal, accent when active | gui_pages.py |
| Sidebar toggle | Small button (\u25e8 icon) | Hides or shows the sidebar. When hidden, chat expands to full width. Silver when visible, dim when hidden | gui_pages.py |

### Chat Area (left column)
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Chat display | SelectableTextbox (CTkTextbox subclass) with native scrollbar, word wrap, 12px left/right margins | Shows conversation: purple for YOU, silver for ENIGMA, orange for SYSTEM (with timestamps), red for errors. Chain-of-thought reasoning is not shown here — it is logged to the CMD page instead. Native scrollbar handles all scrolling — no wrapper frame needed. Fills available space via sticky="nsew" | gui_pages.py |
| File indicator | Tiny cyan text above input | Shows attached filename when a file is attached | gui_pages.py |
| Thinking indicator | Tiny dim text, right side, fixed 140px width | Shows "PROCESSING..." with animated dots while AI generates response. Layout stability is enforced at two layers: `input_area.grid_columnconfigure(1, minsize=140)` locks the control column width, and fixed-size `SelectableLabel` instances do not resize on text updates. This prevents residual jitter during animation | gui_pages.py, widgets.py |
| Chat input | Multi-line text box, 56px default | Type messages here. Enter sends. Shift+Enter for newline. Ctrl+Z/Y undo/redo, Ctrl+A select all. Blocked during generation. Auto-expands from 56px to 200px as content grows, resets to 56px after sending | gui_pages.py |
| SEND button | Green button, right of input | Sends the message (or Enter key). Hidden during generation. Tooltip: "Send message (Enter)" | gui_pages.py |
| STOP button | Red button, same slot as SEND | Shown during generation. Stops AI mid-response. Also Escape key. Tooltip: "Stop AI generation" | gui_pages.py |
| Token counter | SelectableLabel on right side of toolbar | Shows current conversation token count (e.g. "128 tokens"). Updates on page show and new chat | gui_pages.py |
| In-memory history cap | Logic behavior (not a direct widget) | Caps `self.history` at `MAX_CHAT_HISTORY = 500` via `_trim_chat_history()` to prevent RAM growth. Full session still auto-saves to disk in `memory/session_*.json` | gui_logic_chat.py |
| History summary | Logic behavior (not a direct widget) | When chat history is truncated to fit context, dropped messages are summarized by `_summarize_dropped_history()` into a compact topic list. The summary is injected into the system prompt by `_prepare_chat()` so the AI retains awareness of earlier conversation. Persisted as `history_summary` field in session JSON — restored on load, cleared on new chat | engine_chat.py, gui_logic_chat.py |
| Utility toolbar | Row below input | Left side: attach, new, web toggle, reasoning toggle, RAG toggle, edit. Right side: voice, mic — separated from SEND to prevent misclicks | gui_pages.py |
| Attach button | Square button in toolbar (left) | Opens file picker to attach a text file to next message. Tooltip: "Attach file" | gui_pages.py |
| NEW button | Dark button in toolbar (left) | Starts a new conversation: clears chat, history, and KV cache. No confirmation needed — current chat auto-saves | gui_pages.py |
| Web access toggle | ToggleButton (🌐 icon) in toolbar (left) | Toggles AI web access on/off. When ON (cyan), AI can search the web via DuckDuckGo. Flag injected into _build_gui_context() system prompt. Tooltip: "Web access" | gui_pages.py |
| Reasoning toggle | ToggleButton (🧠 icon) in toolbar (left) | Toggles chain-of-thought reasoning on/off. When ON, AI uses step-by-step reasoning (`<think>` blocks) before answering. Sets `self.reasoning_enabled` flag. Tooltip: "Reasoning" | gui_pages.py |
| RAG toggle | ToggleButton (📚 icon) in toolbar (left) | Toggles document Q&A on/off. When ON, indexes `data/` and `information/` directories in a background thread via `RAGIndex`, then uses retrieved context to answer questions. Tooltip: "Document Q&A" | gui_pages.py |
| Edit button | Square button (✎ icon) in toolbar (left) | Edits last sent message: removes last exchange, puts user text back in input. Blocked during generation. Tooltip: "Edit last message" | gui_pages.py |
| Voice toggle | Square toggle button in toolbar (right) | Turns voice output (TTS) on/off. When ON (green), AI responses are spoken aloud via pyttsx3 persistent worker thread. Toggling OFF stops any in-progress speech. Tooltip: "Voice output on/off" | gui_pages.py |
| Mic button | Square button (🎤 icon) in toolbar (right) | Voice input: click to start continuous listening, each recognized phrase auto-sends as a chat message. Click again to stop. Uses listen_in_background() with stopper. Turns red while recording. Tooltip: "Voice input (mic)" | gui_pages.py |

### Sidebar (right column, resizable via PanedWindow)

The sidebar contains four **collapsible panels** (CollapsiblePanel widget). Click the header to expand/collapse. When panels are collapsed, expanded ones take the available space. When all are collapsed, only the header rows are visible. The chat/sidebar boundary is a **draggable sash** (tk.PanedWindow) — users can resize by dragging.

| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| HISTORY panel header | CollapsiblePanel, purple chevron + title | Click to expand/collapse the history section | gui_pages.py |
| History list | Text box with word wrap (inside panel) | Shows saved sessions: AI-generated title (or timestamp fallback), message count, date. Click a session to load it. Hover highlights in purple. Switching sessions syncs model context | gui_pages.py |
| SAVE button | Small white button | Saves current chat as a session JSON to memory/ | gui_pages.py |
| LOAD button | Small dark button | Opens file picker to load a saved session JSON | gui_pages.py |
| DELETE button | Small dark button | Shows inline red bar: "Delete? [YES] [NO]". YES deletes the selected session file, NO cancels | gui_pages.py |
| EXPORT button | Small dark button | Exports chat as plain .txt file | gui_pages.py |
| SYSTEM PROMPT panel header | CollapsiblePanel, silver chevron + title | Click to expand/collapse the prompt section | gui_pages.py |
| Prompt editor | Text box (inside panel) | Edit the system prompt that shapes AI behavior | gui_pages.py |
| APPLY button | Small silver accent button | Applies the edited system prompt to current engine | gui_pages.py |
| RESET button | Small dark button | Resets prompt to default from prompts.json | gui_pages.py |
| EMOTIONAL STATE panel header | CollapsiblePanel, dim chevron + title | Click to expand/collapse the emotional state display | gui_pages.py |
| Emotional state bars | Visual bars inside panel | Shows 5 dimensions: valence, energy, engagement, trust, frustration (0.0–1.0 each). Each row has a label (auto-width, aligned via `minsize=110`), an orange progress bar, and a numeric value. Hover any row to see a description tooltip (e.g. "Negative ↔ Positive", "Calm ↔ Energized"). Hidden when "Show emotional state panel" is unchecked on CONFIG. AI still uses emotional data internally | gui_pages.py |
| JOURNAL panel header | CollapsiblePanel, purple chevron + title | Click to expand/collapse the journal section | gui_pages.py |
| Journal display | SelectableTextbox (inside panel) | Shows the last 5 journal entries (timestamp + text). Populated by `_refresh_journal_display()` which reads from the model's `Journal` instance. Empty until monologue mode is enabled and reflections pass the coherence gate | gui_pages.py |

**Sidebar toggle:** Click the ◨ button in the top bar to hide the entire sidebar (history + system prompt + emotional state + journal). The chat area expands to full width. Click again to restore the sidebar. All collapsible panels retain their state when the sidebar is toggled.

**Collapsible behavior:**
- History and System Prompt start expanded by default; Emotional State and Journal start collapsed
- Emotional State panel has fixed height (no vertical stretching). Can be hidden entirely via CONFIG toggle.
- Click a panel header (chevron + title) to collapse it — remaining expanded panels share the space
- Click again to re-expand — all expanded panels share space equally
- Chevron indicator: ▼ = expanded, ▶ = collapsed
- When all are collapsed, only the thin header rows are visible

**STOP button behavior:**
- SEND button is hidden during generation, replaced by red STOP button in the same grid slot
- Clicking STOP sets `_stop_requested` flag, halts the typewriter animation mid-stream, and shows a SYSTEM message
- Escape key also triggers stop when generation is active
- After stopping, SEND button is restored and input is re-enabled

**Edit button behavior:**
- Removes the last user+assistant message pair from history
- Puts the user's message text back into the input box for editing
- Redisplays the remaining history and auto-saves
- Blocked during active generation and when history is empty

**Send guard:**
- `_is_generating` flag prevents double-send — Enter key and SEND button both check this flag
- Concurrent generation threads are impossible — second send attempt is silently ignored

**Voice output (TTS) behavior:**
- Uses pyttsx3 with a persistent worker thread and Queue — engine initialized once on the worker thread
- `_tts_speak(text)` lazily creates the worker thread on first call, then enqueues text
- Worker thread loops reading from queue, speaks each utterance, auto-recovers on engine errors
- Worker registers a `started-word` callback — checks `_tts_stop_event` each word, calls `engine.stop()` from the **worker thread** (same thread as engine) to avoid cross-thread COM crashes
- `_tts_stop()` sets `_tts_stop_event` — never calls `engine.stop()` directly (SAPI5 COM has thread affinity, cross-thread calls crash on Windows)
- Toggling voice OFF stops any in-progress speech; STOP generation also stops speech
- `_tts_shutdown()` sends poison pill (`None`) to queue for clean exit on window close

**Voice input (mic) behavior:**
- Click mic button to start continuous listening via `listen_in_background()` with stopper callable
- Each recognized phrase triggers `_on_voice_text()` which auto-sends it as a chat message
- Mic stays active between phrases — works like a conversation, not one-shot
- Click mic again (or generation starts) to stop listening via stopper callable
- Button turns red while recording, resets to normal when stopped

---

## PAGE: MODELS (Model Management)

### Top Bar
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| "MODELS" header | Section label | Page title | gui_pages.py |

### Create Form
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Name entry | Text input | Name for the new model | gui_pages.py |
| Memory (GB) entry | Numeric input (int, width 60) | Available memory budget in GB. Auto-detects GPU VRAM (or system RAM). `recommend_preset_for_vram()` picks the largest architecture that fits. Default: auto-detected. Tooltip: "Available memory for training (GB)" | gui_pages.py |
| CREATE button | Silver accent button | Creates a blank untrained model sized to the memory budget. Feedback shown inline and in status bar. Tooltip: "Create a new empty model" | gui_pages.py, gui_forge_models.py |
| IMPORT button | Silver accent button | Opens file picker to import an external model (.gguf, .bin, .safetensors, .pth, .pt). Copies to models/ directory. Tooltip: "Import a model file from disk" | gui_pages.py |
| DOWNLOAD button | Silver accent button | Downloads a model from HuggingFace. Reads the repo ID from the inline HF entry field. Tooltip: "Download a model from HuggingFace" | gui_pages.py, gui_forge_models.py |
| HF repo entry | Text input (width 260) with placeholder "e.g. gpt2 or username/model-name" | Inline entry for HuggingFace repo IDs. Press Enter or click DOWNLOAD to start download. Replaces the old dialog prompt | gui_pages.py |
| Status label | Tiny text below form | Shows create/delete/copy/rename/import/download feedback: white for info, green for success, red for errors. Also updates the bottom status bar | gui_pages.py |

### Model Cards (scrollable, multi-row layout with identity)

Each model gets a card with identity info, format details, and action buttons. Identity data is loaded from the model's context directory.

| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Model name + params | Editable entry (row 0, left) | Shows identity display_name if set, otherwise file name. Param count appended for native models. Read-only by default; becomes editable when EDIT or right-click Rename is used | gui_pages.py |
| EDIT button | Silver accent button (row 0, right) | Makes the name entry editable inline with orange border. Shows SAVE/CANCEL buttons (row 3). Enter saves, Escape cancels. Saves as display name in model context. Tooltip: "Edit identity card" | gui_pages.py |
| EXPORT CARD button | Dark button (row 0, right) | Exports identity card as a standalone JSON file. Tooltip: "Export identity card to JSON" | gui_pages.py |
| COPY button | Dark button (row 0, right) | Creates a copy of the model file with "_copy" suffix. Shows → arrow feedback. Tooltip: "Duplicate this model" | gui_pages.py |
| GROW button | Green accent button (row 0, right, native only) | Progressive growing — expand model to a larger size preset. Shows inline preset picker (dropdown + GROW/CANCEL). Creates new model file with expanded weights. Only shown for native .pth/.pt models. Tooltip: "Expand this model to a larger size (progressive growing)" | gui_pages.py, gui_forge_models.py |
| DELETE button | Dark button, hover red (row 0, right) | Shows inline red delete bar (row 4): "Delete model_name? [YES] [NO]". YES confirms deletion, NO cancels. Tooltip: "Permanently delete this model" | gui_pages.py |
| NATIVE/EXTERNAL tag | Colored label (row 1, left) | Green "NATIVE" for .pth/.pt models, orange "EXTERNAL" for .gguf/.bin/.safetensors | gui_pages.py |
| Format info | Tiny dim text (row 1, after tag) | Shows "PTH // 48M params // ~91 MB RAM // xl" — format, param count, estimated RAM, and preset name (if saved in model context) | gui_pages.py |
| File name subtitle | Tiny dim text (row 1, after format) | Shows original file name in parens when identity display_name differs | gui_pages.py |
| Personality | Normal dim text (row 2) | Short personality description from identity card | gui_pages.py |
| Stats line | Tiny dim text (row 2) | Message count, session count, training run count | gui_pages.py |
| Tags | Tiny accent text (row 2) | User-defined tags displayed as [tag1] [tag2] | gui_pages.py |
| Edit row | Hidden frame (row 3) | SAVE and CANCEL buttons for inline name editing. Only visible when editing | gui_pages.py |
| Delete bar | Hidden frame (row 4) | Red inline bar: "Delete model? [YES] [NO]". Only visible when delete is pending | gui_pages.py |

**Model card layout:**
- **Row 0:** Model name entry (identity name or file name + param count) on left, EDIT / EXPORT / COPY / DELETE buttons on right
- **Row 1:** NATIVE or EXTERNAL tag (color-coded) + format and file size + file name subtitle
- **Row 2 (if identity exists):** Personality, stats (messages/sessions/training runs), tags
- **Row 3 (hidden):** SAVE / CANCEL buttons for inline name editing (shown when EDIT or Rename is active)
- **Row 4 (hidden):** Red delete confirmation bar (shown when DELETE is clicked)
- **Row 5 (if compatible LoRA adapters exist):** LoRA adapter section — see below

**LoRA adapter section (per-card, Pass 156t + Pass 156u-B):** When `scan_lora_adapters(model_path)` finds adapters whose recorded base matches this card's base stem, this section renders inside the card body. **Header row:** `LoRA Adapters: N available` label + (when active) `active: <name>` indicator + `CLEAR` button (clears the active single-adapter route). **Per-adapter rows:** stack-selection `CTkCheckBox` (col 0) + `themed_numeric_entry(mode="float", width=56)` weight entry (col 1, default `"1.0"`, accepts negatives + scientific notation per Dia "no sliders" rule) + adapter name with rank/alpha (col 2) + `APPLY` button (col 3, hidden on the active row). **Bottom button:** `APPLY STACK` rendered when ≥2 adapters exist — merges every ticked row into a single weighted PEFT stack via `EnigmaEngine.apply_adapter_stack`. Empty selection or 0 ticks → chat-system hint, no engine call. Single tick → routes through `_set_chat_adapter` (skips the `_stack` PEFT indirection on the trivial case). Parse error in any weight entry (non-numeric, NaN, Inf) → chat-error per row, all-or-nothing — partial stacks never reach the engine. Apply / Apply Stack / Clear are gated on the model being currently loaded; cross-base clicks surface a chat hint instead of attempting a load.

**Chat session markers (Pass 156v Step 1 + Step 2):** Successful runtime state changes render a divider line in the chat log via the `session_marker` text tag — dim foreground, smaller font, extra vertical breathing room. Step 1 covers LoRA adapter changes: `─── LoRA adapter: foo_lora ───`, `─── LoRA cleared — using base weights ───`, `─── LoRA stack: foo@0.70, bar@0.30 ───`. Step 2 extends to model + RAG seams: `─── Model: ENIGMA_SMALL (742,000,000 params, RTX 5090) ───` on load, `─── Model unloaded — no model active ───` on unload, `─── Document Q&A enabled — N chunks from M files ───` after a successful corpus build, `─── Document Q&A disabled — no corpus active ───` on toggle off. Visually distinct from regular orange `system_msg` lines so the user can scan the chat log and locate the exact point where state changed if answer quality regresses afterwards. Errors, load-first hints, and in-flight progress (`Building document index...`) continue to render as regular system/error messages (unchanged contract). Profile swap and system-prompt edit do NOT render a marker because there is no chat-page surface that swaps them today.

**Right-click context menu:** Right-clicking any model card shows a tk.Menu with two options:
- **Rename file** — Makes the name entry editable with orange border (same as EDIT but tags the entry as a file rename). On Save, renames the actual file on disk, updates route assignments, and renames the model context directory.
- **Delete** — Same as clicking the DELETE button (shows inline red delete bar).
Bound to `<Button-3>` on the card frame, inner frame, and name entry.

**Identity card:** Each model can have an identity — display name, personality, tags, notes, and auto-tracked stats (total messages, sessions, training history). Identity data is stored in `data/model_contexts/<model_key>/context.json` and loaded when building model cards.

**EDIT behavior:** Makes the name entry editable inline — shows an orange border and grids SAVE/CANCEL buttons at row 3. Enter saves, Escape cancels. Saves the new name as a display name in the model's context.json. Other identity fields (personality, tags, notes) are edited directly in the model context files via the DOCS page.

**EXPORT behavior:** Opens a save dialog to export the model's identity card as a standalone `<key>_identity.json` file containing display name, personality, avatar, stats, training history, tags, notes, and memory fact count.

**Param count:** Native models (.pth/.pt) show their parameter count next to the name (e.g. "19.08B", "1.5B", "500.0M"). Computed from the state dict at scan time. External models show name only.

**COPY behavior:** Creates `<name>_copy.<ext>` in the same directory. Shows feedback like "model.pth → model_copy.pth" with green success message. For directory-based models (HuggingFace sharded), copies the entire directory.

**Operation guard:** All heavy model operations (copy, import, create, delete) are protected by a shared `_model_op_in_progress` flag via `_model_op_busy()`. Only one operation can run at a time — additional clicks show an orange warning. The flag always resets via `finally` block.

**Sharded model display:** HuggingFace models split across multiple safetensors files (e.g. model-00001-of-00005.safetensors) are grouped into a single model card showing the combined size and shard count.

**Rename (via right-click):** Makes the name entry editable with orange border and tags it as a file rename. On Save, sanitizes input (alphanumeric + underscore only). If the model is assigned to any route, the route assignment is updated automatically. Unloads the model if it was currently loaded. The model's context directory (chat history, system prompt, config overrides) is also renamed to follow the new name. Handles case-only renames on Windows via a temp file to work around case-insensitive filesystem.

**IMPORT behavior:** Opens a file dialog filtered for model files (.gguf, .bin, .safetensors, .pth, .pt). Copies the selected file to the models/ directory. Shows progress and refreshes the model card list.

**Weight transfer:** `_transfer_weights()` copies weights between different-sized models by using the minimum dimensions for each tensor. Learned features are preserved wherever source and destination dimensions overlap.

---

## PAGE: ROUTER (Route Assignments)

### Top Bar
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| "ROUTER" header | Section label | Page title | gui_pages.py |
| SUSPEND button | Button on right, disabled until model loaded | Suspends model memory while preserving chat route. Saves context, releases engine, sets header to "SUSPENDED" (orange). Button text changes to "RESUME" to reload the model. Blocked during generation or training | gui_pages.py |
| UNLOAD button | Small button on right, disabled until model loaded | Unloads current model, frees memory | gui_pages.py |

### Route Connection Cards
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| CHAT route card | HUDFrame with dot, name, description, dropdown, status | Assign a model to handle conversations | gui_pages.py |
| TRAINER route card | HUDFrame with dot, name, description, dropdown, status | Assign a model to handle training and evaluation | gui_pages.py |
| STUDENT route card | HUDFrame with dot, name, description, dropdown, status | Assign the AI model being trained and evaluated | gui_pages.py |
| Mod route cards | One per mod, auto-generated | Assign a model to each mod independently | gui_pages.py |
| Route status label | Text on right of each card | Shows assigned model name (green), "Running" for mods, or "No model" (dim) | gui_pages.py |
| Route status dot | Colored circle on left of card | Green = model assigned or running, orange = model assigned but mod stopped, gray = nothing | gui_pages.py |
| Model dropdown | CTkOptionMenu per route card | Select which model to assign to this route (None clears it) | gui_pages.py |

**Route behavior:** Each route (CHAT, TRAINER, STUDENT, and each mod) has its own model dropdown. Selecting a model from the CHAT dropdown loads it into the engine. Selecting "None" unloads it. Non-chat routes share the chat engine if assigned the same model (`_get_engine_for_route()`). The STUDENT route is the model being trained — fine-tune in FORGE trains the STUDENT model while TRAINER can evaluate it. During FORGE operations (Evaluate, Adaptive Phase 3, AI-Assisted Phase 3, Dialogue training), STUDENT receives a lean persona prompt via `_build_student_system_prompt()` (identity + behavioral guidance only — no training mechanics or scoring rubrics), while TRAINER gets the full mechanics prompt via `_build_trainer_system_prompt()`. Mod routes also show running/stopped state. All route statuses update live via `_update_route_status()` in gui_logic.py. Assignments are stored in `self.route_assignments` dict. Route changes also update the status bar for cross-page visibility.

---

## PAGE: FORGE (Training)

Resizable 2-column layout via tk.PanedWindow: controls on the left, output log on the right. Users can drag the sash to resize.

### Top Bar
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| "THE FORGE" header | Section label | Page title | gui_pages.py |

### Left Column: Controls

#### Assigned Models (status cards)
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| TRAINER card | HUDFrame with status dot, name, format/size | Shows which model is assigned as TRAINER. Green dot when assigned, dim when not | gui_pages.py |
| STUDENT card | HUDFrame with status dot, name, format/size, param count | Shows which model is assigned as STUDENT. Green dot when assigned, dim when not. After training, displays the updated parameter count | gui_pages.py |

Cards update live via `_update_forge_cards()` whenever route assignments change.

**Param count on STUDENT card:** After each training session completes, the STUDENT card updates to show the model's parameter count (e.g. "Parameters: 19.08M"). The count is cleared when a new training session starts and refreshed when training finishes, ensuring it always reflects the latest trained state. Uses `_update_forge_param_count()` and `_clear_forge_param_count()` from gui_forge.py.

#### Train (unified section)
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Training mode cards | 13 radio-card options (Foundation: `PRE-TRAIN`, `DISTILL`, `BASIC`, `IMAGE` | Advanced: `AI-GUIDED`, `DIALOGUE`, `RLHF`, `SELF-PLAY` | Alignment: `GRPO`, `REMAX`, `SIMPO`, `ORPO`, `APO`) | User chooses one clear path, then only relevant sections are shown | gui_pages_forge.py |
| Include reasoning checkbox | CTkCheckBox | When ON, AI-generated training data can include `<think>` reasoning chains | gui_pages_forge.py |
| Evolutionary selection checkbox | CTkCheckBox | Generate multiple candidate answers per task, score them, keep the best, train on winners. Visible for Basic, AI-Guided, RLHF, and Self-Play modes | gui_pages_forge.py |
| Basic data source dropdown | Option menu | For Basic mode training data. Supports `(none)` and file selection from `data/`. Default selection prefers `data/finetune/combined_finetune.txt` when present (shipped by `collect_finetuning_data.py --all`), else falls back to whatever `scan_training_data` returns first. Pass 156i9 / D-11b. | gui_pages_forge.py |
| **Auto-LoRA trigger** | Automatic detection | When training in Basic mode, detects STUDENT model param count. Auto-selects LoRA if > 7B params, full fine-tuning if ≤ 7B params. No user toggle needed — happens automatically at training start. Shows info log message about detected size and selected method | gui_forge.py |
| AI-guided topic/goal entry | Text input (required for AI-Guided) | Defines what the trainer should teach the student. If empty, training logs guidance and does not start | gui_pages_forge.py |
| AI-guided supplement data dropdown | Option menu (optional) | Optional seed data for AI-guided curriculum generation. Wired to `ai_supplement_var` and used by the adaptive backend/tool flows | gui_pages_forge.py |
| Training stage buttons | 4 CTkButtons (`BASICS/CONVERSATION/COMMANDS/WEB`) | Select the adaptive pipeline start stage. Runtime contract is “start here, then continue forward” | gui_pages_forge.py |
| Training brief panel | CollapsiblePanel with quick profile fields + custom text | Refines trainer instructions for AI-guided runs. Profile labels (Personality, Tone, etc.) auto-size to content | gui_pages_forge.py |
| Image data directory | Text input + Browse button | Folder with image-text pairs used by Image mode | gui_pages_forge.py |
| Encoder size dropdown | Option menu (`tiny/small/medium`) | Vision encoder size for Image mode | gui_pages_forge.py |
| Training preset dropdown | Option menu (`Quick/Balanced/Thorough/Custom`) | Pre-fills epochs, learning rate, and batch size | gui_pages_forge.py |
| Epochs entry | Text input, default `10` | Number of training passes | gui_pages_forge.py |
| Learning rate entry | Text input, default `0.00005` | Training learning rate | gui_pages_forge.py |
| Batch size entry | Text input, default "auto" | Training batch size. "auto" fills available GPU memory. Lower = less VRAM. Tooltip: "How many examples the model learns from at once. 'auto' = fill available GPU memory. Set to 1 if you get out-of-memory errors. Bigger = faster training, more VRAM needed." | gui_pages_forge.py |
| Grad accumulation entry | Text input, default "1" | Gradient accumulation steps | gui_pages.py |
| Gradient checkpointing | Checkbox | Saves VRAM by recomputing activations | gui_pages.py |
| Rolling best K entry | Text input, default "0" | Keep K best checkpoints by loss during training. 0 = disabled | gui_pages_forge.py |
| AI-guided pairs entry | Text input, default "20" | Number of generated examples per stage for AI-guided flow | gui_pages_forge.py |
| LoRA rank/alpha entries | Text inputs (`8`/`16`) | Advanced LoRA controls (used when LoRA path is selected in backend flow) | gui_pages_forge.py |
| Resume from checkpoint | CTkCheckBox | ON: Click TRAIN to continue from where you left off (loads latest checkpoint). OFF: Start training from scratch. Tooltip explains this clearly | gui_pages_forge.py |
| TRAIN button | Green button | Starts training with the selected mode. Shows "RESUMING..." when Resume checkbox is on and a checkpoint is found | gui_pages_forge.py |
| STOP button | Dark button, disabled until training starts | Stops training after current batch (signals Trainer.request_stop()). Shows "STOPPING..." while waiting for batch to finish, resets to "STOP" when training ends | gui_pages_forge.py |
| Auto-train checkbox | CTkCheckBox, tiny dim text | When checked, GENERATE DATA and WEB LEARN automatically select the new file and start training after completion | gui_pages.py |

**Mode-adaptive UI:** Switching between `PRE-TRAIN`, `DISTILL`, `BASIC`, `AI-GUIDED`, and `IMAGE` shows only the relevant controls.

| Section | Pre-Train | Distill | Basic | AI-Guided | Image |
|---------|-----------|---------|-------|-----------|-------|
| Pre-Train section | Shown | Hidden | Hidden | Hidden | Hidden |
| Distill section | Hidden | Shown | Hidden | Hidden | Hidden |
| Data source picker | Hidden | Hidden | Shown | Optional supplement | Hidden |
| Topic/goal | Hidden | Hidden | Hidden | Required | Hidden |
| Stage buttons | Hidden | Hidden | Hidden | Shown | Hidden |
| Training brief | Hidden | Hidden | Hidden | Shown | Hidden |
| Pairs per stage | Hidden | Hidden | Hidden | Shown | Hidden |
| Image folder + encoder | Hidden | Hidden | Hidden | Hidden | Shown |

**Training modes:**
| Display Name | Internal Key | Description | Requirements |
|-------------|-------------|-------------|-------------|
| Pre-Train | Pre-Train | Language pre-training from scratch on large text data. Creates a new model from a selected preset, optionally retrains tokenizer, trains with general_mix_ratio=0.0 | Data file or directory (no model route needed) |
| Distill | Distill | Teacher (GGUF) generates targeted training data across 6 categories (personality, reasoning, knowledge, conversation, commands, creativity), student fine-tunes on it. Saves model + generated data to `data/distilled_{name}.txt` | TRAINER + STUDENT routes + at least 1 category selected |
| Basic | Basic | User trains on selected data file. Backend can route to full fine-tune or LoRA path | STUDENT route + data file |
| AI-Guided | AI-Guided | TRAINER generates/teaches curriculum for STUDENT from user topic and optional supplement data | TRAINER + STUDENT routes + topic/goal |
| Image | Vision | Vision training on image-text pairs from selected folder | STUDENT route + image folder |
| Dialogue | Dialogue | TRAINER↔STUDENT multi-turn conversation with scoring, corrections, and reinforcement | TRAINER + STUDENT routes |
| RLHF | RLHF | 2-phase: reward model training on preference data, then PPO policy optimization | STUDENT route + preference JSONL |
| Self-Play | Self-Play | TRAINER scores STUDENT responses and reinforces high-quality outputs | TRAINER + STUDENT routes |
| GRPO | GRPO | Group Relative Policy Optimization. RL without a critic network — reward model + GRPOTrainer | STUDENT route + preference JSONL (prompt/chosen/rejected) |
| ReMax | ReMax | REINFORCE with mean-reward baseline. Simpler than PPO — reward model + ReMaxTrainer | STUDENT route + preference JSONL (prompt/chosen/rejected) |
| SimPO | SimPO | Simple Preference Optimization. No reference model needed (beta=2.5, gamma=0.5) | STUDENT route + preference JSONL (prompt/chosen/rejected) |
| ORPO | ORPO | Odds Ratio Preference Optimization. SFT + alignment in one step (beta=0.1) | STUDENT route + preference JSONL (prompt/chosen/rejected) |
| APO | APO | Anchored Preference Optimization (zero variant). Both sides anchored to reference independently — avoids the DPO "degrade-rejected" failure mode. Routes through `train_dpo(loss_type="apo_zero")` (beta=0.1) | STUDENT route + preference JSONL (prompt/chosen/rejected) |

#### Pre-Train Section
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Model size dropdown | CTkOptionMenu from MODEL_PRESETS | Selects architecture preset (pi_zero to omega). Inline "name - description" format (e.g. "small - Entry GPU, ~1 GB VRAM"). Default: "small". | gui_pages_forge.py |
| Data path entry + Browse | CTkEntry + browse button | Path to .txt, .jsonl, .json file or directory of text files for pre-training corpus | gui_pages_forge.py |
| Vocab size entry | CTkEntry, default "32000" | Target vocab size for BPE tokenizer retraining (256–100000) | gui_pages_forge.py |
| Retrain tokenizer checkbox | CTkCheckBox, default ON | When checked, trains a new BPE tokenizer on the data before model creation. Tooltip: "Automatic during pre-training, for standalone use FORGE Tools" | gui_pages_forge.py |
| Model name entry | CTkEntry, default "pretrained_model" | Output model filename (saved to models/) | gui_pages_forge.py |

#### Distill Section
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Category checkboxes | 6 CTkCheckBox (personality, reasoning, knowledge, conversation, commands, creativity) | Select which categories of training data the teacher generates. At least 1 required | gui_pages_forge.py |
| Examples per category | CTkEntry, default "50" | Number of examples teacher generates per selected category (1–500) | gui_pages_forge.py |
| Max tokens | CTkEntry, default "512" | Maximum token length per generated example (32–8192) | gui_pages_forge.py |

**Contract note (intentional):** The FORGE page contract is 13 modes in three rows — Foundation (Pre-Train, Distill, Basic, Image), Advanced (AI-Guided, Dialogue, RLHF, Self-Play), and Alignment (GRPO, ReMax, SimPO, ORPO, APO). All 13 are in the radio-card selector. Evolutionary selection is a checkbox toggle visible for Basic, AI-Guided, RLHF, and Self-Play modes (not alignment modes).

**AI-Guided validation:** If topic/goal is empty, training does not start and the output log explicitly tells the user what to fill in and why.

**AI-Guided execution note:** The current backend path goes through the adaptive pipeline. Supplement selection and start-stage selection are now wired. The pipeline still intentionally auto-chains remaining stages after the selected start point.

**Migration note (Completed March 9, 2026):** All GUI tests have been updated to validate the 8-mode contract. Test class renamed from `TestRLHFSelfPlayDropdown` to `TestTrainingModes` with 9 tests covering Basic, AI-Guided, and Image modes. Legacy mode references in tests removed. Code passes linting and tests pass.

**Forge status (March 12, 2026):** Core FORGE work is complete for day-to-day training: 8-mode UX, automatic LoRA routing for large models, automatic before/after perplexity logging, curriculum preview in log, and training report card (letter grades + next-step recommendations) are all active. Training Brief save now verifies writes. Profile labels auto-size (truncation fixed). Remaining roadmap items are tool success-rate persistence and Discovery mode orchestration.

**Backend training improvements (March 11, 2026, no GUI changes):** `TrainingConfig` now includes `adam_beta1`/`adam_beta2`/`adam_eps` (LM-friendly defaults), `val_split` (hold-out fraction for per-epoch validation), and LoRA `weight_decay` is configurable. RMSNorm computes in fp32 for numerical stability. All tokenizers expose `think_start_id`/`think_end_id`. These are backend-only — no new FORGE widgets.

**Hardware-adaptive training (April 17, 2026, no GUI changes):** `TrainingMemoryBudget` in `hardware_detection.py` scales 11 training constants (streaming threshold/window, minhash/curriculum limits, CE chunk size, batch caps, tokenizer sample cap, dedup capacity, replay capacity) to detected RAM/VRAM. Works on Pi 5 (8 GB) through RTX 5090 workstation (64+ GB). `TrainingConfig.training_memory_gb` field (default 0.0 = auto-detect). All GUI training modes now use budget-derived `ce_chunk_size` instead of hardcoded 4096. RAM warning threshold is now 10% of total instead of fixed 4 GB.

**Reality check (March 10, 2026):** The visible FORGE UI is mostly aligned with the backend now:
- The AI-Guided supplement dropdown is wired through the adaptive/tool paths.
- Stage buttons define the adaptive start stage, then the pipeline intentionally continues forward through later stages.
- The adaptive plan records test scores, but progression still auto-advances rather than using score thresholds.
- The old focus-field widget is gone; Training Topic + Training Brief are the active control surfaces.

**Adaptive pipeline output (March 12, 2026):** During Phase 1 (data generation), the log shows each generated example with a truncated preview (`[1/20] User: ... AI: ...`, up to 150 chars). After the full pipeline completes, a training report card is logged: overall letter grade (A-F with verdict), per-stage score breakdown, and tailored next-step recommendations based on performance (score 8+: refine/test/backup; 5-7: more reps/data/epochs; <5: check trainer/lower LR/try basic mode).

**Training generation limits (March 15, 2026):** All Phase 3 testing and Dialogue training generation limits have been raised for better training quality. Test questions: max_gen=100, student answers: max_gen=256, judgments: max_gen=128. Dialogue-specific: questions max_gen=128, answers max_gen=256, corrections max_gen=300. All log truncation has been removed — training output now shows full untruncated text (newlines replaced with spaces for single-line display). Affects gui_forge_advanced.py, gui_forge_adaptive.py, gui_forge_tools.py, and gui_forge_queue.py.

**Training stage buttons:**
| Stage | Tooltip | What TRAINER teaches |
|-------|---------|---------------------|
| BASICS | Teach fundamental language patterns, grammar, and basic responses | If selected, adaptive training starts here and then continues forward through later stages |
| CONVERSATION | Teach natural dialogue flow and contextual responses | If selected, adaptive training starts here and then continues forward through later stages |
| COMMANDS | Teach command recognition and structured outputs | If selected, adaptive training starts here and then continues forward through later stages |
| WEB | Teach web content understanding and information extraction | If selected, adaptive training starts here; no later stages remain |

**Stage data formats:** Each stage generates training data in its own format via `_build_generation_prompt()` and `_format_training_pair()`:
| Stage | Generation Format | Supplement Format |
|-------|------------------|-------------------|
| BASICS | Varied text (paragraphs, lists, examples) | Raw text (no Q/A wrapper) |
| CONVERSATION | User/AI dialogue pairs | User: .../AI: ... format |
| COMMANDS | Q&A with [CMD] blocks | Q: .../A: ... format |
| WEB | Q&A with search context | Q: .../A: ... format |

#### Tools (CollapsiblePanel, collapsed by default)
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| GENERATE DATA button | Dark button | TRAINER autonomously generates training data in stage-appropriate format (basics=varied text, conversation=User/AI dialogue, commands=Q&A+CMD, web=Q&A+search) via `_build_generation_prompt()`. When "Include reasoning" is checked, generated data includes `<think>` reasoning chains. Saves to data/. Updates progress bar. If Auto-train is checked, starts training on completion | gui_pages.py |
| EVALUATE button | Dark button | TRAINER tests STUDENT: generates questions, judges answers 1-10, determines readiness | gui_pages.py |
| BENCHMARK button | Dark button | Runs 20-prompt coherence benchmark on CHAT model, scores reflections via `score_coherence()`, reports readiness for automatic monologue mode (ready/marginal/not_ready) | gui_pages_forge.py |
| HISTORY button | Dark button | Displays past training runs from data/training_history.json in the log. Shows model name, mode, epochs, final loss, and timestamp for each run | gui_pages.py |
| SAVE CHECKPOINT button | Dark button | Saves current model state to models/checkpoints/ with auto-name (modelname_timestamp). During active training, saves live weights via trainer._save_checkpoint() (GPU memory). When idle, copies the on-disk .pth file (already up-to-date after training completes). | gui_pages.py |
| LOAD CHECKPOINT button | Dark button | Loads a checkpoint back into the STUDENT model slot | gui_pages.py |
| Topic entry | Text input | Topic for WEB LEARN search | gui_pages.py |
| WEB LEARN button | Dark themed button | Searches DuckDuckGo via web_utils, fetches pages, TRAINER generates Q/A pairs using trainer system prompt. Updates progress bar. If Auto-train is checked, starts training on completion | gui_pages.py |
| Max pages entry | Text input, default "3" | How many web pages to read | gui_pages.py |
| Vocabulary size entry | Text input, default "8000" | BPE tokenizer vocabulary size | gui_pages.py |
| TRAIN TOKENIZER button | Dark button | Trains a BPE tokenizer on selected data. Tooltip: "Standalone on any data, for pre-training use checkbox" | gui_pages.py |
| Quantize mode dropdown | Option menu (int8/int4/fp16) | Select quantization bitwidth for STUDENT model | gui_pages.py |
| QUANTIZE button | Dark button | Quantize the STUDENT model to reduce size | gui_pages.py |
| Export GGUF mode dropdown | Option menu (Q8_0/Q4_0/Q4_K_M/F16) | Select GGUF quantization type | gui_pages.py |
| EXPORT GGUF button | Dark button | Export STUDENT as a GGUF file for llama.cpp | gui_pages.py |
| ADD TO QUEUE button | Dark button | Adds current FORGE settings (mode, data, epochs, LR, batch) as a job to the training queue | gui_pages_forge.py |
| QUEUE button | Dark button | Displays current queue state and job list in the forge log | gui_pages_forge.py |
| RUN button | Green-accent button | Starts the training queue. Toggles to PAUSE while running. Click again to resume | gui_pages_forge.py |
| SAVE PLAN button | Dark button | Saves current queue as an overnight plan JSON via file dialog. Stores all pending jobs for later resume | gui_pages_forge.py |
| LOAD PLAN button | Dark button | Loads a saved overnight plan JSON via file dialog. Adds remaining jobs to queue, skips completed ones | gui_pages_forge.py |
| REVIEW DATASET button | Dark button | Shows curated dataset summary and pending entries in forge log. Lists source, stage, and text preview | gui_pages_forge.py |
| APPROVE ALL button | Dark button | Approves all pending entries in the curated dataset for training use. Saves to JSONL | gui_pages_forge.py |

**Web Learn behavior:** Uses shared `web_utils.py` (ddg_search + fetch_page_text) to search DuckDuckGo for the topic and fetch top N pages. Extracts text content (limited to 3000 chars per page), breaks into chunks. TRAINER generates one training pair per chunk in stage-appropriate format using `_build_trainer_system_prompt()` (respects training brief and stage) and `_format_training_pair()`. Updates progress bar throughout (search → fetch → generate → save). Saves all pairs as `web_<topic>.txt` in data/. When Auto-train is checked, routes the new file to the active mode selector and starts training.

#### Advanced Settings (CollapsiblePanel, collapsed by default)
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Replay Capacity entry | themed_entry, default "256" | Maximum stored experiences in the prioritized `ReplayBuffer` for RLHF/SelfPlay training. Wired into `RLHFConfig` and `SelfPlayConfig` via `gui_forge_new_modes.py` | gui_pages_forge.py |
| Replay Ratio entry | themed_entry, default "0.25" | Fraction (0.0–1.0) of minibatch drawn from replay buffer vs fresh rollouts. Wired into `RLHFConfig` and `SelfPlayConfig` | gui_pages_forge.py |

**Advanced Settings behavior:** These controls are read by `_read_forge_rl_params()` which returns a dict consumed by RLHF and SelfPlay training launchers in `gui_forge_new_modes.py`. Replay capacity and ratio are numeric. Both settings affect RLHF and Self-Play training modes.

#### Hardcoded Training Parameters (no GUI control)

These values are set in code and not exposed as GUI widgets. Documented for reference — they can be changed by editing the source files directly.

| Parameter | Value | Where | Notes |
|-----------|-------|-------|-------|
| DPO beta | 0.1 | gui_forge_training.py | Controls preference strength in DPO training |
| Dialogue temperature | 0.8 | gui_forge_advanced.py | Teacher generation temp during dialogue training |
| Adaptive temperature | 0.7 | gui_forge_adaptive.py | Teacher generation temp during adaptive pipeline |
| Distill temperature | 0.8 | gui_forge_new_modes.py | Teacher generation temp during distillation |
| Evolutionary temp/top_k | 0.9 / 50 | gui_forge_advanced.py | Candidate generation settings for evolutionary selection |
| Vision batch size | 1 | gui_forge_training.py | Fixed batch size for vision training |
| Reward model epochs | min(epochs, 5) | gui_forge_new_modes.py | RLHF reward model capped at 5 epochs |
| Reward model LR | lr × 10 | gui_forge_new_modes.py | RLHF reward model uses 10× the base learning rate |
| Pre-train warmup | 1% of total steps | gui_forge_new_modes.py | WSD schedule warmup duration |
| Pre-train general_mix | auto 0→10% | gui_forge_new_modes.py | If set to 0, silently overridden to 10% (logged) |
| GRPO/ReMax reward model | Same as RLHF | gui_forge_new_modes.py | Phase 1: reward model (min(epochs,5), lr×10), then Phase 2: RL policy (GRPOTrainer or ReMaxTrainer) |
| SimPO beta/gamma | 2.5 / 0.5 | gui_forge_new_modes.py | Simple Preference Optimization hyperparameters |
| ORPO beta | 0.1 | gui_forge_new_modes.py | Odds Ratio Preference Optimization strength |

### Right Column: Log
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Log panel | HUDFrame, right column (resizable via PanedWindow sash) | Contains training output log | gui_pages.py |
| "OUTPUT LOG" header | Green section label | Labels the log | gui_pages.py |
| Training log | Text box, green text | Shows epoch loss, training status, errors, completion info, loss curves | gui_pages.py |
| Progress bar | CTkProgressBar (green, 6px) + percentage label | Shows progress for training, data generation, web learn, and evaluation. Updates from 0-100% | gui_pages.py |
| Loss chart panel | CollapsiblePanel, collapsed by default | Contains graphical loss chart. Auto-expands when training completes | gui_pages_forge.py |
| Loss chart canvas | tk.Canvas, height 150px | Draws loss curve (green line), moving average (accent line), grid lines, axis labels. Thread-safe via self.after() | gui_forge_tools.py |
| Loss chart info label | SelectableLabel, dim text | Shows "Steps: N | Loss: X.XXXX | Best: X.XXXX | PPL: X.X" below the canvas | gui_forge_tools.py |

**Loss curve (text):** Text-based bar chart rendered in the log after training. Shows per-epoch loss with block characters (█) proportional to loss magnitude.

**Loss curve (graphical):** Canvas line chart in a collapsible panel between the progress bar and log. Green line = actual loss per step/epoch. Accent line = smoothed moving average. Three horizontal grid lines with loss values. Auto-expands when training completes. Includes perplexity info when available from TrainingMonitor.

**Evaluation results:** When training completes (Solo or LoRA modes), the log displays before/after perplexity measurements evaluated on a fixed set of test prompts. Shows "Before: perplexity = X.XX", "After: perplexity = Y.YY", and "Improvement: Z.ZZ (N.N%)". Lower perplexity indicates better language modeling. Evaluation is automatic (enabled via `run_evaluation=True` in TrainingConfig). Uses `evaluate_model()` from `training_evaluation.py`.

---

## PAGE: DOCS (Documentation Browser)

Documentation browser with file editor, inline rename, search filter, and unsaved change detection. Files are organized into categories: Guides (from information/), Prompts (from data/prompts/), Notes (from data/notes/), and Mod docs (from mods/<id>/docs/).

### Top Bar
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| "Documentation" header | SectionLabel | Page title | gui_docs_page.py |
| + NEW button | Small button, silver text | Creates a blank untitled.md file in information/ (auto-numbered: untitled.md, untitled_2.md, etc.) and opens it in the editor | gui_docs_page.py |

### Left Column: File Browser
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Search bar | CTkEntry with placeholder "Search files..." | Filters browser entries in real time by name, filename, or category. Clears to show all files | gui_docs_page.py |
| Browser frame | HUDFrame with scrollable inner | Contains categorized file list | gui_docs_page.py |
| Category headers | Tiny colored labels (GUIDES=cyan, TRAINER=green, TRAINING DATA=green, PROMPTS=orange, NOTES=yellow, MOD:X=silver) | Group files by source | gui_docs_page.py |
| File entries | Clickable buttons, one per file | Click to load file into editor. Highlights when selected. Hover tooltip shows the full file path on disk | gui_docs_page.py |

### Right Column: Editor
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Filename label | Small text above editor, clickable | Shows current file name. Click to rename: swaps to inline CTkEntry, Enter confirms, Escape/FocusOut cancels. Shows "● name" in orange when unsaved, turns green on save, red on delete | gui_docs_page.py |
| File path label | Tiny dim text below filename | Shows the full filesystem path of the currently loaded file. Clears on delete | gui_docs_page.py |
| SAVE button | Green button | Writes editor content back to the file. Also bound to Ctrl+S | gui_docs_page.py |
| DELETE button | Red text button | Shows inline red bar: "Delete filename? [YES] [NO]". YES deletes the file, NO cancels | gui_docs_page.py |
| RELOAD button | Dim button | Refreshes the file browser (re-scans all sources) | gui_docs_page.py |
| Editor textbox | CTkTextbox, word wrap, full height | Edit file content. Supports .md, .txt, and .json files. Tracks edits for unsaved indicator | gui_docs_page.py |
| Stats footer | Tiny dim text, right-aligned | Shows live "X lines · Y words · Z chars" count, updates on every keystroke | gui_docs_page.py |

**File sources:**
| Category | Source Directory | File Types | Color |
|----------|-----------------|------------|-------|
| Guides | information/ | .md, .txt | Cyan |
| Trainer | information/trainer/ | .md, .txt | Green |
| Training Data | data/ | .txt, .jsonl | Green |
| Prompts | data/prompts/ | .md, .txt | Orange |
| Notes | data/notes/ | .md, .txt | Yellow |
| Mod docs | mods/<id>/docs/ | .md, .txt | Silver |

**Default guide files:** how_the_ai_works.md, training_guide.md, commands_reference.md, getting_started.md, prompts_guide.md, external_models.md

**Prompt files:** chat.md (default system prompt for new model contexts), trainer.md (prepended to FORGE trainer system prompt). Edit these to customize AI behavior per route.

**Unsaved changes:** When editor content differs from the saved file, the filename label shows "● filename" in orange. Switching files shows an inline bar: "Unsaved changes [SAVE] [DISCARD] [CANCEL]" — SAVE writes the file, DISCARD abandons changes, CANCEL stays on the current file. Uses `_docs_pending_action` to defer the interrupted navigation until the user responds. No popup dialog.

**Keyboard shortcuts:** Ctrl+S saves the current file. Ctrl+Z undoes the last edit. Ctrl+Y redoes. Unlimited undo history (reset when a new file is loaded). All shortcuts work when the editor has focus.

**Find bar (Ctrl+F):** Toggle via Ctrl+F or right-click menu → "Find (Ctrl+F)". Appears as an inline bar below the toolbar with: Find entry (placeholder "Find..."), Previous (▲) and Next (▼) navigation buttons, match count display ("N matches"), and Close button (✖). All matches highlighted with `find_hl` tag, active match highlighted with `find_current` tag. Wraps around on reaching end/beginning.

**Right-click context menu:** Right-clicking the editor shows a context menu with: Cut, Copy, Paste, Select All, and Find (Ctrl+F).

**Auto-save:** Modified documents are automatically saved every 30 seconds via `_docs_auto_save()`. Silent operation — no status message unless the user also manually saves.

**Search filter:** Type in the search bar at the top of the file browser to filter entries. Matches against file name, filename on disk, and category. Clear the search to show all files again.

**Inline rename:** Click the filename label to enter rename mode. An inline text entry appears with the current name. Press Enter to confirm (renames file on disk, updates browser), Escape or click away to cancel. The filename label has a hand cursor to indicate clickability.

---

## PAGE: CONFIG (Settings)

Uses friendly display names so users understand what each parameter does.

### Top Bar
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| "SETTINGS" header | Section label | Page title | gui_pages.py |
| Intro text | Dim text | "These settings control how the AI generates text." | gui_pages.py |

### Config Cards (scrollable)
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Config card | HUDFrame, one per parameter | Contains friendly name, description, range, and input | gui_pages.py |
| Friendly name | Bold text like "Creativity" | User-friendly name for the parameter | gui_pages.py |
| Description | Tiny dim text, wraps at 500px | Explains what the parameter affects in plain language | gui_pages.py |
| Range label | Tiny silver text like "Range: 0.0 to 2.0" | Shows valid min/max values | gui_pages.py |
| Value entry | Text input on right | Type a number. Validates on focus-out or Enter, clamps to valid range | gui_pages.py |

**Parameters and friendly names:**
| Internal Name | Display Name | Description |
|--------------|-------------|-------------|
| temperature | Creativity | How creative the AI is |
| top_p | Diversity | Controls response diversity |
| top_k | Word Choices | How many word choices the AI considers |
| max_tokens | Response Length | Maximum length of each AI response |
| repetition_penalty | Repetition Control | How strongly the AI avoids repeating itself |

**Backend-only parameters** (in defaults.py, usable via API/CLI, not yet in CONFIG GUI):
| Internal Name | Default | Description |
|--------------|---------|-------------|
| min_p | 0.0 | Minimum probability filter (relative to top token). 0.0 = disabled |

### Theme Card
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Theme card | HUDFrame with accent border | Contains theme selector | gui_pages.py |
| "THEME" | Bold header | Labels the card | gui_pages.py |
| Description | Tiny dim text | "Change the color theme." | gui_pages.py |
| Theme dropdown | themed_dropdown | Lists all 4 themes (dark, midnight, carbon, solarized). Auto-selects current active theme. Changing the selection applies the theme immediately (live switching) | gui_pages.py |

**Theme behavior:** Selecting a theme from the dropdown applies it immediately via `_apply_theme_live()` and `_retheme_tree()` which walk the widget tree and remap all colors in-place. No restart required. C_* constants are updated globally via `reload_theme()` in widgets.py. Theme preference is saved to `gui_settings.json["theme"]`.

### Font Size Card
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Font size card | HUDFrame with accent border | Contains font size offset control | gui_pages_config.py |
| "FONT SIZE" | Bold header | Labels the card | gui_pages_config.py |
| Description | Tiny dim text | "Adjust font size across the entire GUI. Takes effect on restart." | gui_pages_config.py |
| Size Offset entry | Text input, default "0" | Integer offset applied to all font sizes. Range -4 to 8 | gui_pages_config.py |
| Range label | Tiny silver text | Shows "Range: -4 to 8" | gui_pages_config.py |
| APPLY button | Silver accent button | Saves offset to gui_settings.json and auto-restarts GUI | gui_pages_config.py |

**Font size behavior:** The offset is added to every FONT_* tuple's base size at import time via `set_font_size_offset()` in widgets.py. For example, offset=2 makes FONT_BODY go from 16 to 18. The value is persisted in `gui_settings.json["font_size_offset"]` and loaded automatically when the GUI starts. Clamped to [-4, 8] for safety. Requires restart — fonts are module-level constants.

### Memory Card
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Memory card | HUDFrame with accent border | Contains memory mode selector | gui_pages_config.py |
| "MEMORY" | Bold header | Labels the card | gui_pages_config.py |
| "Memory Mode" label | Bold text in row | Labels the dropdown | gui_pages_config.py |
| Memory mode dropdown | themed_dropdown | Values: "automatic - AI extracts facts from chat", "manual - ...", "disabled - ...". Inline description format. Controls how the AI stores facts from chat | gui_pages_config.py |
| Tooltip | Hover text on dropdown | Explains modes: automatic = facts extracted from every message, manual = only explicit memory.remember commands, disabled = no memory storage at all | gui_pages_config.py |

**Memory mode behavior:** Selecting a mode saves to `gui_settings.json["memory_mode"]` and applies live via `_change_memory_mode()`. In "automatic" mode (default), `extract_facts()` runs on every chat message. In "manual" mode, only explicit `memory.remember` commands store facts. In "disabled" mode, `PersistentMemory.add()` is a no-op (returns False). The mode is read from settings by `_get_memory_mode()` in `gui_logic_chat.py` before each auto-extraction call.

### Monologue Card
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Monologue card | HUDFrame with accent border | Contains monologue mode selector | gui_pages_config.py |
| "MONOLOGUE" | Bold header | Labels the card | gui_pages_config.py |
| "Monologue Mode" label | Bold text in row | Labels the dropdown | gui_pages_config.py |
| Monologue mode dropdown | themed_dropdown | Values: "disabled - no background reflection", "journal_only - ...", "automatic - ...". Inline description format. Controls AI self-reflection behavior during idle periods | gui_pages_config.py |
| Tooltip | Hover text on dropdown | Explains modes: disabled = no monologue, journal_only = reflections stored but not shown, automatic = reflections stored and shown in CMD | gui_pages_config.py |

**Monologue mode behavior:** Selecting a mode saves to `gui_settings.json["monologue_mode"]` and applies live via `_change_monologue_mode()`. In "disabled" mode (default), no reflections are generated. In "journal_only" and "automatic" modes, the idle tracker in `desktop.py` polls every 30 seconds — when the user has been idle for 5 minutes and a model is loaded, a reflection is generated in a background thread via `_run_reflection()`. The reflection is scored by `score_coherence()` — only entries above the threshold (0.7) are stored in the model's `Journal`. In "automatic" mode, reflections that pass the gate are also displayed in the CMD activity log. Journal entries are visible in the CORE sidebar Journal panel.

### Emotional State Visibility
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Show emotional state panel | CTkCheckBox | When checked (default), the emotional state panel is visible on the CORE sidebar. When unchecked, the panel is hidden but the AI still uses emotional awareness internally. Persisted to `gui_settings.json["show_emotional_state"]` | gui_pages_config.py |

### History Cap Card
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| History cap card | HUDFrame with accent border | Contains chat history disk persistence cap | gui_pages_config.py |
| "History cap" label | Bold text in row | Labels the entry | gui_pages_config.py |
| History cap entry | themed_entry, default "500" | Integer value (10–10000). Controls `MAX_CONTEXT_HISTORY` — how many messages are persisted to disk in `memory/session_*.json`. In-memory history is unchanged. Saved to `gui_settings.json["history_cap"]`, applied on startup via `set_max_context_history()` | gui_pages_config.py |

**History cap behavior:** Changing the value calls `_change_history_cap()` which validates the range (10–10000), updates `MAX_CONTEXT_HISTORY` at runtime via `set_max_context_history()`, and persists to `gui_settings.json`. The cap only affects disk writes in `ModelContext._save_history()` — in-memory history and token-based context truncation are separate.

### Training Card
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Training card | HUDFrame with border | Contains background training options | gui_pages.py |
| "TRAINING" | Bold section header | Labels the card | gui_pages.py |
| Description | Tiny dim text | "Background training options for chat sessions." | gui_pages.py |
| Learn while chatting | CTkCheckBox | When enabled, chat exchanges are fed to the background trainer so the AI improves over time. Requires TRAINER route assigned. Persisted to gui_settings.json | gui_pages.py |

**Learn while chatting behavior:** Each user↔AI exchange during normal chat is captured and fed to the `BackgroundTrainer` via `_feed_background_trainer()` in LogicMixin. The background trainer accumulates exchanges and periodically runs a short SFT step on the STUDENT model. This causes the AI to gradually learn from its own conversations. Toggling the checkbox immediately saves the preference to `data/gui_settings.json["learn_while_chatting"]`.

### Performance Card
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Performance card | HUDFrame with accent border | Contains launch and memory settings | gui_pages_config.py |
| "PERFORMANCE" | Bold section header | Labels the card | gui_pages_config.py |
| Description | Tiny dim text | "Reduce background memory usage while keeping the app open. Launch settings apply on next start." | gui_pages_config.py |
| Auto-load chat model on launch | CTkCheckBox | When enabled, the CHAT route model loads automatically on launch. Persisted to gui_settings.json | gui_pages_config.py |
| Auto-start mods on launch | CTkCheckBox | When enabled, all mods start automatically on launch. Persisted to gui_settings.json | gui_pages_config.py |
| Unload chat model when minimized | CTkCheckBox | When enabled, minimizing the window releases model memory and reloads when restored. Persisted to gui_settings.json | gui_pages_config.py |
| Confirm AI file operations | CTkCheckBox | When enabled (default), a confirmation dialog appears when the AI wants to write/append a file. When disabled, file operations are auto-approved. Thread-safe (`after(0)` + `threading.Event`). Persisted to `gui_settings.json["confirm_file_operations"]` | gui_pages_config.py |
| APPLY GAMING MODE | Button (silver accent) | Sets all three to low-memory mode: no model autoload, no mod autostart, unload on minimize | gui_pages_config.py |

**Gaming mode behavior:** Clicking APPLY GAMING MODE sets all three checkboxes to their low-overhead state and also enforces a live runtime profile: chat-learning off, router trainer off, slower UI/status timers, exact param counting skipped, and minimize-unload enabled. Described in more detail under Confirmed Fixes in `SUGGESTIONS.md`.

### Mod Info Card
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Mod card | HUDFrame with border | Shows installed mod modules | gui_pages.py |
| "MOD MODULES" | Bold header | Labels the card | gui_pages.py |
| Description | Tiny text | "Mods are plugin programs that connect to the engine. Auto-start can be changed in PERFORMANCE." | gui_pages_config.py |
| Mod rows | One row per mod | Shows "name vX.X" and description snippet | gui_pages.py |
mod rules. modes need to be added without haveing to code the GUI, be able to have there own working page, these are sopposed to be add ons like it is a something entirely seperate do not add it to the main code it just needs to show up here so the user can acess it maybe the AI can acess it too

### Directory Paths Section
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Paths card | HUDFrame at bottom of CONFIG | Contains all directory path settings | gui_pages.py |
| "DIRECTORY PATHS" | Bold section header | Labels the paths section | gui_pages.py |
| Description | Tiny dim text | "Set where the engine reads and writes files. Changes take effect on next launch." | gui_pages.py |
| Path rows | One row per directory key | Each has display name, text entry, and browse button | gui_pages.py |
| Browse button | Small "..." button per row | Opens directory picker dialog | gui_pages.py |
| SAVE PATHS button | Silver accent button | Persists path overrides to data/path_settings.json | gui_pages.py |
| RESET button | Dark button | Restores all paths to defaults | gui_pages.py |

### Backup / Restore Section
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Backup card | HUDFrame with border | Contains backup/restore buttons | gui_pages_config.py |
| "BACKUP / RESTORE" | Bold header | Labels the section | gui_pages_config.py |
| Description | Tiny dim text | "Export your settings, routes, prompts, and notes as a zip file, or restore from a previous backup." | gui_pages_config.py |
| EXPORT BACKUP button | Silver accent button | Exports settings, routes, prompts, and notes to a .zip file via save dialog | gui_pages_config.py |
| IMPORT BACKUP button | Dark button | Opens file picker for a .zip backup, then shows inline yellow bar: "Overwrite settings with backup? [YES] [NO]". YES extracts and restores all settings, NO cancels | gui_pages_config.py |

### Display Names Section
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Names card | HUDFrame below paths | Contains display name settings | gui_pages.py |
| "DISPLAY NAMES" | Bold section header | Labels the names section | gui_pages.py |
| Description | Tiny dim text | "Set how your name and the AI's name appear in chat." | gui_pages.py |
| Your Name entry | Text input, default "YOU" | Sets the user's display name in chat messages | gui_pages.py |
| AI Name entry | Text input, default "ENIGMA" | Sets the default AI display name (overridden by model_info.json) | gui_pages.py |
| SAVE NAMES button | Silver accent button | Persists display names to data/gui_settings.json | gui_pages.py |
| RESET button | Dark button | Restores names to defaults ("YOU" / "ENIGMA") | gui_pages.py |

**Display name priority:** Per-model model_info.json `display_name` > CONFIG AI Name setting > "ENIGMA" default. User name is always from CONFIG.

**Per-model names:** Place a `model_info.json` file in the model's folder with `{"display_name": "Name"}`. When that model loads, the AI name in chat updates automatically. On unload, reverts to the CONFIG setting.

**Editable paths:**
| Key | Display Name | Default |
|-----|-------------|--------|
| models_dir | Models Directory | models/ |
| data_dir | Training Data | data/ |
| outputs_dir | Outputs Directory | outputs/ |
| sessions_dir | Sessions Directory | data/sessions/ |
| memory_dir | Memory Directory | memory/ |
| mods_dir | Mods Directory | mods/ |

**Path behavior:** Saved paths are stored in `data/path_settings.json`. On startup, saved overrides are loaded into the entry fields. RESET clears all overrides and restores defaults. The browse button opens a native directory picker. Path constants and persistence functions live in scanners.py (`PATH_SETTINGS`, `load_path_settings()`, `save_path_settings()`, `get_path()`).

---

## COLOR PALETTE (themes.py → widgets.py)

Colors are defined as `Theme` dataclasses in `themes.py`. The active theme is loaded at import time in `widgets.py`, populating all `C_*` constants. 4 preset themes available: **dark** (default), **midnight**, **carbon**, **solarized**. Theme preference is saved in `data/gui_settings.json["theme"]`. Themes can be switched live via `reload_theme()` in widgets.py and `_apply_theme_live()` / `_retheme_tree()` in desktop.py — the widget tree is walked and all colors are remapped in place without restarting.

Values below are for the default **dark** theme:

| Name | Hex | Used For |
|------|-----|----------|
| C_BG | #080808 | Window background |
| C_PANEL | #0e0e0e | Panel/card backgrounds |
| C_SURFACE | #181818 | Button hover, slightly lighter areas |
| C_INPUT | #1c1c1c | Text input backgrounds |
| C_ACCENT | #8B95A5 | Primary silver/gray: titles, active states, highlights |
| C_ACCENT_DIM | #2a2a2a | Borders, inactive accents |
| C_ACCENT_MUTED | #3d3d3d | Hover states, secondary accents |
| C_PURPLE | #a855f7 | User messages |
| C_PURPLE_DIM | #2a1a3e | Purple button backgrounds (fine-tune) |
| C_PURPLE_MUTED | #3d2a55 | Purple hover states |
| C_CYAN | #22d3ee | CMD page accent, file attachment tag |
| C_TEXT | #b0b0b0 | Normal body text |
| C_TEXT_DIM | #555555 | Dim labels, descriptions |
| C_TEXT_BRIGHT | #e8e8e8 | Bright text, parameter names |
| C_GREEN | #22c55e | Success, loaded status, training log, voice toggle on, SEND button |
| C_GREEN_DIM | #0e3a1e | Green button backgrounds (SEND, voice toggle) |
| C_RED | #ef4444 | Errors, failed status |
| C_ORANGE | #f97316 | Warnings, loading state |
| C_BORDER | #1f1f1f | Default panel borders |
| C_BORDER_ACCENT | #2e2e2e | Section label lines, subtle borders |

---

## FONT PALETTE (widgets.py)

All fonts are Consolas monospace. Defined at lines 63-71.

| Name | Size | Used For |
|------|------|----------|
| FONT_TITLE | Consolas 26 bold | App title |
| FONT_SECTION | Consolas 17 bold | Section headers, nav buttons |
| FONT_BODY | Consolas 16 | General text |
| FONT_SMALL | Consolas 15 | Descriptions, small buttons |
| FONT_TINY | Consolas 14 | Labels, ranges, status text |
| FONT_CHAT | Consolas 16 | Chat messages |
| FONT_INPUT | Consolas 17 | Chat text input |
| FONT_MONO | Consolas 16 | Config entries, training log |
| FONT_CMD | Consolas 15 | Command terminal output |

---
