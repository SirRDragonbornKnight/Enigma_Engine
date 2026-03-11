# GUI Reference - Every Element Explained

This document maps every visible element in the Enigma Engine desktop GUI,
what it is, where it lives in code, and what it does.

Use this to decide what to change, move, remove, or redesign.

**Text selectability:** Almost all display text uses `SelectableLabel` (tk.Entry in readonly state) — supports click-drag selection, Ctrl+C copy, and right-click copy menu with no blinking cursor. Multi-line labels that use `wraplength` remain as `CTkLabel` with right-click copy via `_enable_label_copy`.

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
| Ctrl + Z | Undo (docs editor) |
| Ctrl + Y | Redo (docs editor) |
| Ctrl + S | Save (docs editor) |
| Ctrl + F | Find (docs editor) |
| Up / Down | Command history (CMD page) |

**Nav collapse:** Click the ◀ arrow button in the header to collapse the nav rail to 0px (fully hidden via grid_remove). Content expands to fill the full width. Click ▶ to restore full 170px with labels.

**Mod behavior:** Each mod is a full page. All mods auto-start when the app launches. Mod pages show info, commands, UI widgets from mod.json, and an output log.

---

## PAGE: CMD (Dual-Mode Terminal)

Dual-mode terminal with SYSTEM (real PowerShell) and ENGINE (AI command registry) modes.
AI ACCESS toggle lets the AI execute real system commands when enabled.
Also serves as an activity monitor — all AI operations (chat [CMD] blocks, command execution) are logged here via `_cmd_activity()`.
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
| system.info | System information |
| (all engine commands) | Full command registry available |

**AI command execution flow:**
1. AI responds to "ask" with text and optional [CMD] blocks
2. Each [CMD] block is tried as an engine command first
3. If unknown and AI ACCESS is ON, the command runs as a real system command
4. If unknown and AI ACCESS is OFF, an error is shown

**Keyboard:** Up/Down arrows navigate command history. Enter executes.

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
| Chat display | SelectableTextbox (CTkTextbox subclass) with native scrollbar, word wrap, 12px left/right margins | Shows conversation: purple for YOU, silver for ENIGMA, orange for SYSTEM (with timestamps), red for errors. When AI uses chain-of-thought reasoning, a `🧠 Reasoning:` section appears before the answer showing the `<think>` content in dim text. Native scrollbar handles all scrolling — no wrapper frame needed. Fills available space via sticky="nsew" | gui_pages.py |
| File indicator | Tiny cyan text above input | Shows attached filename when a file is attached | gui_pages.py |
| Thinking indicator | Tiny dim text, right side, fixed 140px width | Shows "PROCESSING..." with animated dots while AI generates response. Layout stability is enforced at two layers: `input_area.grid_columnconfigure(1, minsize=140)` locks the control column width, and fixed-size `SelectableLabel` instances do not resize on text updates. This prevents residual jitter during animation | gui_pages.py, widgets.py |
| Chat input | Multi-line text box, 56px default | Type messages here. Enter sends. Shift+Enter for newline. Blocked during generation. Auto-expands from 56px to 200px as content grows, resets to 56px after sending | gui_pages.py |
| SEND button | Green button, right of input | Sends the message (or Enter key). Hidden during generation. Tooltip: "Send message (Enter)" | gui_pages.py |
| STOP button | Red button, same slot as SEND | Shown during generation. Stops AI mid-response. Also Escape key. Tooltip: "Stop AI generation" | gui_pages.py |
| Token counter | SelectableLabel on right side of toolbar | Shows current conversation token count (e.g. "128 tokens"). Updates on page show and new chat | gui_pages.py |
| In-memory history cap | Logic behavior (not a direct widget) | Caps `self.history` at `MAX_CHAT_HISTORY = 500` via `_trim_chat_history()` to prevent RAM growth. Full session still auto-saves to disk in `memory/session_*.json` | gui_logic_chat.py |
| Utility toolbar | Row below input | Left side: attach, new, web toggle, edit. Right side: voice, mic — separated from SEND to prevent misclicks | gui_pages.py |
| Attach button | Square button in toolbar (left) | Opens file picker to attach a text file to next message. Tooltip: "Attach file" | gui_pages.py |
| NEW button | Dark button in toolbar (left) | Starts a new conversation: clears chat, history, and KV cache. No confirmation needed — current chat auto-saves | gui_pages.py |
| Web access toggle | ToggleButton (🌐 icon) in toolbar (left) | Toggles AI web access on/off. When ON (cyan), AI can search the web via DuckDuckGo. Flag injected into _build_gui_context() system prompt. Tooltip: "Web access" | gui_pages.py |
| Edit button | Square button (✎ icon) in toolbar (left) | Edits last sent message: removes last exchange, puts user text back in input. Blocked during generation. Tooltip: "Edit last message" | gui_pages.py |
| Voice toggle | Square toggle button in toolbar (right) | Turns voice output (TTS) on/off. When ON (green), AI responses are spoken aloud via pyttsx3 persistent worker thread. Toggling OFF stops any in-progress speech. Tooltip: "Voice output on/off" | gui_pages.py |
| Mic button | Square button (🎤 icon) in toolbar (right) | Voice input: click to start continuous listening, each recognized phrase auto-sends as a chat message. Click again to stop. Uses listen_in_background() with stopper. Turns red while recording. Tooltip: "Voice input (mic)" | gui_pages.py |

### Sidebar (right column, resizable via PanedWindow)

The sidebar contains two **collapsible panels** (CollapsiblePanel widget). Click the header to expand/collapse. When one is collapsed, the other takes all available space. When both are collapsed, only the header rows are visible. The chat/sidebar boundary is a **draggable sash** (tk.PanedWindow) — users can resize by dragging.

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

**Sidebar toggle:** Click the \u25e8 button in the top bar to hide the entire sidebar (history + system prompt). The chat area expands to full width. Click again to restore the sidebar. Both collapsible panels retain their state when the sidebar is toggled.

**Collapsible behavior:**
- Both start expanded by default, sharing the sidebar space equally
- Click a panel header (chevron + title) to collapse it — the other panel expands to fill the space
- Click again to re-expand — both panels share space equally again
- Chevron indicator: ▼ = expanded, ▶ = collapsed
- When both are collapsed, only the two thin header rows are visible

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
| CREATE button | Silver accent button | Creates a blank untrained model with a default small architecture. Feedback shown inline and in status bar. Tooltip: "Create a new empty model" | gui_pages.py |
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
| EXPORT button | Dark button (row 0, right) | Exports identity card as a standalone JSON file. Tooltip: "Export identity card to JSON" | gui_pages.py |
| COPY button | Dark button (row 0, right) | Creates a copy of the model file with "_copy" suffix. Shows → arrow feedback. Tooltip: "Duplicate this model" | gui_pages.py |
| DELETE button | Dark button, hover red (row 0, right) | Shows inline red delete bar (row 4): "Delete model_name? [YES] [NO]". YES confirms deletion, NO cancels. Tooltip: "Permanently delete this model" | gui_pages.py |
| NATIVE/EXTERNAL tag | Colored label (row 1, left) | Green "NATIVE" for .pth/.pt models, orange "EXTERNAL" for .gguf/.bin/.safetensors | gui_pages.py |
| TRAINABLE badge | Cyan text label (row 1, after NATIVE tag) | Shown only on native models (.pth/.pt) that support training. Not shown on external formats | gui_pages.py |
| Format info | Tiny dim text (row 1, after tags) | Shows "PTH // 42 MB" format and file size | gui_pages.py |
| File name subtitle | Tiny dim text (row 1, after format) | Shows original file name in parens when identity display_name differs | gui_pages.py |
| Personality | Normal dim text (row 2) | Short personality description from identity card | gui_pages.py |
| Stats line | Tiny dim text (row 2) | Message count, session count, training run count | gui_pages.py |
| Tags | Tiny accent text (row 2) | User-defined tags displayed as [tag1] [tag2] | gui_pages.py |
| Edit row | Hidden frame (row 3) | SAVE and CANCEL buttons for inline name editing. Only visible when editing | gui_pages.py |
| Delete bar | Hidden frame (row 4) | Red inline bar: "Delete model? [YES] [NO]". Only visible when delete is pending | gui_pages.py |

**Model card layout:**
- **Row 0:** Model name entry (identity name or file name + param count) on left, EDIT / EXPORT / COPY / DELETE buttons on right
- **Row 1:** NATIVE or EXTERNAL tag (color-coded) + TRAINABLE badge (cyan, native only) + format and file size + file name subtitle
- **Row 2 (if identity exists):** Personality, stats (messages/sessions/training runs), tags
- **Row 3 (hidden):** SAVE / CANCEL buttons for inline name editing (shown when EDIT or Rename is active)
- **Row 4 (hidden):** Red delete confirmation bar (shown when DELETE is clicked)

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

**Route behavior:** Each route (CHAT, TRAINER, STUDENT, and each mod) has its own model dropdown. Selecting a model from the CHAT dropdown loads it into the engine. Selecting "None" unloads it. Non-chat routes share the chat engine if assigned the same model (`_get_engine_for_route()`). The STUDENT route is the model being trained — fine-tune in FORGE trains the STUDENT model while TRAINER can evaluate it. Mod routes also show running/stopped state. All route statuses update live via `_update_route_status()` in gui_logic.py. Assignments are stored in `self.route_assignments` dict. Route changes also update the status bar for cross-page visibility.

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
| Training mode cards | 3 radio-card options (`BASIC`, `AI-GUIDED`, `IMAGE`) | Replaces old multi-mode dropdown. User chooses one clear path, then only relevant sections are shown | gui_pages_forge.py |
| Include reasoning checkbox | CTkCheckBox | When ON, AI-generated training data can include `<think>` reasoning chains | gui_pages_forge.py |
| Basic data source dropdown | Option menu | For Basic mode training data. Supports `(none)` and file selection from `data/` | gui_pages_forge.py |
| **Auto-LoRA trigger** | Automatic detection | When training in Basic mode, detects STUDENT model param count. Auto-selects LoRA if > 7B params, full fine-tuning if ≤ 7B params. No user toggle needed — happens automatically at training start. Shows info log message about detected size and selected method | gui_forge.py |
| AI-guided topic/goal entry | Text input (required for AI-Guided) | Defines what the trainer should teach the student. If empty, training logs guidance and does not start | gui_pages_forge.py |
| AI-guided supplement data dropdown | Option menu (optional) | Optional seed data for AI-guided curriculum generation. Wired to `ai_supplement_var` and used by the adaptive backend/tool flows | gui_pages_forge.py |
| Training stage buttons | 4 CTkButtons (`BASICS/CONVERSATION/COMMANDS/WEB`) | Select the adaptive pipeline start stage. Runtime contract is “start here, then continue forward” | gui_pages_forge.py |
| Training brief panel | CollapsiblePanel with quick profile fields + custom text | Refines trainer instructions for AI-guided runs | gui_pages_forge.py |
| Image data directory | Text input + Browse button | Folder with image-text pairs used by Image mode | gui_pages_forge.py |
| Encoder size dropdown | Option menu (`tiny/small/medium`) | Vision encoder size for Image mode | gui_pages_forge.py |
| Training preset dropdown | Option menu (`Quick/Balanced/Thorough/Custom`) | Pre-fills epochs, learning rate, and batch size | gui_pages_forge.py |
| Epochs entry | Text input, default `10` | Number of training passes | gui_pages_forge.py |
| Learning rate entry | Text input, default `0.00005` | Training learning rate | gui_pages_forge.py |
| Batch size entry | Text input, default "4" | Training batch size. Lower = less VRAM | gui_pages.py |
| Grad accumulation entry | Text input, default "1" | Gradient accumulation steps | gui_pages.py |
| Gradient checkpointing | Checkbox | Saves VRAM by recomputing activations | gui_pages.py |
| Rolling best K entry | Text input, default "0" | Keep K best checkpoints by loss during training. 0 = disabled | gui_pages_forge.py |
| AI-guided pairs entry | Text input, default "20" | Number of generated examples per stage for AI-guided flow | gui_pages_forge.py |
| LoRA rank/alpha entries | Text inputs (`8`/`16`) | Advanced LoRA controls (used when LoRA path is selected in backend flow) | gui_pages_forge.py |
| TRAIN button | Green button | Starts training with the selected mode | gui_pages.py |
| STOP button | Dark button, disabled until training starts | Stops training after current epoch | gui_pages.py |
| Auto-train checkbox | CTkCheckBox, tiny dim text | When checked, GENERATE DATA and WEB LEARN automatically select the new file and start training after completion | gui_pages.py |

**Mode-adaptive UI:** Switching between `BASIC`, `AI-GUIDED`, and `IMAGE` shows only the relevant controls.

| Section | Basic | AI-Guided | Image |
|---------|-------|-----------|-------|
| Data source picker | Shown | Optional supplement | Hidden |
| Topic/goal | Hidden | Required | Hidden |
| Stage buttons | Hidden | Shown | Hidden |
| Training brief | Hidden | Shown | Hidden |
| Pairs per stage | Hidden | Shown | Hidden |
| Image folder + encoder | Hidden | Hidden | Shown |

**Training modes:**
| Display Name | Internal Key | Description | Requirements |
|-------------|-------------|-------------|-------------|
| Basic | Basic | User trains on selected data file. Backend can route to full fine-tune or LoRA path | STUDENT route + data file |
| AI-Guided | AI-Guided | TRAINER generates/teaches curriculum for STUDENT from user topic and optional supplement data | TRAINER + STUDENT routes + topic/goal |
| Image | Vision | Vision training on image-text pairs from selected folder | STUDENT route + image folder |

**Contract note (intentional):** The FORGE page contract is now these 3 modes. Legacy mode names (Self Study, Dialogue, DPO, LoRA, Evolutionary, RLHF, Self-Play) are not part of the primary FORGE UI contract.

**Backend-only modes (fully implemented but not in dropdown):**
| Display Name | Internal Key | Description | Requirements |
|-------------|-------------|-------------|-------------|
| RLHF | RLHF | Two-phase: trains reward model on preference data, then PPO-trains STUDENT against it with KL penalty | STUDENT route + .jsonl data file |
| Self-Play | SelfPlay | TRAINER scores STUDENT responses as reward signal, REINFORCE policy gradient updates | TRAINER + STUDENT routes |

**AI-Guided validation:** If topic/goal is empty, training does not start and the output log explicitly tells the user what to fill in and why.

**AI-Guided execution note:** The current backend path goes through the adaptive pipeline. Supplement selection and start-stage selection are now wired. The pipeline still intentionally auto-chains remaining stages after the selected start point.

**Migration note (Completed March 9, 2026):** All GUI tests have been updated to validate the 3-mode contract. Test class renamed from `TestRLHFSelfPlayDropdown` to `TestTrainingModes` with 9 tests covering Basic, AI-Guided, and Image modes. Legacy mode references in tests removed. Code passes linting and tests pass.

**Forge status (March 9, 2026):** Core FORGE work is complete for day-to-day training: 3-mode UX, automatic LoRA routing for large models, and automatic before/after perplexity logging are all active. Remaining roadmap items are tool success-rate persistence and Discovery mode orchestration.

**Reality check (March 10, 2026):** The visible FORGE UI is mostly aligned with the backend now:
- The AI-Guided supplement dropdown is wired through the adaptive/tool paths.
- Stage buttons define the adaptive start stage, then the pipeline intentionally continues forward through later stages.
- The adaptive plan records test scores, but progression still auto-advances rather than using score thresholds.
- The old focus-field widget is gone; Training Topic + Training Brief are the active control surfaces.

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
| HISTORY button | Dark button | Displays past training runs from data/training_history.json in the log. Shows model name, mode, epochs, final loss, and timestamp for each run | gui_pages.py |
| SAVE CHECKPOINT button | Dark button | Saves STUDENT model to models/checkpoints/ | gui_pages.py |
| LOAD CHECKPOINT button | Dark button | Loads a checkpoint back into the STUDENT model slot | gui_pages.py |
| Topic entry | Text input | Topic for WEB LEARN search | gui_pages.py |
| WEB LEARN button | Dark themed button | Searches DuckDuckGo via web_utils, fetches pages, TRAINER generates Q/A pairs using trainer system prompt. Updates progress bar. If Auto-train is checked, starts training on completion | gui_pages.py |
| Max pages entry | Text input, default "3" | How many web pages to read | gui_pages.py |
| Vocabulary size entry | Text input, default "8000" | BPE tokenizer vocabulary size | gui_pages.py |
| TRAIN TOKENIZER button | Dark button | Trains a BPE tokenizer on selected data | gui_pages.py |
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

### Training Card
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Training card | HUDFrame with border | Contains background training options | gui_pages.py |
| "TRAINING" | Bold section header | Labels the card | gui_pages.py |
| Description | Tiny dim text | "Background training options for chat sessions." | gui_pages.py |
| Learn while chatting | CTkCheckBox | When enabled, chat exchanges are fed to the background trainer so the AI improves over time. Requires TRAINER route assigned. Persisted to gui_settings.json | gui_pages.py |

**Learn while chatting behavior:** Each user↔AI exchange during normal chat is captured and fed to the `BackgroundTrainer` via `_feed_background_trainer()` in LogicMixin. The background trainer accumulates exchanges and periodically runs a short SFT step on the STUDENT model. This causes the AI to gradually learn from its own conversations. Toggling the checkbox immediately saves the preference to `data/gui_settings.json["learn_while_chatting"]`.

### Mod Info Card
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Mod card | HUDFrame with border | Shows installed mod modules | gui_pages.py |
| "MOD MODULES" | Bold header | Labels the card | gui_pages.py |
| Description | Tiny text | "Mods are plugin programs that connect to the engine. They auto-start when the app launches." | gui_pages.py |
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

## FILE MAP

| File | Lines | What It Controls |
|------|-------|-----------------|
| widgets.py | 666 | Theme-driven C_* color constants, fonts, reload_theme() for live switching, widget classes (HUDFrame, StatusDot, NavButton, SectionLabel, SelectableLabel, ToggleButton, StatusBar, CollapsiblePanel, SelectableTextbox, Tooltip), factory functions (themed_entry, themed_dropdown, themed_scroll) |
| desktop.py | 772 | Window shell: header (pin toggle, shortcuts inline dropdown), nav rail (collapsible via grid_columnconfigure), status bar, label copy (right-click on CTkLabels that still use wraplength), auto-start mods, status ticker, display name loading, Escape-to-stop binding, TTS lifecycle (init/shutdown), deferred boot scanning, window geometry persistence (save/restore), live theme switching (_apply_theme_live, _retheme_tree), parent watchdog. Inherits 7 mixins. |
| gui_pages.py | 1102 | Page builders: CORE (fullscreen + web toggle + token counter + resizable sidebar + clickable history + media tags + STOP/EDIT buttons + auto-expanding input + reasoning display), MODELS (cards with param count + tooltips + COPY/RENAME/DELETE + IMPORT/DOWNLOAD + NATIVE/EXTERNAL tags), ROUTER. Inherits ForgePageMixin + ConfigPageMixin |
| gui_pages_forge.py | 815 | FORGE page layout: 3-mode radio-card selector (`Basic`, `AI-Guided`, `Image`) + reasoning checkbox + mode-specific sections + Auto-train checkbox + stage buttons with tooltips + CollapsiblePanel tools + vision browse + LoRA advanced subsection + student param count label + hyperparameter presets + progress bar + loss chart canvas panel + rolling best K + queue/plan/dataset buttons (ForgePageMixin) |
| gui_pages_config.py | 585 | CONFIG page layout: generation parameters, paths, display names, live theme picker (no restart), font size control, learn-while-chatting toggle, backup/restore (ConfigPageMixin) |
| gui_docs_page.py | 798 | DOCS page: documentation browser with search filter, path tooltips, file editor with path label and stats footer, inline file rename, blank doc creation, unsaved change detection, Ctrl+S shortcut, Ctrl+F find bar with match navigation, right-click context menu, auto-save (30s timer), notes category, CRUD operations |
| gui_logic.py | 881 | Logic hub: config, model loading, routes, display names, toggles, path settings, web access toggle, GUI context, CMD activity pipeline, GGUF param estimation. Inherits LogicChatMixin + LogicMediaMixin |
| gui_logic_chat.py | 1107 | Chat messaging, session management, AI session naming, duplicate save prevention, typewriter, file attachment, history, send guard, stop generation, message editing, auto fact extraction, reasoning display, token counter (LogicChatMixin) |
| gui_logic_media.py | 571 | Media rendering, voice I/O, TTS via pyttsx3, inline media rendering with image cap, chat input auto-resize (LogicMediaMixin) |
| gui_forge.py | 894 | Forge hub: training setup, shared utils, dispatch. Inherits ForgeTrainingMixin + ForgeAdvancedMixin + ForgeAdaptiveMixin + ForgeNewModesMixin + ForgeToolsMixin + ForgeModelsMixin + ForgeQueueMixin |
| gui_forge_training.py | 764 | Basic training modes: solo, DPO, vision, LoRA, CPU-first student loading, loss curves (ForgeTrainingMixin) |
| gui_forge_advanced.py | 1083 | Advanced training: evolutionary, guided (with curated dataset auto-accumulate), dialogue (TRAINER↔STUDENT conversation with corrections + reinforcement + transcript saving) (ForgeAdvancedMixin) |
| gui_forge_adaptive.py | 617 | Adaptive pipeline: TC-C3 continuous adaptive loop, SA-B auto-chain stages, SA-C saveable/resumable JSON plan (ForgeAdaptiveMixin) |
| gui_forge_new_modes.py | 313 | New training modes: RLHF (2-phase: reward model → PPO), Self-Play (TRAINER as reward) (ForgeNewModesMixin) |
| gui_forge_tools.py | 980 | Forge tools: data gen (with curated dataset auto-accumulate + reasoning flag), evaluate, web learn (with curated dataset auto-accumulate), checkpoints, tokenizer training, cards, auto-train, forge param count display, loss chart canvas drawing (ForgeToolsMixin) |
| gui_forge_models.py | 584 | Model ops: import, create, copy, rename, delete, quantize, GGUF export, HuggingFace download, model context rename (ForgeModelsMixin) |
| gui_forge_queue.py | 390 | Queue, overnight plan, curated dataset GUI callbacks: add/show/run queue, save/load plan, review/approve dataset, dataset auto-accumulate helper (ForgeQueueMixin) |
| gui_mods.py | 178 | Mod subprocess lifecycle (start/stop/auto-start, _launch_mod) |
| gui_mod_page.py | 279 | Per-mod page builder from mod.json (dynamic UI rendering incl. dropdown and checkbox widgets) |
| gui_cmd_page.py | 1205 | Dual-mode terminal: SYSTEM shell + ENGINE commands + AI ACCESS + activity monitor + live status strip + 11 info commands (incl. profiles) |
| media.py | 445 | Chat media support: image/GIF/video detection, Pillow loading, GIF animation, video thumbnails (OpenCV), URL detection, clickable links, MAX_CHAT_IMAGES cap |
| scanners.py | 557 | Filesystem scanning, config limits, ROUTE_KEYS, PATH_SETTINGS, path persistence, scan_docs, trainer docs, param counting (zipfile peek + file-size heuristic), target_size display, _format_param_count |
| themes.py | 143 | Color theme system: Theme frozen dataclass (20 fields), 4 presets (dark/midnight/carbon/solarized), load/save preference, theme API |

### Mixin Inheritance Order
```
EnigmaGUI(DocsPageMixin, ForgeMixin, ModMixin, ModPageMixin, CMDPageMixin, LogicMixin, PagesMixin, ctk.CTk)
```

ForgeMixin inheritance:
```
ForgeMixin(ForgeTrainingMixin, ForgeAdvancedMixin, ForgeAdaptiveMixin, ForgeNewModesMixin, ForgeToolsMixin, ForgeModelsMixin, ForgeQueueMixin)
```
