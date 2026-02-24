# GUI Reference - Every Element Explained

This document maps every visible element in the Enigma Engine desktop GUI,
what it is, where it lives in code, and what it does.

Use this to decide what to change, move, remove, or redesign.

---

## WINDOW

| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Window title | "ENIGMA ENGINE" in OS title bar | Identifies the app | desktop.py |
| Window size | 1440x900 default, 800x500 minimum | Sets the app dimensions. Resizable down to 800x500 | desktop.py |
| Background color | Very dark black (#080808) | Base color behind everything | desktop.py |

---

## HEADER BAR (top strip across the window)

| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Header bar | Dark panel (#0e0e0e), 56px tall, 1px border bottom | Contains title and model status | desktop.py |
| "ENIGMA" title | Large bold bright text (#e8e8e8) | First half of app branding | desktop.py |
| " ENGINE" title | Large bold silver text (#8B95A5) | Second half of app branding | desktop.py |
| "1.1.0" | Tiny dim text (no "v" prefix) | Version number | desktop.py |
| Status dot | Colored circle on LEFT of status text | Gray = no model, orange = loading, green = loaded, red = error | desktop.py |
| Model status label | Text like "NO MODEL" or "model_name // CUDA" | Shows current model state (no brackets) | desktop.py |

---

## NAV RAIL (left sidebar, 170px wide)

| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Nav rail | Dark panel (#0e0e0e) with right border only | Contains page buttons and brick launchers. Collapsible via header toggle | desktop.py |
| Nav toggle | Arrow button (◀/▶) in header | Collapses nav to 0px (fully hidden) or expands to 170px with labels. Silver when expanded, dim when collapsed | desktop.py |
| CORE button | NavButton with left-edge accent bar | Switches to CORE page (chat) | desktop.py |
| CMD button | NavButton with left-edge accent bar | Switches to CMD page (command terminal) | desktop.py |
| DOCS button | NavButton with left-edge accent bar | Switches to DOCS page (documentation + profiles) | desktop.py |
| MODELS button | NavButton with left-edge accent bar | Switches to MODELS page (model files) | desktop.py |
| ROUTER button | NavButton with left-edge accent bar | Switches to ROUTER page (route assignments) | desktop.py |
| FORGE button | NavButton with left-edge accent bar | Switches to FORGE page (training) | desktop.py |
| CONFIG button | NavButton with left-edge accent bar | Switches to CONFIG page (settings) | desktop.py |
| Separator line | 1px horizontal rule | Divides nav pages from bricks section | desktop.py |
| "BRICKS" label | Tiny dim text | Labels the brick section | desktop.py |
| Brick NavButtons | NavButton, one per brick | Switches to that brick's dedicated page | desktop.py |
| "(none)" | Tiny dim text | Shown if no bricks found in bricks/ folder | desktop.py |

**Nav button behavior:** Active button shows a 3px left-edge accent bar (silver) and bright text on surface background. Inactive buttons show dim text with no bar. Only one active at a time. No symbols or icons, no "NAV" label.

**Nav collapse:** Click the ◀ arrow button in the header to collapse the nav rail to 0px (fully hidden via grid_remove). Content expands to fill the full width. Click ▶ to restore full 170px with labels.

**Brick behavior:** Each brick is a full page. All bricks auto-start when the app launches. Brick pages show info, commands, UI widgets from brick.json, and an output log.

---

## PAGE: CMD (Dual-Mode Terminal)

Dual-mode terminal with SYSTEM (real PowerShell) and ENGINE (AI command registry) modes.
AI ACCESS toggle lets the AI execute real system commands when enabled.

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
| divider | Border accent (#2e2e2e) | Spacing dividers |

**ENGINE mode commands:**
| Command | Description |
|---------|-------------|
| help | Show all engine commands and terminal info |
| clear / cls | Clear the terminal output (works in both modes) |
| history | Show command history |
| ask \<question\> | Send question to AI, auto-execute any [CMD] blocks in response |
| config.get/set/list | Get, set, or list config values |
| model.info/list | Model information and listing |
| file.read/write/list | File operations |
| system.info | System information |
| (all engine commands) | Full command registry available |

**AI command execution flow:**
1. AI responds to "ask" with text and optional [CMD] blocks
2. Each [CMD] block is tried as an engine command first
3. If unknown and AI ACCESS is ON, the command runs as a real system command
4. If unknown and AI ACCESS is OFF, an error is shown

**Keyboard:** Up/Down arrows navigate command history. Enter executes.

---

## BRICK PAGES (one per brick, built from brick.json)

Each brick gets its own page, dynamically built from the brick's `brick.json` config.

### Top Bar
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Brick name header | Section label | Page title (brick name) | gui_brick_page.py |
| START button | Silver accent button | Starts the brick subprocess | gui_brick_page.py |
| STOP button | Dark button, disabled until running | Stops the brick subprocess | gui_brick_page.py |
| Status dot | Colored circle | Gray = stopped, green = running | gui_brick_page.py |
| Status label | Text "RUNNING" or "STOPPED" | Shows brick process state | gui_brick_page.py |

### Left Column: Info + Commands + Interface
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Info card | HUDFrame | Shows brick name, description, version, port | gui_brick_page.py |
| Commands list | One row per command | Shows command name (silver) and description (dim) from brick.json | gui_brick_page.py |
| Interface card | HUDFrame with accent border | Renders UI widgets from brick.json "ui" section | gui_brick_page.py |
| text_input widgets | CTkEntry | Text input fields defined in brick.json | gui_brick_page.py |
| text_area widgets | CTkTextbox | Multi-line text areas defined in brick.json | gui_brick_page.py |
| number widgets | CTkEntry (numeric) | Number inputs with defaults from brick.json | gui_brick_page.py |
| button widgets | CTkButton (silver accent) | Sends the mapped command to the brick | gui_brick_page.py |

### Right Column: Output Log
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Output header | Green section label | Labels the log | gui_brick_page.py |
| Output log | CTkTextbox, green text | Shows brick start/stop events, command sends, responses | gui_brick_page.py |

---

## STATUS BAR (bottom strip across the window)

| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Status bar | 30px tall strip at very bottom | Three-section info bar | desktop.py |
| Left section | Text like "READY" or "MODEL_NAME LOADED" | Shows current state | desktop.py |
| Center section | Text like "CPU" or "CUDA // 8.0 GB VRAM" | Shows compute device | desktop.py |
| Right section | Text like "UPTIME 00:05:23" | Shows app uptime, updates every second | desktop.py |

---

## PAGE: CORE (Chat Interface)

### Top Bar
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| "NEURAL INTERFACE" header | Section label with accent line | Page title | gui_pages.py |
| Active profile label | Tiny purple text on right | Shows "PROFILE: name" when a profile is active (set from DOCS page) | gui_pages.py |
| Fullscreen toggle | Small button (\u26f6 icon) | Enters fullscreen chat — hides header, nav, status bar. CORE page covers the entire GUI. Dim when normal, accent when active | gui_pages.py |
| Sidebar toggle | Small button (\u25e8 icon) | Hides or shows the sidebar. When hidden, chat expands to full width. Silver when visible, dim when hidden | gui_pages.py |

### Chat Area (left column)
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Chat frame | HUDFrame with border | Container for chat display | gui_pages.py |
| Chat display | Text box, word wrap, 12px left/right margins | Shows conversation: purple for YOU, silver for ENIGMA, dim for system, red for errors | gui_pages.py |
| File indicator | Tiny cyan text above input | Shows attached filename when a file is attached | gui_pages.py |
| Thinking indicator | Tiny dim text, right side, fixed 140px width | Shows "PROCESSING..." with animated dots while AI generates response. Fixed width prevents layout shift | gui_pages.py |
| Chat input | Multi-line text box, 56px tall | Type messages here. Enter sends. Shift+Enter for newline | gui_pages.py |
| SEND button | Green button, right of input | Sends the message (or Enter key). Tooltip: "Send message (Enter)" | gui_pages.py |
| NEW button | Dark button, below SEND | Starts a new conversation: clears chat, history, and KV cache | gui_pages.py |
| Utility toolbar | Row below input | Contains voice, mic, and attach buttons — separated from SEND to prevent misclicks | gui_pages.py |
| Voice toggle | Square toggle button in toolbar | Turns voice output on/off. Green when on, gray when off. Tooltip: "Voice output on/off" | gui_pages.py |
| Mic button | Square button (\U0001f3a4 icon) in toolbar | Voice input: click to start recording, click again to cancel instantly. Uses listen_in_background() with stopper. Turns red while recording. Tooltip: "Voice input (mic)" | gui_pages.py |
| Attach button | Square button in toolbar | Opens file picker to attach a text file to next message. Tooltip: "Attach file" | gui_pages.py |
| Web search button | Square button (🌐 icon) in toolbar, cyan accent | Opens web search dialog — enter a query, results inserted into chat context. Uses DuckDuckGo via search.web engine command. Tooltip: "Web search" | gui_pages.py |

### Sidebar (right column, weight-based)

The sidebar contains two **collapsible panels** (CollapsiblePanel widget). Click the header to expand/collapse. When one is collapsed, the other takes all available space. When both are collapsed, only the header rows are visible.

| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| HISTORY panel header | CollapsiblePanel, purple chevron + title | Click to expand/collapse the history section | gui_pages.py |
| History list | Text box with word wrap (inside panel) | Shows saved sessions: name, message count, date | gui_pages.py |
| SAVE button | Small white button | Saves current chat as a session JSON to memory/ | gui_pages.py |
| LOAD button | Small dark button | Opens file picker to load a saved session JSON | gui_pages.py |
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
| Size dropdown | Option menu | Pick: pi_zero/nano/tiny/small/medium/large | gui_pages.py |
| CREATE button | Silver accent button | Creates a fresh untrained model | gui_pages.py |

### Model Cards (scrollable)
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Model name | Bold text | File name of the model | gui_pages.py |
| Model info | Tiny text like "PTH // 42 MB" | Format and file size | gui_pages.py |
| DELETE button | Dark button, hover red | Deletes the model file after confirmation | gui_pages.py |

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
| TRAINER route card | HUDFrame with dot, name, description, dropdown, status | Assign a model to handle training runs | gui_pages.py |
| Brick route cards | One per brick, auto-generated | Assign a model to each brick independently | gui_pages.py |
| Route status label | Text on right of each card | Shows assigned model name (green), "Running" for bricks, or "No model" (dim) | gui_pages.py |
| Route status dot | Colored circle on left of card | Green = model assigned or running, orange = model assigned but brick stopped, gray = nothing | gui_pages.py |
| Model dropdown | CTkOptionMenu per route card | Select which model to assign to this route (None clears it) | gui_pages.py |

**Route behavior:** Each route (CHAT, TRAINER, and each brick) has its own model dropdown. Selecting a model from the CHAT dropdown loads it into the engine. Selecting "None" unloads it. Other routes track the assignment for display. Brick routes also show running/stopped state. All route statuses update live via `_update_route_status()` in gui_logic.py. Assignments are stored in `self.route_assignments` dict.

---

## PAGE: FORGE (Training)

### Top Bar
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| "THE FORGE" header | Section label | Page title | gui_pages.py |

### Left Column: Controls
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Controls panel | HUDFrame with scrollable inner | Contains all training settings | gui_pages.py |
| "TRAIN MODEL" heading | Bold text with underline | Section divider | gui_pages.py |
| Data source dropdown | Option menu | Pick which file from data/ to train on (shows name + size). Selecting a file loads it into the Data Editor | gui_pages.py |
| Model size dropdown | Option menu | Pick size: pi_zero/nano/tiny/small/medium/large | gui_pages.py |
| Epochs entry | Text input, default "10" | How many training passes | gui_pages.py |
| Batch size entry | Text input, default "4" | Training batch size | gui_pages.py |
| Learning rate entry | Text input, default "0.0001" | Training learning rate | gui_pages.py |
| START TRAINING button | Silver accent button | Begins model training in background thread | gui_pages.py |
| STOP button | Dark button, disabled until training starts | Stops training after current epoch | gui_pages.py |
| "TRAIN TOKENIZER" heading | Bold text with underline | Section divider | gui_pages.py |
| Vocabulary size entry | Text input, default "8000" | BPE tokenizer vocabulary size | gui_pages.py |
| TRAIN TOKENIZER button | Silver accent button | Trains a BPE tokenizer on selected data | gui_pages.py |

### Center Column: Data Editor
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| "Data Editor" header | SectionLabel with cyan accent | Labels the editor panel | gui_pages.py |
| File name label | Tiny text | Shows the name of the currently loaded file, or "No file selected" | gui_pages.py |
| Data editor | CTkTextbox, word wrap | Editable text area showing the contents of the selected training data file | gui_pages.py |
| SAVE button | Green button | Writes the editor content back to the data file | gui_pages.py |
| NEW FILE button | Silver accent button | Creates a new .txt file in data/ (prompts for filename) | gui_pages.py |
| REFRESH button | Dark button | Re-scans data/ and updates the data source dropdown | gui_pages.py |

**Data editor behavior:** When a data source is selected from the dropdown, its content loads into the editor. The first file is auto-loaded when the page builds. Edit the text and click SAVE to persist changes. NEW FILE creates a blank file and selects it. REFRESH updates the dropdown if files were added/removed externally.

### Right Column: Log
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Log panel | HUDFrame, right column | Contains training output log | gui_pages.py |
| "OUTPUT LOG" header | Green section label | Labels the log | gui_pages.py |
| Training log | Text box, green text | Shows epoch loss, training status, errors, completion info | gui_pages.py |

---

## PAGE: DOCS (Documentation + Profiles)

Documentation browser with file editor and profile management. Files are organized into categories: Guides (from information/), Profiles (from profiles/), and Brick docs (from bricks/<id>/docs/).

### Top Bar
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| "Documentation" header | SectionLabel | Page title | gui_docs_page.py |
| + NEW button | Small button, silver text | Creates a new .md file in information/ (prompts for name) | gui_docs_page.py |
| + PROFILE button | Small button, purple text | Creates a new AI profile JSON in profiles/ (prompts for name) | gui_docs_page.py |

### Left Column: File Browser
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Browser frame | HUDFrame with scrollable inner | Contains categorized file list | gui_docs_page.py |
| Category headers | Tiny colored labels (GUIDES=cyan, PROFILES=purple, BRICK:X=silver) | Group files by source | gui_docs_page.py |
| File entries | Clickable buttons, one per file | Click to load file into editor. Highlights when selected | gui_docs_page.py |

### Right Column: Editor
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Filename label | Small text above editor | Shows current file name, turns green on save, red on delete | gui_docs_page.py |
| SAVE button | Green button | Writes editor content back to the file | gui_docs_page.py |
| DELETE button | Red text button | Deletes the current file after confirmation dialog | gui_docs_page.py |
| RELOAD button | Dim button | Refreshes the file browser (re-scans all sources) | gui_docs_page.py |
| Editor textbox | CTkTextbox, word wrap, full height | Edit file content. Supports .md, .txt, and .json files | gui_docs_page.py |

**File sources:**
| Category | Source Directory | File Types | Color |
|----------|-----------------|------------|-------|
| Guides | information/ | .md, .txt | Cyan |
| Profiles | profiles/ | .json | Purple |
| Brick docs | bricks/<id>/docs/ | .md, .txt | Silver |

**Default guide files:** how_the_ai_works.md, training_guide.md, commands_reference.md, getting_started.md, prompts_guide.md

**Profile creation:** + PROFILE creates a JSON with template fields: name, id, system_prompt, personality (tone, verbosity, formality), generation (temperature, top_p, top_k, max_tokens, repetition_penalty).

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

### Profile Card
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Profile card | HUDFrame with purple border | Shows active profile info | gui_pages.py |
| "ACTIVE PROFILE" | Purple header text | Labels the card | gui_pages.py |
| Profile name | Body text | Shows name of currently selected profile or "None selected" | gui_pages.py |
| Description text | Tiny dim text | Explains how to switch profiles (from DOCS page) | gui_pages.py |

### Brick Info Card
| Element | What It Is | What It Does | File |
|---------|-----------|-------------|------|
| Brick card | HUDFrame with border | Shows installed brick modules | gui_pages.py |
| "BRICK MODULES" | Bold header | Labels the card | gui_pages.py |
| Description | Tiny text | "Bricks are plugin programs that connect to the engine. They auto-start when the app launches." | gui_pages.py |
| Brick rows | One row per brick | Shows "name vX.X" and description snippet | gui_pages.py |

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
| profiles_dir | Profiles Directory | profiles/ |
| sessions_dir | Sessions Directory | data/sessions/ |
| memory_dir | Memory Directory | memory/ |
| bricks_dir | Bricks Directory | bricks/ |

**Path behavior:** Saved paths are stored in `data/path_settings.json`. On startup, saved overrides are loaded into the entry fields. RESET clears all overrides and restores defaults. The browse button opens a native directory picker. Path constants and persistence functions live in scanners.py (`PATH_SETTINGS`, `load_path_settings()`, `save_path_settings()`, `get_path()`).

---

## COLOR PALETTE (widgets.py)

All colors defined as constants at the top of widgets.py (lines 15-32). Changing a constant updates every widget that uses it.

| Name | Hex | Used For |
|------|-----|----------|
| C_BG | #080808 | Window background |
| C_PANEL | #0e0e0e | Panel/card backgrounds |
| C_SURFACE | #181818 | Button hover, slightly lighter areas |
| C_INPUT | #1c1c1c | Text input backgrounds |
| C_ACCENT | #8B95A5 | Primary silver/gray: titles, active states, highlights |
| C_ACCENT_DIM | #2a2a2a | Borders, inactive accents |
| C_ACCENT_MUTED | #3d3d3d | Hover states, secondary accents |
| C_PURPLE | #a855f7 | User messages, profiles |
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

All fonts are Consolas monospace. Defined at lines 37-45.

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
| widgets.py | 495 | Colors, fonts, widget classes (HUDFrame, StatusDot, NavButton, SectionLabel, ToggleButton, StatusBar, CollapsiblePanel, SelectableTextbox, Tooltip), factory functions (themed_entry, themed_dropdown, themed_scroll) |
| desktop.py | 398 | Window shell: header, nav rail (collapsible via grid_columnconfigure), status bar, label copy (right-click on CTkLabels), auto-start bricks, status ticker, display name loading. Inherits 7 mixins. |
| gui_pages.py | 1084 | Page builders: CORE (with fullscreen + web search), MODELS, ROUTER, FORGE, CONFIG (with directory paths + display names) |
| gui_logic.py | 1043 | Chat, sessions, profiles, route assignment, model loading, per-model context, voice input, path settings, display names, web search |
| gui_forge.py | 429 | Training, tokenizer training, model create/delete, data file editing |
| gui_bricks.py | 171 | Brick subprocess lifecycle (start/stop/auto-start, _launch_brick) |
| gui_brick_page.py | 321 | Per-brick page builder from brick.json (dynamic UI rendering) |
| gui_cmd_page.py | 577 | Dual-mode terminal: SYSTEM shell + ENGINE commands + AI ACCESS |
| scanners.py | 296 | Filesystem scanning, config limits, ROUTE_KEYS, PATH_SETTINGS, path persistence, scan_docs |

### Mixin Inheritance Order
```
EnigmaGUI(DocsPageMixin, ForgeMixin, BrickMixin, BrickPageMixin, CMDPageMixin, LogicMixin, PagesMixin, ctk.CTk)
```
