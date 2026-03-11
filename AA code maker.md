# AA Code Maker - Working Rules

Read this file before writing code. Update after each session.

**Living document**: Add to "Recent Learnings" as you work. This accumulates project knowledge without repeating the same mistakes.

If this file and `SUGGESTIONS.md` conflict, follow the most recent practical behavior in code and tests.

---

## 0. Special Spot - Verification First

Critical reminder:
- Always inspect current code paths and related tests before editing.
- Never assume old behavior/contracts are still active.
- If uncertain, stop and verify with source reads before writing changes.

---

## 1. RULES (Non-Negotiable)

1. Read this file and SUGGESTIONS.md before doing anything
2. Read target files before editing — verify imports, attributes, and APIs actually exist
3. Write or update tests first, then build features
4. Search for existing implementations before building anything new
5. No cloud dependencies, no new packages unless necessary
6. Challenge ideas that don't solve the real problem — do not be a yes man
7. Reuse existing patterns — no duplicate ways to do the same thing
8. Finish with zero issues, warnings, or broken code
9. Keep it simple — do not over-engineer
10. Make it realistic and practical — working code over perfect code

---

## 2. DO NOT

**Code changes:**
- Do not delete or rewrite existing code without being asked
- Do not add untested code
- Do not duplicate existing functionality — search first
- Do not leave debug prints (`print`, `console.log`, etc.)
- Do not add unused imports or make random formatting changes
- Do not hardcode file paths or magic constants

**Architecture:**
- Do not introduce new dependencies without approval
- Do not change file structure or architecture unprompted
- Do not add global state changes without a thread lock
- Do not add backward-compatible shims

**UI:**
- Do not use sliders — use numeric input only
- Do not add windows/popups unless specified (use inline confirmation)
- Do not add symbols or special characters in names

**Behavior:**
- Do not declare things broken without reproducing them
- Do not create theoretical problems you're not experiencing
- Do not audit for issues you didn't ask about
- Do not assume complexity equals bad

---

## 3. Engineering Patterns (Stable)

These are proven and should remain:

- **Atomic saves**: Use `atomic_torch_save()`, `atomic_write_text()`, `atomic_write_json()` everywhere.
- **Boundaries**: `core/` never imports `gui/`. No circular deps.
- **Deferred imports**: `_ensure_*_imports()` for PyTorch/heavy libs saves 540MB RAM.
- **Runtime config**: Read `pad_token_id`, `max_seq_len` from tokenizer/config, not hardcoded constants.
- **Shared logic**: `_prepare_chat()` keeps chat/stream paths aligned. Extract common code instead of duplicating.
- **Thread safety**: Use `threading.Lock` when GUI + background threads share mutable objects.
- **Security defaults**: No command execution tools by default, no permissive CORS, sanitize AI-generated args.
- **Concurrency guards**: `_model_op_busy()`, `_is_generating` flags prevent race conditions.

---

## 4. Recent Learnings (Add New Discoveries Here)

When something fails or succeeds, record it here. Prune to keep focused.

### Things That Failed
- ❌ Mixing old and new FORGE contracts during GUI migration created test confusion
- ❌ Not updating all related tests when changing core UI led to cascade failures
- ❌ Leaving legacy dispatch assertions in older tests caused false regressions after the 3-mode FORGE contract landed
- ❌ Applying generic command arg sanitization to `code.run` payloads broke fenced/multiline Python (` ```python ... ``` ` became malformed), causing syntax errors in CMD-page AI execution.
- ❌ Made a change before fully checking current implementation state. Result: stale assumptions and avoidable rework. Fix: read code + tests first, every time.
- ❌ **FORGE contract drift** (March 10, 2026): The simplified AI-Guided UI implied supplement selection, stage control, and adaptive gating that the backend did not actually honor. Fix: verify the visible UI contract against the executing path before calling the feature done.
- ❌ **Partial UI migration side effect** (March 10, 2026): Removed `forge_focus_field` from UI but left a `focus_field` pass-through in `_start_adaptive_training`, causing runtime `NameError` when adaptive pipeline starts. Fix: define/normalize optional legacy vars explicitly during transition or remove all call-site references together.

### Things That Worked
- ✅ **Gaming preset needed live enforcement, not just next-launch defaults** (March 10, 2026): The existing CONFIG "Apply Gaming Mode" button only flipped saved settings, so runtime overhead stayed high while the app was open. Fix: derive a live low-overhead mode from the saved preset (`no autoload`, `no mod autostart`, `unload on minimize`, `learn while chatting off`), slow nonessential UI timers, skip exact param counting, and stop router background training when chat-learning is disabled.
- ✅ **FORGE realism audit** (March 10, 2026): Checking the actual adaptive path against the new 3-mode UI exposed the real gaps quickly: supplement still read `train_data_var`, stage buttons were not authoritative, and progression still auto-advanced. This was more valuable than assuming the UI contract was already true.
- ✅ **FORGE tool-path wiring cleanup** (March 10, 2026): Updated `Generate Data`/`Web Learn` to route generated files to the active mode selector (AI-Guided supplement vs Basic data), removed dead `forge_focus_field` reads in active tool flows, and fixed adaptive start to avoid undefined `focus_field`. Added regression tests for these three-mode connections.
- ✅ **Next FORGE cleanup target is structural, not cosmetic** (March 10, 2026): After the runtime wiring fixes, the main remaining work is removing legacy 9-mode config baggage from `gui_forge.py` and adding measurable tool-success evaluation. This is better value than adding more visible FORGE controls right now.
- ✅ **Chat file-link QoL fix** (March 10, 2026): Deferred command side effects until typewriter completion, then rendered explicit clickable `OPEN FILE:` entries in chat. This removed response interruption and made file outputs discoverable/openable immediately without guessing paths.
- ✅ **Lean suggestions doc policy** (March 10, 2026): Keeping `SUGGESTIONS.md` active-only (current state, chosen memory strategy, immediate priorities) reduced drift and made implementation decisions faster.
- ✅ **Processing indicator hard-lock fix** (March 10, 2026): Fixed remaining CORE input-bar movement by locking fixed-size `SelectableLabel` geometry (`pack_propagate(False)` / `grid_propagate(False)`) and preventing width recalculation on text updates when explicit width is set. Root cause was widget-level resize requests, not dot animation length.
- ✅ **Chat history memory leak fix** (March 10, 2026): Capped `self.history` at 500 messages to prevent unbounded RAM growth during long sessions. Pattern matches existing caps: images (200), input history (50), GIFs (5). Added `_trim_chat_history()` method called after each exchange. After overnight sessions, RAM was hitting 100% — now capped and stable. Full conversation still saved to disk in session files.
- ✅ **CMD code-run reliability fix** (March 10, 2026): `CommandRegistry.execute()` now preserves raw payload for `code.run` (no generic split/sanitize on code body), so fenced or multiline code executes correctly. Added regression tests for raw payload preservation and fenced code execution. CMD ask-mode now injects a stricter system policy to avoid speculative `[CMD]` calls and require valid arguments.
- ✅ **FORGE test contract cleanup** (March 10, 2026): Replaced obsolete `Train with AI` toggle tests with current 3-mode FORGE contract tests (`Basic`, `AI-Guided`, `Image`) and removed legacy `_training_mode_desc` assertion. This kept intent coverage while aligning tests to the actual UI architecture in `gui_pages_forge.py`.
- ✅ Auto-LoRA trigger: Detect model param count at training start, auto-select LoRA if > 7B. Cleaner than manual dropdown. Implemented `_get_model_param_count()` helper that loads model briefly to count params, then dispatches to `_start_lora_training()` or `_start_solo_training()`. Tests all pass.
- ✅ **Before/after evaluation**: Added `run_evaluation=True` to TrainingConfig. Trainer now evaluates model on test prompts before and after training, logs perplexity improvement. Created `training_evaluation.py` module with `evaluate_model()` and `evaluate_tool_usage()` functions. GUI logs evaluation results showing perplexity drop and improvement percentage. Tests confirm integration works.
- ✅ **Duplicate widget bug**: Removed accidental duplicate frame construction in gui_pages_forge.py (lines 209, 268, 421) that created orphan UI blocks bypassing mode-switching logic. Caused duplicate sections to appear and wrong sections visible in IMAGE mode.
- ✅ **Layout shift prevention**: Locking grid column dimensions with `grid_columnconfigure(1, minsize=140)` prevented "Processing..." text from resizing input bar horizontally. Widget dimensions alone aren't enough — grid must be locked.
- ✅ **Natural memory system**: Removed passive bulk memory injection from system prompt. Added `memory.search <query>` command for active retrieval. AI now searches memory contextually when needed instead of seeing all facts upfront. Fixed "scripted" feeling. Removes lines 158-166 in gui_logic.py (auto-injection), adds memory_search() handler in builtin_commands.py.
- ✅ **File access for AI**: Added file.read/file.write/file.append instructions to system prompt. AI can now create and edit documentation, notes, and guides in `information/` and `data/notes/` folders when asked. Always confirms before modifying existing files. Enables natural documentation writing in DOCS page.
- ✅ **Processing animation fix**: Changed animated dots from variable width (".", "..", "...") to fixed width ("   ", ".  ", ".. ", "...") by padding with spaces. Prevents layout shifts when animation cycles. Line 670 in gui_logic_chat.py now uses `"." * n + " " * (3 - n)` pattern.
- ✅ FORGE is now stable at the core level: 3-mode contract + Auto-LoRA + before/after perplexity evaluation all wired and tested; UI bugs fixed; remaining work is tool success metric persistence and Discovery mode
- ✅ Converging FORGE to 3 user-facing modes (Basic, AI-Guided, Image) reduced complexity significantly
- ✅ Keeping advanced options hidden in collapsible sections removed UI clutter without removing capability
- ✅ Reusing existing backend methods for new modes avoided duplication
- ✅ Defining one clear UI contract first, then migrating tests to match, is more maintainable than legacy compatibility
- ✅ Test-driven approach to migration: changed tests first to validate new contract, code followed naturally
- ✅ Linting after changes caught all issues early (ruff security checks pass)

---

## 5. TEST LOOP

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

## 6. WHEN STUCK

1. Stop — do not guess or hallucinate solutions
2. State clearly: what you know vs what you don't know
3. Ask the user for clarification on ambiguous requirements
4. If a fix breaks something else: revert completely and rethink — no patches on patches
5. Explain *why* when pushing back on an approach (with alternatives)

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
3. **Update "Recent Learnings"** in this file when something significant failed or worked
4. Update `SUGGESTIONS.md` only when active status, decisions, or near-term priorities change

### Priority Order (when rules conflict)
**Correctness > Simplicity > Performance > Style**

---

## 8. CODE QUALITY

**Before implementing:**
- Confirm the real problem: "What are you experiencing?"
- Propose 2-3 approaches when there are choices — let the user decide
- Consider edge cases and error paths from the start

**While coding:**
- Add clear comments only for non-obvious logic
- Cross-platform: must work on both Linux and Windows
- Everything should actually work — no placeholder implementations
- Think about ripple effects — changes often connect to multiple places

**After coding:**
- Update related files together (tests, docs, imports, callers)
- If changing core/, run full test suite
- Test before claiming something is broken
- Evaluate what you built and note anything worth recording

**Trust the codebase:**
- Do not remove comments or code without a reason
- Trust your existing code until proven otherwise
- Work on what was actually requested, not theoretical improvements
- Ignore the installed mods and base model state — not relevant to most GUI work

---

## 9. QUICK COMMANDS

```bash
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