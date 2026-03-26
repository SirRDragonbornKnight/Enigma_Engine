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
- Show the reasoning *before* writing code — explain why something is broken and what the fix approach is. Don't just jump to edits.

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
11. Trade study before coding — evaluate multiple approaches against cost, complexity, value, and risk, then pick or combine the best
12. Future-proof where cheap — design for tomorrow's scale when it costs nothing extra today, but don't build for problems that don't exist yet
13. Devil's advocate every proposal — actively look for reasons an approach will fail before committing to it
14. Decision support — when there are real choices, present options with tradeoffs and let the user pick
15. Fit the code — check if changes need adapting to match existing patterns, naming, and structure

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
- Do not worry about time or complexity

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

## 4. Learned Principles (Distilled from ~70 implementations)

Reusable patterns extracted from project history. Grouped by theme. (~96 implementations)

### Verification First
- **Read before editing.** Always inspect current code + related tests before making changes. Never assume old contracts still hold.
- **Verify before fixing.** Audit findings can be false positives. Read actual source code to confirm the issue is real before writing a fix.
- **Read reference implementations.** External repos reveal the real gap (e.g. 7 missing PPO components, not 1).
- **Always verify "can't do X because Y" comments** — the constraint may no longer apply (e.g. RMSNorm "circular import" comment was stale).

### Testing
- **Tests define the contract.** Change tests first to match the new behavior, then implement. Code follows naturally.
- **Structural tests work without GPU.** `inspect.getsource()` checks let you verify code paths without running GPU code.
- **After method decomposition,** grep tests for `getsource(ClassName.old_method)` and redirect to the new helper.
- **Embedding/output dimension changes ripple** to tests that check output shapes — expect to update assertions.

### Wiring & Config
- **Grep ALL call sites.** When adding a config field (e.g. AdamW betas), grep the entire codebase for all constructor calls, not just the file you're editing.
- **Save AND load.** When adding a field to a state object (dataclass, checkpoint, context), verify it appears in both the save path and the load path.
- **Search ALL consumers** when changing a core pattern (e.g. buffer → on-demand). Not just the primary class — also RewardModel, test helpers, etc.
- **`getattr(config, 'field', default)`** makes new config fields backward-compatible with old checkpoints.
- **UI controls must persist.** When adding widgets, verify they round-trip to disk. FORGE once had 17 widgets with no persistence.
- **When removing a widget,** remove all call-site references together or define a fallback. Orphaned references cause `NameError`.

### Training & Numerics
- **PyTorch `LambdaLR.__init__` calls `step()` internally** — which evaluates `lambda(0)`. Use `(step + 1) / warmup` to avoid zero LR on the first optimizer step.
- **Never call `encode()` to get a known special token's ID.** `encode(add_special_tokens=True)` wraps input in BOS/EOS. Use the named attribute directly.
- **RMSNorm needs fp32 upcast** in fp16/bf16 to prevent overflow/NaN.
- **AdamW `beta2=0.95`** for LM training (not PyTorch's default 0.999). Matches GPT/LLaMA convention.
- **Training data format must match inference format.** "User:/Assistant:" everywhere — not "Q:/A:" or "User:/AI:".
- **Data quality gates:** clean → validate → deduplicate. Never accept teacher output without checking.
- **Two-tier truncation:** message-count cap for disk, token-count for inference context window.

### GUI & Threading
- **`after(0, callback)` guarantees main-thread execution.** Don't add locks for state that's only accessed via tkinter scheduling.
- **Only lock mutable state that crosses thread boundaries.** Background → GUI reads need locks; main-thread-only state does not.
- **Thread-safe GUI updates:** `self.after(0, self._update_method)` from background threads.
- **Centralize widget setup in factory functions** (e.g. `themed_entry()`) so new instances get standard behavior for free.
- **Late-binding lambda in threaded error handlers:** `lambda: f(exc)` inside a try/except captures the variable name, not the value. Ruff flags this as F821. Fix: `msg = str(exc)` then `lambda m=msg: f(m)`.
- **CTkFrame defaults to 200×200px** when propagation is disabled. If you set `width=N` on a Frame subclass, also compute `height` from font metrics or you get a 200px tall label.
- **`CTkEntry.configure(fg_color="transparent")` crashes** even though the constructor accepts it. Use the parent's actual background color instead.
- **Tkinter `<Leave>` fires when entering child widgets.** Tooltip on a container frame cancels immediately when the cursor moves to any child. Fix: check `winfo_pointerx/y` against `winfo_rootx/y + winfo_width/height` to detect true leaves vs child transitions.
- **Fixed-width labels clip text at larger font offsets.** Use `grid_columnconfigure(col, minsize=N)` for column alignment instead of pinning `width=` on the label itself.

### Security & Boundaries
- **`core/` never imports `gui/`.** Wire cross-boundary features (like emotional hints) in the GUI layer.
- **Forbidden lists must block the primitive** (`__import__(`), not specific examples (`__import__('os')`).
- **Audit config values** to verify they're actually consumed by code, not just decorative defaults.
- **Path traversal `startswith` checks need `+ os.sep`.** Without it, `profiles/../evil` passes if there's a sibling dir like `profiles_evil/`. Found in 3 separate checks — always grep for all `startswith` path guards when fixing one.

### Multilingual & i18n
- **When English keyword lists gate behavior, add a non-Latin heuristic.** Count Latin vs non-Latin alphabetic chars — if non-Latin dominates, use the safe default (e.g. route to AI). English patterns still work for English; non-English gets the conservative path. No language detection dependency needed.

### Feature Gating
- **Default to disabled when model capability is uncertain.** Phase 5 monologue defaults to "disabled" because 125M model can't produce coherent reflections. Infrastructure is ready for when the model grows — no wasted work, no broken UX.
- **Quality gate AI-generated content before surfacing to users.** Heuristic scorer (coherence, length, variety) prevents garbage from reaching the UI. Threshold-based: below = store silently, above = show to user.
- **Idle detection reuses existing polling patterns.** `after(N, callback)` with monotonic timer + activity reset + double-trigger flag. Same approach as status ticker and background trainer.

### Research & Trade Studies
- **Audit existing implementations before searching for research.** Get exact algorithms, parameters, and line numbers for what EXISTS first. Then research becomes targeted ("what upgrades BM25 with k1=1.5?") instead of speculative ("what RAG improvements exist?").
- **Upgrades ≠ gaps.** Two different research categories: "things you don't have" (gap-fill) vs "things you have that could be better" (upgrade). A gap-fill adds new capability. An upgrade replaces an existing algorithm with its next-generation version. Both matter, but upgrades are lower risk because the integration path is known.
- **Cross-reference subagent findings.** Automated review agents report false positives — e.g. claiming DPO is "missing" when it exists in gui_forge_training.py. Always verify claims against actual source before acting on them.
- **Don't brainstorm research from memory.** Systematically enumerate every subsystem (attention, RoPE, optimizer, scheduler, loss, cache, sampling, etc.), audit what's implemented, THEN search for upgrades. Memory-based listing misses entire categories.

### Process
- **Implement first, audit second.** Build the feature, then systematically trace GUI → config → core → training → checkpoint to find wiring gaps.
- **Batch related fixes.** Group related stability/compatibility changes, test each independently.
- **Triage before implementing.** When facing a list of issues, verify each is real first. Saves wasting effort on non-issues.
- **When duplicated logic exists,** trace ALL callers before consolidating — including test helpers and edge-case paths.
- **Run lint with project config, not overridden flags.** `ruff check dir/` uses pyproject.toml ignores. `ruff check --select E,F,W` overrides them and reports intentionally suppressed rules. Always use the project config.

### Concurrency
- **`threading.Lock` is non-reentrant.** If `add()` holds the lock and calls `flush()` which also acquires it — deadlock. Split into public (locks) and private `_unlocked()` (caller holds lock). Or use `RLock`, but split is cleaner.
- **Guard all model forward-pass entry points** with `_generation_lock`. Adding `batch_generate` without the lock meant concurrent calls could corrupt the KV cache.
- **Never `await` inside a `threading.Lock`.** Even when the await "can't" suspend (unbounded queue), it's fragile. Use `put_nowait()` or switch to `asyncio.Lock` for async code.
- **If a loop reads flags under lock, writers must also use the lock.** `training_queue` had `_run_loop()` reading `_running`/`_paused` under `_lock` but `start()`/`stop()` writing them without it — classic stale-value race.
- **Lock scope: read under lock, write to disk outside.** `_save_state()` was reading `_next_id`/`_jobs` without `_lock` because callers released it first. Fix: acquire lock for the snapshot, then do I/O outside it — avoids holding the lock during slow disk writes.
- **`subprocess.PIPE` that is never drained will hang the child.** OS pipe buffer is ~64KB. If the child writes to stderr and no one reads, it blocks. Either close the pipe after startup, use `DEVNULL`, or spawn a drain thread.

### DPO / Preference Training
- **Prompt mask length must match the actual encoded prefix.** If `chosen_ids = encode(f"User: {prompt}\nAssistant: {response}")`, then `prompt_len` must come from `encode(f"User: {prompt}\nAssistant: ")` — not bare `encode(prompt)`. Off-by even a few tokens leaks prompt into the DPO loss.

### Context & Accumulation
- **Cap unbounded string accumulation.** `_history_summary` grew by concatenation every truncation event — never trimmed. Split on structural markers (e.g. `[Earlier conversation summary`), rebuild from newest to oldest within a char budget, drop oldest blocks. Same pattern applies to any accumulator that appends but never shrinks.
- **Structural tests catch mixin namespace collisions.** With 7+ mixins sharing `self`, scan all source files for `self.attr = `, group by mixin family, whitelist known cross-family attrs, fail on any new collision. The test IS the deliverable — zero existing bugs, but prevents future ones automatically.

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
- Mock heavy dependencies (torch, file I/O) — tests should run in ~17s total
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
1. Run `ruff check enigma_engine/` then `python -m pytest tests/ -v`
2. Run `python run.py` to verify
3. **Update "Learned Principles"** in this file when a reusable pattern or anti-pattern emerges
4. Update `SUGGESTIONS.md` when confirmed fixes, backlog items, or priorities change
5. Update `GUI_REFERENCE.md` only when visible GUI behavior has changed

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
- Change the code in any way needed to improve functionality

---

## 9. QUICK COMMANDS

```bash
ruff check enigma_engine/                            # Lint (run before every commit)
python -m pytest tests/ -v                           # Run all tests verbose
python -m pytest tests/ --tb=short -q                # Run all tests (compact output)
ruff check --fix enigma_engine/                      # Auto-fix safe lint issues
ruff check enigma_engine/                            # Run linter standalone
python run.py                                        # Show system info
python run.py --gui                                  # Launch desktop GUI
python run.py --serve                                # Start API server (port from CONFIG)
python run.py --serve --port 8080                     # Start API server on specific port
python run.py --train data/training.txt --epochs 10  # Train model
python run.py --train-tokenizer data/training.txt    # Train BPE tokenizer
python run.py --benchmark                            # Run coherence benchmark on default model
python run.py --benchmark --model models/my.pth      # Benchmark a specific model
python run.py --help                                 # Show all CLI options
```