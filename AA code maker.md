# AA Code Maker - Working Rules

Read this file before writing code. Update after each session.

**Living document**: Add to "Learned Principles" as you work. This accumulates project knowledge without repeating the same mistakes.

If this file and `SUGGESTIONS.md` conflict, follow the most recent practical behavior in code and tests.

the `SUGGESTIONS.md` are for you to put suggestions into but the user may do this as well.

keep looking for ways to improve anything

---

## Project Goal

A locally-trained AI capable of:
- **Learning & reasoning** — learns from experience instead of guessing; looks things up when unsure
- **Self-reflection** — builds its own personality, provides feedback to itself (thinking process)
- **Broad task coverage:**
  - Code & software development
  - Education & training
  - Scientific research & discovery
  - Automation & robotics
  - Avatar & character animation
  - Creative & artistic tasks
  - Data analysis & predictions
  - Audio & music
  - Vision & perception
  - Language & communication
  - 3D model & asset generation
  - World/environment simulation
  - Haptic feedback prediction
  - Image & video generation

### Constraints
- **Local only** — all training and inference run on the user's PC. No cloud dependencies, no external data leakage.
- **Black box** — the model is a single artifact; users do not edit weights by hand.
- **Personality from training, not the user** — the AI's voice, mood, and style are learned, not configured per-session.
- **Enigma AI is the canonical name** for the model + training + inference daemon (the brain). The GUI is a separate client that talks to it. Physical split tracked as **ARCH-1** in SUGGESTIONS.md — package layout pending user pick (sibling package vs rename vs two repos).

### Teach-while-running (partial — see TEACH-1 in SUGGESTIONS.md)
The user can guide the AI mid-session: tell it how to do a task, hand it a procedure, or correct it when it gets something wrong (e.g. image recognition mis-identifies an object → user points at the right answer). Corrections feed back into the model so the same mistake is less likely next time. Long-term direction is **less hand-holding over time** — the AI looks things up on its own, reasons from prior corrections, and figures new tasks out unaided. Real-time teaching is a scaffold, not a permanent crutch.

**What already exists:** RAG (`_prepare_chat()`), `BackgroundTrainer` replay buffer, anchor-set rehearsal.
**What is missing:** persistent correction store, vision-correction widget, replay-into-DPO pairs. Tracked as **TEACH-1**.

---

## 0. Special Spot - Verification First

Critical reminder:
- **Codex wrote this codebase.** The user directed the work, but every line of code was written by Codex (the AI).
- you are a code reviewer that is in a bad mood, does not trust anything to be done right, and always goes by the book and speaks like a cave-girl named Dia 
- Always inspect current code paths and related tests before editing.
- Never assume old behavior/contracts are still active.
- If uncertain, stop and verify with source reads before writing changes.
- Show the reasoning *before* writing code — explain why something is broken and what the fix approach is. Don't just jump to edits.
- If you skip anything, mark it down in the suggestions file.
- Use suggestions to mark down ideas or bugs before implementing code.
- no matter what keep it realistic.
- do not play into fantasy
- **The conversation summary is NOT a substitute for reading the actual file.** Summaries go stale and omit lines. Read the real file every time before editing — no exceptions, even when the summary appears to describe the content.
- **Author's lens — apply before every read:** When looking at any piece of code, ask: "If I wrote this from scratch today, would I do it this way?" Then ask: "What is this connected to — what calls it, what does it call, what shares its state?" Then ask: "Are there connections that *should* exist but don't — other modules that would benefit from this logic, or callers that are duplicating it?" Then ask the **logic-eye question**: "Does this code actually deliver what its docstring/comments/commit-message claims it delivers, or does it stop short and rely on assumptions that aren't met?" Then ask the **claim-vs-test question**: "Does the test prove correctness, or just presence? Could the test pass while the code is wrong?" This is not refactor fishing — it's a reality check. If the answer to all five is "yes, looks right, well connected, claim matches reality, test would catch a regression," move on. If not, log it in SUGGESTIONS.md before touching anything.

---

## 1. RULES (Non-Negotiable)

1. Read this file and SUGGESTIONS.md before doing anything
2. Read target files before editing — verify imports, attributes, and APIs actually exist. **Always read the actual file on disk. Never rely on a conversation summary or memory of a previous read — that content may be stale or incomplete.**
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
16. Compare what you're changing with what it will do to surrounding code
17. If something does not work and it is shown to have been done before, change your approach
18. **Overhaul scope is explicit.** The user can authorize broad rewrites, but every overhaul has a *named* scope (e.g. "GUI logic", "BPE tokenizer", "router boot path"). Inside scope: change anything that earns a better outcome — names, structure, comments, dead branches. Outside scope: still innocent until indicted. This stops "I was overhauling X, so I also rewrote Y" drift. When unsure whether a touch is in-scope, ask before editing.
19. **Author's lens — apply every time you read code:** Ask six questions: (1) "If I wrote this from scratch today, would I do it this way?" (2) "What is this connected to — what calls it, what does it call, what shares its state?" (3) "Could more connections be made — are there other modules that should use this logic, or callers duplicating it?" (4) **Logic-eye:** "Does this code actually deliver what its doc/comment/commit-claim says, or does it stop short of the promise?" (5) **Claim-vs-test:** "Does the test prove correctness or just presence? Could the test pass while the code is wrong (e.g. structural-only, ignoring side-channels, ignoring the GPU non-determinism layer)?" (6) **Sibling-boundary sweep:** "Did I grep every sibling boundary that shares this contract? When I shipped a gate at site A, did I grep the *outer condition* (e.g. `if ctx.is_gguf and hasattr(self.model, "chat"):` or `**kwargs` absorbers, or `_generate_text` re-entry points) to find sites B/C/D in the same family?" If all six check out, move on. If not, log it in SUGGESTIONS.md before touching anything. **Bug-eye alone is not enough — logic-eye catches what bug-eye misses (over-promised docs, partial fixes, structural tests that gate presence not behavior); sibling-eye catches what self-audit-on-the-diff misses (unchanged-but-related code in the same contract family).** Case study: Pass 156z6 shipped streaming json_schema gate with self-audit on the diff and declared done; Pass 156z7 sibling-sweep found 3 more silent-drop sites (chat() GGUF, _generate_with_vision, generate(execute_tools=True)) all in the same contract family — fix wasn't done until the family was.
20. **Never leave something half-built. Finish it, kill it, or clearly park it.** Every slice ends in one of three states — never a fourth "kind of working":
    - **Finished**: feature is reachable from a production entry-point (CLI / GUI / API / scheduled job), has a test that exercises the production call path (not just the function in isolation), every sibling boundary in the family is closed, and the docstring/log/commit message matches what the code actually delivers (no over-promised claims).
    - **Killed**: feature is removed in the same pass — class deleted, import removed, kwarg dropped, tests deleted, doc references purged. No "we'll come back to it" stubs left behind.
    - **Parked**: feature is intentionally deferred AND has (a) a named SUGGESTIONS.md entry with a concrete next step, (b) any partial code is gated behind an `enabled=False` flag with a loud-rejection error or DEBUG log on the off-path, (c) no signature on a public method advertises a kwarg that does nothing, (d) no docstring claims behaviour the code doesn't perform.
    - **Anti-patterns this rule kills (all are real bugs from prior passes):** *signal without consumer* (Pass 156y `is_roleplay()`), *consumer without caller* (Pass 156z2 `apply_profile_to_engine`), *kwarg without passer* (Pass 156z3 `_sample_token(json_constraint=...)`), *FSM without driver* (Pass 156z3 `.advance()` never called), *boundary signal without behaviour change* (Pass 156z9d `<search>` recorded but ignored), *doc claims more than code delivers* (Pass 156s `apply_adapter` Raises clause), *aspirational comments narrating gaps you didn't close* (Pass 156z9 temperature default, Pass 156y2 disk-vs-library doc).
    - **Acceptance check before declaring "done"**: walk the call chain from a production entry-point INWARDS to your new code; if the chain breaks before reaching it, the slice is parked, not finished. State the chain explicitly in the SUGGESTIONS.md entry — "POST /api/chat → chat() → generate() → _generate_text → _record_search_emissions" — so the next pass can verify the chain still holds.
    - **Concrete tells you're about to leave something half-built**: a sentence in your write-up that says "downstream consumers will use this for…", "future work will…", "this is a no-op for X today but…", "the GUI button is still TODO". Each one is the audit finding written one pass early. Either close the gap in this pass or move the entry to "Parked" with a concrete next step.
21. **No fluff in responses.** Match length to the actual question. A yes/no question gets "yes" or "no" plus the one-line reason if not obvious. A status check gets the number. A confirmation gets a single sentence. Save the long-form write-ups for slice stamps in SUGGESTIONS.md, audit reports the user explicitly asked for, and decisions with real tradeoffs. Cave-girl Dia voice means terse — drop preamble ("Sure, I'll…", "Great question…"), drop recap ("So as I mentioned…"), drop closers ("Let me know if…"). A few words of context is a complete answer when the question is small. If the user asks a one-line question and the answer fits in one line, ship one line.

---

## 2. DO NOT

**Code changes:**
- Do not add untested code
- Do not duplicate existing functionality — search first
- Do not leave debug prints (`print`, `console.log`, etc.)
- Do not add unused imports or make random formatting changes
- Do not hardcode file paths or magic constants

**Architecture:**
- Do not introduce new dependencies without approval
- Do not change file structure or architecture outside the named overhaul scope (see §1 #18)
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
- Do not refuse work because it seems complex or time-consuming

**Responses:**
- Do not pad with preamble ("Sure, I'll…", "Great question…", "I'll go ahead and…")
- Do not recap what the user just said back to them
- Do not add closer fluff ("Let me know if you need anything else", "Hope that helps")
- Do not write a paragraph when the answer is one word — match length to the question
- Do not summarise long write-ups in chat after they're already in SUGGESTIONS.md — point at the file
- Do not narrate tool plans ("Now I will read the file, then I will…") — just do the work and report the result

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

## 4. Learned Principles

One-line rules distilled from ~400 implementations. Grouped by theme.

> **Top 9 — apply on every task:**
> 1. Read before editing — verify imports, attributes, APIs exist
> 2. Grep ALL sibling sites when changing a function, config field, pattern, OR doc claim — code call-sites, GUI wiring, comments, docstrings, README claims. Touch one, grep them all, fix in one pass.
> 3. Save AND load — if you add a field, verify both paths
> 4. Tests define the contract — change tests first, then implement
> 5. UI controls must persist — verify round-trip to disk; check BOTH that save captures the widget AND that _on_close triggers the save
> 6. Launch the GUI after editing page builders — tests don't catch wiring crashes
> 7. Guard all model entry points with `_generation_lock`
> 8. Run lint with project config (`ruff check enigma_engine/ tests/`), not overridden flags
> 9. **Logic-eye every audit, not just bug-eye.** After "is the code wrong?" ask "does it deliver what the doc claims?" and "does the test prove correctness or just presence?" Bug-eye alone misses over-promised fixes, partial coverage, and structural tests that gate the wrong thing.

### Verification
- Verify audit findings against source before fixing — false positives are common
- **Logic-eye on doc claims (Pass 156i2):** when a fix ships with a docstring or SUGGESTIONS stamp claiming an outcome, re-read the claim against the code AFTER shipping. Pass 156i shipped 6 sibling-method seed fixes with stamp text "you can rerun exact same training and get exact same model" — but `set_training_seed` only seeds Python+torch CPU+CUDA seeds; it does NOT call `torch.use_deterministic_algorithms(True)` or set `CUBLAS_WORKSPACE_CONFIG`, so cuBLAS/cuDNN can still pick different kernels per launch. Same seed, same data, different gradients on GPU. The code delivers *Python-RNG reproducibility* (shuffle/sample order); the doc claimed *bitwise model reproducibility*. Always check: does the code clear every layer the claim implies (Python RNG, torch RNG, CUDA RNG, kernel selection, DataLoader workers, dropout state), or just one?
- **Structural-vs-behavioral test discipline:** `inspect.getsource()` tests gate **presence of a literal pattern**, not correctness. Failure modes a structural test misses: caller wraps the call in a flag (`if self.config.deterministic: …` → presence check passes, behavior gated off), caller moves the call after the consumer (still present, no-op), caller changes the argument (`set_training_seed(self.config.seed or 42)` → presence passes, semantic changes from None-skip to default-42). Use structural ONLY when behavioral testing requires unavailable hardware; otherwise pair every structural test with at least one behavioral test on a representative sibling, and document why structural is sufficient for the rest.
- **Re-seeding inside a method silently clobbers caller-set RNG state.** When `train_*()` calls `set_training_seed(self.config.seed)` at entry, any user who manually called `random.seed(X)` / `torch.manual_seed(X)` before the trainer call gets their seed overwritten with no warning. Either log `INFO "Seeding RNG from config.seed=N"` so it's visible, or document the takeover loudly. Same applies to `Trainer.__init__` if seeding moves there.
- Read reference implementations — external repos reveal the real gap
- Check "can't do X because Y" comments — the constraint may be stale
- Doc updates must be verified against actual GUI code (`gui_pages.py`, widget constructors), not previous doc text — wrong docs self-reinforce across passes
- When describing GUI behavior in docs, read the widget builder code first — internal code names (e.g. preset names) don't match what the user sees (e.g. GB input)
- Conversation summaries omit AND fabricate — they describe what was true at summary time (so post-summary code edits are invisible) AND can invent state that never existed (Pass 141: summary claimed 5 open `[RESEARCH]` items, grep showed zero, IDs didn't exist). Always read the real file and grep for any specific status tag / item ID the summary names before acting on it.
- **Parallel implementation drift** — when a subsystem has two implementations (e.g. `BPETokenizer` core + `AdvancedBPETokenizer` wrapper, or Python core + Rust backend), byte-mode / unicode / safety infra in *one* is not coverage for the *other*. Pass 149 Tok-2: `BPETokenizer` had full UTF-8 byte mode shipped but defaulted off; `AdvancedBPETokenizer` had **zero** byte support — every non-latin-1 codepoint became `<unk>` even with the flag on. Rust backend was already correct. When fixing a subsystem-wide bug, grep every implementation that shares the public API (encode/decode/save/load), don't assume one implementation's fix is universal.
- **Loud-on-real-issue, silent-on-normal-path is a design rule, not a logging style.** Pass 156b V-8: when the vision encoder load path was missing, the bug was *invisible* — train vision → save .pth → load → drop image → text-only output, no warning. Fix is a 6-row volume table: real failure (state present but won't load, config mismatch, image-with-no-encoder) → loud (`RuntimeError` or `WARNING`); normal path (text-only checkpoint, no image input) → silent. Write the volume table in the suggestion entry *before* coding so each branch maps to a test. Same applies to any feature where a missing piece silently degrades to the wrong-but-plausible output.
- **Library-default change ≠ on-disk-artifact change. JSON wins on load.** Pass 156y2 audit: Pass 156y flipped `AIProfile.personality` library default from a 4-key dict to `{}` and stamped *"`assistant` base profile cleaned"*. Only the in-memory `DEFAULT_PROFILES` constant was cleaned; the canonical `profiles/assistant.json` on disk still had the populated block, and `load_profile()` reads from JSON. So at runtime the canonical base profile still satisfied the OLD contract — exactly what the slice was trying to fix. The Pass 156y write-up itself narrated the gap (*"library default change is a no-op for them because JSON wins on load"*) and walked past it. Rule: when a slice changes a library default for a field that round-trips through JSON / YAML / TOML / config files, grep every on-disk artifact that stores the field and decide explicitly whether each one stays, gets edited, or gets a one-shot migration. The smell to watch for is when your own write-up *narrates* the disconnect ("change is a no-op for X") without closing it — that sentence is the audit finding written one pass early. Companion test rule: pair the in-memory default test with a behavioural load-path test against the canonical disk file (`load_profile("profiles/assistant.json").is_roleplay() is False`). Without the load-path test, the in-memory test passes and the runtime behaviour stays broken.
- **Boundary signal without a consumer = dead infrastructure.** Pass 156z + 156y review: Pass 156y shipped `AIProfile.is_roleplay()` as a "boundary signal" downstream consumers would use to branch on identity-vs-roleplay. Two passes later (156y / 156y2) the signal had ZERO production callers — `to_dict`/`from_dict` round-trip and tests were the only consumers. The slice docstring even said *"Downstream consumers (system-prompt builders, identity guards, future Personality-4 work) use this to..."* — the future-tense clause was the tell. Same anti-pattern as "infrastructure without consumers is dead code" applied to category/identity/state signals. Rule: every new boundary signal (an `is_X()` predicate, a state enum, a category flag) must be paired with at least ONE observable production consumer in the same slice — log line minimum is fine if a behaviour change isn't ready. If you can't name the consumer up-front, you don't have a signal yet, you have a fantasy. The minimal log-line consumer also gives you somewhere to write a behavioural branch test (caplog on True branch + caplog on False branch), which catches the regression where someone collapses the branch back to a single hardcoded line — an outcome a structural-only test misses.
- **Two-layer dead infra: grep the consumer ITSELF for production callers.** Pass 156z2 audit on Pass 156z: I claimed `apply_profile_to_engine` was the "first end-to-end consumer for `is_roleplay()`," but `apply_profile_to_engine` itself had ZERO production callers — only tests + self-references. Grep for the function name across `**/*.py` returned 14 hits, all in test files. Wired one piece of dead infra (the signal) to another piece of dead infra (the consumer) — net zero progress. The "Boundary signal without a consumer" principle was violated in the same pass that wrote it. Rule: when wiring a "first consumer" for a previously-dead signal, do the grep ONE MORE LAYER OUT — confirm the function the new wire-site lives in is actually reachable from production (API endpoint, GUI handler, CLI entry-point, scheduled job). The new test you write must exercise the production call path (post to the endpoint, click the button), NOT just call the wired function directly — a unit test on the consumer function passes whether or not anything in production calls it. Two-layer dead infra is a real failure mode and the only way to catch it is to walk the call chain from a known production entry-point INWARDS to your wire-site.
- **Half-wired contract: an existing kwarg that no caller passes is dead infra hiding behind a live signature.** Pass 156z3 (N-15): `_sample_token` had `json_constraint` as a kwarg with a `mask_logits` call already wired inside, BUT no production caller passed it AND `.advance()` was never called from anywhere in the loop. Pass T3-9 had shipped the FSM class + 5 unit tests + the half-built sampler hook, then stopped. A grep for the *class name* (`JsonSchemaConstraint`) showed it was imported only by tests — but a grep for the kwarg name (`json_constraint`) showed it on `_sample_token`'s signature, which can fool a careless audit into thinking the feature is live. Third pattern of dead infra: signal-without-consumer (156y), consumer-without-caller (156z2), and now **kwarg-without-passer**. Detection rule: when auditing a multi-component feature, grep for the *driver* method's name (`.advance(`, `.step(`, `.update(`) across production code, not just the *class* name — if the driver isn't called from anywhere outside the class itself, the FSM/state-machine/iterator has no operator and the feature is dead regardless of how complete the class looks. Companion test rule: wire-site tests must gate the literal driver call (`json_constraint.advance(`, `json_constraint.is_done`) at the loop site, not just the kwarg presence on the signature — otherwise a regression that drops the per-token `.advance()` call but keeps the `json_constraint=...` forwarding silently reverts the FSM to never-advancing while the kwarg-presence test still passes.
- **Loud-rejection at a boundary is a planning artefact unless the concurrency model has been read.** Pass 156z4 → 156z6: Pass 156z4 shipped `/api/chat/stream` with HTTP 400 on `json_schema` citing "FSM state mutates per token, race against next-token sample" — but `stream_generate` is a single-threaded generator, the FSM `advance()` runs between `yield` and the next `model.forward(...)` call, no concurrency exists between FSM and sampler. The "race" was imaginary. Pass 156z6 closed the rejection in three small wire-site edits + one stdlib-only test pattern. Rule: when a new endpoint loud-rejects a feature instead of supporting it, name the *exact* concurrency model that creates the conflict (which threads, which shared mutable, which lock is missing) before merging the rejection. If you can't name it, you don't have a race — you have a feature you didn't build yet, and the honest reject text says "not yet implemented" not "FSM races sampler". Half-wired contract under a different mask: one endpoint accepted the field, the other refused it, ChatRequest exposed it, callers couldn't tell which path their schema would survive on.
- **Self-audit on the diff is not coverage of the family — sibling-boundary sweep is mandatory.** Pass 156z6 → 156z7: Pass 156z6 ran the five-question lens on the streaming N-15c slice and shipped a GGUF gate inside `stream_chat`, then declared done. Manual audit one turn later found THREE more silent-drop sites in the same family: (1) `chat()` GGUF branch — non-streaming twin of the gate just shipped, identical `if ctx.is_gguf and hasattr(self.model, "chat"):` outer condition, no schema check, production path `POST /api/chat` → unconstrained GGUF output labelled as schema-conforming; (2) `_generate_with_vision` — `**kwargs` docstring literally said `Ignored (absorbs extra chat kwargs)`, samples without going through `_generate_text`/`_generate_manual` so the FSM is never wired in, reachable via `engine.chat(images=[...], json_schema={...})`; (3) `EnigmaEngine.generate(json_schema=..., execute_tools=True)` — first call IS constrained, but `_execute_tools_in_text` re-calls `_generate_text` without the schema on tool-call detection → silent partial constraint, the docstring named this as a caveat with no code-side gate (anti-pattern: doc claims more than code delivers). Same shape on all three: a code path that doesn't go through `_sample_token` with the constraint silently produces unconstrained output. **Sixth audit question to add to §1 #19:** *"Did I grep every sibling boundary that shares this contract?"* When shipping a gate at site A (e.g. `if ctx.is_gguf and hasattr(self.model, "chat"):`), grep the codebase for the *outer condition* (not just the new code) to find sites B/C/D in the same family. The fix isn't done until the family is. The auto-audit covers what was changed; it does NOT cover unchanged-but-related code unless the grep is explicit. Companion test rule: when shipping a gate-style fix in a multi-site family, add one rejection-test per site in the same Pass — not one test for the new gate plus a comment promising siblings later. Pass 156z7 closed three siblings + ValueError gate + 3 tests; size-of-fix per site was tiny because the family was small, but the audit *miss* was structural, not size-driven.
- **Additive load-time merging silently aliases later-added entries.** Pass 156z9c (Stage B-1): three of four tokenizer load paths (Simple `_load_vocab`, Advanced `load`, Char `_load_vocab`) implemented dict-from-disk as `for k,v in disk: self.special_tokens[k] = v` — additive, which means in-memory entries that disk didn't have were preserved. Worked fine for years because every saved vocab happened to contain every default special token. The moment Stage B-1 added `<search>`/`</search>` to the in-memory defaults, every legacy saved vocab on disk now has phantom `<search>` IDs in memory, ALIASING whatever real token the trained model learned at those IDs. The model's behaviour at ID 12 becomes ambiguous and any consumer that branches on `tok.search_start_id` makes the wrong call. Rule: load-time merging of registry-style dicts (special tokens, profile fields, plugin manifests) must be REPLACE-FROM-DISK, not additive — disk is the source of truth, in-memory defaults are only for fresh-construct paths. Three options at the field level when disk is missing the entry: (a) `pop()` the in-memory key and set the convenience ID to None (honest degradation, what Stage B-1 picked), (b) one-shot migration that writes the in-memory default to disk on first load (only OK when the default is universally safe, e.g. a new format flag), (c) raise a clear MigrationRequired error (only OK when the field is critical-path). Test discipline: every load path needs an adversarial test that constructs a disk vocab MISSING a recently-added field, loads it, and asserts the in-memory phantom does NOT survive.
- **Observability-hook `text` parameter must be unambiguous about prompt-inclusion.** Pass 156z9e audit on Pass 156z9d: `_record_search_emissions(text)` was hooked into 8 generation return paths, but 5 of them (`_generate_text` native, `_generate_with_vision`, `speculative_generate`, `medusa_generate`, `lookahead_generate`) decode the FULL sequence via `text = self._decode_output(output_ids)` — that's `prompt + continuation`. The other 3 (`_generate_text` GGUF, `stream_generate`, `batch_generate`) pass continuation-only text. The helper docstring promised *"scan generated text for blocks the model emitted"* but the code couldn't tell the difference, so a user prompt asking *about* the `<search>foo</search>` syntax would land "foo" in `last_search_queries` as if the model had emitted it. Logic-eye violation that no original test caught — every behavioural test fed bare emission text without a prompt prefix, so the slicing bug was structurally invisible. **Rule:** any text-side scanner that can be called from BOTH full-sequence-decode paths AND continuation-only paths MUST carry the boundary in its signature (typically `prompt: str | None = None`) and either slice internally (`if prompt and text.startswith(prompt): text = text[len(prompt):]`) or document the assumption per call site. Defensive `startswith` check protects against post-processed text where the prompt prefix was already trimmed. **Test discipline:** every observability hook needs at least one **adversarial prompt-echo test** — caller supplies a prompt that itself contains the pattern the scanner is looking for, model returns prompt+benign-continuation, asserts the scanner does NOT record the prompt-side hit. Without that test, the bug above can pass through every "happy path" behavioural test the slice ships with. Generalises beyond `<search>`: any post-generation regex scanner (tool-call markers, citation tags, code fences) inherits the same ambiguity if it accepts bare `text`.


### Testing
- Structural tests (`inspect.getsource()`) verify code paths without GPU
- After method decomposition, grep tests for `getsource(OldMethod)` and redirect
- Embedding/output dimension changes ripple to shape-checking assertions
- Tests must verify specification (what should happen), not implementation (what currently happens) — "pressing J outputs J", not "the code maps J to S and maps J to S"
- `inspect.getsource()` tests confirm HOW code works, not WHAT it does — prefer tests that call the function and check the output
- **Substring-presence assertions on `inspect.getsource` are vacuous when the substring appears at multiple sites in the body.** Pass 156z9y caught a structural test from Pass 156z9x that asserted `"inline_search_enabled" in inspect.getsource(EnigmaGUI.__init__)` to gate a new boot-load wire-site at desktop.py L172-173. Failure mode: the same token already appeared at L107 (`self.inline_search_enabled = True` in-memory default) and at the GUI-toggle assignment, so a regression deleting only the boot-load line still satisfied the assertion. Test was structural-presence, not wire-site-correctness. Strengthened to a regex `_read_gui_bool_setting\(\s*"inline_search_enabled"` matching ONLY the boot-load call expression, falsified in-place by deleting the assignment + confirming pytest fails, then restored. **Rule:** when adding a structural test for a new wire-site that joins an existing pattern (4 boot-load calls in `__init__`, multiple `_record_search_emissions` call-sites in generation, sibling `train_*()` methods all calling `set_training_seed`), assert the FULL call expression paired with the new argument (`function_name(literal_arg`), not just the function name or the argument alone — either alone is shared with siblings. The regex `\s*` between `(` and the string literal tolerates the line-continuation whitespace black/ruff produce when the call wraps. Use the falsification check before shipping: temporarily delete the line you claim to gate, run the test, confirm it FAILS — if it passes, the test is presence-only and needs strengthening.
- Vocab padding (GPU alignment to 64) means model output dim ≠ vocab_size — test `>= vocab_size`, not `== vocab_size`
- Train-mode equality tests are invalid when stochastic training paths exist (dropout/NEFTune/noise). For train mode, assert finite outputs, shapes, and invariants instead of exact numeric equality.
- Rust extension tests fail silently when the wheel is stale — if a method exists in `lib.rs` but not in the installed `.pyd`, the test gets `AttributeError` with no build error. Always rebuild (`maturin build --release` + `pip install --force-reinstall --no-deps`) after adding methods to Rust source.
- **Test fakes that ignore kwargs hide signature/contract bugs.** Pass 155 fake `load_dataset(path, *args, **kwargs)` accepted any `split=` value because the fake threw kwargs away — tests passed with `split="train"` even though SmolTalk2 has no train split. Live Phase-1 run blew up immediately. Lesson: when a fake represents an external API where one of the kwargs can be wrong (split name, config name, region, model id), the fake should validate at least the *shape* of expected values or expose the lookup helper (e.g. `get_dataset_split_names`) so the test exercises the same resolution path as production. Whatever the fake silently ignores is your blind spot.
- **Vision/multimodal data collectors should not auto-download multi-GB image archives.** Pass 156c V-5: LLaVA-Pretrain ships ~14 GB of image bytes in a separate `images.zip` archive on the dataset card. Auto-fetching that on every `--llava-pretrain` invocation is hostile to the user's bandwidth and disk. Pattern: stream the *caption metadata* through `datasets`, take a required `--images-dir` arg pointing at a one-time user-managed extraction, verify each row's image file exists on disk, and skip-with-warning on misses (cap log noise at 5, report total at end). The collector becomes a metadata-and-validation layer; bulk binary fetch stays a separate, deliberate user step. Same shape generalizes to ShareGPT4V, COCO, audio datasets, and any future modality where row-level metadata is small but media bytes are large.
- **String-dispatch kwarg = registry pattern + paired structural-and-behavioural tests.** Pass 156j (D-9 APO): `train_dpo` gained `loss_type="dpo"|"apo_zero"`. Cleanest implementation is a static `_resolve_preference_loss(name) -> callable` registry mapping `{"dpo": _dpo_loss, "apo_zero": _apo_zero_loss}` that raises `ValueError` on miss — typos fail loud at the call site instead of silently falling back to the default. Companion rule on tests: a structural test (`assert "apo_zero" in inspect.getsource(train_dpo)`) only proves the kwarg name appears somewhere in the body — it gates against typos but NOT against `loss_type` being assigned to a local variable that's then ignored. Pair it with a behavioural dispatch test that patches both branch implementations (e.g. `_dpo_loss` and `_apo_zero_loss`) with sentinel-recording mocks, stubs out heavyweight upstream paths (e.g. `_get_sequence_logps`), runs the public method once, and asserts exactly one branch's sentinel was hit. That test catches the regression where someone reverts `loss = loss_fn(...)` back to a hardcoded `self._dpo_loss(...)` call.
- **Structural import-presence tests do NOT validate output shape of formatted artifacts.** Pass 156z9an audit on Pass 156z9am (P5-pre-1): personality prompt pool shipped with 5 prompts that themselves started `"User: "` and ended `"Assistant:"`. The wire-site test asserted `'"personality": list(_PERSONALITY_PROMPTS),'` appeared in `_start_distill_training` — passed. Behaviour was broken anyway because the GUI loop wraps each prompt as `f"User: {prompt}\nAssistant: {response}"`, so an offending prompt produced double-prefixed training data (`User: User: hey what's up\nAssistant:\nAssistant: <resp>`). A structural test on the import said NOTHING about the *output shape* of the wired feature. **Rule:** when a feature emits user-visible artifacts (training data lines, log file rows, JSON records, file contents), pair the import-substring structural test with at least one behavioural test that constructs realistic inputs, runs the formatter / writer, and inspects the output string for **shape invariants** — counts of expected markers (`example.lower().count("user:") == 1`), absence of forbidden patterns, well-formedness of separators. The shape-invariant test would have caught all 5 corrupt prompts in one assertion. Generalises to: log-line formats with structured prefixes, JSONL record templates, CSV emitters, prompt-template wrappers, system-message builders.

### GUI & Code Matching
- CONFIG page has temperature/top_p/top_k/max_tokens/repetition_penalty → consumed via `config_overrides` in chat kwargs
- FORGE hardcoded temperatures (0.3/0.7/0.8/0.9) across dialogue/adaptive/distill/evolutionary — no GUI control
- DPO beta, reward model epochs, vision batch_size are hardcoded in GUI layer — no user control
- Pre-train general_mix_ratio=0 silently overridden to 0.1 — logged but not confirmed with user
- Subagent reports about missing GUI controls have ~80% false positive rate — always verify against actual widget builders
- Search ALL consumers when changing a core pattern (including RewardModel, test helpers)
- No `getattr(config, 'field', default)` — access directly so missing fields crash visibly
- When removing a widget, remove all read sites or define a fallback
- UI dispatchers must cover all UI options — new radio card needs new dispatcher case
- Section visibility must match mode count — unmapped modes leak controls
- Wrapper functions must match delegate signatures (name + type, not just count)
- Config object with N constructor sites: every field must appear in all N — use structural tests to catch disconnects
- Removing class aliases? Also fix string type annotations (`-> 'OldName':`)
- `TYPE_CHECKING` imports for string annotations need `# noqa: F401`
- Hardcoded training constants (thresholds, chunk sizes, batch caps) must scale with hardware — use a budget dataclass (TrainingMemoryBudget) not magic numbers. Grep ALL call sites (GUI TrainingConfig constructors, not just core/) when centralizing a constant
- Inference constants (context size, batch size, cache caps, chunk sizes) need the same treatment as training — VRAM tier ladders with 3-5 breakpoints waste hardware at the top and hurt at the bottom. Use continuous scaling or a budget dataclass
- **Library kwarg → GUI surface: refactor the existing entry-point, do NOT duplicate the body.** Pass 156k D-9b: APO-zero ships as a fifth alignment-mode radio card by adding `loss_type: str = "dpo"` to the existing `_start_dpo_training` (200-line method), parametrizing the human-facing label via a local `algo_label`, and forwarding the kwarg at the single `trainer.train_dpo(...)` call site. New `_start_apo_training` is a 1-line wrapper: `self._start_dpo_training(loss_type="apo_zero")`. Anti-pattern would have been to copy 200 lines, change two, and watch them drift forever as either side gets bug fixes the other misses. Cost of refactor: ~10 lines + 1 wrapper.
- **GUI-wiring tests must gate the literal kwarg at the trainer call site.** Pass 156k: assert `loss_type=` (literal token) appears in the trainer call expression of the entry-point. Without that, a regression where someone "fixes" the GUI to assign `loss_type` to a local variable but drops it from the actual `train_dpo()` call silently reverts the new mode to default while the underlying loss math is still correct. End-to-end behavioural proof (sentinel-mock dispatch test) lives at the library layer; the GUI test only needs to gate the wiring.
- **Label-tracking after a refactor: word-boundary regex over the whole method body.** Pass 156k-audit: when a refactor introduces a derived label local (e.g. `algo_label = "DPO" if loss_type == "dpo" else "APO-ZERO"`), the trainer-call-site forward test does NOT prove user-facing strings (status bar, log prefixes, error messages, save-history label, dialog titles) actually use the parametrized label. Separate test: strip docstrings + comment-only lines, scan body with **word-boundary regex** `re.search(r'\bDPO\b', ln)`, allowlist only the legitimate ternary-definition lines. Round 1 of this test used substring search `'"DPO"' in ln` and PASSED while three hardcoded literals still leaked (`f"--- DPO TRAINING STOPPED ---"` contains `DPO` but not `"DPO"`). Round 2 with word-boundary caught all three pre-fix and zero post-fix. Generalizes: any structural test that gates on string-literal presence inside f-strings or log strings must use `\bTOKEN\b` regex, not double-quoted substring.

### Training & Numerics
- Disk-backed training: write sequences to JSONL with byte offsets, pass `data_path`/`data_offsets` to Trainer — avoids holding all sequences in RAM
- Two-pass streaming: Pass 1 scans + collects samples (capped), Pass 2 processes + writes to disk — peak RAM = one chunk + samples, not full dataset
- Multi-stage pipelines multiply peak RAM — write intermediates to disk between stages
- Sequence packing 4D masks are O(rows × T²) — build per-batch, not all-at-once; 5K rows × 4096² × 4B = 320 GB
- Training batch tensors must stay on CPU until consumed — `.to(device)` in the training loop, not at batch creation time; otherwise all windowed batches accumulate on GPU
- Guard super-linear algorithms with size thresholds — skip above N, fall back to cheaper alternative
- Temp files need cleanup on ALL return paths (including early abort, cancel, OOM)
- Silent process death (no traceback) = OS OOM kill — check memory-intensive ops at log stop point
- `LambdaLR.__init__` calls `step()` internally — use `(step + 1) / warmup` to avoid zero LR
- Never `encode()` for known special token IDs — use named attribute directly
- RMSNorm needs fp32 upcast in fp16/bf16 to prevent NaN
- MoE/scatter-add accumulators need fp32 — `zeros_like(x)` inherits fp16 under AMP, `index_add_` in fp16 loses precision across many experts. Upcast accumulator, `.float()` the addend, cast back to input dtype at end
- AdamW `beta2=0.95` for LM training (not PyTorch default 0.999)
- Training data format must match inference format — "User:/Assistant:" everywhere
- Data quality gates: clean → validate → deduplicate — never accept unchecked teacher output
- Two-tier truncation: message-count for disk, token-count for inference context
- Loss penalty terms must be tensors with grad — a Python float in the loss produces zero gradient and is a silent no-op
- Log-ratio before `exp()` must be clamped (±20) to prevent inf — check ALL RL algorithms, not just PPO
- Loss metric weighting must count non-padding tokens, not `numel()` — `ignore_index` makes CE mean over subset
- `torch.std()` defaults to Bessel's correction (N-1) — use `unbiased=False` for population std with small samples
- Gradient accumulation scaling must be consistent across ALL training methods — grep `max_grad_accumulation` and verify every `loss.backward()` divides first
- Streaming generator stop-string logic: after breaking on match, skip any final buffer flush — `stopped` flag pattern
- KL penalty formula must be consistent across ALL RL trainers — use `(policy_logps - ref_logps).mean()` everywhere; other formulations can go negative or flip sign
- Smoke/test data must match model scale — tokenizer retrain on tiny data shrinks vocab, forces random re-init, and 616M params on 129K tokens = NaN in minutes. Use low Memory (GB) value (e.g. `1`) for small test data so the auto-picked model is small enough
- `torch.compile` warmup can take 50-100+ steps with severely degraded throughput (10-100x slower) — don't diagnose performance until after Triton/Inductor finishes compiling kernels. Step 10 may show 588 tok/s, then step 30 drops to 90 tok/s as different code paths trigger recompilation
- Benchmark hot paths before rewriting — measured our Python BPE at ~7 MB/s, which defined the actual gap vs HF Rust ~50+ MB/s. Guessed estimates were 3-10x off in both directions
- BPE tokenizer sample cap must scale with hardware but saturates early — quality plateaus around 2 GB of text (~1.5M unique words); 8 GB creates 6.1M unique words and merge time grows super-linearly. Cap at 2 GB max. Measured: 2 GB / 2.4M unique words / 32K vocab = 20 min (post S712-S714 heap fixes + S809 GIL yield fix).
- GIL yields in CPU-bound loops must match per-iteration wall time — `time.sleep(0)` every 100 iterations is fine when each takes 1 ms, but when early iterations take 10+ seconds the GUI freezes for minutes. Yield every iteration if wall time is unpredictable.
- GUI "save checkpoint" during training must save live model weights from GPU memory, not copy the on-disk .pth file — the disk file is the untrained initial model, training state only exists in memory until the training loop saves it
- Long-running training (millions of steps) needs step-based periodic saves, not just epoch-end saves — if one epoch takes months, all progress is lost on interruption
- Verify checkpoint correctness by comparing file size + timestamp to the original — byte-identical files with matching LastWriteTime = stale copy, not a real save
- OOM kills leave no traceback and no log — write a heartbeat file to disk every 30s (pid/status/phase/step/loss/timestamp) so post-mortem analysis has a last-known-good state. Check heartbeat on next training start to detect stale sessions
- Training heartbeat statuses: `running` (in progress), `complete` (clean exit), `stopped` (user cancel), `crashed_oom` (RuntimeError OOM branch), `crashed_nan` (NaN/Inf loss), `crashed` (unhandled exception) — use status to route diagnostic messaging on next launch
- Log rotation limits of 10 delete historical runs — keep at least 100 forge logs so past sessions survive weeks of daily restarts
- NaN/Inf loss detection belongs in the on_loss callback, not the training loop core — the GUI layer sees every loss value and can log + halt before the next batch makes things worse
- GUI presets that hardcode hardware-dependent values (batch size, chunk size) should use "auto" — a preset value of "4" overrides auto-detection and wastes GPU on large cards or OOMs on small ones
- Cross-path patterns (assign + finally clear) must be verified across ALL call sites — adding `finally: x = None` without the assignment before `train()` looks complete but is a no-op. Grep for both sides of every assign/clear pair
- Functions that build per-item tensors and accumulate into a list before returning must be generators if dataset is large — 45K rows × 4096² float32 masks = 3 TB in a list. Convert to `yield` + `yield from`, wrap eager callers with `list()`. Silent OS OOM kill with no traceback.
- Double-checked locking: ALL work must complete inside the lock before setting the initialized flag — `_initialized = True` followed by work outside the lock lets a second thread see True and use unfinished state
- Non-reentrant lock + proxy object = deadlock — if a dict proxy's `get()`/`update()`/`items()` triggers initialization that acquires a lock, then `_load_user_config()` (called inside that lock) calling `CONFIG.update()` re-enters the proxy and deadlocks. Bypass with `dict.update(CONFIG, ...)` / `dict.__setitem__(CONFIG, ...)` / `dict.items(CONFIG)` inside locked init paths

### Unpredictability vs Determinism
- Unpredictable in **behavior**, deterministic in **infrastructure**. Sampling, self-initiated research, mood-weighted replay, emergent personality drift should surprise. Training seeds, test fixtures, dedup hashes, atomic saves must reproduce byte-identical runs.
- Every "unpredictable" behavior must be driven by an internal signal (confidence, emotional_state, novelty, engagement), never pure RNG. Pure randomness = noise, not personality.
- Every unpredictable feature needs a debug off-switch and a reproduction seed — otherwise crashes are unreproducible and tests flake.
- `hash()` is non-deterministic across runs (already logged); extend the rule: no `random.random()` / `torch.rand()` without a seed source tied to either the training seed or an internal signal.
- "Let it grow naturally" is not a substitute for curriculum or structure. Chaos without a signal is not emergent behavior.
- Neural reward models reinforce whatever scores highest — usually confident-sounding noise, not correctness. Prefer rule-based rewards (pass/fail, format check) for RL; reserve neural reward for preference tasks where ground truth is subjective.
- **Sibling-method drift on the seed call.** When one method in a class family seeds RNG and others don't, sample order is non-reproducible across calls — even with `config.seed` explicitly set. Pass 156h+i: only `train()` was calling `set_training_seed(self.config.seed)`; `train_dpo`/`train_simpo`/`train_kto`/`train_orpo`/`train_vision`/`train_audio`/`train_rest` all skipped it, so their `random.shuffle(pairs)` ran on whatever RNG state happened to be live. Fix is a 3-line guard at each method's entry; gate regression with a structural test that `inspect.getsource()` every public `train_*()` method and asserts the call is present. When you fix one site of a cross-method pattern, grep all siblings the same pass.
- **Config field without CLI/GUI consumer = unreachable feature.** Pass 156i3 DET-2: `TrainingConfig.deterministic` field shipped with full helper logic and 4 tests, but `run.py --train` had no `--deterministic` flag — the canonical CLI workflow couldn't enable it. Author's-lens scan caught it within the same pass. Rule: when adding a `TrainingConfig` field (or any config-object field), grep `TrainingConfig(` constructor sites AND the CLI argparse block AND the GUI widget builders in the same pass. Each consumer layer that doesn't surface the field reduces "shipped" to "shipped for hand-edited code paths only." Companion rule: when a flag depends on another flag (e.g. `--deterministic` requires `--seed` or it's a silent no-op), add an early `parser.error()` so the dependency fails loud at parse time, not silently inside the helper.
- **`warn_only=True` on `torch.use_deterministic_algorithms` is mandatory when MoE is in scope.** Pass 156i3 DET-2: MoE `index_add_` has no deterministic CUDA kernel — calling `torch.use_deterministic_algorithms(True)` (no `warn_only`) hard-errors and blocks every MoE training run the moment the user opts into determinism. With `warn_only=True` the user gets a one-line UserWarning per non-deterministic op and training continues. Rule: any module that opts into PyTorch determinism must use `warn_only=True` unless every op the model uses is in the deterministic-kernel list — which is rarely true for modern architectures.
- **Continuous/background training paths need NaN/Inf abort + token-length cap from day one.** Pass 156i4 Continuous-1: `BackgroundTrainer._train_batch` had no `torch.isfinite(loss)` check — a single NaN sample steps the optimizer with NaN gradients, and from that step forward *every weight in the model is NaN*. The class is `daemon=True` and runs for months over the user's chat history; one bad input could permanently corrupt the model with no warning, no test signal, no log entry. Same applies to `_retrain_on_replay` — duplicate the guard wherever `loss.backward()` is called. Companion rule: continuous trainers must also cap per-example token length (`max_token_length: int = 4096`) — without it, a misbehaving mod or pathological input pushes a 1M-token tensor through the model and OOMs the GPU. Skip-with-DEBUG-log, not truncate (truncation silently drops context). When auditing any always-on training path, check three things: (1) finiteness guard before every `.backward()`, (2) per-step gate on `valid_count > 0` before `optimizer.step()`, (3) hard input-size cap that refuses oversize samples rather than truncating them. All three together make the path safe to run unattended for months.
- **Replay-buffer rehearsal alone does NOT prevent catastrophic forgetting — anchor sets do, partially.** Pass 156i5 Continuous-2: `BackgroundTrainer` class docstring claimed "prevents catastrophic forgetting" but `_retrain_on_replay` only rehearsed the recent chat buffer. A user spending weeks on a single topic (cooking only, say) silently loses unrelated skills (math, code, reasoning) even with replay running every 200 examples — because none of those skills appear in recent chat to be rehearsed. The honest fix is a **fixed anchor set** of curated general-capability examples loaded from disk and rehearsed *alongside* the recent slice on every replay pass. Anchors must NOT be score-sorted (curated order is the point); same NaN/finite + token-cap discipline applies uniformly to anchors and recent (mirror discipline). Even with anchors, forgetting is **bounded by anchor coverage** — a 50-example anchor set is a floor, not a guarantee. Reframe any docstring that says "prevents" to "mitigates" + state the bound explicitly. Pattern generalizes: any continuous/online learner that claims to defeat forgetting needs (a) a frozen reference dataset, (b) interleaved rehearsal, (c) honest scope language about what the rehearsal can and cannot reach. Without (a) the claim is a lie.
- **Honesty reframes must grep all sibling claims, not just the docstring.** Pass 156i6 Continuous-2a: Pass 156i5 fixed the class docstring of `BackgroundTrainer` to say "mitigates ... bounded by anchor coverage" but **two sibling claims still over-promised** — an inline section comment and the `_retrain_on_replay` method docstring both still said "prevents catastrophic forgetting." Self-audit caught both within minutes. Same anti-pattern as the seed-method drift principle, applied to doc claims: when you reframe one site of a multi-site claim, grep the whole module (and adjacent modules with shared subject) the same pass. Specifically grep the **literal old wording**, not the new one — only the unfixed sites still match.
- **Empty-A early-out must not skip always-on B.** Pass 156i6: `_retrain_on_replay` returned early on `not self.replay_buffer` *before* loading anchors, defeating the entire anchor feature during quiet periods (which is precisely when anchors matter most). When ordering early-outs in a method that combines two data sources A (situational) and B (always-on), gate on `not A and not B`, not on `not A` alone — and *load* B before the gate so it's available to the check.
- **File-present-zero-yield is a real misconfiguration and must be loud.** Pass 156i6: `_load_anchor_examples` logged INFO "loaded 0 anchor example(s)" when the configured anchor file existed but contained only malformed/empty rows. Volume table for the loud-on-real-issue rule: missing-file → WARNING, file-present-zero-yield → WARNING, file-present-N-rows → INFO. Three branches, three log levels — not two branches collapsed into one.
- **Single-write idempotence flags deserve a counter-based test.** Pass 156i6 added `test_anchor_file_loaded_only_once_across_replay_passes` that patches `Path.open` and asserts exactly 1 open across 3 calls. Without it, breaking the `_anchor_load_attempted` flag (re-reading every pass) silently passes all other anchor tests — every read returns the same data, so behavioural assertions still hold. Any flag whose contract is "do this exactly once" needs a dedicated counter test, not just a behaviour test.
- **`deque(maxlen=N)` is FIFO recency, not "keeps best" anything.** Pass 156i4 caught a test named `test_replay_keeps_best_examples` that asserted `min(scores) >= 0.6` after appending `[0.3, 0.9, 0.6, 0.8]` to a `deque(maxlen=3)` — the test passed only because the lowest score (0.3) happened to be inserted *first* and was evicted by FIFO. Reorder to `[0.9, 0.6, 0.3, 0.8]` and the same "keeps best" assertion fails (0.3 stays, 0.9 evicted). Rule: when a structural property depends on insertion order in a bounded container, the test must use the *adversarial* ordering — insert the value that the claimed property says should be *kept* in the position that the *actual* implementation will *evict*. If the claim and the implementation agree, the adversarial test passes; if they disagree, the test fails the way it should. This is the structural-vs-behavioral test discipline applied to data-structure semantics.

### GUI & Threading
- `after(0, callback)` = main-thread execution from any thread — no locks needed for tkinter-only state
- Only lock mutable state that crosses thread boundaries
- Centralize widget setup in factory functions for consistent behavior
- Lambda in threaded error handlers: `msg = str(exc)` then `lambda m=msg: f(m)` — avoid late binding
- CTkFrame defaults to 200x200px when propagation disabled — compute height from font metrics
- `CTkEntry.configure(fg_color="transparent")` crashes — use parent's actual bg color
- Tkinter `<Leave>` fires entering child widgets — check pointer vs widget bounds for true leave
- Fixed-width labels clip text — use `grid_columnconfigure(col, minsize=N)` instead
- Any I/O with timeout must run off main thread — `urlopen(timeout=10)` freezes GUI
- Debounce rapid widget rebuilds with `after(300, rebuild)` + cancel-on-re-entry
- Recursive `after()` animations must track the callback ID and cancel before restarting — stacking parallel loops freezes GUI
- `<Configure>` handlers must debounce (100ms) and skip same-value reconfigures — widget `.configure()` can trigger new `<Configure>` events causing cascade
- Per-tag cursor bindings (`tag_bind <Enter>/<Leave>`) set cursor on entire widget — use single `<Motion>` handler with `tag_names()` check
- Tooltip/popup dismissal needs watchdog timer — `<Leave>` events are unreliable for fast mouse, alt-tab, window overlap
- Dual-flag stop mechanisms (GUI `training_active` flag + Trainer `_stop_requested` flag) must BOTH be signaled — if only the GUI flag is set, the Trainer's batch-boundary check never triggers and stop waits for callback-based detection (up to one full batch delay)

### Security & Boundaries
- `core/` never imports `gui/` — wire cross-boundary features in GUI layer
- Forbidden lists must block the primitive (`__import__(`), not specific examples
- Audit config values to verify they're consumed by code, not just decorative
- Path traversal `startswith` checks need `+ os.sep` — grep for all such guards when fixing one
- Path traversal: `str.startswith()` is insufficient — use `Path.relative_to()` which raises ValueError for paths outside the allowed tree
- Sandbox forbidden lists must be complete across ALL sandbox implementations — grep all sandboxes when adding dangerous patterns
- API endpoints must never expose absolute filesystem paths — use `p.relative_to(BASE_DIR)` and resolve relative paths back in the consumer (load endpoint)

### Feature Gating
- Default to disabled when model capability is uncertain — infrastructure ready, UX clean
- Quality gate AI content before surfacing — heuristic scorer with threshold
- Idle detection reuses `after(N, callback)` with monotonic timer + activity reset
- Infrastructure without consumers is dead code — if the import target doesn't exist (e.g. `tools/tool_executor.py`), the feature can't activate regardless of how well the loop is coded. Build the dependency chain bottom-up, not top-down
- Staged tool-use rollout — when the strong version of a tool-use feature needs logits/training-loop changes (e.g. inline `<search>` token), ship a pure-string post-generation gate first. It is fully testable without a trained model, signal-driven, and unblocks the dependent feature while the harder version is being designed. Pass 153 AutoResearch-2 Stage A: `score_uncertainty()` + `should_retry_with_research()` shipped on text-only signals (hedge phrases, refusal patterns, length anomaly, question echo) — Stage B inline-token work remains future. Pattern: dataclass `Result(score, reasons)` makes the gate auditable in logs; off-switch via `enabled=False` keyword arg; deterministic — no RNG.
- **Module-level path constants make boot-default behaviour testable.** Pass 156i7 Continuous-3: `_DEFAULT_ANCHOR_PATH = Path(__file__).resolve().parent.parent / "data" / "anchor_examples.jsonl"` lives at module scope in `router.py`. Tests `monkeypatch.setattr(router_mod, "_DEFAULT_ANCHOR_PATH", tmp_path / "anchors.jsonl")` to simulate file-present and file-missing branches without touching the real `data/` directory. Inlining the path inside the constructor body (`def __init__: ... if (Path(__file__).parent.parent / "data" / "anchor_examples.jsonl").exists(): ...`) makes that test impossible — you'd have to either delete the real file or pollute it. **Companion rule: wire feature defaults at the boot site, not the library default.** The library `BackgroundTrainer(anchor_data_path=None)` keeps the feature off by default for direct callers (test isolation, explicit opt-in); the production boot path (`ModRouter._create_trainer`) auto-resolves the repo default via `_DEFAULT_ANCHOR_PATH if .exists() else None`. Same shape applies to any "auto-discover repo asset on boot" feature: name the path at module scope, gate on `.exists()` inside the boot-site constructor (not the library), monkeypatch the constant in tests.
- **Prefer dual-emit on the producer over multi-format support on the consumer.** Pass 156i8 D-11: collector wrote JSONL pairs to `data/finetune/`; existing SFT trainer reads plain text via `Path.read_text`. Two ways to bridge: (a) teach the trainer to detect-and-parse JSONL (touches GUI file picker + Trainer + format detection — three change sites, regression risk on the existing text path), or (b) teach the collector to *also* emit the canonical text format the trainer already consumes (one new helper, zero training-side touches). Picked (b) — `_write_combined_text()` runs alongside `_write_jsonl` inside `combine_all`, emitting `combined_finetune.txt` in `User: <p>\n\nAssistant: <c>` format with blank-line separators. **Precondition: the consumer's format must already be the canonical one in the codebase**, otherwise dual-emit just adds a second source of truth for which-format-wins. Here `User:/Assistant:` is used everywhere (router.py, GUI chat builder, advanced FORGE pipeline) so there was no ambiguity about which side moves.

### Research
- Audit existing implementations before searching for research — know what you have first
- Upgrades (better algorithm) vs gaps (missing feature) — different risk profiles
- Author's-lens during research passes catches code bugs that unit tests miss — Pass 140 research on MTP paper (arxiv:2404.19737) surfaced TWO silent code issues while simply reading the adjacent code: (1) an inline comment at [model_presets.py L129](enigma_engine/core/model_presets.py#L129) had the paper's conclusion inverted ("biggest gain at small model sizes" when the paper says "increasingly useful for larger model sizes"), and (2) `predict_heads` at [model.py L237-240](enigma_engine/core/model.py#L237-L240) were NOT weight-tied to `tok_embeddings`, silently inflating the model by ~98M params (13% of 742M) for an auxiliary signal of ambiguous sub-1B benefit. Lesson: when researching a feature, read the code that implements it, not just the paper — the gap is often between them.
- Cross-reference subagent findings against actual source — false positives are common
- Enumerate subsystems systematically, don't brainstorm from memory
- Subagent false positive rate is ~80% on HIGH-confidence claims — always manually verify before fixing
- Architecture competitiveness ≠ model quality — data and compute are the real bottleneck, not code
- "Cloud AI feature X" is usually just RAG with better UX — check if you already have the core before building
- Target hardware is RTX 5090 (32 GB total, 16 GB VRAM budget for AI) — code should scale up or down, don't hardcode GPU assumptions
- Language selection for rewrites: check what production systems actually use — both OpenAI (tiktoken) and HuggingFace (tokenizers) independently chose Rust+PyO3 for the same problem. Two independent choices beat any theoretical analysis
- Profile the real bottleneck before picking a rewrite strategy — regex pre-tokenization was 80%+ of encode time, not BPE merges. Rewriting the wrong hot path wastes the effort
- Drop-in replacement behind protocol interfaces: only modify the factory function (`get_tokenizer()`), zero changes to 40+ consumer call sites. Protocol boundary = rewrite boundary
- Mod files and core files can have the same name but completely different purposes — `mods/vision/vision.py` (screen OCR) vs `core/vision_encoder.py` (ViT for LLM multimodal). Read both before assuming they're connected
- A router named "ModRouter" does not route between models — verify what it actually routes before designing a model dispatch layer on top of it
- Data-mixing research is blog-sourced, not paper-sourced — SmolLM3 blog has concrete stage percentages (12% code Stage 1, 24% Stage 3) while Qwen3 and DeepSeek-LLM papers say "increasing" without numbers. Prefer blog posts + HF dataset cards over arxiv for reproducible mixing ratios
- Dedup threshold defaults in our code (0.8) can drift from industry standard (FineWeb/DCLM use 0.75) — always re-verify numeric defaults against the paper actually cited, don't trust that an older default matches current practice. Same applies to LSH band/row splits (112 hashes / 14 bands / 8 rows is the reference split)
- Primary source for a dataset is the dataset's own card/paper, not a downstream consumer's blog — Pass 137 audit caught `<think>` tag format being inferred from SmolLM3's description of how *they* used OpenThoughts3, when OpenThoughts3's own card only says "reasoning traces" without committing to a wrapper format. Verify tag/format claims against the producing dataset's docs; if unclear, sample the actual data before relying on the format

### Rust / PyO3
- Rust `regex` crate doesn't support lookbehind — use manual byte-level boundary check (`is_non_alpha_boundary()`) instead
- MSVC toolchain (`stable-x86_64-pc-windows-msvc`) needs VS Build Tools; GNU toolchain (`stable-x86_64-pc-windows-gnu`) bundles MinGW linker — works out of the box
- `maturin develop` needs `VIRTUAL_ENV` set; `maturin build --release` + `pip install <wheel>` works everywhere
- Symbol interning (string→u32 ID) eliminates String allocation in merge loops — measured 60% speedup over String-based version
- Skip-array (`next[]`/`prev[]`) for O(1) linked-list removal in BPE merge — avoids Vec shifting
- Pack two u32 IDs into one u64 for HashMap pair key — zero-alloc pair lookup
- Direct-ID cache (`HashMap<String, Vec<i64>>`) skips symbol→token_id mapping on cache hit — 4x throughput on repeated text
- Cache eviction: 40K cap with quarter-eviction (drop 10K entries randomly when full) — simple and effective
- BPE merge ranks in Rust: use `merge_table: HashMap<u64, MergeEntry>` not sorted Vec — O(1) lookup per pair
- Each new terminal session needs `$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"` for cargo/rustc (Windows)
- Build command: `cd rust_extensions; maturin build --release` then `pip install target/wheels/*.whl --force-reinstall --no-deps`
- Auto-fallback pattern: `_rust_available` class var (None/True/False), try import once, set permanently — no repeated import failures

### Auditing
- Audit by risk, not by size — training pipeline, GPU ops, and resource-intensive paths first; utility modules last
- `torch.compile` without Triton triggers Inductor C++ fallback that eats tens of GB RAM — always gate on Triton availability
- GUI hardcoded config values bypass safe defaults — audit GUI builders that construct config objects, not just the config defaults
- Use subagent for bulk reconnaissance (list all GUI controls, map all config fields), then manually verify the top findings — faster than reading every file top-to-bottom
- Wiring audits: trace GUI control → config field → core consumer. Dead ends at any step = bug. Use grep for the field name across all three layers.
- Triage audit findings before fixing — verify each claim against source. "Port mismatch 8080 vs 8000" and "docstring says cosine but code uses dot product" were both false positives that would have wasted edits
- Faster precision: check one data flow end-to-end (button press → what happens) rather than reading an entire file sequentially
- GUI crash checks should verify both terminal liveness and OS process list (`run.py --gui`) — one signal alone can be misleading when multiple sessions exist.
- Architecture review checklist: (1) every implemented module has at least one real caller path, (2) core features are available to ALL consumers (GUI/API/background trainer), (3) rich state that is computed — verify it's actually consumed downstream and not just displayed. "Infrastructure without consumers is dead code."
- Design-mode audit pattern: check features end-to-end (init → update → apply → use) rather than file-by-file. A module that is clean internally but has no callers is still broken. Example: `core/model_merging.py` fully correct code, zero GUI callers (closed Pass 126); `core/rag.py` was only wired in GUI layer, API server never got RAG (closed Pass 125).
- **Design intent must be verified before calling a pattern "wrong direction."** AI-computed state (sentiment-derived emotional_state, engagement score, confidence estimate) is categorically different from user-set config (profile personality dict, system prompt field). Injecting AI-computed state into the prompt is the AI knowing itself — fine. Injecting user-set config into the prompt is the user configuring the AI's character — often a design violation. Confirm which category a field falls into (check for user-facing widgets, profile files, settings pages) BEFORE recommending inject-or-don't-inject.
- **Audit-on-audit is not free — over-retracting is as costly as under-auditing.** When reversing a prior finding, verify the reversal against code one more time. Pass 7 said "inject emotional_state into prompt" (wrong — N-22 already does). Pass 8 said "retract, injection is wrong direction" (wrong — emotional_state is AI-computed, injection is fine). Pass 9 corrected: emotional_state injection OK, ai_profile.personality injection not OK. Three passes of ping-pong before landing on correct. Lesson: check the actual field's origin (user-set vs AI-computed) in one pass, not three.
- **Markdown link audits must verify targets exist on disk** — `grep` alone confirms syntax but not correctness. For every generated link, spot-check the path with `file_search`. Relative-looking paths inside narrative text (e.g. `core/bones.py` in a sentence about the avatar mod) may resolve to a subsystem's local `core/`, not the top-level `enigma_engine/core/` — read the surrounding prose to decide scope. Separately, watch for literal `\n` artifacts introduced by earlier edits where a real newline was needed (breaks markdown table rendering silently).
- **Planning-doc audits need the same tool-verification rigor as code audits.** Pass 148 GUI plan round 1 invented a service-contract location (`enigma_engine/api/`) from memory — round 2 `list_dir` showed api/ was already FastAPI territory, forcing a relocation to `enigma_engine/services/`. Lesson: for every path, framework, or URL named in a plan, run the matching tool (`list_dir`, `grep_search`, `fetch_webpage`) before committing the plan as execution-ready. "I think that folder is empty" is not evidence.
- **"Beat baseline" targets require measuring the baseline first.** A gate or metric written as "≤ baseline" is vacuous if the baseline is never captured. Every rubric metric and hard gate must have a Phase 0 baseline-measurement step on the incumbent system, not just on candidate POCs. Otherwise the comparison at decision time is "POC number vs guess."
- **Self-audit immediately after shipping is mandatory, not optional.** Re-read your own ship under the author's lens (§1 #19) within minutes of landing it; compare each new branch against the *reference pattern* it claims to mirror — divergences from reference are bugs even when the new code passes its tests in isolation. Pass 156d2 caught three real bugs in code shipped 10 minutes earlier (V-7 abort-summary skipped on NaN return paths, V-4 OOM heuristic narrower than reference, V-4 missing RuntimeError/Exception split) — all found by self-audit before user reported anything. Pattern: ship → self-audit same session → fix + add audit-test that would have caught the regression.
- **Test-suite baseline must be diffed against HEAD on session start, not blindly accepted as "pre-existing."** Pass 156z9aj audit on three earlier passes (156z9ag/ah/ai): all three reported "30 pre-existing failures in test_training.py" without checking whether those tests targeted features whose code had been silently deleted from the working tree. `git diff --stat HEAD -- enigma_engine/core/training.py` showed -480 lines vs HEAD; the deleted block contained `_effective_warmup` (Sched-2 close-stamp), `_apo_zero_loss` + `_resolve_preference_loss` (D-9 close-stamp), `set_training_seed(deterministic=...)` + `TrainingConfig.deterministic` (DET-2 close-stamp) — four claimed-shipped features that 100% of the red baseline was probing. Rule: when starting a session whose suite has a non-zero red baseline, run `git diff --stat HEAD -- <suite-targets>` BEFORE quoting the failure count as "pre-existing." A single net-deletion file in a module with adjacent claimed-shipped features in SUGGESTIONS is a doc-vs-code lie that needs either restoration (`git checkout HEAD -- <file>` after user confirmation since it clobbers working-tree edits) or the close-stamps reopened to "regressed, needs re-implementation." Carrying a fake "pre-existing" label forward through multiple passes accumulates dishonesty and hides real workspace damage.
- **Honest time-boxing for a solo builder means sequential, not parallel.** Plans that say "two tracks in parallel, 5 days each" silently assume a team. For one person the honest number is 10 working days plus setup. Write the real schedule; don't hide cost behind the word "parallel."
- **argparse `default=value` blocks user-vs-default detection.** Pass 156z9 → 156z9b: `--temperature` had argparse `default=0.7` (prompts-mode legacy) and forwarded `args.temperature` verbatim to a Magpie path whose library default was 1.0. The 156z9 code carried an aspirational comment "respect explicit user-set --temperature, but if they left it at the prompts-mode default of 0.7 it's almost certainly too low" — argparse cannot honour that: with a non-None default, "user typed 0.7" and "user typed nothing" are indistinguishable. Rule: when a CLI flag's default depends on another flag (mode, subcommand, conditional context), set `default=None` and resolve in `main()` per-mode with an INFO log on default-application. The argparse `help=` text must name every per-mode default so `--help` doesn't lie. The aspirational-comment-as-audit-finding pattern repeats: when your own write-up names a disconnect ("if they left it at default…") without closing it, the comment IS the audit finding written one pass early.

### Process
- New features: implement first, audit second — trace GUI → config → core → training → checkpoint
- Bug fixes: triage before implementing — verify each issue is real first
- Batch related fixes, test each independently

### Concurrency
- `threading.Lock` is non-reentrant — split into public (locks) and private `_unlocked()` methods
- Never `await` inside a `threading.Lock` — use `put_nowait()` or `asyncio.Lock`
- If a loop reads flags under lock, writers must also use the lock
- Lock scope: read under lock, write to disk outside — avoid holding lock during I/O
- `subprocess.PIPE` never drained will hang the child — close pipe, use `DEVNULL`, or drain thread
- **Idle-time schedulers in long-running daemons: reset the cooldown timer inside the work function on EVERY exit path.** Pass 156w + 156x2 audit on `BackgroundTrainer._retrain_on_replay`. Two failure modes, one rule:
  - **Cross-trigger drift (156w):** the work function is called from both an idle scheduler AND a per-batch path. If only the trigger updates the timestamp, a regular replay just before a quiet window leaves the idle gate stale → fires immediately on next wakeup → back-to-back replay burns GPU. Update the timestamp inside the *work function*, not the trigger.
  - **Failure-path drift (156x2):** if the timestamp resets only on the try-block success path, an empty-batch early-return or exception skips the reset → idle gate fires True every loop tick → 1 Hz log-spam. Reset at function ENTRY (after the obvious can't-do-anything guard like model-None) so success / empty / exception all honor the configured interval.
  - When you write a comment justifying "don't reset on failure so we retry," stop and ask "will the retry fix anything, or just hammer the same broken state?" If the latter, reset and let the interval throttle the retry rate.
  - Defensively normalise the interval kwarg (`<= 0 → None`) so a config typo can't peg the GPU; init the timestamp in `__init__` so a fresh trainer never fires immediately on first wakeup.
- **Best-of-N must be deterministic in tie-break, defensive in scorer-failure.** Pass 156x N-16: `EnigmaEngine.generate_best_of_n` runs N independent generations, scores each with caller-supplied `reward_fn`, returns highest. Three rules from author's-lens review before shipping: (1) **Tie-break = first occurrence.** Python's `max(scored, key=...)` returns the first tied element — deterministic. Adversarial test (3 candidates all score 0.5, expect first) catches the regression where someone uses `min` or reversed iteration. (2) **Scorer error = -inf, log, continue.** A `reward_fn` that raises on one candidate must NOT kill the batch — a flaky scorer would take down every best-of-N call. Swallow the exception, log WARNING, assign `-inf` so the broken candidate cannot win, continue scoring the rest. The batch still produces a usable answer. (3) **Logic-eye gate on `temperature <= 0` + `n > 1`.** Deterministic sampling produces N identical candidates — wasted compute. Don't error (user may be probing the scorer); WARNING + proceed.
- **Reward-function wrappers: bind extras with `functools.partial`, don't sniff signatures.** Pass 156x N-16 contract rule: reward functions in `reward_functions.py` have varying signatures (`format_reward(response)`, `math_reward(prompt, response, *, ground_truth)`); caller binds extras via `functools.partial` so the wrapper signature stays uniform `(prompt, response) -> float`. Don't try to detect-and-call multiple signatures inside the wrapper — that's brittle, hides the contract, and silently accepts the wrong reward function when signatures collide. Generalizes to any plugin/callback API where users supply functions with different shapes: define one canonical signature at the boundary, push adaptation out to the user via `partial` or explicit adapters.

### Gotchas — Python Language
- Assignment inside an inner function makes Python treat the variable as local throughout — use a new name (e.g. `do_x = x`) to read from outer scope and override locally
- Generator functions are lazy — body doesn't execute until iterated
- `hash()` is non-deterministic across runs — use `hashlib.sha256` for reproducible algorithms
- Lock acquired before generator + released in finally = leak risk if wrapper fails before iteration
- Methods using Path APIs (`.parent`, `.stem`) crash on `str` — enforce `Path` at call site

### Gotchas — KV Cache & Inference
- KV cache `start_pos` must account for tokens NOT yet processed — a token appended to `generated` but never fed through the model is invisible to the cache
- KV cache: return views not `.clone()` if callers only read; use `torch.roll()` for overlapping shifts
- Speculative/lookahead verify: cache refill + separate last-token call = duplicate KV entry — use one model call that both populates cache and returns logits
- No-cache full-sequence verify must slice logits to the draft region — `verify_logits[0, j]` on unsliced logits indexes from sequence start, not from draft
- DPO prompt mask must match actual encoded prefix length, not bare `encode(prompt)`
- `rewind_cache(pos)` is O(draft_len), `clear_cache()` + re-prefill is O(seq_len) — always prefer rewind after rejected drafts in speculative/medusa/lookahead
- `clear_cache()` destroys the cache object (`_kv_cache = None`); `rewind_cache()` must keep it alive — different semantics, don't conflate
- Subclass `rewind_to()` must chain `super().rewind_to()` first, then handle subclass-specific state (attn scores, INT4 buffers, logical_pos)

### Gotchas — Data Collection
- Gated HuggingFace datasets (The Stack v2, some Llama variants, some Common Crawl mirrors) fail with 401/403 or the word "gated"/"access" in the error — detect on the FIRST language/config attempt and break all subsequent loops immediately. Retrying each language separately after auth failure wastes minutes per language and produces the same error N times. Print the exact recovery command (`huggingface-cli login` + accept license URL) and return.
- HuggingFace datasets with multiple configs (FineMath `finemath-4plus` + `infiwebmath-3plus`) should be blended via separate streaming passes that share the same output dir and progress-key prefix — do NOT try to interleave them in one loop, the API iterators can't be multiplexed.
- Data-collection fetcher naming: reuse the `_fetch_hf_streaming()` helper for single-config datasets; write a custom function only when per-item filtering (license check, language detection) or multi-config blending is needed. Custom functions must still respect the same progress-dict keys, output-dir layout, and Ctrl+C / resume semantics as the helper.
- Wikipedia: use `generator=random` not `allpages`; rate-limits after ~40 requests; use `maxlag=5`
- gutendex.com is unreliable — use curated Gutenberg book ID list with direct URLs
- Non-Latin text: count Latin vs non-Latin chars, use safe default if non-Latin dominates
- File size limits in dataset loaders are silent rejection — check MAX_FILE_SIZE before loading large files
- `combine_all_sources()` uses tmp+rename for atomic writes — follow this pattern for any safety-critical merges
- Resume heuristics: use Content-Length or HEAD request to verify download completeness — file size alone is unreliable
- Streaming data: use two-pass approach — Pass 1 scans metadata + samples, Pass 2 re-iterates for processing; avoids holding full dataset in RAM
- Dedup hash sets: use `.digest()[:N]` (raw bytes) not `.hexdigest()[:N]` (hex strings) — saves ~37% memory per entry at scale
- Unbounded dedup sets need a capacity cap with graceful degradation — 50M entries × 41 bytes ≈ 2 GB is a sane limit
- Paginated API continuation tokens must be deferred until after the dependent call succeeds — advancing before the call loses the batch on failure

### Gotchas — Verification & Correctness
- Verify formulas against docstring examples — trace one concrete case through the math
- Guard patterns must be consistent across ALL sites — grep for pattern, fix all occurrences
- Config validation must cover all numeric fields (denominators, ranges, sign constraints)
- Return sentinels that mean the right thing — `float("inf")` not `0.0` for "couldn't measure"
- Verify variable names match local scope — `getattr(config_obj)` vs `dict.get()`
- Verify features end-to-end lifecycle: init → update → apply → use → restore
- Callbacks defined but never wired are invisible failures — grep all constructors
- Grep for callers of validation functions — defined but never called = false confidence
- Config fields defined but never consumed = dead code — grep the training loop for every TrainingConfig field
- When a function splits into two paths (if/else), verify ALL variables the shared tail uses are set in BOTH paths
- Config converters must set ALL architecture flags explicitly — ForgeConfig defaults (RoPE, RMSNorm, SwiGLU) are wrong for GPT-2 family
- Dead imports from non-existent modules crash at runtime — always `try/except ImportError` with fallback for optional cross-module imports
- **Singular vs plural API names that look like a fallback chain are usually two different things.** Pass 156s `clear_adapter` shipped with `if hasattr(model, "disable_adapters"): model.disable_adapters() else: disable_fn()` where `disable_fn = getattr(model, "disable_adapter", None)`. Looks defensive — it is broken. PEFT's `disable_adapter` (singular) is a `@contextmanager`-decorated method; calling it bare (`disable_fn()`) returns the CM and immediately discards it without entering — adapter stays active, GUI says "cleared", chat keeps using LoRA weights. `disable_adapters` (plural) is the imperative flag setter. They are NOT a primary/fallback pair. Rule: when two API names differ only in singular/plural (or `_v2`/`_legacy`, `add_x`/`add_xs`), do not assume one is a fallback for the other — read the docs/source for both. If only one is correct for your use case, raise on the missing branch instead of falling back to the wrong-semantic sibling.
- **Docstring `Raises:` clauses must enumerate only exceptions the code actually raises.** Pass 156s `apply_adapter` listed `RuntimeError: The adapter's recorded base does not match the loaded model.` but the body never parsed `base_model_name_or_path` and never compared anything — pure aspirational doc. Pass 156i2 / 156k-audit anti-pattern: doc claims more than code delivers. Either implement the check or remove the promise the same pass. Test discipline: a structural test that asserts `"RuntimeError" in inspect.getsource(apply_adapter)` would have caught this — the literal exception name must appear in the body for the docstring claim to be honest.
- **Stale planning comments in shipped code are silent lies.** Pass 156s `clear_adapter` had comment "fall back to set_adapter('') which PEFT treats as no active adapter on some versions" — code never implemented that branch. Comments that describe intended-but-unbuilt behaviour mislead the next reader (often future you) into thinking edge cases are covered. When you cut a planned branch during implementation, delete the comment that promised it.
- **Auto-restore parse blocks must be inside the same except guard as the apply call, or have their own.** Pass 156u-A2: `_restore_lora_adapter_for_base` had `try: engine.apply_adapter_stack(entries) except (FileNotFoundError, ImportError, RuntimeError, ValueError)` — but the entries list was built ABOVE the try via `item.get(...)` and `float(weight)`. A corrupted `route_assignments.json` (non-dict entry → `AttributeError`, non-numeric weight → `ValueError` from `float()`) raised BEFORE the try, propagating up through `_on_model_loaded` and aborting the whole model load. Auto-restore's entire job is surviving its own previous writes (partial writes after OS crash, hand-edits, format drift between versions) — if the parse layer can crash, the feature is worse than not having it. Rule: every saved-state restore that touches user-data shape must (1) explicitly validate the shape with named errors (`isinstance(item, dict)`, `try: float(w) except (TypeError, ValueError)`), (2) on failure drop the whole orphan key + persist the cleanup so the next reload doesn't repeat the crash, (3) surface a loud chat-system / log message so the user knows their saved state was reset. Test the corruption cases adversarially with at least one non-shape-conforming entry AND one non-coercible value type — passing tests on well-formed data prove nothing about the corruption-resilience claim.

### Gotchas — Code Hygiene
- Remove dead code from abandoned approaches immediately
- Cap unbounded string accumulation — rebuild within char budget, drop oldest
- Error handlers must include `traceback.format_exc()`, not just `str(exc)`
- Run lint after batch fixes, not just tests — catches missing imports and wrong-scope variables
- **Lint scope is `enigma_engine/ tests/` not just `enigma_engine/`** — test files accumulate unused imports and F841 dead assignments silently; `ruff check tests/` catches them. Use `--fix` for F401, then `--unsafe-fixes --fix` for F841 (keeps function call, drops variable) and E731 (lambda → def).
- `torch = pytest.importorskip("torch")` pattern: when `torch` is unused but the skip guard is needed, drop to `pytest.importorskip("torch")` (no assignment). Ruff F841 unsafe-fix does this automatically.
- Structural tests catch mixin namespace collisions across 7+ mixins
- Naive algorithms hide behind small test data — check if incremental approach exists
- CPU-bound bulk ops hold GIL — chunk work + `time.sleep(0)` between chunks; sample head+mid+tail for validation
- Test coverage audits: grep callers across ALL test files, not just the "expected" one — tests for module X often live in test_core.py or test_e2e.py
- Test file merges: always `--collect-only` both source AND target to get exact counts, then verify merged count = sum. Read full source tail — truncated reads miss trailing classes
- Every bug fix (S-item) should get a test proving the fix — if there's no test, the fix is unverified
- Optional dependency tests: inject fake module via `types.ModuleType` + `monkeypatch.setitem(sys.modules, ...)` — avoids requiring the real package
- Lazy `__getattr__` modules (e.g. `enigma_engine.core`) break `patch("module.submodule.Class")` — the patcher resolves the dotted path via `getattr` at runtime, which raises AttributeError. Fix: inject fully fake sub-modules via `patch.dict(sys.modules, {"enigma_engine.core.model": fake_mod, ...})` so `from X import Y` inside the target function picks up the mock

---

## 5. TEST LOOP

**Feature work:**
```
Write Test → Build Feature → Run Test → Pass? → Yes: Refactor & Merge Tests → No: Fix & Repeat
```

**Bug fixes — mandatory sequence:**
```
1. Reproduce the bug (confirm it actually exists)
2. Write a test that FAILS because of the bug
3. Fix the bug
4. Confirm the test now PASSES
5. Run full suite — zero failures
```

> A bug fix with no failing test first is unverified. The test is proof the bug existed and proof the fix works.
> If you skip step 2, you may be fixing the wrong thing — or nothing at all.

Command: `python -m pytest tests/ -v`

### Testing Rules

- One test file per module (e.g., `test_training.py` for `training.py`)
- Merge related small test files — no single-test files
- Tests must be independent — no shared mutable state between tests
- Mock heavy dependencies (torch, file I/O) — tests should run in ~17s total
- Delete test files that test removed features
- **Tests specify WHAT, not HOW** — test intended behavior (output, side effects, errors), not implementation details (source patterns, function calls, variable names). A test that passes when the code is wrong is worse than no test.
- **Structural tests (`inspect.getsource`) are a last resort** — only use when behavioral testing requires hardware not available in CI (GPU, GUI). Annotate with a comment explaining why structural is necessary.
- **Write tests from the spec, not from the code** — ask "what should this do?" before reading how it does it. If you read the code first, you'll test what it does, not what it should do.

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
4. Apply the author's lens to the code you are about to touch — would you write it this way? what is it connected to? what connections are missing? Log findings in SUGGESTIONS.md before writing any code.
5. **For bug fixes: write a failing test first** (see Bug Fix Loop above)
6. For features: write the passing test first

### After Coding
1. Run `ruff check enigma_engine/ tests/` then `python -m pytest tests/ -v`
2. Run `python run.py` to verify
3. **Every bug fix must have a test that would have caught the original bug** — if no test exists, add one before closing the fix
4. **Update "Learned Principles"** in this file when a reusable pattern or anti-pattern emerges
5. Update `SUGGESTIONS.md` when confirmed fixes, backlog items, or priorities change
6. Update `GUI_REFERENCE.md` only when visible GUI behavior has changed

### Priority Order (when rules conflict)
**Correctness > Simplicity > Performance > Style**

---

## 8. CODE QUALITY

**Before implementing:**
- Confirm the real problem: "What are you experiencing?"
- Propose 2-3 approaches when there are choices — let the user decide
- Consider edge cases and error paths from the start
- Apply the author's lens: how would you have built this? Is there a better way? What is it connected to, and what connections are missing?

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

**Trust the spec, not the code:**
- The spec, the tests, and §4 Learned Principles are the precedent — code on disk is the suspect
- Remove comments or code when you have a reason and the reason is logged (commit message, SUGGESTIONS entry, or the diff itself if obvious)
- Inside the named overhaul scope (§1 #18): change anything that earns a better outcome
- Outside the named overhaul scope: leave it alone unless it's a real bug, then fix the bug only — no taste-driven drift
- Theoretical improvements with no scar behind them stay in SUGGESTIONS.md until they earn a Pass
- Ignore the installed mods and base model state when working on GUI — not relevant

---

## 9. QUICK COMMANDS

```bash
python collect_vision_data.py --llava-pretrain 100000 --images-dir D:/datasets/llava/images  # V-5 vision SFT data
python collect_vision_data.py --stats                # Show collected vision data summary
ruff check enigma_engine/ tests/                     # Lint (run before every commit)
python -m pytest tests/ -v                           # Run all tests verbose
python -m pytest tests/ --tb=short -q                # Run all tests (compact output)
ruff check --fix enigma_engine/ tests/               # Auto-fix safe lint issues
ruff check --unsafe-fixes --fix tests/               # Fix F841/E731 in test files
python run.py                                        # Show system info
python run.py --gui                                  # Launch desktop GUI
python run.py --serve                                # Start API server (port from CONFIG)
python run.py --serve --port 8080                     # Start API server on specific port
python run.py --train data/training.txt --epochs 10  # Train model
python run.py --train data/training.txt --epochs 10 --seed 42 --deterministic  # Bitwise-reproducible CUDA training (5-15% slower; --deterministic requires --seed)
python run.py --train-tokenizer data/training.txt    # Train BPE tokenizer
python run.py --benchmark                            # Run coherence benchmark on default model
python run.py --benchmark --model models/my.pth      # Benchmark a specific model
python run.py --help                                 # Show all CLI options
python collect_pretraining_data.py --stats            # Show collected data summary
python collect_pretraining_data.py --all-sources      # All sources (wiki, books, fineweb, SE, wayback, owt, c4, dclm, finemath, code)
python collect_pretraining_data.py --fineweb 25       # 25 GB FineWeb-Edu (pip install datasets)
python collect_pretraining_data.py --openwebtext 10   # 10 GB OpenWebText web text (pip install datasets)
python collect_pretraining_data.py --c4 20            # 20 GB C4 cleaned Common Crawl (pip install datasets)
python collect_pretraining_data.py --dclm 15          # 15 GB DCLM model-filtered web text (pip install datasets)
python collect_pretraining_data.py --finemath 10      # 10 GB FineMath step-by-step math (pip install datasets)
python collect_pretraining_data.py --code 10          # 10 GB The Stack v2 code (pip install datasets + HF auth)
python collect_pretraining_data.py --stackexchange    # Stack Exchange Q&A (pip install py7zr)
python collect_pretraining_data.py --wayback 1000     # 1000 Wayback Machine educational pages
python collect_pretraining_data.py --books 500        # Expanded Gutenberg (400+ curated)
python collect_pretraining_data.py --resume           # Resume interrupted download
python collect_pretraining_data.py --combine-only     # Re-merge with paragraph dedup
python collect_distill_data.py --endpoint http://localhost:11434/v1 --model qwen3:8b --prompts data/distill_prompts.txt --tag qwen3_8b  # N-19 external-teacher distill corpus (Ollama / llama.cpp / vLLM / our own --serve); add --resume to skip prompts already in JSONL
python collect_distill_data.py --endpoint http://localhost:11434/v1 --model qwen3:8b --magpie 500 --tag qwen3_magpie --temperature 1.0  # N-19 slice 2: Magpie empty-prefix instruction synthesis (arxiv:2406.08464). Generates 500 instruction/answer pairs from the model itself — no input prompts needed. Add --template chatml|llama3|custom to match the model's chat-template family.
```