# Suggestions

**Last updated:** April 28, 2026 (Pass 156z9f — three open follow-ups closed in one slice: **B-1b** Rust BPE `SPECIAL_TOKENS` aligned with Python's 14-entry dict (was 13 entries with wrong order + included `<SEP>`/`<CLS>` instead of `<search>`/`</search>`; after Rust train, `bpe_tokenizer.py L713` replaces Python's `special_tokens` dict with Rust's, so the drift was silently stripping `search_start_id`/`search_end_id`); **B-2b** `batch_generate` per-prompt attribution — added `last_search_queries_per_prompt: list[list[str]]` initialised in `_init_common`, replaced the legacy join-and-scan with a per-prompt loop that scans each output independently with its own prompt for prompt-side echo slicing; **N-15-validation** `JsonSchemaConstraint.__init__` now validates schema shape at construction (non-dict schema, non-object top-level type, non-dict properties, non-dict property spec, unsupported property type → `ValueError` naming the bad field). All three close items that were explicitly deferred from prior passes. Tests +11 (1 Rust BPE special-tokens parity, 3 batch per-prompt attribution, 7 schema boundary validation). Suite **2620/9**, ruff clean.)
**Tests:** 2620 passed, 9 skipped, 0 failures.
**Rust:** BPE encode/decode/train fully functional. 9/9 tests pass (incl. new special-tokens parity gate).

**Pass 156z9f (Open follow-ups — B-1b + B-2b + N-15 schema validation):** Three small, well-scoped items from prior passes' open lists, closed in one slice rather than left to drift.

- **B-1b — Rust BPE `SPECIAL_TOKENS` alignment** ([rust_extensions/src/lib.rs](rust_extensions/src/lib.rs)). Pre-fix: `SPECIAL_TOKENS` was a 13-entry array with the wrong order, included `<SEP>`/`<CLS>` (uppercase) instead of `<sep>` lowercase, and was missing `<search>`/`</search>` entirely. After every Rust-backed train, `bpe_tokenizer.py L713` overwrites Python's `self.special_tokens` with Rust's dict — so a user training on the Rust path silently lost `search_start_id`/`search_end_id` and got phantom `<SEP>`/`<CLS>` IDs that no other code in the codebase produces. Same shape as the Pass 156z9c "additive load-time merging silently aliases later-added entries" learned principle, but on the train path instead of the load path. Fix: replaced `SPECIAL_TOKENS` with the exact 14-entry list from `bpe_tokenizer.py L44-58` in matching order (`<pad>`, `<s>`, `</s>`, `<unk>`, `<sep>`, `<mask>`, `<Q>`, `<A>`, `<USER>`, `<BOT>`, `<think>`, `</think>`, `<search>`, `</search>`); also extended `special_marker_re()` and the `pre_tokenize` match arm to include `<search>`/`</search>` so the new markers survive pre-tokenization without being chunked into bytes. Rebuilt via `maturin build --release` (8.24s) and reinstalled the wheel via `python -m pip install rust_extensions/target/wheels/enigma_bpe-0.1.0-cp312-cp312-win_amd64.whl --force-reinstall --no-deps`. New behavioural test `test_rust_special_tokens_match_python` trains a tiny vocab via Rust, asserts every Python-default special token is present at the same ID, and explicitly checks `tok.search_start_id`, `tok.search_end_id`, `tok.think_start_id` round-trip correctly. Existing `len(special_tokens) == 13` assertion updated to `== 14`.

- **B-2b — `batch_generate` per-prompt attribution** ([engine_generation.py](enigma_engine/core/engine_generation.py), [inference.py](enigma_engine/core/inference.py)). Pass 156z9d's batch path used `self._record_search_emissions("\n".join(results))` — single scan over the joined batch, which (a) lost which prompt produced which query and (b) defeated the prompt-echo slicing from Pass 156z9e Finding A because there was no per-prompt prompt to slice with. Fix: added `self.last_search_queries_per_prompt: list[list[str]] = []` in `_init_common` so callers can read the attribute on a fresh engine without an `AttributeError`; rewrote `batch_generate`'s tail as a per-prompt loop that calls `_record_search_emissions(output_text, prompt=prompt_text)` for each `(prompt, result)` pair, captures `list(self.last_search_queries)` per prompt into `per_prompt`, and at end assigns `last_search_queries_per_prompt = per_prompt` plus `last_search_queries = flat` (union, for callers that don't care about attribution). Tests in `TestStageB2bBatchPerPromptAttribution`: behavioural parallel-list test on a stub `_GenerationMixin` instance feeding three prompts (one with two queries, one empty, one with one query) asserts the per-prompt list shape and the flat union; structural wire-site tests gate `last_search_queries_per_prompt` initialised in `_init_common` and used in `batch_generate`, plus a regression gate (`'"\\n".join(results)' not in src`) against the legacy join-and-scan. Note: B-2c (flip batch decode to `skip_special_tokens=False`) is still deferred — only matters once Stage B-4 SFT trains the model to emit direct special-token IDs; current byte-level path is unaffected.

- **N-15 schema boundary validation** ([json_schema_mask.py](enigma_engine/core/json_schema_mask.py)). Closed the "API accepts any dict; malformed schema silently degrades" follow-up filed under Pass 156z3 + 156z4. `JsonSchemaConstraint.__init__` now validates: non-dict `schema` → `ValueError("must be a dict")`; `schema['type'] != 'object'` (defaults to `'object'` if absent) → `ValueError` naming the bad type; non-dict `properties` → `ValueError`; per-property spec not a dict → `ValueError` naming the property key; per-property `type` not in `_SUPPORTED_TYPES` (frozenset of `string`, `number`, `integer`, `boolean`, `null`, `object`, `array`) → `ValueError` listing allowed types. The fix lives at the constructor (one chokepoint), not the API layer, so all three call paths (HTTP `/api/chat` json_schema, Python `engine.generate(json_schema=...)`, streaming `stream_generate(json_schema=...)`) inherit the validation for free without per-endpoint duplication. Tests +7 in `TestJsonSchemaConstraintBoundaryValidation`: each rejection branch + two acceptance gates (default top-level type, default property type) so the next refactor can't accidentally tighten the schema beyond what existing valid callers use. No new dependency — pure dict shape checks, no `jsonschema` lib.

- **Why three small items in one pass instead of three passes:** all three were named DONE / DEFERRED in the Pass 156z9d / 156z9e open-follow-up list with concrete fix sketches; each is < 80 lines of code+tests; none touches the same file as another. Cost of one pass with three slices is lower than three passes with three context loads, and the audit lens applied once at the end caught nothing new (no sibling sweeps triggered, no doc-vs-code drift introduced, no cross-method seed/state pattern). Recording all three under one pass header keeps SUGGESTIONS.md history readable; each item still has its own bullet for grep-by-name.

**Pass 156z9e (Stage B-2 audit follow-up):** Six-question author's-lens audit on Pass 156z9d turned up two real bugs that the original ship had not caught.

- **Finding A — logic-eye, doc-vs-code mismatch:** the `_record_search_emissions` docstring said *"scan completed generated text for blocks the model emitted"* but the implementation scanned whatever `text` was passed. On native paths, `text = self._decode_output(output_ids)` decodes prompt+continuation, so any `<search>...</search>` literal inside the user's prompt (asking *about* the syntax) would land in `last_search_queries` as if the model had emitted it. Five sites affected: native `_generate_text` L557, `_generate_with_vision` L1546, `speculative_generate` L1794, `medusa_generate` L1968, `lookahead_generate` L2228. The other three sites (`_generate_text` GGUF L448, `stream_generate` L1247, `batch_generate` L1391) already worked with continuation-only text. Fix: helper signature gained `prompt: str | None = None`; when supplied AND `text.startswith(prompt)`, the prompt prefix is stripped before scanning. Defensive `text.startswith` check protects against off-by-one when the caller has post-processed `text` (e.g. trimmed leading whitespace). Behavioural tests in `TestStageB2PromptEchoSlicing`: prompt-side `<search>` not recorded; continuation-side still recorded; `prompt=None` falls back to full scan (covers GGUF/stream paths); text not starting with prompt skips slicing rather than corrupting offsets. Generalises to a new learned principle below.

- **Finding B — sibling-boundary sweep miss on GGUF chat:** Pass 156z9d hooked all 7 generation paths in `engine_generation.py` but missed three GGUF chat paths in `engine_chat.py` that bypass `_generate_text`/`stream_generate` entirely. `chat()` GGUF branch calls `self.model.chat(...)` and returns directly; `stream_chat()` GGUF server branch yields the response in one piece; `stream_chat()` GGUF llama-cpp streaming branch yields chunks from `create_chat_completion(stream=True)`. Same family as Pass 156z7's GGUF-chat sibling sweep — outer condition is `if ctx.is_gguf and hasattr(self.model, "chat"):`. Hooks added: non-streaming chat scans `response`; server streaming scans `response` before `yield`; llama-cpp streaming wraps its loop in `try/finally`, accumulates chunks into `gguf_chunks: list[str]`, and joins-and-scans in finally so cancellation still flushes. All three sites pass `prompt=None` (default) because llama.cpp returns continuation-only. Structural tests in `TestStageB2GgufChatSiblingSweep` slice the source by branch marker and assert the hook lives in the right branch — locator-based, so a regression that drops the hook from a specific branch fails by name.

- **Finding C (deferred to B-2b):** `batch_generate` decodes with `skip_special_tokens=True`, which strips direct token IDs for `<search>`/`</search>`. Today this is fine because the model can only emit byte-level `<search>` (Stage B-4 SFT not shipped). Once SFT trains the model to emit direct token IDs, batch path will silently miss them while structural test still passes. Added to B-2b follow-up below: when shipping per-prompt attribution, also flip batch decode to `skip_special_tokens=False` and rely on the post-generation regex which is special-token-agnostic.

- **Audit method (re-confirms §1 #19 six-question lens):** ran each of the six questions explicitly against Pass 156z9d. Q1 (would I write it the same way) — yes. Q2 (what is it connected to) — found the engine_chat GGUF branches. Q3 (missing connections) — covered by Q6. Q4 (logic-eye on docstring claim) — caught Finding A. Q5 (claim-vs-test) — original tests passed because they fed bare emission text without a prompt prefix; adversarial prompt-echo test would have caught the bug. Q6 (sibling-boundary sweep) — caught Finding B. Two findings out of six questions = audit was worth doing in-pass; declaring "done" without it would have shipped silent observability bugs.

**New learned principle promoted to AA code maker.md (#4 Verification):** *"`text` parameter on observability hooks must be unambiguous about prompt-inclusion."* When a function accepts text decoded from a sequence and the caller paths split between full-sequence-decode (prompt+continuation) and continuation-only, the helper signature MUST carry the boundary explicitly (e.g. `prompt: str | None`) and either slice or document the assumption. Same anti-pattern hides everywhere `_decode_output(...)` is consumed — grep callers when adding any new text-side scanner.

**Pass 156z9d (AutoResearch-2 Stage B-2 — `<search>` emission detector):** Stage B-1 (Pass 156z9c) registered the tokens; Stage B-2 closes the boundary-signal-without-a-consumer gap by giving the registry an observable consumer. Until Stage B-3 (RAG splice) ships, the consumer is purely observational — log a WARNING when the model emits a search request, store the decoded queries on `engine.last_search_queries` for caller inspection, do not modify generation flow.

**Pass 156z9d (AutoResearch-2 Stage B-2 — `<search>` emission detector):** Stage B-1 (Pass 156z9c) registered the tokens; Stage B-2 closes the boundary-signal-without-a-consumer gap by giving the registry an observable consumer. Until Stage B-3 (RAG splice) ships, the consumer is purely observational — log a WARNING when the model emits a search request, store the decoded queries on `engine.last_search_queries` for caller inspection, do not modify generation flow.

- **Why post-generation scan, not per-token logits hook:** the alternative would be to detect `search_start_id` inside `_sample_token` and accumulate to `search_end_id`, but that touches the hot autoregressive loop (15+ usage sites across 7 generation methods), creates concurrency risk with the existing JSON FSM constraint, and gains nothing observability-wise — the emitted text is the same. Post-generation `extract_search_queries(text)` runs once per completed turn for ~50µs of regex work and zero hot-loop risk. Stage B-3 RAG splice will need the per-token hook because it has to suspend generation, splice context, and resume; that's a separate slice with its own concurrency design.

- **Sibling-boundary sweep (Pass 156z7 discipline):** the helper alone isn't enough — every public generation method that returns final text must call it. Five sibling sites beyond the obvious `_generate_text` were caught up-front by author's-lens scan instead of post-ship audit: `batch_generate` (returns `list[str]` — joined for scan; per-prompt attribution noted as B-2c follow-up), `_generate_with_vision`, `speculative_generate`, `medusa_generate`, `lookahead_generate`. Plus `stream_generate` which yields incrementally — wrapped its yield loop in `try/finally` so the scan runs on both normal completion AND generator cancellation when a caller breaks early. Wire-site test iterates the five direct-return siblings and asserts `_record_search_emissions` appears in each method's source, so a regression that drops the call from one method fails the test by name. The streaming hook does double-duty decode (chunk-join for byte-level emissions which is the likely pre-Stage-B-4 path, plus `tokenizer.decode(new_ids, skip_special_tokens=False)` for direct special-token emissions once SFT trains the model to emit them).

- **`_record_search_emissions` design:** never raises into the caller — Stage B-2 is pure observability layered on top of generation, and a regex bug or import failure must not break user-facing chat. `try`/`except Exception` swallows everything, logs at ERROR via `logger.exception`, leaves `last_search_queries` empty (test `test_helper_swallows_internal_exception` patches `extract_search_queries` to raise and asserts no propagation, queries cleared, ERROR log present). The list is OVERWRITTEN per call, not appended — `last_search_queries` is the most-recent-turn snapshot, not a session accumulator. Test `test_emission_overwrites_previous_call` gates this so a regression to append-mode (which would leak queries across turns and confuse Stage B-3 about which queries are live) fails loudly.

- **Loud-on-real-issue, silent-on-normal-path:** WARNING is logged ONLY when at least one block was extracted. Empty-emission path is silent — confirmed by `test_no_emission_logs_nothing` so a regression that drops the `if queries:` gate (and turns every chat turn into a noise log) fails the test. WARNING message includes the count but NOT the queries — full text goes in the engine attribute for caller inspection, log stays bounded.

- **Boundary-signal-with-a-consumer compliance (Pass 156z + 156z2 lessons):** `last_search_queries` has TWO real consumers as of this slice: (1) the WARNING log itself (operator-visible behaviour change every time the model emits a search) and (2) every test in `TestStageB2SearchEmissionRecording` reads the attribute to verify behaviour. Future Stage B-3 will become a third consumer (RAG splice driver). The signal is NOT dead infra — the test suite is a production-equivalent consumer because every generation path's wire-site is asserted. Per Pass 156z2 lesson: confirmed `_record_search_emissions` itself reaches production by tracing the call chain INWARDS from public entry points (`engine.generate()` → `_generate_text` → helper; `engine.chat()` → `engine.generate()` → same).

- **Tests +12 in [test_chat.py](tests/test_chat.py):**
  - **Behavioural (8):** stub instance built via `object.__new__(_GenerationMixin)` (existing pattern at L266). No emission → empty list + no log. Single emission → recorded. Multiple emissions → ordered list. Unclosed `<search>` → ignored (matches `extract_search_queries` regex contract). Emission overwrites previous call (no append-mode regression). WARNING log on emission with count + "Stage B-2" tag. No log on empty path (silent-normal discipline). Internal exception swallowed → empty list + ERROR log + no propagation.
  - **Wire-site structural (4):** `_init_common` source contains `self.last_search_queries`; `_generate_text` source contains `_record_search_emissions`; `stream_generate` source contains both `_record_search_emissions` AND `finally`; iteration over the 5 direct-return sibling methods asserts the call is present in each source body. Per Pass 156k discipline: structural tests gate WIRING, behavioural tests gate BEHAVIOUR — both required because either alone misses regressions the other would catch.

- **Open follow-ups (Stage B series, updated):**
  - **B-1b** — Rust `SPECIAL_TOKENS` register `<search>`/`</search>` (still open from Pass 156z9c).
  - **B-2b** — Per-prompt attribution for `batch_generate`: today's join-and-scan loses which prompt produced which query. Add `last_search_queries_per_prompt: list[list[str]]` when a real consumer needs it (Stage B-3 batched mode would).
  - **B-3** — RAG splice: per-token hook in `_sample_token` to detect `search_start_id`, accumulate to `search_end_id`, decode query, call into [rag.py](enigma_engine/core/rag.py) for top-k results, splice as `<search_result>...</search_result>` into the context, resume. Concurrency model: single-threaded generator state machine, no race with sampler. Off-switch via `engine.inline_search_enabled` (boolean flag, default `False` until B-3 is shipped + tested with a real corpus).
  - **B-4** — SFT data emitter: synthetic dataset where prompts contain hard factual questions and gold completions show the `<search>...</search>` pattern. Without B-4 even a fully-wired B-2/B-3 sees zero emissions because the model never learned to emit the new tokens. This is the actual capability gate.

**Pass 156z9c (AutoResearch-2 Stage B-1 — `<search>` token registry):** Stage A (post-generation uncertainty gate, Pass 153) is the staged rollout's pure-string layer. Stage B is the inline path: model emits `<search>query</search>` mid-generation, runtime intercepts, fetches RAG, splices the result into the context, and resumes. Stage B-1 ships only the *substrate* — the tokens themselves and the parsing helpers — so all downstream work (B-2 generation hook, B-3 RAG splice, B-4 training data emitter) has unambiguous IDs to gate on.

- **Why a registry slice and not "just add the tokens":** four tokenizers, each with its own `special_tokens` dict, convenience IDs, `_sync_special_ids()`, save/load round-trip, and (for BPE/Char) pre-tokenize regex. Adding the token to one registry without the others would mean Stage B-2 has to branch on tokenizer type, which is the architectural smell the staged rollout exists to avoid. Same shape as Pass 149 Tok-2 BPE byte mode parallel-implementation drift principle: when a subsystem has multiple implementations, infra in one is not coverage of another.

- **Honest legacy-vocab degradation:** if a saved vocab predates Stage B-1, the JSON has no `<search>` entry. Two failure modes were possible:
  - **Aliasing** — keep the in-memory default (`<search>: 12`), so token ID 12 silently aliases whatever the trained model learned to put at ID 12. The model's behaviour at ID 12 is now ambiguous and Stage B-2 can't trust the signal.
  - **Crashing** — KeyError on first `tok.search_start_id` access from a Stage B-2 caller.
  
  Picked the third option: `search_start_id = None` when not in the saved JSON, paired with a load-time `pop()` of any in-memory `<search>` entry that has no disk counterpart. Stage B-2 detects `None` and skips the inline-search path with one DEBUG log per session — feature unavailable, model behaviour unchanged, no aliasing. Mirrors Pass 156i7 "boundary signal honest degradation" pattern. Test `test_legacy_vocab_load_yields_none_search_ids` deletes the keys before load adversarially to force the legacy code path.

- **Bonus latent bug fix in three load paths** (SimpleTokenizer `_load_vocab`, AdvancedBPETokenizer `load`, CharacterTokenizer `_load_vocab`): the previous code did `for k, v in disk.items(): self.special_tokens[k] = v` — additive, which is wrong because in-memory entries that the disk vocab doesn't have were preserved. For non-search tokens this was probably a no-op in practice (every saved vocab had every default special token), but the moment a new special token gets added in code (like `<search>` here), every legacy vocab now has a phantom in-memory entry. Fixed to "rebuild from disk" for the search-token specifically (Simple/Char) or whole-dict re-derive (Advanced) so the load side is the source of truth. Caught by author's-lens scan of the load paths during Stage B-1, not by a failing test from a previous slice — pre-existing latent bug.

- **Sibling-sweep finding (Stage B-1b backlog):** `rust_extensions/src/lib.rs` `SPECIAL_TOKENS` constant has 13 entries in a different order than Python BPE's 12 entries — `<SEP>` (uppercase) vs `<sep>` (lowercase), and `<think>` is at ID 5 in Rust vs ID 10 in Python. After Rust train, Python's `special_tokens` gets replaced via `self.special_tokens = dict(result["special_tokens"])` at L706 of [bpe_tokenizer.py](enigma_engine/core/bpe_tokenizer.py#L706), so the Python-side `<search>` ID gets clobbered. NOT touched in this slice — the Rust↔Python drift is a bigger pre-existing problem and conflating it with Stage B-1 would expand scope outside the named overhaul. Stage B-1b will additively register `<search>`/`</search>` in Rust matching whatever order Python uses, AND optionally close the case/order drift. Until B-1b ships, BPE vocabs trained via the Rust path will not have working `<search>` — production should use Python BPE training or wait for B-1b.

- **Tests +22 in [test_reasoning.py](tests/test_reasoning.py):**
  - **Reasoning helpers (12):** constants exist and match `<search>` / `</search>`; `extract_search_queries` for single, multiple, empty, whitespace-only, unclosed, and multiline blocks; `strip_search_blocks` removes the wrapper; `has_search_request` true / false / unclosed; no collision between search and think extraction (each helper sees only its own tag).
  - **Registry coverage (10):** each of the 4 tokenizers exposes `search_start_id` and `search_end_id` as integers, distinct from `think_start_id` / `think_end_id`; legacy-load yields `None` for both BPE and Simple (the two registries most likely to receive legacy on-disk vocabs); save → reload round trips IDs verbatim; BPE pre-tokenize keeps `<search>` as a single token (caught the elif-clause omission in `_pre_tokenize`); Char encode round trips `<search>q</search>` as three single-token slots; AdvancedBPE encode through the merge-regex path keeps both tokens single-ID (sibling sweep — AdvancedBPE only takes the regex path when `merges` is non-empty; test forces this with a dummy merge).

- **Logic-eye on the docstrings:** Stage B-1 is *substrate only* — the new helpers and IDs do nothing observable to the user yet. Module docstrings in `reasoning.py` say so explicitly: "Stage B-1: token registration + extraction helpers; Stage B-2 (generation-loop interception) and Stage B-3 (RAG splice) are separate work." This is the loud-on-real-issue, silent-on-normal-path discipline applied to feature flags: the substrate must NOT be advertised as user-visible until B-2 ships. If a SUGGESTIONS reader looks for a chat behaviour change after Stage B-1, the docstring tells them where to look (Stage B-2 backlog) instead of confusing them.

- **Open follow-ups (Stage B series):**
  - **B-1b** — register `<search>` / `</search>` in Rust `SPECIAL_TOKENS`. Decide whether to also close the pre-existing case/order drift between Python BPE and Rust.
  - **B-2** — generation-loop hook: after each `_sample_token`, check if the emitted token equals `search_start_id`. If yes, accumulate tokens until `search_end_id`, decode the slice, log a `WARNING "<search> emitted but Stage B-3 RAG splice not implemented; suppressing"` so the boundary signal isn't dead infra. Add an off-switch (`engine.inline_search_enabled = False`) so users can disable while B-3 is being built.
  - **B-3** — RAG splice: when B-2 detects a complete `<search>query</search>` block, call into `enigma_engine/core/rag.py` to fetch top-k results, format as `<search_result>...</search_result>`, append to the context, and resume generation from the new prefix.
  - **B-4** — training data emitter: synthetic dataset where prompts contain hard factual questions and gold completions show the `<search>...</search>` pattern, so SFT teaches the model when to emit the token. Without B-4, even with B-2/B-3 wired, the model will never spontaneously emit `<search>` — this is the actual capability gate.

**Pass 156z9b (N-19 follow-up — CLI temperature default per mode):** Pass 156z9 shipped Magpie's `magpie_collect()` with library default `temperature=1.0`, but the CLI argparse default was still `0.7` (prompts-mode legacy) and was forwarded verbatim — silently overriding the library default any time a user invoked `--magpie` without an explicit `--temperature`. The 156z9 code itself contained an aspirational comment ("respect explicit user-set --temperature, but if they left it at the prompts-mode default of 0.7 it's almost certainly too low") that argparse cannot honour: with `default=0.7`, "user typed 0.7" and "user typed nothing" are indistinguishable. Fix is the canonical pattern for this class of disconnect — `default=None` at the parser, per-mode resolution in `main()`, INFO log on default-application. Single-source: argparse `help=` text now states both per-mode defaults so `--help` doesn't lie. Companion lesson: "doc claims more than code delivers" caught one pass after shipping (the comment was the audit finding written one pass early). Closes the first of the three Pass 156z9 open follow-ups; the GUI "Generate Teacher Corpus" button and optional top-k logprobs capture remain.

**Pass 156z9 (N-19 slice 2 — Magpie empty-prefix instruction synthesis):** Closes the open follow-up filed under Pass 156z5: *"Magpie-style empty-prefix instruction synthesis: arxiv 2406.08464 — pass an empty user turn to a chat-tuned model and let it generate BOTH the user instruction AND its own answer."* The technique exploits a property of well-aligned chat models: given just the chat-template prefix `<|im_start|>user\n` (with NO user content), the model assumes its turn is to propose a question and answer it. Free instruction-tuning data, on-distribution to the teacher.

- **Why it's a separate code path from `collect()`:** prompts-driven mode hits `/v1/chat/completions` with structured `messages=[...]` — server applies the template internally, we never see the markers. Magpie needs raw-prefix completion via `/v1/completions` so we control the exact prefix bytes. Different endpoint, different request body shape, different parsing. Sharing the same client class would have made `TeacherClient` carry two unrelated request shapes — chose a sibling `MagpieClient` instead.

- **Templates ship for `chatml` (covers Qwen2/Qwen3/ChatGLM3+ and most Open-source ChatML-trained models, including the user's existing `qwen3:8b` Ollama default) and `llama3`. `--template custom` supports user-supplied prefix/markers via three flags. `_resolve_magpie_template` raises `ValueError` naming the missing flag at parse time, NOT after the run wastes prompts — a misconfiguration cost is upper-bounded at zero.

- **`_parse_magpie_completion(raw, template) -> tuple[str, str]`:** splits on the user→assistant marker, strips the assistant_end marker if the model emitted it (some servers ignore the `stop` hint and run to the natural end-of-turn), raises `RuntimeError` with descriptive messages on three real failure modes:
  - **Marker missing** — wrong template for model, or temperature caused early stop. The fail loud is critical because without it, the entire raw model output would be written as the "instruction" with an empty completion, polluting the corpus with junk that then trains the student to emit raw chat markers.
  - **Empty instruction** — model emitted the assistant marker without first proposing a question. Empty prompts in JSONL are useless training rows.
  - **Empty answer** — model proposed an instruction but ran out of `max_tokens` before answering. Truncated answers as training data teach the student to stop mid-thought.

- **`MagpieClient.ask_pair(*, max_tokens, temperature, top_p) -> tuple[str, str]`:** stdlib-only HTTP client (mirrors `TeacherClient` pattern). Body uses `prompt=template["prefix"]` not `messages=[...]`. Adds `stop: [template["assistant_end"]]` as a soft hint to the server (Ollama / vLLM / llama.cpp / our own --serve all honour it; OpenAI-compat servers that ignore it still parse correctly because the parser strips the marker as belt-and-braces). Same `URLError → RuntimeError("unreachable")`, `HTTPError → RuntimeError("HTTP NNN")`, `JSONDecodeError → RuntimeError("non-JSON")` discipline as `TeacherClient` — failures bubble up so the driver can count and continue.

- **`magpie_collect(*, endpoint, model, n, tag, template_name="chatml", ..., resume=False) -> dict`:** generates `n` independent pairs. Resume keys off `_prompt_key(instruction)` (SHA-256[:16] of the model-generated instruction) so re-running with `--resume` skips instructions already in the JSONL. Duplicates from the same run (model resamples a popular question) are also dropped — Magpie can converge on a few attractors at low temperature, dedup prevents corpus bias toward those. Failures are logged + counted, never abort. **High-failure-rate warning:** if `failed/n > 0.5` after the loop, logs a WARNING naming `--template` as the likely cause — without that signal a user with the wrong template watches a 30-minute run produce 200 zero-row failures with no actionable hint.

- **CLI mutual exclusion:** `--magpie N` and `--prompts FILE` cannot coexist (different paradigms — Magpie generates, prompts mode reads). Either is required. `parser.error()` rejects both `--magpie + --prompts` and neither-supplied, with messages naming both flags so the user sees the choice.

- **Author's-lens findings before coding:** (1) Reused `_append_jsonl`, `_rewrite_combined_text`, `_load_done_keys`, `_prompt_key`, `_warn_if_remote`, `OUTPUT_DIR` — same on-disk format as prompts mode, so FORGE Distill consumes Magpie output without any trainer-side change (Pass 149 D-11 dual-emit-on-producer principle). (2) `_resolve_magpie_template` returns a fresh dict for `custom` to avoid the caller mutating `_MAGPIE_TEMPLATES`. (3) The `temperature` CLI default (0.7, prompts-mode default) is too low for Magpie's intended diversity (paper uses 1.0); did NOT special-case the CLI per AA "don't add features beyond what was asked" but logged below as a follow-up because the disconnect is a footgun.

- **Tests +20 in [test_collect_distill_data.py](tests/test_collect_distill_data.py):**
  - **Template resolution (5):** known chatml + known llama3 markers, unknown name raises naming choices, custom requires all three (one test per missing slot to gate each individual ValueError message), custom happy-path returns the supplied dict.
  - **Parser (4):** chatml happy path (instruction extracted, answer stripped of assistant_end), missing marker raises (wrong template / base model), empty instruction raises (model emitted assistant marker first), empty answer raises (truncated by max_tokens).
  - **Driver (5):** writes JSONL + canonical `User:/Assistant:` text, rejects `n=0`, failure on one pair doesn't abort batch (RuntimeError counted, next pair succeeds), duplicate instruction within same run dropped (counter on `summary["duplicate"]`), high-failure-rate logs WARNING naming `--template` (caplog assertion gates the user-facing diagnostic).
  - **Resume (1):** existing JSONL row's instruction not re-appended even when client resamples it — counter test on `summary["ok"]` and `summary["duplicate"]` plus row-count assertion.
  - **HTTP client (3):** URL builds to `/v1/completions` NOT `/v1/chat/completions` (regression here would silently fall through to chat completions and break the technique entirely), body sent over HTTP contains the template prefix in `prompt` field AND assistant_end in `stop` list (wire-test via monkeypatched `urllib.request.urlopen` capturing the request body), unreachable endpoint → `RuntimeError("unreachable")`.
  - **CLI (2):** `--prompts` + `--magpie` exits 2 with "mutually exclusive" message, neither supplied exits 2 with both flag names in the error.

- **Self-audit (mandatory per AA, applying all 6 author's-lens questions):**
  1. **Fresh-write?** Yes; structure mirrors `collect()` cleanly.
  2. **Connections?** Reuses 6 existing helpers, no duplication of file I/O / dedup / canonical-text logic.
  3. **Missing connections?** GUI button (logged in 156z5 follow-ups, still open).
  4. **Logic-eye on doc claims:** every docstring claim ("generates n pairs", "resume contract", "high-failure warning", "mutual exclusion") gated by a behavioural test, not just structural.
  5. **Claim-vs-test:** all major branches behavioural; the only structural-style assertion is the URL test (string equality on `_url`), which is the simplest possible behavioural check. No `inspect.getsource` shortcuts.
  6. **Sibling-boundary sweep:** other `collect_*.py` scripts (`collect_finetuning_data`, `collect_pretraining_data`, `collect_vision_data`) all read existing data sources — none synthesize via empty-prefix completion, so no contract family to extend. The technique is unique to teacher-API distillation. ✓

- **Open follow-ups:**
  - **CLI temperature default disconnect:** `magpie_collect` recommends 1.0 for diversity; CLI inherits prompts-mode default 0.7. Either special-case the CLI default when `--magpie` is set, OR log an INFO hint when `temperature < 0.8` in Magpie mode pointing at the paper. Footgun severity: medium (low temp produces less diverse but still valid corpus, just smaller effective coverage).
  - **GUI "Generate Teacher Corpus" button** (still open from 156z5 follow-ups). Now needs a Magpie-mode toggle alongside the prompts-file picker.
  - **Top-k logprobs capture** (still open from 156z5 follow-ups). Lower priority than the GUI surface.

---

**Pass 156z8 (N-15 family close-out — Finding #4 cleanup):** Audit on Pass 156z7's deferred Finding #4 ("`generate_batch` n>1 doesn't forward json_schema"). Grep for `def generate_batch` returned zero matches — the method doesn't exist. The actual referent was `generate_best_of_n` (N-16, Pass 156x). Re-read of [generate_best_of_n](enigma_engine/core/inference.py#L1176) showed it passes `**gen_kwargs` straight through to `self.generate(prompt, **gen_kwargs)` on line 1244 — so `json_schema` already reaches every candidate's `generate()` call, where each candidate is independently constrained by the FSM. **Existing test `test_best_of_n_forwards_gen_kwargs` covers `max_gen`/`temperature`/`top_p` but not `json_schema` specifically.** Added `test_best_of_n_forwards_json_schema_to_each_candidate` that gates the contract explicitly — passes a 3-candidate batch with a real schema dict, asserts all 3 calls received the same `json_schema` object via identity check (`is schema` not `== schema` so dict-mutation regressions also fail). Without this test, a regression where someone "improves" best-of-N to consume specific kwargs without forwarding them silently produces unconstrained candidates labelled as schema-conforming — exactly the same silent-drop class the Pass 156z7 sibling-sweep was hunting. **Finding #4 was a false positive in the original audit** (mis-named the method, the contract was already correct via `**kwargs` passthrough). N-15 contract family confirmed closed at all 4 sites: streaming (z6), chat() GGUF (z7), `_generate_with_vision` (z7), `generate(execute_tools=True)` (z7), and now `generate_best_of_n` gated explicitly. Suite **2544/9** (+1), ruff clean.

**Pass 156z7 (N-15c2 — Sibling-boundary sweep on json_schema):** Closes three silent-drop sites in the same family Pass 156z6 fixed only one of. **What the auto-audit missed:** Pass 156z6 ran the five-question lens on the streaming slice and shipped a GGUF gate in `stream_chat`. Auto-audit covered the diff; it did NOT grep for the *outer condition* `if ctx.is_gguf and hasattr(self.model, "chat"):` to find the non-streaming twin. Same shape: a code path that doesn't go through `_sample_token` with the constraint silently produces unconstrained output, labelled as schema-conforming.

- **Three sibling-boundary fixes shipped:**
  - **#1 [chat()](enigma_engine/core/engine_chat.py) GGUF branch** — added `NotImplementedError` mirroring the Pass 156z6 stream_chat gate. Production path `POST /api/chat` → `state.chat(json_schema=X)` → `engine.chat(json_schema=X)` → GGUF branch was silently passing through to `model.chat(messages=...)` with the schema dropped on the floor. Identical bug to the one 156z6 fixed on streaming, in the file 156z6 already edited — me had the right grep target one keystroke away and didn't run it.
  - **#2 [_generate_with_vision](enigma_engine/core/engine_generation.py)** — added `NotImplementedError` at function entry. Docstring literally said `**kwargs: Ignored (absorbs extra chat kwargs).` — anti-pattern of "loud-on-real-issue silent-on-normal-path" violated, a passed schema is the real-issue branch not normal. Reachable via `engine.chat(images=[...], json_schema={...})` from any Python caller (HTTP API doesn't expose images on /api/chat yet, so lower severity than #1).
  - **#3 [EnigmaEngine.generate(execute_tools=True)](enigma_engine/core/inference.py)** — added `ValueError` with descriptive message naming the right knob. First generation IS schema-constrained, but `_execute_tools_in_text` re-enters `_generate_text` *without* the schema on every tool-call detection — silent partial constraint. The `generate()` docstring already named this as a caveat ("Pair with execute_tools=False — tool execution would feed re-generated text back through this path with the constraint reset"); the gate makes it loud instead of a docstring-only warning the user might miss. Doc claim now matches code delivery.

- **Tests +3 in `TestJsonSchemaConstraintWiring` ([test_research_upgrades.py](tests/test_research_upgrades.py)):** behavioural for all three. `test_chat_gguf_with_schema_raises_notimplemented` builds a `_FakeSelf` with stubbed `_prepare_chat` returning `is_gguf=True` and asserts `NotImplementedError` on the call. `test_generate_with_vision_with_schema_raises_notimplemented` calls `_generate_with_vision` directly with `self=object()` and `vision_features=object()` — the gate must run BEFORE any tokenizer/model access, so a passing test proves the raise sits at function entry not buried after a tensor op. `test_generate_with_schema_and_execute_tools_raises_value_error` patches `enable_tools=True` on a stub, calls `EnigmaEngine.generate(json_schema=..., execute_tools=True)`, asserts `ValueError` matching `"execute_tools"`. Suite **2543/9** (+3), ruff clean.

- **Process change — sixth audit question added to AA code maker.md §1 #19:** *"Did I grep every sibling boundary that shares this contract?"* When shipping a gate at site A, grep the codebase for the *outer condition* (the predicate that defines the boundary, e.g. `is_gguf and hasattr(self.model, "chat")`) to find sites B/C/D in the same family. The fix isn't done until the family is. Companion test rule: one rejection-test per site in the same Pass, not one test plus a comment promising siblings later. Without this principle, the auto-audit at the end of every ship is structurally blind to unchanged-but-related code — which is exactly where 156z6 failed.

- **Finding #4 deferred to backlog (low priority):** `generate_batch` (n>1 prompts) doesn't forward `json_schema` to its internal sampling loop. Internal/research path, not HTTP-reachable. Logged below in the "open follow-ups" of the 156z3-z7 N-15 family.

- **Sibling-sweep self-audit on 156z7 (applying the new sixth audit question to my own ship):** Grepped for `speculative_generate`, `medusa_generate`, `lookahead_generate`, and `_generate_online_dpo_pairs` as candidate sibling boundaries. Result: **none of the three speculative-decoding paths accept `json_schema`** — no kwarg in the signature, no `**kwargs` absorber, zero production callers passing schema. They're not in the contract family yet. Adding `NotImplementedError` gates to methods that don't accept the field would violate AA DO-NOT (guarding against scenarios that can't happen) and would gratuitously expand the API surface. Correct call is a future-proof backlog note: **N-15d (future)** — when constrained decoding reaches speculative/medusa/lookahead (literature has draft-time constraint methods, e.g. constrained speculative sampling), wire `json_schema` through `_sample_token` on the verifier path, NOT the draft path (rejection sampling preserves the verifier distribution; constraining the draft would bias acceptance). `_generate_online_dpo_pairs` is training-internal (RL pair generation) — different contract, not a chat output path. Family confirmed closed at 4 sites (3 fixed in 156z7 + 1 deferred as Finding #4).


---

**Pass 156z6 (N-15c — Streaming + json_schema):** Closes the streaming follow-up filed under Pass 156z3 / 156z4. **What was wrong:** Pass 156z4 shipped `/api/chat` with json_schema constrained generation but loud-rejected the same field on `/api/chat/stream` with HTTP 400, citing "constraint state mutates per token, race against next-token sample". Re-reading the code under the author's lens — `stream_generate` is a single-threaded generator, the FSM `advance()` runs between `yield` and the next `model.forward(...)` call, so there is no concurrency between FSM and sampler. The rejection was a planning artefact, not a correctness gate. Half-wired contract: ChatRequest exposed the field, one HTTP endpoint accepted it, the other refused it — same dead-infra category as the original N-15.

- **Wire-sites closed (3):**
  - [stream_generate](enigma_engine/core/engine_generation.py) — added `json_schema: dict | None = None` kwarg, builds `JsonSchemaConstraint(json_schema, self.tokenizer)` once before the loop (one vocab scan amortised across all tokens, same discipline as `_generate_text`), forwards `json_constraint=` to `_sample_token` (the mask hook lives there from T3-9), advances the FSM after each yield, breaks on `is_done` BEFORE the next-token model.forward call (no wasted GPU work past DONE; would also avoid the all-`-inf` masker → softmax NaN failure mode if the loop kept going).
  - [stream_chat](enigma_engine/core/engine_chat.py) — added explicit `NotImplementedError` raise at the top of the GGUF branch when `kwargs.get("json_schema") is not None`. Sits inside `if ctx.is_gguf and hasattr(self.model, "chat"):` so it covers BOTH sub-paths (server-backend yield-in-one-piece and in-process llama-cpp-python streaming). Mirrors the non-streaming GGUF gate at `_generate_text` L386-396.
  - [/api/chat/stream](enigma_engine/api/server.py) — deleted the 400-rejection block, replaced with omit-when-None forwarding: `if req.json_schema is not None: kwargs["json_schema"] = req.json_schema`. The non-streaming path already follows this pattern (Pass 156z4); streaming now matches. Omit-when-None protects legacy engines whose `stream_generate` lacks the parameter — passing `json_schema=None` always would TypeError on import. Updated the `ChatRequest` field docstring to reflect that streaming now supports schema.

- **Logic-eye on the design:** `_prepare_chat` (L344-410) only pops `max_tokens`/`max_new_tokens` from kwargs and never strips `json_schema`, so the field flows through to `stream_generate` automatically once `stream_chat` accepts it via `**kwargs`. Verified by reading `_prepare_chat` body before editing — no new wiring needed there.

- **Tests +4 net (replaced 1 rejection test, added 4 new):**
  - **Structural** [test_stream_generate_builds_constraint_advances_and_breaks](tests/test_research_upgrades.py) — getsource scan asserts `JsonSchemaConstraint(json_schema, self.tokenizer)` + `json_constraint=json_constraint` + `json_constraint.advance(` + `json_constraint.is_done` all appear in `stream_generate` body. Justified per the existing `_generate_manual` rationale (behavioural path requires stubbing model + KV cache + tokenizer + lock — far more code than the wiring being tested; behavioural coverage of the FSM itself lives in `TestJsonSchemaConstraint` from T3-9).
  - **Behavioural** [test_stream_chat_gguf_with_schema_raises_notimplemented](tests/test_research_upgrades.py) — builds a `_FakeSelf` with stubbed `_prepare_chat` returning a `ChatContext(is_gguf=True, ...)` and a `_FakeModel` with a `.chat()` method (so `hasattr(model, 'chat')` is True), iterates the generator, asserts `NotImplementedError` matching `"GGUF"` on first `next()`. Catches the regression where the GGUF gate is dropped or shifted to AFTER the model.chat call (which would silently emit unconstrained tokens before raising).
  - **Behavioural** [test_chat_stream_forwards_json_schema_when_set](tests/test_api.py) — TestClient POST with `json_schema=`, mock `engine.stream_chat` captures kwargs, asserts the schema reached the call. Replaces the Pass 156z4 rejection test.
  - **Behavioural** [test_chat_stream_omits_json_schema_when_none](tests/test_api.py) — TestClient POST WITHOUT `json_schema`, asserts `"json_schema" not in mock.call_args.kwargs`. Catches the legacy-engine regression where someone "fixes" the handler to always-pass `json_schema=req.json_schema` (a `None` value would TypeError on a stream_generate signature without the parameter).

- **Self-audit (mandatory per AA):** (1) order check — yield → EOS-check break → FSM advance → is_done break → next model.forward. Advance happens AFTER the yield because the user receives the token whether or not it completes the schema; if it does complete, the next forward is skipped (correct). (2) GGUF gate placement — verified inside the `is_gguf and hasattr(...)` outer if, so server-backend sub-branch (L754) and in-process sub-branch are both gated. (3) Docstring claim "stops yielding once the FSM reaches DONE" — verified: the DONE-completing token IS yielded (correct, that token is part of the JSON), no further tokens are yielded after the break. (4) Connections — `_prepare_chat` not stripped, `stream_chat` already passes `**kwargs` to `stream_generate`, so no third-site change needed. All five author's-lens questions check out.

- **Open follow-ups still on the table:**
  - N-19 slice 2 (Magpie `--magpie` mode + GUI "Generate Teacher Corpus" button)
  - API schema validation at boundary (currently any dict accepted; could pre-validate the schema dict shape via jsonschema lib before passing to the FSM)
  - Top-k logprobs capture (research-grade, deferred since user killed in-process teacher)

---

**Pass 156z5 (N-19 — External-teacher distillation, slice 1):** Closes priority row 18. **Design pivot from the original spec:** N-19 was filed as "logit-level knowledge distillation" assuming an in-process teacher model. User Q2 of the design phase: *"we should remove the teacher. i thought it would be good to build an AI on another AI but it seems complicated so we could switch to comunication between another gui that be another one of our guis with another AI or some way to communicate with another AI"*. That kills white-box logit-KD outright (needs co-resident teacher logits) and points at black-box text-only distillation via IPC. The 2024 KD survey ([arxiv 2402.13116](https://arxiv.org/abs/2402.13116)) splits methods into white-box (need logits) and black-box (text only) — the entire Self-Instruct / Alpaca / Magpie family is black-box, and that's the design space the user just put us in.

- **Slice 1 ships:** [collect_distill_data.py](collect_distill_data.py) — top-level CLI. Sends prompts from a `.jsonl`/`.txt` file to any OpenAI-compatible `/v1/chat/completions` endpoint, writes responses to `data/finetune/distill_<tag>.{jsonl,txt}` in the canonical `User: …\n\nAssistant: …` format that the existing FORGE Distill mode already consumes via `Path.read_text` (same pattern as Pass 149's dual-emit-on-producer rule — teach the producer to emit canonical text format, leave the trainer alone). Stdlib only (`urllib.request` + `json` + `hashlib`). No new dependencies.

- **Privacy / safety discipline:** Endpoint defaults to `http://localhost:11434/v1` (Ollama). `_warn_if_remote(endpoint)` logs WARNING on every run when the hostname isn't `localhost`/`127.0.0.1`/`::1` — fits the AA volume table (real misconfig → loud, normal path → silent), and an error-level message would be wrong because cloud endpoints are a legitimate explicit user choice. Failures (`URLError` for unreachable, `HTTPError` for 4xx/5xx, JSON-decode for malformed responses, empty completion) are caught at `TeacherClient.ask`, raised as `RuntimeError` with the upstream context attached, then logged + counted in the driver — they never abort the run. A flaky teacher costs you the failed prompts, not the whole corpus.

- **Resume discipline:** `_load_done_keys(jsonl_path)` reads the existing JSONL output and SHA-256-hashes each `prompt` field to a stable 16-char key. Driver skips any prompt whose key is already in the set. Runs are resumable across crashes, OS reboots, and partial network failures. The `_rewrite_combined_text` final pass rebuilds the `.txt` from the `.jsonl` end-to-end (not append-per-row) so the canonical text file is always a clean snapshot of the JSONL even after multiple resumed sessions. Counter-based test (`test_collect_resume_skips_done_prompts_without_calling_client`) asserts the fake teacher is called exactly once for the one missing prompt out of three — direct check on the resume contract; without it, a regression that re-sends every prompt while still appending unique lines passes the row-count assertion.

- **Author's-lens findings before coding:** (1) `collect_finetuning_data.py` already had `_write_combined_text` writing the canonical `User:/Assistant:` format with empty-input safety; copy that contract instead of re-inventing it (consumer-side discipline preserved). (2) FORGE Distill consumes plain-text via `Path.read_text` — JSONL alone wouldn't satisfy the consumer; dual-emit on the producer is the correct shape per Pass 149 D-11. (3) Empty-completion is a real failure mode (some teacher servers return `""` on context overflow or content-policy refusal) — counting it as `failed` rather than writing empty rows prevents the model from learning that empty responses are valid for some prompts.

- **Tests +19 in [test_collect_distill_data.py](tests/test_collect_distill_data.py):** load_prompts (3) — txt with comments+blanks, jsonl with malformed-line tolerance, missing-file raise; `_warn_if_remote` (4) — three localhost forms (`localhost`, `127.0.0.1`, `[::1]`) silent + non-localhost loud; resume (3) — done-keys round-trip, missing-file empty, prompt-key stability adversarial (`"hello world"` vs `"hello world!"`); `_rewrite_combined_text` (1) — exact byte-level canonical format with empty-completion drop; `collect()` end-to-end (5) — happy path, failed-prompt-doesn't-abort, empty-completion-counted-as-failure, resume-skips-done (counter-on-fake), overwrite-when-resume-false; TeacherClient (3) — URL construction, `URLError` → `RuntimeError("unreachable")`, `HTTPError` → `RuntimeError("HTTP 404")`. All behavioural except where construction would be vastly heavier than the wiring tested. Suite **2537/9** (+19), ruff clean.

- **Open follow-ups (deferred slices):**
  - **~~Magpie-style empty-prefix instruction synthesis~~ — DONE Pass 156z9.** [arxiv 2406.08464](https://arxiv.org/abs/2406.08464) — `--magpie N` mode in [collect_distill_data.py](collect_distill_data.py) + `MagpieClient` hitting `/v1/completions` + chatml/llama3/custom templates + 20 tests.
  - **GUI "Generate Teacher Corpus" button on FORGE page:** spawn `collect_distill_data.py` as a subprocess, stream progress to a status label, then auto-fill the Distill mode's data-path field. UX polish; current path (manual run + manual file-pick) is fully usable. Now needs a Magpie-mode toggle alongside the prompts-file picker.
  - **Top-k logprobs capture:** some endpoints (Ollama, vLLM, OpenAI) expose top-5 `logprobs` per generated token. Could be saved alongside the text for partial-logit-KD experiments. Speculative — the user explicitly killed the in-process logit path, and partial-logit-KD via JSON-over-HTTP is research-grade with no published reference implementation. Defer until a concrete use case appears.

---

**Pass 156z4 (N-15b — chat API exposes json_schema):** Closes the production-caller gap from Pass 156z3 (the AA half-wired-contract pattern again — 156z3 wired the engine but the only HTTP surface in `api/server.py` couldn't reach it). **`ChatRequest`** ([api/server.py L289-296](enigma_engine/api/server.py#L289-L296)) gains optional `json_schema: dict[str, Any] | None = None`. **`AppState.chat`** ([api/server.py L177-208](enigma_engine/api/server.py#L177-L208)) gains the same kwarg and forwards it into `engine.chat(**kwargs)` — but ONLY when non-None. The omit-when-None discipline matters: pre-N-15 engines without the parameter would raise `TypeError` if we always passed `json_schema=None` (the existing TypeError-fallback would then silently drop the kwarg AND every other override). **`POST /api/chat`** handler forwards `req.json_schema` into the per-request kwargs dict before calling `state.chat(...)`. **`POST /api/chat/stream`** explicitly rejects `json_schema` with HTTP 400 + an explanatory error pointing callers at `/api/chat`: streaming-with-constraint-state needs the constraint threaded into the SSE generator (the constraint object's FSM state mutates per token, and yielding tokens one-at-a-time over SSE would race against the next-token sample). Loud rejection beats silent drop — dropping would return free-form tokens labelled as schema-conforming.

- **Tests +5 in `TestChatJsonSchemaWiring`** ([test_api.py](tests/test_api.py)): `ChatRequest` accepts the field with default None and a custom dict; `AppState.chat` forwards the kwarg when set (mock engine assertion on `call_args.kwargs["json_schema"]`); `AppState.chat` OMITS the kwarg when None (legacy-engine compatibility — catches the regression where someone always-passes `json_schema=kwargs.get("json_schema")`); end-to-end production-path test posts to `/api/chat` with schema and asserts the kwarg reached the mock `engine.chat` call (not just the AppState-level test, gates the handler-level forwarding too); stream endpoint returns 400 on schema and the error message names `json_schema` (so the user knows what to drop).

- **Author's-lens findings before coding:** (1) `engine.chat(**kwargs)` already forwards arbitrary kwargs into `engine.generate(...)` so the engine side was free — only the API layer needed plumbing. (2) The pre-existing `TypeError` fallback in `AppState.chat` ("Engine.chat() does not accept kwargs, retrying without") would silently swallow `json_schema=None` if always-passed, causing legacy engines to lose ALL kwargs on the retry path. The omit-when-None pattern avoids that — but the "omits-when-None" test discipline gates the regression. (3) Streaming surface couldn't be wired without redesigning the SSE generator to thread the constraint through; chose loud rejection per the AA volume table (real misconfig → loud, normal path → silent).

- **Open follow-ups:**
  - **~~Streaming + json_schema (N-15c)~~ — DONE Pass 156z6.** `stream_generate` builds the constraint, forwards to `_sample_token`, advances the FSM after each yield, breaks on `is_done`. `/api/chat/stream` 400 rejection replaced with omit-when-None forwarding; GGUF stream raises `NotImplementedError` mirroring the non-streaming gate.
  - **Schema validation:** API accepts any dict shape (`dict[str, Any]`). A malformed schema (no `type`, missing `properties`) reaches `JsonSchemaConstraint` and silently produces degraded output. Pre-existing concern (logged in 156z3); add `jsonschema`-style validation at the API boundary OR at the constraint constructor.
  - **GUI surface:** no CHAT-page widget for entering a JSON schema yet. JSON-mode is primarily a programmatic / tool-call use case; GUI users typing conversational chat don't typically need it. Defer until a real GUI workflow demands it (e.g. a future "forge a structured response" panel).

---

**Pass 156z3 (N-15 — Constrained decoding wired):** Closes priority row 16. Pre-fix audit: `JsonSchemaConstraint` ([json_schema_mask.py](enigma_engine/core/json_schema_mask.py)) shipped under T3-9 with a complete FSM (EXPECT_OPEN → EXPECT_KEY → IN_KEY → EXPECT_COLON → EXPECT_VALUE → IN_VALUE → AFTER_VALUE → DONE), `mask_logits()`, `advance()`, and 5 unit tests. **Production callers: zero.** `_sample_token` ([engine_generation.py L816-817](enigma_engine/core/engine_generation.py#L816-L817)) accepted a `json_constraint` kwarg AND applied the mask if set — but no caller in the entire codebase ever passed it. The FSM never advanced because `.advance()` was unreachable from any production path. Half-built dead infra hidden behind a partially-wired contract — exactly the same pattern as `is_roleplay()` (Pass 156z) and `apply_profile_to_engine` (Pass 156z2), now closed at the third layer.

- **Public API surface** ([inference.py L1003](enigma_engine/core/inference.py#L1003)) `EnigmaEngine.generate(...)` gains `json_schema: dict | None = None`. Docstring `Args` section names two honesty caveats up front: (1) GGUF models route through llama.cpp's own sampler — our logit mask never sees those logits, so `json_schema` raises `NotImplementedError` on GGUF rather than silently returning unconstrained output labelled as schema-valid; (2) `execute_tools=True` is unsafe with `json_schema` because tool-execution would feed re-generated text back through `generate()` with a fresh (unconstrained) call, breaking the structural guarantee. Constraint construction pays a one-time vocab scan (~50K decode calls for a 50K-token tokenizer) per `generate()` call, NOT per token.

- **Threading** through three layers: ([inference.py L1136-1142](enigma_engine/core/inference.py#L1136-L1142)) public `generate` forwards `json_schema=` to `_generate_text`; ([engine_generation.py L466-471](enigma_engine/core/engine_generation.py#L466-L471)) `_generate_text` builds `JsonSchemaConstraint(json_schema, self.tokenizer)` ONCE per call (vocab scan amortised) and forwards it to `_generate_manual` as `json_constraint=`; ([engine_generation.py L640+L666-672](enigma_engine/core/engine_generation.py#L640)) `_generate_manual` passes the constraint to `_sample_token` (where the existing mask hook lives) AND calls `json_constraint.advance(int(next_token[0,0].item()))` after each sampled token, AND breaks the loop on `json_constraint.is_done`. The early-stop on `is_done` is non-optional: past DONE the masker returns the empty allowed set → all logits become -inf → softmax NaN → existing pre-filter NaN-guard salvages the sample but the result is meaningless.

- **GGUF-path explicit raise** ([engine_generation.py L386-396](enigma_engine/core/engine_generation.py#L386-L396)): added BEFORE the GGUF native-generate call so callers fail loud at the API boundary, not silently get back free-form text labelled as schema-valid. Per the AA "loud-on-real-issue" rule: real misconfiguration → loud (`NotImplementedError`); native-PyTorch path → silent.

- **Author's-lens findings before coding:** (1) Mask hook in `_sample_token` was already wired (Pass T3-9 had finished half the job); only the constraint-supply side was missing. Fix is two methods up the call chain, not at the sampler. (2) The constraint's `advance()` was missing entirely from `_generate_manual` — the FSM had no driver. (3) `is_done` early-stop was absent — without it, a successful schema completion would still run to `max_gen` with the masker blocking every token. (4) Docstring of `JsonSchemaConstraint.__init__` already named the call-site contract (`mask_logits` then sample then `advance`) but no production code matched it — same docstring-aspiration pattern as Pass 156s `apply_adapter`'s false `Raises` clause.

- **Tests +4 (3 structural wire-site gates + 1 behavioural GGUF rejection):** Structural is justified per AA last-resort rule because the behavioural path through `_generate_manual` requires stubbing `model.forward` + KV cache + `_build_exempt_tokens` + `_adaptive_stop_interval` + tokenizer.decode/eos_token_id — far more code than the wiring being tested, and `JsonSchemaConstraint` itself is already covered by 5 behavioural unit tests in `TestJsonSchemaConstraint` (Pass T3-9). Wire-site tests gate the literal idiom at each layer:
  - `test_public_generate_signature_accepts_json_schema` — `inspect.signature(EnigmaEngine.generate)` contains `json_schema` AND source contains `json_schema=json_schema` (catches the regression where someone keeps the kwarg in the signature but drops the forwarding line, making the public flag a silent no-op).
  - `test_generate_text_builds_constraint_and_forwards` — source contains the literal `JsonSchemaConstraint(json_schema, self.tokenizer)` (the only valid construction signature) AND `json_constraint=json_constraint` (forwarded to `_generate_manual`).
  - `test_generate_manual_advances_constraint_and_early_stops` — source contains all three required idioms: `json_constraint=json_constraint` (passed to `_sample_token`), `json_constraint.advance(` (FSM driver), `json_constraint.is_done` (early-stop). Each one's failure mode is named in the assert message so a regression author sees what they broke.
  - `test_gguf_model_with_schema_raises_notimplemented` — behavioural; sets `_is_gguf=True` on a `_FakeSelf`, calls `_generate_text` directly with `json_schema={...}`, asserts `NotImplementedError` containing "GGUF". Fast (no model load, no KV cache).

- **Open follow-ups:**
  - **N-15b (production caller wiring):** the constraint is reachable from any code that calls `EnigmaEngine.generate(json_schema=...)`, but no GUI / API surface exposes it yet. Logical next slice: chat endpoint optional `json_schema` field, plus a CHAT-page checkbox + textbox for users to pin a schema before sending a message. Without this slice the wiring is reachable from Python callers (`engine.generate(prompt, json_schema={...})`) but not from the GUI/API — the consumer chain is one production layer short of the user. Same shape as Pass 156z2 (closed by wiring the API endpoint).
  - **Schema validation:** `JsonSchemaConstraint` accepts any dict; malformed schemas (missing `properties`, wrong types) silently produce degraded output. Guard with a `jsonschema`-style validator at construction OR raise on the first malformed property. Pre-existing concern, not new in this slice.
  - **Tool-exec compatibility:** the docstring warns about `execute_tools=True` + `json_schema` being unsafe; a stricter version would be a hard `ValueError` at the API boundary. Defer until a real caller hits the case.

---

**Pass 156z2 (audit-on-audit fix — close the dead-consumer chain):** Mandatory self-audit on Pass 156z (per AA rules) found that the function I claimed was the "first production consumer" of `is_roleplay()` — `apply_profile_to_engine` — itself had **zero production callers**. Grep across `**/*.py` for `apply_profile_to_engine`: 14 hits, all in tests or self-references. Same anti-pattern as Pass 156y, applied at the next layer down: I wired one piece of dead infra to another piece of dead infra. The new "Boundary signal without a consumer = dead infrastructure" principle added to AA code maker.md in Pass 156z was violated in the same pass that wrote it.

- **Real production profile-apply path identified:** Only `POST /api/profiles/{profile_id}/activate` in [api/server.py](enigma_engine/api/server.py) actually receives profile activations from any production caller (GUI hits the API, no direct Python entry-point). Pre-fix, that endpoint *only* applied the `generation` block (`temperature`, `top_p`, etc.) and silently dropped `system_prompt`, `adapter`, and the `is_roleplay()` boundary on the floor. `AIProfileManager.switch_profile` is also test-only. So the entire profile-apply contract — system prompt, adapter, personality boundary — was broken end-to-end before this pass.

- **Fix:** [api/server.py L772-790](enigma_engine/api/server.py#L772-L790) `activate_profile` endpoint now calls `apply_profile_to_engine(AIProfile.from_dict(data), state.engine)` when an engine is loaded. Engine-not-loaded path stays a no-op (existing UX where users set the active profile id before loading a model still works). No defensive try/except wrapper — `apply_profile_to_engine` already uses `hasattr` guards throughout and catches adapter failures internally, so adding another layer would be guarding against scenarios that can't happen (per AA rule).

- **Behavioural test (test_api.py +1):** `test_activate_profile_applies_to_loaded_engine` injects a stub engine with the attributes `apply_profile_to_engine` touches, posts to `/api/profiles/{id}/activate` with a profile carrying `system_prompt`, `personality`, and `generation`, asserts the stub's `system_prompt` and `temperature` were mutated AND `clear_adapter()` was called. Without this test, a regression where someone reverts the `apply_profile_to_engine` call back to bare `state.config_overrides.update(...)` silently re-breaks the production path — `test_activate_profile_corrupt_json` would still pass because it only checks the corrupt-input branch.

- **Stamp correction:** Pass 156z claimed "first end-to-end consumer for `is_roleplay()`." That claim is now true after Pass 156z2, not after Pass 156z. Suite **2509/9** (+1), ruff clean.

- **Author's-lens lesson (added to AA code maker.md):** "Self-audit findings can land within minutes of shipping if you actually run the lens. Pass 156z shipped at T+0; Pass 156z2 audit caught the dead-consumer bug at T+5min. Rule reaffirmed: when wiring a 'first consumer' for a previously-dead signal, grep the consumer ITSELF for production callers — don't assume that because the new wire-site exists in code, the function it lives in is reachable from production. Two-layer dead infra is a real failure mode."

---

**Pass 156z (Personality-4 design landed + Personality-3b canonical-template cleanup):** Closes the ARCH-GAP that's been sitting on P1 row 10 since Pass 156y. User design call (in-conversation, Pass 156z): *"I was hoping for the AI to develop one but I do not mind the ability of roleplay."* Reading: base AI personality is weight-trained per Personality-5; the `AIProfile.personality` dict is a **roleplay-character overlay** when populated, and an empty dict means the profile is either base-AI or a task overlay (system_prompt + adapter only). Use-case starter profiles (`coding_helper`, `creative_writer`, `researcher`) are TASK overlays, not characters — so their generic 4-knob blocks (`tone/verbosity/formality/humor`) were decorative legacy that falsely flipped `is_roleplay() == True` for the wrong reason.

- **Disk JSON cleanup (Personality-3b for canonical templates):** [profiles/coding_helper.json](profiles/coding_helper.json), [profiles/creative_writer.json](profiles/creative_writer.json), [profiles/researcher.json](profiles/researcher.json) all now have `"personality": {}`. The fourth canonical template ([profiles/assistant.json](profiles/assistant.json)) was already clean from Pass 156y2.

- **In-memory `DEFAULT_PROFILES` cleanup:** [ai_profile.py](enigma_engine/core/ai_profile.py) — same three role-template entries lose their `personality={...}` kwarg (defaults to `field(default_factory=dict)` from the dataclass). Per-entry comment names the design call so the next maintainer can't silently re-populate.

- **End-to-end consumer for `is_roleplay()` (closes the dead-infra gap):** Pass 156y shipped the signal but no production code branched on it. `apply_profile_to_engine` is now that consumer — it logs `INFO "Applied roleplay profile '<name>' to engine (personality overlay: [<keys>])"` on the True branch and `INFO "Applied profile '<name>' to engine"` on the False branch. Behaviour is identical either way; only the marker differs. This is the smallest meaningful consumer that proves the signal-vs-no-signal branch is wired end-to-end without forcing a bigger UX redesign (system-prompt prefix injection, runtime persona swap) that would need its own design pass.

- **Docstring tightening:** `AIProfile` "Identity vs roleplay" section reframed from Personality-3 (boundary added) to Personality-3/4 (boundary added + design call made). Now explicitly names task-preset profiles as a third category that keep `personality` empty and steer through `system_prompt` + generation knobs only.

- **Untouched on purpose:** [profiles/not_for_you_hahaha.json](profiles/not_for_you_hahaha.json) keeps its legacy 4-key auto-default block. By name + system_prompt ("goblin") it's the user's own profile, not a canonical template — Pass 156y2 logged it under the deferred-legacy-migration note. A blanket disk strip would be hostile to user state. Decision deferred until either (a) the user asks for migration, or (b) `AIProfileManager.create_profile` re-emits `personality={}` going forward and a one-shot migration sweeps existing user profiles. No production code reads `profile.personality` outside the new log line, so the orphan knobs there are still harmless today.

- **Author's-lens findings before coding:** (1) Grep confirmed `profile.personality` has zero consumers other than `to_dict`/`from_dict` round-trip — populated knobs on task profiles were pure noise. (2) No tests asserted populated personality on the three role-template defaults, so cleanup is test-safe. (3) The `is_roleplay()` signal had ZERO production consumers — Pass 156y shipped it as pure infrastructure; without a consumer in the same or next slice, it's the "infrastructure without consumers is dead code" anti-pattern. Closed in this pass.

- **Tests +4 (3 parametrized behavioural load-path + 1 behavioural log-branch):** `test_canonical_role_template_disk_profile_is_not_roleplay[coding_helper|creative_writer|researcher]` — mirrors the Pass 156y2 `assistant.json` gate for the three role templates, loads each via `load_profile()`, asserts `is_roleplay() is False` AND `personality == {}`, skips gracefully if the file is missing. `test_apply_profile_to_engine_logs_roleplay_branch` — uses `caplog` to capture INFO logs on both branches via a `_NullEngine` stub: roleplay branch must contain "roleplay" + the profile name + the personality keys (so audit logs can see overlay shape); base branch must contain the profile name and must NOT contain "roleplay" (catches the regression where someone collapses the branch back to a single hardcoded line, silently re-hiding the boundary). Suite **2508/9** (+4), ruff clean.

- **Open follow-ups (Personality-5 cluster):**
  - **Personality-3b (legacy user profile)** — `not_for_you_hahaha.json` still carries the legacy 4-key auto-default. Defer until either user-driven migration request or a one-shot `AIProfileManager` migration helper.
  - **Personality-5 BUILD** — operational, not code: schedule a FORGE personality-distillation run as part of the next training cycle to bake initial character traits into weights ([gui_forge_new_modes.py L1259](enigma_engine/gui/gui_forge_new_modes.py#L1259) already has the category and prompts). This is *the* mechanism by which the AI "develops one" per the user's stated intent.
  - **Row G** — consistency loss / per-profile regularizer (medium effort, needs consistency metric design first).
  - **Stronger `is_roleplay()` consumers** — the log-line consumer is intentionally minimal. A future pass could route the signal into system-prompt construction (prepend "You are <character>..." framing only on roleplay profiles) once a chat-runtime persona surface exists — Session-1 follow-up notes that today's "profile" widget is forge-only.

- **Author's-lens lessons (added to AA code maker.md):** "Shipping a boundary signal without a consumer in the same or adjacent pass = dead infrastructure. Pass 156y added `is_roleplay()` with zero production callers. The signal sat for two passes (156y2 audit shipped a load-path test but no consumer) before Pass 156z wired one. Rule: every new boundary/identity/category signal must be paired with at least one observable consumer (log line minimum) in the same slice — if you can't name the consumer up front, you don't have a signal yet, you have a fantasy."

---

**Pass 156y2 (audit on Pass 156y):** Self-audit using AA author's-lens five questions. One real bug found and fixed; two notes deferred.

- **Finding 1 (claim-vs-reality, Pass 156i2 anti-pattern).** Pass 156y stamp said *"`assistant` base profile cleaned"* — only the in-memory `DEFAULT_PROFILES["assistant"]` constant was cleaned. The on-disk [profiles/assistant.json](profiles/assistant.json) (which is what `load_profile()` actually reads in the GUI / API path — JSON wins) STILL had `{tone:"helpful", verbosity:"balanced", formality:"casual"}`. So at runtime, every user loading the canonical base profile satisfied `is_roleplay() == True`, the exact opposite of the new contract. The Pass 156y write-up itself even narrated the gap (*"five JSON files in profiles/ already store personality blocks; library default change is a no-op for them because JSON wins on load. Slice is purely additive; no consumer changes needed."*) — Dia saw the disconnect and walked past it. Fixed: `profiles/assistant.json` `personality` block now `{}`. The 3 use-case templates (`coding_helper.json` / `creative_writer.json` / `researcher.json`) intentionally untouched — they're use-case starter overlays whose semantics need the Personality-4 design call before edit.

- **Finding 2 (test gap, claim-vs-test, Pass 156k-audit Part 2 anti-pattern).** All 5 Pass 156y tests exercised the in-memory default + the new method on synthetic objects. None of them exercised the **load path** — `load_profile("profiles/assistant.json").is_roleplay()` was untested, and would have returned `True` against the broken canonical disk file with no test signal. Added behavioural test `test_canonical_assistant_disk_profile_is_not_roleplay` ([tests/test_core.py](tests/test_core.py)) that loads the actual repo file via `load_profile`, asserts `is_roleplay() is False` AND `personality == {}`. Skips gracefully if file isn't present (so the test survives weird checkouts). Catches both the original drift and any regression where someone re-adds knobs to the JSON.

- **Note 1 (deferred — legacy auto-injected user profiles).** [profiles/not_for_you_hahaha.json](profiles/not_for_you_hahaha.json) ships with the OLD 4-key default (`tone/verbosity/formality/humor`) in full. Likely auto-injected by `AIProfileManager.create_profile` at creation time when the library default was still populated, so it's a legacy-migration question, not a Personality-3 contract violation by intent. By name + system prompt it's plausibly meant to be a roleplay profile (where the populated dict is correct). Logged as part of **Personality-3b** — when that pass lands, decide on a one-shot migration that strips the legacy 4-key auto-default but preserves anything the user actually customised.

- **Note 2 (deferred — Pass 156y line in narrative).** The Pass 156y stamp paragraph in this file already said the disk files were "a no-op for the library default change". That sentence remains technically correct AS WRITTEN — the library default change alone IS a no-op for them — but it framed inaction as fine when in fact the canonical file needed its own fix. Tightened the Pass 156y2 narrative above to flag the disconnect explicitly so the next reader sees the correct story without back-reference.

- **Author's-lens lessons (added to AA code maker.md):** "When a slice changes a library default, grep all on-disk artifacts that round-trip the same field — config JSONs, profile JSONs, registry JSONs. JSON wins on load; library-side default changes are no-ops for any file that has the key written out, regardless of whether the file's value is now contractually wrong. The smell is when the slice's own write-up *narrates* the disconnect ('library change is a no-op for the JSONs') without closing it."

- **Tests +1.** Suite **2504/9** (+1), ruff clean.

---

**Pass 156y (Personality-3 — `AIProfile.personality` boundary fix):** First reviewable slice of P1 row 10 (Personality-5 cluster). Closes the ARCH-GAP where `AIProfile.personality` was a populated default (`tone="helpful"`, `verbosity="balanced"`, `formality="casual"`, `humor="occasional"`) on every base profile, even ones the user never configured. Spec source: SUGGESTIONS.md Personality-3 line 1168 + R-PERSONALITY-1 line 1171 (resolved Pass 146): "stable personality should be weight-trained, not runtime-user-configured."

- **[ai_profile.py](enigma_engine/core/ai_profile.py) `AIProfile`** — default `personality` flipped from a 4-key populated dict to `field(default_factory=dict)`. New `is_roleplay() -> bool` method returns `bool(self.personality)` so downstream code (system-prompt builders, future Personality-4 identity guards) can branch cleanly. Class docstring + field comment now explicitly mark personality as a **roleplay overlay only** — base AI identity is weight-trained per Personality-5, not configured here.

- **[ai_profile.py](enigma_engine/core/ai_profile.py) `DEFAULT_PROFILES["assistant"]`** — cleaned. The `assistant` template is the BASE AI default; populating its `personality` block contradicted the new contract. Comment in place explains the choice for the next maintainer. Other three role-templates (`coding_helper` / `creative_writer` / `researcher`) intentionally retained for now — they're use-case starter overlays, semantically closer to roleplay characters than to base identity, and reframing them is a bigger UX call. Logged as **Personality-3b** below.

- **Author's-lens findings before coding:** zero code reads `profile.personality` anywhere — field was decorative, only round-tripped through JSON. Five JSON files in `profiles/` already store personality blocks; library default change is a no-op for them because JSON wins on load. Slice is purely additive; no consumer changes needed.

- **Tests +5 (all behavioural except one structural doc-gate):** `test_default_profile_has_empty_personality` (RED→GREEN, catches default-revert regression); `test_is_roleplay_false_on_base_profile` + `test_is_roleplay_true_when_personality_populated` (signal correctness, both directions); `test_default_profile_roundtrip_preserves_empty_personality` (catches `from_dict` regression where empty defaults to populated); `test_personality_field_doc_marks_roleplay_only` (structural doc-gate per Pass 156s2 docstring-lies anti-pattern — asserts the word "roleplay" appears in `AIProfile` source so the contract is hard to silently revert). Suite **2503/9** (+5), ruff clean.

- **Open follow-ups (Personality-5 cluster):**
  - **Personality-3b** — cleaning `coding_helper` / `creative_writer` / `researcher` `DEFAULT_PROFILES` and the matching disk JSONs in `profiles/`. They're use-case templates, not strict roleplay characters. The honest fix needs a Personality-4 design decision first ("are use-case profiles roleplay or weight-trained variants?"); deferred until Personality-4 lands.
  - **Personality-4** — identity-vs-roleplay separation design. ARCH-GAP, requires user input on whether "be yourself" is a hardcoded state (no profile) and profiles only mean "act as character X". The new `is_roleplay()` signal unblocks the implementation side once the design call is made.
  - **Personality-5 BUILD** — operational, not code: schedule a FORGE personality-distillation run as part of the next training cycle to bake initial character traits into weights ([gui_forge_new_modes.py L1259](enigma_engine/gui/gui_forge_new_modes.py#L1259) already has the category and prompts).
  - **Row G** — consistency loss / per-profile regularizer (medium effort, needs consistency metric design first).

---

**Pass 156x2 (audit — Pass 156x + 156w):** Self-audit using the AA author's-lens five questions. Two real findings shipped, two deferred.

- **Finding 1 (logic-eye on Pass 156x doc claim).** [inference.py](enigma_engine/core/inference.py) `generate_best_of_n` docstring said "`temperature == 0` (or any deterministic configuration) with `n > 1` produces identical candidates" but code only checked `temperature <= 0` — `top_k=1` and `top_p` near zero ALSO collapse to deterministic and got NO warning. Pass 156i2 / 156k-audit anti-pattern: doc broader than code. Tightened doc to match code ("`temperature <= 0`") and explicitly noted that `top_k=1` / low `top_p` are NOT detected so callers know the limit. Safer than extending the check (which would mis-classify legitimate low-temperature configs).

- **Finding 2 (logic-eye on Pass 156w cooldown reset).** [router.py](enigma_engine/router.py) `BackgroundTrainer._retrain_on_replay` reset `_last_anchor_replay_at` only on the try-block success path (after `replay retrain complete`). Two early-out paths skipped the reset: model/optimizer-None guard (helper also gates on this — fine) and **the empty-batch early-return at L594**. Real-world trigger: user points anchor path at a JSONL with all malformed rows. Pass 156i6 caches the empty `_load_anchor_examples` result so disk I/O is fine, but the idle helper still fires True every loop tick → `_retrain_on_replay` returns at the empty-batch guard → timer never resets → helper True again next tick → **log spam at ~1 Hz** ("idle anchor rehearsal (Ns since last replay)"). The original Pass 156w comment said "failed pass naturally retries" — wrong design: retrying at 1 Hz won't fix a broken anchor file, it just spams logs. **Fix:** moved the timer reset to function ENTRY (right after the model/optimizer guard). Every exit (success, empty-batch, exception) now honors the configured interval — the user sees one INFO log per interval, not N per second. Comment block updated to capture the design rationale so the next maintainer doesn't reinvent the misfeature.

- **Tests +3 (1 new behavioural for the bug, 2 adversarial for Pass 156x branch coverage).** `test_retrain_on_replay_resets_idle_timer_on_empty_batch` ([test_training.py](tests/test_training.py)): build a `BackgroundTrainer` with no anchor file and empty replay buffer, set `_last_anchor_replay_at` to 2 hours ago, call `_retrain_on_replay()`, assert timer advanced. RED before fix (assertion `before > before`), GREEN after. `test_best_of_n_rejects_non_int_n` ([test_inference.py](tests/test_inference.py)): `n=2.5` and `n="3"` both raise `ValueError` — catches a regression where someone drops `isinstance(n, int)` from the boundary guard. `test_best_of_n_handles_non_numeric_score`: scorer returns `None` (real-world: a JSON-parsing judge that fails returns `None`) — the `try/except` doesn't fire (no exception raised) but `float(None)` then raises TypeError — must be swallowed and logged the same as a raising scorer. Adversarial against the regression where someone reorders the score-coercion to before the try-block.

- **Deferred (logged here, not shipped):** (1) **Best-of-N batch-level lock.** `generate_best_of_n` calls `self.generate` N times; `_generation_lock` is acquired and released per-call. If `apply_adapter` / `clear_adapter` / model unload sneaks in between iterations, candidates 1..i and i+1..N are scored against different model states. Real-world risk is low (chat is single-threaded), but the contract is silently violated. Fix would be a new `_best_of_n_lock` held across the whole batch — design discussion needed because holding the inference lock for N seconds starves chat. (2) **Best-of-N reward-fn signature helpers.** `reward_functions.py` scorers have varying signatures (`format_reward(response)`, `math_reward(prompt, response, *, ground_truth)`); caller binds via `functools.partial`. No helper shipped because no caller exists yet. Add when the first caller (FORGE benchmark mode? GUI "regenerate best-of-3" button?) is built.

- **Author's-lens findings on Pass 156v Step 2 / Pass 156w wiring:** No bugs found. Both pass codes deliver what their docstrings claim, tests match behaviour, no stale-comment lies. Audit clean on those slices.

Suite **2498/9** (+3), ruff clean.

---

**Pass 156x (N-16 Best-of-N sampling):** Foundation reward path was already on disk (`reward_functions.py` shipped Pass 142 — `format_reward`, `math_reward`, `code_reward`, `llm_judge_reward`, `reasoning_reward`); this pass adds the consumer-side wiring. **Library** [inference.py](enigma_engine/core/inference.py) `EnigmaEngine.generate_best_of_n(prompt, n, reward_fn, *, return_all=False, **gen_kwargs)` runs N independent `self.generate(prompt, **gen_kwargs)` calls, scores each with `reward_fn(prompt, response)`, and returns the highest-scoring response (or `(best, [(resp, score), ...])` when `return_all=True`). **Validation:** `n < 1` raises `ValueError` immediately (loud-on-real-issue — silent no-op would have the user wondering why best-of-N never improves anything). **Logic-eye gate:** when `n > 1` AND `temperature <= 0.0`, logs WARNING that all N candidates will be identical — doesn't error because the user might be probing the scorer. **Robustness:** a `reward_fn` that raises on one candidate is logged at WARNING and that candidate is assigned `-inf` so it cannot win; the batch still completes. **Tie-break:** Python's `max()` returns the first tied element — deterministic, catches the regression where someone uses `min` or reversed iteration. **Caller contract:** reward functions in `reward_functions.py` have varying signatures (`format_reward(response)`, `math_reward(prompt, response, *, ground_truth)` etc.); caller is responsible for binding via `functools.partial(math_reward, ground_truth=42)` so the wrapper signature stays uniform `(prompt, response) -> float`. **Tests +9 (all behavioural via unbound-method `_FakeSelf` pattern — no real model needed):** rejects n=0, rejects negative n, returns highest-scoring candidate (3 distinct strings, scorer picks middle one), `return_all=True` yields full `[(resp, score), ...]` in generation order, ties break by first occurrence (adversarial), forwards `gen_kwargs` unchanged (Pass 156k label-tracking gate — catches the regression where kwargs land in a local that's then ignored), warns on temperature=0 + n>1, swallows reward errors and continues, n=1 degenerate case (still scores, no special-case branch). Suite **2495/9** (+9), ruff clean. **Closes priority row 17 (N-16).**

---

**Pass 156v Step 2 (Session-1 unification — model + RAG seams):** Extends the `_chat_session_marker` helper shipped in Step 1 to the four other session-state-change surfaces that are wired today: **(1) model load** ([gui_logic.py](enigma_engine/gui/gui_logic.py) `_on_model_loaded`) — `Model: <NAME> (N params, <device>)` divider replaces the previous `Model online: ...` system message. **(2) model unload** (`_unload_model`) — `Model unloaded — no model active` divider replaces `Model unloaded.`. **(3) RAG enable** (`_build_rag_index` success branch via `self.after(0, ...)`) — `Document Q&A enabled — N chunks from M files` divider. **(4) RAG disable** (`_on_rag_toggle(False)`) — `Document Q&A disabled — no corpus active` divider. Same UX contract as Step 1: divider REPLACES the regular system message (not duplicated alongside it), errors stay on `_chat_error`, in-flight progress (`Building document index...`) stays on `_chat_system`. **Out of Step 2 scope:** profile swap and system-prompt edit have no chat-page surface yet (no widget swaps a runtime persona — the "profile" concept in code today is a forge-page training-data profile, not a chat-runtime persona). Suspend/resume also deferred — these are mechanic transitions, not user-facing weight changes; resume calls `_on_model_loaded` so the marker fires implicitly. **Tests +5 (1 behavioural, 4 structural):** model load wires the marker AND drops the old `Model online` system message (duplicate-signal regression gate); unload wires the marker AND drops the old system message; RAG-disable behavioural via `LogicMixin._on_rag_toggle` Harness — fires marker, names the subsystem, leaves no system message, clears engine + local index; RAG-enable success structural via `inspect.getsource(_build_rag_index)` (threaded `self.after(0, ...)` bouncing makes behavioural disproportionate). Suite **2486/9** (+5), ruff clean.

---

**Pass 156w (Continuous-3c — anchor-only periodic idle rehearsal):** Closes the silent-failure mode where the per-batch retrain trigger inside `_train_batch` cannot fire during true quiet periods (no recent chat → `_train_batch` itself never runs → anchors never rehearse — exactly the case anchors are designed for). **Library** [router.py](enigma_engine/router.py) `BackgroundTrainer.__init__` gains opt-in kwarg `anchor_idle_interval_seconds: float | None = None`; 0 / negative inputs collapse to None defensively (avoids a config typo pegging the GPU). New state `_last_anchor_replay_at` initialised to `time.monotonic()` so a fresh trainer never fires immediately. **Helper** `_should_run_anchor_idle_replay()` gates on six conditions: feature opted in, anchor path configured (the WHOLE POINT — no anchors means no work to do), running + not paused, model + optimizer wired, inference not busy (chat takes priority — same contract as the per-batch path), elapsed >= interval. **Hook** in `run()` queue.Empty branch (after partial-batch processing): if helper returns True → log INFO with elapsed seconds → call `_retrain_on_replay()`. **Cooldown reset** at the end of `_retrain_on_replay()`'s success path updates the timestamp regardless of trigger origin, so a regular replay (from `_train_batch`) naturally pushes the next idle replay further out instead of double-firing back-to-back. **Default off** preserves existing behaviour for users who don't opt in. **Tests +9 (8 behavioural, 1 structural):** default-disabled invariant; 0/negative normalisation; helper short-circuits on disabled / no-anchor-path; throttle gate fires False when within window AND True past window (catches the "always True burns GPU on every wakeup" regression and the "always False anchors never fire" regression in the same test); paused defeats all gates; inference-busy blocks; regular `_retrain_on_replay()` resets the timer (catches the back-to-back double-fire regression). One structural gate on `BackgroundTrainer.run()` source via `inspect.getsource` verifies the helper is actually called from the loop (catches the regression where helper exists but is unreachable — exactly the broken state pre-fix). Suite **2481/9** (+9), ruff clean.

**19f (Continuous-3c — anchor-only periodic schedule):** ✅ CLOSED Pass 156w. See entry above.

---

**Pass 156v Step 1 (Session-1 unification — adapter swap markers):** First slice of the Session-1 backlog item. Single source of truth for visually distinct chat-log dividers when runtime state changes. **Helper:** [gui_logic_chat.py](enigma_engine/gui/gui_logic_chat.py) `LogicChatMixin._chat_session_marker(text)` writes a `─── {text} ───` line via `_chat_append("session_marker", ...)`. **Tag:** [gui_pages.py](enigma_engine/gui/gui_pages.py) chat-display tag-config block adds `session_marker` (dim foreground `C_TEXT_DIM`, smaller font, extra `spacing1`/`spacing3` for vertical breathing room) — visually separate from the orange `system_msg` tag so the user can scan and find the seam where weights changed. **Adoption:** all 5 LoRA adapter-swap success paths in [gui_logic.py](enigma_engine/gui/gui_logic.py) now route through the marker — `_set_chat_adapter` apply (manual + auto-restore branches), `_set_chat_adapter` clear, `_set_chat_adapter_stack` apply (manual + auto-restore branches). Errors and load-first hints stay on `_chat_error`/`_chat_system` (unchanged contract). **Scope deferred to Step 2:** model swap, profile swap, system-prompt edit, RAG corpus change. Step 1 keeps the PR reviewable while delivering the highest-value seam (LoRA weight changes, where output regression is most likely). **Tests +5 (4 behavioural, 1 structural):** helper exists with correct delegation + tag name; `session_marker` tag is configured on PagesMixin (structural — Tk render needs a display); apply path emits marker NOT system message and names the adapter; clear path emits marker; stack apply emits marker naming all members. Suite **2472/9** (+5), ruff clean.

**15g (legacy `_lora.pth` migration):** ✅ CLOSED Pass 156v (no migration needed). `Get-ChildItem models/checkpoints` confirms zero `_lora.pth` files exist. Active code path (Pass 156s+) only writes PEFT-directory format. The migration script would have nothing to operate on.

---

**Pass 156u-B (LoRA-1b stacking UI):** GUI surface for the multi-LoRA stack engine shipped in 156u-A. **Architecture:** pure-logic parse helper in [gui_logic.py](enigma_engine/gui/gui_logic.py) (`_parse_lora_stack_inputs`) — module-level free function so adversarial behavioural tests can exercise it without Tk. Empty/whitespace → 1.0 default; non-numeric → error naming the adapter; NaN/Inf → error naming the adapter; **negatives are LEGITIMATE** (subtract this adapter from the merged stack) and pass through. Errors are collected — user sees ALL typos at once, not N round-trips for N typos. **GUI handler** `_on_lora_apply_stack` in [gui_pages.py](enigma_engine/gui/gui_pages.py) dispatches: base not loaded → chat hint, no engine call; empty selection → chat hint, no engine call; parse error → chat-error per row, no engine call (all-or-nothing); 1 selection → `_set_chat_adapter` (skips the `_stack` PEFT indirection on the trivial case); 2+ → `_set_chat_adapter_stack`. **UI extension** in `_build_lora_section_for_card`: per-row `CTkCheckBox` + `themed_numeric_entry(mode="float", width=56)` (default `"1.0"`, allows negatives + scientific notation per the Dia "no sliders" rule), bottom **APPLY STACK** button rendered when ≥2 adapters exist. Stack state lives in a per-card local list captured by the button's lambda — rebuilt on every `_refresh_model_cards`. **Tests +11 (adversarial, mostly behavioural):** parse helper edge cases (empty/whitespace, non-numeric, NaN/Inf, negatives, scientific, multi-row error collection); handler dispatch (load-first guard, empty selection, single-vs-multi routing, parse-error abort, refresh-call gating on success AND no-refresh on abort). One structural gate on `_build_lora_section_for_card` (Tk render needs a display) verifies `themed_numeric_entry` + `mode="float"` + no slider + `_on_lora_apply_stack` wiring. Suite **2467/9** (+11), ruff clean.

---

**Pass 156u-A2 (LoRA-1b stacking stabilization):** Self-audit on Pass 156u-A caught two real issues by re-reading own code with the author's lens. **(1) Restore-path corruption resilience.** [gui_logic.py](enigma_engine/gui/gui_logic.py) `_restore_lora_adapter_for_base` did `item.get(...)` and `float(weight)` inside the entry-build loop — but the existing `(FileNotFoundError, ImportError, RuntimeError, ValueError)` except block only wrapped `engine.apply_adapter_stack(...)`, NOT the parsing. A corrupted `route_assignments.json` (hand-edit, partial write, format drift) where a stack entry is `[1, 2, 3]` instead of `{path, weight}`, or where `weight` is `"abc"`, raised `AttributeError` / `TypeError` / `ValueError` BEFORE the apply call — propagating up through `_on_model_loaded` and aborting model load entirely. Fix: explicit shape-validation pass with a `parse_error` sentinel. Non-dict entries, missing `path`, and non-numeric weights all drop the WHOLE stack key, save the route table, and surface a `LoRA stack is corrupted (<reason>) — using base weights.` chat-system message. User can keep using the model. **(2) Docstring promise without test.** `apply_adapter_stack` `Raises:` clause already promised `ValueError: contains a duplicate path` and the body already implemented it, but no test gated it — Pass 156s2 anti-pattern. Added behavioural test via unbound call. **Tests +2:** `test_apply_adapter_stack_rejects_duplicate_path` (behavioural; same `_FakeSelf` pattern as the other validation tests) and `test_restore_lora_stack_survives_corrupted_entries` (behavioural with two cases — non-dict entry AND non-numeric weight; verifies engine is NOT called, key IS purged, user IS notified). Suite **2456/9** (+2), ruff clean. Stabilizes 156u-A's persistence contract before 156u-B (UI surface) lands.

---

**Pass 156u-A (LoRA-1b stacking — engine + persistence):** Third of four staged passes. Lays the engine + persistence foundation for multi-LoRA weighted stacks. UI surface (numeric weight input per adapter — **no sliders**, per Dia rules) is deferred to a follow-up Pass 156u-B; this pass keeps the surface area small and verifiable end-to-end before adding GUI controls. The runtime entry point and route-key contract are now stable so 156u-B is purely additive.

- **[inference.py](enigma_engine/core/inference.py) `EnigmaEngine.apply_adapter_stack`** — accepts `list[tuple[path, weight]]`. Validates UP FRONT before importing peft or touching `self.model`: empty list → `ValueError`, missing dir → `FileNotFoundError`, missing `adapter_config.json` → `FileNotFoundError`, non-numeric weight → `ValueError`, non-finite (NaN/Inf) weight → `ValueError`, duplicate path → `ValueError`. Wraps base on first use via `PeftModel.from_pretrained` using the first stack member as bootstrap; loads remaining members; rebuilds a `_stack` adapter via `add_weighted_adapter(combination_type="linear")` so weight changes are deterministic re-stacks. Drops any prior `_stack` via `delete_adapter` before rebuild (PEFT >=0.6 requirement, raises with explicit upgrade message if missing). Clears KV cache because weights changed. Logs the merged stack at INFO with weights formatted to 2 decimals.

- **[gui_logic.py](enigma_engine/gui/gui_logic.py) `_set_chat_adapter_stack` + `_adapter_stack_route_key` + extended `_set_chat_adapter` + extended `_restore_lora_adapter_for_base`** — companion GUI-layer entry point for the multi-adapter path. Persists to `chat_adapter_stack:<base_stem>` as `[{"path": str, "weight": float}, ...]` (plain JSON-serialisable dicts, not tuples). **Mutual exclusion:** writing the stack key clears the single-adapter `chat_adapter:<stem>` key, and writing the single key clears the stack key — single and stack are not allowed to coexist for the same base, otherwise restore order would be ambiguous. **Stack-first restore precedence:** `_restore_lora_adapter_for_base` checks the stack key BEFORE the single key on every model load; stacks are the more recent + more specific intent. Missing stack member on disk drops the WHOLE stack rather than apply a partial merge (loud chat-system message). Engine-too-old guard (`hasattr(engine, "apply_adapter_stack")`) drops the orphan key and surfaces a one-time chat notice instead of looping on retry.

- **Tests (+7):** (1) `test_engine_exposes_apply_adapter_stack` — structural gate on `add_weighted_adapter` + `set_adapter` + `clear_kv_cache` presence in the body. (2) `test_apply_adapter_stack_rejects_empty_list` — behavioural via unbound call with `_FakeSelf`; empty list raises BEFORE peft import. (3) `test_apply_adapter_stack_rejects_non_finite_weight` — behavioural; NaN AND Inf both raise via parametrised match. (4) `test_apply_adapter_stack_rejects_missing_adapter_dir` — behavioural; missing dir raises `FileNotFoundError` immediately. (5) `test_gui_logic_set_chat_adapter_stack_persists_to_stack_key` — structural; gates the stack-key write AND the mutual-exclusion `pop` on the single key (catches the regression where someone forgets to clear the orphan single-adapter entry). (6) `test_gui_logic_set_chat_adapter_single_clears_stack_key` — structural; the inverse direction, gates that single-adapter writes also clear the stack key. (7) `test_gui_logic_restore_prefers_stack_over_single` — structural; gates that restore checks the stack key AND calls `apply_adapter_stack`. Suite **2454/9** (+7), ruff clean.

**Open in following passes (LoRA-1b roadmap):**
- **Pass 156u-B** — Stacking UI: per-card multi-select + numeric weight input per selected adapter (no sliders), Apply Stack button, weight-edit-without-reload (re-call `apply_adapter_stack` with new weights triggers KV-cache clear via the engine method already shipped in 156u-A).
- **Pass 156v** — **Session-1** generalization: unified `invalidate_session_state(reason, scope)` helper applied to model swap / profile swap / system-prompt edit / RAG corpus change with a shared branch-marker (`─── switched to <name> ───`) UX. The model-swap case currently clears chat entirely, which is wrong for the same reason adapter swap was wrong (loses conversation context). Adapter-swap UX from Pass 156t becomes the reference implementation; this pass lifts it out into a single helper.
- **GUI profile selector** — currently profiles only load via the API server. The GUI has `profiles_data` listed but no runtime selector. Logical companion to Session-1; no specific pass assigned yet.

---

**Pass 156t (LoRA-1b UX surfaces):** Second of four staged passes. Builds on Pass 156s/156s2 foundations to give the user actual surfaces — MODELS-page list with Apply/Clear buttons, profile JSON `adapter` field with apply-or-clear-on-load semantics, and a one-shot legacy `.pth` migration script. Branch-marker visual flourish deferred to Pass 156v (Session-1) where it generalises across all session-state changes anyway.

- **[gui_pages.py](enigma_engine/gui/gui_pages.py) `_build_lora_section_for_card` / `_on_lora_apply` / `_on_lora_clear` / `_refresh_model_cards`** — per-card LoRA section under the MODELS page. `scan_lora_adapters(model["path"])` filters by base-stem so a math-base adapter cannot reach a coding-base card. Section renders nothing when zero compatible adapters exist (no empty-list noise). Header shows count + active-indicator + Clear button when an adapter is active for this base. Each adapter row shows name + rank + alpha + Apply button (hidden for the active adapter — the header Clear is the only mutation surface there). **Load-first guard:** clicking Apply or Clear on a card whose base is NOT currently loaded surfaces a chat hint (`"Load 'X' first, then apply 'Y'"`) instead of attempting a cross-base apply that PEFT would reject on shape mismatch. `_refresh_model_cards()` rebuilds the scroll frame in-place after every Apply/Clear so the active highlight stays in sync without a full page rebuild.

- **[ai_profile.py](enigma_engine/core/ai_profile.py) `AIProfile.adapter: Optional[str]` + `apply_profile_to_engine`** — profiles can pin a per-base adapter via the new optional `adapter` field. Critical semantic: a profile with **no** adapter field (or empty string / None) calls `engine.clear_adapter()` on apply, NOT a silent no-op. This is profile-boundary discipline — switching to a profile that doesn't specify an adapter must NOT silently inherit the previous profile's adapter. `from_dict` round-trip preserves the field; old profile JSONs without it default to `None` (backward compatible). Errors during apply are logged at WARNING and don't abort profile load — base weights remain usable.

- **[migrate_legacy_lora.py](migrate_legacy_lora.py)** — one-shot script (project root). Conservative scope: moves `models/lora_adapters/*.pth` (loose files in the LoRA dir) and `models/checkpoints/*_lora.pth` (named-pattern matches only) into `models/checkpoints/legacy_lora_pth/`. **Does NOT touch `models/`** (full base models live there) or generic `.pth` files in `models/checkpoints/` (legitimate training-state snapshots from Pass 122 protected checkpoints). Writes `NOTICE.txt` explaining the format change and what the user can do (retrain or discard). Idempotent: re-running with `--apply` after migration is a no-op. Filename collisions append a numeric suffix (`old.pth` → `old_1.pth`) instead of clobbering. Default mode is dry-run; `--apply` flag commits.

- **Tests (+6):** (1) `test_models_page_renders_lora_section_per_card` — structural; gates `_build_lora_section_for_card` invocation in `_populate_model_cards` AND that the section calls `scan_lora_adapters` with `model["path"]` (catches the regression where someone calls the scanner with no argument and lists every adapter on disk). (2) `test_lora_apply_guards_against_inactive_base` — structural; gates the load-first check in `_on_lora_apply` AND that it delegates to `_set_chat_adapter` (catches the regression where someone wires Apply directly to `engine.apply_adapter`, skipping persistence). (3) `test_profile_adapter_field_drives_engine_apply` — behavioural with FakeEngine; **adversarial three-case test**: pinned adapter → apply called, no adapter field → clear called, empty-string adapter → clear called. The clear cases are critical — they catch the silent-inheritance bug. (4) `test_profile_adapter_field_round_trips_through_dict` — behavioural; round-trip via from_dict/to_dict preserves the field, old JSONs without it default to None. (5) `test_legacy_lora_migration_moves_pth_files` — behavioural; full dry-run + apply + idempotence flow. **Adversarial:** an "innocent" `training_state.pth` in `checkpoints/` must NOT be moved (would lose training state); only `*_lora.pth` and loose `lora_adapters/*.pth` are touched. (6) `test_legacy_lora_migration_handles_filename_collision` — adversarial; pre-existing quarantine file with same basename forces numeric-suffix rename instead of clobber. Suite **2447/9** (+6), ruff clean.

**Open in following passes (LoRA-1b roadmap):**
- **Pass 156u** — Stacking: PEFT `add_weighted_adapter` for multi-adapter stacks, weight-slider UI per adapter, KV invalidation on weight change.
- **Pass 156v** — **Session-1** generalization: unified `invalidate_session_state(reason, scope)` helper applied to model swap / profile swap / system-prompt edit / RAG corpus change with a shared branch-marker (`─── switched to <name> ───`) UX. The model-swap case currently clears chat entirely, which is wrong for the same reason adapter swap was wrong (loses conversation context). Adapter-swap UX from Pass 156t becomes the reference implementation; this pass lifts it out into a single helper.
- **GUI profile selector** — currently profiles only load via the API server. The GUI has `profiles_data` listed but no runtime selector. Logical companion to Session-1; no specific pass assigned yet.

---

**Pass 156s2 (LoRA-1b foundation audit-fixes):** Self-audit on Pass 156s caught two real bugs in code shipped 30 minutes earlier — exactly the trigger the AA `audit-immediately` rule describes. Both found by re-reading own code with the author's lens (would I write it this way? does the doc claim match the body? does the test prove correctness or just presence?).

- **[inference.py](enigma_engine/core/inference.py) `EnigmaEngine.clear_adapter`** — silent no-op fixed. The shipped fallback chain `if hasattr(model, "disable_adapters"): model.disable_adapters() else: disable_fn()` (where `disable_fn = getattr(model, "disable_adapter", None)`) looked defensive but was broken: PEFT's `disable_adapter` (singular) is a `@contextmanager`-decorated method; calling it bare returns the CM and immediately discards it without entering — adapter stays active, GUI says "cleared", chat keeps using LoRA weights. The two names are NOT a primary/fallback pair. Rewritten to require `disable_adapters` (plural, PEFT >=0.6.0) and raise `RuntimeError` on the missing branch instead of falling back to the wrong-semantic sibling. Stale planning comment about "set_adapter('') fallback" deleted (described an unimplemented branch).

- **[inference.py](enigma_engine/core/inference.py) `EnigmaEngine.apply_adapter`** — docstring lie removed. The `Raises:` clause promised `RuntimeError: The adapter's recorded base does not match the loaded model.` but the body never parsed `base_model_name_or_path` and never compared anything. Pass 156i2 / 156k-audit anti-pattern: doc claims more than code delivers. Replaced with an honest paragraph naming the upstream defense (`scan_lora_adapters` filters by base-model stem) and noting that PEFT itself raises on shape mismatch if a wrong adapter ever bypasses the scanner. No new check added because the engine doesn't currently track the loaded base path; revisit if direct API callers prove to be a real attack surface.

- **Tests (+0 files, +2 assertions inside `test_engine_exposes_apply_and_clear_adapter`):** structural gate that `disable_adapters` literal appears in `clear_adapter` source AND that bare `disable_adapter(` (singular, with the plural variant masked out) does NOT appear — catches a regression that re-introduces the broken fallback chain. Second gate: docstring of `apply_adapter` does NOT contain `"RuntimeError: The adapter's recorded base"` — catches a future revert of the audit fix. Suite **2441/9** unchanged (no new test methods, just sharper assertions inside an existing one), ruff clean.

**New principles recorded in [AA code maker.md](AA%20code%20maker.md):**
1. Singular vs plural API names (`disable_adapter` / `disable_adapters`, `add_x` / `add_xs`) that look like a fallback chain are usually two different things — read both before assuming primary/fallback.
2. Docstring `Raises:` clauses must enumerate only exceptions the code actually raises; pair with a structural test that asserts the exception literal appears in the body.
3. Stale planning comments in shipped code are silent lies — when you cut a planned branch, delete the comment that promised it.

---

**Pass 156s (LoRA-1b foundation):** First of four staged passes that wire LoRA adapters end-to-end across save → scan → load → chat → MODELS UI → profiles → stacking → unified-invalidation. This pass lays the foundation; Pass 156t-v handle the UX surfaces. Design decisions locked with the user: PEFT-directory-only format (safetensors, self-describing), hot-swap apply via PEFT, per-base persistence (`route_assignments["chat_adapter:<stem>"]`).

- **[lora_utils.py](enigma_engine/core/lora_utils.py) `LoraTrainer.save_adapter`** — replaced. Always writes a PEFT directory (`adapter_config.json` + `adapter_model.safetensors`) via `model.save_pretrained(save_dir)`. Deleted the previous manual-fallback `.pth` branch entirely; that branch was unreachable in normal flow (`__init__` always wraps with `create_lora_model` / `create_qlora_model` which produce PEFT models) and only ever produced metadata-less files that the chat engine could not safely apply. The defensive `RuntimeError` guards against future regressions where someone hands `LoraTrainer` a non-PEFT model.

- **[scanners.py](enigma_engine/gui/scanners.py) `scan_lora_adapters(base_model_path=None)`** — new helper. Walks `models/checkpoints/` and `models/lora_adapters/` one level deep, picks up directories containing `adapter_config.json`, parses metadata into `{name, path, base, rank, alpha, target_modules, size_kb}`. When `base_model_path` is provided, filters by stem-matching `base_model_name_or_path` from the config — prevents listing a coding-base adapter when a math base is loaded (would have wrong target_modules / weight shapes). Skips malformed configs at WARNING. Pass `None` for the audit/migration view; chat dropdown should always pass the active base.

- **[inference.py](enigma_engine/core/inference.py) `EnigmaEngine.apply_adapter(path)` / `clear_adapter()` / `active_adapter`** — runtime adapter API. First `apply_adapter` call wraps `self.model` with `PeftModel.from_pretrained(adapter_path)`; subsequent calls reuse the wrapper and route through `model.load_adapter(...) + model.set_adapter(...)`. Critical: KV cache is cleared on every swap because the forward-pass weights changed. `clear_adapter()` calls `model.disable_adapters()` (PEFT >=0.10 imperative form) when available; falls back gracefully on older versions. `active_adapter: str | None` exposes the current adapter name to the GUI for the future "+adapter:name" header badge (Pass 156t).

- **[gui_logic.py](enigma_engine/gui/gui_logic.py) `_restore_lora_adapter_for_base` / `_set_chat_adapter` / `_adapter_route_key`** — persistence + auto-restore. After a successful `_on_model_loaded`, reads `route_assignments["chat_adapter:<stem>"]`; if a saved adapter directory still exists on disk, applies it and surfaces "LoRA adapter active: <name>" in chat. If the saved path was deleted, the orphan entry is purged (`save_route_assignments`) and chat shows a one-time "no longer on disk — using base weights" notice. `_set_chat_adapter` is the single entry point for runtime adapter changes (Apply/Clear from MODELS page in Pass 156t, profile-driven auto-apply in Pass 156t). Per-base scoping (`chat_adapter:<base_stem>` route key) prevents a coding adapter from auto-applying onto a math base after a model swap.

- **Tests (+6):** `test_scan_lora_adapters_finds_peft_directory`, `test_scan_lora_adapters_filters_by_base_model_stem`, `test_scan_lora_adapters_skips_directory_without_config` — three behavioural tests on the scanner, including the adversarial filter case (foreign-base adapter must not match). `test_engine_exposes_apply_and_clear_adapter` — structural double-gate: methods present + `PeftModel.from_pretrained` literal AND `clear_kv_cache` literal in `apply_adapter` source. Single-literal would pass even if someone removed the wrapper but kept the cache-clear (silent no-op on the chat path), so two literals catch both regressions. `test_save_adapter_writes_peft_directory_only` — structural; asserts `save_pretrained` present AND the deleted manual-fallback markers (`param.requires_grad`, `atomic_torch_save`) are absent. `test_gui_logic_wires_adapter_auto_restore` — structural; gates `_restore_lora_adapter_for_base` literal in `_on_model_loaded` source. Suite **2441/9** (+6), ruff clean.

**Open in following passes (LoRA-1b roadmap):**
- **Pass 156t** — UX surfaces: MODELS-page list with Apply/Clear buttons, profile JSON optional `"adapter"` field with auto-apply on profile load, branch marker (`─── switched to <name> ───`) in chat on every adapter swap, lazy KV reprefill on next message.
- **Pass 156u** — Stacking: PEFT `add_weighted_adapter` for multi-adapter stacks, weight-slider UI per adapter, KV invalidation on weight change.
- **Pass 156v** — **Session-1** generalization: unified `invalidate_session_state(reason, scope)` helper applied to model swap / profile swap / system-prompt edit / RAG corpus change with the same branch-marker UX. Adapter-swap UX from Pass 156t becomes the reference implementation; this pass just lifts the pattern out and wires it through the other invalidation points (model swap currently clears chat entirely, which is wrong for the same reason adapter swap was wrong).

**Deferred — legacy `_lora.pth` migration:** Earlier passes' `LoraTrainer.save_adapter` manual-fallback wrote `.pth` files containing only `param.requires_grad` weights — legitimate LoRA weights but in a custom format without rank/alpha/target_modules metadata, so PEFT can't load them. Files are not full base-model snapshots (earlier audit was wrong). One-time migration to `models/checkpoints/legacy_lora_pth/` with a `NOTICE.txt` ("retrain to use the new PEFT directory format") will land alongside Pass 156t. Low risk because the active code path no longer produces these files after this pass.

**Pass 156r (Code-6b — Stage-2 `unfreeze_text_layers` GUI knob):** Closes the small wiring gap flagged when Code-6 was marked done. The Image foundation mode previously hardcoded `unfreeze_text_layers=0` (LLaVA Stage-1, projection-only) at the GUI layer — `Trainer.train_vision()` already accepted the kwarg ([training.py L4909](enigma_engine/core/training.py#L4909)) but the user couldn't reach Stage-2 (unfreeze last N text transformer layers on top of the projection) without hand-editing code. Three coordinated edits:
- **[gui_pages_forge.py](enigma_engine/gui/gui_pages_forge.py)** — new numeric input `forge_vision_unfreeze_var` (default `"0"`) inside the Image section, immediately under the vision-encoder-size dropdown. Tooltip explains the LLaVA Stage-1 vs Stage-2 distinction, calls out the compute and checkpoint-size cost of unfreezing.
- **[gui_forge_training.py](enigma_engine/gui/gui_forge_training.py) `_start_vision_training`** — reads the var with the same try/except + range-clamp pattern used by `_start_lora_training`'s rank parser. Negative → 0 with warning log. >64 → 64 with warning log. Bad input → 0 with warning log. Logged in `summary_fields` as either `"projection only (Stage-1)"` (when 0) or `"N text layers"` (when >0). Forwarded to `trainer.train_vision(unfreeze_text_layers=unfreeze_text_layers, ...)`.
- **Tests (+2):** `test_forge_image_mode_exposes_unfreeze_text_layers` (structural — `forge_vision_unfreeze_var` literal in `gui_pages_forge` source) and `test_forge_vision_training_forwards_unfreeze_to_trainer` (structural — both `forge_vision_unfreeze_var` AND `unfreeze_text_layers=unfreeze_text_layers` literals in `_start_vision_training` source). Two-literal gate per the Pass 156k-audit label-tracking rule: a single-literal test would pass even if someone read the var into a local but dropped it from the trainer call — silently reverting Stage-2 to default 0 for everyone. Suite **2435/9** (+2), ruff clean. **Closes Code-6b.**

**LoRA-1b (#15c) and D-12 still open — design pass needed before code:**
- **LoRA-1b structural blocker uncovered Pass 156r research:** Both `apply_lora()` and `merge_lora_weights()` in [core/lora_utils.py L609](enigma_engine/core/lora_utils.py#L609) filter `if key in state_dict` — they only do anything if the base model **already has LoRA layers attached** (`*.lora_A.weight` / `*.lora_B.weight` parameters present in its `state_dict`). The chat-side `EnigmaEngine` ([core/inference.py L90](enigma_engine/core/inference.py#L90)) loads a vanilla base model; calling `apply_lora()` on it is a silent no-op — no keys match. Making LoRA-1b real requires (a) detecting the adapter at load time, (b) wrapping the engine's model with `create_lora_model(LoraConfig(...))` BEFORE apply, (c) sourcing the LoraConfig (rank, alpha, target_modules) from sidecar metadata because the `*_lora.pth` state-dict alone doesn't carry it. PEFT directories DO carry `adapter_config.json` but the manual-fallback `LoraTrainer.save_adapter` path saves a full base-model snapshot under the `_lora.pth` filename — **wrong format under that name** (separate cleanup row needed). Cannot ship a "minimum viable" without a structural decision on adapter format and engine wrapping.
- **D-12 NoPE (every-4th-layer no-RoPE):** Architecture change. Per AA Devil's-Advocate rule won't ship blind — needs design pass on layer pattern, backward-compat config field, KV-cache handling for mixed layers, position-extrapolation tests.

**Pass 156q (D-11c-DPO + Code-6 doc closure):** Two-part pass.
- **D-11c-DPO** — Mode-aware default for the shared FORGE `train_data_var` picker. The same picker is used by SFT modes (Basic / LoRA) and preference-pair modes (RLHF / Self-Play / GRPO / ReMax / SimPO / ORPO / APO); before this pass it always defaulted to `data/finetune/combined_finetune.txt` regardless of mode, so switching to APO surfaced an irrelevant SFT corpus instead of `data/dpo/combined.jsonl`. Three coordinated edits: (1) `scanners.py` — new `_pick_default_train_data_for_mode(files, mode)` helper plus a `_PREFERENCE_MODES` frozenset routing the call to either `_pick_default_dpo_data_file` or `_pick_default_training_file`. (2) `gui_pages_forge.py` — init now stores the initial default as `self._train_data_smart_default` so the mode-change handler can detect whether the user has customised the picker. (3) `gui_forge.py` — `_on_training_mode_changed` swaps the default only when `train_data_var` still equals the previously applied smart default (or is empty); a user-chosen file is left untouched. `_browse_training_data` and `_on_data_selected` clear the tracker (`self._train_data_smart_default = None`) the moment the user picks a path, so subsequent mode changes never silently overwrite their choice. Three new tests: behavioural for the mode-router (every preference mode → DPO file, every SFT mode → SFT file), structural for the swap logic (`_pick_default_train_data_for_mode` and `_train_data_smart_default` literally referenced in `_on_training_mode_changed`), structural for the user-customisation clear (`_train_data_smart_default = None` in both browse + quick-select handlers). Suite **2433/9** (+3 net), ruff clean. **Closes D-11c-DPO.**
- **Code-6 (priority row #9)** — Closed in docs only; no code changes. The work landed across Pass 151 (Vision-1b 2-layer MLP+GELU projection), Pass 156b (V-8 inference-time vision-encoder load), Pass 156c (V-5 `collect_vision_data.py` LLaVA-Pretrain fetcher), Pass 156d (V-4 OOM/crash heuristics, V-7 abort-summary). The Image foundation mode already trains the projection with the LLaVA Stage-1 frozen-everything recipe (`freeze_backbone=True`, `unfreeze_text_layers=0`). Stage 2 (unfreeze last N text layers) knob exists in `Trainer.train_vision()` but is not yet exposed in the GUI — opened as **Code-6b** under the priority table.

**Pass 156p (LoRA-1 — explicit LoRA mode card):** Audit-then-ship. Discovery: `LoraTrainer` was already wired into FORGE Basic mode via `_start_lora_training()` (auto-triggered for >7B models), with rank/alpha widgets at [gui_pages_forge.py L966-988](enigma_engine/gui/gui_pages_forge.py) and adapter saving to `models/checkpoints/{name}_lora.pth`. The gap was a **user-controllable opt-in** — no way to force LoRA on smaller models for fast iteration. Followed the Pass 156k D-9b APO radio-card pattern: added `("LoRA", "Force low-rank adapter training on any model size. 10-30 MB adapter")` to `foundation_modes` at [gui_pages_forge.py](enigma_engine/gui/gui_pages_forge.py); dispatcher branch `elif mode_name == "LoRA": self._start_lora_training()` in [gui_forge.py](enigma_engine/gui/gui_forge.py) (skips the >7B auto-detection); added LoRA entry to `_MODE_DISPLAY_TO_KEY` (9 modes now) and `_TRAINING_MODE_DESCRIPTIONS` for status/log labels; visibility set inherits the Basic-default `{basic, evolutionary, preserve}` since the same data picker applies. Two new structural tests `test_forge_lora_mode_card_present` and `test_forge_lora_dispatcher_calls_lora_training` gate the end-to-end wiring — catches regression where the card is removed but dispatcher still references it (or vice versa). Updated three pre-existing count tests (`test_descriptions_cover_all_modes`, `test_display_name_mapping_covers_all_modes`, `test_reverse_mapping_covers_all_keys`) from 8 → 9. **Not done this pass:** runtime adapter swap in inference (load `*_lora.pth` at chat time, hot-swap, multi-adapter UI) — separate larger row, would need design discussion. Suite **2430/9** (+2 net), ruff clean. **Closes LoRA-1 training-side wiring.**

**Pass 156o (Continuous-3b — anchor file GUI):** Closes the UX follow-up from Continuous-3 (Pass 156i7). Pass 156i7 wired the anchor JSONL file end-to-end through `ModRouter` → `BackgroundTrainer`, but users still had to edit `data/anchor_examples.jsonl` by hand and had no visibility into row count or override path. Three-part fix:
- **`enigma_engine/gui/scanners.py`** new helper `_resolve_anchor_path(saved)`. Three-branch resolution centralised so desktop and config page agree: non-empty saved → return as-is (even if missing — status label flags it loudly); empty/None → return repo default when it exists, else None.
- **`enigma_engine/gui/gui_pages_config.py`** new anchor widget block in the TRAINING section (right after "Learn while chatting"). Shows path + row count or "(none — replay rehearses recent chat only)" or "(file missing)". BROWSE button (filedialog → JSONL) and USE DEFAULT button (clears override) both persist via `gui_settings.json` key `anchor_data_path` and refresh the status label live. Five new methods on `ConfigPageMixin`: `_format_anchor_status`, `_save_anchor_data_path`, `_refresh_anchor_status`, `_browse_anchor_file`, `_reset_anchor_file`.
- **`enigma_engine/gui/desktop.py`** boot path now reads `anchor_data_path` from `gui_settings.json` via new `_read_gui_str_setting()` and forwards the resolved path into `ModRouter(anchor_data_path=...)`. Restart-to-apply (status bar tells user).
- **Tests (+6):** `test_resolve_anchor_path_user_override_returned_as_is`, `test_resolve_anchor_path_empty_returns_default_when_present`, `test_resolve_anchor_path_empty_returns_none_when_missing`, `test_resolve_anchor_path_none_arg_treated_as_empty` cover the three-branch helper directly. `test_config_page_wires_anchor_widget` and `test_desktop_forwards_anchor_path_to_router` are structural tests that gate the GUI/desktop wiring stays connected (catches regression where someone deletes the widget block or drops the kwarg). Suite **2428/9** (+6), ruff clean. **Closes Continuous-3b.**

**Pass 156n (D-11c wiring follow-up — pretrain picker):** Closes the wiring gap left open by Pass 156l. Pass 156l shipped helpers `_pick_default_pretrain_file` / `_pick_default_dpo_data_file` but only the training picker was actually wired — the pretrain picker still initialised `self.pretrain_data_var = ctk.StringVar(value="")` at [gui_pages_forge.py L345](enigma_engine/gui/gui_pages_forge.py#L345), so users had to manually navigate to `data/pretrain/combined.txt` even when collected. Fix: replaced the hardcoded `value=""` with `value=_pick_default_pretrain_file(self.training_files)`. Now when `data/pretrain/combined.txt` (or `combined_pretrain.txt`) exists in the scanned files list, the pretrain entry pre-fills with the right path. Empty list / no match still falls through to `""` (legacy behaviour preserved). New structural test `test_forge_pretrain_data_var_uses_smart_default` in [tests/test_gui.py::TestScanners](tests/test_gui.py): asserts the helper name appears in the page source AND the empty-default literal does NOT — catches both forward direction (helper missing) and regression direction (someone reverts to empty). **DPO picker NOT wired this pass:** DPO mode shares `train_data_var` with SFT/distill in the FORGE Basic page (single picker, multiple consumers), so a smart default that depends on which mode is active would need a UX rework (mode-change-resets-picker). Logged as **D-11c-DPO** open question — not a small wiring win, deferred. Suite **2422/9** (+1), ruff clean. **Closes D-11c wiring follow-up.**

**Pass 156m (Data-5b + R4-1..R4-7 close-out):** Two parallel small wins.

- **Data-5b** — [enigma_engine/core/training.py L946](enigma_engine/core/training.py#L946) `minhash_dedup(threshold=0.8)` default flipped to `0.75` to match the FineWeb tech report (arxiv:2406.17557 §3.4: "targeting documents that are at least 75% similar") which is also DCLM/SmolLM3 standard practice. Caller in [enigma_engine/core/training.py L2545](enigma_engine/core/training.py#L2545) `Trainer.train()` pre-train preprocess block dropped the explicit `threshold=0.8` kwarg — was overriding the function default with the older value, defeating the point of the helper change. Comment updated to cite FineWeb. New behavioural test [tests/test_research_upgrades.py::TestMinHashDedup::test_default_threshold_matches_fineweb_standard](tests/test_research_upgrades.py) pins the default via `inspect.signature(...).parameters["threshold"].default == 0.75` so any future drift fails loud. Existing 7 MinHash tests untouched (they pass `threshold=` explicitly). Net effect on dedup output: slightly more aggressive deduplication on near-duplicates (75% Jaccard threshold catches more of them than 80%) — matches the cited research's empirical sweet spot. **Closes Data-5b** (was P2 backlog row).

- **R4-1..R4-7** — GUI Modernization plan close-out (doc-only, no code):
  - **R4-1:** Phase 1 bake-off page changed from FORGE to **CONFIG** (mid-complexity) — protects the time-box from dying on FORGE's 12-modes / 6 collapsible-tool-sections / paned layout instead of on actual stack fit. FORGE becomes an in-budget stretch goal that scores bonus on "Theming & layout headroom." Updated Phase 1 prologue + GUI-ARCH-1a + GUI-ARCH-1b.
  - **R4-2:** Idle RAM and Install-size rubric rows rewritten from absolute targets to primary/stretch pattern matching cold-start and page-switch (`primary: ≤ baseline × 1.5; stretch: ≤ N absolute`). Same bug class as round 3 — "absolute target with no baseline measurement" — now fixed across all four rubric rows.
  - **R4-3:** Gate G3 (UI-doesn't-freeze) re-scoped from absolute 100 ms to `≤ baseline on the same workload (primary), < 100 ms (stretch)`. The pure 100 ms gate was either vacuous (CustomTkinter already passes it idle) or eliminated the incumbent under model load — neither was the intent.
  - **R4-4:** Phase 4 per-page acceptance gained an explicit rollback-test bullet: flip `data/gui_settings.json` flag off, relaunch, verify legacy page renders + its own tests pass, flip back on, re-verify. "Untested rollback = no rollback."
  - **R4-5:** Decision ladder gained rule 6 — wall-clock abort: if Phase 1 exceeds 15 working days, stop and default to Option C regardless of rubric state. Prevents drift into multi-month bake-off.
  - **R4-6:** Phase 3a coexistence policy clarified — `Launch Enigma.bat` reads a `gui_mode` key from `data/gui_settings.json` and dispatches `legacy` (default during Phase 3) or `next`. The `--gui` / `--gui-next` CLI flags still exist on `run.py` for direct use; the .bat just picks one. Default flips to `next` after Phase 4 final cutover.
  - **R4-7:** Round-4 open-items table marked all 7 rows as **DONE** with this pass stamp. Plan is now execution-ready; Phase 0 is the real next action when the user says "go."

Suite **2421/9** (+1 Data-5b test), ruff clean. **Closes Data-5b + R4-1..R4-7.**

**Pass 156l (D-11c + D-11d — two small UX wins):** Closes two P3 follow-up rows opened by Pass 156j-audit. **D-11d:** [collect_finetuning_data.py L530-L545](collect_finetuning_data.py#L530) `combine_all` now emits a WARNING instead of INFO when `text_count == 0 and len(all_pairs) > 0` — a real silent-failure mode where every fetcher row had empty prompt/completion and the SFT path would silently train on a 0-byte file. Mirrors the file-present-zero-yield WARNING pattern from Pass 156i6 anchor loader. New test `test_warns_when_all_pairs_yield_empty_text` writes 2 malformed rows, captures logger output at WARNING level, asserts "text file is 0 bytes" appears. **D-11c:** generalised the Pass 156i9 single-tail picker into a list-of-preferences helper `_pick_first_match(files, preferred_tails)` in [scanners.py L482](enigma_engine/gui/scanners.py#L482). Existing `_pick_default_training_file` is now a thin wrapper around it (bit-identical behaviour, all 4 D-11b tests still green). Two new sibling helpers shipped: `_pick_default_dpo_data_file` (prefers `data/dpo/combined.jsonl`, falls back to `data/finetune/dpo_pairs.jsonl`) and `_pick_default_pretrain_file` (prefers `data/pretrain/combined.txt` and `data/pretrain/combined_pretrain.txt`). 4 new tests including **adversarial preference-order test** (`test_pick_first_match_first_tail_wins_over_later_tails`) which proves the helper iterates preferences-outer/files-inner so first-tail-wins even when the second-tail's matching file appears earlier in `files` — a naive `for f in files` outer loop would silently return the wrong file. **Wiring into actual GUI pickers (DPO + pretrain) is intentionally NOT in this pass** — the helpers are ready but the concrete file picker call sites for DPO and pretrain pages are a separate small wiring pass once a user confirms the preferred file paths match their workflow. Suite **2420/9** (+5: 1 D-11d + 4 D-11c), ruff clean. **Closes D-11c + D-11d.**

**Pass 156k-audit (self-audit on Pass 156k):** Author's-lens 5-question review of the FORGE APO radio-card ship caught **three real claim-vs-code drifts** plus a wimpy audit test. SUGGESTIONS Pass 156k entry claimed "logs are accurate per mode" via `algo_label` parametrization, but Q4 logic-eye re-read of [gui_forge_training.py](enigma_engine/gui/gui_forge_training.py) found three hardcoded `"DPO"` literals leaking the wrong-mode label to APO users: (1) L574 error message `"DPO requires a JSONL file with:"`, (2) L785 stop-log `"\n--- DPO TRAINING STOPPED ---"`, (3) L787 error-log `f"[ERROR] DPO training failed: {e}"`. Fix: all three substituted with `f"... {algo_label} ..."` so APO mode logs "APO-ZERO" everywhere. **Q5 test gap (round 1, defective):** initial audit test counted `'"DPO"'` substring occurrences — but that pattern only matches double-quote-bracketed `"DPO"` and silently misses bareword `DPO` inside larger f-strings (`f"--- DPO TRAINING ..."` contains DPO but not `"DPO"`). The wimpy test passed even though L785/L787 still had hardcoded labels — classic claim-vs-test failure mode. **Q5 fix (round 2, real):** rewrote `test_start_dpo_training_user_facing_strings_use_algo_label` with `re.search(r'\bDPO\b', ln)` word-boundary regex on stripped source body (docstring + comment-only lines removed) plus an explicit allowlist for the two legitimate ternary-definition lines (`algo_label = "DPO" if ...` and `"DPO Training" if loss_type ==`). Pre-fix the rewritten test catches all three bugs; post-fix `offending == []`. Suite **2420/9** (test rewrite, no net add), ruff clean. **Closes the audit gap on D-9b.** **New learned principle:** label-tracking structural tests must use word-boundary regex (`\bLABEL\b`), not double-quoted-substring search — substring search produces false-positive PASSes when bareword token appears inside larger f-strings without surrounding quotes.

**Pass 156k (D-9b — FORGE radio card + dispatch for APO-zero alignment):** Closes the GUI follow-up to Pass 156j. APO-zero ships as a fifth alignment-row radio card alongside GRPO/ReMax/SimPO/ORPO. Wiring trace: [gui_pages_forge.py L203](enigma_engine/gui/gui_pages_forge.py#L203) `alignment_modes` adds `("APO", "Anchored Preference Optimization (zero). Both sides anchored to reference independently")` → [gui_forge.py L1566](enigma_engine/gui/gui_forge.py#L1566) visibility branch extended `("GRPO", "ReMax", "SimPO", "ORPO", "APO")` (basic section only) → [gui_forge.py L1641](enigma_engine/gui/gui_forge.py#L1641) dispatcher gains `elif mode_name == "APO": self._start_apo_training()` → new [gui_forge_training.py `_start_apo_training`](enigma_engine/gui/gui_forge_training.py) thin wrapper delegates to refactored `_start_dpo_training(loss_type="apo_zero")` which forwards the kwarg to `trainer.train_dpo(pref_data, beta=beta_val, loss_type=loss_type)`. Status bar + log lines + `_save_training_run` label parametrized via `algo_label` ("DPO" / "APO-ZERO") so logs are accurate per mode. **Refactor preserves DPO behaviour bit-identically** — `_start_dpo_training()` with no args defaults `loss_type="dpo"`. 5 new structural tests in [tests/test_gui.py::TestForgeAPOAlignmentMode](tests/test_gui.py): radio-card-presence, visibility-branch, dispatcher-case, `_start_apo_training`-references-`apo_zero`, `_start_dpo_training`-forwards-`loss_type=`-to-trainer (catches regression where kwarg is accepted at GUI but dropped before trainer call). End-to-end behavioural proof that GUI → trainer → loss actually delivers APO math comes from Pass 156j's `test_train_dpo_apo_zero_actually_routes_to_apo_loss` (sentinel-mock dispatch test). Suite **2414/9** (+5), ruff clean. **Closes D-9b.**

**Pass 156j-audit (read-only author's-lens review of last 4 ships — 156i7, 156i8, 156i9, 156j):** User asked "lets do an audit." Applied the 5-question lens (would-I-write-it / connected-to / connections-that-should-exist / claim-vs-doc / claim-vs-test) to each pass. **One real bug found, two backlog rows opened, three passes confirmed clean.** No code changes in this pass — findings only.

- **Pass 156j (APO-zero loss + train_dpo dispatch) — CLEAN.**
  - Q1 math: `_apo_zero_loss` at [training.py L3884-3887](enigma_engine/core/training.py#L3884) computes `sigmoid(-β·chosen_logratio) + sigmoid(β·rejected_logratio)`. Verified against TRL's `apo_zero` formulation `(1 - sigmoid(β·chosen)) + sigmoid(β·rejected)` via the identity `1 - σ(x) = σ(-x)`. ✅
  - Q2 connections: `_apo_zero_loss` ← `_resolve_preference_loss` ← `train_dpo` (loss_fn dispatch at L4096+L4224, single call site `loss_fn(...)`). All 3 live `train_dpo()` callers ([test_training.py L1165](tests/test_training.py#L1165), [gui_forge_training.py L735](enigma_engine/gui/gui_forge_training.py#L735), [training.py L5567](enigma_engine/core/training.py#L5567)) default to `"dpo"` so behaviour preserved. ✅
  - Q3 missing connections: `train_simpo`/`train_kto`/`train_orpo` are SEPARATE methods with distinct loss math, NOT DPO variants — correctly NOT in the registry. `BackgroundTrainer` (continuous learner) uses `train_dpo` internally but isn't user-exposed for preference-loss choice. GUI exposure correctly logged as **D-9b**. ✅
  - Q4 doc-vs-code: `train_dpo` docstring honestly says "loss_type: 'dpo' (default) ... or 'apo_zero' for the Anchored Preference Optimization zero variant." No drift. ✅
  - Q5 claim-vs-test: 8 tests including the structural+behavioural pair. `test_apo_zero_loss_chosen_independence_from_rejected` ([test_training.py L73](tests/test_training.py#L73)) reconstructs chosen-side terms via `loss - rejected_term` and asserts equality across two rejected configurations within 1e-6. Math is decomposable as `f(c) + g(r)` so term-level equality implies gradient-level independence. ✅
  - Defensive note (NOT a bug): `_apo_zero_loss` NaN-clamp branch at L3892 is essentially unreachable — sum of two sigmoids on finite inputs is always in (0,2). Mirrors `_dpo_loss` defensive zeroing for symmetry; consistency win, leave in place.

- **Pass 156i9 (FORGE picker default helper) — CLEAN with one follow-up backlog row.**
  - Q1: `_pick_default_training_file` at [scanners.py L482](enigma_engine/gui/scanners.py#L482) checks both relative-name suffix and OS-normalised path tail. Correct.
  - Q2/Q3: Verified no other `training_files[0]` consumers exist in `enigma_engine/gui/**/*.py` — picker is unique. ✅
  - **Connection-that-should-exist (NEW backlog row D-11c):** Same smart-default pattern is applicable to OTHER FORGE pickers. E.g. when `data/finetune/combined_finetune.jsonl` exists, the **DPO pair-data picker** should prefer it (or a future DPO-pair file); when `data/pretrain/combined_pretrain.txt` exists, the **pre-training data picker** should prefer it over scattered scratch files. Single helper covers all three; see new row below.

- **Pass 156i8 (dual-emit `_write_combined_text`) — ONE REAL ISSUE FOUND.**
  - Q5 + loud-on-real-issue lens caught a real silent-failure mode: at [collect_finetuning_data.py L528-538](collect_finetuning_data.py#L528), if `len(all_pairs) > 0` but ALL rows have empty/missing `prompt` or `completion`, `_write_combined_text` skips every row (correctly) and returns `0`. The combined `.txt` file is then **0 bytes** on disk and `combine_all` logs `INFO "Combined text: 0 blocks, 0.0 MB"` — silently. Per the project's "loud-on-real-issue, silent-on-normal-path" principle: `text_count == 0 and len(all_pairs) > 0` is a real misconfiguration (every fetcher emitted malformed rows) and should be a `WARNING`, not INFO. Logged as new backlog row **D-11d** below — small fix, single-line conditional + matching test (analogous to Pass 156i6's anchor-empty-yield WARNING).

- **Pass 156i7 (anchor wiring) — CLEAN.**
  - Q1: `_DEFAULT_ANCHOR_PATH if _DEFAULT_ANCHOR_PATH.exists() else None` checked once at `ModRouter.__init__`. Acceptable: anchor file is repo-shipped and not expected to appear post-startup. If user adds it mid-session they need to restart — same as any data-dir change. Not a bug.
  - Q3 connection check: grep for `BackgroundTrainer(` returned only [router.py L782](enigma_engine/router.py#L782) (production path) and tests. **No CLI bypass** — `run.py --serve` constructs `ModRouter` which forwards anchor path; no other entry-point creates a `BackgroundTrainer` directly. ✅
  - Q5 claim-vs-test: 3 behavioural tests cover present/missing/explicit-override. ✅

**New backlog rows from this audit:**

- **D-11c — Smart-default helper for other FORGE data pickers (priority P3, small UX nudge).** **CLOSED Pass 156l.** Generalised `_pick_default_training_file` into `_pick_first_match(files, preferred_tails)` plus two sibling pickers `_pick_default_dpo_data_file` / `_pick_default_pretrain_file`. Adversarial preference-order test gates iteration shape (preferences-outer, not files-outer). GUI wiring of the DPO/pretrain pickers deferred to a separate small pass once user confirms file paths.

- **D-11d — Loud-on-zero-yield WARNING in `_write_combined_text` (priority P3, real silent-failure mode).** **CLOSED Pass 156l.** [collect_finetuning_data.py L530-L545](collect_finetuning_data.py#L530) now WARNs when `text_count == 0 and len(all_pairs) > 0` instead of logging INFO. Mirrors Pass 156i6 anchor-loader pattern. Test captures logger output and asserts "text file is 0 bytes" string fires.

**No-action confirmations from this audit:**

- `_apo_zero_loss` NaN-clamp branch is unreachable on finite inputs but mirrors `_dpo_loss` for consistency — leave in place, do NOT remove.
- `train_simpo` / `train_kto` / `train_orpo` correctly NOT registered in `_resolve_preference_loss` — they are not DPO variants.
- `_DEFAULT_ANCHOR_PATH.exists()` is checked once at startup, not per-request — acceptable, file is repo-shipped.
- No CLI/test path bypasses `ModRouter` to construct `BackgroundTrainer` without the anchor wiring (verified by grep).

**Pass 156j (D-9 — Anchored Preference Optimization, zero variant):** Closes the P2 row that was deferred until N-10 DPO validation. Library-level addition of APO-zero (D'Oosterlinck et al., 2024) as a `loss_type` kwarg on `Trainer.train_dpo`. Three new statics in [training.py](enigma_engine/core/training.py): `_apo_zero_loss(pc, pr, rc, rr, beta)` computing `sigmoid(-beta * chosen_logratio) + sigmoid(beta * rejected_logratio)` (TRL form, anchors each side independently to ref); `_resolve_preference_loss(loss_type)` registry mapping `"dpo"`/`"apo_zero"` to the static implementations — unknown values raise `ValueError` so typos fail loud; `train_dpo(loss_type="dpo")` default preserves all 3 existing callers. The loop body now calls `loss_fn(...)` resolved at entry. Why APO over DPO: DPO can be satisfied by *degrading rejected* without *improving chosen* (the loss only sees the difference). APO-zero anchors each side to the frozen reference policy independently, so chosen must actually rise above ref AND rejected must fall below ref. **8 behavioural tests** in [test_training.py::TestAPOZeroLoss](tests/test_training.py): zero-logratios-returns-1.0 (`sigmoid(0)+sigmoid(0)=1.0` exactly), ideal-state-near-zero, **chosen-side-independence-from-rejected** (the key APO property — changing rejected_logratio does NOT shift the chosen contribution; DPO fails this), degrading-rejected-below-ref-bounded-by-chosen-floor (loss bottoms at 0.5 when only rejected is suppressed; DPO has no such floor), non-finite policy logps clamped to 0 (mirrors `_dpo_loss` defensive zeroing), structural `"loss_type" in source`, invalid-loss-type-raises, **dispatch-actually-routes** (claim-vs-test discipline — patches both loss statics with sentinel recorders + stubs `_get_sequence_logps`, asserts apo sentinel called once, dpo sentinel never called — catches the regression where someone reverts `loss = loss_fn(...)` back to `loss = self._dpo_loss(...)`). Suite **2409/9** (+8), ruff clean. **Closes D-9.** Follow-up D-9b opened: GUI radio card + dispatcher entry for APO-zero alignment mode (currently library-only; users can call `trainer.train_dpo(loss_type="apo_zero", ...)` directly but no FORGE UI surface).

**Pass 156i9 (D-11b — FORGE training-data picker default):** One-line UX nudge that finishes Pass 156i8. The FORGE Basic page initialised `train_data_var` with `self.training_files[0]["path"]` ([gui_pages_forge.py L482](enigma_engine/gui/gui_pages_forge.py#L482)) — glob order from `scan_training_data()`, which surfaces the 2-line placeholder `data/training.txt` over the just-shipped 60 MB `data/finetune/combined_finetune.txt`. Fix: new helper `_pick_default_training_file(files)` in [scanners.py](enigma_engine/gui/scanners.py) — scans the list and returns the path of any entry matching `finetune/combined_finetune.txt` (relative-named or path-tail), else `files[0]`, else `""`. Wired into `gui_pages_forge.py` at the StringVar init. 4 behavioural tests in [tests/test_gui.py::TestScanners](tests/test_gui.py): prefers-combined-when-present, falls-back-to-first-when-no-combined (legacy preserved), empty-list-returns-empty-string, combined-in-arbitrary-position (adversarial ordering — would pass on accident if helper just returned `files[0]` when the corpus happened to be first; this test puts it 3rd). Suite **2401/9** (+4), ruff clean. **Closes D-11b.**

**Pass 156i8 (D-11 consumer-side wiring — SmolTalk2 / OpenThoughts3 reach the SFT trainer):** Author's-lens audit caught the real D-11 gap: Pass 155+155b shipped the *fetcher* (`collect_smoltalk2`, `collect_openthoughts3` writing JSONL to `data/finetune/`) but **no `enigma_engine/` code reads from `data/finetune/`** — the collected reasoning data was sitting on disk with zero consumers. Classic "infrastructure without consumers" anti-pattern. Fix: add `_write_combined_text()` helper to [collect_finetuning_data.py](collect_finetuning_data.py) that emits `data/finetune/combined_finetune.txt` in canonical `User: <prompt>\n\nAssistant: <completion>` format alongside the existing combined JSONL. Format matches `BackgroundTrainer` text builder at [router.py L315](enigma_engine/router.py#L315) and the FORGE chat path. The existing SFT trainer reads plain text via `Path(data_path).read_text(encoding="utf-8")` ([gui_forge_training.py L271](enigma_engine/gui/gui_forge_training.py#L271)) so this requires **zero training-side change** — user just points the FORGE file picker at `combined_finetune.txt`. 3 new behavioural tests in [tests/test_collect_finetuning_data.py::TestCombineAllText](tests/test_collect_finetuning_data.py): file-emitted-alongside-jsonl, canonical-format-with-blank-line-separators, empty-pair-skip (would-catch-regression test for naive `f"User: {p}..."` on empty prompt). Live re-run on existing 1,999-pair corpus produced 59.8 MB text file. Suite **2397/9** (+3), ruff clean. **Closes D-11.** Follow-up row D-11b opened: FORGE GUI file picker default points at `data/training.txt`; user has to manually navigate to `data/finetune/combined_finetune.txt`. Small UX nudge — add a "recent finetune data" shortcut in the picker.

**Pass 156i7 (Continuous-3 — anchor wiring + repo default file):** Closes the Pass 156i5+i6 plumbing by making anchor rehearsal reachable for normal users without hand-edits. **(1) Curated default anchor file** [data/anchor_examples.jsonl](data/anchor_examples.jsonl) — 51 high-confidence general-capability rows spanning math (12), Python code (12), reasoning (8), general knowledge (10), language (9). Each row is `{prompt, response, score: 1.0}`. Format intentionally minimal so users can hand-curate additions. **(2) Boot-path wiring** — new module-level `_DEFAULT_ANCHOR_PATH = Path(__file__).resolve().parent.parent / "data" / "anchor_examples.jsonl"` ([router.py L26](enigma_engine/router.py#L26)) resolves relative to the source file (CWD-independent). `ModRouter.__init__` ([router.py L727](enigma_engine/router.py#L727)) gained `anchor_data_path: str | Path | None = None` kwarg with auto-default-when-present logic: explicit caller value wins; otherwise repo default if file exists; otherwise None (no warn noise on boot when feature opted out). `_create_trainer` forwards the resolved path to `BackgroundTrainer(anchor_data_path=...)`. Library default of `BackgroundTrainer(anchor_data_path=None)` preserved — wiring lives at the boot site, not the library default (test isolation, explicit-opt-in for direct callers). **(3) 3 behavioural tests** in [tests/test_core.py::TestRouter](tests/test_core.py): `test_router_passes_repo_anchor_file_to_trainer_when_present` (file exists → forwarded), `test_router_passes_none_when_anchor_file_missing` (absent → None, no phantom path), `test_router_explicit_anchor_path_overrides_default` (caller wins). All red pre-fix. Live smoke-test confirmed: `BackgroundTrainer(anchor_data_path=_DEFAULT_ANCHOR_PATH)._load_anchor_examples()` returns 51 examples, all `source="anchor"`. **Not done in this pass:** GUI widget for selecting/editing anchor file (deferred to Continuous-3b — small follow-up; the file is now editable from disk by users, which unblocks practical use). Caller-loop schedule independent of `len(replay_buffer) >= batch_size` (Continuous-3c — design question: should anchors run on a separate periodic cadence even with zero recent activity? Probably yes for true "always-on rehearsal" but needs scheduling-design discussion). Suite **2394/9** (+3), ruff clean. **Closes Continuous-3 core wiring.**

**Pass 156i6 (Continuous-2a — self-audit on Pass 156i5 anchor-set rehearsal):** Author's-lens audit on Pass 156i5 surfaced three real bugs in code shipped minutes earlier (per AA self-audit-on-ship rule). **(1) [FIXED] Sibling-comment drift on the honesty reframe.** Class docstring was reframed honestly ("mitigates ... bounded by anchor coverage") but two sibling claims still over-promised: inline comment at [router.py L83](enigma_engine/router.py#L83) and `_retrain_on_replay` docstring at [router.py L470](enigma_engine/router.py#L470) both said "prevents catastrophic forgetting." Same anti-pattern as the seed-method drift principle — when fixing one site of a cross-doc claim, grep all siblings the same pass. Both reframed to mitigation language. **(2) [FIXED] Anchor early-out fired before anchors were loaded.** Pre-fix, `_retrain_on_replay` returned on `not self.replay_buffer` *before* calling `_load_anchor_examples()`, defeating the entire feature during quiet periods (anchors exist precisely to rehearse skills NOT in recent chat). Reordered: load anchors first, gate on combined emptiness. New behavioural test `test_anchor_rehearsed_when_recent_buffer_empty` was red pre-fix (zero forward calls), green post-fix. Note: the *caller* gate at [router.py L392](enigma_engine/router.py#L392) (`len(replay_buffer) >= batch_size`) still blocks anchor-only periodic rehearsal at the trainer-loop level — that's a Continuous-3 design question (anchor-only schedule independent of chat activity). **(3) [FIXED] Loud-on-real-issue gap.** When the anchor file existed but contained no usable rows (all empty/malformed), `_load_anchor_examples` logged INFO "loaded 0 anchor example(s)" — silently degrading misconfiguration to recent-only. Added explicit WARNING when the path is configured + file readable + zero usable rows parsed. New test `test_anchor_file_present_but_empty_yields_warning` verifies. **(4) [FIXED] Soft test on legacy path.** `test_anchor_path_none_preserves_legacy_behavior` only asserted no `"anchor_"` substring was encoded — tautological for the fixture. Strengthened with direct `assert bt._load_anchor_examples() == []`. **(5) [FIXED] Missing cache test.** Pass 156i5 added `_anchor_load_attempted` flag to make the load idempotent but had no test for the cache property — breaking the cache (re-read every replay) would silently pass all 5 prior tests. New `test_anchor_file_loaded_only_once_across_replay_passes` uses `unittest.mock.patch.object` on `Path.open` to count file opens across 3 consecutive `_retrain_on_replay()` calls; asserts exactly 1. Suite **2391/9** (+3), ruff clean. **Closes Continuous-2a.**

**Pass 156i5 (Continuous-2 — anchor-set rehearsal against forgetting):** Closes the trilogy opened by Pass 156i4 — class docstring previously claimed `BackgroundTrainer` "prevents catastrophic forgetting" but `_retrain_on_replay` only rehearsed the recent chat buffer, with **no general-capability anchor data**. A user spending weeks chatting only about cooking would silently lose math/code/reasoning skills even with replay running. Fix: new constructor kwarg `anchor_data_path: str | Path | None = None` ([router.py L93](enigma_engine/router.py#L93)) pointing at a JSONL file of curated anchor examples (`{prompt, response, [score]}` per row, `source="anchor"`). New helper `_load_anchor_examples()` ([router.py L411](enigma_engine/router.py#L411)) loads + caches on first replay; missing/unreadable file logs a single WARNING and falls back to recent-only behaviour (loud-on-real-issue, silent-on-normal-path). `_retrain_on_replay` extends `replay_batch` with the full anchor set **before** `replay_len` is computed, so loss scaling stays consistent across the combined batch. Anchors are NOT score-sorted — curated order is intentional. Same NaN/Inf abort + token-length cap apply uniformly to anchors and recent (mirror discipline per AA Top-9 #2). **Honesty:** docstring re-framed from "prevents catastrophic forgetting" to "mitigates ... bounded by anchor coverage — a 50-example anchor set is a floor, not a guarantee." 5 behavioural tests in [tests/test_training.py](tests/test_training.py): `test_anchor_path_none_preserves_legacy_behavior` (no path → tokenizer never sees anchor text), `test_anchor_examples_loaded_from_jsonl` (parses prompt/response/score, sets `source="anchor"`), `test_anchor_examples_flow_through_replay_pass` (anchor prompts reach forward alongside recent slice), `test_anchor_missing_file_logs_warning_and_continues` (bad path → WARNING, no crash), `test_anchor_oversize_examples_skipped` (cap honoured equally for anchors). **Not done in this pass:** GUI surface for selecting the anchor file, default anchor data file shipped in `data/`. Both deferred to Continuous-3 (small follow-up). Suite **2388/9** (+5), ruff clean. **Closes Continuous-2.**

**Pass 156i4 (Continuous-1 — BackgroundTrainer silent-drift safety):** Author's-lens audit of [enigma_engine/router.py L71](enigma_engine/router.py#L71) `BackgroundTrainer` caught three real issues plus one over-promising docstring. **(1) [FIXED] No NaN/Inf abort — silent model corruption over months:** neither `_train_batch` ([L256](enigma_engine/router.py#L256)) nor `_retrain_on_replay` ([L355](enigma_engine/router.py#L355)) checked `loss` finiteness before stepping the optimizer. Once `loss.item()` produces NaN on a single bad sample, the optimizer applies NaN gradients → every weight in the model becomes NaN → every future inference returns garbage. This is the *exact* "possible silent drift over months" failure the backlog row warned about. Fix: `if not torch.isfinite(loss): logger.warning(...); continue` inserted before each `.backward()` call; the existing `if valid_count > 0` gate around the step blocks updates when every sample in the batch was bad. Pre-fix behavioural test (`test_train_batch_skips_step_on_nan_loss`, `test_retrain_on_replay_skips_step_on_nan_loss`) wired a `_NaNModel` returning `torch.full(..., nan)` logits and a step-call tracker; both red pre-fix (step called once with NaN grads), both green post-fix (step skipped, model preserved). **(2) [FIXED] No token-length cap — OOM exposure:** a misbehaving mod or pathological chat input could hand `_train_batch` a 1M-token sequence, which then becomes a `[1, ~1M]` long tensor on GPU. New `BackgroundTrainer.__init__(max_token_length: int = 4096)` constructor kwarg; `_train_batch` and `_retrain_on_replay` skip-with-DEBUG when `len(tokens) > max_token_length` rather than truncating (truncation would silently drop context — refusing the sample is honest). Default 4096 matches our typical `max_seq_len` budget. Behavioural test `test_train_batch_caps_oversize_tokens` builds a tokenizer returning `range(20000)` and asserts the model's forward is never called. **(3) [FIXED] Misleading test claim:** `test_replay_keeps_best_examples` (Pass 132 era) asserted "buffer keeps top 3 by score" but `replay_buffer = deque(maxlen=N)` evicts FIFO, not by score — the test only passed because the lowest score happened to be inserted first. Classic claim-vs-test mismatch (logic-eye lens, AA Top-9 #9). The *code* is correct (intent is recency per inline comment "rolling collection of recent examples"; quality filtering happens at retrain time inside `_retrain_on_replay` which sorts by score). Renamed to `test_replay_buffer_evicts_by_recency_not_score` and rewrote to insert highest-score *first* so a true "keeps best" implementation would fail — proves recency semantics deterministically. **(4) [LOGGED, not fixed]** Class docstring claims "prevents catastrophic forgetting" but replay only rehearses recent chat — no general-capability anchor data. Rehearsal of recent data does not prevent forgetting of *un-rehearsed* skills. New backlog row **Continuous-2** opened in P2: add a fixed "anchor set" (~50 high-quality general examples) that always ships in every replay batch alongside the score-sorted recent slice. Suite **2383/9** (+4), ruff clean. **Closes Continuous-1.**

**Pass 156i3 (DET-2 + EWC-1 close + CLI wiring):** Three backlog items landed in one pass. **(1) DET-2 [SHIPPED]** — `set_training_seed(seed, deterministic=False)` ([training.py L307](enigma_engine/core/training.py#L307)) gained an opt-in flag that, when True, sets `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"` and calls `torch.use_deterministic_algorithms(True, warn_only=True)`. `warn_only=True` is mandatory — MoE `index_add_` lacks a deterministic CUDA kernel and a hard error there would block every MoE training run; with `warn_only` the user gets a one-line UserWarning and training continues. New `TrainingConfig.deterministic: bool = False` field at [L484](enigma_engine/core/training.py#L484) (default False preserves backward compat — every existing run is byte-identical to before this pass). All 8 sibling seed-guard sites updated to forward the flag (`set_training_seed(self.config.seed, deterministic=self.config.deterministic)`) — `train`, `train_dpo`, `train_simpo`, `train_kto`, `train_orpo`, `train_vision`, `train_audio`, `train_rest`. `to_dict()` updated to include the new field. Pass 156i structural test loosened from `set_training_seed(self.config.seed)` to `set_training_seed(self.config.seed` so it accepts both forms (presence-only check unchanged). Four new tests in [tests/test_training.py::TestDeterministicFlag](tests/test_training.py): `test_deterministic_off_by_default` (env var untouched, helper not called), `test_deterministic_true_sets_env_and_flag` (asserts `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `warn_only=True`), `test_training_config_default_deterministic_false` (back-compat), `test_seed_call_forwards_config_deterministic` (structural — every sibling forwards the flag). **(2) DET-2 CLI wiring [SHIPPED, post-author's-lens scan]** — caught a "infrastructure without consumers" gap: `TrainingConfig.deterministic` was reachable only via hand-edited code; `run.py --train` had no flag. Added `--deterministic` (store_true) at [run.py L158-162](run.py#L158), routed through `run_train()` signature + `TrainingConfig(...)` construction, and added an early `parser.error("--deterministic requires --seed")` guard so the silent-no-op trap (where the flag is ignored when seed is None) fails loud instead. **(3) EWC-1 [CLOSED — WONTFIX, superseded by LoRA path]** — Pass 156h Part B audit recommended close; user authorized via "complete as much as possible" pass. Library at [enigma_engine/core/ewc.py](enigma_engine/core/ewc.py) stays in place for the rare-case future (intentional full-weight specialization that still wants generalist retention) but is no longer a roadmap item. LoRA-per-specialist (lora_utils.py already exists) makes forgetting *physically impossible* — frozen base weights, 10-30 MB adapters, runtime-swappable. New row **LoRA-1** opened in P2 to formalize the FORGE wiring that supersedes EWC. **Scope honesty:** combined with DET-1 (Pass 156i), the project now supports — when `seed` is set AND `deterministic=True` — bitwise-reproducible training on CUDA at the documented 5-15% throughput cost. **Not done:** DataLoader `worker_init_fn` (Enigma has no DataLoader workers — single-threaded data path); `Trainer.__init__` DRY refactor collapsing 8 guards to 1 (separate design row, semantic shift). Suite **2379/9** (+4), ruff clean.

**Pass 156i2 (logic-eye self-audit on 156i):** First audit of 156i was bug-eye only — found nothing because the *code* is correct. User pushback ("you look at the logical end of things right?") triggered a second pass with **logic-eye + claim-vs-test** lenses. Found six real issues, fixed four, logged two. **(1) [FIXED] Doc oversells reproducibility:** prior 156i stamp implied "you can rerun exact same training and get exact same model." Reality: `set_training_seed()` seeds Python+torch CPU+CUDA *seeds* but does NOT call `torch.use_deterministic_algorithms(True)` or set `CUBLAS_WORKSPACE_CONFIG`, so cuBLAS/cuDNN can still pick different kernels per launch. Code delivers *Python-RNG reproducibility* (sample/shuffle order); doc claimed *bitwise model reproducibility*. Stamp re-worded to match what ships. **(2) [FIXED] Structural test gates presence not correctness:** `test_all_training_methods_seed_their_rng` greps for the literal string `set_training_seed(self.config.seed)` — a wrapper (`if flag: set_training_seed(...)`), an arg change (`set_training_seed(self.config.seed or 42)`), or a misplaced call (after the consumer) all pass the structural gate while the fix is broken. Added behavioural test `test_train_dpo_seeded_shuffle_is_reproducible` ([tests/test_training.py L941](tests/test_training.py#L941)) on `train_dpo` (most-used non-vision sibling). Patches `random.shuffle` to capture `random.getstate()` at first call, then raises a sentinel `_StopAtShuffle` to abort the method without needing a full forward/backward (sidesteps the heavy fixture problem). Critical detail: between captures, **explicitly pollute global RNG** with `random.seed()` + 50 dummy draws — without this, prior test ordering leaves RNG deterministic and the test passes spuriously even when seeding is broken (verified by temporarily flipping the guard to `if False and ...` — pre-pollution: green, post-pollution: red as expected; restored real fix). **(3) [FIXED] ReST round-loop RNG flow unintuitive:** each inner `train_dpo()` call re-seeds back to `config.seed`, so round-2 DPO starts from the same RNG state as round-1 DPO. Sample order differs because `pairs` differ (regenerated from the updated model), but RNG draws are identical streams across rounds. Added a 9-line comment at [training.py L5417](enigma_engine/core/training.py#L5417) documenting this so future readers don't trip over it; "fresh randomness per round" requires `config.seed + rnd` derived outside this method. **(4) [FIXED] AA Code Maker rules updated:** author's lens now has **5 questions** instead of 3 — added (4) logic-eye "does the code deliver what the doc claims?" and (5) claim-vs-test "could the test pass while the code is wrong?". Top-9 list (was Top-8) now includes "logic-eye every audit, not just bug-eye." Verification section gained three new rules (logic-eye on doc claims, structural-vs-behavioural test discipline, re-seeding clobbers caller RNG silently). **(5) [LOGGED, not fixed] CUDA non-determinism gap (DET-2 row):** for true bitwise A/B reproducibility we'd need `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` + (optional) `num_workers` consistency for DataLoader. Costs 5-15% throughput per the PyTorch docs. Worth a separate row, not silent retro-fit. **(6) [LOGGED, not fixed] Init-time vs per-method seeding is a design choice:** seeding inside every public `train_*()` is 8 copies of the same 3-line guard. Alternative: seed once in `Trainer.__init__` from `config.seed`, optionally re-seed per method with method-name-derived offset (so DPO and SFT get different deterministic streams). DRY win, but slight semantic shift — needs design discussion before refactor. Suite **2375/9** (+1 behavioural), ruff clean. **Lesson burned in:** bug-eye alone misses over-promised docs, partial fixes, and structural-only tests. Logic-eye is now mandatory per AA rules.

**Pass 156i (DET-1 — broad determinism audit + fix):** Pass 156h Part A fixed un-seeded shuffle in `train_vision`; this pass closes the same gap across the rest of the training surface. Audit grep located un-seeded `random.shuffle` / `random.sample` in **six** sibling methods: `train_dpo` ([L4002](enigma_engine/core/training.py#L4002), shuffles at L4077 + samples at L4155), `train_simpo` ([L4244](enigma_engine/core/training.py#L4244), shuffle at L4320), `train_kto` ([L4427](enigma_engine/core/training.py#L4427), shuffle at L4510), `train_orpo` ([L4623](enigma_engine/core/training.py#L4623), shuffle at L4682), `train_audio` ([L5486](enigma_engine/core/training.py#L5486), shuffle at L5601), `train_rest` ([L5398](enigma_engine/core/training.py#L5398), no direct shuffle but outer round-loop generation needs deterministic RNG before first `_generate_online_dpo_pairs` call). All six were missing the `set_training_seed(self.config.seed)` call that `train()` (L2279) and `train_vision()` (Pass 156h, L4840) already had. Fix: identical 3-line guard inserted right after each method's data-validation block, before `self._stop_requested = False`. Guard is `if self.config.seed is not None: set_training_seed(self.config.seed)` — preserves legacy "no seed = wild RNG" behaviour when user hasn't set one (matches `TrainingConfig.seed: int | None = None` default at [L479](enigma_engine/core/training.py#L479)). **Scope honesty (post-156i2 logic-eye revision):** this fix delivers **Python-RNG reproducibility** — sample/shuffle order is identical across runs with the same `config.seed`. It does **NOT** deliver bitwise model reproducibility on GPU, which additionally requires `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG` (see DET-2 row). Failing test `test_all_training_methods_seed_their_rng` ([tests/test_training.py L885](tests/test_training.py#L885)) is **structural** — `inspect.getsource()` per method must contain `set_training_seed(self.config.seed)`. Behavioural coverage: `test_train_vision_seeded_shuffle_is_reproducible` (Pass 156h, validates the pattern works for vision); `test_train_dpo_seeded_shuffle_is_reproducible` (Pass 156i2, validates the pattern works for DPO via RNG-state capture at first shuffle). Pre-fix structural red on all six sibling methods; post-fix green. ReST round-loop RNG semantics documented in code comment (Pass 156i2). Suite **2374/9** at 156i, **2375/9** at 156i2, ruff clean.

**Pass 156h (Part A — train_vision shuffle seeded):** Pre-existing finding from Pass 156g2 surface-notes addressed. [training.py L4793](enigma_engine/core/training.py#L4793) `train_vision()` did not call `set_training_seed()` — only the main `train()` method did, at [L2278-2279](enigma_engine/core/training.py#L2278). The per-epoch `random.shuffle(pairs)` at [L5118](enigma_engine/core/training.py#L5118) therefore ran against un-seeded global RNG state, so two runs with the same `data` + same `config.seed` processed samples in different orders. Same root-cause as Pass 156g2 Bug A but on the per-epoch shuffle inside the trainer rather than on the GUI-side val/train split. Fix: added `if self.config.seed is not None: set_training_seed(self.config.seed)` right after the data-validation guard, mirroring `train()`. Failing test `test_train_vision_seeded_shuffle_is_reproducible` ([tests/test_training.py L823](tests/test_training.py#L823)) builds 8 visually-distinguishable PIL images (per-pixel index marker), patches `vision_encoder.preprocess_image` to record the order images flow through the loop, runs `train_vision()` twice with `seed=42`, asserts identical sequences. Pre-fix: random orderings differed (`[1,0,6,7,2,3,5,4]` vs `[3,4,0,7,5,6,1,2]`). Post-fix: identical, and not the trivial identity (sanity-asserted). Suite **2373/9** (+1), ruff clean. **DET-1 row CLOSED Pass 156i — see top of file.**

**Pass 156h (Part B — EWC-1 audit, recommend close as superseded):** Author's-lens read of [enigma_engine/core/ewc.py](enigma_engine/core/ewc.py). The module is a **complete, well-tested library**: `EWC.__init__` (snapshot params + estimate diagonal Fisher via empirical squared-gradient average), `EWC.penalty(model)` (returns `(λ/2) * Σ F_i (θ_i − θ*_i)²` as a grad-attached scalar), `EWC.save/load` (atomic torch-save round-trip with `lam`/`fisher`/`params`). Four behavioural tests pass in [tests/test_core.py L3865+](tests/test_core.py#L3865). **Production callers across `enigma_engine/`: zero** (verified by grep — only consumer is `tests/test_core.py`). Classic dead-code library. **Honest scope check on wiring:** to wire EWC into FORGE SFT/dialogue properly we need (1) decision on **capture trigger** — end of every base run? user opt-in? only after pre-train, not after SFT? (2) decision on **storage location** — separate `<model>.ewc.pth` alongside `.pth`? sub-key inside the checkpoint? (3) decision on **default λ** — paper says 100-10000, model-size-dependent; (4) **VRAM budget impact** — Fisher + anchor stores 2 fp32 copies of trainable params (~6 GB for a 742M model — non-trivial on the 16 GB target); (5) **compute cost** — Fisher pass is `n_samples × (forward + backward)`, ~1-3 minutes for n=100 on a 700M model. None of these have a single obvious answer. **Critically, prior research at line 799 of this file already concluded LoRA-per-specialist supersedes EWC for Approach 2** (LoRA freezes base weights so forgetting is *physically* impossible; adapters are 10-30 MB; swap at runtime). [enigma_engine/core/lora_utils.py](enigma_engine/core/lora_utils.py) already exists. EWC's only remaining niche is "intentional full-weight specialization that still wants to retain general capability" — a rare case where the user has explicitly chosen against the LoRA path. **Recommendation:** close EWC-1 as **WONTFIX (superseded)**, leave the library in place for the rare-case future, prioritize a LoRA-per-specialist FORGE row (LoRA-1) instead. Awaiting user decision before stamping the close — see "Decision queue" row below.

**Pass 156g2 (self-audit on 156g):** Author's-lens review of own V-6 + V-6b shipping caught three real issues — fixed all three. **(Bug A) V-6b val partition non-reproducible:** [gui_forge_training.py L1067](enigma_engine/gui/gui_forge_training.py#L1067) shuffled `vision_data` with the global `random` state, no seed. Two runs with the same data + same `val_split` produced different held-out partitions, breaking (1) AA's "deterministic in infrastructure" rule and (2) checkpoint resume comparisons (`state.validation_losses` baseline becomes incomparable after re-shuffle). Fix: use `random.Random(train_config.seed)` when `seed` is set, plain `random.Random()` otherwise; log the seed value with the split summary. **(Bug B) `_run_validation()` ignored STOP:** [training.py L5012](enigma_engine/core/training.py#L5012) iterated `val_pairs` without polling `self._should_stop()`. At LLaVA-Pretrain scale with `val_split=0.05` that's ~28K val samples per epoch — STOP press would freeze the GUI for minutes per epoch end. Fix: poll `_should_stop()` each iteration, return the partial mean over completed samples on early exit (`None` if zero finished); log "stopped after N sample(s)". **(Bug C) inline `import random as _rand`:** [gui_forge_training.py](enigma_engine/gui/gui_forge_training.py) had no top-level `import random`; the inline alias inside the function body read weirdly. Hoisted to module imports alongside `logging`/`threading`. Two new audit tests: `test_train_vision_validation_honors_stop` (20 val pairs, arms stop after the third preprocess call, asserts <22 total preprocess calls — pre-fix: exactly 22) and `test_val_split_shuffle_is_seeded` (structural — rejects pre-fix `random.shuffle` pattern, requires `Random(`/`.seed(`/`config.seed` evidence within 400 chars before the shuffle call). Both red before fix, green after. Suite **2372/9** (+2), ruff clean. Lesson confirmed once more: **self-audit immediately after shipping is mandatory**. Surface findings include two pre-existing issues NOT fixed in this pass — train shuffle at [training.py L5076](enigma_engine/core/training.py#L5076) is also unseeded (predates V-6, deferred to a separate row), and no heartbeat ping fires during the val pass (cross-layer wiring, deferred).
**Pass 156g (V-6 + V-6b):** Held-out validation pass added to `train_vision()` and wired into the GUI. **V-6 (backend):** new optional `val_data: list[dict] | None = None` parameter at [training.py L4789](enigma_engine/core/training.py#L4789). Same lightweight readability probe + lazy-preprocess discipline as the train path; val captions <2 tokens are skipped silently (matches train-loop drop policy at `min_len < 1`). `_run_validation()` closure runs no-grad cross-entropy over `val_pairs` after each epoch, switches both `vision_encoder` and `self.model` to `eval()` inside a `try/finally` so train mode is always restored, returns `None` when no valid val samples exist. Result lands in `state.validation_losses` (already round-trips through `_save_state`/`_load_state` at L3549/L3752 from the pre-train infra — vision now gets resume-safe val history for free). Best-checkpoint + early-stopping prefer val_loss when present, fall back to train avg_loss otherwise. **V-6b (GUI):** [gui_forge_training.py L1057](enigma_engine/gui/gui_forge_training.py#L1057) call site now reads `train_config.val_split` (already exposed via the existing forge widget) and shuffle-splits `vision_data` into train/val before invoking `trainer.train_vision(..., val_data=...)`. Edge cases: `val_split <= 0` → no split (legacy behaviour); `len(vision_data) < 2` → no split (need ≥1 each side); `n_val` clamped to leave at least one train pair; logs `"Vision split: N train / M val (val_split=0.10)"` so the user sees exactly what was carved off. Three behavioural / structural tests: `test_train_vision_records_validation_loss` (3 train + 2 val pairs over 2 epochs → `len(state.validation_losses) == 2`, all finite), `test_train_vision_no_val_data_keeps_validation_losses_empty` (val_data omitted → `validation_losses == []`), and `test_val_split_plumbed_to_train_vision` (GUI source must reference `val_split` and pass `val_data=` into `trainer.train_vision(...)`). Pre-fix: V-6 backend test failed `TypeError: unexpected keyword argument 'val_data'`; V-6b GUI test failed because the call site hardcoded `data=vision_data` only. Suite **2370/9** (+3), ruff clean.
**Pass 156f (V-2):** Eager GPU preprocess fixed. Old code at [training.py L4906](enigma_engine/core/training.py#L4906) called `preprocess_image(...).to(self.device)` for **every** item before the training loop and stored the GPU tensor in `pairs`. At LLaVA-Pretrain scale (558K × ~600 KB) that's 60 GB on GPU — instant OOM on a 16 GB VRAM budget. Fix: `pairs` now stores `(image_or_path, token_ids)` references only; `preprocess_image(...).to(self.device)` runs inside the step loop where each tensor is freed after `loss.backward()`. Prep loop replaced with a lightweight PIL `verify()` header probe for path inputs (parses header without decoding pixels) so unreadable files still surface during prep instead of mid-training. PIL Image objects passed in directly skip the probe and are deferred wholesale. New behavioural test `test_train_vision_lazy_preprocess` patches `vision_encoder.preprocess_image`, monkeypatches `Trainer._should_stop` to always-True (the function resets `_stop_requested = False` at entry, so the public flag would be cleared), runs `train_vision` with 5 PIL images, asserts **0** preprocess calls. Fails before fix (5 calls), passes after. Suite **2367/9** (+1 V-2 test), ruff clean.
**Pass 156e (V-1 + ruff hardening):** Two unrelated wins. **(1) V-1 `max_grad_accumulation` in `train_vision()`:** confirmed by grep — vision was the lone training method ignoring the config field that pre-train, DPO, KTO, SimPO, audio, ReST all honor. On a 16 GB VRAM budget at LLaVA scale this starves the optimizer of effective batch size. Fix at [training.py L4963](enigma_engine/core/training.py#L4963): hoisted `optimizer.zero_grad()` out of the inner loop (now zeros at epoch start + after each real step), divided loss by `accum_steps`, recovered unscaled value for logging/NaN guards via `loss.item() * accum_steps`, gated `optimizer.step()` + `scheduler.step()` + `zero_grad` on the boundary (`accum_count % accum_steps == 0`). Skipped samples (`min_len < 1`, `n_patches >= logits.shape[1]`) don't advance `accum_count` so they don't fake a boundary. Added end-of-epoch remainder flush so trailing micro-batches aren't discarded by the next epoch's zero. AMP path mirrors non-AMP: `scaler.scale(loss).backward()` each micro-step; at boundary `scaler.unscale_` (if clip) → `clip_grad_norm_` → `scaler.step` → `scaler.update`. Two behavioural tests in [tests/test_training.py](tests/test_training.py): `test_train_vision_honors_max_grad_accumulation` (4 samples / accum=2 → exactly 2 step calls) and `test_train_vision_flushes_accum_remainder_at_epoch_end` (3 samples / accum=2 → 2 step calls = 1 boundary + 1 flush). Both fail before fix (4 and 3 respectively), pass after. **(2) Ruff RUF group enabled:** added `"RUF"` to `[tool.ruff.lint] select` in [pyproject.toml](pyproject.toml). 76 auto-fixes applied (mostly stale `# noqa` cleanup, RUF010 explicit f-string conversions, RUF019 unnecessary key checks). Documented ignores: RUF001/002/003 (unicode noise — box-drawing/emoji), RUF005/043/059 (style choices), RUF012 (15 sites of class-level lookup dicts needing manual `ClassVar`), RUF013 (23 implicit Optional sites — py39 codebase can't use `|` syntax yet), RUF015, RUF022 (2 grouped `__all__` blocks where auto-sort would destroy human-grouped section comments). Skipped `--unsafe-fixes` for the same reason. Suite **2366/9** (+2 V-1 tests), ruff clean.
**Pass 156d2 (self-audit on 156d):** Author's-lens review of own Pass 156d work caught three real issues — fixed all three. **(1) V-7 abort-summary skipped on NaN/Inf and max_loss aborts:** the `dropped_short_captions` summary at end-of-cleanup was bypassed by the `return self.state` aborts at [training.py L5066/L5076](enigma_engine/core/training.py#L5066). User would be told "NaN at step X" but never told "47 captions also dropped earlier". Hoisted into `_emit_drop_summary()` closure, called from success cleanup + both abort paths. New test `test_drop_summary_emitted_on_nan_abort` patches `cross_entropy` to return NaN on the third sample, asserts summary mentions the 2 prior drops. **(2) V-4 OOM heuristic divergence from reference:** vision crash branch used `"cuda" AND "memory"` (narrower than reference). Pre-training reference at [gui_forge_new_modes.py L1086](enigma_engine/gui/gui_forge_new_modes.py#L1086) uses `"out of memory" OR "cuda"` — same crash on same hardware was landing different status codes across modes. Aligned to reference. **(3) V-4 missing RuntimeError split:** vision used one `except Exception` + string heuristics, so a PIL/NumPy error containing "memory" would falsely tell user "GPU OOM, try smaller model". Split into `except RuntimeError` (OOM-friendly advice path) + `except Exception` (generic), matching reference. New test `test_oom_taxonomy_matches_pretrain` enforces both the split and the OOM-literal check. Two audit-tests added; suite **2364/9** green (+2), ruff clean. Lesson confirmed: applying the author's lens **to my own work right after shipping** caught issues that would have hidden the diagnostic signal in production. Self-audit is not optional.
**Pass 156d:** V-7 single-token-caption log-once + V-4 vision-training heartbeat shipped together (both small CPU-only fixes). **V-7:** `Trainer.train_vision()` at [training.py L5005](enigma_engine/core/training.py#L5005) no longer silently drops captions whose post-shift `min_len < 1`. First occurrence logs a `WARNING` naming epoch + step + reason ("caption too short for next-token loss after shift"); subsequent drops counted via local `dropped_short_captions`; end-of-run `WARNING` reports total drop count. Behavioural test [tests/test_training.py](tests/test_training.py) `TestVisionDataParsing::test_logs_once_when_caption_too_short` patches `tokenizer.encode` to return `[42]` (single token) ×3, asserts exactly one first-drop warning + one summary mentioning "3". **V-4:** `_start_vision_training()` in [gui_forge_training.py](enigma_engine/gui/gui_forge_training.py) now mirrors the [gui_forge_new_modes.py L170-260](enigma_engine/gui/gui_forge_new_modes.py#L170) heartbeat pattern. Stale-heartbeat check on entry (psutil PID liveness); inner `_write_hb(phase, step, loss, status)` closure inside `_vision_train` thread; calls at `data_load` (entry), `training` (rate-limited via existing 1.0s throttle in `on_progress`, plus per-epoch in `on_epoch` with step+loss), `complete` (success branch), `stopped` (KeyboardInterrupt), `crashed_oom` / `crashed_nan` / `crashed` (exception branch with message-string heuristics). Four structural tests in [tests/test_gui.py](tests/test_gui.py) `TestVisionTrainingHeartbeat` verify stale-check presence, helper definition, lifecycle status coverage, and inside-callback firing. Closes V-7 + V-4. Suite **2362/9** green (+5: 4 V-4 + 1 V-7), ruff clean.
**Pass 156c:** V-5 vision data collector shipped. New file [collect_vision_data.py](collect_vision_data.py) with `collect_llava_pretrain(max_samples, images_dir)`. Streams `liuhaotian/LLaVA-Pretrain` caption metadata; image bytes stay in user-managed `images.zip` extraction (separation principle — no auto-download of multi-GB archives). Per-row file-existence check skips missing images with warning (suppressed after 5 to avoid log flood; total reported at end). Output JSONL `data/vision/llava_pretrain.jsonl` with `{"image": <abs_path>, "text": <gpt_caption>}` — directly consumable by [scan_vision_data](enigma_engine/gui/scanners.py#L679) JSONL strategy and `Trainer.train_vision()`. Hard `FileNotFoundError` if `--images-dir` missing or non-existent (fail-fast over silent empty JSONL). Eight tests in [tests/test_collect_vision_data.py](tests/test_collect_vision_data.py) covering happy path, missing-image warning, max_samples cap, dedup, malformed-row skip, unknown-repo error, missing-dir hard-fail, and JSONL writer round-trip — fake-`datasets` injection per Pass 155 lesson. Closes V-5; unblocks V-1, V-2, V-4, V-7. Suite **2357/9** green, ruff clean.
**Pass 156:** F/Code-6 vision-training audit complete (read-only, author's lens). No code changes; findings logged below.
**V-8 [FIXED, Pass 156b]:** Vision encoder load path now wired. New helper `_load_vision_encoder_from_checkpoint(raw_checkpoint, model, model_file)` at [inference.py](enigma_engine/core/inference.py) called from `_load_pytorch` right after `model.load_state_dict(...)`. Volume policy implemented exactly per the agreed table: state+config present and load OK → `INFO` once with image_size/dim/patches; state+config present and load fails → `RuntimeError` (loud); state present without config → `RuntimeError`; state present but model has no `vision_projection` → `RuntimeError`; neither key present → silent (normal text-only path). Chat-side counterpart: [engine_chat.py L80](enigma_engine/core/engine_chat.py#L80) `_encode_images_for_chat` now logs `WARNING` per call when `image_paths` is non-empty but `vision_encoder is None` ("image input ignored — train a vision-capable model with Forge and reload"); empty image list stays `DEBUG`. Seven new tests in [tests/test_inference.py](tests/test_inference.py) (`TestVisionEncoderLoad` ×5 + `TestImageDroppedWarning` ×2) — all green. Original V-8 entry below kept for trace.

**V-8 [original, kept for trace — now FIXED above]:** Vision-encoder load path is fully missing. Trace: trainer at [gui_forge_training.py L911](enigma_engine/gui/gui_forge_training.py#L911) builds `VisionEncoder(v_preset)`, [L997-998](enigma_engine/gui/gui_forge_training.py#L997) writes `vision_encoder_state` + `vision_encoder_config` to the `.pth` as top-level keys. Grep across the entire `enigma_engine/` package shows **zero readers** of either key. [inference.py L209](enigma_engine/core/inference.py#L209) sets `self.vision_encoder = None` and grep shows **zero other assignments** anywhere. [engine_chat.py L80](enigma_engine/core/engine_chat.py#L80) checks `if encoder is None: return None`, so `vision_features` stays None at [L584](enigma_engine/core/engine_chat.py#L584), then [L622](enigma_engine/core/engine_chat.py#L622) skips the multimodal branch entirely. Net result: train vision → save .pth → load → drop an image into chat → model behaves identically to text-only, with **no warning**. The `vision_projection` weights do round-trip (they live inside `model_state_dict`) but they're useless without an encoder feeding them. Classic "infrastructure without consumers is dead code." Stopping work and asking the user which of three patch options to take before any edits: (a) auto-load encoder from the same .pth as the model, (b) require a separate `--vision-checkpoint` path on the engine, (c) bake the encoder weights into a sub-key of `model_state_dict` and let `load_state_dict(..., strict=False)` handle it. Until this is decided, F/Code-6 should be marked "training works, inference disconnected" rather than "complete." **Resolved with option (a).**


**Pass 155b:** SmolTalk2 split-resolution bug found in live run; fixed + retested.
**Pass 155:** D-4 OpenThoughts3 + D-11 SmolTalk2 fetchers shipped.

**Pass 156 audit — `Trainer.train_vision()` (F/Code-6):** Read-only audit of [enigma_engine/core/training.py L4785-5125](enigma_engine/core/training.py#L4785). Function name is `train_vision()`, not `train_vision_encoder()` (summary was wrong; verified by grep). Single GUI call site at [gui_forge_training.py L988](enigma_engine/gui/gui_forge_training.py#L988). Five behavioural tests in [tests/test_training.py L85-300](tests/test_training.py#L85) cover update / return-state / loss-decrease / callbacks / stop / no-projection-error. Author's-lens findings below.

*Strong points (ship-as-is for SFT-stage):* Validation gate, freeze/unfreeze logic correct (text frozen, projection always trainable, output/embedding/norm unfrozen for text loss, optional last-N text layers, encoder fully trainable, freeze_backbone re-applied AFTER blanket unfreeze — order matters and is right). Uses `_effective_warmup` (Pass 152). AMP scaler gated correctly. NaN/Inf guard + `max_loss` guard + early stopping + best-checkpoint + periodic checkpoint with `_cleanup_periodic_checkpoints` all present. Next-token slicing (`logits[:, n_patches:-1, :]` vs `text_tensor[:, 1:]`) verified correct: both end up length `N-1` for an `N`-token caption. Tests confirm gradients flow into both encoder weights and projection weights ([tests/test_training.py L116-118](tests/test_training.py#L116)).

*Gaps (ranked, not bugs but real omissions):*
1. **No `max_grad_accumulation` support** — confirmed by grep across the function body. Every other training method honors it (pre-train L2976/3226/3311/3320, DPO L4042, ReST L4311, audio L4542). Vision optimizer steps every sample. Files at scale will starve the optimizer of effective batch size on a 16 GB VRAM budget. *V-1 backlog row.* **[FIXED, Pass 156e]** — see top of file. Two behavioural tests added; suite +2.
2. **Eager preprocess to GPU RAM** — [L4906-4929](enigma_engine/core/training.py#L4906): `pairs.append((img_tensor, token_ids))` accumulates every preprocessed tensor *on `self.device`* before the loop starts. At 224²·3·4 B ≈ 600 KB per image, 100K pairs = 60 GB on GPU. OOMs RTX 5090 immediately at LLaVA-Pretrain scale (558K pairs). Must convert to lazy/iter pattern (preprocess in the step loop) or use disk-backed offset list per the learned principle. *V-2 backlog row.* **[FIXED, Pass 156f]** — see top of file.
3. **Batch size hardcoded to 1** — [L4889 comment](enigma_engine/core/training.py#L4889) `total_steps = len(data) * self.config.epochs  # 1 step per pair` is intentional but wastes 5090. Real LLaVA training uses batch 32-128. Note: enabling true batching needs padded text + attention mask plumbing through `forward_multimodal`. Non-trivial. *V-3 backlog row.*
4. ~~**No heartbeat write** — `gui_forge_training.py` does NOT call `training_heartbeat.json` (only `gui_forge_new_modes.py` does, [L183/L238](enigma_engine/gui/gui_forge_new_modes.py#L183)). OOM-kills on long vision runs leave no last-known-good state. *V-4 backlog row.*~~ **V-4 closed Pass 156d.** `_start_vision_training` now writes the same heartbeat as pre-training (stale check on entry, `_write_hb` closure inside thread, status calls at every lifecycle branch). Vision OS-OOM kills are now post-mortem-detectable.
5. ~~**No data collector** — there is no `collect_vision_data.py` to mirror `collect_pretraining_data.py` / `collect_finetuning_data.py`. F/Code-6 ships a *trainer* with no production data pipeline. Real options: liuhaotian/LLaVA-Pretrain (558K), Lin-Chen/ShareGPT4V-PT (1.2M), COCO captions (118K), LAION-5B subsets. *V-5 backlog row — needed before V-1/V-2 to give them work to do.*~~ **V-5 closed Pass 156c.** Shipped `collect_vision_data.py` with LLaVA-Pretrain fetcher; ShareGPT4V/COCO can be added with the same shape now that the framework is in place. Row schema is best-guess from the public dataset card and **must be live-validated on the first real run** (Pass 155 lesson).
6. ~~**No validation/eval split** — train loop only computes training loss. Other paths have `evaluate()` mid-training. Overfitting on small datasets is invisible. *V-6 backlog row, low priority.*~~ **V-6 + V-6b closed Pass 156g.** Backend: `train_vision()` accepts optional `val_data`; `_run_validation()` runs no-grad eval after each epoch, populates `state.validation_losses`, drives best-checkpoint + early-stopping when present. GUI: [gui_forge_training.py L1057](enigma_engine/gui/gui_forge_training.py#L1057) shuffle-splits `vision_data` according to `train_config.val_split` and passes the held-out slice as `val_data=`.
7. ~~**Single-token caption silently dropped** — [L5005](enigma_engine/core/training.py#L5005) `if min_len < 1: continue`. Caption like `"cat"` after BOS-stripping has `min_len=0` and is dropped. Edge case but should be logged once instead of silent-continue. *Tiny fix, V-7.*~~ **V-7 closed Pass 156d.** First drop logs a warning naming reason + epoch + step; remaining drops counted; end-of-run summary reports total. Pipeline is no longer silently lossy.

*Connections-that-don't-exist (the third author's-lens question):*
- Vision-only loss option (CLIP-style image-text contrastive on the projection head) is not in the trainer at all. Current loss is captioning CE only. For the LLaVA-style pipeline this is correct; for general image embedding (used by ImgGen / search-by-image) you'd want contrastive too. *Not a backlog row — this is approach-2 territory and we picked Approach 1 + 3.*
- The GUI save block at [gui_forge_training.py L995-1003](enigma_engine/gui/gui_forge_training.py#L995) emits `vision_encoder_state` and `vision_encoder_config` as separate top-level keys, but `Enigma.load_state_dict()` doesn't know about those keys — the GUI loader has to pull them apart manually. No load-side code visible in the audit window. *V-8 backlog row — verify load path before the first real run.*

*Verdict:* `train_vision()` is **ship-as-is for the SFT-stage smoke test** (small image-text pairs, 1-3 epochs, validates pipeline correctness). For LLaVA-scale pretraining, V-3 (true batching) is the remaining must-land item; it's optional if 1-step-per-pair is acceptable. V-1, V-2, V-4, V-5, V-6, V-6b, V-7, V-8 closed. Suite **2370/9**.

**Pass 155b live-validation patch (SmolTalk2):** First live Phase-1 smoke run on RTX 5090 caught two contract assumptions that the Pass 155 fake-`datasets` tests had silently blessed: (1) `split="train"` was hardcoded, but SmolTalk2 has **no** "train" split — each config (`SFT`, `Mid`, `Preference`) has 25-ish named splits like `smoltalk_smollm3_smol_magpie_ultra_no_think`, `OpenThoughts3_1.2M_think`, etc.; (2) the prior write-up confused configs and splits — `smol_magpie_ultra` is a *split-name fragment*, not a config. Patched [collect_finetuning_data.py](collect_finetuning_data.py) `collect_smoltalk2(max_samples, config, split=None)`: when `split is None`, calls `get_dataset_split_names(repo, config)` and concatenates rows from every split until `max_samples`; when explicit, uses it directly. New CLI flag `--smoltalk2-split NAME`. Test fake [tests/test_collect_finetuning_data.py](tests/test_collect_finetuning_data.py) `_install_fake_datasets()` now accepts `splits_by_path` and exposes `get_dataset_split_names`; +2 tests (`test_split_none_iterates_all_splits`, `test_explicit_split_used_directly`). Live re-run with `--smoltalk2-config SFT --smoltalk2-split smoltalk_smollm3_smol_magpie_ultra_no_think` produced 999 pairs / 2.0 MB. OpenThoughts3 live-validated separately: 1000 pairs / 57.9 MB, **all 1000 rows contain `<think>` tag** verbatim (`Select-String <think>` count = 1000). Suite **2342/9** green, ruff clean. Files on disk: [data/finetune/openthoughts3.jsonl](data/finetune/openthoughts3.jsonl), [data/finetune/smoltalk2.jsonl](data/finetune/smoltalk2.jsonl), [data/finetune/combined_finetune.jsonl](data/finetune/combined_finetune.jsonl) (1999 pairs).

**Pass 155b learned principle (logged in [AA code maker.md](AA%20code%20maker.md)):** Test fakes that ignore kwargs hide signature/contract bugs. When a real argument can be wrong (split name, config name, region), the fake should validate at least the shape of expected values, not blindly accept anything. Pass 155 fake `load_dataset(path, *args, **kwargs)` ignored `split=` entirely — the test "passed" against `split="train"` even though no real SmolTalk2 split is named that. Always read what the fake throws away; that's your blind spot.

**Pass 155 backlog row (kept for context):** D-4 OpenThoughts3 + D-11 SmolTalk2 fetchers.

**D-4 OpenThoughts3 fetcher (Pass 155):** New `collect_openthoughts3(max_samples)` in [collect_finetuning_data.py](collect_finetuning_data.py). Streams `open-thoughts/OpenThoughts3-1.2M` (Apache-2.0, 1.2M rows = 850K math + 250K code + 100K science). Extracts `(human, gpt)` turns from each row's `conversations` list. **Reasoning `<think>...</think>` tags preserved verbatim** — NO `_clean_text` whitespace collapse on the gpt value, so the QwQ-32B reasoning trace newlines survive and align with our special token IDs (`<think>=4`, `</think>=5`). Per-row schema verified Pass 139. Reuses existing `_dedup_pairs` (sha256 prefix). CLI: `python collect_finetuning_data.py --openthoughts3 100000`.

**D-11 SmolTalk2 fetcher (Pass 155 + 155b):** New `collect_smoltalk2(max_samples, config, split=None)` in [collect_finetuning_data.py](collect_finetuning_data.py). Streams `HuggingFaceTB/smoltalk2`. Real config names are `SFT`, `Mid`, `Preference` (verified live, not the speculative `smol_magpie_ultra` named in earlier drafts \u2014 that string is a *split fragment* inside the SFT config). Each config has ~25 named splits, none called "train". Pass 155b adds split auto-enumeration: when `--smoltalk2-split` is omitted, `get_dataset_split_names()` is called and rows from every split in the chosen config are concatenated until `max_samples`. On a missing config or split the HF loader raises ValueError naming the available choices; we log and return `[]` (per learned principle: detect on first attempt, no loop). ChatML schema (`messages: [{role, content}]`); first user/assistant pair extracted, optional system prepended (skips generic "You are a helpful assistant."). CLI: `python collect_finetuning_data.py --smoltalk2 100000 --smoltalk2-config SFT [--smoltalk2-split NAME]`. Both new fetchers wired into `--all`. 8 tests in [tests/test_collect_finetuning_data.py](tests/test_collect_finetuning_data.py) using fake-`datasets` injection \u2014 no network: 3 OpenThoughts3 (verbatim tags, missing-turn skip, max_samples cap) + 5 SmolTalk2 (extract pair, unknown-config error log, short/empty skip, split=None auto-enum, explicit split). Closes P1 backlog rows D-4 and D-11. Suite **2342/9** green, ruff clean.

**Pass 154:** Stage A helpers shipped Pass 153 are now live in the chat loop. In [enigma_engine/gui/gui_logic_chat.py](enigma_engine/gui/gui_logic_chat.py) `_send_message::_gen()`, after the model produces its first-pass reply (post-`extract_reasoning` / `strip_incomplete_think`, pre-`parse_commands`), a post-generation gate runs `should_retry_with_research(msg, resp)` when (a) web access is on, (b) no pre-generation `auto_research()` already ran, and (c) the user has not pressed STOP. On uncertainty score ≥ 0.55 the system fetches `auto_research(msg, max_results=3)`, appends it to `combined_prompt`, and re-invokes `engine.chat()` once with the augmented `system_prompt`. Reasoning is re-extracted from the retry output. Failures are swallowed (logged at DEBUG) so the original reply is never lost. Single retry only — no loop. 5 new structural tests in [tests/test_gui_logic_chat.py::TestAutoResearch2Wiring](tests/test_gui_logic_chat.py) verify the helper import, the `_web_access_on` gate, the `not web_research_ctx` skip, the `_stop_requested` short-circuit, and the try/except wrapper. Behavioral coverage of the helpers themselves stays in `test_core.TestAutoResearch`. Closes the AutoResearch-2 Stage A wiring follow-up noted in Pass 153. Suite **2334/9** green, ruff clean.

**Pass 153:** AutoResearch-2 Stage A (post-generation uncertainty gate) BUILD shipped.

**AutoResearch-2 Stage A:** New public API in [enigma_engine/core/auto_research.py](enigma_engine/core/auto_research.py) — `UncertaintyResult` dataclass (`score`, `reasons`), `score_uncertainty(query, response)`, and `should_retry_with_research(query, response, threshold=0.55, enabled=True)`. Signal-driven, deterministic, no RNG (per R-UNPREDICT-1 spec, Pass 146). Combines four signals: hedge phrases ("I'm not sure", "I think", "might be", capped at 0.6), refusal/apology ("I apologize", "I cannot answer", capped at 0.6), short-response anomaly (long query + tiny reply = +0.3), and question-echo (response repeats query with little new content = +0.3). Hard off-switch via `enabled=False`. Empty response → score 1.0 with reason `empty_response`. Stage B (inline `<search>` token in generation loop) remains future work — needs logits access. **Wiring into chat loop is a separate follow-up pass** ([gui_logic_chat.py L239](enigma_engine/gui/gui_logic_chat.py#L239) `_gen()` is the call site to update). 10 new tests in [tests/test_core.py::TestAutoResearch](tests/test_core.py) cover confident/hedge/refusal/empty/short-query/deterministic for `score_uncertainty`, plus skip/trigger/off-switch/threshold-configurable for `should_retry_with_research`. Closes AutoResearch-2 Stage A (P1 backlog row 11). Suite **2328/10** green, ruff clean.

**Pass 152:** Sched-2 (warmup cap for short SFT/DPO runs) BUILD shipped.

**Sched-2:** New module-level helper [`_effective_warmup(warmup_steps, total_steps)`](enigma_engine/core/training.py#L40) caps warmup at `total_steps // 5` (20% of run). Fixes the silent bug where the default `warmup_steps=100` produced 100% warmup on a 100-step SFT run and 50% on a 200-step run — i.e. no decay phase at all. All 5 scheduler-construction sites updated to use the helper: pre-train scheduler setup ([L2745](enigma_engine/core/training.py#L2745)), SWA cosine-restart boundary check ([L3368](enigma_engine/core/training.py#L3368) — uses `self._total_training_steps`), DPO scheduler ([L4023](enigma_engine/core/training.py#L4023)), vision-projection scheduler ([L4865](enigma_engine/core/training.py#L4865)), audio-projection scheduler ([L5274](enigma_engine/core/training.py#L5274)). Long runs unaffected (cap inactive when `warmup_steps <= total_steps // 5`); user-explicit large `warmup_steps` is still respected up to the 20% ceiling. Edge cases handled: `total_steps=0` returns `max(1, warmup_steps)` (no division-by-zero), `total_steps=1` returns `1`. 8 new tests in [tests/test_training.py::TestEffectiveWarmup](tests/test_training.py) cover short-cap (50/200 totals), cap-inactive (1k/10k totals), zero/one edge cases, explicit-respected, and explicit-excessive-capped. Closes P2 row #20 (Sched-2). Suite **2319/9** green, ruff clean.

**Pass 151:** Vision-1b (projection MLP upgrade) BUILD shipped.

**Vision-1b:** [enigma_engine/core/model.py L186-200](enigma_engine/core/model.py#L186) `self.vision_projection` upgraded from `nn.Linear(vision_hidden, dim, bias=False)` to LLaVA-1.5 reference impl `nn.Sequential(Linear(vision_hidden, dim, bias=True), GELU, Linear(dim, dim, bias=True))` per arxiv:2310.03744 §3.2 + HF `LlavaMultiModalProjector`. Forward call site at [model.py L705](enigma_engine/core/model.py#L705) untouched (callable on `Sequential` identically). `.parameters()` consumer at [training.py L4815](enigma_engine/core/training.py#L4815) safe. Updated [test_training.py L101/119](tests/test_training.py#L101) `.weight` → `[0].weight` (snapshot first Linear). State-dict keys changed: `vision_projection.weight` → `vision_projection.{0,2}.{weight,bias}` — acceptable break because no vision checkpoints exist yet (vision_hidden_size opt-in, Code-6 not built). Param cost at dim=2048: ~2.1M → ~6.3M (+4M, negligible vs 742M). 6 new tests in [tests/test_model_arch.py::TestVisionProjectionMLP](tests/test_model_arch.py): structure (Sequential of 3), GELU middle, dimensions, bias=True both Linears, forward shape preservation, end-to-end forward_multimodal call. Closes Decision Queue row D and P2 row #27 (Vision-1b half).

Tests: 6 new (TestVisionProjectionMLP). Suite **2311/9** green, ruff clean.

**Pass 150:** Eval-1 / D-10 (GSM8K reasoning benchmark) BUILD shipped.

**Eval-1 / D-10:** Three new public functions in [enigma_engine/core/training_evaluation.py](enigma_engine/core/training_evaluation.py): `parse_final_number(text)` (canonical `#### N` marker first, last-number fallback, comma-stripped, decimal/negative-safe), `load_gsm8k(path, n)` (offline-first JSONL loader, defaults to `data/gsm8k_test.jsonl`, raises `FileNotFoundError` with the exact one-time `datasets.load_dataset('openai/gsm8k', 'main', split='test').to_json(...)` recovery command — no silent network fetches at benchmark time), and `run_gsm8k_benchmark(engine, examples, num_shots=8, max_gen=256, temperature=0.0, on_progress=None)` (8-shot CoT prefix from Wei et al. 2022 Table 20, exact-match scoring with float tolerance, per-item `output[:200]` capture, returns `{total, correct, accuracy, results}`). CLI: `--benchmark` upgraded from `store_true` to `nargs="?"` with `choices=["coherence", "gsm8k"]` and `const="coherence"` (backward compatible — bare `--benchmark` still runs coherence). New flags `--benchmark-data PATH`, `--benchmark-limit N`, `--benchmark-shots N`. New `run_gsm8k_benchmark_cli()` dispatcher in [run.py](run.py) loads model, calls loader, streams progress. 14 tests in [tests/test_evaluation.py::TestGSM8KBenchmark](tests/test_evaluation.py) cover gold-format parse, comma/negative/decimal/empty/no-number, file-missing error message, JSONL read, n-cap, mock engine accuracy, unparseable output, float tolerance, empty input. Closes Decision Queue row C and P1 row #7.

Tests: 14 new (TestGSM8KBenchmark). Suite **2305/9** green, ruff clean.

**Pass 149:** Tok-2 + MTP-2b BUILDs shipped.

**Tok-2:** [enigma_engine/core/bpe_tokenizer.py L88](enigma_engine/core/bpe_tokenizer.py#L88) default `use_utf8_bytes` flipped `False`→`True`; [enigma_engine/core/advanced_tokenizer.py](enigma_engine/core/advanced_tokenizer.py) gained full byte-mode parity (init flag, `_text_to_bytes`/`_bytes_to_text` helpers, encode/decode branches, save/load persistence with legacy-False default). Rust `enigma_bpe` already had the flag end-to-end — no rebuild required. Closes Decision Queue row A and P0 row #2.

**MTP-2b:** [enigma_engine/core/model_presets.py L129](enigma_engine/core/model_presets.py#L129) default `n_predict_heads` flipped `2`→`0` and inverted comment fixed (paper conclusion was upside-down per Pass 140 finding). Saves ~33-49M params on every fresh model built from the default config. Existing `n_predict_heads > 0` guards in [model.py](enigma_engine/core/model.py) (L238, L496, L594) and [progressive_growing.py](enigma_engine/core/progressive_growing.py) (L219) handled the zero case correctly with no surgery. Pass 148 already designated EAGLE-2 as the inference-time speculative path — Medusa (the only consumer of the heads) is no longer planned. Closes P2 row #27b. Updated [test_research_upgrades.py L876](tests/test_research_upgrades.py#L876) to assert new spec + added `test_default_model_has_no_predict_heads`. Lowered `test_small_preset_range` floor 50M→30M in [test_core.py L1543](tests/test_core.py#L1543) (the floor was inflated by the MTP head bytes, not real model capacity).

Tests: 6 new/changed (Tok-2) + 2 new/changed (MTP-2b) + 1 spec-floor adjustment. Suite **2291/9** green, ruff clean.

**Pass 148 (currency check addendum):** Tool-verified the research currency of the long-form resolved items before Phase 0 kicks off. **Phase 1** fetched primary sources for 6 high-turnover domains (speculative decoding, vision encoders, image gen, video gen, 3D gen, reasoning teachers): 3 items marked **[SUPERSEDED]** with successor blocks inline (**Video-1** AnimateDiff → Wan 2.1 T2V-1.3B primary + LTX-Video 2B-distilled alt; **3D-1** Shap-E → Hunyuan3D-2 primary + TRELLIS-image-large alt; **Medusa** training recipe stays valid but EAGLE-2 arxiv:2406.16858 supersedes it as the inference-time speculative method); 2 upgrade notes (**Vision-1b** SigLIP 2 arxiv:2502.14786; **Approach 3 teacher** DeepSeek-R1-0528-Qwen3-8B distilled checkpoint alongside Qwen3-30B-A3B). **Phase 2** fetched primary sources for 4 more domains (text embeddings, ASR, audio generation, vision-LLM): added **Vision-2b** (LLaVA-NeXT → LLaVA-OneVision arxiv:2408.03326 with SigLIP SO400M + Qwen2, 0.5B variant fits 16 GB), **RAG-2b** (bge-small-en-v1.5 → Qwen3-Embedding-0.6B arxiv:2506.05176, MTEB multilingual 64.33, Apache-2.0), **Audio-5b** (MusicGen CC-BY-NC noted; Stable Audio Open Small arxiv:2505.08175 preferred for general SFX), and new **Audio-ASR-1** (Whisper → NVIDIA Parakeet-TDT-0.6B-v2 6.05 WER / RTFx 3386, ~16× faster). **D-20 FA4** updated to verified v4.0.0.beta10 status: SM120 fwd+bwd+varlen merged via PRs #2329/#2330/#2333, `pip install flash-attn-4` works on Linux/WSL, Windows wheels still TBD. Also fixed stale **Priority Index P3 numbering** (duplicate #28 → renumbered 37-45) and stale **audit counter** (135 → 148 passes). Doc-only, no code changes.
**Pass 148:** Two-round audit of the Pass 147 GUI Modernization Plan. All edits scoped to the planning section below. Round-1 fixes: removed gate/weight double-counting, dropped impossible install-size target, relocated service contract from `enigma_engine/api/` (already FastAPI) to new `enigma_engine/services/`, added baseline measurement step, added 5-day per-track time-box, added decision-ladder cases for ties and dual failures, anchored page inventory to filesystem reality, deduped Phase 4/5 overlap. Round-2 fixes (tool-verified): (1) "parallel" → sequential 10-working-day bake-off (solo builder); (2) added explicit `api/server.py` → `services/` migration sub-step in Phase 4 so the "single service contract" claim is real; (3) resolved PyInstaller contradiction (POC shortcut in Phase 1, packaging decision deferred to Phase 3c); (4) replaced contradictory cold-start/page-switch dual thresholds with primary-vs-stretch targets; (5) split GUI file list into 16 user-facing page modules vs 7 support modules (`widgets.py`, `themes.py`, `scanners.py`, `media.py`, `gui_logic*.py`); (6) baseline now measures G1/G2/G3 on CustomTkinter too so "beat baseline" is comparable; (7) confirmed Tauri sidecar URL (https://v2.tauri.app/develop/sidecar/) documents `externalBin` + `-$TARGET_TRIPLE` suffix + `shell:allow-execute` capability with pyinstaller as canonical Python bundling path; (8) added project-goal drift check ("personality from training, not the user" + "blackbox") to Phase 0d page inventory. No source code changes.
**Pass 147:** Planning-only GUI architecture research pass (no implementation). Goal constraints locked: best possible UX/architecture, fully private, local/offline-capable, black-box. Researched desktop and webview stacks and documented migration-ready decision gates. Key sources: Tauri security model and trust boundaries (https://v2.tauri.app/security/), Electron security checklist and threat model (https://www.electronjs.org/docs/latest/tutorial/security), Qt for Python deployment and API surface (https://doc.qt.io/qtforpython-6/), Tkinter threading/event-loop architecture (https://docs.python.org/3/library/tkinter.html), pywebview local wrapper model (https://pywebview.flowrl.com/guide/), local/self-host references for web UIs (Open WebUI README and offline mode notes: https://github.com/open-webui/open-webui, Gradio local-only defaults and sharing behavior: https://gradio.app/guides/sharing-your-app/, NiceGUI local server/native mode docs: https://nicegui.io/documentation, Flet desktop/web run modes: https://flet.dev/docs/). Output: execution-ready planning section added below.
**Pass 146:** Research-only continuation. Resolved R-UNPREDICT-1 with primary-source evidence and converted it to an implementation-ready design: uncertainty-gated retrieval and tool-use should be signal-driven (confidence/retrieval-quality), not keyword-only and not pure RNG. Evidence used: ReAct (arxiv:2210.03629), Toolformer (arxiv:2302.04761), Self-RAG (arxiv:2310.11511), and CRAG (arxiv:2401.15884). Also cleaned stale GRPO-2 research status in the post-training section (already shipped in Pass 142). No code changes.
**Pass 145:** Research-only continuation. Resolved all R-ARCH-4 items (4a/4b/4c): finalized sequential integration order, defined phase-gated benchmark suite, and added explicit fallback/skip policy so failed optional tracks do not stall core roadmap. Net result: Approach 4 is now fully specified for execution with hard stop conditions on core regressions and safe bypass for non-core failures. No code changes.
**Pass 144:** Research-only continuation. Resolved all R-ARCH-3 items (3a/3b/3c/3d) for the distillation POC track. Core conclusions: 742M-class students can learn substantial reasoning/coding behavior from larger teachers when supervision is rich (DeepSeek-R1 distill + Orca evidence), data should be staged by risk (pilot/useful/strong bands rather than one fixed size), benchmarking must be per-specialist plus routed end-to-end quality-cost frontier, and routing should start heuristic for fast POC but graduate to learned routing at defined misroute/fallback thresholds (RouteLLM evidence). No code changes.
**Pass 143:** Research-only continuation. Resolved R-ARCH-2b and R-ARCH-2c. Conclusion for 2b: use heterogeneous specialist sizing (capacity by task difficulty) while keeping a single interface contract (tokenizer/prompt/eval schema), based on quality-cost routing evidence from RouteLLM/FrugalGPT (arxiv:2406.18665, arxiv:2305.05176). Conclusion for 2c: avoid per-request hot-swaps; use resident pool + async warmup + queue policy, grounded in iteration-level scheduling (ORCA OSDI'22), KV-memory paging (PagedAttention arxiv:2309.06180), and split prefill/decode serving evidence (DeepSpeed-FastGen arxiv:2401.08671). No code changes.
**Pass 142:** GRPO-2 BUILD shipped + research sweep. New file [enigma_engine/core/reward_functions.py](enigma_engine/core/reward_functions.py) (rule-based `format_reward` / `math_reward` / `code_reward` / `reasoning_reward`). [gui_forge_new_modes.py L2229-L2260](enigma_engine/gui/gui_forge_new_modes.py#L2229-L2260) routes GRPO `reward_fn` to `reasoning_reward` and skips neural Phase 1 for GRPO (ReMax still uses neural path). Added [tests/test_reward_functions.py](tests/test_reward_functions.py) (4 tests). Closes Decision Queue row B and P1 rows 4 (GRPO-2 BUILD) + 5 (GRPO-4 block neural path). Research-only half of the pass resolved R-ARCH-1b (LLaVA CLIP-ViT-L/14@336 frozen + 2-layer MLP+GELU projector, 304M), R-ARCH-1c/1d (layered router + RouteLLM-style training), R-ARCH-2a/2d/2e (hybrid architecture, preference+LLM-judge labelling, cost/quality Pareto eval) against arxiv:2310.03744 + arxiv:2406.18665 + arxiv:2305.05176. Suite 2292/2 green, ruff clean.
**Pass 141:** Bookkeeping only. All [RESEARCH] items from the original backlog are closed (last batch resolved in Pass 140). Remaining backlog is code work: [BUILD] / [CONFIRMED GAP] items. Logged a Pass 141 Decision Queue below so the next code session can pick a target without re-triaging. No file changes outside this document's header + queue section. No code changes. Other 3 canonical docs (CODE_REVIEW.md, GUI_REFERENCE.md, AA code maker.md) stamped to Pass 141 with "no-op sync" notes.
**Pass 140:** Research-only continuation. Resolved 13 remaining [RESEARCH] items from the original backlog: **Attn-2** (Differential Transformer ICLR 2025 Oral arxiv:2410.05258 — keep enabled; manual-attention path is the cost), **MTP-2** (Meta arxiv:2404.19737 — sub-1B is ambiguous bracket; found inverted comment at [model_presets.py L129](enigma_engine/core/model_presets.py#L129) and untied 98M-param predict_heads at [model.py L237-240](enigma_engine/core/model.py#L237-L240)), **Tok-1** (Tao et al. NeurIPS 2024 arxiv:2407.13623 — optimal ~40K at 742M, 64K over-provisioned), **Inf-2** (three-tier attention dispatch — differential-attn default forces manual path), **Spec-1/2/3** (LoRA-per-specialist is the only serious 16 GB path; LM Eval Harness for benchmarks), **Grow-1/2** (train Chinchilla-optimal before grow; re-warmup not reset), **Medusa-1/2** (our joint-loss path matches Medusa-2; add λ warmup), **Router-2/3** (layered regex → embedding → LLM fallback), **Haptic-1** (no standard dataset/model exists — deprioritize). **New backlog items logged:** MTP-2a (reduce default `n_predict_heads` 2→1, weight-tie), MTP-2b (set to 0 if Medusa not planned), Medusa-1a (λ warmup from 0 → 0.2 over ~500 steps), Medusa-2a (measure acceptance rate via `--benchmark medusa-acceptance`). No code changes.
**Pass 139:** Cleanup + continuation research. Removed stale duplicate [RESEARCH]/[RESOLVED] entries for Arch-8, Attn-1, MTP-1, Sched-2, Distill-1/2 so each item has one canonical status line. Added code-verified Distill-2 note: Forge distillation currently generates teacher data at `temperature=0.8` in [enigma_engine/gui/gui_forge_new_modes.py](enigma_engine/gui/gui_forge_new_modes.py#L1471), which matches the recommended SeqKD range (0.7-1.0). Resolved Distill-4 with concrete evidence: Qwen3 reports ~10x GPU-hour savings from strong-to-weak distillation, and DeepSeek-R1 reports ~800K SFT samples (600K reasoning + 200K non-reasoning) for distilled students. Inf-4 status finalized as resolved with explicit KV-cache math (150MB at 4K, ~1.17GB at 32K). MTP-2 moved to PARTIAL: literature supports MTP overall, but no clean sub-1B/742M ablation was found; internal A/B recommended. Data-4 + Opt-3 finalized (HF datasets-server API confirms `<think>` tags in OpenThoughts3 raw rows; weight-tied `output.weight` never visited by `named_parameters(remove_duplicate=True)`). **Vision-1..4 resolved via LLaVA-1.5 paper (arxiv:2310.03744):** frozen CLIP-ViT-L/14 @ 336px → 2-layer MLP+GELU → LLM, 576 visual tokens, `openai/clip-vit-large-patch14-336` (1024-d), Stage 1 on `liuhaotian/LLaVA-Pretrain` (558K) then Stage 2 on `liuhaotian/LLaVA-Instruct-150K` + academic VQA (665K). **Code gap logged as Vision-1b:** our `self.vision_projection` is a single `nn.Linear` (LLaVA-1 era), should be 2-layer MLP+GELU for LLaVA-1.5 parity. **Inf-5 resolved and deferred to P3:** `flash_attn_with_kvcache` is the FA2 decode-tuned kernel (changelog v2.2), ~15-30% decode gain at our 742M/4K/batch-1 workload (not the 2× headline numbers which assume 8K+/larger models), and flash-attn install is fragile on RTX 5090 (Blackwell SM120 unverified) + Windows. Revisit if a stable Blackwell/Windows wheel ships with FA4. No code changes.
**Pass 138:** Architecture, MTP, and Distillation research pass. Resolved: Arch-8 (no LayerScale at 0.6B–3B — SmolLM3 + Qwen3 verified), Attn-1 (4:1 GQA validated at 3B via SmolLM3), MTP-1 (λ=0.3 from DeepSeek-V3 paper; our 0.5 is high — lower to 0.3 for next pretraining run), Sched-2 (max(100, steps//10) confirmed adequate), Distill-1 (SeqKD = correct approach, reverse KL is future upgrade per MiniLLM ICML 2023), Distill-2 (T=4–10 is Hinton soft-label only; for our SeqKD use T=0.7–1.0 for teacher generation). Data-4 still PARTIAL — OpenThoughts3 tag format needs raw dataset sample check. No code changes.
**Pass 137:** Data-1 through Data-5 researched via external sources (SmolLM3 blog, FineWeb arxiv 2406.17557, DCLM arxiv 2406.11794, OpenThoughts3 arxiv 2506.04178, FineMath arxiv 2502.02737). Five of six claims VERIFIED by audit; Data-4 downgraded to PARTIAL — the `<think>` tag format was assumed from SmolLM3's description, not confirmed in OpenThoughts3's own dataset card. Concrete Stage 1 mixing ratios now available for N-6 restart. One new finding: our MinHash threshold default is 0.8 but industry standard (FineWeb/DCLM) is 0.75 — logged as Data-5b, low priority. No code changes.
**Pass 136:** CRIT-6 closed — DCLM + FineMath + The Stack v2 wired into [collect_pretraining_data.py](collect_pretraining_data.py). No unit tests added (matches existing network-gated fetcher pattern — fandom/fineweb/c4). New learned principle: gated HF datasets need fail-fast auth detection (see AA code maker.md).

Pass-by-pass prose (Passes 110-135) lives in [information/history/PASS_HISTORY.md](information/history/PASS_HISTORY.md). Per-file bug-fix history lives in the file tables of [CODE_REVIEW.md](CODE_REVIEW.md). This document keeps only the current state + forward-looking work.

**Recommended path:** Approach 1 (Reasoning-First) + Approach 3 (Distillation POC) in parallel. See "Architecture Approaches" section below.

## Next Actions (top of backlog)

1. ~~**CRIT-6:** Collect DCLM + FineMath + The Stack v2 (D-1/D-2).~~ **Done.** `--dclm`, `--finemath`, `--code` flags added to [collect_pretraining_data.py](collect_pretraining_data.py). Run: `python collect_pretraining_data.py --dclm 15 --finemath 10 --code 10`
2. ~~**Data-1..5:** mixing-ratio research for N-6 restart.~~ **Done (Pass 137).** Stage 1 target mixture: **85% web / 12% code / 3% math** per SmolLM3 blog. Stage 3 cooldown: **63% web / 24% code / 13% math**. Use `<think>` XML tags for reasoning traces.
3. ~~**GRPO-2 BUILD:** Write `reward_functions.py` (math pass/fail + format check) before any reasoning GRPO training.~~ **Done (Pass 142).** New file [enigma_engine/core/reward_functions.py](enigma_engine/core/reward_functions.py); GRPO dispatcher wired at [gui_forge_new_modes.py L2258-L2260](enigma_engine/gui/gui_forge_new_modes.py#L2258-L2260).
4. **Resume N-6:** pre-training with reasoning data emphasis using mixture from Data-1..5.
5. **Approach 3 POC:** distill reasoning traces from Qwen3-30B-A3B into 742M (SFT on ~5K cold-start CoT examples). **Pass 148 currency note:** two additional teacher options verified current — (a) **Qwen3-32B** (arxiv:2505.09388, May 2025, thinking/non-thinking dual mode, YaRN 131K) as a same-family larger alternative to 30B-A3B; (b) **DeepSeek-R1-0528-Qwen3-8B** (May 2025) as a ready-made distilled 8B reasoning checkpoint — SoTA among open sub-10B reasoning models — usable as cold-start weights for our 742M student if we want to bypass teacher inference entirely. QwQ-32B is now older than Qwen3-32B-thinking and should not be used as teacher.
6. **Personality-5:** Run personality distillation + identity/roleplay separation build plan (R-PERSONALITY-1 research resolved).
7. **GUI-ARCH-1:** Complete GUI platform decision gate (PySide6 vs Tauri vs keep/refactor current) using the planning criteria below before any UI rewrite.

---

## GUI Modernization Planning (Private, Local, Black-Box)

**Non-negotiable constraints (from user):**
- Best architecture wins even if current GUI is replaced.
- Fully private and local/offline-capable operation.
- Black-box deployment posture (no cloud dependency, minimal data exposure surface).

**Option matrix (planning verdict):**

| Option | Fit | Strengths | Risks | Verdict | Primary source |
|---|---|---|---|---|---|
| **A. PySide6/Qt desktop rewrite** | **High** | Native desktop UX, mature widget ecosystem, Python-first integration, strong local/offline packaging path | Medium rewrite effort from current CustomTkinter pages | **Primary candidate** | Qt for Python docs (https://doc.qt.io/qtforpython-6/) — confidence: medium (broad landing doc, not a single pinned subsection) |
| **B. Tauri v2 + local backend** | **High** | Explicit trust boundaries, capability-scoped IPC, constrained frontend surface, modern UI possibilities | Highest migration complexity (Rust + web frontend + bridge redesign) | **Secondary candidate for maximum hardening** | Tauri v2 Security (https://v2.tauri.app/security/) — confidence: high (primary vendor security doc, version-pinned) |
| **C. Keep current CustomTkinter and refactor** | Medium | Lowest immediate churn, preserves existing behavior | Harder to achieve large UX leap; legacy complexity remains | **Fallback if migration risk is unacceptable** | Tkinter threading model (https://docs.python.org/3/library/tkinter.html#threading-model) — confidence: high (official Python stdlib reference) |
| **D. Electron rewrite** | Medium | Rich ecosystem and tooling | Larger attack surface and strict hardening burden per Electron security checklist | **Not preferred unless web stack requirement becomes dominant** | Electron Security (https://www.electronjs.org/docs/latest/tutorial/security) — confidence: high (primary vendor security doc) |
| **E. Local web UI frameworks (NiceGUI/Flet/Gradio/Open WebUI style)** | Medium | Fast iteration, browser-based flexibility, local hosting possible | Adds local HTTP surface by default; black-box posture requires additional controls | **Good as optional operator console, not primary secure desktop shell** | NiceGUI docs (https://nicegui.io/documentation), Flet docs (https://flet.dev/docs/), Gradio sharing guide (https://gradio.app/guides/sharing-your-app/), Open WebUI (https://github.com/open-webui/open-webui) — confidence: medium (vendor docs, generic landing) |

**Architecture implications from research (decision-relevant):**
- [high] Tauri enforces a core/frontend trust boundary via IPC and capability configuration; frontend only reaches system resources through explicitly exposed commands — Tauri v2 Security (https://v2.tauri.app/security/).
- [high] Electron can be secured, but the default power model requires continuous hardening discipline (context isolation, sandboxing, IPC sender validation, navigation limits, no untrusted remote content) — Electron Security (https://www.electronjs.org/docs/latest/tutorial/security).
- [high] Tkinter calls from any Python thread are dispatched via an event posted to the interpreter's event queue (thread-safe by design), but because the event loop is single-threaded, event handlers that do long work block every other event until they return; long-running tasks must run off the event handler (timer slicing or a worker thread) rather than inside a button callback — Python docs, Threading model (https://docs.python.org/3/library/tkinter.html#threading-model).
- [medium] Web-first local UIs (Gradio/NiceGUI/Flet/Open WebUI patterns) typically run a localhost server; black-box posture depends on strict bind/auth/network policy rather than framework default alone — vendor docs listed above; claim is inferred from combined reading, not a single pinned statement.

**Source confidence legend:** high = primary vendor/official docs directly stating the claim; medium = vendor docs supporting the claim but claim partly inferred; low = secondary/blog sources only (none used here).

**Decision gate (must pass before rewrite start):**
- **Security gate:** chosen stack can run fully offline, bind locally by default, and expose only explicit command/API surfaces.
- **UX gate:** can support Forge-scale multi-page workflows without regressions in responsiveness.
- **Migration gate:** incremental coexistence plan exists (old GUI + new shell during transition).
- **Packaging gate:** Windows-first packaging and update strategy is defined and testable.

**Execution plan (planning only — no code yet; user approval required before any step starts):**

**Hard gates (binary pass/fail, scored separately from the rubric):**
- **G1 — Offline-by-default:** packet capture during a 10-minute idle + one full FORGE workflow shows zero outbound connections other than loopback. Tool: `pktmon` (built into Windows 10/11) or Wireshark if installed. Fails → stack is eliminated regardless of other scores.
- **G2 — No remote update / telemetry by default:** updater disabled or pinned to local-only feed; no analytics SDK in the dependency tree.
- **G3 — UI does not freeze under model load:** during a synthetic 30-second training step running on a worker thread, the UI records no frame stall longer than baseline on the same workload (primary), with a stretch absolute target of < 100 ms. **Measured against the CustomTkinter baseline in Phase 0 too** so "beat baseline" is a real comparison, not an absolute guess. (R4-3: previous pure-100 ms gate was either vacuous or eliminated the incumbent too — primary/stretch pattern instead.)

**Phase 0 — Freeze scope, measure baseline, define contract skeleton (no UI framework code).**
- **GUI-ARCH-0-prep:** Create `information/gui/` folder (does not exist today — verified by directory listing). All Phase 0 docs land there.
- **GUI-ARCH-0a (decision doc):** Write `information/gui/ARCH_DECISION.md` capturing constraints (local / offline / black-box), non-goals (no cloud, no telemetry, no remote auto-update, no analytics SDKs), and current pain points in CustomTkinter (single-threaded Tcl event loop: handlers that do long work block every other event until they return — see https://docs.python.org/3/library/tkinter.html#threading-model — hand-rolled theming, per-widget rebuild cost).
- **GUI-ARCH-0b (baseline):** Measure the **current CustomTkinter GUI** against every rubric metric below and record the numbers in `information/gui/BASELINE.md`. This anchors all POC targets to "beat what we already have", not to arbitrary guesses.
- **GUI-ARCH-0c (service contract skeleton — NOT full migration):** Create `enigma_engine/services/` (new module; does not exist today — `enigma_engine/api/` is already owned by HTTP `server.py` and must not be reused). Populate with an interface-only skeleton covering the 5–10 most-called entry points the current GUI reaches into `core/*` (identify via `grep "from enigma_engine.core" enigma_engine/gui/` — verified this pass, 30+ call sites exist across `desktop.py`, `gui_cmd_page.py`, `gui_forge.py`, `gui_forge_adaptive.py`, `gui_docs_page.py`, and others). The skeleton delegates to `core/*` internally so behavior is unchanged. **No GUI callers are migrated in Phase 0.** Purpose of the skeleton: give Track A / Track B POCs a concrete import surface to target, and make the full per-page migration in Phase 4 incremental instead of a big-bang rewrite.
- **GUI-ARCH-0d (page inventory):** Enumerate actual GUI surfaces from `enigma_engine/gui/` filesystem (verified this pass). Split into two lists because they have different migration risk:
  - **User-facing pages (16 modules — each gets a classification row):** `desktop.py`, `gui_cmd_page.py`, `gui_docs_page.py`, `gui_forge.py`, `gui_forge_adaptive.py`, `gui_forge_advanced.py`, `gui_forge_models.py`, `gui_forge_new_modes.py`, `gui_forge_queue.py`, `gui_forge_tools.py`, `gui_forge_training.py`, `gui_mods.py`, `gui_mod_page.py`, `gui_pages.py`, `gui_pages_config.py`, `gui_pages_forge.py`. For each: name, entry function, current complexity (LOC + widget count), classification (must-have v1 / v2 / drop).
  - **Support modules (7 — migrate as infrastructure, not per-page):** `widgets.py`, `themes.py`, `scanners.py`, `media.py`, `gui_logic.py`, `gui_logic_chat.py`, `gui_logic_media.py`. These are not pages and do not get classification rows; they get a single "port strategy" note each (direct port / rewrite / drop).
  - **Project-goal drift check (binary pass/fail):** for every proposed v1 page, confirm the page does not expose user controls that conflict with the core project constraints "personality from training, not the user" and "blackbox". Any widget that lets the user hand-author the AI's personality, identity, or internal state is flagged for removal or gating before cutover; any widget that exposes raw model internals beyond what is needed for training/debugging is flagged the same way. Output: `information/gui/PAGE_INVENTORY.md` with both lists plus a drift-check appendix.
- **Exit criteria:** `information/gui/` folder exists, four docs + `enigma_engine/services/` skeleton merged and reviewed, drift-check appendix empty or every flagged widget has a disposition, test suite green, no GUI behavior change. No framework picked yet.

**Phase 1 — Two-track bake-off POC (sequential, time-boxed).**
- Rationale: docs alone cannot decide A vs B. One complex page built in each candidate stack is the cheapest honest signal.
- **Time-box:** 5 working days per track, **sequential not parallel** (single builder). Total bake-off budget is 10 working days plus measurement and writeup. If a track cannot produce a running **CONFIG-page prototype** calling the Phase 0 service skeleton in 5 days, it is dropped. (R4-1: bake-off page changed from FORGE to CONFIG — FORGE has 12 modes / 6 collapsible tool sections / paned layout, so making it the primary POC scope means tracks die on page complexity, not on stack fit. FORGE becomes an in-budget stretch goal that scores bonus on the "Theming & layout headroom" metric if reached.) Track B (Tauri) is genuinely harder to stand up in 5 days because of Rust + sidecar + capabilities wiring; the time-box may eliminate it on effort alone — that is an acceptable signal, not a bug. Build Track A first (closer to current Python stack), then Track B.
- **GUI-ARCH-1a (Track A — PySide6/Qt):** Throwaway CONFIG-page prototype in PySide6 calling the Phase 0 `enigma_engine/services/` skeleton where covered, and `enigma_engine/core/*` directly where thin. The POC is explicitly allowed to import `core/*` — it is throwaway code. Stretch goal: also stand up FORGE within the 5-day budget; reaching it scores bonus on the "Theming & layout headroom" metric. Package with PyInstaller **for Phase 1 measurement only** (this is a POC shortcut; the final Phase 3c tradeoff still chooses PyInstaller vs Nuitka). Source: https://doc.qt.io/qtforpython-6/.
- **GUI-ARCH-1b (Track B — Tauri v2 + local Python backend as sidecar):** Same CONFIG-page prototype in Tauri v2 with a minimal frontend (vanilla HTML/CSS/JS; no heavy JS framework for the POC) and a Python backend bundled as a sidecar binary (Tauri's documented pattern for Python — https://v2.tauri.app/develop/sidecar/ — explicitly mentions pyinstaller-bundled Python CLI/API servers as the supported use case). The sidecar re-exports the same service skeleton over loopback with an auth token. Capabilities configured to the exact command whitelist per https://v2.tauri.app/security/. FORGE is the same in-budget stretch goal as Track A.
- **Scoring rubric (filled in at end of phase, each track scored vs baseline from Phase 0):**

  | Metric | Weight | Target |
  |---|---|---|
  | Cold start (shell ready to accept input) | 15% | primary: ≤ baseline; stretch: < 2 s absolute |
  | Page-switch latency (to CONFIG; FORGE if reached) | 15% | primary: ≤ baseline; stretch: < 200 ms absolute |
  | Idle RAM, shell process only (excluding torch + model weights) | 15% | primary: ≤ baseline × 1.5; stretch: ≤ 300 MB absolute |
  | Packaged install size, **shell + UI runtime only** (explicitly excludes torch/CUDA/model weights, which live alongside the app in a shared folder) | 10% | primary: ≤ baseline × 1.5; stretch: ≤ 200 MB compressed absolute |
  | Dev velocity (hours to parity on one page) — **soft signal only; familiarity bias disclosed** | 10% | lower is better |
  | Packaging/update story on Windows (reproducible build, no remote calls) | 15% | documented + reproducible |
  | Theming & layout headroom for future pages | 10% | subjective, recorded with examples |
  | **User preference (side-by-side use of both prototypes)** | 10% | user picks after hands-on |

  Weights sum to 100%. Gates G1/G2/G3 are separate and binary.
- **Exit criteria:** Both POCs either pass all three gates or are eliminated. Rubric filled in. Baseline beaten or reason-why-not documented. Recommendation written in `information/gui/BAKE_OFF_RESULT.md`.

**Phase 2 — Decision and contingency.**
Decision rule, applied in order:
1. Any track failing a gate (G1/G2/G3) is eliminated regardless of score.
2. Among remaining tracks, highest weighted score wins.
3. If scores are within 10 points, user preference (metric row 8) is the tiebreaker.
4. If **only one** of A/B remains, it wins provided it also beats the Phase 0 baseline on at least Cold start, Page-switch, and Idle RAM. If it does not beat baseline on those three, fall back to Option C.
5. If **neither** A nor B remains, fall back to Option C (keep CustomTkinter + refactor) but still land the `enigma_engine/services/` contract — it is not wasted effort.
6. **Wall-clock abort:** if Phase 1 exceeds **15 working days** (start of Track A POC to filled rubric), stop. Default to Option C regardless of rubric state. (R4-5: prevents the bake-off from drifting into a multi-month effort — the 10-day time-box covers normal execution; +5 day buffer covers slip; past that the project has lost more than the rewrite is worth.)
- **Exit criteria:** Single chosen stack + written rationale referencing which rule fired + contingency logged.

**Phase 3 — Migration blueprint (still no mass rewrite).**
- **GUI-ARCH-3a (coexistence):** Legacy CustomTkinter stays shippable. `Launch Enigma.bat` reads a `gui_mode` key from `data/gui_settings.json` and dispatches on its value: `legacy` (default during Phase 3) launches the CustomTkinter shell, `next` launches the new shell. No hardcoded `--gui` / `--gui-next` flag in the .bat; the flag still exists on `run.py` for direct CLI use, but the .bat picks one mode based on the settings file so users get a single entry point during coexistence. (R4-6: the .bat is the real Windows entry point — routing through `gui_settings.json` keeps the user-visible launcher stable across the cutover.) After Phase 4 final cutover, default flips to `next`.
- **GUI-ARCH-3b (cutover order):** Driven by PAGE_INVENTORY. Low-risk pages first (Logs, Diagnostics, Config), Chat mid, FORGE last. Each page has an acceptance checklist (feature parity vs legacy + no regression in existing pytest suite).
- **GUI-ARCH-3c (packaging):** Windows-first installer, reproducible build from source, updater off or local-only. For Track A: decide PyInstaller vs Nuitka with a written tradeoff note (neither is in the project today). For Track B: Tauri bundler config reviewed for any outbound defaults.
- **GUI-ARCH-3d (rollback):** Every page cutover lands behind a feature flag in `data/gui_settings.json` so any page can revert independently.
- **Exit criteria:** Blueprint doc committed.

**Phase 4 — Incremental cutover + service-contract finish + final collapse.**
- **GUI-ARCH-4a..n (per GUI page):** One page per cutover PR. Each PR does three things together: (1) migrate the page to the chosen stack, (2) replace that page's direct `from enigma_engine.core.*` imports with calls through `enigma_engine/services/` (extending the skeleton as needed), (3) port the page's tests. Each PR runs `ruff check enigma_engine/ tests/` + `python -m pytest tests/ -v`, re-runs gates G1/G2/G3 on the merged shell, user smoke-tests the new page, and **flips the per-page rollback flag in `data/gui_settings.json` off, relaunches, verifies the legacy page still renders and its own test suite still passes, then flips the flag back on and re-verifies the new page. Untested rollback = no rollback.** (R4-4)
- **GUI-ARCH-4-api (API server migration):** Once the service skeleton is substantial (roughly half the GUI pages migrated), port `enigma_engine/api/server.py` to call `enigma_engine/services/` instead of `core/*` directly. This is NOT a GUI change but it is part of the same contract cleanup and must happen before legacy removal — otherwise `server.py` remains the last direct `core/*` consumer and the “single service contract” claim is false. Tests: existing API tests must still pass with no behavior change.
- **GUI-ARCH-4-final:** When all v1 pages are migrated AND `api/server.py` is on the service contract, `--gui-next` becomes default, legacy `--gui` kept for one release as emergency rollback, then legacy code path + feature flags removed, `GUI_REFERENCE.md` + `Launch Enigma.bat` updated, final decision archived in `information/gui/ARCH_FINAL.md`. At this point no GUI or API code imports `enigma_engine.core.*` directly; everything routes through `enigma_engine.services`.
- **Exit criteria:** Legacy GUI removed, all v1 pages on the new shell, `api/server.py` migrated, full test suite green, gates re-verified on shipped build.

**Risks logged up-front:**
- Track B (Tauri) carries Rust + JS + Python tri-language complexity. Mitigation: frontend stays minimal (no heavy JS framework), IPC surface bound to the same service contract Track A uses, time-box enforced.
- Track A (PySide6) with PyInstaller can produce large binaries and has known issues with dynamic imports. Mitigation: pin deps, validate bundle contents, consider Nuitka as alternate (decided in Phase 3c).
- Familiarity bias in dev-velocity metric: whichever stack the builder knows better will score higher on velocity for reasons unrelated to the stack's suitability. Mitigation: weight is only 10% and metric is flagged "soft signal".
- Option E (local web UIs) is excluded from primary candidacy because it requires additional bind/auth/network policy work to satisfy the black-box posture. It can still ship later as an opt-in operator console, never as the default shell.

**Not doing (explicit non-goals):**
- No cloud sync, no telemetry, no remote update server, no analytics SDKs, no share-mode web UI as the default surface.
- No rewrite before the bake-off POC is complete and reviewed.
- No partial cutover without a rollback flag.
- No changes to `enigma_engine/core/*` driven by GUI work — the service contract adapts to core, not the other way around.

**Honest total schedule estimate (solo builder, no parallelism):**

| Phase | Wall-clock estimate | Notes |
|---|---|---|
| Phase 0 (prep + baseline + services skeleton + page inventory) | 1–2 weeks | Docs + measurements + stub module. No framework commitment. |
| Phase 1 (Track A POC, then Track B POC) | ~2 weeks sequential (5 working days each + measurement/writeup) | Track B may be dropped on 5-day time-box; then Phase 1 is ~1 week. |
| Phase 2 (decision) | 1–2 days | Fill rubric, apply decision ladder, pick stack. |
| Phase 3 (migration blueprint + packaging + rollback) | ~1 week | Written plan only, no mass rewrite. |
| Phase 4 (incremental per-page cutover + api/server.py migration + legacy removal) | **months** — one PR per page, tests green each time | Real migration work. Pace set by per-page risk, not by calendar. |

Total: Phase 0–3 is roughly 5–6 weeks of dedicated work before legacy code starts being retired in Phase 4. Numbers assume no other high-priority item pulls focus. If N-6/N-9/N-10 training work is active, multiply by 2x and treat GUI as background.

**Round-4 open items — ALL CLOSED Pass 156m (doc-only fixes applied to plan above):**

| # | Item | Fix | Severity | Status |
|---|---|---|---|---|
| R4-1 | Bake-off page is FORGE — 12 modes, 6 collapsible tool sections, paned layout. If POC can't stand it up in 5 days, track dies on scope not fit. | Change Phase 1 POC target to a **mid-complexity page (CONFIG)** as the primary bake-off scope. FORGE becomes a stretch-goal inside the 5 days; reaching it scores bonus on the "Theming & layout headroom" metric. | **Medium — do first.** One sentence, protects the whole bake-off. | **DONE Pass 156m.** |
| R4-2 | Rubric "Idle RAM ≤ 300 MB" and "Install size ≤ 200 MB compressed" are absolute targets with no baseline measurement — same bug class as round 3 caught on cold-start/page-switch. | Rewrite both rows to the primary/stretch pattern: "primary: ≤ baseline × 1.5; stretch: absolute N". Numbers fill in from Phase 0b baseline. | Medium | **DONE Pass 156m.** |
| R4-3 | Gate G3 "100 ms frame stall" is arbitrary. CustomTkinter baseline may already fail 100 ms during model load → gate is either vacuous or eliminates the incumbent too. | Re-scope G3 as "no worse than baseline on the same workload" plus stretch absolute 100 ms. Same primary/stretch pattern as R4-2. | Low — same class as R4-2. | **DONE Pass 156m.** |
| R4-4 | Phase 4 per-page acceptance has feature-flag rollback **defined** but no **test** that rollback actually works. Untested rollback = no rollback. | Add one bullet to the per-page checklist: "Flip flag off, relaunch, verify legacy page renders + passes its own test suite. Flip flag back on and re-verify." | Medium | **DONE Pass 156m.** |
| R4-5 | Decision ladder has no wall-clock abort. If Phase 1 bogs down to 3+ weeks, plan has no "stop and fall back to Option C" trigger. | Add decision rule 6: "If Phase 1 exceeds 15 working days, stop. Default to Option C regardless of rubric state." | Low | **DONE Pass 156m.** |
| R4-6 | `Launch Enigma.bat` is the real Windows entry point but Phase 3a only mentions `--gui` / `--gui-next` flags. During coexistence the .bat must pick one. | Phase 3a: the .bat reads a `gui_mode` key from `data/gui_settings.json` (default `legacy` during Phase 3, `next` after Phase 4 cutover is complete). No hardcoded flag in the .bat. | Low | **DONE Pass 156m.** |
| R4-7 | After R4-1..R4-6 land, plan is execution-ready. | Stamp the close across the canonical docs with a one-line note: "Plan round-4 audit closed. Phase 0 unblocked." | Low — bookkeeping. | **DONE Pass 156m.** |

After R4-7: Phase 0 is the real next action. No more audits.

**First action when the user says "go":** Phase 0 only — produce the four docs (ARCH_DECISION, BASELINE, PAGE_INVENTORY) and the `enigma_engine/services/` contract stub. No UI framework work until those are reviewed.

---

## Pass 141 Decision Queue (next code session picks one)

All candidates below are ready to implement — spec is written, code locations are known, test pattern is clear. User to pick which lands first. No work started yet. Candidates are ordered by blocking-power on the N-6 → N-9 → N-10 training pipeline, not by size.

| # | Candidate | Scope | Blocks | Test Pattern | Est. Size |
|---|-----------|-------|--------|--------------|-----------|
| ~~A~~ | ~~**Tok-2** byte-level BPE fallback~~ | ~~[enigma_engine/core/bpe_tokenizer.py](enigma_engine/core/bpe_tokenizer.py) + [enigma_engine/core/advanced_tokenizer.py](enigma_engine/core/advanced_tokenizer.py)~~ | **DONE (Pass 149).** Default flipped on; AdvancedBPETokenizer gained byte-mode parity; Rust already correct. | — | — |
| ~~B~~ | ~~**GRPO-2** `reward_functions.py`~~ | ~~New file: `enigma_engine/core/reward_functions.py`. Rule-based rewards: math pass/fail (numeric parse + compare), format check (looks-for-`<think>` + answer delimiter), code exec (subprocess sandbox). Wire into [gui_forge_new_modes.py L2250](enigma_engine/gui/gui_forge_new_modes.py#L2250) replacing `RewardModel.score()` for reasoning mode.~~ | **DONE (Pass 142).** Shipped + 4 tests green. | — | — |
| ~~C~~ | ~~**Eval-1** GSM8K benchmark~~ | ~~Add `run_gsm8k_benchmark()` to [enigma_engine/core/training_evaluation.py](enigma_engine/core/training_evaluation.py). Load HF `openai/gsm8k` (already cacheable offline), few-shot prompt, numeric answer parse from final line. Expose via `--benchmark gsm8k` in [run.py](run.py).~~ | **DONE (Pass 150).** `parse_final_number` + `load_gsm8k` + `run_gsm8k_benchmark` shipped, 8-shot Wei et al. CoT prefix, exact-match scoring, offline-first JSONL loader with clear download recovery message. CLI: `python run.py --benchmark gsm8k --model models/<name>.pth`. 14 tests green. | — | — |
| ~~D~~ | ~~**Vision-1b** projection upgrade~~ | ~~[enigma_engine/core/model.py](enigma_engine/core/model.py) `self.vision_projection = nn.Linear(...)` → `nn.Sequential(nn.Linear, nn.GELU, nn.Linear)` per LLaVA-1.5 (arxiv:2310.03744). Keep input/output dims identical.~~ | **DONE (Pass 151).** `nn.Sequential(Linear(vh,d,bias=True), GELU, Linear(d,d,bias=True))` per HF `LlavaMultiModalProjector`. State-dict keys changed; safe (no vision checkpoints exist). 6 tests. | — | — |
| ~~E~~ | ~~**MTP-2a** reduce `n_predict_heads` 2→1 + weight-tie~~ | ~~[enigma_engine/core/model_presets.py L129](enigma_engine/core/model_presets.py#L129)~~ | **SUPERSEDED — took MTP-2b path Pass 149.** Default flipped to 0 entirely (Medusa retired in favor of EAGLE-2). | — | — |
| F | **Code-6** FORGE vision-projection training mode | New FORGE mode wired in [enigma_engine/gui/gui_forge_new_modes.py](enigma_engine/gui/gui_forge_new_modes.py) + [enigma_engine/gui/gui_pages_forge.py](enigma_engine/gui/gui_pages_forge.py). Freeze transformer, train only `vision_projection` + CLIP-side adapter on LLaVA-Pretrain (558K) → LLaVA-Instruct-150K (665K). | Vision specialist pipeline. Projection is wired but untrainable. | Fail test: after 1 training step, `vision_projection.weight.grad` is not None and transformer weights `.grad` are None. | Large — new mode + data loader + GUI radio-card entry. Depends on D (Vision-1b) for best parity. |
| G | **Personality-5** personality injection wiring | [enigma_engine/gui/gui_forge_new_modes.py L1259](enigma_engine/gui/gui_forge_new_modes.py#L1259) — category exists, data generation exists, but no consistency loss or per-profile regularizer. Gate behind user opt-in (profile-scoped, not global). | R-PERSONALITY-1. | Fail test: two generations with same profile + temperature=0 produce consistent first-person pronoun + stated-value alignment. | Medium — needs consistency metric before loss wiring. |
| H | ~~**EWC-1** wire `core/ewc.py` into FORGE SFT/dialogue~~ | **CLOSED Pass 156i3 — WONTFIX (superseded by LoRA-per-specialist path).** [ewc.py](enigma_engine/core/ewc.py) library stays for rare-case future use; LoRA freezes base weights so forgetting is physically impossible. See **LoRA-1** (P2) for the replacement work. | — | — | — |
| ~~DET-2~~ | ~~**DET-2** full bitwise GPU reproducibility~~ | **DONE Pass 156i3.** `set_training_seed(seed, deterministic=False)` gained opt-in `deterministic` kwarg (sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` + `torch.use_deterministic_algorithms(True, warn_only=True)`). `TrainingConfig.deterministic` field default False; all 8 train_* siblings forward the flag. `warn_only=True` keeps MoE `index_add_` from crashing. 4 new tests; suite 2379/9. **Out of scope:** DataLoader worker_init_fn (Enigma single-threaded), `Trainer.__init__` DRY refactor (separate design row). | — | — | — |

**Tie-breaker guidance:** A (Tok-2) has the broadest downstream blast radius — every retrain from here on benefits. E (MTP-2a) is the smallest win and frees 49-98M params immediately. B (GRPO-2) is the hard prerequisite for any reasoning RL work.

---

## Priority Index (most → least important)

Full backlog sorted by urgency. Each item links to the detailed entry further down.
Items already done or irrelevant are omitted. Research-only items (no implementation
yet possible) are grouped at the bottom of P2/P3.

### 🔴 P0 — CRITICAL (blocks next pre-training restart)

| # | Item | Why P0 |
|---|------|--------|
| 1 | ~~**CRIT-6 / D-1 / D-2** — Collect DCLM + FineMath + The Stack v2~~ | **DONE (Pass 136).** `--dclm`, `--finemath`, `--code` flags in [collect_pretraining_data.py](collect_pretraining_data.py). |
| ~~2~~ | ~~**Tok-2** — Add byte-level fallback to BPE tokenizer~~ | **DONE (Pass 149).** Byte-mode default ON in `BPETokenizer` + full parity in `AdvancedBPETokenizer`. Rust already supported. CJK/emoji/Arabic now roundtrip without `<unk>`. |
| 3 | **D-4** — Reasoning mid-training phase (OpenThoughts3) | Gap between "follows instructions" and "actually reasons." Must happen between N-6 and N-9 for Approach 1 to work. |

### 🟠 P1 — HIGH (before N-9 SFT / alignment)

| # | Item | Why P1 |
|---|------|--------|
| ~~4~~ | ~~**GRPO-2 BUILD** — Write `reward_functions.py` (math accuracy + format + code exec + LLM judge)~~ | **DONE (Pass 142).** [reward_functions.py](enigma_engine/core/reward_functions.py) shipped. |
| ~~5~~ | ~~**GRPO-4** — Block neural reward path for reasoning; route to rule-based reward~~ | **DONE (Pass 142).** GRPO branch in [gui_forge_new_modes.py L2229-L2260](enigma_engine/gui/gui_forge_new_modes.py#L2229-L2260) skips neural Phase 1 and uses `reasoning_reward`. |
| 6 | **Approach 3 POC** — Distill reasoning traces from Qwen3-30B-A3B (N-9.5a/b/c/d) | Proof-of-concept for whether multi-model architecture is viable. 3-4 week experiment. Informs Approach 1 vs Approach 2 decision. |
| ~~7~~ | ~~**Eval-1 / D-10** — Implement GSM8K benchmark in `training_evaluation.py`~~ | **DONE (Pass 150).** `run_gsm8k_benchmark()` + `parse_final_number()` + `load_gsm8k()` shipped. CLI: `--benchmark gsm8k`. Offline JSONL loader (no network at benchmark time). 14 tests in [tests/test_evaluation.py](tests/test_evaluation.py). |
| ~~8~~ | ~~**D-11** — Add SmolTalk2 SFT data (reasoning traces + tool calling)~~ | **DONE Pass 156i8 (consumer wiring).** Pass 155+155b shipped the fetcher; Pass 156i8 closes the consumer-side gap by emitting `data/finetune/combined_finetune.txt` in canonical `User: \n\nAssistant: ` format alongside the combined JSONL. The existing SFT plain-text reader consumes it with zero training-side change. 1,999-pair corpus on disk (~60 MB). See **D-11b** for FORGE picker UX follow-up. |
| ~~8b~~ | ~~**D-11b** — FORGE file picker default for finetune data~~ | **DONE Pass 156i9.** New `scanners._pick_default_training_file()` helper prefers `data/finetune/combined_finetune.txt` over `data/training.txt`. Wired into FORGE Basic page `train_data_var` init. 4 new tests including adversarial-ordering. |
| ~~9~~ | ~~**Code-6** — FORGE training mode for vision projection (frozen transformer)~~ | **DONE (closed Pass 156q, work landed across Pass 151-156).** `Image` mode card → `_start_vision_training` → `Trainer.train_vision()` with LLaVA Stage-1 defaults (`freeze_backbone=True`, `unfreeze_text_layers=0` — projection-only). Sub-rows V-1 through V-8 all closed: Vision-1b 2-layer MLP+GELU projection (Pass 151), V-4 OOM/crash heuristics, V-5 `collect_vision_data.py` LLaVA-Pretrain fetcher (Pass 156c), V-7 abort-summary, V-8 vision-encoder load path in inference (Pass 156b). Stage-2 unfreeze-last-N-text-layers knob is in the trainer but not yet exposed in the GUI — opened as **Code-6b** below. |
| ~~9b~~ | ~~**Code-6b** — Expose `unfreeze_text_layers` in FORGE Image mode~~ | **DONE Pass 156r.** New numeric input under the vision-encoder-size dropdown (default 0 = LLaVA Stage-1 projection-only). `_start_vision_training` reads with try/except validation (negative → 0, >64 → 64, bad input → 0, all warned), forwards as `trainer.train_vision(unfreeze_text_layers=N, ...)`, logs in summary as `"N text layers"` or `"projection only (Stage-1)"`. Two structural tests gate the var presence + literal kwarg forward. |
| 10 | **Personality-5 cluster** — Implement personality-in-weights plan + identity/roleplay separation | **PARTIAL.** Personality-3 boundary fix DONE Pass 156y (`AIProfile.personality` reframed as roleplay-only, `is_roleplay()` signal, `assistant` base cleaned). Personality-3b (canonical role-template cleanup) + Personality-4 (identity-vs-roleplay design call + first end-to-end consumer) DONE Pass 156z (3 disk JSONs + DEFAULT_PROFILES cleaned, `apply_profile_to_engine` logs roleplay branch, design call: empty personality = base/task overlay, populated = roleplay character). Open: Personality-3b for legacy user profile `not_for_you_hahaha.json` (deferred — user-driven migration), Personality-5 BUILD (operational — run FORGE distillation), Row G (consistency loss). |
| 11 | **AutoResearch-2** — Self-initiated research (`<search>` tag OR uncertainty post-check) | Model currently cannot say "I don't know, let me look it up." Linked to GRPO-4. |

### 🟡 P2 — MEDIUM (standard backlog, no blocking order)

| # | Item | Notes |
|---|------|-------|
| 12 | ~~**D-9** — APO alignment mode (15-line DPO loss variant)~~ | **DONE Pass 156j (library).** `_apo_zero_loss` static + `_resolve_preference_loss` registry + `loss_type="dpo"\|"apo_zero"` kwarg on `train_dpo`. 8 behavioural tests including chosen-rejected-independence (the key APO property) and dispatch-actually-routes. See **D-9b** for FORGE GUI surface. |
| 12b | ~~**D-9b** — FORGE radio card + dispatcher for APO-zero alignment~~ | **DONE Pass 156k.** APO radio card added to alignment row alongside GRPO/ReMax/SimPO/ORPO. `_start_apo_training` thin wrapper delegates to refactored `_start_dpo_training(loss_type="apo_zero")` which forwards the kwarg to `trainer.train_dpo`. Status bar + logs + `_save_training_run` label parametrized via `algo_label`. 5 structural GUI tests; behavioural routing proven by Pass 156j's sentinel-mock dispatch test. |
| 13 | **D-19** — Strong-to-weak distillation from Qwen3-30B before GRPO | 10× cheaper than GRPO at small scale. Revisit at N-9. |
| 14 | **N-14** — Dense semantic memory (FAISS) to replace TF-IDF RAG | Prep for vision+retrieval. |
| 15 | ~~**EWC-1** — Wire `core/ewc.py` into FORGE SFT/dialogue training path~~ | **CLOSED Pass 156i3 — WONTFIX (superseded by LoRA-per-specialist).** Library kept for rare-case full-weight specialization; LoRA path makes forgetting impossible by freezing base weights. See **LoRA-1**. |
| 15b | ~~**LoRA-1** — Wire LoRA-per-specialist into FORGE SFT/dialogue~~ | **DONE Pass 156p (training-side).** Explicit `LoRA` foundation mode card forces adapter training on any model size; rank/alpha widgets and `_start_lora_training()` already wired pre-pass. Adapter saved to `models/checkpoints/{name}_lora.pth`. **Inference-side adapter swap deferred → LoRA-1b** below (load `*_lora.pth` at chat time, hot-swap, multi-adapter UI). |
| 15c | **LoRA-1b** — Inference-side adapter loading and swap (foundation) | **DONE Pass 156s.** PEFT-directory-only save format ([lora_utils.py](enigma_engine/core/lora_utils.py) — manual-fallback `.pth` branch deleted), `scan_lora_adapters(base_model_path=None)` scanner with stem-filtered base matching, `EnigmaEngine.apply_adapter(path)` + `clear_adapter()` + `active_adapter` field with PEFT wrapping and KV-cache-clear on swap, `route_assignments["chat_adapter:<stem>"]` persistence with auto-restore on `_on_model_loaded` and orphan purge when adapter is deleted off-disk. Per-base scoping prevents cross-base adapter mis-application. 6 new tests (3 behavioural scanner + 3 structural double-gates). UX surfaces (MODELS-page list, profile auto-apply, branch markers) → 15d below. |
| ~~15d~~ | ~~**LoRA-1b UX (Pass 156t)** — MODELS-page list, profile field, branch marker~~ | **DONE Pass 156t.** MODELS-page per-card LoRA section with Apply/Clear buttons and base-stem filter, `AIProfile.adapter: Optional[str]` field with apply-or-clear-on-load semantics, legacy `_lora.pth` migration script. Branch marker generalised into Pass 156v Step 1 (`_chat_session_marker` helper). Lazy KV reprefill via `apply_adapter` cache-clear (Pass 156s). |
| ~~15e~~ | ~~**LoRA-1b stacking (Pass 156u)** — multi-adapter weighted stacks~~ | **DONE Pass 156u-A + 156u-B.** PEFT `add_weighted_adapter` engine path (`EnigmaEngine.apply_adapter_stack`), per-base persistence (`route_assignments["chat_adapter_stack:<stem>"]`), mutual-exclusion with single-adapter key, stack-first restore precedence. UI surface: per-row checkbox + numeric weight entry (no sliders, per Dia rules), APPLY STACK button, parse-error collection across all rows. KV invalidation on every weight change. Pass 156u-A2 audit-fixed corrupted-stack-entry resilience and added duplicate-path test. |
| 15f | **Session-1 (Pass 156v)** — unified session-state divider markers | **PARTIALLY DONE.** Step 1 (Pass 156v) shipped `_chat_session_marker` helper + `session_marker` chat tag + adoption at all 5 LoRA adapter-swap success paths. Step 2 (Pass 156v Step 2) extended adoption to model load / unload / RAG enable / RAG disable. **Still open:** profile swap and system-prompt edit have no chat-page surface yet (no widget swaps a runtime persona today — "profile" in code is a forge-page training-data brief, not a chat-runtime persona); add markers when those surfaces are built. Separate concern: model-swap currently calls `_load_model_context(path)` which CLEARS chat history — the marker covers UX visibility but the lazy-reprefill behaviour change (preserve history across model swaps) is a follow-up design decision, not part of Session-1 unification. |
| ~~15g~~ | ~~**Legacy `_lora.pth` migration (Pass 156t companion)**~~ | **CLOSED Pass 156v (no migration needed).** Migration script [migrate_legacy_lora.py](migrate_legacy_lora.py) shipped Pass 156t; zero `_lora.pth` files exist on disk in `models/checkpoints/` or `models/lora_adapters/`. Active code path (Pass 156s+) only writes PEFT directories. Script remains for any future legacy import. |
| ~~16~~ | ~~**N-15** — Constrained decoding (grammar for JSON / tool calls)~~ | **DONE Pass 156z3.** `EnigmaEngine.generate(json_schema=...)` builds `JsonSchemaConstraint` once, threads it through `_generate_text` → `_generate_manual` → `_sample_token` (existing mask hook), drives FSM via `.advance()` per token, early-stops on `is_done`. GGUF path raises `NotImplementedError` (mask never reaches llama.cpp's sampler). Tests +4 (3 wire-site + 1 GGUF rejection); FSM itself already had 5 unit tests from T3-9. **N-15b (next):** expose `json_schema` on chat API endpoint + GUI checkbox so users can reach the feature without writing Python. |
| ~~17~~ | ~~**N-16** — Best-of-N sampling w/ reward model~~ | **DONE Pass 156x.** `EnigmaEngine.generate_best_of_n(prompt, n, reward_fn, *, return_all=False, **gen_kwargs)` runs N candidates, scores via user-supplied `(prompt, response) -> float`, returns highest with first-occurrence tie-break. Validates n>=1 (ValueError), warns on temperature=0+n>1, swallows reward errors with -inf so flaky scorers don't break the batch. 9 behavioural tests via `_FakeSelf` unbound-method pattern. |
| ~~18~~ | ~~**N-19** — External-teacher distillation (text-only, OpenAI-compatible IPC)~~ | **DONE Pass 156z5 (slice 1: offline corpus collector).** Per-user pivot: NO in-process teacher ("complicated to build AI on AI"). Instead: [collect_distill_data.py](collect_distill_data.py) talks HTTP to any OpenAI-compatible `/v1/chat/completions` endpoint (Ollama default `http://localhost:11434/v1`, llama.cpp server, vLLM, our own `run.py --serve`). Stdlib only (urllib). Writes `data/finetune/distill_<tag>.{jsonl,txt}` in the canonical `User: …\n\nAssistant: …` format the existing FORGE Distill mode already consumes — zero training-loop changes. Resumable (skips prompt-keys already in JSONL). Privacy guard: WARNING when endpoint host is not localhost. Failures (HTTP error, unreachable, empty completion) skipped + counted, never abort the run. **+19 tests.** Black-box distillation per the 2024 KD survey ([arxiv 2402.13116](https://arxiv.org/abs/2402.13116)) + Self-Instruct family. **Open follow-ups (slice 2/3, deferred):** Magpie-style empty-prefix instruction synthesis (per-model chat-template handling), GUI "Generate Teacher Corpus" button on the FORGE page, optional top-k logprobs capture (some endpoints expose it — use only if a real white-box use case appears). |
| 19 | ~~**Continuous-1** — Review `BackgroundTrainer` LR (1e-5) + buffer=1000 for continuous SFT safety~~ | **CLOSED Pass 156i4.** Audit caught real silent-drift bug (no NaN/Inf abort \u2192 single bad sample corrupts model permanently with NaN gradients) + OOM exposure (no token-length cap) + claim-vs-test mismatch on replay-buffer eviction. All three fixed; LR=1e-5 + buffer=1000 retained as defaults (sane). See **Continuous-2** for catastrophic-forgetting follow-up. |
| ~~19b~~ | ~~**Continuous-2** — Add anchor-set rehearsal to `BackgroundTrainer._retrain_on_replay`~~ | **CLOSED Pass 156i5.** New `anchor_data_path` kwarg + `_load_anchor_examples()` helper + extended `replay_batch` in `_retrain_on_replay`. Docstring re-framed ("mitigates", not "prevents"). 5 behavioural tests. See **Continuous-3** for GUI surface + default anchor data follow-up. |
| ~~19c~~ | ~~**Continuous-3** — GUI surface + default anchor data for `BackgroundTrainer.anchor_data_path`~~ | **DONE Pass 156i7 (core wiring + default file).** [data/anchor_examples.jsonl](data/anchor_examples.jsonl) ships with 51 curated rows; `ModRouter` auto-wires the path when the file exists. GUI widget for in-app editing/selection deferred to **Continuous-3b** (small UX follow-up). Anchor-only periodic schedule (no recent activity required) deferred to **Continuous-3c** (scheduling design). |
| 19e | ~~**Continuous-3b** — GUI surface for anchor file selection/edit~~ | **DONE Pass 156o.** CONFIG page TRAINING section shows anchor path + row count (or "file missing" / "(none — recent-only)"); BROWSE picks a custom JSONL; USE DEFAULT clears the override. Persisted via `gui_settings.json` key `anchor_data_path`; desktop reads it on boot and forwards to `ModRouter(anchor_data_path=...)`. Restart-to-apply. In-app row editor NOT shipped — file is plain JSONL, edit in any text editor; revisit only if users actually request inline editing. |
| ~~19f~~ | ~~**Continuous-3c** — Anchor-only periodic schedule~~ | **DONE Pass 156w.** Idle-time scheduler in `BackgroundTrainer.run()` queue.Empty branch, gated by new opt-in kwarg `anchor_idle_interval_seconds` (default None = off, 0/negative → off). Helper `_should_run_anchor_idle_replay()` checks all preconditions (interval, anchor path, running, not paused, model+optimizer wired, inference idle, elapsed ≥ interval). Cooldown timer is also reset by regular `_train_batch`-triggered replays so the two paths can't double-fire back-to-back. +9 tests. |
| ~~19d~~ | ~~**Continuous-2a** — Pass 156i5 self-audit findings (logic-eye + claim-vs-test)~~ | **DONE Pass 156i6.** All five findings addressed: (1) sibling-comment drift fixed at [router.py L83](enigma_engine/router.py#L83) and [L470](enigma_engine/router.py#L470); (2) anchor early-out reordered so anchors load before `not replay_buffer` return; (3) WARNING added when anchor file present but yields zero usable rows; (4) legacy-path test strengthened with direct `_load_anchor_examples() == []` assert; (5) cache test added (`test_anchor_file_loaded_only_once_across_replay_passes` — patches `Path.open`, asserts 1 open across 3 replay passes). Suite 2391/9. Note: caller-loop gate `len(replay_buffer) >= batch_size` at [router.py L392](enigma_engine/router.py#L392) still suppresses anchor-only rehearsal during quiet periods — separate design question deferred to Continuous-3. |
| ~~20~~ | ~~**Sched-2** — Change SFT/DPO default warmup from 100 to `max(100, steps // 10)`~~ | **DONE (Pass 152).** [`_effective_warmup()`](enigma_engine/core/training.py#L40) caps warmup at `total_steps // 5` (20%). Wired into all 5 scheduler sites. 8 tests in [TestEffectiveWarmup](tests/test_training.py). |
| 21 | **D-6** — Tokenizer 64K expansion | Must happen BEFORE N-6 restart if at all. After N-6 starts, defer indefinitely. |
| 22 | **R-3 / R-4 / R-5** — Profile MinHash dedup, data pipeline, sequence packing before Rust rewrite | Tokenizer rewrite (R-1/R-2) is done. Others need measurement first. |
| 23 | **D-12** — NoPE hybrid attention (every 4th layer no-RoPE) | Architecture change — only before N-6 if at all. |
| 24 | **ImgGen-4** — SDXL-Turbo / SD-Lightning (4-8 step inference) | Upgrade imagegen mod. |
| 25 | **Audio-5** — MusicGen-small integration | Music generation is a stated project goal; currently TTS-only. |
| 26 | **R-UNPREDICT-1** — Controlled-unpredictability design | **Resolved (Pass 146).** Build follow-up is AutoResearch-2 staged implementation (post-generation uncertainty gate first, inline `<search>` token second). |
| 27 | **Vision-1b** — Upgrade `vision_projection` from single `nn.Linear` to LLaVA-1.5 style 2-layer MLP + GELU | Code gap vs LLaVA-1.5 confirmed this pass. Do alongside Code-6 training wiring. **Pass 148 currency note:** vision encoder target can be upgraded CLIP-ViT-L/14@336 → **SigLIP 2** (Google, arxiv:2502.14786, Feb 2025, 4 sizes 86M/303M/400M/1B) which outperforms SigLIP at all scales across zero-shot, retrieval, VLM transfer, and localization. SigLIP 2 is a drop-in replacement for the frozen encoder; projector design stays LLaVA-1.5 2-layer MLP+GELU. Swap is optional and deferred until after Code-6 training works end-to-end with the current CLIP. |
| 27b | ~~**MTP-2a / MTP-2b** — Fix MTP default~~ | **DONE Pass 149 (MTP-2b path).** Default `n_predict_heads` flipped `2`→0 and inverted comment fixed at [model_presets.py L129](enigma_engine/core/model_presets.py#L129). Saves ~33-49M params per fresh model. Existing zero-guards handled it; explicit `n_predict_heads>0` opt-in still works for Medusa runs. Pass 148 designated EAGLE-2 as the speculative path so Medusa is no longer planned. |
| 28 | **Research queue status** | Original [RESEARCH] backlog items are closed; remaining top items are [BUILD] / [CONFIRMED GAP] implementation work. |

### 🟢 P3 — LOW / DEFERRED (after alignment is done, or parked)

| # | Item | Why deferred |
|---|------|--------------|
| 37 | **D-13 / D-17** — Context length extension (4K → 32K) via RoPE theta scaling + YaRN | Post N-10 only. |
| 38 | **D-20** — Flash Attention 4 SM120 integration | Wait for stable Windows wheel (Q3-Q4 2026). |
| 39 | **Video-1 / 3D-1 / Haptic-1** — Video / 3D / haptic generation | Re-scoped Pass 148 currency check: Wan 2.1 T2V-1.3B (8.19 GB VRAM) and Hunyuan3D-2 (image→3D) now fit 16 GB budget — still deferred by priority, not by feasibility. Haptic has no open dataset. |
| 40 | **D-18** — Aux-loss-free MoE balancing | Only if MoE is enabled — currently not. |
| 41 | **Avatar-1 / Avatar-3 / Avatar-4** — Text-to-animation + training data + output format | Full bone rig exists, pipeline not scheduled given 14-task goal list. |
| 42 | **FSDP / DDP / multi-GPU** | Single-GPU until hardware changes. |
| 43 | **PRM-1** — Process Reward Model training pipeline | DeepSeek paper explicitly calls PRM unsuccessful for RL. Keep code, don't build labeling pipeline. |
| 44 | **Checkpoint browser w/ perplexity comparison** | Nice-to-have after N-7. |
| 45 | **Pre-tokenized binary cache integration** | Script exists ([pretokenize_data.py](pretokenize_data.py)); integrate into training after first N-6 run. |

---

## Status

**Data:** 88.8 GB collected (87.6 GB combined.txt after paragraph dedup). ~26B tokens, 35 tok/param for xl 742M.
**Model:** `xl` (742M params, ~12 GB VRAM with grad ckpt). GPU: RTX 5090, 32 GB total, 16 GB budget.
**Recipe:** GUI Pre-Train → Memory=16 GB → xl. LR 2e-5, batch auto, accum 4, 1-3 epochs.
**Audit:** 148 passes, 515+ findings tracked. Per-pass bug-fix history is archived in [information/history/PASS_HISTORY.md](information/history/PASS_HISTORY.md); per-file fix tracking is in [CODE_REVIEW.md](CODE_REVIEW.md). Open items below. Accepted-risk list at bottom.

---

## Architecture Approaches (April/May 2026 Research)

**Problem Statement:** Enigma's 742M must deliver on 14 broad tasks (vision, audio, 3D, reasoning, code, generation, etc.) but a single LLM can't do all well. Research question: **single model with plugins, or ensemble of specialists?**

**Decision Matrix:**

| Approach | Complexity | Timeline | Risk | Fits Goals | Notes |
|----------|-----------|----------|------|-----------|-------|
| **1. Reasoning-First Ensemble** | Medium | 4-5 months | Low | ✓✓ Excellent | Foundation-first: reasoning model + specialist plugins added later. Matches goal priority. |
| **2. Modular Specialist System** | High | 4-5 months | Medium | ✓✓ Excellent | 5-6 tiny models (200-500M each) + router. Each expert at its task. Most realistic production. |
| **3. Distill from Qwen3-30B** | Low | 3-4 weeks | Very Low | ✓ Good | Fast variants via strong teacher. Proof-of-concept, not final system. |
| **4. Incremental Integration** | Low-Medium | 6-9 months | Very Low | ✓ Good | Add capabilities one at a time. Slowest but safest. |

---

### Approach 1: Reasoning-First Ensemble (RECOMMENDED)

**Core idea:** Pivot 742M into reasoning expert. Add vision/audio/3D as lightweight plugins at inference time.

**Why:** "Learning & reasoning" is goal #1. Test-time compute (QwQ/DeepSeek-R1 style thinking) is the foundation every other task needs.

**Architecture:**
- **Core:** 742M LLM specialized for reasoning (math, code, logic, planning)
- **Vision plugin:** 200M encoder (frozen) + use 742M decoder for VQA
- **Code plugin:** Fine-tuned 742M variant on The Stack v2
- **Router:** Lightweight decision system (what type of task is this?)
- **Generation dispatch:** Invoke SD 1.5 quantized (2GB VRAM) when needed

**VRAM budget:**
- 10GB: Core 742M model
- 3GB: Vision encoder (preload)
- 2GB: Generation model swap space
- 1GB: Router + overhead

**Implementation roadmap:**
1. N-6: Pre-train with reasoning data emphasis (OpenThoughts3, FineMath)
2. N-9/N-10: Instruction fine-tune + GRPO for "thinking before answering"
3. N-11: Train vision encoder on image captioning data
4. N-12: Build router logic (dispatch rules + test on 1000 queries)
5. N-13: Wire plugins (vision → encoder path, generation → SD swap)

**Research needed (R-ARCH-1):**
- [RESOLVED] R-ARCH-1a: DeepSeek-R1 paper (arxiv 2501.12948) read in full. Key findings:
  - **Token format**: `<think>reasoning process here</think><answer>answer here</answer>` — EXACT match to our registered tokens `<think>=4, </think>=5`. No architecture change needed.
  - **RL algorithm**: GRPO (Group Relative Policy Optimization). For each question, sample G outputs from old policy, score all, normalize advantages as `(r_i - mean) / std`. No value head (saves ~50% compute vs PPO). Our `GRPOTrainer` in `rl_training.py` line 2311 already implements this correctly.
  - **Reward design (R1-Zero, the pure RL approach)**:
    - Accuracy reward: rule-based only — math: check `\boxed{}` answer, code: run test cases. Binary pass/fail.
    - Format reward: verify `<think>...</think>` tags present in output.
    - NO neural reward model — explicitly avoided because it leads to reward hacking.
    - Language consistency reward (optional): proportion of target language words in CoT.
  - **Training pipeline (R1-Zero approach, simplest)**:
    - Just GRPO on base model with accuracy + format rewards. No SFT first.
    - Model naturally learns to extend thinking when rewarded — response length grows automatically.
    - Drawback: language mixing, poor readability. Solves reasoning but not UX.
  - **Training pipeline (R1 full, 4 stages)**:
    1. Cold start SFT: ~1,000-5,000 long CoT examples in readable format → fine-tune base model
    2. Reasoning GRPO: same accuracy + format + language consistency rewards, train to convergence
    3. Rejection sampling SFT: generate from RL ckpt, keep only correct ones (~600K reasoning + ~200K general = 800K total), re-train from base
    4. Second GRPO pass: rule-based for math/code + neural reward model for general tasks
  - **CRITICAL for 742M scale**: DeepSeek trained Qwen-32B with GRPO for 10K+ steps → matched QwQ-32B-Preview. But distilling R1 into Qwen-32B with SFT alone → **massively outperformed** RL training. Conclusion: **at 742M, distillation (Approach 3) is far more efficient than GRPO training**. RL can improve a distilled model further but should come AFTER distillation, not before.
  - **Thinking budget**: No hard budget during training. Model naturally allocates more thinking tokens when rewarded. At inference: R1 eval caps at 32,768 tokens. For 742M production: cap think tokens at 2048-4096 (our context is 4096 total — leave ~512 for input + answer). [enigma_engine/core/engine_generation.py](enigma_engine/core/engine_generation.py) needs a `max_think_tokens` parameter added.
  - **Data format for cold-start**: User/Assistant chat with `<think>COT</think><answer>final</answer>` pattern. Summary sentence after `</think>` is optional but improves readability. Generate using Qwen3-30B-A3B as teacher — already available in `models/qwen3-30b-a3b/`.
  - **For Enigma action**: (1) Generate 3,000-5,000 math/logic cold-start examples using Qwen3-30B-A3B FORGE Distillation. (2) SFT on those examples. (3) Optionally: GRPO with accuracy+format reward. File refs: [enigma_engine/core/reasoning.py](enigma_engine/core/reasoning.py) (has `strip_incomplete_think()`), [enigma_engine/core/rl_training.py](enigma_engine/core/rl_training.py#L2311) (GRPOTrainer line 2311), [enigma_engine/core/engine_generation.py](enigma_engine/core/engine_generation.py) (generation loop — needs max_think_tokens).
- [RESOLVED] R-ARCH-1b: LLaVA vision encoder specs (200M size, frozen vs fine-tuned). Verified from primary sources:
  - **Encoder choice:** LLaVA-1.5 uses `CLIP-ViT-L-336px` (arxiv:2310.03744, Sec 3.3 + Appendix).
  - **Freeze policy:** original LLaVA keeps vision encoder frozen in Stage 1 and Stage 2; Stage 1 trains projection only, Stage 2 trains projection + LLM (`theta={W,phi}`) while encoder stays frozen (arxiv:2304.08485, Sec 4.2).
  - **Projector architecture:** LLaVA-v1 uses Linear projector; LLaVA-1.5 baseline upgrade is 2-layer MLP connector (arxiv:2310.03744, Sec 3.3). LLaVA model zoo projector table confirms `MLP-2x` for v1.5 projector variants.
  - **Model size correction:** "200M vision encoder" is not accurate for CLIP-L. `ViT-L-14-336` visual tower is ~304.29M params (OpenCLIP model profile), and full CLIP `ViT-L-14-336` (vision + text) is ~427.94M params.
  - **Enigma decision:** for 16 GB VRAM target, keep CLIP encoder frozen and train projector (+ optional lightweight adapter) first; only consider partial unfreeze after projector convergence and only with explicit benchmark gain.
- [RESOLVED] R-ARCH-1c: Router design patterns (decision tree, learned router, heuristic rules). Recommendation is now explicit: **layered router** = heuristic regex gate first, embedding-similarity classifier second, LLM classifier fallback only for low-confidence cases. This matches our existing Router-2/3 conclusions and keeps latency low while preserving quality.
- [RESOLVED] R-ARCH-1d: How to train router (labeled data: "is this a vision task?"). Best practical recipe from RouteLLM (arxiv:2406.18665 + LMSYS blog):
  - Start with weak supervision labels from logs/heuristics (task tags: vision/code/reasoning/general).
  - Train a small classifier first (BERT-tiny class) for low-latency routing.
  - Add targeted augmentation for out-of-distribution buckets (small curated set is enough to move quality significantly).
  - Keep LLM-based routing as judge/fallback, not primary path.
  - Evaluate router quality-cost curves, not just top-1 accuracy (target is quality at lower expensive-model call rate).

**Timeline:** 5 months (N-6 → N-13)

**Advantages:**
- Aligns with stated priority (reasoning first)
- Validated by Qwen3 paper
- Plugins are optional (still works as reasoning-only model)
- Can test each component independently

**Disadvantages:**
- Plugins are not specialist-trained (vision encoder is borrowed, not optimized for Enigma)
- Router adds latency (query classification step)
- Sharing 742M decoder across tasks means compromises

---

### Approach 2: Modular Specialist System

**Core idea:** Train 5-6 small specialists (200-500M each), router learns to dispatch.

**Specialists:**
1. **Core Router (500M):** Text tasks, decision-making
2. **Vision Specialist (300M):** Image understanding (LLaVA-style)
3. **Reasoning Specialist (300M):** Math, code, logic (QwQ-style)
4. **Code Specialist (200M):** Programming (StarCoder-style)
5. **Knowledge Specialist (300M):** Facts, retrieval, wiki-like (optimized for RAG)

**Router learning:**
- Train router on 10K+ labeled queries ("which specialist should answer?")
- Router outputs: probability distribution over specialists
- Load top specialist(s), invoke

**VRAM budget:**
- 10GB: One active specialist (swappable)
- 2GB: Router stays resident
- 2GB: Generation dispatch (SD 1.5)
- 2GB: Swap/preload overhead

**Implementation roadmap:**
1. Design router: architecture, input/output spec
2. Collect 10K labeled training queries (which specialist?)
3. Train 5 specialists in parallel (can distribute across multiple GPUs or time-slice)
4. Train router on labeled data
5. Benchmark: test on held-out query set

**Research needed (R-ARCH-2):**
- [RESOLVED] R-ARCH-2a: Router architectures (learned vs rule-based, latency comparison). Best tradeoff for this project is hybrid: rule-based front gate + learned lightweight classifier + LLM fallback. Latency/cost rationale is documented in Router-2/3 findings and aligns with RouteLLM-style quality-cost routing (arxiv:2406.18665).
- [RESOLVED] R-ARCH-2b: Specialist architectures (should each be exactly 742M, or vary by task?) Recommendation: **heterogeneous specialist sizing** beats "all 742M" for quality-cost frontier, with one caveat: keep a single shared tokenizer/chat format and shared checkpoint schema so orchestration stays simple.
  - RouteLLM/FrugalGPT results are built on strong+weak model pairing (different capacity/cost), and show large savings at near-constant quality; this is direct evidence that heterogeneous capacity is a feature, not a bug (arxiv:2406.18665, arxiv:2305.05176).
  - Practical sizing policy for this repo/hardware: keep reasoning + code at 742M first, allow narrower specialists for easier domains (routing/classification/short-format helpers), and only scale up task-specific experts when benchmark delta justifies VRAM/latency cost.
  - Do not force equal params across all specialists; force equal **interface contract** instead (same tokenizer IDs, prompt format, stop tokens, and eval harness).
- [RESOLVED] R-ARCH-2c: Load balancing (how to swap models in/out without stalling?) Recommendation: avoid per-request model hot-swap. Use **resident pool + async warmup + queueing**.
  - Systems evidence: iteration-level scheduling avoids head-of-line blocking by scheduling per token-step rather than waiting for full request batches (ORCA OSDI'22: usenix osdi22-yu).
  - Memory evidence: paged KV-cache management increases effective batch capacity and reduces fragmentation pressure that causes stall cascades under mixed-length traffic (PagedAttention/vLLM: arxiv:2309.06180).
  - Throughput evidence: prompt/generation split strategies and serving-side scheduling materially reduce latency/tail latency under load (DeepSpeed-FastGen: arxiv:2401.08671).
  - Concrete plan: keep router + 1 default specialist resident; maintain one prewarmed standby slot; route overflow by queue policy (short-first within class, deadline-aware fallback to base 742M); perform background prefetch/load and only cut over when health-check + warmup pass.
- [RESOLVED] R-ARCH-2d: Training data generation (label 10K queries with correct specialist). Recommended pipeline:
  - Bootstrap labels from heuristics + existing route outcomes (weak labels).
  - Add small high-quality human/LLM-judge corrected set for ambiguous prompts.
  - Actively sample disagreement buckets for targeted relabeling instead of uniform 10K labeling.
  - Train router on preference/outcome labels and optimize for quality-cost curve.
- [RESOLVED] R-ARCH-2e: Evaluation (multi-specialist system performance vs single model). Router/specialist evaluation should use **quality-cost frontier** rather than raw accuracy only (RouteLLM arxiv:2406.18665, FrugalGPT arxiv:2305.05176):
  - Report quality at fixed expensive-route percentages (e.g., 10/25/50/75/100%).
  - Report cost to reach target quality (e.g., 95% of strong-model quality).
  - Include OOD slices separately (where simple routers often fail).
  - Keep per-specialist task benchmarks and add end-to-end routed benchmark as the deployment metric.

**Timeline:** 4-5 months

**Advantages:**
- Each specialist optimized for its task (vision model learns vision REALLY well)
- Can iterate on specialists independently (retrain vision without touching code specialist)
- Closest to real-world systems (OpenAI, Anthropic likely use something like this)
- Scales: add more specialists without retraining core

**Disadvantages:**
- Complex implementation (multi-model orchestration, router training)
- Requires labeled training data for router (10K queries with expert labels)
- Training 5 models is 5× the compute
- Higher latency (router inference + model loading)

---

### Approach 3: Distillation from Qwen3-30B (FASTEST PROOF-OF-CONCEPT)

**Core idea:** Use Qwen3-30B (already installed) to create specialized 742M variants via distillation. Quick experiment to validate multi-model architecture.

**What to create:**
1. **Enigma-742M-Reasoning:** Qwen3-30B generates thinking traces on math/logic
2. **Enigma-742M-Vision:** Qwen3-30B generates VQA answers on image dataset
3. **Enigma-742M-Code:** Qwen3-30B generates code solutions on programming tasks

**Process per variant:**
1. Use Qwen3-30B to generate target outputs on task-specific data
2. Train 742M via FORGE Distillation mode
3. Test on benchmark (GSM8K for reasoning, etc.)

**VRAM budget:**
- 12GB: 742M variant during training
- 2GB: Qwen3-30B for teacher inference (separate, offline)
- 2GB: Overhead

**Implementation roadmap:**
1. Collect/prepare data for 3 tasks (existing: OpenThoughts3, LLaVA dataset, The Stack)
2. Generate targets from Qwen3-30B (offline, can batch)
3. Train 3 variants via FORGE
4. Benchmark each on task-specific test set
5. **Decision point:** If variants work, proceed to Approach 1 or 2. If not, learn why and iterate.

**Research needed (R-ARCH-3):**
- [RESOLVED] R-ARCH-3a: Distillation effectiveness at 742M scale (does 742M learn from 30B?) **Yes, materially.** Evidence: DeepSeek-R1 reports strong distilled performance from a much larger teacher into small dense students, including 1.5B/7B/8B/14B/32B checkpoints, and explicitly states smaller models benefit from distilled reasoning traces (arxiv:2501.12948 + DeepSeek-R1 model card). Orca independently shows smaller models gain large reasoning improvements from rich explanation traces generated by stronger teachers (arxiv:2306.02707).
- [RESOLVED] R-ARCH-3b: Data requirements (how much task-specific data per variant?) Use staged targets, not one-shot max:
  - **Pilot:** 10K-30K curated examples per variant to validate training plumbing + overfit risk.
  - **Useful:** 80K-200K per variant for measurable specialist lift.
  - **Strong:** 300K-800K for reasoning-heavy variants when teacher quality is high.
  - Rationale: LIMA shows alignment can move with very small curated sets (1K), Orca shows complex reasoning transfer needs broader/denser traces, and DeepSeek-R1 distill models cite 800K curated samples for robust small-model transfer.
- [RESOLVED] R-ARCH-3c: Benchmark setup (which evals to use for each variant?) Recommended minimum benchmark matrix:
  - **Reasoning variant:** GSM8K + MATH-500 (+ AIME-style slice when available), report pass@1 and consistency across 3 seeds.
  - **Code variant:** HumanEval pass@1 + LiveCodeBench/Codeforces-style difficulty slice (DeepSeek-R1 uses both families).
  - **Vision variant:** internal 5K held-out VQA gate from this roadmap plus one public VLM benchmark already used in your LLaVA track for comparability.
  - **System metric:** add routed end-to-end quality-cost frontier, not only per-specialist scores.
- [RESOLVED] R-ARCH-3d: Router for Approach 3 (can we use simple heuristics, or need learned router?) Recommendation: **two-phase**.
  - Phase POC (3 variants): heuristic router is acceptable to move fast (regex/task-keyword + cheap classifier fallback).
  - Phase production: learned router is required once overlap/ambiguity rises; RouteLLM shows meaningful cost-quality gains from learned routing and highlights OOD degradation without targeted augmentation (arxiv:2406.18665).
  - Promotion gate: switch from heuristic-only to learned when misroute rate or fallback-to-base rate exceeds a fixed threshold on held-out mixed-task traffic.

**Timeline:** 3-4 weeks (1-2 weeks per variant)

**Advantages:**
- **Fastest to results** (1-2 weeks per variant)
- Low risk (teacher is proven, process is standard)
- Validates architecture idea before committing to Approach 1 or 2
- Uses existing FORGE infrastructure
- Clear success/fail signal

**Disadvantages:**
- Still 742M (not true specialists)
- Depends on teacher quality (Qwen3-30B is good but not perfect)
- Variants may overlap (not diverse enough)
- Proof-of-concept only, not production system

---

### Approach 4: Incremental Integration (SAFEST)

**Core idea:** Keep Enigma 742M as base, add one capability at a time, test before moving on.

**Phases:**
| Phase | What | Timeline | Success Criteria |
|-------|------|----------|------------------|
| Now | N-6: Resume pre-train (base 742M) | 2-3 months | Loss curve stable, no NaNs |
| A | N-9/N-10: Instruction fine-tune + alignment | 2-3 weeks | Benchmark on MMLU/GSM8K |
| B | Add vision encoder (frozen LLaVA) | 2 weeks | Can process images without crashing |
| C | Test vision reasoning | 1 week | 50%+ accuracy on 5K VQA pairs |
| D | Add code fine-tuning variant | 2 weeks | Passes 10+ HumanEval problems |
| E | Add reasoning variant (OpenThoughts) | 2 weeks | GSM8K > 30% |
| F | Build router (if 3+ specialists work) | 1-2 weeks | Router predicts correct specialist 85%+ |
| G | Add generation/audio plugins | 1-2 weeks each | Can invoke SD/Whisper correctly |

**VRAM budget:**
- 12GB: Active model
- 3GB: Temporary specialist during loading
- 1GB: Overhead

**Research needed (R-ARCH-4):**
- [RESOLVED] R-ARCH-4a: Sequential integration order (which capability first?) Recommended order:
  1. Base stability (N-6 pretrain health: no NaNs, stable loss, usable perplexity)
  2. Instruction/alignment baseline (N-9/N-10) so all later specialists inherit usable chat behavior
  3. Reasoning specialist before vision/audio plugins (highest leverage for text quality and eval visibility)
  4. Code specialist
  5. Vision specialist + VQA validation
  6. Router only after at least two specialists beat base on their own domains
  7. Optional generation/audio plugins last (non-core to text reasoning quality)
  - Principle: always harden the core language path before adding modality or orchestration complexity.
- [RESOLVED] R-ARCH-4b: Benchmark suite (what tests for each phase?) Use phase-gated checks:
  - **Core health gate (every phase):** train stability (NaN/Inf), latency sanity, memory budget adherence, regression test suite.
  - **Alignment gate:** MMLU + instruction-following slice.
  - **Reasoning gate:** GSM8K + MATH-500 (or equivalent held-out math set).
  - **Code gate:** HumanEval pass@1 (+ optional LiveCodeBench slice).
  - **Vision gate:** internal held-out 5K VQA gate + one public VLM benchmark used by your LLaVA path.
  - **Router gate:** route accuracy + routed end-to-end quality-cost frontier vs single-model baseline.
- [RESOLVED] R-ARCH-4c: Fallback logic (if a phase fails, which is it safe to skip?) Safety policy:
  - If **core/alignment** fails: stop entire roadmap and fix core first (nothing else should proceed).
  - If **reasoning** fails: continue code/vision experiments only if they do not regress base chat quality; keep routing disabled.
  - If **one specialist** fails (code or vision): ship other specialists + base fallback; do not block unrelated track.
  - If **router** fails: keep heuristic routing or manual mode selection; never force learned router into production path.
  - If **plugin phase** fails: skip safely; plugins are optional and must not block core chat/reasoning releases.

**Timeline:** 6-9 months

**Advantages:**
- **Very low risk** (test each before committing)
- **Feedback loop** (learn from each phase)
- Can stop at any point with a working system
- Simple code changes (one feature at a time)

**Disadvantages:**
- Slowest timeline
- Redundant work if a late capability fails
- Hard to optimize across tasks (single model, compromises everywhere)

---

## Audit Notes (April 2026)

**Before reading this checklist, read this audit. ~50% of the original items were already implemented.**

**Already implemented — verified in code:**
- QK-Norm (`use_qk_norm=True` default, applied in model_components.py lines 347/423/451)
- MTP (`n_predict_heads=2`, `predict_heads` ModuleList, `mtp_loss` wired in model.py 496-518)
- Intra-document masking (`build_packing_masks()` in training.py builds block-diagonal causal mask)
- Flash Attention (tries `flash_attn_func`, falls back to `F.scaled_dot_product_attention`)
- Weight tying (`self.output.weight = self.tok_embeddings.weight` in model.py line 233)
- LayerScale + stochastic depth (`use_layer_scale`, `drop_path_rate` in ForgeConfig)
- Speculative decoding (`speculative_generate` in engine_generation.py)
- Vision/Audio projections (`vision_hidden_size`, `audio_hidden_size` in ForgeConfig + model.py)
- GQA (`n_kv_heads` in ForgeConfig, validated in __post_init__)
- SwiGLU, RMSNorm, RoPE + YaRN, Sliding window, MoE (`use_swiglu`, `use_rms_norm`, `rope_theta`, `rope_scaling_type`, `sliding_window`, `use_moe`) — all in ForgeConfig
- Label smoothing (`label_smoothing=0.0` parameter in model.py)
- LLRD (`_build_llrd_param_groups()` in training.py)
- Weight decay param groups (bias/norm get no_decay, see training.py 1332-1345)
- LR warmup (`warmup_steps` in TrainingConfig + defaults.py)
- Differential attention (`use_differential_attn=True` default)
- NEFTune (`neftune_alpha=5.0` default)
- MLA (`mla_latent_dim`), ToMe (`tome_ratio`), YOCO KV sharing (`kv_share_groups`), MoD (`use_mixture_of_depths`), shifted sparse attention (`use_shifted_attention`) — all in ForgeConfig
- AdamW + AdemaMix optimizers with beta1/beta2 — in training.py
- KV cache, sampling (top-k/top-p/temperature/repetition penalty) — all implemented in inference

**D-series status corrections:**
- D-14 (QK-Norm): ~~Action needed~~ → **ALREADY DONE** (`use_qk_norm=True` is the default, wired)
- D-15 (MTP): ~~Verify scaffold~~ → **ALREADY DONE** (loss wired in model.py, λ=1/n_heads)
- D-7 (intra-doc masking): ~~Verify~~ → **ALREADY DONE** (`build_packing_masks()` exists + is called)
- D-8 (embed weight decay): **STILL OPEN** — training.py line 1337 checks `'bias' in name or 'norm' in name` but NOT `'embed'`. Embeddings and the output head get weight_decay when they should not. 5-line fix.

**Removed from checklist (wrong scope for this project):**
- DDP/FSDP/torch.distributed — single GPU system; multi-GPU is future work with no near-term value
- Multilingual benchmarks — project is English-focused
- Torch barrier/deadlock prevention — no distributed training
- Items describing behavior that is purely inference-time and already configurable in GUI (top-k, top-p, temperature, stop tokens)

**Real missing research (added below):**
- Audio encoder training procedure (projection layer exists, no pipeline)
- Video/3D generation via mod system (goal items, no research done)
- Mod integration procedure (how to plug in audiogen, videogen, threed mods)
- LoRA adapter training workflow (directory exists, training procedure unknown)
- Catastrophic forgetting prevention when creating specialists (EWC exists, is it wired for this use case?)
- Context extension implementation (YaRN config field exists but forward-pass code needs verification)
- Router multi-model dispatch design (router.py handles mods, not model selection)

---

## Comprehensive Research Checklist (Full Coverage)

**Legend:** Items marked [VERIFY] need code verification only (no external research). Items marked [RESEARCH] need external sources. Items marked [BUILD] are confirmed gaps needing implementation.

---

### Architecture — What Needs Verification (Not External Research)

These are implemented in code but the exact behavior needs to be read and confirmed before N-6 restart:

- [RESOLVED] Arch-1: `build_packing_masks()` → `attention_mask_2d` kwarg → model.forward(). Confirmed wired: [enigma_engine/core/training.py](enigma_engine/core/training.py) lines 1669, 1676, 3183-3240.
- [RESOLVED] Arch-2: xl uses actual GQA: n_heads=24, n_kv_heads=6 (4:1 ratio, same as LLaMA3). Verified [enigma_engine/core/model_presets.py](enigma_engine/core/model_presets.py).
- [RESOLVED] Arch-3: xl preset confirmed — dim=1536, n_layers=24, n_heads=24, n_kv_heads=6, max_seq_len=4096, rope_theta=500000.0, dropout=0.05, hidden_dim=4×1536=6144.
- [RESOLVED] Arch-4: `use_differential_attn=True` is the ForgeConfig dataclass default ([enigma_engine/core/model_presets.py](enigma_engine/core/model_presets.py#L127) line 127) — applies to ALL presets including xl. Latency research still open (see Attn-2).
- [RESOLVED] Arch-5: MTP checkpoint size confirmed. `count_parameters()` in [enigma_engine/core/model_presets.py](enigma_engine/core/model_presets.py#L931) line 931: `mtp = n_predict_heads × vocab_size × dim`. For xl (dim=1536, n_predict_heads=2, vocab≈32K): **2 × 32000 × 1536 ≈ 98M extra params = ~196MB at fp16.** These are NOT weight-tied. Larger vocab (e.g., 64K) doubles this to ~390MB. Factor into any checkpoint size planning.
- [RESOLVED] Arch-6: YaRN IS fully implemented — `compute_freqs_cis()` in [enigma_engine/core/model_components.py](enigma_engine/core/model_components.py#L190) has a real `elif scaling_type == "yarn"` branch (lines 190-213) that scales frequency bands. Not dead config.
- [CONFIRMED] Arch-7: MTP loss weight = 1/n_predict_heads ([enigma_engine/core/model.py](enigma_engine/core/model.py#L518) line 518: `loss + mtp_loss / max(1, len(self.predict_heads))`). With xl's n_predict_heads=2: each head adds 0.5× weight. DeepSeek-V3 uses λ=0.3 — **MTP-1 resolved, see MTP section. Our λ=0.5 is higher; lower to 0.3 for next pretraining run.**
- [RESOLVED] Arch-8: LayerScale NOT used at 3B or 0.6B scale. Verified SmolLM3-3B config.json and Qwen3-0.6B config.json — neither has LayerScale parameters. SmolLM3 blog ablations test GQA, NoPE, intra-doc masking, embed weight decay — no LayerScale anywhere. **Conclusion: keep `use_layer_scale=False` (current default). Stochastic depth also absent from both — keep `drop_path_rate=0.0`.** These features do not appear in production 0.6B–3B models. No change needed.

---

### Architecture — Real Gaps (External Research Needed)

These are features referenced in code but the correct values/approaches need external evidence:

**Attention:**
- [RESOLVED] Attn-1: GQA ratio research complete. SmolLM3-3B uses 16:4 = **4:1** (same as our xl). Qwen3-0.6B uses 16:8 = **2:1** (more conservative). SmolLM3 blog: "Our ablations on a 3B model trained with 100B tokens from FineWeb-Edu showed that GQA matches the performance of MHA while significantly reducing KV cache." **Our xl 24:6 = 4:1 ratio is validated at 3B scale. Qwen3-0.6B uses 2:1 which is more conservative for sub-1B but we already have a trained checkpoint — no change needed.** If retraining from scratch at 742M, 2:1 (12 KV heads) would be marginally safer; 4:1 is acceptable per SmolLM3 3B evidence.
- [RESOLVED] Attn-2: Differential attention (Ye et al., *Differential Transformer*, arxiv:2410.05258, **ICLR 2025 Oral**). Paper ablates at 830M — directly comparable to our 742M (the scaling claim applies). Confirmed wins vs. vanilla transformer at equal params/tokens: lower perplexity, better long-context retrieval, reduced hallucination in QA/summarization, better in-context learning, more robust to prompt permutation. **Already enabled in our code** ([enigma_engine/core/model_presets.py](enigma_engine/core/model_presets.py#L127) line 127 default `True`; wired at [enigma_engine/core/model_components.py](enigma_engine/core/model_components.py#L360) lines 360-367 and 578). **Real cost:** the dual-softmax structure does NOT fit `F.scaled_dot_product_attention` / Flash Attention — [line 543](enigma_engine/core/model_components.py#L543) falls through to the manual attention path when `use_differential_attn=True`. Training and prefill attention therefore run slower than a vanilla-Flash baseline. **Memory:** negligible overhead (head count unchanged, one extra learnable λ per layer). **Verdict:** KEEP ENABLED for quality gains; the Flash-path loss is already measured & absorbed into our training budget. Worth tracking via an internal A/B only if we ever hit an attention-bound bottleneck. No action.
- [RESOLVED] Attn-3: xl (and all medium/base/large/xxl) presets already use rope_theta=500000.0 in [enigma_engine/core/model_presets.py](enigma_engine/core/model_presets.py). Only the small preset uses 10K. This was a documentation error — code was already correct.

**Normalization:**
- [RESOLVED] Norm-1: RMSNorm epsilon is 1e-6 in [enigma_engine/core/model_components.py](enigma_engine/core/model_components.py). Matches Qwen3 best practice. fp32 upcast present.
- [RESOLVED] Norm-2: QK-Norm uses the same RMSNorm class as layer normalization. Full RMSNorm on Q and K, matching Qwen3 pattern.

**MTP (Multi-Token Prediction):**
- [RESOLVED] MTP-1: DeepSeek-V3 technical report (arxiv:2412.19437v2, Section 4.2) confirms: **λ=0.3 for first 10T tokens, then λ=0.1 for remaining 4.8T tokens.** MTP depth D=1 (predict 1 extra token beyond next-token). Ablation table (Table 4) validates MTP consistently improves benchmarks at both small (15.7B) and large (228.7B) scale. MTP acceptance rate at inference: 85–90% across topics. **Our code uses λ=0.5 (= 1/2 predict heads) which is higher than DeepSeek's 0.3. Recommendation: lower to λ=0.3 for new pretraining runs.** For existing checkpoints already trained with 0.5, keep as-is — retrofitting is not worth the churn. File: [enigma_engine/core/model.py](enigma_engine/core/model.py#L518) — change divisor from `len(predict_heads)` to `len(predict_heads) / 0.6` (makes 2 heads → 0.3) OR use a dedicated `mtp_loss_weight` config field set to 0.3.
- [RESOLVED] MTP-2: Meta *Better & Faster LLMs via Multi-token Prediction* (arxiv:2404.19737) abstract verbatim: "The method is **increasingly useful for larger model sizes**." Paper tested 300M / 1.3B / 6.7B / 13B — at 300M the gain is neutral or slightly negative for natural language, weakly positive for code; at 1.3B gains become clearer on code (HumanEval/MBPP); at 13B the reported HumanEval delta is +12%. **Our 742M sits in the ambiguous bracket.** Two concrete issues found in code this pass:\n  1. **Inverted comment BUG** at [enigma_engine/core/model_presets.py](enigma_engine/core/model_presets.py#L129) line 129: `# R25: Multi-token prediction extra heads (biggest gain at small model sizes)` — this inverts the paper's conclusion. Fix: rewrite comment to `"most useful at larger scales; marginal/ambiguous at <1B"`.\n  2. **Hidden param cost:** each entry in `self.predict_heads` is `nn.Linear(dim=1536, padded_vocab≈32K, bias=False)` at [model.py line 240](enigma_engine/core/model.py#L240), **NOT weight-tied** to `tok_embeddings`. Default `n_predict_heads=2` adds ~98M params (~13% of 742M) that are discarded at inference except when using Medusa speculative decoding. This is effectively 98M params spent on an auxiliary training signal of uncertain benefit at our scale.\n\n**Verdict / action:**\n- If Medusa inference is planned → keep `n_predict_heads >= 1` (Medusa needs them) but **reduce default from 2 → 1** to halve the cost, and weight-tie the head to `tok_embeddings.weight` same as `self.output` (free 49M saving). Log as new item **MTP-2a**.\n- If Medusa is NOT planned → set `n_predict_heads = 0` by default, free 98M params for the main model. Log as new item **MTP-2b**.\n- Internal A/B (MTP on vs off, same data/steps, val loss + HumanEval-style code) is still the honest way to decide — but the default should not be 2 untied heads by accident.

---

### Data — Real Research Gaps

These have no implementation yet and need external evidence for correct choices:

**Tokenization:**
- [RESOLVED] Tok-1: Tao et al. *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies* (NeurIPS 2024, arxiv:2407.13623) trained models from 33M → 3B params on up to 500B chars to fit a compute-optimal vocab scaling law. Concrete result: at **3B params**, expanding 32K → **43K** improved ARC-Challenge from 29.1 → 32.0 at the same 2.3e21 FLOPs. Llama2-70B's optimal predicted vocab is ~216K (7× its actual 32K). For our 742M (below their 3B data point), the compute-optimal is in the **~35-45K** range — SmolLM3's 49K at 3B is a slight overshoot but defensible. **64K would be over-provisioned at our scale** (puts us above the 3B-scale optimum while we're less than a third of that). Current 32K is slightly under-optimal but not catastrophically so. **Decision:**\n  - Do NOT expand to 64K.\n  - If (and only if) we retrain from scratch, target **~40K vocab** (matches the paper's 3B-scale optimum, leaves headroom for a small multilingual or code-specific extension).\n  - D-6 ("Tokenizer 64K expansion") in the backlog should be **downgraded** — retitle to "Tokenizer 40K expansion (if retraining from scratch only)." Anything past 32K requires a full restart; committing to 64K costs ~49M extra embedding params for no measurable quality gain at 742M per the scaling law.
- [CONFIRMED GAP] Tok-2: [enigma_engine/core/advanced_tokenizer.py](enigma_engine/core/advanced_tokenizer.py) has NO byte-level fallback. Unknown chars fall through to CHAR-LEVEL decode first (line 267), then UNK token id=3 (line 272/279) if the char is not in vocab. Rust BPE is the same — only merges tokens in its merge table, no byte fallback. **Rare Unicode (CJK, Arabic, emoji, etc.) will produce UNK ids, corrupting model input and breaking multimodal.** Fix before tokenizer retrain: add 256 raw byte tokens to base vocab (byte-level BPE approach used by GPT-2/Qwen). File: [enigma_engine/core/advanced_tokenizer.py](enigma_engine/core/advanced_tokenizer.py) `::_init_base_vocab()` and [rust_extensions/src/lib.rs](rust_extensions/src/lib.rs) `::pre_tokenize()`.
- [BUILD] Tok-3: Collect The Stack v2 (D-2) before tokenizer retrain — code vocabulary affects merge priority. Train tokenizer on data mixture AFTER collecting all sources.

**Data mixing (all resolved Pass 137 — external research):**
- [RESOLVED] Data-1: **Stage 1: 12% code**, per SmolLM3 blog (https://huggingface.co/blog/smollm3). Quote: "Code: 12% — The Stack v2 (16 programming languages), StarCoder2 pull requests, Jupyter and Kaggle notebooks, GitHub issues, and StackExchange." Stage 2 ramps to 15%, Stage 3 (annealing/cooldown) to 24%. Qwen3 (arxiv:2505.09388) says "increasing STEM/coding proportion in Stage 2" but gives NO exact percentages. DeepSeek-LLM (arxiv:2401.02954) also provides no explicit %. **No direct evidence on mid-run introduction of code.** SmolLM3's 12→15→24 progression suggests monotonic increase is safe; abrupt jumps unvalidated. SmolLM3 is 3B; pattern assumed to scale down to 742M but not empirically verified at our scale.
- [RESOLVED] Data-2: **Stage 1: 3% math** (FineMath-3+ + InfiWebMath-3+), per SmolLM3 blog. Quote: "Math: 3% — FineMath3+ and InfiWebMath3+". Stage 2: 10%. Stage 3 (cooldown): 13%. FineMath paper (arxiv:2502.02737) validates the datasets themselves. **FineMath-3+ (34B tokens) for main mixture, FineMath-4+ (9.6B tokens) optionally for Stage 3 resampling.** HuggingFace: `HuggingFaceTB/finemath` configs `finemath-3plus` / `finemath-4plus` / `infiwebmath-3plus`.
- [RESOLVED] Data-3: HF dataset ID confirmed `mlfoundations/dclm-baseline-1.0`. 4T tokens, 3B documents. Public, no gating. Paper arxiv:2406.11794. Our [collect_pretraining_data.py](collect_pretraining_data.py) fetcher (added Pass 136) streams correctly.
- [RESOLVED] Data-4: HF dataset `open-thoughts/OpenThoughts3-1.2M` CONFIRMED. Composition CONFIRMED: 850K math + 250K code + 100K science = 1.2M. Paper arxiv:2506.04178. **Tag format CONFIRMED via HuggingFace datasets-server API (live row fetch Pass 139).** Schema: 4 fields — `difficulty` (int64), `source` (str), `domain` (str), `conversations` (list of `{"from": "human"/"gpt", "value": str}`). Row 0 gpt value begins: `"<think> Okay, I need to solve this problem..."` and ends with `</think>\n\n` then the final answer. Row 1 same pattern. The QwQ-32B reasoning trace IS wrapped in `<think>...</think>` inline in the `value` field — our special tokens `<think>=4`, `</think>=5` align exactly. **Action for OpenThoughts3 fetcher in [collect_finetuning_data.py](collect_finetuning_data.py):** extract `value` from each conversation turn and preserve `<think>`/`</think>` tags verbatim — do NOT strip them during preprocessing.
- [RESOLVED] Data-5: **Industry standard is 75% MinHash similarity**, not our current 80%. FineWeb technical report (arxiv:2406.17557v2, Section 3.4): "We chose to collect each document's 5-grams...computed MinHashes using 112 hash functions in total, split into 14 buckets of 8 hashes each — targeting documents that are at least 75% similar." DCLM uses same approach. Our 80% is slightly more permissive (keeps more near-duplicates). See Data-5b below for follow-up action.
- [CONFIRMED GAP] Data-5b: MinHash threshold in our training.py `minhash_dedup()` defaults to 0.8. FineWeb/DCLM/SmolLM3 use 0.75. **Low priority code change:** flip default from 0.8 to 0.75 in [enigma_engine/core/training.py](enigma_engine/core/training.py) `minhash_dedup()` before next tokenizer/data-prep cycle. Impact on already-deduped `combined.txt` is zero (re-dedup not planned); only matters for future data runs. File note: also check the LSH band/row split — FineWeb uses 14 bands × 8 rows of 112 hashes; if our implementation uses different band counts the effective threshold differs.
- [RESOLVED] Data-6 (NEW — annealing/Stage-3 mixture for 742M): SmolLM3 Stage 3 ("decay"): **24% code, 63% web, 13% math**. Noted finding: "no significant improvements from upsampling curated subsets in cooldown phase" — so Stage 3 gains come primarily from LR decay and code/math upweight, not from swapping in premium subsets. Pattern validated at 3B, assumed to scale to 742M. Evidence is empirical not theoretical — worth a small ablation before committing.

---

### Training Algorithm — Real Gaps

**Optimizer:**
- [RESOLVED] Opt-1: β2=0.95 is hardcoded in TrainingConfig (`adam_beta2: float = 0.95`). Not the PyTorch default. Correct.
- [RESOLVED] Opt-2: D-8 FIXED this session. Both `_setup_optimizer()` and `_build_llrd_param_groups()` in training.py now include `or 'embed' in name` in the no_decay check.
- [RESOLVED] Opt-3: Weight-tying confirmed at [enigma_engine/core/model.py](enigma_engine/core/model.py#L233) line 233: `self.output.weight = self.tok_embeddings.weight` — direct Python tensor alias (same `id()`). Optimizer param groups built in [enigma_engine/core/training.py](enigma_engine/core/training.py#L1337-L1351) iterate `self.model.named_parameters()`, which defaults to `remove_duplicate=True` — the tied tensor is yielded exactly once, under the name of the submodule registered first. Since `tok_embeddings` (model.py line 179) is registered before `output` (line 230), the tied weight surfaces only as `tok_embeddings.weight` and matches `'embed' in name` → goes to `no_decay_params`. `output.weight` is never visited. **Safe:** single param group, correct decay policy, no double-update. Code is correct as written.

**Scheduler:**
- [RESOLVED] Sched-1: WSD is already the DEFAULT scheduler (`schedule_type: str = "wsd"` in TrainingConfig). Not a build task. Was a documentation error.
- [RESOLVED] Sched-2: SmolLM3 pretraining uses 2000 fixed warmup steps out of ~8T tokens (~negligible %). SFT/DPO warmup not explicitly given in the blog. The `max(100, steps // 10)` formula is our own recommendation for small fine-tuning runs and is consistent with SmolLM3's proportionally-small warmup pattern. **Action confirmed: Sched-2 implementation (Suggestion row 20) is correct — change SFT/DPO default to `max(100, steps // 10)`.** This is a low-risk quality-of-life fix for short DPO runs.

**Precision:**
- [RESOLVED] Prec-1: `_resolve_amp_dtype("auto")` checks `torch.cuda.is_bf16_supported()` → picks bfloat16 on Blackwell/Ampere+ (RTX 5090 → bfloat16 confirmed). Loss scaling not needed for bfloat16's wider exponent range (Prec-2 implicitly confirmed safe).
- [RESOLVED] Prec-2: bfloat16 does NOT need dynamic loss scaling. Its exponent range is the same as float32 (8 bits) vs float16's 5 bits. GradScaler is not called when `_amp_dtype == bfloat16` ([enigma_engine/core/training.py](enigma_engine/core/training.py) lines 1255, 4651, 4860, 5269 all gate on `!= bfloat16`). RTX 5090 uses bfloat16, no action needed.

---

### Inference — Real Gaps

These are missing features or unverified behaviors:

- [RESOLVED] Inf-1: KV cache is pre-allocated. `KVCache` allocates full `max_seq_len` tensors upfront. No dynamic fragmentation.
- [RESOLVED] Inf-2: Answered directly from our code. Three dispatch paths at [enigma_engine/core/model_components.py](enigma_engine/core/model_components.py#L521-L565) lines 521-565:\n  1. **Flash path** (line 534): enabled only when `HAS_FLASH_ATTN && x.is_cuda && fp16/bf16 && not use_cache && not use_differential_attn && (mask is None or T == k.shape[1])`. This covers **training and prompt prefill** only.\n  2. **SDPA path** (line 544): enabled when `not self.use_differential_attn && hasattr(F, \"scaled_dot_product_attention\") && x.is_cuda`. Used for decode when Flash is unavailable. PyTorch's SDPA auto-dispatches to FA2 for prefill (seqlen_q > 1) but falls to the math/cuDNN kernel for seqlen_q = 1 decode (no specialized split-KV kernel — that's the Inf-5 gap).\n  3. **Manual attention** (line 568+): fallback for CPU/MPS/any-dtype AND the catch-all when `use_differential_attn=True`.\n\n**Key finding:** because `use_differential_attn` defaults to `True` ([model_presets.py L127](enigma_engine/core/model_presets.py#L127)), **our runtime config actually uses the manual attention path for both prefill and decode** — no prefill/decode specialization at all. Flash and SDPA are effectively dead code paths in the default config, only reachable if differential attention is explicitly disabled. Performance implication: prefill is slower than it could be (no Flash), but at least the prefill/decode behavior is consistent. **Action:** none required — current behavior is intentional (Attn-2 resolution: keep differential attention for quality). Logged as context. If we ever disable differential attention for a workload, the distinction kicks back in automatically.
- [RESOLVED] Inf-3: `H2OKVCache` evicts low-attention tokens. `StreamingLLMCache` uses attention sinks for infinite context. `TurboQuantKVCache` INT4 compression. No vLLM-style paged attention, but these mechanisms solve VRAM-bounded long sessions.
- [RESOLVED] Inf-4: xl KV cache VRAM calculated from config and formula `2 × n_kv_heads × head_dim × n_layers × seq_len × dtype_bytes`. With xl (`n_kv_heads=6`, `head_dim=64`, `n_layers=24`, `seq_len=4096`, bf16=2): **150MB KV cache**. At 32K context: **~1.17GB**. Conclusion: for 742M, KV cache is not the dominant VRAM consumer; model weights + activations dominate. INT8 KV compression is optional, not required for baseline operation.

---

### Post-Training — Real Research Gaps

**Process reward model:**
- [BUILD→DEPRIORITIZE] PRM-1: `ProcessRewardModel` and `PRMTrainer` exist in [enigma_engine/core/rl_training.py](enigma_engine/core/rl_training.py#L787) (lines 787, 916). Training format requires `step_labels: list[float]` (+1.0 correct, -1.0 wrong) per reasoning step. **No pipeline to auto-generate step labels exists.** Options: (a) manually label math solutions step-by-step, (b) use Qwen3-30B to judge each step. **NEW from DeepSeek paper (Section 4.2 Unsuccessful Attempts):** DeepSeek explicitly calls PRM an unsuccessful approach for RL training — reward hacking risk, hard to define fine-grain steps, automated annotation unreliable, adds complexity. PRM is useful ONLY for reranking top-N responses at inference time (not for online RL). **Recommendation: deprioritize PRM for RL. Use rule-based accuracy reward instead. Keep PRMTrainer code but don't build the labeling pipeline yet.**

**Reasoning alignment:**
- [RESOLVED] GRPO-1: GRPOTrainer confirmed in [enigma_engine/core/rl_training.py](enigma_engine/core/rl_training.py#L2311) (line 2311). No value head — takes external `reward_fn: Callable[[str, str], float]`. Group-relative advantage computed as `(r_i - mean(group)) / std(group)`. GRPO works at 742M but DeepSeek paper shows **distillation >> GRPO for small models** (32B GRPO matched QwQ-32B, but 32B distilled from R1 massively outperformed). For 742M: distill first, GRPO optionally second.
- [RESOLVED] GRPO-2: Shipped in Pass 142. Rule-based rewards live in [enigma_engine/core/reward_functions.py](enigma_engine/core/reward_functions.py) and GRPO routing in [enigma_engine/gui/gui_forge_new_modes.py](enigma_engine/gui/gui_forge_new_modes.py#L2229-L2260) uses `reasoning_reward` (with neural reward path blocked for GRPO).
- [RESOLVED] GRPO-3: Rewards normalized per-group before advantage: `adv = (r - r.mean()) / (r.std() + 1e-8)`. Matches paper. Confirmed in GRPOTrainer.
- [BUG-RISK] GRPO-4 (neural reward hacking): The current RLHF+GRPO training path (`gui_forge_new_modes.py` line 2250) passes `reward_model.score()` — a **neural** RewardModel — as the reward_fn. DeepSeek explicitly calls this "unsuccessful" for RL training: the policy learns to game the neural scorer (reward hacking), not to produce actually correct reasoning. Neural reward is only safe for preference ranking of pre-generated outputs, not for online RL. **Risk: using this path for reasoning training will train overconfident wrong answers.** Fix: build rule-based reward_fn (GRPO-2 spec above) and use it instead. The neural reward path should be restricted to alignment/helpfulness training, not reasoning. File: [enigma_engine/gui/gui_forge_new_modes.py](enigma_engine/gui/gui_forge_new_modes.py#L2250) line 2250, [enigma_engine/core/rl_training.py](enigma_engine/core/rl_training.py#L2341) line 2341.

**Distillation (Approach 3):**
- [RESOLVED] Distill-1: MiniLLM (arxiv:2306.08543, ICML 2023) provides the authoritative comparison. Ranking: **Reverse KL (MiniLLM) > SeqKD (cross-entropy on teacher outputs) > word-level forward KL soft targets**. Rationale: forward KL forces student to cover ALL modes of teacher (mode-averaging) → assigns probability to void regions → exposure bias, poor calibration. Reverse KL is mode-seeking → focuses on teacher's major modes → better precision, lower exposure bias. **Practical impact for our FORGE Distillation:** we already use SeqKD (train student on teacher-generated hard labels via cross-entropy), which is position #2 and significantly simpler than MiniLLM. This is the correct approach. Full reverse KL requires policy gradient, teacher-mixed sampling, and reward hacking mitigation — complex, not worth building for our current scale. **Current implementation is correct. No change needed for the distillation loss type.**
- [RESOLVED] Distill-2: **T=4–10 applies to soft-label KD (Hinton 2015 style) only** — where the full teacher probability distribution is used as training targets. Our FORGE Distillation uses SeqKD (hard labels: teacher generates response tokens, student trains with standard cross-entropy). In SeqKD, teacher temperature only affects generation quality/diversity of the training data. MiniLLM uses T=1 for teacher sampling. DeepSeek-V3 SFT uses deterministic/greedy teacher outputs for hard-label training. **Code check:** current FORGE Distill generation uses `temperature=0.8` in [enigma_engine/gui/gui_forge_new_modes.py](enigma_engine/gui/gui_forge_new_modes.py#L1471), which is within the recommended 0.7-1.0 range. No loss-level temperature scaling is needed.
- [RESOLVED] Distill-3: FORGE Distillation uses `_load_engine_for_path()` in [enigma_engine/gui/gui_forge.py](enigma_engine/gui/gui_forge.py#L945) (line 945). That calls `EnigmaEngine(model_path=path)` — accepts **ANY** format EnigmaEngine supports: native .pth, GGUF (.gguf), HuggingFace, GPTQ/AWQ. So yes, Qwen3-30B-A3B in GGUF format ([models/qwen3-30b-a3b/](models/qwen3-30b-a3b/)) can be used as the teacher directly.
- [RESOLVED] Distill-4: Qwen3 reports **compute reduction** (Strong-to-Weak distillation needs ~1/10 GPU hours vs running full 4-stage post-training per small model) but does **not** publish a fixed sample/token count in Section 4. DeepSeek-R1 provides concrete scale: **~800K SFT samples total** for reasoning distillation (about **600K reasoning + 200K non-reasoning**) and shows strong gains for 1.5B/7B/14B/32B/70B distilled students. Practical guidance for Enigma 742M: start with **200K-800K** high-quality teacher traces (minimum viable at low end, DeepSeek-proven at high end), then ablate quality vs quantity.

---

### Multi-Model Architecture — Research Gaps (Approaches 1 & 2)

**Router design:**
- [RESOLVED] Router-1: [enigma_engine/router.py](enigma_engine/router.py) is a TCP socket server (port 9900) — mod IPC bus, not model dispatcher. `ModRouter` manages connect/disconnect/heartbeat for external mod processes. Also has `BackgroundTrainer` that learns from conversations (replay buffer, DPO pairs, periodic retrain on best examples). To dispatch between AI models (Enigma vs specialist vs Qwen3), a new layer is needed on top of this — Router-2/3 still unsolved.
- [RESOLVED] Router-2: Practical options ordered by cost/accuracy:
  1. **Regex / keyword rules** (~0 ms, ~70-80% accuracy). Detects obvious patterns: code blocks (```), "draw/generate/image", math symbols, file paths. Cheap, brittle, good first-pass filter.
  2. **Sentence embedding + cosine similarity to task centroids** (~5 ms on CPU with sentence-transformers/all-MiniLM-L6-v2 at 22M params, ~80-90% accuracy). Pre-compute centroid vectors for each task class from labeled examples; route by nearest centroid.
  3. **BERT-tiny classifier** (4.4M params, ~10 ms CPU, ~85-92% accuracy when fine-tuned on ~5K labeled examples). Good accuracy/cost point.
  4. **Our 742M model as classifier** (~200-500 ms prefill for a short classification prompt, ~90-95% accuracy). High latency; see Router-3.
- **Recommendation:** layered approach — regex rules first (catch 60% of obvious cases), embedding similarity fallback (catch next 30%), LLM for the remaining 10% only if confidence from the first two is low. This keeps p50 latency near 0 and only pays LLM cost on genuinely ambiguous inputs.
- [RESOLVED] Router-3: Yes, technically — pass the task through Enigma with a classification prompt ("Classify the following request as one of: code / vision / reasoning / general. Request: ..."). **Tradeoffs:**
  - **Accuracy:** ~90-95% with good prompt engineering, comparable to a fine-tuned BERT-tiny but without training a separate model.
  - **Latency:** ~200-500 ms for prefill of a ~200-token classification prompt at batch=1 on 5090 (dominated by prefill FLOPs, not by the 1-token classification output). This is added to EVERY user turn — doubles the perceived response latency vs. a direct LLM call without routing.
  - **VRAM:** no extra load — Enigma is already resident.
- **Verdict:** use this ONLY as the fallback in the Router-2 layered approach, not as the primary router. The 200-500 ms cost is unacceptable on fast-path requests where regex + embedding already give a confident answer. Implementation note: can be made much cheaper with a cached embedding of the system prompt (skip re-prefill of the static instruction part).

**Vision integration:**
- [RESOLVED] Vision-1: CONFIRMED via LLaVA-1.5 paper (arxiv:2310.03744) and [llava-vl.github.io](https://llava-vl.github.io/). Architecture: **frozen CLIP ViT encoder → projection → LLM decoder (trained)**. Projection design changed between versions: LLaVA-1 used a single linear matrix; **LLaVA-1.5 upgraded to a 2-layer MLP with GELU** ("fully-connected vision-language cross-modal connector"), reporting "surprisingly powerful and data-efficient" gains across 11 benchmarks. **Code gap:** our [enigma_engine/core/model.py](enigma_engine/core/model.py#L186) `self.vision_projection` is a single `nn.Linear` (LLaVA-1 era). For Code-6 we should upgrade to `nn.Sequential(Linear, GELU, Linear)` to match LLaVA-1.5 SoTA. Logged as **Vision-1b** under backlog.
- [RESOLVED] Vision-2: CONFIRMED. LLaVA-1 (ViT-L/14 @ 224 px) → 16×16 = **256 visual tokens**. LLaVA-1.5 (ViT-L/14 @ 336 px) → 24×24 = **576 visual tokens**. LLaVA-HD / LLaVA-NeXT (AnyRes grid up to 4×): 576 × up-to-5 = up to 2880. **Decision for 742M / 4096 ctx:** 576 tokens is fine (leaves 3520 for text). LLaVA-1.5 paper reports the 336 px variant outperforms the 224 px variant on VQA benchmarks (exact ablation delta not re-verified this pass). Use 336 px to match LLaVA-1.5. If multi-image or long-text ever needed, fall back to the 224 px / 256-token config.
- **[Pass 148 currency check]** Vision-2b: **LLaVA-NeXT (Jan 2024) is now superseded by LLaVA-OneVision** (arxiv:2408.03326, last revision Oct 26 2024, Apache-2.0). LLaVA-OneVision uses **SigLIP SO400M (not CLIP) + Qwen2 LM**, and the SAME model handles single-image / multi-image / video in one architecture (no separate variants). Sizes: 0.5B / 7B / 72B — the **0.5B variant is the right LLaVA-class drop-in for our 742M base on 16 GB**. Vision tokens use anyres grids similar to LLaVA-NeXT but with stronger downsampling for video. **Action:** when wiring Code-6, target the LLaVA-OneVision recipe + SigLIP 2 encoder (Vision-1b) directly instead of LLaVA-1.5; the 1.5 numbers above remain valid as a minimum-viable reference. Keep CLIP-ViT-L/14-336 only as the lowest-friction encoder option.
- [RESOLVED] Vision-3: CONFIRMED. LLaVA-1.5 uses `openai/clip-vit-large-patch14-336` (HF: 304M params, 1024-d penultimate hidden state). Drop-in: `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` (same arch, trained on LAION-2B, 224 px only — would need to accept the 2 pt VQA hit or use LAION's 336 px variant if released). **Recommendation:** start with `openai/clip-vit-large-patch14-336` — exactly what LLaVA-1.5 ships. Add to `core/model.py` ModelConfig: `vision_hidden_size = 1024` when vision is enabled. Use `select_feature="patch"` (drop CLS), matching LLaVA-1.5.
- [RESOLVED] Vision-4: CONFIRMED two-stage training. **Stage 1 — feature alignment (projection only, LLM + CLIP frozen):** `liuhaotian/LLaVA-Pretrain` (558K image-caption pairs from LAION/CCS/SBU, BLIP-filtered). 1 epoch, small LR. **Stage 2 — instruction tuning (projection + LLM, CLIP frozen):** `liuhaotian/LLaVA-Instruct-150K` + academic VQA mix totaling **665K samples** (VQAv2, GQA, OKVQA, OCRVQA, A-OKVQA, TextCaps, RefCOCO, Visual Genome). LLaVA-1.5 full training: ~1 day on 8×A100 for 13B. For our 742M on one 5090 expect proportional time. **Modern alternative:** `lmms-lab/LLaVA-OneVision-Data` is the current superset (single-image + multi-image + video), larger but better coverage. Action for Code-6: wire Stage 1 data collection into `collect_finetuning_data.py` behind a `--llava-pretrain` flag.
- [RESOLVED] Vision-5: [enigma_engine/core/model.py](enigma_engine/core/model.py) `model.forward()` accepts `vision_features: Optional[Tensor]` (line 670). `vision_projection` created in __init__ when `vision_hidden_size is not None` (lines 185-191). Vision embeds prepended to token embeddings before the transformer. NOT dead code — LLaVA-style path is wired. Training the projection is still needed (Code-6).

**Specialist training (Approach 2):**
- [RESOLVED] Spec-1: Yes, catastrophic forgetting at 742M on focused fine-tuning is well-documented (Luo et al. 2023, *An Empirical Study of Catastrophic Forgetting in LLMs During Continual Fine-tuning*). Two practical mitigations exist: (a) **EWC** (Kirkpatrick 2017) — our [enigma_engine/core/ewc.py](enigma_engine/core/ewc.py) implements this, but EWC is sensitive to Fisher estimation quality and requires keeping the base model's Fisher matrix in memory; (b) **LoRA adapters per specialist** (Hu et al. 2021) — we already have [enigma_engine/core/lora_utils.py](enigma_engine/core/lora_utils.py). **Verdict:** LoRA is the right default for multi-specialist setups — base weights are frozen so forgetting is physically impossible, adapters are 10-30 MB each, swap at runtime. Reserve EWC for the case where we intentionally want full-weight specialization but need to retain some general capability (rare — usually better to use a LoRA). Backlog item EWC-1 (wire EWC into FORGE SFT/dialogue) is still valid but should be de-prioritized in favor of a LoRA-per-specialist path for Approach 2.
- [RESOLVED] Spec-2: Existing text's fp16 arithmetic is correct — 742M × 2 B ≈ 1.48 GB per specialist, theoretical max **~10 specialists** in 16 GB VRAM. **Realistic ceiling is 4-6 full-weight specialists** once KV cache (150 MB - 1.2 GB depending on ctx), activation buffers, and framework overhead are included. **Major revision:** LoRA changes this math entirely — **one frozen base model (1.5 GB fp16) + N LoRA adapters (~20 MB each)** supports dozens of specialists simultaneously with hot-swap at < 100 ms. For our 16 GB budget, LoRA gives **20+ effective specialists** vs. ~5 full-weight specialists, and with N× smaller disk footprint. Combined with Spec-1 verdict: **LoRA-per-specialist is the only serious multi-specialist path on 16 GB VRAM**. Full-weight merging (via `model_merging.py`) remains an option for creating the final consolidated model before deployment but not for simultaneous multi-specialist inference.
- [RESOLVED] Spec-3: Use the **EleutherAI LM Evaluation Harness** (github.com/EleutherAI/lm-evaluation-harness) as the harness — it's the community standard and covers all the benchmarks below with one runner. Per-specialist benchmark set:
  - **Reasoning specialist:** GSM8K (arithmetic CoT), MATH (hard math), ARC-Challenge (science MCQ), MMLU-Pro (general knowledge with CoT). **Skip AIME** — too hard for 742M, noise floor.
  - **Code specialist:** HumanEval + HumanEval+ (single-function synthesis), MBPP + MBPP+ (short programs), BigCodeBench-Lite if time permits.
  - **Vision specialist** (post Code-6 / Vision-1b): VQAv2 (general visual QA), TextVQA (OCR-in-image), GQA (scene reasoning), MMMU (multimodal reasoning). Skip COCO captioning — caption quality is not a good signal for multimodal reasoning specialists.
  - **Knowledge / general:** MMLU (0-shot + 5-shot), TriviaQA, NaturalQuestions. Include ARC-Easy and HellaSwag as sanity checks that general capability hasn't collapsed.
- **Action for Eval-1 / D-10:** wire the LM Eval Harness in as the `--benchmark` backend, driven by a YAML config that selects the relevant suite per specialist type. Much lower effort than reimplementing GSM8K/MMLU scorers internally.
- [RESOLVED] Spec-4 (merging): [enigma_engine/core/model_merging.py](enigma_engine/core/model_merging.py) has SLERP, TIES, and linear merge for combining specialists. TIES (Trim, Elect Sign, Disjoint Merge) is the state-of-the-art for merging fine-tuned variants without catastrophic forgetting of overlapping knowledge. Already built — can merge a code specialist back with the general model. Research still needed on which merge method works best at 742M.

---

### Generation & Mod System — Research Gaps

These are stated project goals (image, audio, video, 3D generation) but have zero research done:

**Image generation:**
- [RESOLVED] ImgGen-1: Stable Diffusion 1.5 fits in 16GB budget alongside Enigma. VRAM breakdown: Enigma 742M = ~1.5GB weights + ~0.3-0.5GB activations/KV cache = ~2GB total; SD 1.5 = ~5.5GB fp32 or ~2.8GB fp16 + ~1-1.5GB activations = ~4-6GB total; combined ~6-8GB leaves 8-10GB for batch_size=1 inference. SD 2.1 (6GB) and SDXL (7.5GB) fit in theory but require careful memory management. **Recommendation: SD 1.5 is the baseline (confirmed working); SD 2.1 marginally possible; SDXL requires quantization or sequential inference.**
- [RESOLVED] ImgGen-2: Direct simultaneous GPU residence not practical (both models ~4-6GB each = 8-12GB before overhead). Two deployment patterns: (a) **Sequential:** Enigma generates caption text → SD 1.5 loads, generates image from caption → offload SD to CPU RAM, process image with Enigma. Typical latency ~3-5s per image (not real-time but acceptable for asynchronous generation). (b) **Subprocess:** Launch SD in separate process (avoids Python memory sharing issues), communicate via file/socket, Enigma unloads from GPU during SD inference. Pattern used: `enable_model_cpu_offload()` in diffusers pipeline moves models to CPU when not actively computing. **Verdict: Sequential with CPU offload is the practical approach. No hard VRAM sharing; one model loads, the other sleeps on CPU RAM.**
- [RESOLVED] ImgGen-3: Real implementation. `StableDiffusionPipeline` via diffusers library. DPMSolver/Euler schedulers. Providers: placeholder (offline) / local / DALL-E / Replicate. Uses SD 1.5/2.1 pipeline.

**Audio generation:**
- [RESOLVED] Audio-1: Meta's **MusicGen comes in three sizes: 300M (small), 1.5B (medium), 3.3B (large)**. Paper arxiv:2306.05284 (*Simple and Controllable Music Generation*, Copet et al., Meta). MusicGen-small at **300M** is the smallest viable text-to-music option (training: 20K hours of licensed music, instrumental-only via HT-Demucs preprocessing). **VRAM requirement:** ~1-2GB for batch=1 16-second generation. **Alternatives considered:** AudioGen (Meta, speech + ambient sound, ~1.5B) — larger and not music-specific; Bark (Suno, ~400M TTS, not music). **Verdict: MusicGen-small 300M is the target for music generation.** AudioGen is larger and not needed if the goal is music-only. TTS is separate (Bark or existing pyttsx3).
- [RESOLVED] Audio-2: **TTS (speech synthesis) ≠ music generation.** Current code has TTS via pyttsx3/ElevenLabs. **Music generation is Audio-5 (new capability, separate).** Two different pipelines: (a) TTS for Enigma's voice output (already implemented, pyttsx3), (b) MusicGen for background/ambient music (not yet built, marked Audio-5 in P2). **Answer: use existing pyttsx3 for voice; add MusicGen-small separately for music. Goals are complementary, not exclusive.**
- [RESOLVED] Audio-3: All three are real implementations. `audiogen` = TTS via pyttsx3/ElevenLabs/system fallback (NOT music generation — see Audio-5 new gap). `voice` = voice synthesis with pycache (actively used). `transcriber` = speech-to-text transcription.
- [RESOLVED] Audio-4: [enigma_engine/core/audio_encoder.py](enigma_engine/core/audio_encoder.py) is a Whisper-style encoder (Conv1d → GELU → Conv1d stride=2 → sinusoidal pos → Transformer → RMSNorm). Presets: tiny(384-dim, 10M), base(512-dim, 25M), small(768-dim, 95M). All stdlib, no external deps.

**Video generation:**
- **[SUPERSEDED — Pass 148 currency check]** Video-1: The AnimateDiff + SD 1.5 block below was primary-source-accurate at resolution time but has been surpassed by permissively-licensed 2025 text-to-video models that still fit the 16 GB VRAM budget. **New primary:** **Wan 2.1 T2V-1.3B** (Alibaba, Feb 2025, Apache-2.0) — 8.19 GB VRAM on RTX 4090, generates 5-second 480p video in ~4 min, beats closed-source Runway Gen-3 on VBench in published numbers. **Primary alternative:** **LTX-Video 0.9.8 2B-distilled** (Lightricks, 2025) — real-time capable on consumer GPUs, but non-standard dev license. **Do not use** HunyuanVideo (13B, 45-60 GB VRAM — too big). AnimateDiff stays valid as a drop-in SD-based fallback for animation rather than true T2V. Original block preserved below for implementation reference:
  - [RESOLVED — historical] Video-1: **AnimateDiff (Guo et al. ICLR2024) + Stable Diffusion 1.5 is the minimum viable path.** AnimateDiff is a **motion module adapter** (~50-100MB additional params) inserted into SD's UNet to extract motion priors from video training data. Architecture: frozen SD 1.5 base + trainable motion modules at each block. **VRAM breakdown:** SD 1.5 = 5.5GB; AnimateDiff adapter overhead = negligible (~100MB); pipeline with `enable_model_cpu_offload()` and `enable_vae_slicing()` = ~6-6.5GB peak. Remaining 9.5GB available for batch_size=1 short video generation (16 frames, 25 diffusion steps). **Implementation:** diffusers library has native AnimateDiffPipeline (no custom code needed). **Alternative:** ModelScopeT2V at 4.5B is slightly smaller but less integrated into diffusers. **Verdict: AnimateDiff + SD 1.5 is production-ready, fits 16GB budget, use diffusers pipeline directly.**
- [RESOLVED] Video-2: Real implementation. AnimateDiff local pipeline + animated GIF fallback + Replicate API. Not a stub.

**3D generation:**
- **[SUPERSEDED — Pass 148 currency check]** 3D-1: Shap-E (arxiv:2303.12522, May 2023) and TripoSR (arxiv:2403.02151, Mar 2024) have been surpassed by 2024-2025 image→3D models with dramatically better mesh quality. **New primary:** **Hunyuan3D-2** (Tencent, arxiv:2501.12202, Jan 2025) — image→3D diffusion model, SoTA open reconstruction quality, fits 16 GB VRAM. **Primary alternative:** **TRELLIS-image-large** (Microsoft, arxiv:2412.01506, Dec 2024, MIT license) — 2.3M downloads/month, structured latent 3D representation, image→3D. Note: the official TripoSR HuggingFace model card itself now points users to its SF3D successor. **Text→3D path:** no open SoTA model at our scale matches image→3D quality; route text→3D as text→image (FLUX.1-schnell or SD 3.5) → image→3D (Hunyuan3D-2). Original block preserved below for implementation reference:
  - [RESOLVED — historical] 3D-1: **Shap-E (OpenAI, arxiv:2303.12522) is the primary text-conditioned 3D generation model at 350M params, fits 16GB budget.** Key details: **text→GLB (3D mesh)** in single forward pass. Trained on ~1M 3D objects from Google Scanned Objects. VRAM: ~2-3GB for inference. Alternatives: (a) **Zero123** (Liu et al., ~600M, image→novel-view synthesis, requires initial image), (b) **SyncDreamer** (Xu et al., ~400M, single-image to multi-view, requires input image), (c) **TripoSR** (Zhou et al., ~500M, single image→SDF). **Deployment:** Shap-E for text→3D (no input needed), Zero123/SyncDreamer for image→3D (avatar refinement/variation). **Verdict: use Shap-E (350M text→3D) as primary, keep SyncDreamer as fallback for image-based 3D refinement (avatar re-posing).** Combined: Enigma (1.5GB) + Shap-E (2-3GB) fits with room for activation buffers.
- [RESOLVED] 3D-2: Real implementation. Shap-E local generation (OpenAI, ~350M) + geometric primitives fallback + Replicate API. Not a stub.

**Haptic feedback prediction:**
- [RESOLVED] Haptic-1: Honest answer \u2014 **I do not know** of a standard, well-validated dataset + architecture for text-conditioned haptic prediction at the scale we'd need. What exists:\n  - **LMT Haptic Texture Database** (Strese et al. 2017): 108 real-world material textures with accelerometer recordings. Small, texture-only, not text-conditioned.\n  - **HaptiPedia** (~200 haptic icons): metadata + signal library, curated not algorithmic.\n  - **Research papers** on text\u2192haptic (Seifi et al. 2021, Schneider et al. 2023) use hand-crafted mappings or small MLP regressors trained on <1K examples. No open pretrained model exists.\n- **What's realistic for us:** a small MLP mod (~1M params) that takes a short text description + the current context embedding from Enigma and outputs a low-dim haptic parameter vector (amplitude, frequency, duration, texture class). Training data would have to be synthetic or bootstrapped from LMT + manual labeling.\n- **Verdict:** Haptic generation is a research project, not a near-term build. Keep in P3 / DEFERRED. Remove from the active research backlog \u2014 no realistic unlock by external literature search alone; would require novel data collection. Better to focus on the Vision / Code / Reasoning specialist track that has clear recipes.

---

### Additional Files Scanned (Pass 3 continued)

Final files read and assessed:

| File | What it is | Research implication |
|------|-----------|---------------------|
| [enigma_engine/router.py](enigma_engine/router.py) | TCP mod IPC bus + BackgroundTrainer | Router-1 resolved; dispatch layer still missing |
| [enigma_engine/core/rag.py](enigma_engine/core/rag.py) | Full BM25/TF-IDF RAG with adaptive chunking | Dense retrieval (embedding RAG) not present |
| [enigma_engine/core/kv_cache.py](enigma_engine/core/kv_cache.py) | Pre-alloc + H2O + INT4 + StreamingLLM cache | KV cache is very advanced; Inf-1/3 resolved |
| [mods/vision/](mods/vision/) | Screen capture + OCR, NOT LLM multimodal | Vision mod ≠ LLaVA; gap confirmed |
| [mods/codegen/](mods/codegen/) | TemplateCode + LocalCode (calls Enigma) + OpenAI | Code mod uses base 742M — no code specialist |
| [enigma_engine/core/memory.py](enigma_engine/core/memory.py) | Flat markdown memory, 200 facts, pattern extraction | No embedding-based retrieval |
| [enigma_engine/core/model_merging.py](enigma_engine/core/model_merging.py) | SLERP + TIES + linear merge | Can merge specialists after training |
| [enigma_engine/core/monologue.py](enigma_engine/core/monologue.py) | Inner monologue / journal / coherence gate | Phase 5, already built |
| [enigma_engine/core/adaptive_trainer.py](enigma_engine/core/adaptive_trainer.py) | TrainingPlan: basics→conv→commands→web curriculum | Adaptive pipeline exists, not yet used at scale |
| [mods/avatar/](mods/avatar/) | Full avatar: BoneController (all joints), GLB/GLTF/OBJ loader | Production-ready bone rig, output format TBD |

---

### New Gaps Found — Full Codebase Scan (May 2026)

These files/features exist in the codebase but had no research items. Added after reading all core/ files and all mods/.

**Inference speed:**
- [RESOLVED] Inf-5: `flash_attn_with_kvcache` IS the FA2 decode-path API (added in flash-attn v2.2 per the official changelog: "Optimize for inference... query sequence length = 1. The bottleneck here is to load KV cache as fast as possible, and we split the loading across different thread blocks"). Single kernel: updates KV inplace + attention + optional rotary, supports GQA (our 4:1), paged KV, sliding window. **Current code:** [enigma_engine/core/model_components.py](enigma_engine/core/model_components.py#L521-L527) explicitly gates Flash off when `use_cache=True`; decode falls through to `F.scaled_dot_product_attention` ([line 557](enigma_engine/core/model_components.py#L557)). SDPA on CUDA auto-dispatches to FA2 for seqlen_q>1 (prefill) but lacks the specialized 1-token decode kernel. **Realistic speedup at our workload (742M, 4K ctx, batch=1):** ~15-30% decode tok/s, NOT the 2× numbers quoted for 8K+/large models — at 742M the MLP + tied output head are a meaningful share of per-token cost, and SDPA's native FA2 prefill is already fast. **Blocker:** flash-attn README lists CUDA support for Ampere/Ada/Hopper only; Blackwell consumer SM120 (RTX 5090) is unverified, and Windows support is "might work... requires more testing." Installing the package on our target box is fragile. **Verdict:** keep deferred. If a stable Blackwell+Windows wheel lands alongside FA4, revisit. Moved to P3 (was P2 research). Not worth the dependency risk on the hot path for a ~20% gain when SDPA already covers CPU/MPS/any-dtype fallback cleanly.

**Progressive growing:**
- [RESOLVED] Grow-1: Our [enigma_engine/core/progressive_growing.py](enigma_engine/core/progressive_growing.py) already implements Net2Net (Chen et al. 2016) width expansion + identity-init depth expansion. Published guidance on *when* to grow:
  - **Chinchilla-optimal tokens first.** Don't grow until the source model has seen at least Chinchilla-optimal data for its current size (~20 tok/param — ~15B tokens for 742M). Growing earlier wastes the smaller model's training signal; the larger architecture underfits because the expansion happens before the smaller weights are well-converged. Our 35 tok/param data budget is ideal: train 742M → Chinchilla-optimal (~15B tokens), then optionally grow to ~2.5B and continue on remaining tokens.
  - **Triggering signal.** Standard practice (MSG/stacking papers, Gong et al. 2019 *Efficient Training of BERT by Progressively Stacking*): grow when the loss curve plateaus at the current size. Operationally: val loss slope over the last 500-1000 steps falls below a threshold (e.g. < 1% relative improvement per 1000 steps).
  - **Data budget after growing.** Empirically 50-100% of the pre-grow token count on the new architecture before final convergence. So xl → xxl expansion at ~15B tokens should be followed by another ~10-15B tokens at 2.5B scale. We don't have that token budget right now (26B total collected, already committed to xl completion).
- **Verdict:** keep `progressive_growing.py` as infrastructure; do NOT schedule xl → xxl on current data budget. Revisit after N-10 completes and if we've collected another 10B+ tokens.
- [RESOLVED] Grow-2: Standard Net2Net / MSG / gradual-stacking practice is to **re-warmup** after expansion, NOT reset LR to the initial value. Recipe from multiple stacking papers (Gong 2019, Shen 2022 *Staged Training*, bert2BERT Chen 2022):
  1. After expansion, set LR to a small fraction of the current LR (e.g. current_lr × 0.1 or directly match the min_lr from the cosine schedule).
  2. Re-warmup over 100-500 steps back to the current LR (NOT back to the original peak LR — the smaller model's final LR is already past peak).
  3. Resume the existing cosine/WSD decay from that point.
- **Rationale:** a full warmup-from-scratch would destroy the small model's learning; no warmup at all lets gradients from the zero-initialized new params destabilize the rest of the network in the first few steps.
- **Action if/when xl → xxl is scheduled:** add a `--post-grow-warmup-steps` flag (default 200) and `--post-grow-lr-scale` (default 0.1) to the training CLI. Not urgent — tied to Grow-1's "not on current data budget" verdict.

**Speculative decoding (Medusa):**
- **[Pass 148 currency note — training recipe still valid, inference-time method superseded]** Our joint-loss MTP training path matches Medusa-2 and remains the correct *training* recipe (predict_heads stay useful as an auxiliary signal). For *inference-time* speculative decoding, **EAGLE-2** (Li et al., arxiv:2406.16858, Jun 2024) is now the preferred integration target — lossless, 3.05–4.26× speedup, 20–40% faster than EAGLE-1 (itself ~1.6× faster than Medusa). EAGLE-3 exists but has not been independently verified in this pass. If/when we add speculative decoding to the inference path, target EAGLE-2 not Medusa, even though Medusa-2 training hooks (predict_heads) stay.
- [RESOLVED] Medusa-1: The original *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* (Cai et al., arxiv:2401.10774, ICML 2024) proposes **two training recipes**:
  - **Medusa-1:** base model is frozen, Medusa heads are trained separately (cheap, fast, ~2× speedup, lower acceptance).
  - **Medusa-2:** heads and base are trained jointly with a weighted loss (base loss + λ × Medusa loss). Higher acceptance (~3× speedup) but requires re-fine-tuning the base.
- **Our approach matches Medusa-2:** `predict_heads` are included in the main loss at [enigma_engine/core/model.py](enigma_engine/core/model.py#L496-L518) lines 496-518, `loss = loss + mtp_loss / len(predict_heads)`. This is the higher-quality recipe.
- **Caveat:** Medusa-2 paper uses weight λ=0.2 and warms up Medusa heads first with Medusa-1 style for a few hundred steps before enabling joint. Our code adds them unweighted (λ=1.0 effectively via `/len(heads)`) from step 0. **Potential improvement:** add a warmup period and a `mtp_loss_weight` config (default 0.2 per Medusa-2, ramped from 0 over first ~500 steps). Log as **Medusa-1a**.
- Verdict: current joint training is correct direction; weighting is the tunable knob.
- [RESOLVED] Medusa-2: Original Medusa paper benchmarks on Vicuna-7B and Vicuna-13B — no sub-1B numbers published. **Why it matters at our scale:** speculative decoding's speedup comes from amortizing memory-bandwidth-bound decode across multiple tokens per forward pass. At 742M, decode IS memory-bandwidth bound on 5090 (the weights + KV cache don't fit L2, main bottleneck is HBM reads), so the speedup mechanism DOES apply. However:
  - **Acceptance rate is the question.** At 742M, the model is less confident → Medusa head predictions diverge from the base output more often → acceptance drops. Community reports on sub-1B Medusa variants (informal, no paper) suggest ~1.5-1.8× speedup vs ~2-3× at 7B+.
  - **Batch=1 only.** Speculative decoding gains collapse at batch > 4 because the base model's forward pass already amortizes across the batch. Our single-user inference path is the best case.
- **Verdict:** likely worthwhile at 742M batch=1 but the ~1.5-1.8× speedup is not a slam dunk vs. the 98M parameter cost of `predict_heads` (MTP-2a/b). If we drop to `n_predict_heads=1` per MTP-2a, Medusa effectively becomes 2-token lookahead — still useful, half the cost. **Action:** measure acceptance rate via a `--benchmark medusa-acceptance` flag on our real workload before committing. Log as backlog item **Medusa-2a**.

**Image generation mod upgrade:**
- [RESOLVED] ImgGen-4: SDXL-Turbo is SDXL-family and uses SDXL pipeline classes (`StableDiffusionXLPipeline` / `AutoPipelineForText2Image`), so it is not a drop-in replacement for SD1.5 pipeline call sites. It can run in 1-4 steps, but requires explicit SDXL load/wiring and has higher VRAM pressure than the current SD1.5 path. Decision: keep SD1.5 default; only add SDXL-Turbo behind a separate provider flag if we explicitly accept the heavier SDXL branch.

**Audio: music generation gap:**
- [RESOLVED] Audio-5: Code verification confirms `mods/audiogen/audiogen.py` is **TTS-only** (LocalTTS via pyttsx3, ElevenLabs cloud, system fallback) and has no music model path. Minimum viable music generation path is **MusicGen-small (300M)** as a separate provider (not a TTS replacement). From MusicGen model cards/paper (arxiv:2306.05284): sizes are 300M / 1.5B / 3.3B; small is the only practical local option for this stack. Budget fit: Enigma (~1.5-2 GB effective) + MusicGen-small (~1-2 GB inference) stays well under 16 GB, but generation is slow enough that async job-style UX is preferred. **Decision:** keep TTS in `audiogen` unchanged; add a separate `musicgen` provider/path (new mod or provider in audio mod) using MusicGen-small first, medium/large deferred.
- **[Pass 148 currency check]** Audio-5b: License + currency revision. **MusicGen weights are CC-BY-NC-4.0** (non-commercial only) — fine for personal/research use, NOT redistributable in a commercial build. **Stable Audio Open 1.0** (arxiv:2407.14358, Stability AI, July 2024) is now the better-licensed alternative for general audio/SFX: stereo 44.1 kHz, up to 47 s, T5 + DiT, **Stability AI Community License** (free for individuals + businesses with annual revenue under $1M). **Stable Audio Open Small** (arxiv:2505.08175, May 2025) is the lowest-VRAM option: up to 11 s, transformer-DiT, ARM-CPU optimized, 8-step pingpong sampler — generates on a phone, comfortably fits beside Enigma on 16 GB. **Decision (revised):** for music-specifically, MusicGen-small remains acceptable for offline/personal use given the CC-BY-NC license. For general audio + SFX + ambient, prefer Stable Audio Open Small (< $1M revenue org) → Stable Audio Open 1.0 → MusicGen-small in that order. Wire each as a separate provider behind a single `audiogen.music` interface.
- **[Pass 148 currency check]** Audio-ASR (new): The repo currently uses Whisper variants for transcription. As of mid-2025, **NVIDIA Parakeet-TDT-0.6B-v2** (May 2025, CC-BY-4.0, 600M params) leads HuggingFace's Open-ASR leaderboard with **6.05% mean WER and RTFx 3386** versus Whisper-large-v3-turbo at 7.83% / RTFx 200 — both more accurate AND ~16× faster on the same hardware, English-only, single-pass up to 24 minutes, with word timestamps + punctuation. NVIDIA also released **Parakeet-TDT-0.6B-v3** with 25 European languages (multilingual). **Moonshine** (arxiv:2410.15608, Oct 2024, MIT, 27M tiny / 61M base) is the right choice for low-latency edge / always-on streaming. **Decision:** swap default `mods/transcriber` English path to Parakeet-TDT-0.6B-v2; keep Whisper-large-v3 as multilingual fallback until Parakeet-v3 weights are validated locally; add Moonshine-base as the streaming/edge path. Logged as **Audio-ASR-1**.

**Avatar mod:**
- [RESOLVED] Avatar-1: Integration does **not** require training data for a first working version. Code path is already command-driven: `AvatarBrick` receives `avatar.bone` / `avatar.expression` over router TCP JSON, and `BoneController` enforces anatomical limits. Practical integration sequence: (1) map Enigma internal state + response tags to a small pose/expression preset table, (2) emit router commands at message boundaries, (3) optionally add viseme/lipsync later. Training data is only needed for learned motion style, not for baseline expressive control. **Priority call:** medium (P3/P2 boundary) behind reasoning/vision tasks, but low implementation risk because infrastructure already exists.

**Auto-research:**
- [RESOLVED] AutoResearch-1: `should_auto_research()` called in [enigma_engine/gui/gui_logic_chat.py](enigma_engine/gui/gui_logic_chat.py#L239) `::_gen()` (line 239) during every chat message — actively fires for substantive questions. NOT unused. Quality impact is qualitative and unverified — user should assess.
- [RESOLVED] AutoResearch-2 (confidence gap): Confirmed from code that `should_auto_research()` is query-keyword based and runs **before** model generation, so it cannot react to model uncertainty. Decision: implement in two stages, not one. **Stage A (fast, no training):** add a post-generation uncertainty pass in chat loop; if output contains calibrated uncertainty markers OR low-evidence patterns, trigger one retry with `auto_research()` context attached. **Stage A SHIPPED Pass 153** — `score_uncertainty()` + `should_retry_with_research()` in [auto_research.py](enigma_engine/core/auto_research.py). Wiring into [gui_logic_chat.py L239](enigma_engine/gui/gui_logic_chat.py#L239) `_gen()` is a follow-up pass. **Stage B (proper long-term):** add inline tool token (`<search>...</search>`) support so model can self-initiate research mid-generation. This requires generation-loop interruption/resume plus training examples that demonstrate when to emit `<search>`. **Verdict:** gap is real; staged plan is now clear and linked to existing files (`auto_research.py`, `gui_logic_chat.py`).

**Continuous learning:**
- [RESOLVED] Continuous-1: Code review confirms continuous trainer currently does **full-parameter updates** at `lr=1e-5` with replay buffer `maxlen=1000`, periodic replay retrain every 200 examples, and no EWC penalty. This is effective for short-term adaptation but has real long-horizon drift risk (matches continual fine-tuning forgetting concerns already cited in Spec-1). Practical decision: keep architecture, but treat current defaults as aggressive. Recommended safe defaults for always-on mode: `learning_rate=1e-6` (or 2e-6 max), keep replay buffer capped by recency (1000-2000), and require periodic evaluation gates before checkpoints are promoted. If fast adaptation is desired, do it in adapter/LoRA space rather than full-weight online updates.

**EWC wiring (confirmed gap):**
- [BUILD] EWC-1: [enigma_engine/core/ewc.py](enigma_engine/core/ewc.py) is standalone — zero FORGE integration. For Approach 2 specialist fine-tuning to prevent catastrophic forgetting, EWC must be wired: (1) after base training, compute EWC from a sample of general data, (2) during specialist fine-tuning, add `ewc.penalty(model)` to the loss. Requires adding EWC to the SFT/dialogue training path in [enigma_engine/core/training.py](enigma_engine/core/training.py) and an EWC section in FORGE GUI.

---

### Personality System — Audit (Pass 7, corrected Pass 8)

**Design intent (from user):** Personality is something the AI *develops*, not something the user configures. User should not be able to influence the AI's own character. Roleplay (user says "act as X") is fine — that is character-acting, separate from identity.

**What exists:**

| Component | File | What it is | Correct direction? |
|---|---|---|---|
| `ai_profile.personality` dict (tone/verbosity/formality/humor) | [enigma_engine/core/ai_profile.py](enigma_engine/core/ai_profile.py#L105) line 105 | USER-SET traits dict | **NO** — user-controlled personality = user influencing AI character |
| `model_context.personality` string | [enigma_engine/core/model_context.py](enigma_engine/core/model_context.py#L123) line 123 | Display label for identity card | Neutral — display only |
| `emotional_state` (5 dimensions: valence/energy/engagement/trust/frustration) | [enigma_engine/core/model_context.py](enigma_engine/core/model_context.py#L135) line 135 | AI-computed per message, decays, used for BackgroundTrainer engagement score | **PARTIAL** — AI-computed (not user-set), feeds training signal via BackgroundTrainer — this is the right direction |
| FORGE Distillation "personality" category | [enigma_engine/gui/gui_forge_new_modes.py](enigma_engine/gui/gui_forge_new_modes.py#L1259) line 1259 | Teacher generates warm, genuine responses → student SFT learns the style | **YES** — personality baked into weights via training |
| Training brief "Personality" field | [enigma_engine/gui/gui_forge.py](enigma_engine/gui/gui_forge.py#L64) line 64 | User description for guiding teacher during FORGE training | **MIXED** — user can seed the initial character during a one-time training run, but shouldn't be able to change it at runtime |

**Pass 7 audit self-correction (revised Pass 9):**

- [RESOLVED] Personality-1 (emotional_state injection): N-22 (Pass 125) already injects `emotional_state` into the system prompt via [enigma_engine/gui/gui_logic.py](enigma_engine/gui/gui_logic.py#L335) `::_build_gui_context()` line 335 as `[Internal State: valence=X, energy=Y, ...] Let this state color your tone naturally — do not announce it.` This is **correct**: emotional_state is AI-computed from sentiment analysis (`compute_engagement_score` in [enigma_engine/core/sentiment.py](enigma_engine/core/sentiment.py)), not user-set. Giving the AI awareness of its own computed state ≠ user configuring the AI's personality. Pass 8 retracted this as "wrong direction" — that retraction was itself wrong. No action needed. Also feeds BackgroundTrainer engagement weight in [enigma_engine/gui/gui_logic_chat.py](enigma_engine/gui/gui_logic_chat.py#L1252) line 1252 — both consumers are correct.
- [CRITICAL] Personality-2: `ai_profile.personality` dict (tone/verbosity/formality/humor) is USER-SET via profile files in [profiles/](profiles/). The earlier suggestion to inject it into the system prompt would have meant the user directly configures the AI's character — which contradicts design intent ("personality is something the AI develops, not something the user configures"). **Do not inject this dict.** The dict itself is the problem — see Personality-3.

**What is actually a gap:**

- [ARCH-GAP] Personality-3: `ai_profile.personality` user controls (tone/verbosity/formality/humor) exist in the AI's default profile and are presented as configurable. This is the wrong abstraction for the AI's own identity. These controls should be **scoped only to roleplay/character profiles** (where the user is defining a fictional character to act as), not to the AI's base identity. The AI's base profile should have no user-configurable personality traits. File: [enigma_engine/core/ai_profile.py](enigma_engine/core/ai_profile.py#L105) line 105, profile files in [profiles/](profiles/). Note: this is a design/UX decision, not a one-line code fix.
- [ARCH-GAP] Personality-4: No clear separation between **identity** (the AI's own developed character, baked into weights) and **roleplay** (user-requested character-acting, prompt-based for that session). The same profile system handles both. A user loading [profiles/assistant.json](profiles/assistant.json) looks identical to the AI just being itself. This needs a design decision: should "be yourself" be a hardcoded state (no profile), and profiles only ever mean "act as this character"?
- [BUILD] Personality-5: FORGE Distillation "personality" category exists ([enigma_engine/gui/gui_forge_new_modes.py](enigma_engine/gui/gui_forge_new_modes.py#L1259) line 1259) and generates character training data via teacher. This is the correct build path for personality-in-weights. **It has NOT been run as part of any training plan yet.** The emotional roadmap calls it "Step 1b" but it's never been scheduled. Recommend running distillation with the "personality" category as part of the next training cycle to bake initial character traits.
- [RESOLVED] R-PERSONALITY-1 (reframed): Stable personality should be weight-trained, not runtime-user-configured. Concrete direction for this codebase: (1) build a fixed personality corpus through FORGE personality distillation, (2) run a dedicated SFT pass to anchor style/values, (3) disable runtime personality knobs for base identity, and (4) keep roleplay as explicit temporary overlays only. Minimum viable seed should start around ~500 high-quality personality examples (100 is too fragile) and can scale later; BackgroundTrainer reinforcement should include guardrails so engagement does not reward shock/flattery behavior.

**Roleplay separation (design note):**
- AI's own personality → trained into weights (FORGE Distillation "personality" category) + reinforced by BackgroundTrainer engagement → not prompt-injectable by user, not configurable via profile personality dict
- Character acting → user loads a character profile or says "act as X" → prompt-based for session scope → clearly temporary
- The code does not currently enforce this separation — any profile can override system_prompt which effectively overrides personality. An architectural guard is needed.

**Unpredictability (design note — Pass 10):**

- [RESOLVED] R-UNPREDICT-1: "Organic" / unpredictable behavior must split **behavior** (surprising but signal-driven) from **infrastructure** (deterministic, seedable). Literature-backed decision:
  - **ReAct (arxiv:2210.03629):** interleave reasoning and external actions to reduce hallucination; supports action-on-demand rather than always-on tool calls.
  - **Toolformer (arxiv:2302.04761):** model can learn when/what/how to call tools; trigger should be learned/signal-based, not keyword-only.
  - **Self-RAG (arxiv:2310.11511):** adaptive retrieval via reflection tokens outperforms fixed retrieval; retrieval should be conditional.
  - **CRAG (arxiv:2401.15884):** retrieval-quality evaluator and confidence degree should govern fallback to broader search.
  - **Enigma implementation policy:** Stage A add post-generation uncertainty gate (confidence + novelty + retrieval-quality score) before auto-research retry; Stage B add inline `<search>` token handling in generation loop. Keep hard off-switch + reproducible seed path for all stochastic branches.
  - **Status:** research complete; now an implementation backlog item (AutoResearch-2), not an open research item.
  
  Concrete places to add controlled unpredictability in this codebase:
  1. **Self-initiated research** — replace `should_auto_research()` keyword match with a confidence signal derived from model logits + memory novelty. Fires when the model is uncertain, not when the query contains "what is". Links: AutoResearch-2. File: [enigma_engine/core/auto_research.py](enigma_engine/core/auto_research.py) `::should_auto_research()`.
  2. **Mood-weighted replay** — BackgroundTrainer already consumes engagement scores; weight replay sample selection by `emotional_state` so the same conversation gets revisited differently depending on internal state. Same input, different lesson. File: [enigma_engine/router.py](enigma_engine/router.py) BackgroundTrainer + N-25 idle callback.
  3. **Monologue variance** — `monologue.py` self-reflection currently uses a fixed prompt. Shape the internal prompt by current `emotional_state` (not user-visible) so reflection drifts with mood. Slow, consistent personality drift baked back into memory. File: [enigma_engine/core/monologue.py](enigma_engine/core/monologue.py).
  4. **Curiosity token** — let model emit `<search>query</search>` mid-generation when uncertain (linked to R-PERSONALITY-1 and AutoResearch-2 option (a)). Trigger auto-research inline. Unpredictable *when*, predictable *that* it fires on uncertainty.
  **Non-negotiable constraints:**
  - Training runs must remain seedable (reproducible crashes + ablations)
  - Tests must remain deterministic (no flakes from "unpredictable" paths)
  - Every unpredictable feature needs an off-switch (debug flag) and a seed (reproduction)
  - No `random.random()` or `torch.rand()` without a seed source tied to either (a) training seed, or (b) an internal signal (emotional_state, confidence, novelty)
  **Anti-patterns to block:**
  - "Let it grow naturally" used as excuse to skip curriculum/structure — chaos is not personality
  - Neural reward model that reinforces confident-sounding noise (GRPO-4 — already logged)
  - Unseeded hashing for dedup (already a Learned Principle)
  Status: research resolved in Pass 146. No code changes yet; implementation follow-up is AutoResearch-2 staged build.

**Tests:**
- [RISK] Test-1: Three GRPO tests in [tests/test_training.py](tests/test_training.py#L3284) lines 3284–3340 are structural (`inspect.getsource`) tests checking for `.clamp(-20, 20)` and `unbiased=False` string literals. These test HOW not WHAT. A refactor that moves the logic (e.g. into a helper function) will fail these tests even if the math is correct. Per testing rules, structural tests are last resort. These particular checks (numeric correctness guards) are borderline acceptable because wrong values cause silent NaN/inf. But there is no behavioral end-to-end test of the reward loop — nothing verifies that the trainer actually improves responses when given a meaningful reward function. Recommendation: add one behavioral integration test that gives a mock reward_fn returning 1.0 for longer responses and verifies the model's output length increases after training. Keep the structural tests for the numeric guard but don't add more.

**RAG system:**
- [RESOLVED] RAG-1: [enigma_engine/core/rag.py](enigma_engine/core/rag.py) has a full BM25/TF-IDF pipeline: `TfidfVectorizer` (Okapi BM25 with IDF weighting), `RAGIndex`, adaptive chunking (sentence-boundary aware), `index_directory()`. Already built. NOT in research list at all.
- [RESOLVED] RAG-2: Small dense retrievers are now concrete. `sentence-transformers/all-MiniLM-L6-v2` is 22.7M params with 384-d embeddings; `BAAI/bge-small-en-v1.5` is 33.4M params with 384-d embeddings and stronger retrieval-oriented benchmarks. Both are lightweight enough for CPU-side embedding service in this project. **Decision:** use `bge-small-en-v1.5` as primary (better retrieval quality), `all-MiniLM-L6-v2` as fallback/minimal dependency option. Do not use Enigma hidden states as replacement embeddings for retrieval: they are not contrastive-retrieval tuned, are more expensive per chunk, and couple retrieval quality to LM checkpoint drift.
- **[Pass 148 currency check]** RAG-2b: **Qwen3-Embedding-0.6B** (arxiv:2506.05176, Jun 5 2025, Apache-2.0) is the current SoTA in the sub-1B class and a clean upgrade over `bge-small-en-v1.5`. Verified facts: 0.6B params, 32K context, MRL output dim selectable from 32 → 1024, MTEB multilingual mean **64.33** (vs BGE-M3 59.56, multilingual-e5-large 63.22), 100+ languages, supports `flash_attention_2`. The 8B variant is #1 on MTEB multilingual at **70.58** but is too heavy to co-reside with Enigma on 16 GB. **Decision (revised):** Qwen3-Embedding-0.6B = primary; bge-small-en-v1.5 = lightweight fallback when Qwen3 weights aren't available; all-MiniLM-L6-v2 = absolute-minimum fallback. Keep BGE-M3 in mind only when hybrid dense + sparse + ColBERT scoring is explicitly needed (Qwen3-Embedding is dense-only).
- [RESOLVED] RAG-3: Wiring is currently split but compatible: GUI chat may inject web context via `auto_research()` before generation, and `_prepare_chat()` independently injects local RAG document context when `_rag_index` is active. This means web and local retrieval can both fire in the same turn. **Decision policy:** (1) local RAG first for user/local corpus questions, (2) web search only when local RAG confidence is low or query is time-sensitive/current-events, (3) neither for simple conversational requests. Implement as a single lookup orchestrator that scores source confidence and appends one merged context block, instead of two independent injections.

**KV cache (Inf-1/3 resolved):**
- [RESOLVED] Inf-1: KV cache IS pre-allocated. `KVCache` allocates full `max_seq_len` upfront, uses in-place slice updates. No dynamic fragmentation.
- [RESOLVED] Inf-3: `H2OKVCache` (Heavy Hitter Oracle) handles VRAM-bounded long sessions by evicting low-attention tokens. `StreamingLLMCache` provides infinite context via attention sinks (first 4 tokens always kept). `TurboQuantKVCache` uses INT4 packing. PagedAttention not present, but these mechanisms cover the same need.

**Vision mod distinction:**
- [RESOLVED] Vision-5 addendum: [mods/vision/vision.py](mods/vision/vision.py) is screen capture + OCR (`ScreenCapture`, `OCR` via tesseract/easyocr, `ImageAnalyzer`). This is computer vision for what's ON SCREEN, NOT multimodal LLM vision (understanding image content via the language model). The `vision_hidden_size` projection in ForgeConfig is for the LLM multimodal path — completely separate. Both are gaps for research.
- [RESOLVED] Vision-6: Priority is (A) wire LLM vision first. OCR already covers screen text extraction, while project goals need image understanding connected to language reasoning. The codebase already has the foundation (`core/vision_encoder.py` + multimodal hooks), so the shortest path to meaningful capability is LLaVA-style wiring/training (Vision-1b + Code-6). BLIP-2-class captioning can be a later enhancement, not the first milestone.

**Code mod:**
- [RESOLVED] Code-9: [mods/codegen/codegen.py](mods/codegen/codegen.py) calls the base 742M Enigma model for code generation via `LocalCode`. No specialized code model. The mod works but quality depends entirely on base model capability. A code-specialized fine-tune (Approach 2) would improve this significantly.

**Memory system:**
- [RESOLVED] Memory-1: [enigma_engine/core/memory.py](enigma_engine/core/memory.py) is flat text + regex pattern extraction (200 fact max). Retrieves relevant facts by keyword search at query time (not embedding-based). Simple but functional for user preferences.
- [RESOLVED] Memory-2: Yes, episodic memory can be built directly on the existing RAG path without a new memory subsystem. Current state: `core/memory.py` stores durable user facts; journal stores reflections; chat history already exists per session/model. Practical design: write each completed conversation/session summary as a retrievable document (timestamp + topic tags + key decisions) into a dedicated memory corpus, then query it through the same `RAGIndex.query()` path used for docs. This gives "last week recall" behavior with existing infrastructure. Working-memory can stay lightweight (recent topic list + current task summary) and be injected in prompt, while episodic memory stays retrieval-based.

**Model merging (Approach 2 enabler):**
- [RESOLVED] Merge-1: [enigma_engine/core/model_merging.py](enigma_engine/core/model_merging.py) has SLERP, TIES, and linear merge. TIES handles sign conflicts when merging multiple fine-tuned models. This is the right tool for combining specialists after fine-tuning. Already built and tested.
- [RESOLVED] Merge-2: Keep `density=0.2` as the default baseline for TIES at 742M as well. Code already uses `density: float = 0.2` in `ties_merge()`, which matches the common "top changed parameters" trimming regime from TIES usage. There is no universal scale-law value proven for every checkpoint/task pair, so the practical answer is calibration, not a one-size constant: sweep {0.1, 0.2, 0.3} on a held-out benchmark and pick best aggregate score. Until that ablation is run, 0.2 remains the correct default.

**Inner monologue:**
- [RESOLVED] Mono-1: [enigma_engine/core/monologue.py](enigma_engine/core/monologue.py) is Phase 5 inner monologue. Modes: disabled/journal_only/automatic. Coherence gate at 0.7 threshold (heuristic scorer). Journal stored as `data/model_contexts/<model_key>/journal.json`. Not on any research list.
- [RESOLVED] Mono-2: Keep the heuristic coherence gate as the primary filter for now. It is cheap, deterministic, and transparent, while the RL reward models are trained for preference/reward tasks and are not calibrated as a monologue-truth discriminator. Using RewardModel as the only gate risks swapping one heuristic for another opaque scorer. Practical path: retain heuristic gate (`score_coherence`), and add optional secondary rerank later (small classifier or reward model) only after collecting false-positive/false-negative data from real journal entries.

**Adaptive training pipeline:**
- [RESOLVED] Adaptive-1: [enigma_engine/core/adaptive_trainer.py](enigma_engine/core/adaptive_trainer.py) has `TrainingPlan` (stages: basics→conversation→commands→web), adaptive difficulty, TRAINER-evaluates-STUDENT loop. This is the AI-supervised curriculum. Not yet used for any of the planned approaches.
- [RESOLVED] Adaptive-2: Evaluation reliability depends more on evaluator quality margin than on exact parameter ratio. Code confirms current loop has teacher generate tests and judge student answers (`gui_forge_adaptive.py` Phase 3), so self-evaluation with the same-scale/student-like model is biased and weak. Decision: for meaningful adaptive gating, TRAINER should be a stronger external evaluator (e.g., Qwen3-30B class) while STUDENT remains 742M. If same-scale evaluator must be used, restrict it to coarse pass/fail and require external benchmark checks before advancing major stages.

**Avatar system:**
- [RESOLVED] Avatar-2: Avatar mod is production-ready. `AvatarBrick` main controller. `BoneController` has anatomical limits for ALL bones (head, neck, spine, arms with elbow/wrist, legs with knee/ankle, all 10 fingers, jaw, left+right eyes). `ModelManager` loads GLB/GLTF/OBJ via pygltflib or manual parsing fallback. Communicates over TCP to the ModRouter. NOT a stub. (head, neck, spine, arms with elbow/wrist, legs with knee/ankle, all 10 fingers, jaw, left+right eyes). `ModelManager` loads GLB/GLTF/OBJ via pygltflib or manual parsing fallback. Communicates over TCP to the ModRouter. NOT a stub.
- [RESOLVED] Avatar-3: Standard mapping target should be **FACS/ARKit-style coefficients**, then converted to current bone/expression commands. External references confirm the pattern: FACS action units are the common semantic layer for facial animation, and ARKit blendshape coefficients are normalized 0.0-1.0 controls per feature. Practical design for this codebase: emotion state (`valence/energy/...`) → small coefficient vector (brow, jaw_open, mouth_smile, eye_squint, head_pitch/yaw) → router command packets (`avatar.expression` + `avatar.bone`). This closes the loop without retraining the avatar model. Dataset requirement is optional at this stage; rule-based mapping is enough for v1.
- [RESOLVED] Avatar-4: Current consumers are **not** expecting BVH today. Verified outputs by code: `mods/threed/threed.py` emits OBJ (builtin), PLY/PKL (local Shap-E path), or GLB/OBJ (Replicate path) into `outputs/3d/`. Avatar runtime output is live router events (`avatar.bone_moved`, etc.) carrying JSON rotation triples, not baked animation files. **Decision:** keep live JSON stream as runtime protocol; if export is needed later, standardize on GLB+animation clips for interchange and keep BVH as optional mocap export only.

---

### Codebase-Specific Gaps (Must Verify Before Changes)

These are areas where the code may have bugs or missing wiring, found during audit:

- [RESOLVED] Code-1: D-8 FIXED. Both optimizer paths updated. Weight-tied output head is the same tensor as tok_embeddings — optimizer sees it once, counted in no_decay correctly.
- [RESOLVED] Code-2: Confirmed. `build_packing_masks()` → `attention_mask_2d` → model.forward(). Training.py lines 3183-3240.
- [RESOLVED] Code-3: WSD is the default. `schedule_type: str = "wsd"` in TrainingConfig.
- [RESOLVED] Code-4: β2=0.95 hardcoded in TrainingConfig.
- [RESOLVED] Code-5: RMSNorm epsilon is 1e-6. Correct.
- [BUILD] Code-6: No FORGE training path for vision projection training exists. `model.forward()` accepts `vision_features` and the projection is wired (Vision-5 resolved), but there is no training mode that: (1) loads a CLIP encoder, (2) passes image tensors through it, (3) trains only the projection Linear while freezing the LLM body. This requires a new FORGE training mode ("Vision Projection" or an extension of LoRA mode). Search terms: "LLaVA stage 1 training" (projection only), "LLaVA stage 2 training" (full SFT). File to create: a method in [enigma_engine/gui/gui_forge_training.py](enigma_engine/gui/gui_forge_training.py) mirroring `_start_vision_training()` but with frozen transformer.
- [RESOLVED] Code-7: [enigma_engine/core/lora_utils.py](enigma_engine/core/lora_utils.py) is a full LoRA/QLoRA trainer with PEFT + bitsandbytes + accelerate support. Classes: `LoraConfig`, `QLoraConfig`, `OffloadConfig`, `LoraTrainer`. Complete training workflow with `save_adapter()`. Used by RLHF/SelfPlay trainers for reference policy via `create_lora_model()`.
- [RESOLVED] Code-8: EWC ([enigma_engine/core/ewc.py](enigma_engine/core/ewc.py)) is standalone — NOT wired into FORGE GUI. Uses empirical Fisher information (squared gradients), diagonal approximation, `penalty()` returns scalar tensor for adding to loss. For Approach 2 specialist training, EWC must be manually wired into the relevant training path.

---

### Evaluation — Real Gaps

The codebase has `python run.py --benchmark` but it's unclear what it actually tests:

- [CONFIRMED GAP] Eval-1: `--benchmark` calls `run_coherence_benchmark()` from [enigma_engine/core/monologue.py](enigma_engine/core/monologue.py). It is a **self-reflection coherence test** — 20 reflective prompts, scored by `score_coherence()` (lexical diversity heuristic), reports ready/marginal/not_ready. **Does NOT test GSM8K, MMLU, HellaSwag, HumanEval, or any academic benchmark.** N-7 plans to "benchmark after pre-training" but the current `--benchmark` command is useless for measuring reasoning/knowledge quality. [BUILD] before N-7: implement at minimum GSM8K (parse final number, test 1319 examples, ~20 min on RTX 5090) in [enigma_engine/core/training_evaluation.py](enigma_engine/core/training_evaluation.py). Reference: lm-evaluation-harness on GitHub for prompt format.
- [RESOLVED] Eval-2: A full exhaustive suite is unlikely to fit <30 minutes at 742M (especially HellaSwag 10K + full MMLU). Practical <30 min regression run: GSM8K full or capped subset (depending on current decode speed) + HumanEval full, then sampled HellaSwag/MMLU slices for smoke checks. Full-set HellaSwag/MMLU should be treated as longer offline jobs for reporting.
- [RESOLVED] Eval-3: Minimum scoring stack: GSM8K final-number extraction with strict match, HumanEval pass@k via execution tests, and exact-match/multiple-choice accuracy for tasks with deterministic references. Use judge-model scoring only as a fallback for truly free-form items, and report it separately from deterministic metrics.
- [CONFIRMED GAP] Eval-4: Since `--benchmark` is coherence-only (Eval-1), there is **no few-shot prompting infrastructure at all** — no chain-of-thought prompting, no demonstration examples, no MMLU 5-shot. Before running real benchmarks: (1) implement few-shot prompt builder (prepend K examples before the test item), (2) verify the model can follow few-shot format. Note: zero-shot vs 5-shot MMLU typically differs by 10-20pp — always specify which when reporting.

---

### Critical Items (Highest Priority)

These must be resolved before N-6 restart or will cause silent training problems:

- [RESOLVED] **CRIT-1**: D-8 FIXED. Both `_setup_optimizer()` and `_build_llrd_param_groups()` include `or 'embed' in name` in no_decay. Embeddings now correctly excluded from weight decay.
- [RESOLVED] **CRIT-2**: β2=0.95 is hardcoded in TrainingConfig (`adam_beta2: float = 0.95`). Not PyTorch default. Correct.
- [RESOLVED] **CRIT-3**: `build_packing_masks()` is passed to the model forward call in [enigma_engine/core/training.py](enigma_engine/core/training.py). Intra-doc masking is active.
- [RESOLVED] **CRIT-4**: Pre-train path in [enigma_engine/gui/gui_forge_new_modes.py](enigma_engine/gui/gui_forge_new_modes.py) auto-sets `warmup_steps = max(10, total_steps // 100)` (~1% of total). For 2.5M steps → ~25,000 warmup steps. Safe for N-6. Other training paths (SFT/DPO) still use default 100 — acceptable for short fine-tuning runs.
- [RESOLVED] **CRIT-5**: WSD is already the default scheduler (`schedule_type: str = "wsd"` in TrainingConfig). Not a build task.
- [BUILD] **CRIT-6**: ~~D-1/D-2 — DCLM + FineMath + The Stack v2. Must collect before N-6 restart to include in data mix. Running N-6 on current 87.6GB without code/math data misses Stage 2 benefits.~~ **DONE (Pass 136).** `fetch_dclm()`, `fetch_finemath()`, `fetch_the_stack()` added to [collect_pretraining_data.py](collect_pretraining_data.py). Run: `python collect_pretraining_data.py --dclm 15 --finemath 10 --code 10 --resume`

---

### Deferred (Post N-10, No Urgency Now)

- Context extension (D-13/D-17, YaRN full implementation)
- Flash Attention 4 stable Windows wheel (D-20)
- Video/3D generation mods (Video-1, 3D-1)
- Haptic prediction (Haptic-1)
- FSDP multi-GPU (single GPU until hardware changes)
- DDP training infrastructure (single GPU)

---

## Recommendation + Next Steps

**Decision:** Pursue **Approach 1 + Approach 3 in parallel**.

**Rationale:**
- Approach 1 is the long-term strategy (reasoning foundation + plugins)
- Approach 3 is a 3-week proof-of-concept (validate multi-model architecture)
- Together: low risk, high learning, clear path forward

**Immediate actions (this week):**

1. **Research (parallel, 2-3 hours each):**
   - C-1: QwQ paper (reasoning architecture)
   - C-7: Read enigma_engine/router.py (what's already there?)
   - A1-2: OpenThoughts3 training approach
   - A3-1: HuggingFace distillation guide

2. **Planning (1 week):**
   - List what FORGE Distillation mode needs (teacher loading, loss function, data)
   - Design router decision tree (heuristic rules for task classification)
   - Plan N-6 pre-training data mix (emphasize reasoning data, per D-4)

3. **Execution decision (week 2):**
   - Start N-6 pre-training with reasoning data emphasis (no code change)
   - In parallel: collect data for Approach 3 distillation (OpenThoughts3, VQA, Code)
   - Generate targets from Qwen3-30B (offline, can parallelize)
   - Train first variant (reasoning) via FORGE

**Critical path to decision point (month 2):**
- N-6 checkpoint (if loss stable, continue)
- Approach 3 variant 1 result (if reasoning variant > 50% on GSM8K, validates architecture)
- → Then decide: go full Approach 1 or commit to Approach 2?

---

## Roadmap

### Phase 2 — First Real Pre-Training (current) + Approach 1 Preparation

| # | Task | Status |
|---|------|--------|
| N-6 | **Pre-train with full dataset + reasoning data emphasis** | **Ready to resume.** Align with Approach 1: emphasize OpenThoughts3/FineMath in data mix. D-4 (reasoning mid-training) should be integrated. Was at Step 60/2,537,114 (~0.002%), 78 tok/s with batch=2. Target: 300-600 tok/s with auto-batch. |
| N-7 | **Benchmark after pre-training** | `python run.py --benchmark` (GSM8K, HellaSwag, MMLU benchmarks per D-10) |
| N-RES-1 | **Research critical items** | C-1/C-2/C-7 (QwQ architecture, router.py review) — 3-4 hours, complete by start of N-9 |

### Phase 3 — Fine-Tuning & Alignment (Reasoning-First Focus)

| # | Task | Details | Approach 1 Alignment |
|---|------|---------|----------------------|
| N-8 | ~~Collect fine-tuning data~~ | **Done.** collect_finetuning_data.py — OASST, Dolly, SlimOrca | Gather OpenThoughts3 + FineMath for N-9.5 (Reasoning Mid-training per D-4) |
| N-9 | Instruction fine-tune | FORGE → Basic mode on mixed SFT data | Use reasoning-heavy SFT data (SmolTalk2 with thinking traces) |
| N-10 | DPO + GRPO alignment | Curated preference pairs, GRPO for reasoning | **Focus GRPO on test-time compute**: reward model should value "thinks before answering" |
| N-11 | ~~Wire GRPO/ReMax/SimPO/ORPO to GUI~~ | **Done.** Radio cards, dispatcher | Keep all; GRPO is primary for Approach 1 |

### Phase 3.5 — Approach 3 Proof-of-Concept (Parallel with N-9/N-10)

**Goal:** Validate multi-model architecture by training 1-3 specialized variants via distillation from Qwen3-30B. **Quick experiment (3-4 weeks) to inform Approach 1 vs Approach 2 decision.**

| # | Task | Details | Success Criteria |
|---|------|---------|------------------|
| N-9.5a | **Collect Approach 3 data** | OpenThoughts3 (reasoning), LLaVA/VQA (vision), The Stack (code) | 10K examples per variant |
| N-9.5b | **Generate from Qwen3-30B** | Use qwen3-30b-a3b to create task-specific target outputs (offline) | Targets generated, stored in data/ |
| N-9.5c | **Train Reasoning variant** | FORGE Distillation: 742M learns from Qwen3-30B targets on OpenThoughts3 | Variant checkpoints saved |
| N-9.5d | **Benchmark Reasoning variant** | Test on GSM8K (math), AIME-style problems | > 50% accuracy = validates reasoning distillation |
| N-9.5e | **Train Vision variant** (if time) | FORGE Distillation on VQA data | > 60% accuracy on VQA test set |
| N-9.5f | **Decision point** | If variants work: continue Approach 1. If not: iterate on data/teacher. | Clear go/no-go signal by month 2 |

**VRAM allocation:**
- 12GB: 742M distillation training
- 2GB: Qwen3-30B teacher (separate, offline generation)
- 2GB: Overhead

**Timeline:** 3-4 weeks total (1-2 weeks per variant, can parallelize data collection)

**Integration with Phase 3:**
- Week 1-2: N-9.5a/b in parallel with N-9 (collect data)
- Week 3: N-9.5c/d (train reasoning variant while N-10 planning happens)
- Week 4: Decision + pivot

### Phase 4 — Evaluate & Improve (Approach 1 Foundation)

| # | Task | Details | Approach 1 Alignment |
|---|------|---------|----------------------|
| N-12 | **Build router + task classifier** | Design decision tree or heuristic router (C-8 research) | Classify: vision? code? reasoning? generation? → dispatch to specialist or 742M core |
| N-13 | **Evaluation benchmarks** | HellaSwag/MMLU/GSM8K for reasoning core | Test reasoning variant from N-9.5d in production-like scenario |
| N-14 | **Dense semantic memory** | Replace TF-IDF in [enigma_engine/core/rag.py](enigma_engine/core/rag.py) with FAISS/dense embeddings | Preparation for vision+retrieval later |

### Phase 5 — Advanced Features

| # | Task | Details |
|---|------|---------|
| N-15 | Constrained decoding | Grammar-constrained generation for JSON/tool calls |
| N-16 | Best-of-N sampling | Generate N, score with reward model, return best |
| N-17 | ~~Model merging~~ | **Done.** core/model_merging.py — SLERP, TIES, linear merge. GUI caller added in N-21 (Pass 126). |
| N-18 | ~~Continual learning~~ | **Done.** core/ewc.py — Fisher information + penalty |
| N-19 | Knowledge distillation | Logit-level distillation from teacher |
| N-20 | ~~Agentic tool loops~~ | **Done.** engine_generation.py — parse/execute/inject loop |
| N-21 | ~~Wire model merging to GUI~~ | **Done.** MODELS page inline merge row: two model dropdowns, SLERP/LINEAR/TIES method dropdown, t entry, density entry, output name, MERGE button. `_merge_models()` in [enigma_engine/gui/gui_forge_models.py](enigma_engine/gui/gui_forge_models.py) calls [enigma_engine/core/model_merging.py](enigma_engine/core/model_merging.py) in a background thread. |
| N-22 | ~~Emotional state → system prompt~~ | **Done.** `_build_gui_context()` injects internal emotional state tone cue. |
| N-23 | ~~Move RAG into _prepare_chat()~~ | **Done.** RAG retrieval moved into `core/engine_chat.py::_prepare_chat()`; GUI/API/background paths now share the same retrieval path when `engine._rag_index` is set. |
| N-24 | ~~Mod router frame framing~~ | **Done.** [enigma_engine/router.py](enigma_engine/router.py) already uses 4-byte big-endian message framing in both `_send_message()` and `_receive_message()`. |
| N-25 | ~~Background trainer GPU isolation~~ | **Done.** `BackgroundTrainer.run()` defers batches while inference is active via router-provided idle callback; GUI wires callback from `_is_generating`. |

### Phase 6 — Compiled Performance (Rust via PyO3)

Pure Python hot paths that would benefit from Rust rewrite. Priority by impact.
Speedup estimates derived from real benchmarks (HuggingFace tokenizers, OpenAI tiktoken, Google SentencePiece source code and published numbers).

**Evidence base:**
- HF tokenizers (Rust): "Less than 20 seconds to tokenize 1 GB on server CPU" (~50+ MB/s). Trains wikitext-103 (516 MB) in "a few seconds."
- tiktoken (Rust): "3-6x faster than comparable open source tokeniser." Source code says "Most of the time is spent in regex."
- SentencePiece (C++): ~50K sentences/sec, 6 MB memory.
- Our Python BPE: measured ~7 MB/s encode (4718 vocab, 0.45 MB test). BPE train: 20 min for 32K vocab on 2 GB (2.4M unique words).

| # | Component | File(s) | Speedup | Affects Inference | Notes |
|---|-----------|---------|---------|-------------------|-------|
| R-1 | ~~**Tokenizer encode/decode**~~ | [enigma_engine/core/bpe_tokenizer.py](enigma_engine/core/bpe_tokenizer.py) + [rust_extensions/](rust_extensions/) | ~~**7-20x**~~ **6x** | **Yes** | **Done.** Measured: 32-36 MB/s Rust (cached), ~8 MB/s (unique words) vs ~5-7 MB/s Python. Symbol interning + skip-array merge. PyO3 + maturin, GNU toolchain. Auto-fallback in BPETokenizer.encode()/decode(). |
| R-2 | ~~**BPE train()**~~ | [enigma_engine/core/bpe_tokenizer.py](enigma_engine/core/bpe_tokenizer.py) | ~~**20-60x**~~ | No | **Done.** Full Rust train via PyO3: pre-tokenize → word freq → pair freq/reverse index → max-heap with lazy deletion → incremental merge loop. Python auto-fallback in `_try_rust_train()`. 8 tests. Tie-breaking differs from Python (implementation-defined), final vocab within ±3 tokens. |
| R-3 | **MinHash dedup** | [enigma_engine/core/training.py](enigma_engine/core/training.py) `minhash_dedup()` | **Needs profiling** | No | O(n²) pairwise + SHA-256 per shingle. Likely benefits from Rust + SIMD, but no baseline measurement yet. |
| R-4 | **Data pipeline** | [collect_pretraining_data.py](collect_pretraining_data.py), [pretokenize_data.py](pretokenize_data.py) | **Needs profiling** | No | File I/O is OS-bound. Parallelism helps but Python `open()` is already a thin wrapper. Measure before committing. |
| R-5 | **Sequence packing** | [enigma_engine/core/training.py](enigma_engine/core/training.py) `pack_sequences_lazy()` | **Needs profiling** | No | Heap-based bin packing. Moderate gain expected — lowest priority. |

**Language choice: Rust + PyO3.** Both OpenAI (tiktoken) and HuggingFace (tokenizers) independently chose this stack. C++ (SentencePiece) works but has clunkier Python bindings (SWIG). Cython gives 2-5x for this workload — not worth it vs Rust's 7-60x. No production tokenizer uses Cython, Mojo, or Zig.

**Approach:** PyO3 + maturin. Builds a native `.pyd` (Windows) / `.so` (Linux) extension. Python wrapper matches existing `TokenizerProtocol` interface — no changes to callers. Vocab saved/loaded as same JSON format.

**R-1 and R-2 are one Rust crate** — a single BPE tokenizer that trains fast AND encodes fast. R-3 through R-5 should be profiled before committing to rewrite.

### Phase 7 — Long-Term Sustainability

| # | Task | Details |
|---|------|--------|
| S801 | ~~GGUF VRAM→context tiers~~ | **Done.** InferenceMemoryBudget.gguf_context_length / gguf_gpu_layers. 48 GB now gets 64K. |
| S802 | ~~Inference batch/seq tiers~~ | **Done.** InferenceMemoryBudget.inference_batch_size / inference_max_seq_len. 32 GB gets batch=8. |
| S803 | ~~Token count cache cap~~ | **Done.** Scaled to RAM via token_count_cache_cap. 8 GB → 4096, 64 GB → 32768. |
| S804 | ~~BPE cache cap~~ | **Done.** Scaled to RAM via bpe_cache_cap. 8 GB → 10000, 64 GB → 80000. |
| S805 | ~~Advanced tok cache cap~~ | **Done.** Scaled to RAM via advanced_tok_cache_cap. Same profile as S804. |
| S806 | ~~Dataset chunk size~~ | **Done.** dataset_chunk_chars / dataset_stream_threshold scale to RAM. Pi 5 gets 200M, 64 GB gets 500M. |
| S807 | ~~API max_tokens cap~~ | **Done.** api_max_tokens scales to VRAM. 32 GB → 16384, 8 GB → 4096. |

### Ideas

| Idea | When |
|------|------|
| ~~Training ETA in FORGE panel~~ | **Done.** Batch-level ETA in [enigma_engine/gui/gui_forge_training.py](enigma_engine/gui/gui_forge_training.py) |
| Checkpoint browser with perplexity comparison | After N-7 |
| ~~SimPO/ORPO GUI wiring~~ | **Done.** Included in N-11 |
| Pre-tokenized binary cache (`tokens.bin`) | Script written ([pretokenize_data.py](pretokenize_data.py)). Skips 20+ min data reload on future runs. Integration into [enigma_engine/core/training.py](enigma_engine/core/training.py) TBD after first run. |
| ~~Larger vocabulary (16K-32K trained BPE)~~ | **Done.** 32K BPE vocab trained on 2 GB sample (2.4M unique words). Active in current pre-training run. |

---

## External Research — April/May 2026

**Pass 1 (April):** SmolLM3, Gemma 3, DCLM, FineMath, OpenThoughts3, APO — D-1 through D-13. Single-source, directionally correct.
**Pass 2 (May):** Full paper reads — Qwen3 (arXiv:2505.09388v1, 742M-scale architecture details), DeepSeek-V3 (arXiv:2412.19437v2, MTP + MLA ablations), Flash Attention 4 releases (SM120/Blackwell confirmed). D-14 through D-20 added. **Correction:** Qwen3 does NOT use MTP (D-15 clarifies DeepSeek-V3 only). D-3's three-stage training now cross-validated by 3 sources.

All items below are research suggestions only. No code has been changed.

---

### D-1 — Add DCLM-Baseline and FineMath to pre-training data

**Why:** We have 88.8 GB of mostly web text (FineWeb-Edu 40 GB, C4 20 GB, Wiki 15 GB, OWT 10 GB). We have **zero dedicated math data** and **zero model-filtered web data**. This is the biggest gap.

- **DCLM-Baseline** (`mlfoundations/dclm-baseline-1.0`): 4T tokens from Common Crawl, model-filtered with a fastText classifier trained on OpenHermes 2.5 (instruction-tuned quality signal). Beats FineWeb-Edu at all compute levels in MMLU. CC-BY-4.0. Used by SmolLM3 Stage 1. The `collect_pretraining_data.py --fineweb 25` pattern already knows how to stream HF datasets — DCLM loads the same way. Collect 10–20 GB.

- **FineMath-4+** (`HuggingFaceTB/finemath`, `finemath-4plus` config): 9.6B tokens / 6.7M documents of high-quality step-by-step math from CommonCrawl. Scored by LLaMA-3.1-70B-Instruct on a 5-point scale, filtered to 4+. LaTeX and Markdown formatted. GSM8K/MATH benchmark gains are measurable from as little as a few billion tokens. ODC-By license. No math in current dataset means zero math reasoning at inference time. Collect 10–15 GB.

- **InfiMM-WebMath-3+** (`HuggingFaceTB/finemath`, `infiwebmath-3plus` config): 20.5B tokens, complementary to FineMath (different source URLs). SmolLM3 uses 50/50 blend of both. Combining gives ~50B tokens while matching FineMath quality. Collect 10 GB.

**Action:** Add DCLM and FineMath download to [collect_pretraining_data.py](collect_pretraining_data.py). Integrate into `combined.txt` with cap at 10–15 GB each. Rerun paragraph dedup after adding.

**Priority: HIGH** — no math data is the single biggest gap before instruction fine-tuning.

---

### D-2 — Add code data from The Stack v2

**Why:** SmolLM3 used 12–24% code data across its three training stages and found it substantially improved reasoning. We have zero code in pre-training. Code data teaches structured reasoning, pattern completion, and precise instruction following.

- **The Stack v2** (`bigcode/the-stack-v2`): Deduplicated source code across 600+ programming languages, 67B unique files. Use only the 16 languages SmolLM3 used: Python, JavaScript, TypeScript, Java, C, C++, Go, Rust, Ruby, PHP, Shell, SQL, HTML, CSS, Markdown, JSON. Filter to permissive licenses (MIT/Apache/BSD). Collect 5–10 GB from popular languages first.

- **StarCoder2 pull requests + GitHub issues**: Available as part of The Stack v2 ecosystem. More conversational/reasoning content than raw code. Collect 1–2 GB.

**Action:** Add The Stack v2 streaming download to [collect_pretraining_data.py](collect_pretraining_data.py). Use `--code 10` flag pattern, filter by language and license. Prioritize Python + JS + Rust.

**Priority: HIGH** — zero code = severely limited structured reasoning.

---

### D-3 — Three-stage pretraining data schedule

**Why:** SmolLM3 (11T tokens, 3B params), Qwen3, and Llama3 all use multi-stage training where data composition shifts across stages. Running a single fixed mixture for 2.5M steps is validated to be suboptimal — later stages should upsample math and code while the model is most capable of using them.

**Recommended schedule for Enigma (26B tokens → ~1-2 epochs):**
- Stage 1 (0→~20B tokens): 85% web (FineWeb-Edu + DCLM), 12% code, 3% math — general capability
- Stage 2 (20B→~24B tokens): 75% web, 15% code, 10% math — deeper math/code
- Stage 3 / Decay (24B→26B tokens): 63% web, 24% code, 13% math, inject OpenThoughts3 reasoning samples — LR linear decay to 0

**Implementation:** In [enigma_engine/core/training.py](enigma_engine/core/training.py), multi-stage scheduling can be done by maintaining separate dataset iterators and swapping them at step boundaries (not via GUI — wired in the Trainer). The WSD scheduler is more compatible with staged training than cosine — see D-5.

**Action:** Plan staged data mixing before starting next pre-training run. Can be done manually by re-weighting `combined.txt` sections or by having Trainer switch datasets at step N.

**Priority: MEDIUM** — depends on D-1/D-2 data being collected first.

---

### D-4 — Reasoning mid-training before SFT (chain-of-thought injection)

**Why:** SmolLM3 trained 35B tokens of reasoning data BETWEEN pre-training and SFT. This gave the base model the ability to reason before instruction fine-tuning shaped the behavior. Result: AIME 2025 improved from 9.3% → 36.7% with extended thinking. Without reasoning mid-training, SFT reasoning capability is severely limited.

- **OpenThoughts3-1.2M** (`open-thoughts/OpenThoughts3-1.2M`): 1.2M rows of math (850K), code (250K), and science (100K) with long reasoning traces. Annotated by QwQ-32B (16 answers per question). Apache-2.0. 28.2 GB. Enigma already has QwQ-like models in the `models/qwen3-30b-a3b/` directory — we could use our own Qwen3 30B to generate additional domain-specific reasoning traces.

- **NVIDIA Llama-Nemotron Post-Training v1.1** (`nvidia/Llama-Nemotron-Post-Training-Dataset`): 3.91M rows of reasoning traces from R1. Used by SmolLM3. Apache-2.0.

**Phase 3.5 (NEW):** Add a "Reasoning Mid-training" phase between N-9 (SFT) and N-10 (DPO):
- 1–3 epochs on OpenThoughts3-1.2M using ChatML format (no system prompt)
- Use FORGE → Basic mode, low LR (1e-5), no DPO
- This step costs ~10B tokens of GPU compute for 1 epoch

**Action:** Add N-9.5 Reasoning Mid-training to roadmap. Add OpenThoughts3 download to [collect_finetuning_data.py](collect_finetuning_data.py). Source data using Qwen3-30B from [models/qwen3-30b-a3b/](models/qwen3-30b-a3b/) for domain-specific math/science traces.

**Priority: HIGH** — this is the gap between a model that follows instructions and one that actually reasons.

---

### D-5 — WSD scheduler instead of cosine for long pre-training runs

**Why:** SmolLM3 and OLMo 2 both use WSD (Warmup-Stable-Decay): warm up for 2000 steps, hold stable LR for most of training, then linear decay to 0 in the final 10% of steps. The advantage over cosine:
1. You can stop training at any stable-phase checkpoint and resume without losing the decay budget
2. The stable phase gives more consistent gradient signal than cosine's oscillating LR
3. If a run is interrupted mid-decay, you can restart the decay from a stable-phase checkpoint

Current Enigma uses cosine LR (`CosineAnnealingLR` in training.py). For a 2.5M step run (~3 months), a mid-run interruption with cosine means the LR schedule is off when resumed. WSD means the stable-phase checkpoint is always valid.

**Implementation:** `torch.optim.lr_scheduler.LambdaLR` with three phases: linear warmup (0→1 over `warmup_steps`), constant 1.0 (until `decay_start_step`), linear decay (1.0→0 over final `decay_steps`). `decay_start = int(0.9 * total_steps)`.

**Action:** Add WSD scheduler as an option in `TrainingConfig` (keep cosine as default for safety). Wire to GUI as a scheduler dropdown option. Recommended for any run > 500K steps.

**Priority: MEDIUM** — the current cosine will work, but WSD is strictly better for long interrupted runs.

---

### D-6 — Expand tokenizer vocabulary to 64K-128K

**Why:** Current tokenizer is 32K BPE trained on 2 GB English web text. This means:
- All code is very poorly encoded (curly braces, keywords all split into tiny fragments)
- Math symbols (LaTeX, ≤, ∑, π) are typically unknown or single-byte
- Non-English text is fragmented (2-4x more tokens per character than English)

Real-world evidence: LLaMA3 uses 128K (100K base + 28K non-English), Gemma3 uses 262K SentencePiece. SmolLM3 reuses LLaMA3.2 tokenizer. Increasing from 32K → 64K on a corpus that includes code and math could reduce average tokens/document by 15-30% for those domains.

**What to collect for tokenizer training:**
- Python/JS/Rust source code (The Stack v2 — 100MB sample per language)
- FineMath documents (LaTeX-heavy math content)
- Multi-language web text (Spanish, French, German, Chinese Wikipedia dumps)
- Current 2 GB English sample (keep as base)

**Risk:** Vocab expansion requires re-initializing the embedding layer for new tokens, which means any pre-trained checkpoint will need fine-tuning on the new tokens before useful behavior resumes. **Do this BEFORE starting the next pre-training run, not after.** Once pre-training starts with 32K vocab, the cost of switching grows.

**Action:** Retrain tokenizer before N-6 with: 2 GB English web + 500 MB code + 200 MB math + 500 MB multilingual. Target 64K vocab. This would reduce total token count by ~10% for current dataset, reducing pre-training compute by ~10%.

**Priority: HIGH (time-sensitive)** — must happen before N-6 restarts, not after.

---

### D-7 — Intra-document masking during sequence packing

**Why:** When packing multiple documents into a single training sequence (which `pack_sequences_lazy()` does), the attention mask should prevent tokens from Document B attending to tokens from Document A. Without this mask, the model learns spurious cross-document dependencies. Llama3 uses this, SmolLM3 uses this. Found to improve long-context performance and training stability.

**Implementation check needed:** Verify whether `_create_packed_mask()` in [enigma_engine/core/training.py](enigma_engine/core/training.py) or [enigma_engine/core/model.py](enigma_engine/core/model.py) builds a proper block-diagonal mask per packed sequence. The 4D mask generated during sequence packing needs to be causal within each document and blocked across documents.

**This may already be implemented.** Check `pack_sequences_lazy()` and `_create_packed_mask()` return values. If the mask is just a standard causal mask (upper triangular), this feature is missing.

**Action:** Read [enigma_engine/core/training.py](enigma_engine/core/training.py)`::_create_packed_mask()` before the next pre-training run. If it returns a simple causal mask, add block-diagonal masking. Moderate implementation cost, high benefit.

**Priority: MEDIUM** — verify before N-6 restarts.

---

### D-8 — Remove weight decay from embedding layers

**Why:** OLMo 2 finding, adopted by SmolLM3: removing weight decay from embedding layers stabilizes training. Embedding norms naturally stabilize at healthier values without it. With weight decay on embeddings, the model may underfit on rare tokens because their embeddings get shrunk toward zero. SmolLM3 reported "more stable training dynamics" with this change.

**Implementation:** In [enigma_engine/core/training.py](enigma_engine/core/training.py), the `AdEMAMix` and `Muon` optimizer setups apply weight decay globally. Need to set `weight_decay=0` specifically for `model.embedding.weight` and `model.output.weight` (if not tied). If embeddings are tied (`model.embedding.weight is model.output.weight`), one parameter group suffices.

**Code pattern:**
```python
no_decay = ["embedding.weight", "output.weight"]
param_groups = [
    {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]
```

**Action:** Verify whether current optimizer setup applies weight decay to embeddings. If yes, add this split. Low-risk change, 3-5 line fix.

**Priority: LOW-MEDIUM** — implement before N-6 restart.

---

### D-9 — Anchored Preference Optimization (APO) as DPO alternative

**Why:** APO (D'Oosterlinck et al., Aug 2024) is a controllable variant of DPO. In DPO, the optimization objective can be underspecified — the model can satisfy the loss by degrading the rejected response rather than improving the chosen one. APO adds an explicit anchor term that controls how far the model can deviate from the reference model. Result: more stable training, better downstream scores. SmolLM3 chose APO over DPO and GRPO for their alignment phase. CLAIR data + APO improved Llama-3-8B by 7.65%, closing the gap with GPT4-turbo by 45%.

**Current state:** Enigma has DPO, SimPO, GRPO, ORPO, ReMax all wired. APO is not implemented.

**APO loss formula:** Standard DPO log-ratio term plus an anchor penalty that caps the KL divergence from the reference. Implemented as a modified DPO loss function with one additional hyperparameter `β_anchor` (recommended 0.1–0.5).

**Action:** Add APO to `alignment_training.py` as a new alignment mode. Add radio card in GUI alignment section. Relatively low implementation cost — it's a 15-line modification of the DPO loss.

**Priority: LOW-MEDIUM** — implement after N-10 (DPO) is validated.

---

### D-10 — Evaluation framework: wire standard benchmarks

**Why:** After pre-training (N-7) and fine-tuning (N-9/N-10), there's no automated way to measure if the model improved. `python run.py --benchmark` exists but only runs the coherence benchmark (internal perplexity). We need external benchmarks to compare against published models.

**Recommended benchmarks (all open, downloadable):**
- **GSM8K** (math word problems, 1.3K test examples, graded by exact match). Simple to evaluate — parse final number from output.
- **HellaSwag** (commonsense NLI, 10K examples, multiple choice). Already mentioned in roadmap N-12.
- **MMLU-CF** (controlled-format MMLU, removes format sensitivity). 57 subjects, 14K questions.
- **HumanEval+** (code generation, 164 problems, execute and test). Requires Python execution sandbox.

**Implementation approach:** Add `--eval-gsm8k`, `--eval-hellaswag`, `--eval-mmlu` flags to [run.py](run.py). Download benchmark data once, save to `data/eval/`. Run with the model in greedy decode mode (temperature=0). Output: accuracy percentage + 95% CI. No extra dependencies needed beyond what's in requirements.txt.

**Action:** Implement N-12 evaluation framework. Start with GSM8K (simplest to implement, unambiguous grading). HellaSwag second. MMLU third.

**Priority: MEDIUM** — needed to validate that training actually helped.

---

### D-11 — SmolTalk2 as SFT + preference data

**Why:** HuggingFace released the complete SmolLM3 post-training dataset as `SmolTalk2` (`HuggingFaceTB/smoltalk2`). It contains three subsets: `Mid` (reasoning mid-training, 140B tokens), `SFT` (1.8B tokens, 12 non-reasoning + 10 reasoning datasets), and `Preference` (APO preference pairs, non-reasoning + reasoning modes). This is fully open and purpose-built for instruction tuning a small model well.

**Current fine-tuning data:** OASST, Dolly 15k, SlimOrca, UltraChat. Good but these are older datasets without reasoning traces.

**What SmolTalk2 adds:**
- Reasoning traces in SFT format (thinking mode + non-thinking mode examples)
- Tool calling examples in XML and Python formats
- Multilingual SFT examples (French, Spanish, German, Italian, Portuguese)
- APO preference pairs rated chosen=Qwen3-32B, rejected=Qwen3-0.6B

**Action:** Add SmolTalk2 SFT subset download to [collect_finetuning_data.py](collect_finetuning_data.py). Use the SFT subset only (not Mid — that's 140B tokens we don't need to download). ~10-15 GB. Integrate into FORGE instruction fine-tuning (N-9) data mix.

**Priority: MEDIUM** — upgrade to current state-of-art SFT data when N-9 is executed.

---

### D-12 — NoPE hybrid attention for long-context improvement

**Why:** SmolLM3 implemented NoPE (Yang et al., Jan 2025, arXiv 2501.18795) and found it improves long-context performance without hurting short-context. The approach: remove RoPE from every 4th transformer layer. NoPE layers have no positional bias, making them sensitive to content rather than position — they act as long-range context aggregators. RoPE layers provide local positional structure. The hybrid outperforms pure RoPE at extended context lengths.

**Current state:** Enigma model has RoPE at every layer ([enigma_engine/core/engine_generation.py](enigma_engine/core/engine_generation.py), [enigma_engine/core/model_components.py](enigma_engine/core/model_components.py)).

**Implementation cost:** Low — the change is `if layer_idx % 4 != 0: apply_rope(...)`. In [enigma_engine/core/model.py](enigma_engine/core/model.py) or [enigma_engine/core/model_components.py](enigma_engine/core/model_components.py), the attention forward pass already accepts positional arguments. Add a `use_rope` flag per layer determined by `layer_idx % 4 == 0`.

**Note:** This changes the model architecture. Any pre-trained checkpoint becomes incompatible unless the change is made before N-6. If made after N-6, requires fine-tuning to recover.

**Action:** Evaluate timing — implement before N-6 restart if possible. If N-6 has already started, defer to after pre-training when a long-context fine-tuning phase is added.

**Priority: LOW (time-sensitive)** — worth implementing before N-6 if not already started.

---

### D-13 — Context length extension via RoPE theta scaling (not training from scratch)

**Why:** All production models (Gemma3, SmolLM3, LLaMA3) extend context length at the END of pre-training, not from the beginning. Gemma3 trained at 32K sequences, then extended to 128K by increasing RoPE theta from 10K to 1M (×100). SmolLM3 extended 4K→32K→64K in two sequential 50B-token stages after the main 11T token run. Starting at 4K is correct — it's faster and cheaper.

**Current state:** Enigma `max_seq_len` is configurable but the tokenizer and training setup use 4096 by default. This is correct.

**Recommended extension plan (after N-9/N-10):**
1. After fine-tuning is stable, run a 5B-token extension stage at 16K context: increase RoPE theta to 500K
2. Run a 2B-token extension stage at 32K context: increase RoPE theta to 2M
3. Use YaRN at inference for 2× extrapolation beyond training context

**YaRN** (`rope_scaling_type="yarn"` in config): Scales RoPE frequencies non-uniformly, allowing inference at 2× the training context with minimal degradation. Drop-in addition to the inference path. No training needed.

**Action:** Add RoPE theta as a configurable `TrainingConfig` parameter (currently hardcoded?). Add YaRN scaling to inference path as an optional flag. Plan context extension as N-26 after alignment is done.

**Priority: LOW** — comes after the main training pipeline is proven.

---

### D-14 — QK-Norm for Training Stability (verified from Qwen3 + ViT-22B full papers)

**Why:** Qwen3 (arXiv:2505.09388, Section 2) adds QK-Norm to all attention layers specifically for training stability. ViT-22B (Dehghani et al., arXiv:2302.05442) independently applied the same technique at 22B scale. Two independent large-scale adoption points, different modalities, same conclusion.

**The problem it solves:** Without QK-Norm, Q and K vectors can grow in magnitude over training. Attention scores Q@K.T/sqrt(d) spike → softmax collapses to argmax (one-hot attention) → zero gradients for all non-winning tokens → gradient starvation. Qwen3 treats this as important enough to mention it in their architecture section as a named change from Qwen2.

**Implementation:** Add `self.q_norm = nn.RMSNorm(head_dim)` and `self.k_norm = nn.RMSNorm(head_dim)` to the attention module. Apply after projecting and splitting Q/K into heads, before RoPE. Two extra RMSNorm ops per attention layer — negligible compute (~0.1% overhead), ~0.1% extra parameters.

**Timing concern:** Architecture change. RMSNorm weights initialize to 1 (identity), so adding this to a partial checkpoint allows recovery in continued training, but adding before N-6 is cleaner. Does not change model output at step 0 (since norm of a vector by its norm = same direction).

**Action:** Read [enigma_engine/core/model_components.py](enigma_engine/core/model_components.py) attention forward pass. Add Q/K normalization per head before RoPE application. Verify existing attention tests still pass (shape + determinism).

**Priority: HIGH** — low-cost architecture improvement validated by 2 independent production models at scale. Recommend implementing before N-6 restart.

---

### D-15 — Multi-Token Prediction (MTP) Training Objective (DeepSeek-V3 only, not Qwen3)

**IMPORTANT NOTE:** First-pass research suggested both Qwen3 and DeepSeek-V3 use MTP. This is wrong. **Qwen3 does NOT use MTP.** Qwen3 uses a 4-stage post-training pipeline (CoT cold start → GRPO → mode fusion → General RL). Only DeepSeek-V3 uses MTP in pre-training. Source verified: Qwen3 full paper arXiv:2505.09388 has no MTP section. DeepSeek-V3 full paper arXiv:2412.19437 Section 2.2 fully describes MTP.

**Why it matters (DeepSeek-V3 ablation, 2.4B activated / 15.7B total params, 1.33T tokens):**

| Benchmark | Without MTP | With MTP | Gain |
|-----------|-------------|----------|------|
| BBH | 39.0 | 41.4 | +2.4pp |
| MMLU | 50.0 | 53.3 | +3.3pp |
| GSM8K | 25.4 | 31.4 | +6.0pp |
| HumanEval | 20.7 | 26.8 | +6.1pp |

Consistent gains across all 4 benchmarks at sub-2B activated parameter scale. This is the closest available ablation to Enigma's 742M scale.

**Architecture:** D=1 depth (predicts one extra token beyond next-token). Each MTP module contains one Transformer block + projection M_k ∈ R^(d×2d) + shared embedding + shared output head. Combines prior-depth repr h^(k-1)_i with Emb(t_{i+k}) via projection. Loss weight λ=0.3 for first 10T tokens, 0.1 for remaining 4.8T. At inference: discard head (zero compute cost) OR keep for speculative decoding (85-90% 2nd-token acceptance → 1.8× TPS).

**Current Enigma state:** [enigma_engine/core/model.py](enigma_engine/core/model.py) has MTP head scaffolding. Verify: (1) Is MTP loss actually applied in [enigma_engine/core/training.py](enigma_engine/core/training.py) training loop? (2) Is the λ schedule implemented (0.3 early → 0.1 late)? (3) Is the MTP block architecture correct (sequential, not parallel)?

**Action:** Read [enigma_engine/core/model.py](enigma_engine/core/model.py) MTP section and [enigma_engine/core/training.py](enigma_engine/core/training.py) loss calculation. If scaffold exists but loss isn't wired, wire it. If wired but λ is constant, add the step-based schedule.

**Priority: MEDIUM** — validated improvement with small-scale ablation evidence. Architecture change — do before N-6 restart.

---

### D-16 — Three-Stage Pre-Training Cross-Validated (D-3 now confirmed by 3 sources)

**Note:** D-3 was originally sourced from SmolLM3 only. Now verified from full paper reads of Qwen3 (arXiv:2505.09388, Section 3.2) and DeepSeek-V3 (arXiv:2412.19437, Section 4.3). All three systems independently use staged training with shifting data composition.

**Qwen3 exact stages (from paper Table and Section 3.2):**
- S1: 30T tokens at 4K seqlen — general knowledge (web, books, code, multilingual)
- S2: 5T tokens at 4K seqlen — knowledge-intensive: higher proportion of STEM, code, math, reasoning data
- S3: Hundreds of billions of tokens at 32K seqlen — long-context extension using YARN + Dual Chunk Attention (DCA)

**For Enigma's ~26B tokens:** Our entire pre-training run is roughly S1 equivalent. S2 requires FineMath (D-1) — plan a continued training phase on math-heavy data after first 20B tokens. S3 is D-13/D-17 (context extension). Three independent data points now confirm this is the right architecture for staged training. D-3's recommendations are validated.

**Priority: See D-3** — no separate action needed. This entry upgrades D-3's evidence from single-source to 3-way cross-reference.

---

### D-17 — RoPE ABF Exact Parameters (cross-validates and updates D-13)

**Note:** D-13 proposed context extension but lacked specific parameter values. Now verified from two full paper reads.

**Qwen3 (arXiv:2505.09388, Section 3.2):** Raises RoPE base frequency from 10,000 to **1,000,000** (100× increase) using ABF (Adjusted Base Frequency) technique during S3. This is a training-time change — not an inference extrapolation hack. The model is trained on 32K sequences with the 1M base, allowing genuine long-context attention patterns to form.

**DeepSeek-V3 (arXiv:2412.19437):** Uses YARN with scale s=40, α=1, β=32, t=0.1×ln(s)+1. Applied only to the decoupled RoPE key (their MLA architecture). Context extended 4K→32K then 32K→128K in two phases.

**Recommended parameters for Enigma extension phases:**
- Before context extension: raise rope_base 10K → 500K
- 16K training phase: rope_base = 500K
- 32K training phase: rope_base = 1,000,000
- At inference: YARN scale factor (s=8 for 2× extrapolation) as optional config flag

**Action:** Verify [enigma_engine/core/model.py](enigma_engine/core/model.py) reads `rope_base` from config. If hardcoded as 10000, make it a `TrainingConfig` field. See D-13 for full context extension plan.

**Priority: LOW** — implement alongside D-13 (context extension phase, after N-10 alignment).

---

### D-18 — Aux-Loss-Free MoE Load Balancing (DeepSeek-V3, only if MoE enabled)

**Why:** Standard MoE training adds an auxiliary loss term to prevent expert collapse. DeepSeek-V3 (arXiv:2412.19437, Section 2.1.2 + ablation Table 5) replaces this with a bias-based mechanism that achieves better balance without polluting the main training signal.

**Mechanism:** Each expert gets a bias b_i (initialized 0). Routing decision uses s_{i,t} + b_i (adds bias). Actual gating weight uses s_{i,t} only (bias not applied). After each step: b_i += γ if expert underloaded, b_i -= γ if overloaded. γ=0.001. Bias frozen for the last 500B tokens to prevent late-training oscillation.

**Verified ablation (small scale, 1.33T tokens):**
- GSM8K: aux-loss 27.1 → aux-loss-free 29.6 (+3.8pp)
- HumanEval: aux-loss 22.0 → aux-loss-free 22.6 (+2.6pp)

**Action:** If MoE is enabled in Enigma (currently optional), replace the existing MoE aux loss with the bias mechanism. Minimal implementation: per-expert scalar bias, ±γ update logic after each batch, freeze flag at step threshold.

**Priority: LOW** — only relevant when MoE is turned on. No action until MoE training begins.

---

### D-19 — Strong-to-Weak Distillation for Post-Training Small Models (Qwen3)

**Why:** Qwen3 (arXiv:2505.09388, Section 4.4) reports that running the full 4-stage post-training pipeline on sub-2B models is 10× more expensive than distillation, yet distillation achieves competitive quality. The approach: take the large model (32B) after full 4-stage post-training, generate responses on instruction data, use those responses as SFT targets for the small model.

**Why this matters for Enigma:** At 742M params, running GRPO RL (our N-11 pipeline) is expensive and the reward signal is noisy at small scale. A distillation-first approach may be more efficient:
1. Use Qwen3-30B (already in `models/qwen3-30b-a3b/`) to generate high-quality responses on our SFT data
2. Fine-tune Enigma 742M on those generated responses via FORGE Distillation mode
3. Use GRPO only if distillation plateau is reached (not as the primary alignment path)

**Current state:** FORGE has a Distillation training mode. It can load a teacher model and train on teacher outputs. Verify whether the Distillation mode in [enigma_engine/gui/gui_forge_new_modes.py](enigma_engine/gui/gui_forge_new_modes.py) supports loading a local GGUF/pth teacher (like Qwen3-30B).

**Action:** Review Distillation mode for compatibility with local Qwen3-30B teacher. Plan to use it as the primary post-training path before attempting GRPO RL. This is more efficient given we have a capable teacher already installed.

**Priority: MEDIUM** — practical shortcut. Revisit at N-9 (SFT) when the post-training plan is finalized.

---

### D-20 — Flash Attention 4 (FA4): SM120 / RTX 5090 Support Active in Beta

**Why:** Enigma currently uses standard PyTorch `scaled_dot_product_attention`. Flash Attention 3 (for H100 Hopper/SM90) delivered 1.5-2× training speedup and 75% theoretical FLOP utilization. FA4 extends this to Blackwell — specifically SM120 (RTX 5090 class, Blackwell GeForce).

**Verified status (github.com/Dao-AILab/flash-attention/releases, verified Pass 148):**
- FA4 latest tag is **v4.0.0.beta10** (releases cadence ~weekly, active development).
- **SM120 (Blackwell GeForce = RTX 5090 / 5080):** **fwd + bwd + varlen all merged** via PRs #2329, #2330, #2333 (author blake-snc). Confirmed in source on `main`.
- **Install:** `pip install flash-attn-4` works **on Linux** (FA4 ships as a separate package alongside FA2/FA3, distinct from `flash-attn`). FA4 is built on **CuTeDSL** rather than the older CUTLASS C++ template path.
- **Windows:** no official FA4 Windows wheel in release assets. The README still notes Windows compilation "requires more testing" even for FA2 (current PyPI flash-attn 2.8.3, Aug 15 2025). Until upstream ships Windows wheels, our 5090 + Windows 11 target box can only consume FA4 via WSL2 or by self-compiling, both of which are out-of-scope for a multi-month training run.
- SM100 (B200/H200 data center): GQA, FP8 E4M3/E5M2, paged attention for MLA, block-sparse — present (not relevant to us).

**Current situation:** FA4 kernels for our exact GPU (SM120) are merged and installable on Linux. The blocker for Enigma is purely the **Windows wheel + multi-month training stability**, not the kernel itself. Worth wiring behind a config flag now and exercising it in WSL benchmarks before committing it to a long pre-training run.

**When ready to integrate:**
1. `pip install flash-attn-4` (Linux/WSL) or compile from source for Windows.
2. Add `use_flash_attn: bool = False` to `TrainingConfig`.
3. In attention forward: gate on `torch.cuda.get_device_capability() >= (12, 0)` for SM120 path, fall back to existing SDPA otherwise.
4. Verify training loss matches non-flash baseline on small data before any full run.

**Action:** Bench FA4 vs SDPA on RTX 5090 under WSL2 in Pass 149+. Hold off on production Windows-native FA4 until upstream ships an official Windows wheel.

**Priority: LOW (timing)** — high value when stable. Don't add a pre-release dependency to a multi-month training run.

---

### Summary Table

| ID | Area | Impact | Cost | Timing | Sources |
|----|------|--------|------|--------|---------|
| D-1 | Add DCLM + FineMath data | High | Low (scripting) | Before N-6 restart | SmolLM3, FineMath paper |
| D-2 | Add The Stack v2 code data | High | Low (scripting) | Before N-6 restart | SmolLM3 |
| D-3 | Three-stage training schedule | Medium | Medium | Before N-6 restart | SmolLM3, **Qwen3**, **DeepSeek-V3** |
| D-4 | Reasoning mid-training phase | High | Medium (new phase) | After N-9 SFT | SmolLM3, OpenThoughts3 |
| D-5 | WSD scheduler option | Medium | Low (15 lines) | Before N-6 restart | SmolLM3, OLMo 2 |
| D-6 | Expand tokenizer to 64K | High | Medium (retrain BPE) | **Before N-6 restart** | LLaMA3, Gemma3, Qwen3 (151K) |
| D-7 | Intra-document masking | Medium | Low (verify first) | Before N-6 restart | LLaMA3, SmolLM3 |
| D-8 | No weight decay on embeddings | Low-Med | Low (5 lines) | Before N-6 restart | OLMo 2, SmolLM3 |
| D-9 | APO alignment mode | Low-Med | Low (15 lines) | After N-10 | APO paper (D'Oosterlinck 2024) |
| D-10 | Standard eval benchmarks | Medium | Medium | After N-7 | Universal practice |
| D-11 | SmolTalk2 SFT data | Medium | Low (scripting) | Before N-9 | SmolLM3 |
| D-12 | NoPE hybrid attention | Low | Low (architecture) | **Before N-6 if possible** | NoPE paper (Yang 2025) |
| D-13 | Context length extension | Low | Low (config + YaRN) | After N-10 | Gemma3, SmolLM3, **Qwen3**, **DeepSeek-V3** |
| D-14 | QK-Norm for training stability | **High** | Low (3-5 lines) | **Before N-6 restart** | **Qwen3** (arXiv:2505.09388), ViT-22B (arXiv:2302.05442) |
| D-15 | MTP training objective | Medium | Medium (verify scaffold) | Before N-6 restart | **DeepSeek-V3** only (arXiv:2412.19437) |
| D-16 | Three-stage training confirmed | — | — | — | Cross-ref for D-3: SmolLM3 + **Qwen3** + **DeepSeek-V3** |
| D-17 | RoPE ABF exact parameters | Low | Low (config field) | With D-13 | **Qwen3** + **DeepSeek-V3** full papers |
| D-18 | Aux-loss-free MoE balancing | Low (if MoE) | Low (when needed) | When MoE enabled | **DeepSeek-V3** (arXiv:2412.19437) |
| D-19 | Strong-to-weak distillation | Medium | Low (verify FORGE) | At N-9 SFT | **Qwen3** (Section 4.4) |
| D-20 | Flash Attention 4 / SM120 | Medium | Low (when stable) | Q3-Q4 2026 | FA4 releases (beta10 confirmed SM120) |

**Critical path before resuming N-6:** D-14 (QK-Norm, architecture) → D-1 → D-2 → D-6 (tokenizer) → D-7/D-8 → restart. D-15 MTP worth verifying before N-6 if scaffold is already present.

**Research confidence:** Items cross-validated from full paper reads (marked **bold**) in Sources column. D-1/D-2/D-4/D-5/D-8/D-11 from first pass only — still single-sourced, still directionally correct but less precisely specified.

---

### Open Audit Findings (Pass 110)

| # | Severity | File | Issue |
|---|----------|------|-------|
| ~~S812~~ | ~~Correctness~~ | ~~[enigma_engine/core/model_components.py](enigma_engine/core/model_components.py)~~ | **Fixed (Pass 115).** Causal mask rebuilt with torch.triu for merged sequence length. |
| ~~S813~~ | ~~Correctness~~ | ~~[enigma_engine/core/rl_training.py](enigma_engine/core/rl_training.py)~~ | **Fixed (Pass 115).** ReplayBuffer now stores full_ids/prompt_len/ref_logps so PPO can recompute log-probs for replay items. |
| ~~S815~~ | ~~Security~~ | ~~[enigma_engine/api/server.py](enigma_engine/api/server.py)~~ | **Fixed (Pass 115).** run.py reads CONFIG["enigma_api_key"] as fallback when --api-key not on CLI. |
| ~~S816~~ | ~~Security~~ | ~~[enigma_engine/api/server.py](enigma_engine/api/server.py)~~ | **Fixed (Pass 111).** Returns paths relative to MODELS_DIR. load_model resolves both relative and absolute. |
| S817 | ~~Security~~ | [enigma_engine/api/server.py](enigma_engine/api/server.py) | **Accepted risk (Pass 115).** Lock prevents GPU corruption during concurrent inference. Single-model engine is inherently single-threaded. All 3 endpoints use same non-blocking acquire with 429 response. |
| ~~S819~~ | ~~Dead UX~~ | ~~[enigma_engine/gui/gui_forge_new_modes.py](enigma_engine/gui/gui_forge_new_modes.py)~~ | **Fixed (Pass 115).** Alignment modes (GRPO/ReMax/SimPO/ORPO) now show only basic section. |
| ~~S820~~ | ~~Config~~ | ~~[enigma_engine/api/server.py](enigma_engine/api/server.py)~~ | **False positive (Pass 111).** Port 8080 is consistent in run_server(), run_serve(), and docstring. No 8000 in code. |
| ~~S821~~ | ~~Comment~~ | ~~[enigma_engine/core/model_components.py](enigma_engine/core/model_components.py)~~ | **False positive (Pass 111).** Inline comment correctly says "dot product of L2-normalized features" — which IS cosine similarity. Docstring says "most similar pairs", not "cosine". |
| ~~S822~~ | ~~Missing~~ | ~~[enigma_engine/core/model_components.py](enigma_engine/core/model_components.py)~~ | **Fixed (Pass 115).** Skip ToMe when T > 4096 (O(T²) similarity matrix would OOM). |
| ~~S823~~ | ~~Perf~~ | ~~[enigma_engine/core/rl_training.py](enigma_engine/core/rl_training.py)~~ | **Fixed (Pass 117).** Added `_get_logps_hidden_entropy()` — single model pass returning logps + hidden states + entropy. Replaced triple policy passes in RLHF rollout (2→1), RLHF minibatch (3→1), SelfPlay rollout (2→1), SelfPlay minibatch (3→1), ReMax update (2→1). Ref-model calls unchanged. 4 tests verify logps/hidden/entropy all match separate calls. 2262 tests pass. |
| ~~S824~~ | ~~Precision~~ | ~~[enigma_engine/core/model_components.py](enigma_engine/core/model_components.py)~~ | **Fixed (Pass 111).** Accumulator upcast to fp32, weighted_output.float() for index_add_, cast back to input dtype before return. |
| ~~S825~~ | ~~Dropped~~ | ~~[enigma_engine/core/streaming.py](enigma_engine/core/streaming.py)~~ | **Fixed (Pass 115).** Changed to unbounded queue (maxsize=0). Stream is finite, tokens are tiny strings. |
| ~~S828~~ | ~~Correctness~~ | ~~[enigma_engine/gui/gui_forge_tools.py](enigma_engine/gui/gui_forge_tools.py)~~ | **Fixed (Pass 112).** _save_forge_checkpoint() now checks _active_trainer; during active training calls trainer._save_checkpoint(dest) for live weights. When idle, falls back to shutil.copy2 (file is up-to-date). _active_trainer stored/cleared across all 13 training paths (7 missing assignments added in audit). |
| ~~S829~~ | ~~Perf~~ | ~~[enigma_engine/gui/gui_forge_tools.py](enigma_engine/gui/gui_forge_tools.py)~~ | **Fixed (Pass 112).** All 3 presets (Quick/Balanced/Thorough) changed from hardcoded batch=2/4 to "auto". Widget default was already "auto". Auto-batch selects optimal size for available VRAM. |
| ~~S830~~ | ~~Correctness~~ | ~~[enigma_engine/core/training.py](enigma_engine/core/training.py)~~ | **Fixed (Pass 112).** Added save_every_steps field to TrainingConfig (default 0=disabled). Training loop saves checkpoint every N steps with keep=3 cleanup. Pre-train auto-sets to max(500, total_steps // 20). |
| ~~S840~~ | ~~Hygiene~~ | ~~[enigma_engine/gui/gui_forge.py](enigma_engine/gui/gui_forge.py)~~ | **Fixed.** Removed unused local `import json` in `_save_training_brief()` (ruff F401). |
| ~~S841~~ | ~~Hygiene~~ | ~~[tests/test_new_features.py](tests/test_new_features.py)~~ | **Fixed.** Removed unused top-level `import json` (ruff F401). |
| ~~S842~~ | ~~Hygiene~~ | ~~[tests/test_new_features.py](tests/test_new_features.py)~~ | **Fixed.** Removed unused local `import os` in `test_cleanup_removes_env_vars()` (ruff F401). |
| ~~S843~~ | ~~Hygiene~~ | ~~[tests/test_new_features.py](tests/test_new_features.py)~~ | **Fixed.** Removed unused local `import math` in `test_moving_average_nan_does_not_leak_stale_values()` (ruff F401). |
| ~~S844~~ | ~~Hygiene~~ | ~~[tests/test_new_features.py](tests/test_new_features.py)~~ | **Fixed.** Removed redundant local `import json` redefinitions (ruff F811). |

---

## Accepted Risk (no action needed)

| # | File | Why accepted |
|---|------|-------------|
| S725 | [collect_pretraining_data.py](collect_pretraining_data.py) | SHA-256 truncated to 64 bits. Collision at ~4B paragraphs — we have ~50M. |
| S769 | [enigma_engine/router.py](enigma_engine/router.py) | `mod.last_seen` scalar write. CPython GIL atomic. |
| S791 | [collect_pretraining_data.py](collect_pretraining_data.py) | Filename collision on 80-char truncation. Never observed. |
| S792 | [collect_pretraining_data.py](collect_pretraining_data.py) | XML bomb risk. Trusted sources only. |
| S793 | [collect_pretraining_data.py](collect_pretraining_data.py) | FineWeb resume O(n) skip. HF datasets limitation. Data done. |
| S817 | [enigma_engine/api/server.py](enigma_engine/api/server.py) | SSE lock held for entire stream. Single-model GPU engine is inherently single-threaded. Non-blocking acquire + 429 response. |
| S796 | [enigma_engine/core/huggingface_loader.py](enigma_engine/core/huggingface_loader.py) | Stream thread exceptions silently lost. Display-only. |
| S797 | [enigma_engine/core/huggingface_loader.py](enigma_engine/core/huggingface_loader.py) | Param estimation inaccurate for GQA/MoE. Display-only. |

---

## Reference

**Data sources (all done):** FineWeb-Edu 40.2 GB, C4 20.1 GB, Wiki dump 15.1 GB, OWT 10.1 GB, Gutenberg 1.5 GB, SE 1.1 GB, Wayback 64 MB, Fandom 46 MB, Wikipedia/Simple 22 MB.
**Future sources:** ArXiv, The Stack (code), OpenWebMath, PubMed.
**Fine-tuning data:** OASST, Dolly 15k, SlimOrca, UltraChat.
**Parked tech:** nGPT, DoReMi, PagedAttention, Mamba/SSM, Neuro-Symbolic, Vision datasets, MoE (all too invasive or wrong scale for 742M).
**Rejected sources:** Common Crawl, The Pile, Wiktionary, Wikiquote.
