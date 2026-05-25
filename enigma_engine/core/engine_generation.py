"""
Generation and sampling methods for EnigmaEngine.

This module is split from ``inference.py`` to keep file sizes manageable.
The :class:`_GenerationMixin` is mixed into :class:`EnigmaEngine` so that
every method is available on the engine instance as before.

**Do not import from ``inference`` here** — that would create a circular
import.  All instance attributes (``self.model``, ``self.tokenizer``,
``self.device``, etc.) are set by ``EnigmaEngine.__init__`` which lives
in ``inference.py``.
"""
from __future__ import annotations

import logging
import re
import typing
from collections.abc import Generator

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Pre-compiled patterns for direct tool routing
_GENERATION_PATTERNS = [
    re.compile(r'(?:draw|paint|create|generate|make|produce)\s+(?:me\s+)?(?:a\s+)?(?:picture|image|photo|illustration|artwork|video|clip|animation|sound|audio|speech|model|mesh|gif)?\s*(?:of\s+)?(.+)', re.IGNORECASE),
    re.compile(r'(?:draw|paint|create|generate|make|produce|speak|say|read)\s+(?:me\s+)?(.+)', re.IGNORECASE),
    re.compile(r'(?:picture|image|photo|video|audio|model)\s+of\s+(.+)', re.IGNORECASE),
]
_WEB_SEARCH_PATTERNS = [
    re.compile(r'(?:search|google|look up|find|browse)\s+(?:for\s+)?(.+)', re.IGNORECASE),
    re.compile(r'what is\s+(.+)', re.IGNORECASE),
    re.compile(r'who is\s+(.+)', re.IGNORECASE),
]


class _GenerationMixin:
    """Generation, sampling, routing, and batch methods mixed into EnigmaEngine."""

    # =========================================================================
    # 🔀 ROUTING HELPERS — decide whether to use AI or direct tool dispatch
    # =========================================================================

    def _needs_ai_creativity(self, prompt: str) -> bool:
        """
        Check if the prompt requires AI creativity/context rather than direct execution.

        Returns True for ambiguous or creative requests that need AI interpretation.
        Also returns True for non-Latin script prompts (CJK, Arabic, Cyrillic, etc.)
        where English keyword matching cannot work — routing to the AI is the safe
        default since the AI can still handle direct commands.
        """
        prompt_lower = prompt.lower()

        # --- Non-Latin script heuristic ---
        # If the prompt is predominantly non-Latin characters, we can't
        # pattern-match creativity indicators.  Safe default: let the AI handle it.
        non_latin = sum(1 for ch in prompt if ch.isalpha() and ord(ch) > 0x024F)
        latin = sum(1 for ch in prompt if ch.isalpha() and ord(ch) <= 0x024F)
        if non_latin > latin:
            return True

        # --- English creativity indicators ---
        creativity_indicators = [
            "what do you think",
            "surprise me",
            "something cool",
            "something interesting",
            "be creative",
            "your choice",
            "you decide",
            "like before",
            "like last time",
            "similar to",
            "in the style of",
            "mood",
            "feeling",
            "vibe",
            "not sure",
            "maybe",
            "suggest",
            "recommend",
            "what would",
            "how about",
            "can you think of",
        ]

        for indicator in creativity_indicators:
            if indicator in prompt_lower:
                return True

        # Very short prompts might be ambiguous
        words = prompt.split()
        if len(words) <= 2:
            # Single word commands are usually direct ("draw cat")
            # But single words alone are ambiguous
            if len(words) == 1 and words[0].lower() not in ['draw', 'paint', 'generate', 'create', 'make', 'speak', 'say']:
                return True

        return False

    def _try_direct_routing(self, intent: str, prompt: str) -> str | None:
        """
        Try to handle the request directly without main AI.

        Returns the response string if handled, None if should fall through to AI.
        """
        if intent == "image":
            logger.info("Direct routing to image generation")
            return self._direct_generation(prompt, "image", "generate_image")

        elif intent == "video":
            logger.info("Direct routing to video generation")
            return self._direct_generation(prompt, "video", "generate_video")

        elif intent == "audio":
            logger.info("Direct routing to audio/speech generation")
            return self._direct_generation(prompt, "audio", "speak_text")

        elif intent == "3d":
            logger.info("Direct routing to 3D generation")
            return self._direct_generation(prompt, "3D model", "generate_3d")

        elif intent == "gif":
            logger.info("Direct routing to GIF generation")
            return self._direct_generation(prompt, "GIF", "generate_gif")

        elif intent == "code" and hasattr(self._tool_router, 'generate_code'):
            logger.info("Using specialized code generation model")
            return self._tool_router.generate_code(prompt)

        elif intent == "vision" and hasattr(self._tool_router, 'describe_image'):
            logger.info("Vision routing detected, but no features provided")
            # Fall through - vision needs actual image input
            return None

        elif intent == "web":
            logger.info("Direct routing to web search")
            return self._direct_web_search(prompt)

        # Unknown intent or no direct handler - let AI handle it
        return None

    def _direct_generation(self, prompt: str, content_type: str, tool_name: str) -> str:
        """
        Generic direct generation handler for image/video/audio/3D/gif.

        Extracts description and calls the appropriate tool directly.
        """
        # Common patterns for extracting the actual content description
        description = prompt

        for pattern in _GENERATION_PATTERNS:
            match = pattern.search(prompt)
            if match:
                description = match.group(1).strip()
                break

        # Clean up
        description = description.strip('.,!? ')
        if not description:
            description = prompt

        logger.info(f"Direct {content_type} generation: {description}")

        if not self._tool_executor:
            return f"To generate {content_type}, please use the {content_type.title()} tab. Direct generation not available."

        # Execute the tool (tool executor handles auto-loading)
        result = self._tool_executor.execute_tool(tool_name, {"prompt": description})

        if result.get("success"):
            path = result.get("path", result.get("result", {}).get("path", ""))
            duration = result.get("duration", 0)
            return f"I've generated {content_type} of '{description}' for you.\n\nSaved to: {path}\nGeneration time: {duration:.1f}s"
        else:
            error = result.get("error", "Unknown error")
            tab_name = content_type.title().replace("3d", "3D")
            return f"I tried to generate {content_type} but encountered an error: {error}\n\nYou can try using the {tab_name} tab directly."

    def _direct_web_search(self, prompt: str) -> str:
        """Direct web search without AI intermediary."""
        # Check if web access is enabled
        if not getattr(self, '_web_enabled', False):
            return "Web access is disabled. Click the 'Web' button in the chat header to enable internet access."

        # Extract search query
        query = prompt

        for pattern in _WEB_SEARCH_PATTERNS:
            match = pattern.search(prompt)
            if match:
                query = match.group(1).strip()
                break

        query = query.strip('?,!. ')

        if not self._tool_executor:
            return f"To search for '{query}', please use the web tools. Direct search not available."

        result = self._tool_executor.execute_tool("web_search", {"query": query})

        if result.get("success"):
            search_results = result.get("result", result.get("results", "No results found"))
            return f"Search results for '{query}':\n\n{search_results}"
        else:
            error = result.get("error", "Unknown error")
            return f"Web search failed: {error}"

    # =========================================================================
    # 🔧 INTERNAL GENERATION - Where the magic happens!
    # =========================================================================

    # Pattern for structured tool calls emitted by the model.
    # Format: <tool_call>{"name": "tool_name", "args": {...}}</tool_call>
    _TOOL_CALL_RE = re.compile(
        r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)

    def _execute_tools_in_text(self, text: str, **kwargs) -> str:
        """Scan generated text for tool-call markers and execute them.

        Implements the agentic tool loop:
          1. Parse <tool_call>{"name":"...", "args":{...}}</tool_call>
          2. Execute via _tool_executor
          3. Replace marker with <tool_result>...</tool_result>
          4. Re-generate from the updated context
          5. Repeat until no more tool calls or max_iterations reached

        Called inside ``generate()`` while ``_generation_lock`` is held.
        """
        import json as _json

        max_iterations = kwargs.pop("max_iterations", 5)
        if not self._tool_executor:
            return text

        for iteration in range(max_iterations):
            match = self._TOOL_CALL_RE.search(text)
            if not match:
                break

            # Parse the tool call JSON
            try:
                call_data = _json.loads(match.group(1))
            except _json.JSONDecodeError:
                logger.warning(
                    "Malformed tool call JSON: %s", match.group(1))
                # Remove the bad marker so we don't loop forever
                text = (text[:match.start()]
                        + "<tool_result>Error: malformed tool call"
                        + "</tool_result>"
                        + text[match.end():])
                continue

            tool_name = call_data.get("name", "")
            tool_args = call_data.get("args", {})

            if not tool_name:
                text = (text[:match.start()]
                        + "<tool_result>Error: missing tool name"
                        + "</tool_result>"
                        + text[match.end():])
                continue

            # Sanitize: only allow string/number/bool values in args
            safe_args = {}
            for k, v in tool_args.items():
                if isinstance(v, (str, int, float, bool)):
                    safe_args[str(k)] = v

            logger.info(
                "Tool call [%d/%d]: %s(%s)",
                iteration + 1, max_iterations,
                tool_name, safe_args)

            # Execute the tool
            try:
                result = self._tool_executor.execute_tool(
                    tool_name, safe_args)
                if result.get("success"):
                    result_text = str(
                        result.get("result",
                                   result.get("output", "OK")))
                else:
                    result_text = (
                        "Error: "
                        + result.get("error", "Tool execution failed"))
            except Exception as exc:
                logger.warning("Tool execution error: %s", exc)
                result_text = f"Error: {exc}"

            # Truncate long results to avoid blowing up context
            max_result_len = 2000
            if len(result_text) > max_result_len:
                result_text = (
                    result_text[:max_result_len] + "... (truncated)")

            # Replace tool call with result
            replacement = (
                f"<tool_result>{result_text}</tool_result>")
            text = (text[:match.start()]
                    + replacement
                    + text[match.end():])

            # Re-generate continuation after the tool result
            max_gen = kwargs.get("max_gen", 200)
            temperature = kwargs.get("temperature", 0.7)
            top_k = kwargs.get("top_k", 50)
            top_p = kwargs.get("top_p", 0.9)
            repetition_penalty = kwargs.get(
                "repetition_penalty", 1.1)
            stop_strings = kwargs.get("stop_strings")
            use_cache = kwargs.get("use_cache", True)
            min_p = kwargs.get("min_p", 0.0)
            # Pass 156z9fe (Pass A fix #1): forward ``json_schema`` so
            # the tool-call continuation stays constrained.  Without
            # this, ``engine.generate(json_schema=..., execute_tools=
            # True)`` produces a constrained first chunk and then an
            # unconstrained continuation after every ``<tool_call>``
            # marker.  Sibling-boundary site Pass 156z7 named (the
            # original close-stamp wrote "the docstring named this as
            # a caveat with no code-side gate") — closed now.
            json_schema = kwargs.get("json_schema")

            continuation = self._generate_text(
                text, max_gen, temperature, top_k, top_p,
                repetition_penalty, stop_strings, use_cache,
                min_p=min_p, json_schema=json_schema)

            # The continuation includes the full prompt+response;
            # _generate_text returns just the new tokens.
            text = text + continuation

        return text

    def _record_search_emissions(
        self,
        text: str,
        prompt: str | None = None,
        *,
        path: str = "native",
    ) -> None:
        """AutoResearch-2 Stage B-2: scan model-generated text for inline
        ``<search>...</search>`` blocks, store the decoded queries on
        ``self.last_search_queries``, and log a WARNING per call
        (rate-limited to one log line, query count only — full queries
        go in the list for caller inspection).

        Called from every public-facing return path of
        :meth:`_generate_text` so observability is uniform across
        native, GGUF, and tool-execution flows.  Boundary signal:
        when Stage B-3 RAG splice ships, this method becomes the
        wire-site that triggers the splice.

        Args:
            text: The raw decoded sequence.  May be the full
                ``prompt + continuation`` (native paths that decode the
                whole sequence) or just the continuation (GGUF / stream
                paths).  When ``prompt`` is provided, it is stripped
                from the front of ``text`` before scanning so user
                prompts that contain a literal ``<search>foo</search>``
                (e.g. asking about the syntax itself) are NOT recorded
                as model emissions.  Pass 156z9e audit: this slicing
                was missing on 5 of 8 call sites — the docstring
                promised "what the model emitted" but the code was
                scanning prompt+continuation.
            prompt: Optional original prompt string.  When supplied
                and ``text`` starts with it, the prompt prefix is
                removed before scanning.  Pass ``None`` for paths that
                already return continuation-only (GGUF native, stream).

        Pass 156z9c (Stage B-1) ``honest degradation``: if the active
        tokenizer doesn't have ``<search>`` registered (legacy
        vocab), ``search_start_id`` is ``None`` on the tokenizer.
        We don't gate on that here — the helper is purely text-side
        and works on the decoded string regardless of tokenizer
        registry state.  The two layers are decoupled by design.

        Pass 156z9u (B-2c) off-switch: when
        ``self.inline_search_enabled`` is False the scan is skipped
        entirely and ``last_search_queries`` is reset to ``[]`` so
        callers see a clean state.  The flag exists so users
        intentionally probing the ``<search>`` syntax (e.g. asking
        the model about it) can suppress the WARNING noise and the
        future Stage B-3 RAG-splice trigger.
        """
        if not getattr(self, "inline_search_enabled", True):
            self.last_search_queries = []
            return
        if prompt and isinstance(text, str) and text.startswith(prompt):
            scan_text = text[len(prompt):]
        else:
            scan_text = text
        try:
            from .reasoning import extract_search_queries
            queries = extract_search_queries(scan_text)
        except Exception:
            # Stage B-2 must NEVER raise into the caller — it's pure
            # observability layered on top of generation.  Log and
            # leave the list empty.
            logger.exception(
                "Stage B-2: search-emission scan crashed; "
                "last_search_queries left empty")
            self.last_search_queries = []
            return

        self.last_search_queries = queries
        if queries:
            logger.warning(
                "Stage B-2: model emitted %d <search> request(s) but "
                "Stage B-3 RAG splice is not implemented; queries "
                "recorded on engine.last_search_queries for inspection",
                len(queries))
            # B-3a sibling-boundary warning.  When the splice opt-in
            # flag is ON, ``native`` (`_generate_text`), ``stream``
            # (`stream_generate`, B-3d Pass 156z9al),  ``speculative``
            # / ``medusa`` / ``lookahead`` (Pass 156z9cp), ``vision``
            # (`_generate_with_vision`, Pass 156z9do), and ``batch``
            # (`batch_generate`, Pass 156z9dp) honour the splice
            # contract.  Only the GGUF path remains WARNING-only and
            # is parked PERMANENTLY (Pass 156z9dq): llama-cpp-python
            # exposes a string ``stop=[...]`` parameter and a
            # logits-processor hook, but our ``_maybe_rag_splice``
            # helper builds raw text-completion prompts while the
            # GGUF chat() path goes through ``create_chat_completion``
            # with role-bounded message templating.  Splicing
            # ``<search_result>…</search_result>`` into the middle
            # of an assistant message produces undefined behaviour
            # on most instruction-tuned chat templates (the
            # ``<|im_end|>`` boundary is implicit in the template,
            # not exposed by the API).  Shipping a half-correct
            # splice here would silently corrupt every GGUF chat
            # call that triggered a search request.  See the Pass
            # 156z9dq SUGGESTIONS.md stamp for the parked-permanently
            # decision rationale and the concrete next step needed
            # (chat-template-aware retrieval injection at the
            # messages-list level, not the text level).
            if (
                getattr(self, "inline_search_splice_enabled", False)
                and path not in (
                    "native", "stream", "speculative",
                    "medusa", "lookahead", "vision", "batch",
                )
            ):
                logger.warning(
                    "B-3a: inline_search_splice_enabled=True but the "
                    "'%s' generation path does not yet support "
                    "</search> auto-stop or splice; %d query(ies) "
                    "recorded on engine.last_search_queries with no "
                    "splice applied.",
                    path, len(queries))

    def _generate_text(
        self,
        prompt: str,
        max_gen: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        stop_strings: list[str] | None,
        use_cache: bool,
        min_p: float = 0.0,
        json_schema: dict | None = None,
    ) -> str:
        """
        Internal method for standard text generation.

        📖 THIS IS THE CORE GENERATION LOOP!

        📐 THE AUTOREGRESSIVE LOOP:
        ┌────────────────────────────────────────────────────────────────────┐
        │  tokens = [15496, 11, 703, 389]  # "Hello, how are"               │
        │                                                                    │
        │  REPEAT max_gen times:                                            │
        │    1. Feed tokens to model → logits                               │
        │    2. Apply repetition penalty to logits                          │
        │    3. Apply temperature scaling                                   │
        │    4. Apply top-k filtering                                       │
        │    5. Apply top-p (nucleus) filtering                             │
        │    6. Sample next token from probabilities                        │
        │    7. Add new token to sequence                                   │
        │    8. Check for stop strings                                      │
        │    9. Check for EOS token                                         │
        │                                                                    │
        │  tokens = [15496, 11, 703, 389, 499, 1804, 2651]                  │
        │                                    └─ newly generated             │
        └────────────────────────────────────────────────────────────────────┘

        📐 REPETITION PENALTY:
        Discourages the model from repeating the same tokens.
        For each token that already appeared in the sequence,
        divide its probability by repetition_penalty.

        📐 TEMPERATURE SCALING:
        logits = logits / temperature
        - Low temp (0.3): Makes high-prob tokens even more likely → focused
        - High temp (1.5): Flattens distribution → more random

        📐 TOP-K FILTERING:
        Keep only the K highest probability tokens, zero out the rest.
        Prevents sampling very unlikely tokens.

        📐 TOP-P (NUCLEUS) FILTERING:
        Sort tokens by probability, keep tokens until cumulative prob >= p.
        Dynamic cutoff - keeps more tokens when uncertain, fewer when confident.
        """
        # ─────────────────────────────────────────────────────────────────────
        # GGUF MODEL HANDLING (llama.cpp)
        # GGUF models have their own generation - use it directly
        # ─────────────────────────────────────────────────────────────────────
        if getattr(self, '_is_gguf', False) or (hasattr(self.model, 'is_loaded') and hasattr(self.model, 'chat')):
            # This is a GGUFModel - use its native generation
            if json_schema is not None:
                # GGUF models use llama.cpp's own sampler — our logit
                # mask never gets a chance to run. Be loud about the
                # mismatch instead of silently producing unconstrained
                # output that the caller will trust as schema-valid.
                raise NotImplementedError(
                    "json_schema constrained decoding is not supported "
                    "on GGUF models (llama.cpp uses its own sampler). "
                    "Load a native PyTorch model or drop the schema."
                )
            try:
                text = self.model.generate(
                    prompt,
                    max_tokens=max_gen,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repeat_penalty=repetition_penalty,
                    stop=stop_strings
                )
                # GGUF path: continuation-only text, sibling-boundary
                # WARNING fires from inside the helper when the splice
                # flag is ON.
                self._record_search_emissions(text, path="gguf")
                return text
            except Exception as e:
                logger.error(f"GGUF generation failed: {e}")
                return f"Error: {e}"

        # ─────────────────────────────────────────────────────────────────────
        # INPUT VALIDATION
        # ─────────────────────────────────────────────────────────────────────
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a string, got {type(prompt).__name__}")

        if not prompt.strip():
            logger.warning("Empty prompt provided")
            return ""

        if max_gen <= 0:
            raise ValueError(f"max_gen must be positive, got {max_gen}")

        if temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {temperature}")

        if top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {top_k}")

        if not 0 <= top_p <= 1:
            raise ValueError(f"top_p must be between 0 and 1, got {top_p}")

        if repetition_penalty < 1.0:
            raise ValueError(f"repetition_penalty must be >= 1.0, got {repetition_penalty}")

        # ─────────────────────────────────────────────────────────────────────
        # TOKENIZE: Convert text to numbers the model understands
        # "Hello" → [15496]
        # ─────────────────────────────────────────────────────────────────────
        input_ids = self._encode_prompt(prompt)

        # ─────────────────────────────────────────────────────────────────────
        # PREFIX KV CACHE: Check for cached system-prompt KV (set by chat())
        # ─────────────────────────────────────────────────────────────────────
        prefix_cache = getattr(self, "_pending_prefix_cache", None)
        prefix_build = getattr(self, "_pending_prefix_build", None)
        self._pending_prefix_cache = None
        self._pending_prefix_build = None

        # ─────────────────────────────────────────────────────────────────────
        # GENERATE: Run the autoregressive loop
        # All native generation goes through _generate_manual() which has
        # KV-cache, windowed repetition penalty, and min-p — one path.
        # ─────────────────────────────────────────────────────────────────────
        # Pass 156z3 (N-15): build JSON schema constraint once per call.
        # Construction iterates the full vocab to build a first-char →
        # token-id lookup; cost is paid once here, not per token.
        json_constraint = None
        if json_schema is not None:
            from .json_schema_mask import JsonSchemaConstraint
            json_constraint = JsonSchemaConstraint(json_schema, self.tokenizer)

        # B-3a: when ``inline_search_splice_enabled`` is True, append
        # ``</search>`` to the stop list so the model halts cleanly on
        # the closing tag instead of streaming past it into more text
        # we'll discard later.  Defensive copy so we don't mutate the
        # caller's list (some callers pass module-level constants).
        # Native non-GGUF path here; sibling paths that also honour
        # the splice contract: ``stream`` (Pass 156z9al), ``speculative``
        # / ``medusa`` / ``lookahead`` (Pass 156z9cp).  The remaining
        # sibling paths (vision / batch / GGUF) warn via
        # ``_record_search_emissions(path=...)`` instead.
        effective_stop_strings = stop_strings
        if getattr(self, "inline_search_splice_enabled", False):
            effective_stop_strings = list(stop_strings or [])
            if "</search>" not in effective_stop_strings:
                effective_stop_strings.append("</search>")

        with torch.no_grad():  # Disable gradient computation (inference only)
            output_ids = self._generate_manual(
                input_ids, max_gen, temperature, top_k, top_p,
                repetition_penalty, min_p,
                stop_strings=effective_stop_strings,
                prefix_cache=prefix_cache,
                json_constraint=json_constraint,
            )

            # Snapshot prefix KV on cache miss so next turn can skip prefill
            if prefix_build is not None and prefix_cache is None:
                try:
                    sys_ids = self._encode_prompt(
                        prefix_build["system_prefix_text"])
                    n = sys_ids.shape[1]
                    # Verify tokenization boundary consistency
                    if (n <= input_ids.shape[1]
                            and torch.equal(
                                input_ids[0, :n], sys_ids[0])):
                        from .kv_cache import PrefixKVCache
                        new_cache = PrefixKVCache()
                        new_cache.build_from_layers(
                            self.model.layers, prefix_len=n)
                        new_cache.prompt_hash = prefix_build["hash"]
                        self._prefix_kv_cache = new_cache
                        self._prefix_prompt_hash = prefix_build["hash"]
                        logger.info(
                            "Built prefix KV cache: %d tokens, hash=%s",
                            n, prefix_build["hash"])
                    else:
                        logger.debug(
                            "Skipped prefix cache build: token boundary "
                            "mismatch (sys=%d, input=%d)",
                            n, input_ids.shape[1])
                except Exception:
                    logger.debug(
                        "Prefix cache build failed", exc_info=True)

        # Decode
        text = self._decode_output(output_ids)

        # Apply stop strings - only check in generated part (after prompt)
        # This prevents cutting off at stop strings that exist in the prompt itself
        # Use ``effective_stop_strings`` so the B-3a auto-stop on
        # ``</search>`` also trims the decoded text (defence in depth
        # if the manual loop yielded one extra token past the match).
        if effective_stop_strings:
            prompt_len = len(prompt)
            generated_part = text[prompt_len:] if len(text) > prompt_len else text
            for stop_str in effective_stop_strings:
                if stop_str in generated_part:
                    stop_pos = generated_part.find(stop_str)
                    text = text[:prompt_len + stop_pos]
                    break

        # B-3b/B-3c: when the splice flag is ON, the auto-stop fired
        # on ``</search>``, and a RAG index is attached, retrieve and
        # splice ``<search_result>...</search_result>`` then continue
        # generation.  Up to ``self.max_search_rounds`` rounds (default
        # 3).  Returns ``None`` when any precondition fails so the
        # caller keeps the original text.
        # Pass the round-0 emitted-token count so the helper can budget
        # remaining rounds against the original ``max_gen`` instead of
        # multiplying it by N.
        tokens_round0 = max(0, output_ids.shape[1] - input_ids.shape[1])
        spliced = self._maybe_rag_splice(
            text, prompt, max_gen, temperature, top_k, top_p,
            repetition_penalty, min_p,
            effective_stop_strings=effective_stop_strings,
            json_constraint=json_constraint,
            tokens_already_generated=tokens_round0,
        )
        if spliced is not None:
            text = spliced

        self._record_search_emissions(text, prompt=prompt)
        return text

    def _maybe_rag_splice(
        self,
        text: str,
        prompt: str,
        max_gen: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        min_p: float,
        *,
        effective_stop_strings: list[str] | None,
        json_constraint: object | None,
        tokens_already_generated: int = 0,
    ) -> str | None:
        """B-3b/B-3c: retrieve + splice ``<search_result>`` after a
        ``</search>`` auto-stop, then continue generation.  Bounded
        multi-round recursion gated by
        ``self.max_search_rounds`` (default 3, set in
        :meth:`EnigmaEngine._init_common`).

        Round budget semantics:

        * Round 1..N-1: continuation calls KEEP ``</search>`` in
          ``stop_strings`` so the model can request another search.
          If it does, we splice and loop.
        * Round N (budget exhausted): one final continuation call
          with ``</search>`` STRIPPED from stops so the model wraps
          up using the accumulated context.  Any further `<search>`
          tags appear as plain text — no recursion past N.

        Token budget (Pass 156z9ak): the caller passes the round-0
        emitted-token count via ``tokens_already_generated``.  Each
        helper round subtracts its own emitted-token delta from a
        running cumulative; per-round ``max_gen`` is set to
        ``max_gen - cumulative`` so the total user budget is
        respected across all rounds instead of being multiplied by
        N.  When the remaining budget hits zero, the loop exits
        cleanly.

        Preconditions for the FIRST splice (any failure returns
        ``None`` and the caller keeps the original text):

        * ``self.inline_search_splice_enabled`` is True.
        * ``self._rag_index`` is attached and built.
        * ``text`` starts with ``prompt`` (Pass 156z9e prompt-echo
          discipline — generated portion only).
        * Generated portion ends with an unclosed ``<search>``
          carrying a non-empty query.
        * Retrieval returns a non-empty formatted context.

        Within the loop, the same preconditions are re-checked each
        round before issuing another retrieval; any miss exits the
        loop and returns the most recent text.
        """
        if not getattr(self, "inline_search_splice_enabled", False):
            return None
        rag_index = getattr(self, "_rag_index", None)
        if rag_index is None or not getattr(rag_index, "is_built", False):
            return None
        if not isinstance(text, str) or not isinstance(prompt, str):
            return None
        if not text.startswith(prompt):
            return None

        max_rounds = max(1, int(getattr(self, "max_search_rounds", 3)))
        current_text = text
        current_prompt = prompt  # advances each round so prompt-echo
        # discipline applies to the freshly-generated tail, not the
        # cumulative spliced prompt.
        spliced_any = False
        # B-3c-2 (Pass 156z9ak): cumulative emitted-token count across
        # round 0 (passed in by caller) plus all helper rounds, so each
        # round's per-call ``max_gen`` is decremented from the original
        # user budget instead of using the full budget every round.
        cumulative_tokens = max(0, int(tokens_already_generated))

        for round_idx in range(max_rounds):
            generated = current_text[len(current_prompt):]
            open_pos = generated.rfind("<search>")
            if open_pos < 0:
                break
            after_open = generated[open_pos + len("<search>"):]
            if "</search>" in after_open:
                # Closed pair ⇒ auto-stop did not fire on this
                # ``<search>``; nothing to splice.
                break
            query = after_open.strip()
            if not query:
                break
            try:
                from .rag import RAGIndex
                results = rag_index.query(query, top_k=5)
                ctx = RAGIndex.format_context(results, max_chars=2000)
            except Exception:
                logger.exception(
                    "B-3b/c: RAG retrieval failed at round %d; "
                    "exiting splice loop",
                    round_idx + 1)
                break
            if not ctx:
                break

            splice_block = (
                f"</search>\n<search_result>\n{ctx}\n"
                f"</search_result>\n"
            )
            new_prompt = current_text + splice_block
            logger.info(
                "B-3b/c: round %d/%d spliced RAG result for query "
                "'%s' (%d chars ctx)",
                round_idx + 1, max_rounds, query[:60], len(ctx))
            try:
                new_input_ids = self._encode_prompt(new_prompt)
            except Exception:
                logger.exception(
                    "B-3b/c: encode of spliced prompt failed at "
                    "round %d; exiting splice loop",
                    round_idx + 1)
                break

            # B-3c-2: respect remaining token budget.  If round 0 +
            # prior splice rounds already consumed the user's full
            # ``max_gen``, no continuation tokens left — exit cleanly
            # instead of issuing another full-budget call.
            remaining = max_gen - cumulative_tokens
            if remaining <= 0:
                logger.info(
                    "B-3b/c: max_gen=%d budget exhausted before round "
                    "%d (cumulative=%d); exiting splice loop",
                    max_gen, round_idx + 1, cumulative_tokens)
                break
            round_max_gen = remaining

            # Final round (budget exhaustion): strip ``</search>``
            # from stops so the model wraps up rather than triggering
            # another auto-stop we can't service.
            is_final_round = (round_idx == max_rounds - 1)
            if is_final_round:
                round_stops = [
                    s for s in (effective_stop_strings or [])
                    if s != "</search>"
                ] or None
            else:
                round_stops = effective_stop_strings or None

            try:
                with torch.no_grad():
                    cont_ids = self._generate_manual(
                        new_input_ids, round_max_gen, temperature, top_k,
                        top_p, repetition_penalty, min_p,
                        stop_strings=round_stops,
                        prefix_cache=None,
                        json_constraint=json_constraint,
                    )
            except Exception:
                logger.exception(
                    "B-3b/c: continuation generation failed at "
                    "round %d; returning pre-round text",
                    round_idx + 1)
                break
            # Account this round's emitted tokens against the user
            # budget.  ``cont_ids`` includes the spliced prompt; the
            # delta from ``new_input_ids`` is what the model actually
            # produced this round.
            try:
                this_round_tokens = max(
                    0,
                    int(cont_ids.shape[1]) - int(new_input_ids.shape[1]),
                )
            except Exception:
                this_round_tokens = 0
            cumulative_tokens += this_round_tokens
            cont_text = self._decode_output(cont_ids)
            if round_stops:
                cp_len = len(new_prompt)
                cont_gen = (
                    cont_text[cp_len:]
                    if len(cont_text) > cp_len else cont_text
                )
                for s in round_stops:
                    if s in cont_gen:
                        pos = cont_gen.find(s)
                        cont_text = cont_text[:cp_len + pos]
                        break

            current_text = cont_text
            current_prompt = new_prompt
            spliced_any = True

            if is_final_round:
                # Budget exhausted; loop terminates regardless of
                # whether another <search> was emitted.
                if "<search>" in current_text[len(current_prompt):]:
                    logger.warning(
                        "B-3c: max_search_rounds=%d budget exhausted "
                        "but model emitted another <search> tag; "
                        "left as plain text",
                        max_rounds)
                break

        return current_text if spliced_any else None

    # How often (in tokens) to check for stop strings during generation.
    # Lower = catches sooner, higher = less decode overhead.
    _STOP_CHECK_INTERVAL: int = 16

    # How many trailing tokens to decode when checking for stop strings.
    # Stop strings are short, so a window of 50 tokens always captures
    # the match while keeping decode cost O(1) instead of O(generated).
    _STOP_CHECK_WINDOW: int = 50

    @staticmethod
    def _adaptive_stop_interval(
        confidence_history: list[float],
        min_interval: int = 8,
        max_interval: int = 32,
    ) -> int:
        """Compute stop-check interval from rolling model confidence (T2-7).

        High confidence → less likely to hit a stop string → check less often.
        Low confidence → check more frequently.
        """
        if not confidence_history:
            return 16  # default fallback
        avg = sum(confidence_history) / len(confidence_history)
        # Linear interpolation: high confidence → longer interval
        frac = max(0.0, min(1.0, avg))
        return int(min_interval + frac * (max_interval - min_interval))

    def _generate_manual(
        self,
        input_ids: torch.Tensor,
        max_gen: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        min_p: float = 0.0,
        stop_strings: list[str] | None = None,
        prefix_cache=None,
        json_constraint: object | None = None,
    ) -> torch.Tensor:
        """Manual autoregressive generation with KV-cache acceleration.

        Args:
            prefix_cache: Optional :class:`PrefixKVCache` whose KV
                tensors are restored instead of re-computing the
                prefix via a forward pass.  The first
                ``prefix_cache.prefix_len`` tokens of *input_ids*
                are skipped during prefill.
        """
        generated = input_ids
        prompt_len = input_ids.shape[1]
        max_len = self.model.config.max_seq_len

        # Check if model supports KV cache
        has_cache = hasattr(self.model, 'clear_cache')

        # Prefix cache path: restore cached KV, only prefill new tokens
        if prefix_cache is not None and has_cache and prefix_cache.prefix_len > 0:
            try:
                self.model.clear_cache()
                self.model.restore_prefix_cache(prefix_cache)
                prefix_len_tokens = prefix_cache.prefix_len
                suffix = input_ids[:, prefix_len_tokens:]
                if suffix.shape[1] > 0:
                    logits = self.model(
                        suffix, use_cache=True,
                        start_pos=prefix_len_tokens)
                else:
                    # Prompt is exactly the prefix — get logits for last token
                    last_tok = input_ids[:, -1:]
                    logits = self.model(
                        last_tok, use_cache=True,
                        start_pos=prefix_len_tokens - 1)
                logger.debug(
                    "Prefix cache hit: skipped %d-token prefill, "
                    "processed %d new tokens",
                    prefix_len_tokens, suffix.shape[1])
            except Exception as exc:
                # S620: Corrupted/mismatched cache — fall back to full prefill
                logger.warning(
                    "Prefix cache restore failed (%s) — "
                    "falling back to full prefill.", exc)
                if has_cache:
                    self.model.clear_cache()
                logits = self.model(input_ids, use_cache=has_cache)
        else:
            if has_cache:
                self.model.clear_cache()
            else:
                logger.warning(
                    "Model lacks KV cache support — generation will use "
                    "O(n²) full-recompute fallback (slow for long sequences)."
                )

            # Prefill: process entire prompt at once
            curr_input = input_ids
            if curr_input.shape[1] > max_len:
                logger.warning(
                    "Input length %d exceeds max_seq_len %d — "
                    "truncating from the left.",
                    curr_input.shape[1], max_len)
                curr_input = curr_input[:, -max_len:]

            logits = self.model(curr_input, use_cache=has_cache)

        # Build rep-penalty exemption set (T1-6): prompt tokens +
        # special tokens + proper nouns that should legitimately repeat.
        exempt_tokens = self._build_exempt_tokens(input_ids, repetition_penalty)

        # T2-7: Track model confidence for adaptive stop-check interval
        _confidence_history: list[float] = []
        _confidence_window = 4

        for step in range(max_gen):
            # Pass B (Pass 156z9ff): honour user stop signal.
            if self._check_cancel():
                logger.info("Generation cancelled by user at step %d", step)
                break
            # Sample next token
            next_token = self._sample_token(
                logits[:, -1, :],
                generated,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                min_p,
                exempt_tokens=exempt_tokens,
                json_constraint=json_constraint,
            )

            # Track max softmax probability for adaptive stop-check interval.
            # Only needed when stop_strings are active — avoids per-token
            # GPU→CPU sync (.item()) and full-vocab softmax otherwise.
            # Pass 156z9fe (Pass A fix #5): the stop-check itself and
            # the ``_adaptive_stop_interval`` call below are now both
            # gated on ``stop_strings`` so the per-iter helper call is
            # skipped entirely on the common no-stop-strings path.
            if stop_strings:
                with torch.no_grad():
                    max_prob = torch.softmax(
                        logits[:, -1, :], dim=-1).max().item()
                _confidence_history.append(max_prob)
                if len(_confidence_history) > _confidence_window:
                    _confidence_history.pop(0)

            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            eos_id = getattr(self.tokenizer, 'eos_token_id', 2)
            if next_token[0, 0].item() == eos_id:
                break

            # Pass 156z3 (N-15): advance JSON FSM with the new token. The
            # constraint tracks structural state (key/value position,
            # brace depth, closing string detection) by decoding each
            # token's text. Stop early if the FSM reaches DONE — extra
            # tokens past schema completion would have nothing valid to
            # generate and the masker would mask everything to -inf.
            if json_constraint is not None:
                json_constraint.advance(int(next_token[0, 0].item()))
                if json_constraint.is_done:
                    break

            # Periodic stop-string check (amortised decode cost).
            # T2-7: Adaptive interval based on model confidence.
            # Pass 156z9fe (Pass A fix #5): gate the helper call on
            # ``stop_strings`` so the no-stop-strings path skips it.
            if stop_strings:
                check_interval = self._adaptive_stop_interval(
                    _confidence_history)
                if (step + 1) % check_interval == 0:
                    tail_start = max(prompt_len, generated.shape[1] - self._STOP_CHECK_WINDOW)
                    recent_ids = generated[0, tail_start:].tolist()
                    recent_text = self.tokenizer.decode(
                        recent_ids, skip_special_tokens=True)
                    if any(ss in recent_text for ss in stop_strings):
                        break

            # Decode step: only feed new token with start_pos
            if has_cache:
                logits = self.model(
                    next_token,
                    use_cache=True,
                    start_pos=generated.shape[1] - 1,
                )
            else:
                # Fallback: full recompute (no cache support)
                curr_input = generated
                if curr_input.shape[1] > max_len:
                    curr_input = curr_input[:, -max_len:]
                logits = self.model(curr_input)

        return generated

    # Default repetition window. Actual window is adaptive: scales
    # with generation length between 64 and 256 tokens.
    REPETITION_WINDOW: int = 128

    @staticmethod
    def _adaptive_rep_window(seq_len: int) -> int:
        """Compute repetition window proportional to generated length.

        Short outputs get a small window (avoids over-suppressing),
        long outputs get a larger one (prevents loops).
        Clamped to [64, 256].
        """
        return max(64, min(256, seq_len // 2))

    def _start_proper_noun_scan(self):
        """Build proper-noun exemption set in a background thread.

        Scans the full vocab (32K+ tokens) to find capitalized words.
        The result is cached on ``self.tokenizer._proper_noun_ids`` so
        subsequent generation calls use it instantly.  The first
        generation proceeds without it (minor rep-penalty inaccuracy)
        rather than blocking for 10-30 seconds.
        """
        import threading

        tokenizer = self.tokenizer
        try:
            tokenizer._proper_noun_scan_started = True
        except AttributeError:
            return  # frozen tokenizer — skip

        def _scan():
            vocab_size = getattr(tokenizer, 'vocab_size',
                                 getattr(tokenizer, 'get_vocab_size',
                                         lambda: 0)()
                                 if callable(getattr(tokenizer, 'get_vocab_size', None))
                                 else getattr(tokenizer, 'vocab_size', 0))
            proper_ids: set[int] = set()
            for tid in range(vocab_size):
                try:
                    tok = tokenizer.decode([tid])
                    stripped = tok.strip()
                    if stripped and stripped[0].isupper() and stripped.isalpha():
                        proper_ids.add(tid)
                except Exception:
                    continue
            try:
                tokenizer._proper_noun_ids = proper_ids
            except AttributeError:
                pass
            logger.debug("Proper-noun scan complete: %d tokens", len(proper_ids))

        threading.Thread(target=_scan, daemon=True,
                         name="proper-noun-scan").start()

    def _build_exempt_tokens(
        self,
        input_ids: torch.Tensor,
        repetition_penalty: float,
    ) -> set[int] | None:
        """Build rep-penalty exemption set (prompt + special + proper nouns).

        Returns ``None`` when *repetition_penalty* is 1.0 (disabled) so
        callers can short-circuit cheaply.
        """
        if repetition_penalty == 1.0:
            return None
        prompt_ids = input_ids[0].tolist()
        exempt: set[int] = set(prompt_ids)
        for attr in ('bos_token_id', 'eos_token_id', 'pad_token_id'):
            tid = getattr(self.tokenizer, attr, None)
            if tid is not None:
                exempt.add(tid)
        special = getattr(self.tokenizer, 'special_tokens', {})
        if isinstance(special, dict):
            exempt.update(special.values())
        cached = getattr(self.tokenizer, '_proper_noun_ids', None)
        if cached is not None:
            exempt.update(cached)
        elif not getattr(self.tokenizer, '_proper_noun_scan_started', False):
            self._start_proper_noun_scan()
        return exempt if exempt else None

    def _sample_token(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        min_p: float = 0.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        typical_p: float = 0.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        exempt_tokens: set[int] | None = None,
        json_constraint: object | None = None,
    ) -> torch.Tensor:
        """Sample next token with various strategies.

        New optional strategies (R10, R11):
          typical_p: Typical sampling threshold (0 = disabled,
              ~0.2-0.95 reasonable). Keeps tokens whose surprise is
              close to the expected information content.
          mirostat_mode: 0 = disabled, 2 = Mirostat v2.
          mirostat_tau: Target surprise (perplexity = e^tau).
          mirostat_eta: Mirostat learning rate.
          exempt_tokens: Token IDs exempt from repetition penalty
              (T1-6).  Prompt tokens, proper nouns, and special
              tokens that should legitimately repeat.
        """
        # Apply repetition penalty over an adaptive recent window
        # Pre-compute window + token counts once for all penalty types
        _needs_rep = repetition_penalty != 1.0
        _needs_additive = frequency_penalty != 0.0 or presence_penalty != 0.0
        if _needs_rep or _needs_additive:
            vocab_size = logits.shape[-1]
            window_size = self._adaptive_rep_window(generated.shape[-1])
            window = generated[0, -window_size:]
            token_ids = window.clamp(0, vocab_size - 1)
            token_counts = torch.bincount(token_ids, minlength=vocab_size)

        if _needs_rep:
            appeared_mask = token_counts > 0
            # Exclude exempt tokens from penalty
            if exempt_tokens:
                for tid in exempt_tokens:
                    if 0 <= tid < vocab_size:
                        appeared_mask[tid] = False
            # Apply penalty: divide positive logits, multiply negative logits
            scores = logits[0, appeared_mask]
            logits[0, appeared_mask] = torch.where(
                scores > 0, scores / repetition_penalty,
                scores * repetition_penalty)

        # OpenAI-style additive penalties (R7)
        if _needs_additive:
            counts_f = token_counts.float()
            if frequency_penalty != 0.0:
                logits[0] -= frequency_penalty * counts_f
            if presence_penalty != 0.0:
                logits[0] -= presence_penalty * (counts_f > 0).float()

        # T3-9: JSON schema constraint mask (before temperature/sampling)
        if json_constraint is not None:
            logits = json_constraint.mask_logits(logits)

        # Temperature scaling
        logits = logits / max(temperature, 1e-8)

        # Save pre-filter logits for NaN fallback (S720)
        pre_filter_logits = logits.clone()

        # Min-p filtering: remove tokens below min_p * max_probability
        if min_p > 0.0:
            probs_for_filter = F.softmax(logits, dim=-1)
            max_prob = probs_for_filter.max(dim=-1, keepdim=True).values
            logits = logits.masked_fill(
                probs_for_filter < min_p * max_prob, float('-inf'))

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k)
            min_value = values[:, -1, None]
            logits = torch.where(logits < min_value, float('-inf'), logits)

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False

            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # Typical sampling (R10, Meister et al.)
        # Keep tokens whose information content (-log p) is close to
        # the entropy of the full distribution.
        if typical_p > 0.0:
            probs_t = F.softmax(logits, dim=-1)
            log_probs = torch.log(probs_t + 1e-10)
            neg_entropy = (probs_t * log_probs).sum(dim=-1, keepdim=True)
            # Shift = |surprise - entropy|
            shifted = torch.abs(-log_probs - neg_entropy)
            sorted_shifted, sorted_idx = torch.sort(shifted, dim=-1)
            sorted_probs_t = probs_t.gather(-1, sorted_idx)
            cum_probs = torch.cumsum(sorted_probs_t, dim=-1)
            keep = cum_probs <= typical_p
            keep[:, 0] = True  # always keep at least one
            # Scatter back
            remove = torch.ones_like(logits, dtype=torch.bool)
            remove.scatter_(-1, sorted_idx, ~keep)
            logits = logits.masked_fill(remove, float('-inf'))

        # Mirostat v2 sampling (R11, Basu et al.)
        # Dynamically adjusts top-k to target a specific perplexity.
        if mirostat_mode == 2:
            probs_m = F.softmax(logits, dim=-1)
            sorted_probs_m, sorted_idx = torch.sort(probs_m, descending=True, dim=-1)
            surprisals = -torch.log2(sorted_probs_m + 1e-10)
            # Find the cutoff where surprise exceeds 2*tau
            exceed = surprisals > 2 * mirostat_tau
            if exceed.any():
                cutoff = exceed.float().argmax(dim=-1).item()
                cutoff = max(cutoff, 1)
            else:
                cutoff = sorted_probs_m.shape[-1]
            # Zero out everything beyond cutoff
            remove_idx = sorted_idx[:, cutoff:]
            if remove_idx.numel() > 0:
                logits.scatter_(-1, remove_idx, float('-inf'))

        # Sample — single final softmax
        probs = F.softmax(logits, dim=-1)
        # Guard: if all logits were -inf, softmax produces NaN.
        # Fall back to pre-filter distribution (S720).
        if torch.isnan(probs).any():
            probs = F.softmax(pre_filter_logits, dim=-1)
            # Second-level guard: if pre-filter logits were also
            # all -inf (model produced garbage), uniform sample.
            if torch.isnan(probs).any():
                probs = torch.ones_like(probs) / probs.shape[-1]
        return torch.multinomial(probs, num_samples=1)

    def _sample_token_batch(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        min_p: float = 0.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        exempt_tokens: set[int] | None = None,
    ) -> torch.Tensor:
        """Sample next tokens for an entire batch in one vectorized pass.

        Args:
            logits: Shape ``[batch_size, vocab_size]`` — last-position logits.
            generated: Shape ``[batch_size, seq_len]`` — tokens generated so far.
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Nucleus (top-p) filtering.
            repetition_penalty: Penalise repeated tokens.
            frequency_penalty: Additive penalty proportional to count.
            presence_penalty: Additive penalty for any token seen at all.
            exempt_tokens: Token IDs exempt from repetition penalty
                (prompt tokens, special tokens, proper nouns).

        Returns:
            ``[batch_size, 1]`` tensor of sampled token IDs.
        """
        batch_size, vocab_size = logits.shape

        # Pre-compute window + counts once for all penalty types
        _needs_rep = repetition_penalty != 1.0
        _needs_additive = frequency_penalty != 0.0 or presence_penalty != 0.0
        if _needs_rep or _needs_additive:
            window_size = self._adaptive_rep_window(generated.shape[-1])
            window = generated[:, -window_size:]
            gen_clamped = window.clamp(0, vocab_size - 1)
            counts = torch.zeros_like(logits)
            counts.scatter_add_(
                1, gen_clamped,
                torch.ones_like(gen_clamped, dtype=logits.dtype))

        # Repetition penalty — windowed, vectorised via scatter_add
        if _needs_rep:
            appeared = counts > 0
            # Exclude exempt tokens from penalty (S784)
            if exempt_tokens:
                for tid in exempt_tokens:
                    if 0 <= tid < vocab_size:
                        appeared[:, tid] = False
            scores = logits[appeared]
            logits[appeared] = torch.where(
                scores > 0, scores / repetition_penalty,
                scores * repetition_penalty)

        # OpenAI-style additive penalties (R7, batched)
        if _needs_additive:
            if frequency_penalty != 0.0:
                logits -= frequency_penalty * counts
            if presence_penalty != 0.0:
                logits -= presence_penalty * (counts > 0).float()

        # Temperature
        logits = logits / max(temperature, 1e-8)

        # Min-p filtering
        if min_p > 0.0:
            probs_for_filter = F.softmax(logits, dim=-1)
            max_prob = probs_for_filter.max(dim=-1, keepdim=True).values
            logits = logits.masked_fill(
                probs_for_filter < min_p * max_prob, float('-inf'))

        # Top-k
        if top_k > 0:
            top_k_clamped = min(top_k, vocab_size)
            values, _ = torch.topk(logits, top_k_clamped, dim=-1)
            min_values = values[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_values,
                torch.tensor(float('-inf'), device=logits.device),
                logits,
            )

        # Top-p (nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                logits, descending=True, dim=-1)
            cum_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_remove = cum_probs > top_p
            sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
            sorted_remove[:, 0] = False
            remove = torch.zeros_like(logits, dtype=torch.bool)
            remove.scatter_(1, sorted_indices, sorted_remove)
            logits = logits.masked_fill(remove, float('-inf'))

        # Sample
        probs = F.softmax(logits, dim=-1)
        # Guard: if all logits were -inf, softmax produces NaN.
        # Fall back to the single highest-logit token per row.
        if torch.isnan(probs).any():
            probs = torch.zeros_like(probs)
            probs.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return torch.multinomial(probs, num_samples=1)

    # =========================================================================
    # Streaming Generation
    # =========================================================================

    def _stream_round_tokens(
        self,
        input_ids: torch.Tensor,
        max_gen: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        min_p: float,
        *,
        json_constraint: object | None,
        stop_on_close: bool,
        state: dict,
    ) -> Generator[str]:
        """B-3d (Pass 156z9al): inner streaming round.  Yields token
        strings as they're produced.  Updates ``state`` dict so the
        caller can decide whether to splice + start another round.

        State keys written:

        * ``emitted_count``: number of tokens this round produced.
        * ``emitted_text``: concatenation of all yielded chunks.
        * ``terminated_on``: one of ``"max"``, ``"eos"``, ``"search"``,
          ``"json_done"``.

        ``stop_on_close=True`` causes the loop to exit early as soon as
        ``</search>`` appears in the joined emitted text — used by
        rounds 1..N-1 of the splice loop.  Final round and
        non-splice-enabled streams pass ``stop_on_close=False``.

        Caller MUST hold ``self._generation_lock`` and be inside a
        ``torch.no_grad()`` block — this helper does NOT acquire either.
        Caller is responsible for clearing / repopulating the KV cache
        between rounds (this helper clears + prefills on entry).
        """
        state["emitted_count"] = 0
        state["emitted_text"] = ""
        state["terminated_on"] = "max"

        max_len = self.model.config.max_seq_len
        has_cache = hasattr(self.model, 'clear_cache')
        if has_cache:
            self.model.clear_cache()
        else:
            logger.warning(
                "Model lacks KV cache support — generation will use "
                "O(n²) full-recompute fallback (slow for long "
                "sequences)."
            )

        generated = input_ids
        curr_input = input_ids
        if curr_input.shape[1] > max_len:
            curr_input = curr_input[:, -max_len:]
        logits = self.model(curr_input, use_cache=has_cache)

        exempt_tokens = self._build_exempt_tokens(
            input_ids, repetition_penalty,
        )

        eos_id = getattr(self.tokenizer, 'eos_token_id', 2)

        for _ in range(max_gen):
            # Pass B (Pass 156z9ff): honour user stop signal.
            if self._check_cancel():
                if state is not None:
                    state["terminated_on"] = "cancel"
                logger.info("Stream generation cancelled by user")
                break
            next_token = self._sample_token(
                logits[:, -1, :],
                generated,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                min_p,
                exempt_tokens=exempt_tokens,
                json_constraint=json_constraint,
            )

            generated = torch.cat([generated, next_token], dim=1)

            token_id = next_token[0, 0].item()

            if hasattr(self.tokenizer, 'decode'):
                token_str = self.tokenizer.decode(
                    [token_id], skip_special_tokens=True)
            else:
                token_str = self.tokenizer.id_to_token.get(token_id, "")

            state["emitted_count"] += 1
            state["emitted_text"] += token_str
            yield token_str

            if token_id == eos_id:
                state["terminated_on"] = "eos"
                return

            if json_constraint is not None:
                json_constraint.advance(int(token_id))
                if json_constraint.is_done:
                    state["terminated_on"] = "json_done"
                    return

            # B-3d: rounds 1..N-1 stop early on </search> so the
            # orchestrator can run RAG + splice + start the next round.
            if stop_on_close and "</search>" in state["emitted_text"]:
                state["terminated_on"] = "search"
                return

            if has_cache:
                logits = self.model(
                    next_token,
                    use_cache=True,
                    start_pos=generated.shape[1] - 1,
                )
            else:
                curr_input = generated
                if curr_input.shape[1] > max_len:
                    curr_input = curr_input[:, -max_len:]
                logits = self.model(curr_input)

    def stream_generate(
        self,
        prompt: str,
        max_gen: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        min_p: float = 0.0,
        max_tokens: int | None = None,  # Alias for max_gen (backward compatibility)
        max_new_tokens: int | None = None,  # Alias for max_gen (Forge model compatibility)
        max_length: int | None = None,  # Alias for max_gen (common parameter name)
        json_schema: dict | None = None,
    ) -> Generator[str]:
        """
        Stream generated tokens one at a time.

        Args:
            prompt: Input text to continue
            max_gen: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty
            max_tokens: Alias for max_gen (backward compatibility)
            max_new_tokens: Alias for max_gen (Forge model compatibility)
            max_length: Alias for max_gen (common parameter name)
            json_schema: Optional JSON schema dict. When set, masks logits
                each step so only schema-conforming tokens are emitted, and
                stops yielding once the FSM reaches DONE. Mirrors the
                non-streaming path in ``_generate_text``. NOT supported on
                GGUF — caller (``stream_chat``) raises NotImplementedError
                up the stack before reaching here.

        Yields:
            Each newly generated token as it's produced
        Raises:
            ValueError: when more than one of ``max_tokens`` /
                ``max_new_tokens`` / ``max_length`` is set.  Pass
                156z9fe (Pass A fix #3): the previous code overwrote
                them sequentially so the last-checked alias won
                silently, hiding caller mistakes (e.g. a
                HuggingFace-style ``max_new_tokens=100`` plus a
                native ``max_tokens=200`` would silently honour
                ``max_length``).
        """
        # Aliases for backward / cross-API compatibility.  At most one
        # of max_tokens / max_new_tokens / max_length may be set.
        _aliases = [
            ("max_tokens", max_tokens),
            ("max_new_tokens", max_new_tokens),
            ("max_length", max_length),
        ]
        _set_aliases = [(n, v) for n, v in _aliases if v is not None]
        if len(_set_aliases) > 1:
            raise ValueError(
                "Conflicting max-length aliases set: "
                f"{[n for n, _ in _set_aliases]}. Pass only one of "
                "max_gen / max_tokens / max_new_tokens / max_length."
            )
        if _set_aliases:
            max_gen = _set_aliases[0][1]

        # N-15c: build JSON schema constraint once per call (vocab scan
        # amortised across all tokens, same discipline as _generate_text).
        json_constraint = None
        if json_schema is not None:
            from .json_schema_mask import JsonSchemaConstraint
            json_constraint = JsonSchemaConstraint(json_schema, self.tokenizer)

        input_ids = self._encode_prompt(prompt)
        # Stage B-2 (Pass 156z9d): accumulate yielded token strings so
        # we can scan for ``<search>`` emissions on stream completion
        # (or generator cancellation via try/finally).  Same observability
        # contract as :meth:`_generate_text` non-streaming path.
        # B-3d (Pass 156z9al) extends this with multi-round splice
        # orchestration when ``inline_search_splice_enabled`` is True
        # and a built RAG index is attached.  Splice block strings
        # yielded mid-stream are also appended to ``full_emitted_text``
        # so the tail observability scan sees them.
        full_emitted_text = ""

        # B-3d: gate splice on the same preconditions as the
        # non-streaming helper (`_maybe_rag_splice`).
        rag_index = getattr(self, "_rag_index", None)
        splice_enabled = (
            getattr(self, "inline_search_splice_enabled", False)
            and rag_index is not None
            and getattr(rag_index, "is_built", False)
        )
        max_rounds = (
            max(1, int(getattr(self, "max_search_rounds", 3)))
            if splice_enabled else 1
        )
        cumulative_tokens = 0
        current_prompt = prompt
        last_terminated_on = "max"
        last_emitted_text = ""

        # Acquire generation lock to protect KV-cache state
        with self._generation_lock, torch.no_grad():
            try:
                for round_idx in range(max_rounds):
                    is_final_round = (round_idx == max_rounds - 1)
                    stop_on_close = splice_enabled and not is_final_round
                    remaining = max_gen - cumulative_tokens
                    if remaining <= 0:
                        logger.info(
                            "B-3d: max_gen=%d budget exhausted before "
                            "round %d (cumulative=%d); ending stream",
                            max_gen, round_idx + 1, cumulative_tokens)
                        break

                    if round_idx == 0:
                        round_input_ids = input_ids
                    else:
                        round_input_ids = self._encode_prompt(current_prompt)

                    state: dict = {}
                    for tok_str in self._stream_round_tokens(
                        round_input_ids, remaining, temperature, top_k,
                        top_p, repetition_penalty, min_p,
                        json_constraint=json_constraint,
                        stop_on_close=stop_on_close,
                        state=state,
                    ):
                        full_emitted_text += tok_str
                        yield tok_str

                    cumulative_tokens += int(state.get("emitted_count", 0))
                    last_terminated_on = state.get("terminated_on", "max")
                    last_emitted_text = state.get("emitted_text", "")

                    if last_terminated_on != "search":
                        # natural stop (eos / max / json_done) — no
                        # splice path.  WARNING if final-round plain-text
                        # ``<search>`` slipped through (mirrors B-3c).
                        if (is_final_round and splice_enabled
                                and "<search>" in last_emitted_text):
                            logger.warning(
                                "B-3d: max_search_rounds=%d budget "
                                "exhausted but model emitted another "
                                "<search> tag; left as plain text",
                                max_rounds)
                        break

                    # Splice path: extract query, retrieve, yield block.
                    open_pos = last_emitted_text.rfind("<search>")
                    close_pos = last_emitted_text.rfind("</search>")
                    if open_pos < 0 or open_pos > close_pos:
                        # Malformed pair (close before open / no open):
                        # exit splice loop.
                        break
                    query = last_emitted_text[
                        open_pos + len("<search>"):close_pos
                    ].strip()
                    if not query:
                        break
                    try:
                        from .rag import RAGIndex
                        results = rag_index.query(query, top_k=5)
                        ctx = RAGIndex.format_context(
                            results, max_chars=2000)
                    except Exception:
                        logger.exception(
                            "B-3d: RAG retrieval failed at round %d; "
                            "ending stream",
                            round_idx + 1)
                        break
                    if not ctx:
                        break

                    splice_block = (
                        f"\n<search_result>\n{ctx}\n"
                        f"</search_result>\n"
                    )
                    logger.info(
                        "B-3d: round %d/%d spliced RAG result for "
                        "query '%s' (%d chars ctx)",
                        round_idx + 1, max_rounds, query[:60], len(ctx))
                    full_emitted_text += splice_block
                    yield splice_block

                    # Build prompt for next round: original prompt
                    # extended with this round's emit (up to and
                    # including the closing tag) plus the splice block.
                    emit_through_close = last_emitted_text[
                        :close_pos + len("</search>")
                    ]
                    current_prompt = (
                        current_prompt + emit_through_close + splice_block
                    )
            finally:
                # Stage B-2 / B-3d tail scan: full_emitted_text already
                # includes every yielded chunk (model tokens + splice
                # blocks).  Splice blocks contain ``<search_result>`` not
                # ``<search>`` so the recorder does NOT spuriously log
                # them as model-emitted queries.  Runs on normal
                # completion AND on generator cancellation (caller
                # breaks early).
                try:
                    self._record_search_emissions(
                        full_emitted_text, path="stream")
                except Exception:
                    logger.exception(
                        "Stage B-2: stream scan crashed; "
                        "last_search_queries left at previous value")

    # =========================================================================
    # Batch Generation
    # =========================================================================

    def batch_generate(
        self,
        prompts: list[str],
        max_gen: int = 100,
        **kwargs
    ) -> list[str]:
        """
        Generate text for multiple prompts in a single batched forward pass.

        Args:
            prompts: List of input prompts
            max_gen: Maximum tokens to generate per prompt
            **kwargs: Additional generation parameters (temperature, top_k, top_p, repetition_penalty)

        Returns:
            List of generated texts

        Raises:
            NotImplementedError: when ``json_schema`` is passed via
                ``**kwargs``.  The batched sampler
                (``_sample_token_batch``) does not accept a
                ``json_constraint`` parameter and the FSM is
                single-sequence by design, so a batched call cannot
                guarantee schema-conforming output across rows.
                Silent drop would let callers receive unconstrained
                batched output labelled as schema-conforming.  Pass
                156z9fe (Pass A fix #2): fourth sibling-boundary site
                in the json_schema family (after GGUF chat, stream
                chat, and vision — Pass 156z7 closed those three).
                Drop the schema or call ``generate()`` per prompt.
        """
        if kwargs.get("json_schema") is not None:
            raise NotImplementedError(
                "json_schema constrained decoding is not supported "
                "on the batch_generate path. The batched sampler "
                "shares one FSM across rows which would not produce "
                "schema-conforming output. Drop the schema or call "
                "generate() per prompt."
            )

        if not prompts:
            return []

        # If only one prompt, use regular generate
        if len(prompts) == 1:
            return [self.generate(prompts[0], max_gen=max_gen, **kwargs)]

        # Extract generation parameters
        temperature = kwargs.get('temperature', 0.8)
        top_k = kwargs.get('top_k', 50)
        top_p = kwargs.get('top_p', 0.9)
        repetition_penalty = kwargs.get('repetition_penalty', 1.1)
        min_p = kwargs.get('min_p', 0.0)

        # Encode all prompts
        if hasattr(self.tokenizer, 'encode'):
            encoded = [self.tokenizer.encode(p) for p in prompts]
        else:
            encoded = [[self.tokenizer.token_to_id.get(t, 3) for t in p] for p in prompts]

        # Pad all sequences to the same length
        max_input_len = max(len(e) for e in encoded)
        pad_id = getattr(self.tokenizer, 'pad_token_id', 0)

        # Create padded batch tensor
        batch_size = len(prompts)
        input_ids = torch.full(
            (batch_size, max_input_len),
            pad_id,
            dtype=torch.long,
            device=self.device
        )

        # Fill in the actual tokens
        for i, tokens in enumerate(encoded):
            input_ids[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)

        # Track which sequences are still generating
        eos_id = getattr(self.tokenizer, 'eos_token_id', 2)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # Generate tokens autoregressively — hold lock to protect model state
        generated = input_ids
        all_finished = False

        # S784: Build exempt_tokens for batch rep-penalty.  Use first row
        # (all rows share the same special / proper-noun tokens; per-row
        # prompt tokens vary but the penalty window handles that).
        exempt_tokens = self._build_exempt_tokens(
            input_ids[:1], repetition_penalty,
        )

        with self._generation_lock:
            for step in range(max_gen):
                # Pass B (Pass 156z9ff): honour user stop signal.
                if self._check_cancel():
                    logger.info("Batch generation cancelled by user at step %d", step)
                    break
                # Early exit if all sequences finished (check every 5 steps starting from step 5 to reduce overhead)
                if all_finished or (step >= 5 and step % 5 == 0 and finished.all()):
                    all_finished = True
                    break

                # Truncate if needed
                curr_input = generated
                max_len = self.model.config.max_seq_len
                if curr_input.shape[1] > max_len:
                    curr_input = curr_input[:, -max_len:]

                # Forward pass for entire batch
                with torch.no_grad():
                    logits = self.model(curr_input)

                # Extract last-position logits: [batch_size, vocab_size]
                last_logits = logits[:, -1, :].clone()

                # Force pad distribution for finished sequences
                last_logits[finished] = float('-inf')
                last_logits[finished, pad_id] = 0.0

                # Sample entire batch at once (vectorized)
                next_tokens_tensor = self._sample_token_batch(
                    last_logits, generated,
                    temperature, top_k, top_p, repetition_penalty, min_p,
                    exempt_tokens=exempt_tokens,
                )  # [batch_size, 1]

                # Check EOS for unfinished sequences
                newly_done = (next_tokens_tensor.squeeze(-1) == eos_id) & ~finished
                finished = finished | newly_done

                # Append next tokens
                generated = torch.cat([generated, next_tokens_tensor], dim=1)

        # Decode all outputs
        results = []
        for i in range(batch_size):
            # Handle case where generated[i] might not be a tensor
            try:
                ids = generated[i].cpu().tolist()
            except AttributeError:
                if isinstance(generated[i], str):
                    results.append(generated[i])
                    continue
                ids = list(generated[i]) if hasattr(generated[i], '__iter__') else [generated[i]]

            if hasattr(self.tokenizer, 'decode'):
                text = self.tokenizer.decode(ids, skip_special_tokens=True)
            else:
                # Fallback
                text = "".join(
                    self.tokenizer.id_to_token.get(idx, "?")
                    for idx in ids
                )

            results.append(text)

        # Stage B-2b: per-prompt attribution.  Earlier slice (B-2)
        # joined results and scanned once, which lost which prompt
        # produced which query.  Now we scan each result independently,
        # populate ``last_search_queries_per_prompt`` (parallel to
        # ``prompts``), and set the flat ``last_search_queries`` to
        # the union so single-prompt callers still see everything.
        # Errors in one prompt's scan do NOT corrupt other prompts'
        # results — each call to ``_record_search_emissions`` is
        # already exception-safe.
        #
        # B-3 sibling closure (Pass 156z9do): when
        # ``inline_search_splice_enabled`` is True, per-prompt trim at
        # ``</search>`` and then call ``_maybe_rag_splice`` on each
        # output independently so multi-round retrieval + continuation
        # works for batched calls too.  Continuation rounds run through
        # ``_generate_manual`` (single-sequence text-only), so the
        # batch efficiency advantage applies only to round 0 — splice
        # rounds are serial per prompt.  Acceptable trade-off because
        # most batched calls won't trigger splice on every row.
        splice_enabled = getattr(self, "inline_search_splice_enabled", False)
        effective_stop_strings_batch = (
            ["</search>"] if splice_enabled else None
        )
        per_prompt: list[list[str]] = []
        flat: list[str] = []
        for i, (prompt_text, output_text) in enumerate(zip(prompts, results)):
            current_text = output_text
            if splice_enabled and isinstance(current_text, str):
                # Per-sequence post-decode trim at ``</search>`` so the
                # splice helper sees an unclosed ``<search>q`` tail.
                pl = len(prompt_text)
                generated_part = (
                    current_text[pl:] if len(current_text) > pl else current_text
                )
                if "</search>" in generated_part:
                    stop_pos = generated_part.find("</search>")
                    current_text = current_text[:pl + stop_pos]

                # Round-0 token count for this sequence: generated total
                # minus the original (unpadded) input length for row ``i``.
                tokens_round0 = max(
                    0, generated.shape[1] - len(encoded[i])
                )
                spliced = self._maybe_rag_splice(
                    current_text, prompt_text, max_gen,
                    temperature, top_k, top_p,
                    repetition_penalty, min_p,
                    effective_stop_strings=effective_stop_strings_batch,
                    json_constraint=None,
                    tokens_already_generated=tokens_round0,
                )
                if spliced is not None:
                    current_text = spliced

            results[i] = current_text
            self._record_search_emissions(
                current_text, prompt=prompt_text, path="batch")
            per_prompt.append(list(self.last_search_queries))
            flat.extend(self.last_search_queries)
        self.last_search_queries_per_prompt = per_prompt
        self.last_search_queries = flat
        return results

    # =========================================================================
    # 🖼️ Multimodal (Vision) Generation
    # =========================================================================

    def _generate_with_vision(
        self,
        prompt: str,
        vision_features: torch.Tensor,
        max_gen: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_strings: list[str] | None = None,
        min_p: float = 0.0,
        **kwargs,
    ) -> str:
        """Generate text conditioned on both a text prompt and vision features.

        Uses the shared ``_sample_token`` helper (same windowed repetition
        penalty, min-p filtering, and sampling logic as every other path)
        and the KV cache for O(1)-per-token decoding after the prefill.

        Args:
            prompt: Text prompt (already formatted by the chat builder).
            vision_features: Pre-encoded vision features with shape
                ``[1, vision_seq_len, vision_dim]``.
            max_gen: Maximum new tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Nucleus (top-p) sampling threshold.
            repetition_penalty: Penalty for repeated tokens.
            stop_strings: Strings that terminate generation.
            min_p: Min-p filtering threshold.
            **kwargs: Ignored (absorbs extra chat kwargs) EXCEPT
                ``json_schema`` — see below.

        Raises:
            NotImplementedError: when ``json_schema`` is passed. The
                vision generation path samples without going through
                ``_generate_text``/``_generate_manual``, so the
                constraint FSM never gets wired in. Silent drop would
                let multimodal callers receive unconstrained output
                labelled as schema-conforming. Pass 156z7 (N-15c2)
                sibling-boundary site missed by Pass 156z6.

        Returns:
            The full generated text (prompt + new tokens).
        """
        if kwargs.get("json_schema") is not None:
            raise NotImplementedError(
                "json_schema constrained decoding is not supported on "
                "the vision (multimodal) generation path. Drop the "
                "schema or use a text-only prompt."
            )
        if not isinstance(prompt, str) or not prompt.strip():
            return ""

        # B-3a sibling closure (Pass 156z9do): when the splice flag is
        # ON, append ``</search>`` to ``stop_strings`` so the vision
        # loop halts cleanly on the closing tag, mirroring
        # ``_generate_text`` and the speculative siblings.  Continuation
        # rounds run through ``_maybe_rag_splice`` → ``_generate_manual``
        # which is a text-only path: image grounding is lost on splice
        # rounds (the cache is rebuilt from the spliced text prompt
        # without a fresh ``forward_multimodal`` prefill).  This is an
        # accepted degradation — the splice exists to inject retrieved
        # text knowledge, and the image content the model needed was
        # already in the emission up to ``</search>``.
        effective_stop_strings = stop_strings
        if getattr(self, "inline_search_splice_enabled", False):
            effective_stop_strings = list(stop_strings or [])
            if "</search>" not in effective_stop_strings:
                effective_stop_strings.append("</search>")

        # Tokenise prompt
        input_ids = self._encode_prompt(prompt)  # [1, seq_len]
        device = getattr(self, "device", torch.device("cpu"))
        input_ids = input_ids.to(device)
        vision_features = vision_features.to(device)

        generated = input_ids  # [1, seq_len] — running token buffer
        prompt_len = input_ids.shape[1]

        # Determine EOS token
        eos_id: int | None = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is None:
            eos_id = getattr(self.tokenizer, "eos_id", None)

        has_cache = hasattr(self.model, 'clear_cache')

        # S783: Build exempt_tokens for rep-penalty (same as other paths)
        exempt_tokens = self._build_exempt_tokens(
            input_ids, repetition_penalty,
        )

        with self._generation_lock, torch.no_grad():
            if has_cache:
                self.model.clear_cache()
            # Prefill: vision + full text prompt (populates KV cache)
            logits = self.model.forward_multimodal(
                input_ids=input_ids,
                vision_features=vision_features,
                use_cache=has_cache,
            )

            for step in range(max_gen):
                # Pass B (Pass 156z9ff): honour user stop signal.
                if self._check_cancel():
                    logger.info("Vision generation cancelled by user at step %d", step)
                    break
                # Shared sampling (windowed penalty, min_p, top-k/p)
                next_token = self._sample_token(
                    logits[:, -1, :],
                    generated,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                    min_p,
                    exempt_tokens=exempt_tokens,
                )

                generated = torch.cat([generated, next_token], dim=1)
                token_id = next_token[0, 0].item()

                # EOS check
                if eos_id is not None and token_id == eos_id:
                    break

                # Periodic stop-string check (windowed decode)
                if (effective_stop_strings
                        and (step + 1) % self._STOP_CHECK_INTERVAL == 0):
                    tail_start = max(prompt_len, generated.shape[1] - self._STOP_CHECK_WINDOW)
                    recent_ids = generated[0, tail_start:].tolist()
                    recent_text = self.tokenizer.decode(
                        recent_ids, skip_special_tokens=True)
                    if any(ss in recent_text for ss in effective_stop_strings):
                        break

                # Decode step: only new token with start_pos
                if has_cache:
                    logits = self.model(
                        next_token,
                        use_cache=True,
                        start_pos=generated.shape[1] - 1,
                    )
                else:
                    logits = self.model(
                        torch.tensor([[token_id]], dtype=torch.long, device=device)
                    )

        # Decode
        generated_ids = generated.squeeze(0).tolist()
        if hasattr(self.tokenizer, "decode"):
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        else:
            text = "".join(
                self.tokenizer.id_to_token.get(idx, "?")
                for idx in generated_ids
            )

        # Apply stop strings
        if effective_stop_strings:
            prompt_text_len = len(prompt)
            generated_part = text[prompt_text_len:] if len(text) > prompt_text_len else text
            for stop_str in effective_stop_strings:
                if stop_str in generated_part:
                    stop_pos = generated_part.find(stop_str)
                    text = text[:prompt_text_len + stop_pos]
                    break

        # B-3b/B-3c sibling closure (Pass 156z9do): same retrieve+splice
        # contract as ``_generate_text`` / speculative siblings.  When
        # the splice flag is ON and a built RAG index is attached, an
        # unclosed ``<search>q`` tail triggers up to ``max_search_rounds``
        # rounds of retrieve and continuation via ``_maybe_rag_splice``.
        # Helper returns ``None`` on any precondition miss, leaving
        # ``text`` unchanged.  ``tokens_already_generated`` is the
        # round-0 emission count so the helper budgets remaining rounds
        # against the original ``max_gen`` instead of multiplying it.
        # NOTE: continuation rounds are text-only (no vision features
        # re-injected) — see the docstring on the stop-string augment
        # block above for the rationale.
        tokens_round0 = max(0, generated.shape[1] - prompt_len)
        spliced = self._maybe_rag_splice(
            text, prompt, max_gen, temperature, top_k, top_p,
            repetition_penalty, min_p,
            effective_stop_strings=effective_stop_strings,
            json_constraint=None,
            tokens_already_generated=tokens_round0,
        )
        if spliced is not None:
            text = spliced

        # Stage B-2 sibling sweep: vision path returns final text here.
        # Pass 156z9e: pass prompt= so user prompts containing literal
        # ``<search>foo</search>`` aren't falsely recorded as emissions.
        self._record_search_emissions(text, prompt=prompt, path="vision")
        return text

    # =========================================================================
    # R23 + R24 — Speculative Decoding with Adaptive K
    # =========================================================================

    def speculative_generate(
        self,
        prompt: str,
        draft_model: "torch.nn.Module",
        max_gen: int = 256,
        initial_k: int = 5,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        min_p: float = 0.0,
        stop_strings: list[str] | None = None,
        adaptive: bool = True,
        min_k: int = 1,
        max_k: int = 12,
    ) -> str:
        """Generate text using speculative decoding (R23/R24).

        A smaller ``draft_model`` proposes ``K`` tokens, then the full
        ``self.model`` (the "verifier") scores them in a single forward
        pass.  Accepted draft tokens are kept; on the first rejection
        the verifier's own sample replaces the draft token and
        generation continues.

        This produces the **exact same distribution** as standard
        sampling from the verifier model (via rejection sampling),
        while typically running 2-3x faster when the draft model is
        well-aligned with the verifier.

        R24 — Adaptive speculation length:
        When ``adaptive=True``, ``K`` is adjusted at runtime based on
        the recent acceptance rate.  High acceptance -> increase K
        (more aggressive speculation).  Low acceptance -> decrease K
        (save wasted draft compute).

        Args:
            prompt: Input text.
            draft_model: Smaller/faster model for draft proposals.
            max_gen: Maximum new tokens.
            initial_k: Starting speculation length.
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Nucleus sampling threshold.
            repetition_penalty: Repetition penalty.
            min_p: Min-p filtering.
            stop_strings: Early-stop strings.
            adaptive: Enable adaptive K (R24).
            min_k: Minimum speculation length.
            max_k: Maximum speculation length.

        Returns:
            Generated text (prompt + new tokens).
        """
        # B-3a sibling closure (Pass 156z9cp): when the splice flag is
        # ON, append ``</search>`` to ``stop_strings`` so the
        # speculative loop halts cleanly on the closing tag, mirroring
        # ``_generate_text``'s ``effective_stop_strings`` pattern.  The
        # post-decode trim already iterates ``stop_strings`` so no
        # additional trim wiring is needed.
        effective_stop_strings = stop_strings
        if getattr(self, "inline_search_splice_enabled", False):
            effective_stop_strings = list(stop_strings or [])
            if "</search>" not in effective_stop_strings:
                effective_stop_strings.append("</search>")

        input_ids = self._encode_prompt(prompt)
        device = input_ids.device
        generated = input_ids
        prompt_len = input_ids.shape[1]
        eos_id = getattr(self.tokenizer, 'eos_token_id',
                         getattr(self.tokenizer, 'eos_id', 2))

        k = initial_k
        accept_history: list[float] = []
        tokens_generated = 0

        has_cache_main = hasattr(self.model, 'clear_cache')
        has_cache_draft = hasattr(draft_model, 'clear_cache')

        with self._generation_lock, torch.no_grad():
            if has_cache_main:
                self.model.clear_cache()
            if has_cache_draft:
                draft_model.clear_cache()

            # Prefill both models
            main_logits = self.model(input_ids, use_cache=has_cache_main)
            draft_model(input_ids, use_cache=has_cache_draft)

            while tokens_generated < max_gen:
                # Pass B (Pass 156z9ff): honour user stop signal.
                if self._check_cancel():
                    logger.info(
                        "Speculative generation cancelled by user (%d tokens generated)",
                        tokens_generated,
                    )
                    break
                # --- Draft phase: generate K tokens from draft model ---
                draft_tokens: list[int] = []
                draft_probs_list: list[torch.Tensor] = []
                draft_input = generated

                for _di in range(k):
                    if has_cache_draft:
                        if _di == 0 and tokens_generated == 0:
                            d_logits = main_logits
                        else:
                            last_tok = torch.tensor(
                                [[draft_tokens[-1] if draft_tokens
                                  else generated[0, -1].item()]],
                                dtype=torch.long, device=device)
                            d_logits = draft_model(
                                last_tok, use_cache=True,
                                start_pos=draft_input.shape[1] + len(draft_tokens) - 1)
                    else:
                        seq = torch.cat([
                            draft_input,
                            torch.tensor([draft_tokens], dtype=torch.long,
                                         device=device)
                        ], dim=1) if draft_tokens else draft_input
                        d_logits = draft_model(seq)

                    d_logits_last = d_logits[:, -1, :] / max(temperature, 1e-8)
                    d_probs = F.softmax(d_logits_last, dim=-1)
                    draft_token = torch.multinomial(d_probs, 1)[0, 0].item()
                    draft_tokens.append(draft_token)
                    draft_probs_list.append(d_probs[0])

                    if draft_token == eos_id:
                        break

                if not draft_tokens:
                    break

                # --- Verify phase: score all draft tokens in one pass ---
                draft_tensor = torch.tensor(
                    [draft_tokens], dtype=torch.long, device=device)

                if has_cache_main:
                    verify_logits = self.model(
                        draft_tensor, use_cache=True,
                        start_pos=generated.shape[1])
                else:
                    full_seq = torch.cat([generated, draft_tensor], dim=1)
                    verify_logits = self.model(full_seq)
                    verify_logits = verify_logits[:, -len(draft_tokens):, :]

                # --- Rejection sampling ---
                accepted = 0
                for j in range(len(draft_tokens)):
                    if j == 0:
                        v_logits = main_logits[:, -1, :] / max(temperature, 1e-8)
                    else:
                        v_logits = verify_logits[:, j - 1, :] / max(temperature, 1e-8)
                    v_probs = F.softmax(v_logits, dim=-1)

                    draft_id = draft_tokens[j]
                    p_draft = draft_probs_list[j][draft_id].item()
                    p_verify = v_probs[0, draft_id].item()

                    if p_draft > 0:
                        accept_prob = min(1.0, p_verify / p_draft)
                    else:
                        accept_prob = 1.0

                    if torch.rand(1).item() < accept_prob:
                        accepted += 1
                        generated = torch.cat([
                            generated,
                            torch.tensor([[draft_id]], dtype=torch.long,
                                         device=device)
                        ], dim=1)
                        tokens_generated += 1

                        if draft_id == eos_id or tokens_generated >= max_gen:
                            break
                    else:
                        adjusted = torch.clamp(
                            v_probs[0] - draft_probs_list[j], min=0)
                        adj_sum = adjusted.sum()
                        if adj_sum > 0:
                            adjusted = adjusted / adj_sum
                            new_token = torch.multinomial(
                                adjusted.unsqueeze(0), 1)[0, 0].item()
                        else:
                            new_token = torch.multinomial(
                                v_probs, 1)[0, 0].item()

                        generated = torch.cat([
                            generated,
                            torch.tensor([[new_token]], dtype=torch.long,
                                         device=device)
                        ], dim=1)
                        tokens_generated += 1
                        break

                # If all accepted, also sample one more from verifier
                if accepted == len(draft_tokens) and tokens_generated < max_gen:
                    # S740: Use _sample_token for consistent filtering
                    bonus_token = self._sample_token(
                        verify_logits[:, -1, :], generated,
                        temperature, top_k, top_p,
                        repetition_penalty, min_p)
                    generated = torch.cat([generated, bonus_token], dim=1)
                    tokens_generated += 1

                # Update main_logits for next iteration
                if has_cache_main:
                    last = generated[:, -1:]
                    main_logits = self.model(
                        last, use_cache=True,
                        start_pos=generated.shape[1] - 1)
                else:
                    main_logits = self.model(generated)

                # Re-sync draft model cache
                if has_cache_draft:
                    rewind_pos = generated.shape[1] - 1
                    if hasattr(draft_model, 'rewind_cache'):
                        draft_model.rewind_cache(rewind_pos)
                        draft_model(
                            generated[:, rewind_pos:],
                            use_cache=True, start_pos=rewind_pos)
                    else:
                        draft_model.clear_cache()
                        draft_model(generated, use_cache=True)

                # --- R24: Adaptive K ---
                if adaptive and len(draft_tokens) > 0:
                    rate = accepted / len(draft_tokens)
                    accept_history.append(rate)
                    window = accept_history[-32:]
                    avg_rate = sum(window) / len(window)
                    if avg_rate > 0.8:
                        k = min(k + 1, max_k)
                    elif avg_rate < 0.4:
                        k = max(k - 1, min_k)

                # EOS / stop check
                if generated[0, -1].item() == eos_id:
                    break
                if (effective_stop_strings
                        and tokens_generated % self._STOP_CHECK_INTERVAL == 0):
                    tail_start = max(prompt_len, generated.shape[1] - self._STOP_CHECK_WINDOW)
                    recent_text = self.tokenizer.decode(
                        generated[0, tail_start:].tolist(),
                        skip_special_tokens=True)
                    if any(ss in recent_text for ss in effective_stop_strings):
                        break

        # Decode
        text = self._decode_output(generated)
        if effective_stop_strings:
            pl = len(prompt)
            gen_part = text[pl:] if len(text) > pl else text
            for ss in effective_stop_strings:
                if ss in gen_part:
                    text = text[:pl + gen_part.find(ss)]
                    break

        # B-3b/B-3c sibling closure (Pass 156z9cp): same retrieve+splice
        # contract as ``_generate_text`` — when the splice flag is ON
        # and a built RAG index is attached, an unclosed ``<search>q``
        # tail triggers up to ``max_search_rounds`` rounds of retrieve
        # and continuation via :meth:`_maybe_rag_splice`.  Helper
        # returns ``None`` on any precondition miss, leaving ``text``
        # unchanged.  ``tokens_already_generated`` is the round-0
        # emission count so the helper budgets remaining rounds against
        # the original ``max_gen`` instead of multiplying it.
        spliced = self._maybe_rag_splice(
            text, prompt, max_gen, temperature, top_k, top_p,
            repetition_penalty, min_p,
            effective_stop_strings=effective_stop_strings,
            json_constraint=None,
            tokens_already_generated=tokens_generated,
        )
        if spliced is not None:
            text = spliced

        # Stage B-2 sibling sweep: speculative decoding return path.
        # Pass 156z9e: prompt-slice so prompt-side <search> isn't recorded.
        self._record_search_emissions(
            text, prompt=prompt, path="speculative")
        return text

    def medusa_generate(
        self,
        prompt: str,
        max_gen: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        min_p: float = 0.0,
        stop_strings: list[str] | None = None,
    ) -> str:
        """Generate text using Medusa multi-head speculation (T3-3).

        Uses the model's MTP predict_heads to draft multiple future
        tokens in parallel from a single hidden state.  The main model
        verifies all draft tokens in one forward pass.

        Unlike :meth:`speculative_generate`, no separate draft model
        is needed — the MTP heads trained during pre-training serve as
        the draft heads.

        Args:
            prompt: Input text.
            max_gen: Maximum new tokens.
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Nucleus sampling threshold.
            repetition_penalty: Repetition penalty.
            min_p: Min-p filtering.
            stop_strings: Early-stop strings.

        Returns:
            Generated text (prompt + new tokens).
        """
        # B-3a sibling closure (Pass 156z9cp): when the splice flag is
        # ON, append ``</search>`` to ``stop_strings`` so the medusa
        # loop halts cleanly on the closing tag, mirroring
        # ``_generate_text``'s ``effective_stop_strings`` pattern.
        effective_stop_strings = stop_strings
        if getattr(self, "inline_search_splice_enabled", False):
            effective_stop_strings = list(stop_strings or [])
            if "</search>" not in effective_stop_strings:
                effective_stop_strings.append("</search>")

        input_ids = self._encode_prompt(prompt)
        device = input_ids.device
        generated = input_ids
        prompt_len = input_ids.shape[1]
        max_len = self.model.config.max_seq_len
        eos_id = getattr(self.tokenizer, 'eos_token_id',
                         getattr(self.tokenizer, 'eos_id', 2))
        tokens_generated = 0
        has_cache = hasattr(self.model, 'clear_cache')

        with self._generation_lock, torch.no_grad():
            if has_cache:
                self.model.clear_cache()

            # Prefill + get first round of draft logits
            main_logits, draft_logits = self.model.medusa_forward(
                input_ids, use_cache=has_cache)

            while tokens_generated < max_gen:
                # Pass B (Pass 156z9ff): honour user stop signal.
                if self._check_cancel():
                    logger.info(
                        "Medusa generation cancelled by user (%d tokens generated)",
                        tokens_generated,
                    )
                    break
                # --- Sample from main head (position +1) ---
                # S740: Use _sample_token for consistent filtering
                tok_1_tensor = self._sample_token(
                    main_logits[:, -1, :],
                    generated, temperature, top_k, top_p,
                    repetition_penalty, min_p)
                tok_1 = tok_1_tensor[0, 0].item()

                if tok_1 == eos_id:
                    generated = torch.cat([
                        generated,
                        torch.tensor([[tok_1]], dtype=torch.long, device=device),
                    ], dim=1)
                    tokens_generated += 1
                    break

                # --- Sample draft tokens from MTP heads (positions +2, +3, ...) ---
                draft_tokens = [tok_1]
                for dl in draft_logits:
                    dt = self._sample_token(
                        dl[:, -1, :],
                        generated, temperature, top_k, top_p,
                        repetition_penalty, min_p)
                    draft_tokens.append(dt[0, 0].item())

                # --- Verify: feed all draft tokens in one pass ---
                draft_tensor = torch.tensor(
                    [draft_tokens], dtype=torch.long, device=device)

                if has_cache:
                    verify_logits = self.model(
                        draft_tensor, use_cache=True,
                        start_pos=generated.shape[1])
                else:
                    full_seq = torch.cat([generated, draft_tensor], dim=1)
                    verify_logits = self.model(full_seq)
                    verify_logits = verify_logits[
                        :, -len(draft_tokens):, :]

                # --- Verify: accept longest matching prefix ---
                accepted = 0
                for j in range(len(draft_tokens)):
                    if j == 0:
                        v_logits = main_logits[:, -1, :]
                    else:
                        v_logits = verify_logits[:, j - 1, :]

                    vt = self._sample_token(
                        v_logits.clone(),
                        generated, temperature, top_k, top_p,
                        repetition_penalty, min_p)
                    verify_token = vt[0, 0].item()
                    if verify_token == draft_tokens[j]:
                        generated = torch.cat([
                            generated,
                            torch.tensor([[draft_tokens[j]]],
                                         dtype=torch.long, device=device),
                        ], dim=1)
                        tokens_generated += 1
                        accepted += 1
                        if draft_tokens[j] == eos_id or tokens_generated >= max_gen:
                            break
                    else:
                        # Replace with verifier's token
                        generated = torch.cat([
                            generated,
                            torch.tensor([[verify_token]],
                                         dtype=torch.long, device=device),
                        ], dim=1)
                        tokens_generated += 1
                        break

                # Rewind cache to actual accepted position
                if has_cache:
                    if hasattr(self.model, 'rewind_cache'):
                        self.model.rewind_cache(generated.shape[1] - 1)
                    else:
                        self.model.clear_cache()
                        self.model(generated, use_cache=True)

                # Next iteration: get fresh main + draft logits
                if tokens_generated < max_gen and generated[0, -1].item() != eos_id:
                    if has_cache:
                        medusa_input = generated[:, -1:]
                    else:
                        medusa_input = generated
                        if medusa_input.shape[1] > max_len:
                            medusa_input = medusa_input[:, -max_len:]
                    main_logits, draft_logits = self.model.medusa_forward(
                        medusa_input,
                        use_cache=has_cache,
                        start_pos=generated.shape[1] - 1 if has_cache else 0,
                    )

                # Stop string check
                if (effective_stop_strings
                        and tokens_generated % self._STOP_CHECK_INTERVAL == 0):
                    tail_start = max(
                        prompt_len,
                        generated.shape[1] - self._STOP_CHECK_WINDOW)
                    recent_text = self.tokenizer.decode(
                        generated[0, tail_start:].tolist(),
                        skip_special_tokens=True)
                    if any(ss in recent_text for ss in effective_stop_strings):
                        break

                if generated[0, -1].item() == eos_id:
                    break

        text = self._decode_output(generated)
        if effective_stop_strings:
            pl = len(prompt)
            gen_part = text[pl:] if len(text) > pl else text
            for ss in effective_stop_strings:
                if ss in gen_part:
                    text = text[:pl + gen_part.find(ss)]
                    break

        # B-3b/B-3c sibling closure (Pass 156z9cp): same retrieve+splice
        # contract as ``_generate_text`` — when the splice flag is ON
        # and a built RAG index is attached, an unclosed ``<search>q``
        # tail triggers up to ``max_search_rounds`` rounds of retrieve
        # and continuation via :meth:`_maybe_rag_splice`.
        spliced = self._maybe_rag_splice(
            text, prompt, max_gen, temperature, top_k, top_p,
            repetition_penalty, min_p,
            effective_stop_strings=effective_stop_strings,
            json_constraint=None,
            tokens_already_generated=tokens_generated,
        )
        if spliced is not None:
            text = spliced

        # Stage B-2 sibling sweep: medusa decoding return path.
        # Pass 156z9e: prompt-slice so prompt-side <search> isn't recorded.
        self._record_search_emissions(text, prompt=prompt, path="medusa")
        return text

    # =========================================================================
    # T5-9: Self-Consistency Decoding
    # =========================================================================

    def self_consistent_generate(
        self,
        prompt: str,
        n_samples: int = 5,
        max_gen: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        min_p: float = 0.0,
        stop_strings: list[str] | None = None,
        extract_answer: typing.Callable[[str], str] | None = None,
    ) -> str:
        """Generate N responses and return the most common answer (T5-9).

        Implements self-consistency decoding (Wang et al. 2022). Samples
        multiple chain-of-thought responses, extracts the final answer
        from each, and picks the majority vote. Dramatically improves
        accuracy on reasoning/math/logic tasks at the cost of N×
        generation time.

        Args:
            prompt: Input text.
            n_samples: Number of independent responses to generate.
            max_gen: Maximum tokens per response.
            temperature: Sampling temperature (should be > 0 for diversity).
            extract_answer: Optional callable that extracts the "answer"
                substring from a full response for voting. Default: use
                the last line of the response.

        Returns:
            The full response whose extracted answer matches the majority.
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")

        responses: list[str] = []
        for _ in range(n_samples):
            # Pass B (Pass 156z9ff): honour user stop signal between samples.
            if self._check_cancel():
                logger.info(
                    "Self-consistent generation cancelled by user after %d samples",
                    len(responses),
                )
                break
            # S780: acquire _generation_lock per sample — _generate_text
            # clears and uses the KV cache, so concurrent calls corrupt it.
            with self._generation_lock:
                text = self._generate_text(
                    prompt, max_gen, temperature, top_k, top_p,
                    repetition_penalty, stop_strings, use_cache=True,
                    min_p=min_p,
                )
            # Strip the prompt if it was echoed
            if text.startswith(prompt):
                text = text[len(prompt):]
            responses.append(text.strip())

        if not responses:
            return ""

        # Extract answers for voting
        extractor = extract_answer or self._default_answer_extractor
        answers = [extractor(r) for r in responses]

        # Majority vote — pick the most common answer
        from collections import Counter
        counts = Counter(answers)
        winner, _ = counts.most_common(1)[0]

        # Return the first full response whose answer matches the winner
        for resp, ans in zip(responses, answers):
            if ans == winner:
                return resp

        return responses[0]  # fallback

    @staticmethod
    def _default_answer_extractor(response: str) -> str:
        """Extract the last non-empty line as the "answer" for voting."""
        lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
        return lines[-1] if lines else response.strip()

    # =========================================================================
    # T5-5: Lookahead Decoding (Jacobi Iteration)
    # =========================================================================

    def lookahead_generate(
        self,
        prompt: str,
        max_gen: int = 512,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        min_p: float = 0.0,
        stop_strings: list[str] | None = None,
        lookahead_size: int = 4,
        ngram_pool_size: int = 64,
    ) -> str:
        """Generate tokens using lookahead decoding (T5-5).

        Uses Jacobi iteration to speculatively predict multiple future
        positions in parallel, verified in a single forward pass.
        Maintains an N-gram pool from past generations for prediction.
        No draft model or extra heads needed — uses only the main model.

        Args:
            prompt: Input text.
            max_gen: Maximum tokens to generate.
            lookahead_size: Number of lookahead positions.
            ngram_pool_size: Max N-grams in the lookup pool.

        Returns:
            Generated text.
        """
        # B-3a sibling closure (Pass 156z9cp): when the splice flag is
        # ON, append ``</search>`` to ``stop_strings`` so the lookahead
        # loop halts cleanly on the closing tag, mirroring
        # ``_generate_text``'s ``effective_stop_strings`` pattern.
        effective_stop_strings = stop_strings
        if getattr(self, "inline_search_splice_enabled", False):
            effective_stop_strings = list(stop_strings or [])
            if "</search>" not in effective_stop_strings:
                effective_stop_strings.append("</search>")

        input_ids = self._encode_prompt(prompt)
        generated = input_ids
        prompt_len = input_ids.shape[1]
        max_len = self.model.config.max_seq_len
        device = input_ids.device
        eos_id = getattr(self.tokenizer, 'eos_token_id', 2)

        # N-gram pool: maps (tok_i, tok_i+1) -> tok_i+2 (bigram -> next token)
        ngram_pool: dict[tuple[int, int], int] = {}
        # Pass 156z9fe (Pass A fix #4): track the next bigram index to
        # update so each call walks only NEW tokens instead of the full
        # generated sequence.  Was O(n²) cumulative over the run.
        ngram_pool_idx = 0

        with self._generation_lock, torch.no_grad():
            has_cache = hasattr(self.model, 'clear_cache')
            if has_cache:
                self.model.clear_cache()

            # Prefill
            curr = input_ids
            if curr.shape[1] > max_len:
                curr = curr[:, -max_len:]
            logits = self.model(curr, use_cache=has_cache)

            tokens_generated = 0
            while tokens_generated < max_gen:
                # Pass B (Pass 156z9ff): honour user stop signal.
                if self._check_cancel():
                    logger.info(
                        "Lookahead generation cancelled by user (%d tokens generated)",
                        tokens_generated,
                    )
                    break
                # Sample the verified next token
                next_token = self._sample_token(
                    logits[:, -1, :], generated, temperature,
                    top_k, top_p, repetition_penalty, min_p,
                )
                next_id = next_token[0, 0].item()
                generated = torch.cat([generated, next_token], dim=1)
                tokens_generated += 1

                if next_id == eos_id:
                    break

                # Build lookahead candidates from N-gram pool
                draft: list[int] = [next_id]
                for _ in range(lookahead_size - 1):
                    seq = generated[0].tolist()
                    if len(seq) >= 2:
                        key = (seq[-2], seq[-1]) if len(draft) == 1 else (draft[-2], draft[-1])
                        predicted = ngram_pool.get(key)
                        if predicted is not None:
                            draft.append(predicted)
                        else:
                            break
                    else:
                        break

                if len(draft) <= 1:
                    # No N-gram predictions available, fall back to single step
                    if has_cache:
                        logits = self.model(
                            next_token, use_cache=True,
                            start_pos=generated.shape[1] - 1)
                    else:
                        logits = self.model(generated[:, -max_len:])
                    # Update N-gram pool from generated tokens
                    ngram_pool_idx = self._update_ngram_pool(
                        ngram_pool, generated[0].tolist(),
                        ngram_pool_size, start_index=ngram_pool_idx)
                    continue

                # Verify draft: feed all draft tokens through model in one pass
                draft_tensor = torch.tensor(
                    [draft], dtype=torch.long, device=device)
                if has_cache:
                    # Cache has 0..generated[-2]; draft[0]=next_id not cached yet.
                    # Send the full draft so next_id enters the KV cache at
                    # position generated.shape[1]-1 and preds follow at N, N+1…
                    verify_logits = self.model(
                        draft_tensor, use_cache=True,
                        start_pos=generated.shape[1] - 1)
                else:
                    full = torch.cat([generated[:, :-1], draft_tensor], dim=1)
                    verify_logits = self.model(full[:, -max_len:])
                    # Slice to the draft region so verify_logits[0,j] aligns
                    # with draft[j] (prediction for draft[j+1]).
                    verify_logits = verify_logits[:, -len(draft):, :]

                # Check which draft tokens match the verifier's greedy picks
                accepted = 0
                for j in range(len(draft) - 1):
                    if j >= verify_logits.shape[1]:
                        break
                    verify_token = verify_logits[0, j].argmax().item()
                    if verify_token == draft[j + 1]:
                        # Accept the draft token
                        generated = torch.cat([
                            generated,
                            torch.tensor([[draft[j + 1]]],
                                         dtype=torch.long, device=device),
                        ], dim=1)
                        accepted += 1
                        tokens_generated += 1
                        if draft[j + 1] == eos_id or tokens_generated >= max_gen:
                            break
                    else:
                        # Divergence — accept verifier's token instead
                        generated = torch.cat([
                            generated,
                            torch.tensor([[verify_token]],
                                         dtype=torch.long, device=device),
                        ], dim=1)
                        tokens_generated += 1
                        break

                # Update N-gram pool with accepted tokens
                ngram_pool_idx = self._update_ngram_pool(
                    ngram_pool, generated[0].tolist(),
                    ngram_pool_size, start_index=ngram_pool_idx)

                # Reset cache to match actual generated sequence
                if has_cache and hasattr(self.model, 'rewind_cache'):
                    self.model.rewind_cache(generated.shape[1] - 1)
                    logits = self.model(
                        generated[:, -1:], use_cache=True,
                        start_pos=generated.shape[1] - 1)
                else:
                    if has_cache:
                        self.model.clear_cache()
                    logits = self.model(
                        generated[:, -max_len:], use_cache=has_cache)

                # Stop string check
                if (effective_stop_strings
                        and tokens_generated % self._STOP_CHECK_INTERVAL == 0):
                    tail_start = max(
                        prompt_len,
                        generated.shape[1] - self._STOP_CHECK_WINDOW)
                    recent_text = self.tokenizer.decode(
                        generated[0, tail_start:].tolist(),
                        skip_special_tokens=True)
                    if any(ss in recent_text for ss in effective_stop_strings):
                        break

                if generated[0, -1].item() == eos_id:
                    break

        text = self._decode_output(generated)
        if effective_stop_strings:
            pl = len(prompt)
            gen_part = text[pl:] if len(text) > pl else text
            for ss in effective_stop_strings:
                if ss in gen_part:
                    text = text[:pl + gen_part.find(ss)]
                    break

        # B-3b/B-3c sibling closure (Pass 156z9cp): same retrieve+splice
        # contract as ``_generate_text`` — when the splice flag is ON
        # and a built RAG index is attached, an unclosed ``<search>q``
        # tail triggers up to ``max_search_rounds`` rounds of retrieve
        # and continuation via :meth:`_maybe_rag_splice`.
        spliced = self._maybe_rag_splice(
            text, prompt, max_gen, temperature, top_k, top_p,
            repetition_penalty, min_p,
            effective_stop_strings=effective_stop_strings,
            json_constraint=None,
            tokens_already_generated=tokens_generated,
        )
        if spliced is not None:
            text = spliced

        # Stage B-2 sibling sweep: lookahead decoding return path.
        # Pass 156z9e: prompt-slice so prompt-side <search> isn't recorded.
        self._record_search_emissions(
            text, prompt=prompt, path="lookahead")
        return text

    @staticmethod
    def _update_ngram_pool(
        pool: dict[tuple[int, int], int],
        tokens: list[int],
        max_size: int,
        start_index: int = 0,
    ) -> int:
        """Update bigram → next-token pool from a token sequence.

        Args:
            pool: Bigram → next-token map (mutated in place).
            tokens: Token sequence to scan.
            max_size: Max pool entries; FIFO-evict oldest above this.
            start_index: First bigram index to scan.  Pass 156z9fe
                (Pass A fix #4): callers track the last-scanned index
                and pass it back here so each call walks only NEW
                tokens.  Previously called with ``start_index=0`` on
                every iteration, producing O(n²) cumulative work over
                a long generation.

        Returns:
            The new ``start_index`` callers should pass on the next
            call (i.e. ``max(0, len(tokens) - 2)``).
        """
        for i in range(max(0, start_index), len(tokens) - 2):
            key = (tokens[i], tokens[i + 1])
            pool[key] = tokens[i + 2]
        # Evict oldest entries if pool exceeds max size
        while len(pool) > max_size:
            oldest_key = next(iter(pool))
            del pool[oldest_key]
        return max(0, len(tokens) - 2)

    def _check_cancel(self) -> bool:
        """Consume the cancel-generation signal.

        Pass 156z9ff (Pass B): closes the signal-without-consumer
        sibling-boundary dead-infra (§4 Learned Principles). The
        ``_cancel_generation`` attribute is set by
        :func:`builtin_commands.stop_cmd` (gated on
        ``_generation_lock.locked()`` so the flag is only ever set
        while a generation is active). Every token-loop in this
        mixin calls this helper first thing in each iteration and
        breaks out on True. Read-and-clear (one-shot semantics) so
        a single set never cancels more than one in-flight loop.

        Returns:
            True if a stop was requested (flag was True; now cleared).
            False otherwise.
        """
        if getattr(self, "_cancel_generation", False):
            self._cancel_generation = False
            return True
        return False
