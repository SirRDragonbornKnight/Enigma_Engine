"""
Chat, streaming-chat, and tool-aware generation methods for EnigmaEngine.

This module is split from ``inference.py`` to keep file sizes manageable.
The :class:`_ChatMixin` is mixed into :class:`EnigmaEngine` so that every
method is available on the engine instance as before.

**Do not import from ``inference`` here** — that would create a circular
import.  All instance attributes (``self.model``, ``self.tokenizer``,
``self.device``, etc.) are set by ``EnigmaEngine.__init__`` which lives
in ``inference.py``.
"""
from __future__ import annotations

import logging
from collections.abc import Generator
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =============================================================================
# ChatContext — shared preparation result for chat() and stream_chat()
# =============================================================================

@dataclass
class ChatContext:
    """Prepared context returned by ``_prepare_chat()``.

    Both ``chat()`` and ``stream_chat()`` call ``_prepare_chat()`` to
    build this once, then diverge only for the generation path.

    Attributes:
        messages: GGUF-style message list (system + history + user).
        prompt: Formatted text prompt for native ``generate()``.
        stop_strings: Sequences that terminate generation.
        max_gen: Effective max generation length (boosted when reasoning).
        temperature: Sampling temperature.
        repeat_penalty: Repetition penalty.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling count.
        is_gguf: Whether the model uses the GGUF backend.
        has_server_backend: Whether the GGUF model uses llama-server.
    """
    messages: list[dict[str, str]]
    prompt: str
    stop_strings: list[str]
    max_gen: int
    temperature: float
    repeat_penalty: float
    top_p: float
    top_k: int
    is_gguf: bool
    has_server_backend: bool


class _ChatMixin:
    """Chat and tool-aware generation methods mixed into EnigmaEngine."""

    # =========================================================================
    # Vision Encoding for Chat
    # =========================================================================

    def _encode_images_for_chat(
        self,
        image_paths: list[str],
    ):
        """Encode image paths into vision features for multimodal chat.

        Uses the loaded ``vision_encoder`` to preprocess and encode each
        image, then concatenates the features along the sequence dimension.

        Args:
            image_paths: List of file paths to images.

        Returns:
            Batched vision features tensor ``[1, total_patches, dim]``,
            or ``None`` if no vision encoder is loaded.
        """
        encoder = getattr(self, "vision_encoder", None)
        if encoder is None:
            # Loud when caller actually expected images to be processed —
            # silent on the normal text-only path (V-8).
            if image_paths:
                logger.warning(
                    "Image input provided (%d image(s)) but model has no "
                    "vision encoder loaded — images will be ignored. Train a "
                    "vision-capable model with Forge and reload.",
                    len(image_paths),
                )
            else:
                logger.debug("No vision encoder loaded — skipping image encoding")
            return None

        try:
            import torch
            from .vision_encoder import preprocess_image
        except ImportError:
            logger.warning("Vision encoder dependencies not available")
            return None

        features_list = []
        device = getattr(self, "device", None)
        if device is not None:
            encoder = encoder.to(device)
        for path in image_paths:
            try:
                img_tensor = preprocess_image(
                    path, image_size=encoder.config.image_size,
                )
                if device is not None:
                    img_tensor = img_tensor.to(device)
                with torch.no_grad():
                    feat = encoder(img_tensor)  # [1, patches, dim]
                features_list.append(feat)
            except Exception as exc:
                logger.warning("Failed to encode image %s: %s", path, exc)

        if not features_list:
            return None

        # Concatenate along sequence dim → [1, total_patches, dim]
        return torch.cat(features_list, dim=1)

    # =========================================================================
    # Chat Interface
    # =========================================================================
    @staticmethod
    def _summarize_dropped_history(
        messages: list[dict[str, str]],
    ) -> str:
        """Summarize messages that are about to be dropped from context.

        Produces a compact text summary of the conversation that was
        trimmed, preserving key topics and context without requiring
        model inference.  Groups by topic shifts and extracts
        decisions/conclusions from assistant responses.

        Args:
            messages: The messages being dropped from history.

        Returns:
            A compact summary string, or "" if messages is empty.
        """
        if not messages:
            return ""

        # Extract key info from dropped messages
        user_msgs = [m["content"] for m in messages
                     if m.get("role") == "user" and m.get("content")]
        asst_msgs = [m["content"] for m in messages
                     if m.get("role") == "assistant" and m.get("content")]

        if not user_msgs and not asst_msgs:
            return ""

        parts: list[str] = []
        parts.append(
            f"[Earlier conversation summary — {len(messages)} "
            f"messages were trimmed from context]")

        # --- Topic grouping ---
        # Detect topic shifts by comparing consecutive user messages.
        # A "topic" is the first ~40 chars of each user message.
        # When consecutive messages share few words, it's a new topic.
        topic_groups: list[list[str]] = []
        current_group: list[str] = []
        prev_words: set[str] = set()

        for msg in user_msgs:
            words = set(msg.lower().split()[:8])
            # If less than 20% word overlap with previous, new topic
            if prev_words and len(words & prev_words) < max(1, len(words) * 0.2):
                if current_group:
                    topic_groups.append(current_group)
                current_group = [msg]
            else:
                current_group.append(msg)
            prev_words = words

        if current_group:
            topic_groups.append(current_group)

        # Summarize topic groups
        if len(topic_groups) > 1:
            topic_labels = []
            for group in topic_groups[:6]:  # cap at 6 topics
                label = group[0][:60]
                if len(group) > 1:
                    label += f" ({len(group)} messages)"
                topic_labels.append(label)
            parts.append("Topics discussed: " + " | ".join(topic_labels))
        elif user_msgs:
            parts.append(f"Started with: {user_msgs[0][:120]}")

        # --- Extract decisions/conclusions from assistant messages ---
        # Look for decisive language patterns in assistant responses.
        decision_markers = (
            "recommend", "decided", "go with", "choose", "best",
            "should use", "let's use", "agreed", "conclusion",
            "the answer is", "in summary", "to summarize",
        )
        decisions: list[str] = []
        for msg in asst_msgs:
            lower = msg.lower()
            if any(marker in lower for marker in decision_markers):
                # Take the sentence containing the marker
                for sentence in msg.replace("!", ".").replace("?", ".").split("."):
                    sentence = sentence.strip()
                    if sentence and any(m in sentence.lower() for m in decision_markers):
                        if len(sentence) > 10:
                            decisions.append(sentence[:100])
                            break
        if decisions:
            parts.append("Key decisions: " + " | ".join(decisions[:4]))

        # --- Last exchange ---
        if user_msgs:
            parts.append(f"Last user message: {user_msgs[-1][:100]}")
        if asst_msgs:
            parts.append(f"Last AI response: {asst_msgs[-1][:100]}")

        return "\n".join(parts)

    @staticmethod
    def _cap_history_summary(summary: str, max_chars: int = 4096) -> str:
        """Keep only the most recent summary blocks within *max_chars*.

        Each truncation event produces a block starting with
        ``[Earlier conversation summary``.  When the combined summary
        grows past the budget, the oldest blocks are dropped.
        """
        if not summary or len(summary) <= max_chars:
            return summary

        marker = "[Earlier conversation summary"
        blocks = summary.split(marker)
        # blocks[0] is text before the first marker (usually "")
        # Rebuild from newest to oldest, keeping what fits
        kept: list[str] = []
        total = 0
        for block in reversed(blocks):
            piece = marker + block if kept or block != blocks[0] else block
            if total + len(piece) > max_chars and kept:
                break
            kept.append(piece)
            total += len(piece)
        kept.reverse()
        result = "".join(kept).strip()
        return result if result else summary[-max_chars:]

    def _truncate_history(
        self,
        history: list[dict[str, str]],
        current_message: str,
        system_prompt: str | None = None,
        max_history_tokens: int | None = None,
        reserve_for_response: int = 200
    ) -> list[dict[str, str]]:
        """
        Truncate conversation history to fit within context window.

        This prevents hallucinations caused by context overflow!

        Args:
            history: Full conversation history
            current_message: Current user message
            system_prompt: Optional system prompt
            max_history_tokens: Max tokens for history (auto-calculated if None)
            reserve_for_response: Tokens to reserve for AI response

        Returns:
            Truncated history that fits in context window
        """
        if not history:
            return []

        # Calculate available space
        max_context = self.get_max_context_length()

        # Reserve space for: system prompt + current message + response
        reserved = reserve_for_response
        try:
            if system_prompt:
                reserved += self.count_tokens(f"System: {system_prompt}\n")
            reserved += self.count_tokens(f"User: {current_message}\nAssistant:")
        except Exception:
            # Fallback: rough char-based estimate if tokenizer unavailable.
            # CJK text averages ~1.5 chars/token vs ~4 for Latin.
            sample = (system_prompt or "") + current_message
            has_cjk = any('\u4e00' <= c <= '\u9fff' for c in sample[:200])
            chars_per_token = 2 if has_cjk else 4
            if system_prompt:
                reserved += len(f"System: {system_prompt}\n") // chars_per_token
            reserved += len(f"User: {current_message}\nAssistant:") // chars_per_token

        max_history_tokens = max_history_tokens or max(0, max_context - reserved)

        # If very limited context, keep only last exchange
        if max_history_tokens < 100:
            logger.warning(f"Very limited context ({max_context} tokens), keeping only last exchange")
            return history[-2:] if len(history) >= 2 else history[-1:]

        # Build history from most recent, counting tokens
        truncated = []
        total_tokens = 0

        for msg in reversed(history):
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            msg_text = f"{role}: {content}\n"
            try:
                msg_tokens = self.count_tokens(msg_text)
            except Exception:
                msg_tokens = len(msg_text) // 4

            if total_tokens + msg_tokens > max_history_tokens:
                # Don't add this message, we're at limit
                break

            truncated.insert(0, msg)
            total_tokens += msg_tokens

        if len(truncated) < len(history):
            # Summarize the dropped messages so context isn't fully lost
            dropped = history[:len(history) - len(truncated)]
            new_summary = self._summarize_dropped_history(dropped)
            # Append to any existing summary
            old_summary = getattr(self, "_history_summary", "")
            if old_summary and new_summary:
                combined = f"{old_summary}\n{new_summary}"
            elif new_summary:
                combined = new_summary
            else:
                combined = old_summary
            self._history_summary = self._cap_history_summary(
                combined, max_chars=4096)
            logger.info(
                f"Truncated history: {len(history)} -> "
                f"{len(truncated)} messages ({total_tokens} tokens)")

        return truncated

    def _prepare_chat(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
        max_gen: int = 2048,
        auto_truncate: bool = True,
        reasoning: bool = False,
        **kwargs,
    ) -> ChatContext:
        """Shared preparation for ``chat()`` and ``stream_chat()``.

        Handles history truncation, reasoning injection, GGUF message
        list construction, prompt building, and kwarg extraction.

        Returns:
            A :class:`ChatContext` with everything both callers need.
        """
        # ── Override max_gen from kwargs if caller passed max_tokens ──────
        max_gen = kwargs.pop("max_tokens", kwargs.pop("max_new_tokens", max_gen))

        # ── Truncate history to prevent context overflow ─────────────────
        if auto_truncate and history:
            history = self._truncate_history(
                history,
                current_message=message,
                system_prompt=system_prompt,
                reserve_for_response=max_gen,
            )
        elif not history:
            # Clear stale summary when history is empty/reset
            self._history_summary = ""

        # ── Inject history summary if earlier messages were trimmed ────
        history_summary = getattr(self, "_history_summary", "")
        if history_summary:
            if system_prompt:
                system_prompt = (
                    f"{system_prompt}\n\n{history_summary}")
            else:
                system_prompt = history_summary

        # ── RAG — prepend retrieved document context if index is active ──
        rag_index = getattr(self, "_rag_index", None)
        if rag_index is not None and getattr(rag_index, "is_built", False) and message:
            try:
                from .rag import RAGIndex
                results = rag_index.query(message, top_k=5)
                doc_ctx = RAGIndex.format_context(results, max_chars=2000)
                if doc_ctx:
                    rag_block = (
                        "[RETRIEVED DOCUMENT CONTEXT]\n"
                        f"{doc_ctx}\n"
                        "[END DOCUMENT CONTEXT]"
                    )
                    if system_prompt:
                        system_prompt = f"{system_prompt}\n\n{rag_block}"
                    else:
                        system_prompt = rag_block
            except Exception:
                logger.debug(
                    "RAG query failed — continuing without document context",
                    exc_info=True,
                )

        # ── Reasoning: inject chain-of-thought instruction ───────────────
        if reasoning:
            from .reasoning import build_reasoning_instruction
            reasoning_instruction = build_reasoning_instruction()
            if system_prompt:
                system_prompt = f"{system_prompt}\n\n{reasoning_instruction}"
            else:
                system_prompt = reasoning_instruction
            # Give extra token budget for the thinking section
            max_gen = int(max_gen * 1.5)
            # Clamp to model's max_seq_len to avoid context overflow
            seq_limit = getattr(
                getattr(self, 'model', None), 'config', None)
            if seq_limit is not None:
                seq_limit = getattr(seq_limit, 'max_seq_len', None)
            if isinstance(seq_limit, int) and max_gen > seq_limit:
                max_gen = seq_limit

        # ── Build GGUF-style messages list ───────────────────────────────
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": message})

        # ── Build native text prompt ─────────────────────────────────────
        try:
            from .prompt_builder import get_prompt_builder
            builder = get_prompt_builder()
            full_prompt = builder.build_chat_prompt(
                message=message,
                history=history,
                system_prompt=system_prompt,
                include_generation_prefix=True,
            )
            stop_strings = builder.get_stop_sequences()
        except ImportError:
            prompt_parts = []
            if system_prompt:
                prompt_parts.append(f"System: {system_prompt}\n")
            if history:
                for msg in history:
                    role = msg.get("role", "user").capitalize()
                    content = msg.get("content", "")
                    prompt_parts.append(f"{role}: {content}")
            prompt_parts.append(f"User: {message}")
            prompt_parts.append("Assistant:")
            full_prompt = "\n".join(prompt_parts)
            stop_strings = ["\nUser:", "\n\n", "User:"]

        # ── Extract common kwargs ────────────────────────────────────────
        temperature = kwargs.get("temperature", 0.8)
        repeat_penalty = kwargs.get(
            "repeat_penalty",
            kwargs.get("repetition_penalty", 1.1),
        )
        top_p = kwargs.get("top_p", 0.9)
        top_k = kwargs.get("top_k", 50)

        is_gguf = bool(getattr(self, "_is_gguf", False))
        has_server = bool(
            is_gguf and getattr(getattr(self, "model", None), "_server", None)
        )

        return ChatContext(
            messages=messages,
            prompt=full_prompt,
            stop_strings=stop_strings,
            max_gen=max_gen,
            temperature=temperature,
            repeat_penalty=repeat_penalty,
            top_p=top_p,
            top_k=top_k,
            is_gguf=is_gguf,
            has_server_backend=has_server,
        )

    def chat(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
        max_gen: int = 2048,
        auto_truncate: bool = True,
        reasoning: bool = False,
        images: list[str] | None = None,
        **kwargs
    ) -> str:
        """Chat-style generation with conversation history.

        Builds a structured prompt from the conversation history and the
        current user message, runs it through ``generate()``, and extracts
        only the assistant's reply.

        When ``auto_truncate`` is enabled (default), long conversation
        histories are automatically trimmed so they fit inside the model's
        context window.  This prevents the model from receiving a prompt
        that exceeds ``max_seq_len`` -- a common cause of hallucinations
        and garbled output.

        When ``reasoning`` is enabled, a chain-of-thought instruction is
        injected into the system prompt so the model produces
        ``<think>...</think>`` blocks before answering.  The raw response
        (including reasoning tags) is returned; the caller (e.g. the GUI)
        is responsible for extracting and displaying the thinking section.

        Args:
            message: The user's current message.
            history: Previous turns as a list of dicts, each with
                ``"role"`` (``"user"`` or ``"assistant"``) and
                ``"content"`` keys.  ``None`` starts a fresh
                conversation.
            system_prompt: An optional system instruction prepended to the
                prompt (e.g. ``"You are a helpful coding assistant."``).
            max_gen: Maximum number of new tokens to generate for the
                assistant reply.
            auto_truncate: If ``True``, older history entries are dropped
                when the prompt would exceed the model's context window.
            reasoning: If ``True``, inject chain-of-thought instructions
                so the model thinks step-by-step inside
                ``<think>...</think>`` tags before answering.
            images: Optional list of image file paths to include as
                visual context.  Requires a loaded ``vision_encoder``.
                Each image is encoded through the vision encoder and the
                resulting features are passed to ``forward_multimodal``.
            **kwargs: Extra keyword arguments forwarded to ``generate()``
                (e.g. ``temperature``, ``top_k``, ``top_p``).

        Returns:
            The assistant's response text (without prompt or history).
            When ``reasoning=True`` the response may contain
            ``<think>...</think>`` blocks.

        Raises:
            RuntimeError: If the underlying model is not loaded or the
                tokenizer fails to encode the prompt.

        Example:
            >>> engine = EnigmaEngine()
            >>> reply = engine.chat("What is Python?")
            >>> print(reply)
            'Python is a high-level programming language...'
            >>>
            >>> # Multi-turn with history
            >>> history = [
            ...     {"role": "user", "content": "Hi!"},
            ...     {"role": "assistant", "content": "Hello! How can I help?"},
            ... ]
            >>> reply = engine.chat("Tell me a joke", history=history)
            >>>
            >>> # Chain-of-thought reasoning
            >>> reply = engine.chat("What is 15 * 23?", reasoning=True)
            >>> # reply may contain <think>...</think> block
        """
        # ── Shared preparation (truncation, reasoning, prompt/messages) ──
        ctx = self._prepare_chat(
            message,
            history=history,
            system_prompt=system_prompt,
            max_gen=max_gen,
            auto_truncate=auto_truncate,
            reasoning=reasoning,
            **kwargs,
        )

        # ── GGUF model: use native chat completion ──────────────────────
        if ctx.is_gguf and hasattr(self.model, "chat"):
            # Pass 156z7 (N-15c2): mirror the streaming GGUF gate from
            # stream_chat. llama.cpp uses its own sampler — our logit
            # mask never reaches it, so silently returning unconstrained
            # output labelled as schema-conforming would be a correctness
            # lie. Loud-reject at the boundary instead. Sibling-boundary
            # site that Pass 156z6 missed.
            if kwargs.get("json_schema") is not None:
                raise NotImplementedError(
                    "json_schema constrained decoding is not supported "
                    "on GGUF models (llama.cpp uses its own sampler). "
                    "Load a native PyTorch model or drop the schema."
                )
            try:
                effective_max = kwargs.get("max_tokens", ctx.max_gen)
                with self._generation_lock:
                    response = self.model.chat(
                        messages=ctx.messages,
                        max_tokens=effective_max,
                        temperature=ctx.temperature,
                        top_p=ctx.top_p,
                        top_k=ctx.top_k,
                        repeat_penalty=ctx.repeat_penalty,
                    )
                # Pass 156z9e (Stage B-2 sibling sweep): GGUF chat
                # bypasses _generate_text, so the hook there never
                # ran on this path.  llama.cpp returns continuation
                # only, so prompt=None.  Loud-on-real-issue: a model
                # emitting <search> here gets one WARNING.
                self._record_search_emissions(response)
                return response
            except Exception:
                # Let the error propagate — no silent fallback (Suggestion #9A)
                raise

        # ── Vision: encode attached images ───────────────────────────────
        vision_features = None
        if images:
            vision_features = self._encode_images_for_chat(images)
            if vision_features is not None:
                logger.info(
                    "Encoded %d image(s) → vision features %s",
                    len(images), tuple(vision_features.shape),
                )

        # ── Prefix KV cache: skip system-prompt prefill on cache hit ─────
        # Only applicable for native models (not GGUF, not vision)
        self._pending_prefix_cache = None
        self._pending_prefix_build = None
        effective_system = (
            ctx.messages[0]["content"]
            if ctx.messages and ctx.messages[0].get("role") == "system"
            else None
        )
        if (effective_system
                and vision_features is None
                and hasattr(self, "model")
                and hasattr(self.model, "clear_cache")):
            import hashlib
            prompt_hash = hashlib.sha256(
                effective_system.encode("utf-8", errors="replace")
            ).hexdigest()[:16]
            if (prompt_hash == self._prefix_prompt_hash
                    and self._prefix_kv_cache is not None):
                self._pending_prefix_cache = self._prefix_kv_cache
                logger.debug("Prefix KV cache hit (hash=%s)", prompt_hash)
            else:
                self._pending_prefix_build = {
                    "hash": prompt_hash,
                    "system_prefix_text": f"System: {effective_system}\n",
                }
                logger.debug(
                    "Prefix KV cache miss — will snapshot after generation")

        if vision_features is not None and not ctx.is_gguf:
            response = self._generate_with_vision(
                prompt=ctx.prompt,
                vision_features=vision_features,
                max_gen=ctx.max_gen,
                stop_strings=ctx.stop_strings,
                **kwargs,
            )
        else:
            response = self.generate(
                ctx.prompt,
                max_gen=ctx.max_gen,
                stop_strings=ctx.stop_strings,
                **kwargs,
            )

        # Extract assistant's response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        # Clean up truncated <think> tags so API callers get clean output
        # (the GUI also does this, but programmatic callers shouldn't need to)
        from .reasoning import strip_incomplete_think
        response = strip_incomplete_think(response)

        return response

    def chat_with_tools(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
        max_gen: int = 200,
        fallback_to_chat: bool = True,
        **kwargs
    ) -> str:
        """
        Chat with automatic tool routing based on user intent.

        Uses the UniversalToolRouter which detects tool intent from
        keywords in the user message. Works regardless of whether the
        model was trained to use tools.

        Args:
            message: User's message
            history: Conversation history
            system_prompt: Optional system prompt
            max_gen: Maximum tokens to generate
            fallback_to_chat: If tool fails, use chat instead
            **kwargs: Additional generation parameters

        Returns:
            Response (either from tool execution or chat)
        """
        try:
            from .universal_router import chat_with_tools as universal_chat
        except ImportError:
            if fallback_to_chat:
                logger.warning("universal_router module not available, falling back to chat")
                return self.chat(message, history=history, system_prompt=system_prompt, max_gen=max_gen, **kwargs)
            raise RuntimeError("universal_router module is not installed — cannot use chat_with_tools") from None

        # Create a chat function that preserves history/system prompt
        def chat_fn(msg, **kw):
            merged = {**kwargs, **kw}
            return self.chat(
                msg,
                history=history,
                system_prompt=system_prompt,
                max_gen=max_gen,
                **merged
            )

        return universal_chat(
            message,
            chat_fn,
            fallback_to_chat=fallback_to_chat
        )

    def stream_chat(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
        max_gen: int = 2048,
        auto_truncate: bool = True,
        reasoning: bool = False,
        **kwargs
    ) -> Generator[str]:
        """
        Stream chat-style generation token-by-token.

        Works with both native PyTorch models (via ``stream_generate``)
        and GGUF models (via ``create_chat_completion(stream=True)``).

        When ``reasoning`` is enabled, chain-of-thought instructions are
        injected into the system prompt (same as ``chat()``).

        Args:
            message: User's message
            history: Conversation history
            system_prompt: Optional system prompt
            max_gen: Maximum tokens to generate
            auto_truncate: If True, truncate history to fit context
            reasoning: If True, inject chain-of-thought instructions
            **kwargs: Additional parameters

        Yields:
            Generated tokens one at a time
        """
        # ── Shared preparation (truncation, reasoning, prompt/messages) ──
        ctx = self._prepare_chat(
            message,
            history=history,
            system_prompt=system_prompt,
            max_gen=max_gen,
            auto_truncate=auto_truncate,
            reasoning=reasoning,
            **kwargs,
        )

        # ── GGUF streaming path ──────────────────────────────────────────
        if ctx.is_gguf and hasattr(self.model, "chat"):
            # N-15c: GGUF uses llama.cpp's own sampler — our logit mask
            # never gets a chance to run on streaming GGUF either. Match
            # the non-streaming GGUF rejection in `_generate_text` so
            # callers fail loud at the API boundary instead of silently
            # receiving unconstrained tokens labelled as schema-valid.
            if kwargs.get("json_schema") is not None:
                raise NotImplementedError(
                    "json_schema constrained decoding is not supported "
                    "on GGUF models (llama.cpp uses its own sampler). "
                    "Load a native PyTorch model or drop the schema."
                )
            # Server backend — no streaming helper yet, yield in one piece
            if ctx.has_server_backend:
                with self._generation_lock:
                    response = self.model.chat(
                        ctx.messages,
                        max_tokens=ctx.max_gen,
                        temperature=ctx.temperature,
                        repeat_penalty=ctx.repeat_penalty,
                        top_p=ctx.top_p,
                        top_k=ctx.top_k,
                    )
                # Pass 156z9e (Stage B-2 sibling sweep): GGUF server
                # streaming returns the full response in one chunk
                # without going through stream_generate's finally hook.
                self._record_search_emissions(response)
                yield response
                return

            # In-process llama-cpp-python — true streaming
            try:
                # Pass 156z9e (Stage B-2 sibling sweep): GGUF llama-cpp
                # streaming bypasses stream_generate, so accumulate
                # chunks and scan in finally — same shape as the
                # native stream_generate hook.  Runs on normal
                # completion AND on generator cancellation.
                gguf_chunks: list[str] = []
                try:
                    with self._generation_lock:
                        stream_resp = self.model.model.create_chat_completion(
                            messages=ctx.messages,
                            max_tokens=ctx.max_gen,
                            temperature=ctx.temperature,
                            repeat_penalty=ctx.repeat_penalty,
                            top_p=ctx.top_p,
                            top_k=ctx.top_k,
                            stream=True,
                        )
                        for chunk in stream_resp:
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                text = delta.get("content", "")
                                if text:
                                    gguf_chunks.append(text)
                                    yield text
                    return
                finally:
                    try:
                        self._record_search_emissions("".join(gguf_chunks))
                    except Exception:
                        logger.exception(
                            "Stage B-2: GGUF stream scan crashed; "
                            "last_search_queries left at previous value")
            except Exception:
                # Let the error propagate — no silent fallback (Suggestion #9A)
                raise

        # ── Native model streaming path ──────────────────────────────────
        # Hold back tokens that could be a prefix of a stop string so we
        # don't leak partial stop sequences to consumers.
        pending = ""
        max_stop_len = max((len(s) for s in ctx.stop_strings), default=0)
        stopped = False

        for token in self.stream_generate(ctx.prompt, max_gen=ctx.max_gen, **kwargs):
            pending += token

            # Check if a full stop string has appeared
            stopped = False
            for stop in ctx.stop_strings:
                pos = pending.find(stop)
                if pos != -1:
                    # Yield everything before the stop string
                    safe = pending[:pos]
                    if safe:
                        yield safe
                    stopped = True
                    break

            if stopped:
                break

            # Check if the tail of pending could be the start of a stop
            # string — if so, hold those characters back
            hold = 0
            if max_stop_len > 0:
                for k in range(1, min(len(pending), max_stop_len) + 1):
                    tail = pending[-k:]
                    for stop in ctx.stop_strings:
                        if stop[:k] == tail:
                            hold = max(hold, k)
                            break

            safe_end = len(pending) - hold
            if safe_end > 0:
                yield pending[:safe_end]
                pending = pending[safe_end:]

        # Flush remaining pending at EOF (generation ended, no stop string hit)
        if pending and not stopped:
            yield pending


