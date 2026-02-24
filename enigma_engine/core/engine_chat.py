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
from typing import Any

logger = logging.getLogger(__name__)


class _ChatMixin:
    """Chat and tool-aware generation methods mixed into EnigmaEngine."""

    # =========================================================================
    # 💬 Chat Interface
    # =========================================================================

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
        if system_prompt:
            reserved += self.count_tokens(f"System: {system_prompt}\n")
        reserved += self.count_tokens(f"User: {current_message}\nAssistant:")
        
        max_history_tokens = max_history_tokens or (max_context - reserved)
        
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
            msg_tokens = self.count_tokens(msg_text)
            
            if total_tokens + msg_tokens > max_history_tokens:
                # Don't add this message, we're at limit
                break
            
            truncated.insert(0, msg)
            total_tokens += msg_tokens
        
        if len(truncated) < len(history):
            logger.info(f"Truncated history: {len(history)} -> {len(truncated)} messages ({total_tokens} tokens)")
        
        return truncated

    def chat(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
        max_gen: int = 200,
        auto_truncate: bool = True,
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
            **kwargs: Extra keyword arguments forwarded to ``generate()``
                (e.g. ``temperature``, ``top_k``, ``top_p``).

        Returns:
            The assistant's response text (without prompt or history).

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
        """
        # ─────────────────────────────────────────────────────────────────────
        # GGUF MODEL: Use native chat completion (handles templates properly)
        # ─────────────────────────────────────────────────────────────────────
        if getattr(self, '_is_gguf', False) and hasattr(self.model, 'chat'):
            # Build messages list for chat API
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": message})
            
            try:
                # Use native chat completion - handles Qwen/Llama/etc templates
                temperature = kwargs.get('temperature', 0.8)
                response = self.model.chat(
                    messages=messages,
                    max_tokens=max_gen,
                    temperature=temperature,
                    top_p=kwargs.get('top_p', 0.9),
                    top_k=kwargs.get('top_k', 50),
                )
                return response
            except Exception as e:
                logger.warning(f"GGUF chat failed, falling back to generate: {e}")
                # Fall through to standard generation
        
        # ─────────────────────────────────────────────────────────────────────
        # TRUNCATE HISTORY TO PREVENT HALLUCINATIONS
        # This is critical! Without this, long conversations overflow the
        # context window and cause the model to hallucinate.
        # ─────────────────────────────────────────────────────────────────────
        if auto_truncate and history:
            history = self._truncate_history(
                history,
                current_message=message,
                system_prompt=system_prompt,
                reserve_for_response=max_gen
            )
        
        # Use centralized prompt builder for consistent formatting
        try:
            from .prompt_builder import get_prompt_builder
            builder = get_prompt_builder()
            full_prompt = builder.build_chat_prompt(
                message=message,
                history=history,
                system_prompt=system_prompt,
                include_generation_prefix=True
            )
            stop_strings = builder.get_stop_sequences()
        except ImportError:
            # Fallback to inline prompt building
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

        # Generate
        response = self.generate(
            full_prompt,
            max_gen=max_gen,
            stop_strings=stop_strings,
            **kwargs
        )

        # Extract assistant's response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

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
            raise RuntimeError("universal_router module is not installed — cannot use chat_with_tools")

        # Create a chat function that preserves history/system prompt
        def chat_fn(msg, **kw):
            return self.chat(
                msg, 
                history=history, 
                system_prompt=system_prompt,
                max_gen=max_gen, 
                **kwargs
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
        max_gen: int = 200,
        **kwargs
    ) -> Generator[str]:
        """
        Stream chat-style generation.

        Args:
            message: User's message
            history: Conversation history
            system_prompt: Optional system prompt
            max_gen: Maximum tokens
            **kwargs: Additional parameters

        Yields:
            Generated tokens one at a time
        """
        # Use centralized prompt builder for consistent formatting
        try:
            from .prompt_builder import get_prompt_builder
            builder = get_prompt_builder()
            full_prompt = builder.build_chat_prompt(
                message=message,
                history=history,
                system_prompt=system_prompt,
                include_generation_prefix=True
            )
            stop_strings = builder.get_stop_sequences()
        except ImportError:
            # Fallback to inline prompt building
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
            stop_strings = ["\nUser:", "\n\n"]

        # Stream generation
        buffer = ""
        for token in self.stream_generate(full_prompt, max_gen=max_gen, **kwargs):
            buffer += token

            # Check for stop conditions
            stopped = False
            for stop in stop_strings:
                if stop in buffer:
                    buffer = buffer[:buffer.find(stop)]
                    stopped = True
                    break
            
            if stopped:
                break

            yield token

    # =========================================================================
    # 🛠️ Tool-Aware Generation
    # =========================================================================
    
    def generate_with_tools(
        self,
        prompt: str,
        module_manager=None,
        max_gen: int = 200,
        max_tool_iterations: int = 5,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        include_system_prompt: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text with tool execution support.
        
        The AI can invoke tools during generation, and the results are fed back
        for continued generation. This enables the AI to:
          - Generate images when asked
          - Control avatar expressions
          - Search the web for information
          - Read/write files
          - And more
        
        Args:
            prompt: Input text or user query
            module_manager: ModuleManager instance for tool access
            max_gen: Maximum tokens per generation step
            max_tool_iterations: Maximum number of tool calls in sequence
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty
            include_system_prompt: Prepend tool usage instructions
            **kwargs: Additional generation parameters
            
        Returns:
            Complete generated text with tool results
        """
        try:
            from .tool_interface import ToolInterface
            from .tool_prompts import get_tool_enabled_system_prompt
        except ImportError:
            raise RuntimeError(
                "tool_interface / tool_prompts modules are not installed — "
                "generate_with_tools is not available yet"
            )

        # Create tool interface
        tool_interface = ToolInterface(module_manager)
        
        # Prepend system prompt if requested
        if include_system_prompt:
            system_prompt = get_tool_enabled_system_prompt()
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        else:
            full_prompt = prompt
        
        # Generate with tool support
        current_prompt = full_prompt
        full_output = ""
        iterations = 0
        
        while iterations < max_tool_iterations:
            # Generate next chunk
            output = self.generate(
                current_prompt,
                max_gen=max_gen,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                use_cache=True
            )
            
            # Extract new content (remove the prompt if it's in the output)
            if current_prompt in output:
                new_content = output[len(current_prompt):]
            else:
                new_content = output
            
            # Check for tool calls in the new content
            tool_call = tool_interface.parse_tool_call(new_content)
            
            if tool_call:
                # Execute the tool
                result = tool_interface.execute_tool(tool_call)
                result_str = tool_interface.format_tool_result(result)
                
                # Append tool call and result to output
                full_output += new_content[:tool_call.end_pos - tool_call.start_pos]
                full_output += "\n" + result_str + "\n"
                
                # Update prompt for next iteration
                current_prompt = full_prompt + full_output
                iterations += 1
                
                # Continue generation after tool result
                continue
            else:
                # No tool call found, we're done
                full_output += new_content
                break
        
        return full_output
    
    def stream_generate_with_tools(
        self,
        prompt: str,
        module_manager=None,
        max_gen: int = 200,
        max_tool_iterations: int = 5,
        include_system_prompt: bool = True,
        **kwargs
    ) -> Generator[str]:
        """
        Stream generation with tool execution support.
        
        Yields tokens as they're generated, pausing for tool execution
        when tool calls are detected.
        
        Args:
            prompt: Input text
            module_manager: ModuleManager for tool access
            max_gen: Maximum tokens per step
            max_tool_iterations: Maximum tool calls
            include_system_prompt: Include tool instructions
            **kwargs: Additional parameters
            
        Yields:
            Generated tokens, including tool results
        """
        try:
            from .tool_interface import ToolInterface
            from .tool_prompts import get_tool_enabled_system_prompt
        except ImportError:
            raise RuntimeError(
                "tool_interface / tool_prompts modules are not installed — "
                "stream_generate_with_tools is not available yet"
            )
        
        tool_interface = ToolInterface(module_manager)
        
        if include_system_prompt:
            system_prompt = get_tool_enabled_system_prompt()
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        else:
            full_prompt = prompt
        
        current_prompt = full_prompt
        buffer = ""
        iterations = 0
        
        while iterations < max_tool_iterations:
            # Stream generate
            for token in self.stream_generate(current_prompt, max_gen=max_gen, **kwargs):
                buffer += token
                yield token
                
                # Check if we have a complete tool call
                if '<|tool_end|>' in buffer:
                    tool_call = tool_interface.parse_tool_call(buffer)
                    if tool_call:
                        # Execute tool
                        result = tool_interface.execute_tool(tool_call)
                        result_str = tool_interface.format_tool_result(result)
                        
                        # Yield result
                        yield "\n" + result_str + "\n"
                        
                        # Update prompt
                        current_prompt = full_prompt + buffer + "\n" + result_str + "\n"
                        buffer = ""
                        iterations += 1
                        break
            else:
                # Generation completed without tool call
                break
