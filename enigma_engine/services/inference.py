"""Inference engine factory and chat-call service.

Wraps :class:`enigma_engine.core.inference.EnigmaEngine` construction and the
public chat / generate entry points. The GUI calls these from background
threads via ``after(0, callback)`` — no concurrency primitives are owned here.

Phase 0 (this slice): signatures only; bodies forward verbatim to core.
"""

from __future__ import annotations

from typing import Any


def create_engine(model: Any, tokenizer: Any, **kwargs: Any) -> Any:
    """Construct an :class:`EnigmaEngine` for the given model + tokenizer.

    All keyword arguments are forwarded verbatim. Phase 0 does NOT enumerate
    every possible kwarg in the service signature — that would couple the
    service to the engine constructor's evolution. ``**kwargs`` passthrough
    is the contract until the first GUI consumer migrates and pins the names
    it actually uses.
    """

    from enigma_engine.core.inference import EnigmaEngine

    return EnigmaEngine(model, tokenizer, **kwargs)


def chat(engine: Any, prompt: str, **kwargs: Any) -> str:
    """One-shot chat call. Forwards to ``engine.chat(prompt, **kwargs)``.

    Streaming variants will land as separate functions (``chat_stream``) once
    the first GUI consumer migrates and exercises the iterator contract.
    """

    return engine.chat(prompt, **kwargs)
