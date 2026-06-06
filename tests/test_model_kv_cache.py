"""KV-cache correctness for the from-scratch Enigma — the serving path.

The model's ``generate``/``generate_stream`` decode token-by-token with
``use_cache=True`` (prefill once, then one new token per step at an
advancing ``start_pos``). For that to be *correct*, the logits produced by
the incremental cached path must equal the logits from a full no-cache
recompute over the same realized sequence — otherwise served output silently
diverges from the model's true distribution.

These tests lock that equivalence (deep model coverage that was removed in
the Modkit refocus; see HANDOFF.md "Restore Forge test coverage"). They run on
CPU with a nano preset so they're fast and GPU-independent, and they exercise
GQA (nano = 4 query heads / 2 KV heads) plus both position schemes
(RoPE and learned positional embeddings).
"""

from dataclasses import replace

import pytest
import torch

from enigma_engine.core.model import Enigma
from enigma_engine.core.model_presets import MODEL_PRESETS


def _tiny(use_rope: bool = True, seed: int = 0) -> Enigma:
    """A small, deterministic, CPU Enigma in eval mode (dropout off)."""
    torch.manual_seed(seed)
    cfg = replace(
        MODEL_PRESETS["nano"],
        vocab_size=64,
        max_seq_len=64,
        use_rope=use_rope,
    )
    # Sanity: the preset we lean on must keep n_kv_heads < n_heads so the
    # cache↔GQA-repeat interaction is actually under test.
    assert cfg.n_kv_heads < cfg.n_heads, "nano preset must use GQA for this test"
    return Enigma(cfg).eval()


@pytest.mark.parametrize("use_rope", [True, False], ids=["rope", "learned_pos"])
@torch.no_grad()
def test_kv_cache_decode_matches_no_cache(use_rope: bool):
    """Incremental cached decode == full no-cache recompute, logit-for-logit.

    Walk a realized sequence one token at a time: at each step compare the
    cached path's next-token logits against recomputing the whole sequence
    from scratch with ``use_cache=False``. They must agree to within float32
    numerical noise at *every* step (prefill + each decode step).
    """
    model = _tiny(use_rope=use_rope)
    vocab = model.config.vocab_size
    torch.manual_seed(1)
    prefix = torch.randint(0, vocab, (1, 5))

    def ref_last_logits(seq: torch.Tensor) -> torch.Tensor:
        return model(seq, use_cache=False)[0, -1]

    # Prefill: cache the prompt, compare against a plain forward over it.
    model.clear_cache()
    cached = model(prefix, use_cache=True)[0, -1]
    seq = prefix.clone()
    assert torch.allclose(cached, ref_last_logits(seq), atol=1e-4), "prefill diverged"

    # Decode: feed each new token (argmax → deterministic) with an advancing
    # start_pos, exactly as Enigma.generate does, and re-verify each step.
    for step in range(1, 9):
        nxt = cached.argmax().view(1, 1)
        seq = torch.cat([seq, nxt], dim=1)
        cached = model(nxt, use_cache=True, start_pos=seq.shape[1] - 1)[0, -1]
        ref = ref_last_logits(seq)
        assert torch.allclose(cached, ref, atol=1e-4), (
            f"decode step {step} (len {seq.shape[1]}) diverged: "
            f"max|Δ|={(cached - ref).abs().max().item():.2e}"
        )
        assert cached.argmax() == ref.argmax(), f"argmax disagrees at step {step}"


@torch.no_grad()
def test_generate_is_deterministic_and_clear_cache_resets():
    """``generate`` is greedy-deterministic and ``clear_cache`` fully resets state.

    Two back-to-back greedy generations (top_k=1) from the same prompt must
    produce byte-identical token sequences — proving the per-call
    ``clear_cache()`` wipes the previous run's KV entries (a stale-cache leak
    would corrupt the second call)."""
    model = _tiny()
    prompt = torch.tensor([[1, 7, 11, 3]])

    out1 = model.generate(prompt, max_new_tokens=12, temperature=1.0, top_k=1)
    out2 = model.generate(prompt, max_new_tokens=12, temperature=1.0, top_k=1)

    assert out1.shape[0] == 1 and out1.shape[1] > prompt.shape[1], "nothing generated"
    assert torch.equal(out1, out2), "greedy generate is not deterministic / cache leaked between runs"
    # The prompt is preserved verbatim as the prefix of the output.
    assert torch.equal(out1[:, : prompt.shape[1]], prompt)


@torch.no_grad()
def test_generate_respects_stop_token():
    """Generation halts as soon as a stop token is emitted."""
    model = _tiny()
    prompt = torch.tensor([[1, 7, 11, 3]])

    # Discover the first token greedy-generation would emit, then make THAT a
    # stop token: generation should append it and immediately halt.
    full = model.generate(prompt, max_new_tokens=12, temperature=1.0, top_k=1)
    first_gen = int(full[0, prompt.shape[1]].item())

    stopped = model.generate(
        prompt, max_new_tokens=12, temperature=1.0, top_k=1, stop_tokens=[first_gen]
    )
    assert stopped.shape[1] == prompt.shape[1] + 1, "did not stop on the stop token"
    assert int(stopped[0, -1].item()) == first_gen
