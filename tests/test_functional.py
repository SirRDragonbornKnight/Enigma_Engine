"""
Functional tests for Enigma Engine.

These tests go beyond import checks — they exercise real logic:
- Create a tiny model and generate tokens
- Tokenizer encode/decode round-trips
- KV-cache update/get
- Command execution
- TrainingConfig validation
- AI profile JSON serialization
"""

import json
import sys
from pathlib import Path

import pytest
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Model creation & generation ─────────────────────────────────────────────


class TestModelCreation:
    """Test creating a tiny model and running basic ops."""

    def test_create_nano_model(self):
        """create_model('nano') should return a working Module."""
        from enigma_engine.core.model import create_model

        model = create_model("nano", vocab_size=256)
        assert isinstance(model, torch.nn.Module)
        assert model.num_parameters > 0

    def test_forward_pass(self):
        """A single forward pass should return logits of correct shape."""
        from enigma_engine.core.model import create_model

        model = create_model("nano", vocab_size=256)
        model.eval()

        input_ids = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            logits = model(input_ids)

        # logits shape: [batch, seq_len, vocab_size]
        assert logits.shape == (1, 8, 256)

    def test_generate_tokens(self):
        """model.generate() should produce more tokens than the input."""
        from enigma_engine.core.model import create_model

        model = create_model("nano", vocab_size=256)
        model.eval()

        input_ids = torch.randint(0, 256, (1, 4))
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=5, temperature=1.0)

        # Output should be longer than input
        assert output.shape[1] > input_ids.shape[1]
        # But not more than input + max_new_tokens
        assert output.shape[1] <= input_ids.shape[1] + 5

    def test_model_presets_available(self):
        """list_presets() should return a non-empty dict."""
        from enigma_engine.core.model_presets import list_presets

        presets = list_presets()
        assert isinstance(presets, dict)
        assert "tiny" in presets
        assert "small" in presets


# ── Tokenizer round-trips ───────────────────────────────────────────────────


class TestTokenizer:
    """Test tokenizer encode → decode round-trip."""

    def test_simple_tokenizer_round_trip(self):
        """encode → decode should recover the original text (approximately)."""
        from enigma_engine.core.tokenizer import SimpleTokenizer

        tok = SimpleTokenizer()
        text = "hello world"
        ids = tok.encode(text)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        decoded = tok.decode(ids)
        # SimpleTokenizer is char-level; decoded should contain the original words
        assert "hello" in decoded
        assert "world" in decoded

    def test_empty_string(self):
        """Encoding an empty string should not crash."""
        from enigma_engine.core.tokenizer import SimpleTokenizer

        tok = SimpleTokenizer()
        ids = tok.encode("")
        decoded = tok.decode(ids)
        # Should get at most special tokens back
        assert isinstance(decoded, str)

    def test_special_tokens(self):
        """SimpleTokenizer should have BOS/EOS ids."""
        from enigma_engine.core.tokenizer import SimpleTokenizer

        tok = SimpleTokenizer()
        assert hasattr(tok, "bos_token_id")
        assert hasattr(tok, "eos_token_id")
        assert isinstance(tok.bos_token_id, int)
        assert isinstance(tok.eos_token_id, int)


# ── KV-Cache ────────────────────────────────────────────────────────────────


class TestKVCache:
    """Test KVCache update/get cycle."""

    def test_update_and_get(self):
        """Updating the cache and reading back should return the same data."""
        from enigma_engine.core.kv_cache import KVCache

        cache = KVCache(
            batch_size=1,
            max_seq_len=32,
            n_kv_heads=2,
            head_dim=16,
            device=torch.device("cpu"),
        )

        # Create a fake key/value for 4 tokens
        k = torch.randn(1, 4, 2, 16)
        v = torch.randn(1, 4, 2, 16)

        cache.update(k, v)
        k_out, v_out = cache.get()

        # Should get back exactly the tokens we put in (first 4 positions)
        assert k_out.shape[1] == 4
        assert v_out.shape[1] == 4
        assert torch.allclose(k_out, k, atol=1e-5)
        assert torch.allclose(v_out, v, atol=1e-5)

    def test_cache_clear(self):
        """After clear, cache should report 0 position."""
        from enigma_engine.core.kv_cache import KVCache

        cache = KVCache(
            batch_size=1,
            max_seq_len=32,
            n_kv_heads=2,
            head_dim=16,
            device=torch.device("cpu"),
        )
        k = torch.randn(1, 2, 2, 16)
        v = torch.randn(1, 2, 2, 16)
        cache.update(k, v)

        cache.clear()
        assert cache.current_pos == 0


# ── Command system ──────────────────────────────────────────────────────────


class TestCommandExecution:
    """Test that registered commands actually execute."""

    def test_system_info_command(self):
        """system.info command should return a successful CommandResult."""
        from enigma_engine.core.commands import get_registry

        registry = get_registry()
        result = registry.execute("system.info")
        assert result.success is True
        assert len(result.message) > 0

    def test_unknown_command(self):
        """Executing a non-existent command should return a failed CommandResult."""
        from enigma_engine.core.commands import get_registry

        registry = get_registry()
        result = registry.execute("does.not.exist")
        assert result.success is False


# ── Training config validation ──────────────────────────────────────────────


class TestTrainingConfig:
    """Test TrainingConfig guardrails added in this session."""

    def test_valid_config(self):
        """Default config should validate without error."""
        from enigma_engine.core.training import TrainingConfig

        config = TrainingConfig()
        config.validate()  # Should not raise

    def test_bad_epochs(self):
        """epochs < 1 should raise ValueError."""
        from enigma_engine.core.training import TrainingConfig

        config = TrainingConfig(epochs=0)
        with pytest.raises(ValueError, match="epochs"):
            config.validate()

    def test_bad_learning_rate(self):
        """learning_rate <= 0 should raise ValueError."""
        from enigma_engine.core.training import TrainingConfig

        config = TrainingConfig(learning_rate=-1e-4)
        with pytest.raises(ValueError, match="learning_rate"):
            config.validate()

    def test_bad_batch_size(self):
        """batch_size < 1 should raise ValueError."""
        from enigma_engine.core.training import TrainingConfig

        config = TrainingConfig(batch_size=0)
        with pytest.raises(ValueError, match="batch_size"):
            config.validate()

    def test_to_dict_has_new_fields(self):
        """to_dict() should include early_stopping and max_loss fields."""
        from enigma_engine.core.training import TrainingConfig

        d = TrainingConfig().to_dict()
        assert "early_stopping_patience" in d
        assert "max_loss" in d
        assert "max_training_seconds" in d


# ── AI profile serialization ────────────────────────────────────────────────


class TestAIProfileSerialization:
    """Test profile JSON round-trip."""

    def test_profile_to_dict_round_trip(self):
        """AIProfile should serialize to dict and back."""
        from enigma_engine.core.ai_profile import AIProfile

        profile = AIProfile(
            id="test",
            name="Test Bot",
            system_prompt="You are helpful.",
        )
        d = profile.to_dict()
        assert d["id"] == "test"
        assert d["name"] == "Test Bot"

        restored = AIProfile.from_dict(d)
        assert restored.id == profile.id
        assert restored.name == profile.name
        assert restored.system_prompt == profile.system_prompt


# ── Router ──────────────────────────────────────────────────────────────────


class TestRouterFunctional:
    """Test BrickRouter prompt logic without starting the server."""

    def test_get_set_prompt(self):
        """set_prompt/get_prompt round-trip."""
        from enigma_engine.router import BrickRouter

        router = BrickRouter(enable_training=False)
        router.set_prompt("test_purpose", "Be concise.")
        assert router.get_prompt("test_purpose") == "Be concise."

    def test_combined_prompt(self):
        """get_combined_prompt merges multiple purposes."""
        from enigma_engine.router import BrickRouter

        router = BrickRouter(enable_training=False)
        combined = router.get_combined_prompt("chat", "safety")
        assert "helpful" in combined.lower()
        assert "harmless" in combined.lower()

    def test_status_without_start(self):
        """get_status should work even before start()."""
        from enigma_engine.router import BrickRouter

        router = BrickRouter(enable_training=False)
        status = router.get_status()
        assert status["running"] is False
        assert status["connected_bricks"] == 0


# ── Per-Model Context ──────────────────────────────────────────────────────


class TestModelContext:
    """Test per-model context save/load/delete cycle."""

    def test_save_and_load_round_trip(self, tmp_path):
        """Context + history should survive save/load cycle."""
        import enigma_engine.core.model_context as mc

        # Redirect storage to tmp
        original = mc._CONTEXTS_DIR
        mc._CONTEXTS_DIR = tmp_path
        try:
            ctx = mc.ModelContext("test_model")
            ctx.system_prompt = "Be creative."
            ctx.config = {"temperature": 0.9}
            ctx.profile_id = "creative_writer"
            ctx.history = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ]
            ctx.save()

            # Load into fresh instance
            ctx2 = mc.ModelContext("test_model")
            ctx2.load()
            assert ctx2.system_prompt == "Be creative."
            assert ctx2.config == {"temperature": 0.9}
            assert ctx2.profile_id == "creative_writer"
            assert len(ctx2.history) == 2
            assert ctx2.history[0]["content"] == "hello"
            assert ctx2.last_used > 0
        finally:
            mc._CONTEXTS_DIR = original

    def test_load_missing_is_noop(self, tmp_path):
        """Loading a nonexistent model context keeps defaults."""
        import enigma_engine.core.model_context as mc

        original = mc._CONTEXTS_DIR
        mc._CONTEXTS_DIR = tmp_path
        try:
            ctx = mc.ModelContext("nonexistent")
            ctx.load()
            assert ctx.system_prompt == "You are a helpful AI assistant."
            assert ctx.history == []
        finally:
            mc._CONTEXTS_DIR = original

    def test_model_key_from_path(self):
        """model_key_from_path strips path to a clean stem."""
        from enigma_engine.core.model_context import model_key_from_path

        assert model_key_from_path("models/enigma_small.pth") == "enigma_small"
        assert model_key_from_path("models/My Model.gguf") == "my_model"
        # Generic stem uses parent dir name
        assert model_key_from_path(
            "models/qwen2.5/model.safetensors") == "qwen2.5"

    def test_clear_history(self, tmp_path):
        """clear_history empties in-memory list."""
        import enigma_engine.core.model_context as mc

        original = mc._CONTEXTS_DIR
        mc._CONTEXTS_DIR = tmp_path
        try:
            ctx = mc.ModelContext("test_clear")
            ctx.history = [{"role": "user", "content": "x"}]
            ctx.clear_history()
            assert ctx.history == []
        finally:
            mc._CONTEXTS_DIR = original

    def test_delete_removes_directory(self, tmp_path):
        """delete() removes the model context directory."""
        import enigma_engine.core.model_context as mc

        original = mc._CONTEXTS_DIR
        mc._CONTEXTS_DIR = tmp_path
        try:
            ctx = mc.ModelContext("to_delete")
            ctx.save()
            assert ctx.context_dir.exists()
            ctx.delete()
            assert not ctx.context_dir.exists()
        finally:
            mc._CONTEXTS_DIR = original

    def test_list_model_contexts(self, tmp_path):
        """list_model_contexts returns saved contexts."""
        import enigma_engine.core.model_context as mc

        original = mc._CONTEXTS_DIR
        mc._CONTEXTS_DIR = tmp_path
        try:
            # Create two contexts
            ctx1 = mc.ModelContext("model_a")
            ctx1.system_prompt = "Prompt A"
            ctx1.history = [{"role": "user", "content": "hi"}]
            ctx1.save()

            ctx2 = mc.ModelContext("model_b")
            ctx2.save()

            contexts = mc.list_model_contexts()
            keys = [c["model_key"] for c in contexts]
            assert "model_a" in keys
            assert "model_b" in keys
            # model_a should have 1 message
            for c in contexts:
                if c["model_key"] == "model_a":
                    assert c["message_count"] == 1
        finally:
            mc._CONTEXTS_DIR = original

    def test_load_model_context_helper(self, tmp_path):
        """load_model_context creates and loads in one call."""
        import enigma_engine.core.model_context as mc

        original = mc._CONTEXTS_DIR
        mc._CONTEXTS_DIR = tmp_path
        try:
            # Save first
            ctx = mc.ModelContext("helper_test")
            ctx.system_prompt = "Testing"
            ctx.save()

            # Use the helper
            loaded = mc.load_model_context(
                "models/helper_test.pth")
            assert loaded.system_prompt == "Testing"
        finally:
            mc._CONTEXTS_DIR = original

    def test_history_validates_messages(self, tmp_path):
        """Invalid messages in history.json are filtered out."""
        import enigma_engine.core.model_context as mc

        original = mc._CONTEXTS_DIR
        mc._CONTEXTS_DIR = tmp_path
        try:
            ctx_dir = tmp_path / "bad_data"
            ctx_dir.mkdir()
            # Write history with some invalid entries
            (ctx_dir / "history.json").write_text(json.dumps({
                "messages": [
                    {"role": "user", "content": "valid"},
                    {"role": "assistant"},  # missing content
                    {"content": "no role"},  # missing role
                    {"role": "user", "content": 123},  # wrong type
                    {"role": "assistant", "content": "also valid"},
                ]
            }), encoding="utf-8")

            ctx = mc.ModelContext("bad_data")
            ctx.load()
            assert len(ctx.history) == 2
            assert ctx.history[0]["content"] == "valid"
            assert ctx.history[1]["content"] == "also valid"
        finally:
            mc._CONTEXTS_DIR = original
