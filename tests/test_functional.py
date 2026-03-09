"""
Functional tests for Enigma Engine.

These tests go beyond import checks — they exercise real logic:
- Create a tiny model and generate tokens
- Tokenizer encode/decode round-trips
- KV-cache update/get
- Command execution
- TrainingConfig validation
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

    def test_parse_param_target_billions(self):
        """parse_param_target handles '8b', '1.5b', '70b' etc."""
        from enigma_engine.core.model_presets import parse_param_target

        assert parse_param_target("8b") == 8_000_000_000
        assert parse_param_target("1.5b") == 1_500_000_000
        assert parse_param_target("70B") == 70_000_000_000
        assert parse_param_target("0.5b") == 500_000_000

    def test_parse_param_target_millions(self):
        """parse_param_target handles '500m', '27m' etc."""
        from enigma_engine.core.model_presets import parse_param_target

        assert parse_param_target("500m") == 500_000_000
        assert parse_param_target("27M") == 27_000_000
        assert parse_param_target("85m") == 85_000_000

    def test_parse_param_target_raw_number(self):
        """parse_param_target handles plain numbers like '8000000000'."""
        from enigma_engine.core.model_presets import parse_param_target

        assert parse_param_target("8000000000") == 8_000_000_000
        assert parse_param_target("500000000") == 500_000_000

    def test_parse_param_target_invalid(self):
        """parse_param_target returns None for invalid input."""
        from enigma_engine.core.model_presets import parse_param_target

        assert parse_param_target("") is None
        assert parse_param_target("abc") is None
        assert parse_param_target("b8") is None

    def test_config_for_param_target_returns_preset(self):
        """config_for_param_target returns a name and ForgeConfig."""
        from enigma_engine.core.model_presets import (
            config_for_param_target, ForgeConfig)

        name, config = config_for_param_target(7_000_000_000)
        assert isinstance(name, str)
        assert isinstance(config, ForgeConfig)
        assert config.dim > 0
        assert config.n_layers > 0

    def test_config_for_param_target_small(self):
        """config_for_param_target matches small targets to small presets."""
        from enigma_engine.core.model_presets import config_for_param_target

        name, config = config_for_param_target(27_000_000)
        # Should pick a small preset, not a giant one
        assert config.dim <= 1024

    def test_config_for_param_target_even_head_dim(self):
        """config_for_param_target always produces even head_dim (RoPE needs it)."""
        from enigma_engine.core.model_presets import config_for_param_target

        # Test a range of targets and vocab sizes, including combos that
        # previously produced odd head_dim (e.g. 1B with vocab=2181).
        test_cases = [
            (1_000_000_000, 2181),
            (1_000_000_000, 32000),
            (500_000_000, 5000),
            (2_000_000_000, 10000),
            (300_000_000, 1500),
            (50_000_000, 256),
            (10_000_000_000, 32000),
        ]
        for target, vocab in test_cases:
            _, config = config_for_param_target(target, vocab_size=vocab)
            head_dim = config.dim // config.n_heads
            assert head_dim % 2 == 0, (
                f"Odd head_dim={head_dim} for target={target}, "
                f"vocab={vocab} (dim={config.dim}, n_heads={config.n_heads})"
            )


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


# ── Router ──────────────────────────────────────────────────────────────────


class TestRouterFunctional:
    """Test ModRouter prompt logic without starting the server."""

    def test_get_set_prompt(self):
        """set_prompt/get_prompt round-trip."""
        from enigma_engine.router import ModRouter

        router = ModRouter(enable_training=False)
        router.set_prompt("test_purpose", "Be concise.")
        assert router.get_prompt("test_purpose") == "Be concise."

    def test_combined_prompt(self):
        """get_combined_prompt merges multiple purposes."""
        from enigma_engine.router import ModRouter

        router = ModRouter(enable_training=False)
        combined = router.get_combined_prompt("chat", "safety")
        assert "helpful" in combined.lower()
        assert "harmless" in combined.lower()

    def test_status_without_start(self):
        """get_status should work even before start()."""
        from enigma_engine.router import ModRouter

        router = ModRouter(enable_training=False)
        status = router.get_status()
        assert status["running"] is False
        assert status["connected_mods"] == 0


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
            # Default prompt comes from data/prompts/chat.md or builtin
            assert len(ctx.system_prompt) > 10
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

    # ── Identity card fields ────────────────────────────────────────

    def test_identity_defaults(self):
        """New ModelContext has identity defaults: display_name, etc."""
        import enigma_engine.core.model_context as mc

        ctx = mc.ModelContext("identity_test")
        assert ctx.display_name == ""
        assert ctx.personality == ""
        assert ctx.avatar == ""
        assert isinstance(ctx.created_at, str)
        assert ctx.total_messages == 0
        assert ctx.total_sessions == 0
        assert ctx.training_history == []
        assert ctx.tags == []
        assert ctx.notes == ""

    def test_identity_save_load_round_trip(self, tmp_path):
        """Identity fields survive save/load cycle."""
        import enigma_engine.core.model_context as mc

        original = mc._CONTEXTS_DIR
        mc._CONTEXTS_DIR = tmp_path
        try:
            ctx = mc.ModelContext("identity_rt")
            ctx.display_name = "Enigma"
            ctx.personality = "Helpful and direct assistant"
            ctx.avatar = "data/avatar/images/default.png"
            ctx.total_messages = 42
            ctx.total_sessions = 5
            ctx.tags = ["coding", "general"]
            ctx.notes = "Fine-tuned on Python documentation"
            ctx.training_history = [
                {"date": "2026-03-02", "mode": "Self Study",
                 "epochs": 10, "best_loss": 0.42}
            ]
            ctx.save()

            ctx2 = mc.ModelContext("identity_rt")
            ctx2.load()
            assert ctx2.display_name == "Enigma"
            assert ctx2.personality == "Helpful and direct assistant"
            assert ctx2.avatar == "data/avatar/images/default.png"
            assert ctx2.total_messages == 42
            assert ctx2.total_sessions == 5
            assert ctx2.tags == ["coding", "general"]
            assert ctx2.notes == "Fine-tuned on Python documentation"
            assert len(ctx2.training_history) == 1
            assert ctx2.training_history[0]["mode"] == "Self Study"
        finally:
            mc._CONTEXTS_DIR = original

    def test_migration_from_old_context(self, tmp_path):
        """Loading old context.json (without identity) preserves base fields."""
        import enigma_engine.core.model_context as mc

        original = mc._CONTEXTS_DIR
        mc._CONTEXTS_DIR = tmp_path
        try:
            # Write old-format context.json
            ctx_dir = tmp_path / "old_model"
            ctx_dir.mkdir()
            (ctx_dir / "context.json").write_text(json.dumps({
                "model_key": "old_model",
                "system_prompt": "Be helpful.",
                "config": {"temperature": 0.8},
                "last_used": 1000.0,
            }), encoding="utf-8")

            ctx = mc.ModelContext("old_model")
            ctx.load()
            # Base fields preserved
            assert ctx.system_prompt == "Be helpful."
            assert ctx.config == {"temperature": 0.8}
            assert ctx.last_used == 1000.0
            # Identity fields are defaults
            assert ctx.display_name == ""
            assert ctx.total_messages == 0
            assert ctx.tags == []
        finally:
            mc._CONTEXTS_DIR = original

    def test_increment_messages(self, tmp_path):
        """increment_messages bumps total_messages count."""
        import enigma_engine.core.model_context as mc

        original = mc._CONTEXTS_DIR
        mc._CONTEXTS_DIR = tmp_path
        try:
            ctx = mc.ModelContext("msg_counter")
            assert ctx.total_messages == 0
            ctx.increment_messages()
            assert ctx.total_messages == 1
            ctx.increment_messages(5)
            assert ctx.total_messages == 6
        finally:
            mc._CONTEXTS_DIR = original

    def test_increment_sessions(self, tmp_path):
        """increment_sessions bumps total_sessions count."""
        import enigma_engine.core.model_context as mc

        original = mc._CONTEXTS_DIR
        mc._CONTEXTS_DIR = tmp_path
        try:
            ctx = mc.ModelContext("session_counter")
            assert ctx.total_sessions == 0
            ctx.increment_sessions()
            assert ctx.total_sessions == 1
        finally:
            mc._CONTEXTS_DIR = original

    def test_record_training_run(self, tmp_path):
        """record_training_run appends to training_history."""
        import enigma_engine.core.model_context as mc

        original = mc._CONTEXTS_DIR
        mc._CONTEXTS_DIR = tmp_path
        try:
            ctx = mc.ModelContext("train_rec")
            ctx.record_training_run(
                mode="Self Study", epochs=10, best_loss=0.42)
            assert len(ctx.training_history) == 1
            entry = ctx.training_history[0]
            assert entry["mode"] == "Self Study"
            assert entry["epochs"] == 10
            assert entry["best_loss"] == 0.42
            assert "date" in entry

            # Add another
            ctx.record_training_run(
                mode="DPO", epochs=5, best_loss=0.31)
            assert len(ctx.training_history) == 2
        finally:
            mc._CONTEXTS_DIR = original

    def test_memory_fact_count(self, tmp_path):
        """memory_fact_count reads from PersistentMemory if available."""
        import enigma_engine.core.model_context as mc

        ctx = mc.ModelContext("fact_count_test")
        # Without a memory file, should return 0
        count = ctx.memory_fact_count
        assert isinstance(count, int)
        assert count >= 0

    def test_identity_in_list_contexts(self, tmp_path):
        """list_model_contexts includes identity fields."""
        import enigma_engine.core.model_context as mc

        original = mc._CONTEXTS_DIR
        mc._CONTEXTS_DIR = tmp_path
        try:
            ctx = mc.ModelContext("listed_model")
            ctx.display_name = "My AI"
            ctx.tags = ["helper"]
            ctx.total_messages = 100
            ctx.save()

            contexts = mc.list_model_contexts()
            assert len(contexts) == 1
            entry = contexts[0]
            assert entry["display_name"] == "My AI"
            assert entry["tags"] == ["helper"]
            assert entry["total_messages"] == 100
        finally:
            mc._CONTEXTS_DIR = original

    def test_session_path_default_empty(self):
        """New ModelContext has empty session_path."""
        import enigma_engine.core.model_context as mc
        ctx = mc.ModelContext("sp_test")
        assert ctx.session_path == ""

    def test_session_path_save_load(self, tmp_path):
        """session_path survives save/load round trip."""
        import enigma_engine.core.model_context as mc

        original = mc._CONTEXTS_DIR
        mc._CONTEXTS_DIR = tmp_path
        try:
            ctx = mc.ModelContext("sp_roundtrip")
            ctx.session_path = "memory/session_20260304_120000_1.json"
            ctx.save()

            ctx2 = mc.ModelContext("sp_roundtrip")
            ctx2.load()
            assert ctx2.session_path == (
                "memory/session_20260304_120000_1.json")
        finally:
            mc._CONTEXTS_DIR = original

    def test_old_context_missing_session_path(self, tmp_path):
        """Loading old context.json without session_path defaults to empty."""
        import enigma_engine.core.model_context as mc

        original = mc._CONTEXTS_DIR
        mc._CONTEXTS_DIR = tmp_path
        try:
            ctx_dir = tmp_path / "old_sp"
            ctx_dir.mkdir()
            (ctx_dir / "context.json").write_text(json.dumps({
                "model_key": "old_sp",
                "system_prompt": "Hello",
                "config": {},
                "last_used": 1000.0,
            }), encoding="utf-8")

            ctx = mc.ModelContext("old_sp")
            ctx.load()
            assert ctx.session_path == ""
        finally:
            mc._CONTEXTS_DIR = original


# ── CLI flags ────────────────────────────────────────────────────────────────


class TestCLIFlags:
    """Verify run.py has correct CLI arguments and security defaults."""

    def test_serve_defaults_to_localhost(self):
        """run_serve must default to 127.0.0.1, not 0.0.0.0."""
        import inspect
        import importlib
        run = importlib.import_module("run")
        source = inspect.getsource(run.run_serve)
        # Must NOT have host="0.0.0.0" as the default
        assert 'host="0.0.0.0"' not in source, (
            "run_serve must not default to 0.0.0.0 — exposes API to network")

    def test_serve_accepts_host_and_api_key(self):
        """run_serve must accept host and api_key parameters."""
        import inspect
        import importlib
        run = importlib.import_module("run")
        sig = inspect.signature(run.run_serve)
        assert "host" in sig.parameters, (
            "run_serve missing host parameter")
        assert "api_key" in sig.parameters, (
            "run_serve missing api_key parameter")

    def test_chat_accepts_profile_and_temperature(self):
        """run_chat must accept profile and temperature parameters."""
        import inspect
        import importlib
        run = importlib.import_module("run")
        sig = inspect.signature(run.run_chat)
        assert "profile" in sig.parameters, (
            "run_chat missing profile parameter")
        assert "temperature" in sig.parameters, (
            "run_chat missing temperature parameter")

    def test_cli_has_host_argument(self):
        """CLI parser must have --host argument."""
        import importlib
        run = importlib.import_module("run")
        source = open("run.py", encoding="utf-8").read()
        assert "--host" in source, (
            "CLI missing --host argument for server bind address")

    def test_cli_has_api_key_argument(self):
        """CLI parser must have --api-key argument."""
        source = open("run.py", encoding="utf-8").read()
        assert "--api-key" in source, (
            "CLI missing --api-key argument for server authentication")

    def test_cli_has_profile_argument(self):
        """CLI parser must have --profile argument."""
        source = open("run.py", encoding="utf-8").read()
        assert "--profile" in source, (
            "CLI missing --profile argument for chat")

    def test_cli_train_uses_atomic_save(self):
        """CLI train must use atomic_torch_save, not raw torch.save."""
        source = open("run.py", encoding="utf-8").read()
        assert "atomic_torch_save" in source, (
            "run.py --train should use atomic_torch_save")
