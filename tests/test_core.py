"""
Tests for Enigma Engine core functionality.

Run with: python -m pytest tests/ -v
"""

import inspect
import pytest
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestAIProfile:
    """Test AI profile system."""

    def test_profile_create_and_list(self):
        from enigma_engine.core.ai_profile import AIProfile, AIProfileManager
        profile = AIProfile(
            id="test_profile", name="Test Profile",
            system_prompt="You are a test assistant.")
        assert profile.id == "test_profile"
        assert "test assistant" in profile.system_prompt
        manager = AIProfileManager()
        assert isinstance(manager.list_profiles(), list)


class TestRouter:
    """Test the router module."""

    def test_background_trainer_inference_busy_guard(self):
        """N-25: trainer should defer when inference is active."""
        from enigma_engine.router import BackgroundTrainer

        trainer = BackgroundTrainer()

        trainer.inference_idle_check = lambda: False
        assert trainer._inference_busy() is True

        trainer.inference_idle_check = lambda: True
        assert trainer._inference_busy() is False

        trainer.inference_idle_check = (
            lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        # Fail-open on callback errors so training does not deadlock.
        assert trainer._inference_busy() is False

    def test_router_training_can_toggle_runtime(self, monkeypatch):
        """ModRouter can create and remove its trainer after init."""
        from enigma_engine import router as router_mod

        created = []

        class _DummyTrainer:
            def __init__(self, **kwargs):
                created.append(self)
                self.started = False
                self.stopped = False

            def start(self):
                self.started = True

            def stop(self):
                self.stopped = True

        monkeypatch.setattr(router_mod, "BackgroundTrainer", _DummyTrainer)

        router = router_mod.ModRouter(enable_training=False)
        router.running = True

        router.set_training_enabled(True)

        assert router.trainer is created[0]
        assert created[0].started is True

        router.set_training_enabled(False)

        assert router.trainer is None
        assert created[0].stopped is True

    def test_router_training_toggle_is_locked(self, monkeypatch):
        """S770: concurrent set_training_enabled uses _train_lock."""
        from enigma_engine import router as router_mod
        import threading

        class _DummyTrainer:
            def __init__(self):
                self.started = False
                self.stopped = False

            def start(self):
                self.started = True

            def stop(self):
                self.stopped = True

        monkeypatch.setattr(router_mod, "BackgroundTrainer", _DummyTrainer)
        router = router_mod.ModRouter(enable_training=False)
        router.running = True

        # Rapid concurrent toggles should not orphan trainers

        def toggle():
            router.set_training_enabled(True)
            router.set_training_enabled(False)

        threads = [threading.Thread(target=toggle) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # After all toggles, trainer should be cleanly None
        assert router.trainer is None

    def test_router_inference_idle_check_propagates_to_trainers(self, monkeypatch):
        """N-25: idle callback should propagate to current and future trainer."""
        from enigma_engine import router as router_mod

        class _DummyTrainer:
            def __init__(self, **kwargs):
                self.started = False
                self.stopped = False
                self.inference_idle_check = None

            def start(self):
                self.started = True

            def stop(self):
                self.stopped = True

        monkeypatch.setattr(router_mod, "BackgroundTrainer", _DummyTrainer)
        router = router_mod.ModRouter(enable_training=True)

        def checker() -> bool:
            return True

        router.set_inference_idle_check(checker)
        assert router.trainer is not None
        assert router.trainer.inference_idle_check is checker

        router.set_training_enabled(False)
        router.set_training_enabled(True)
        assert router.trainer is not None
        assert router.trainer.inference_idle_check is checker

    def test_router_passes_repo_anchor_file_to_trainer_when_present(
        self, monkeypatch, tmp_path,
    ):
        """Continuous-3: ModRouter wires the curated anchor JSONL through.

        When `data/anchor_examples.jsonl` exists in the repo, the boot
        path constructs `BackgroundTrainer(anchor_data_path=<that path>)`
        so anchor rehearsal is on by default â€” no hand-edit needed.
        """
        from enigma_engine import router as router_mod

        captured: dict = {}

        class _CapturingTrainer:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.inference_idle_check = None

            def start(self):
                pass

            def stop(self):
                pass

        # Point the default at a temp file that exists
        anchor_file = tmp_path / "anchor_examples.jsonl"
        anchor_file.write_text('{"prompt": "x", "response": "y"}\n', encoding="utf-8")
        monkeypatch.setattr(router_mod, "_DEFAULT_ANCHOR_PATH", anchor_file)
        monkeypatch.setattr(router_mod, "BackgroundTrainer", _CapturingTrainer)

        router_mod.ModRouter(enable_training=True)

        assert captured.get("anchor_data_path") == anchor_file, (
            f"Router must forward repo anchor file when it exists; "
            f"got kwargs={captured!r}"
        )

    def test_router_passes_none_when_anchor_file_missing(
        self, monkeypatch, tmp_path,
    ):
        """Continuous-3: missing anchor file â†’ `None`, not a phantom path.

        The user can ship without `data/anchor_examples.jsonl` and
        `BackgroundTrainer` will fall back to recent-only replay
        without WARNING noise on every boot.
        """
        from enigma_engine import router as router_mod

        captured: dict = {}

        class _CapturingTrainer:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.inference_idle_check = None

            def start(self):
                pass

            def stop(self):
                pass

        missing = tmp_path / "does_not_exist.jsonl"
        monkeypatch.setattr(router_mod, "_DEFAULT_ANCHOR_PATH", missing)
        monkeypatch.setattr(router_mod, "BackgroundTrainer", _CapturingTrainer)

        router_mod.ModRouter(enable_training=True)

        assert captured.get("anchor_data_path") is None, (
            f"Router must pass None when default anchor file is absent; "
            f"got kwargs={captured!r}"
        )

    def test_router_explicit_anchor_path_overrides_default(
        self, monkeypatch, tmp_path,
    ):
        """Continuous-3: caller-supplied anchor path wins over repo default."""
        from enigma_engine import router as router_mod

        captured: dict = {}

        class _CapturingTrainer:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.inference_idle_check = None

            def start(self):
                pass

            def stop(self):
                pass

        # Repo default exists, but user supplies a different path
        repo_default = tmp_path / "repo_default.jsonl"
        repo_default.write_text("{}\n", encoding="utf-8")
        explicit = tmp_path / "user_choice.jsonl"
        explicit.write_text("{}\n", encoding="utf-8")
        monkeypatch.setattr(router_mod, "_DEFAULT_ANCHOR_PATH", repo_default)
        monkeypatch.setattr(router_mod, "BackgroundTrainer", _CapturingTrainer)

        router_mod.ModRouter(enable_training=True, anchor_data_path=str(explicit))

        assert captured.get("anchor_data_path") == str(explicit), (
            f"Explicit anchor_data_path must override repo default; "
            f"got kwargs={captured!r}"
        )
    """Test that packaging config is correct."""

    def test_no_setup_py(self):
        """setup.py should be deleted â€” pyproject.toml is the single source."""
        assert not (PROJECT_ROOT / "setup.py").exists()


@pytest.mark.structural
class TestDeadImports:
    """Verify dead imports have been removed.

    Uses ruff F401 (unused imports) rule on critical modules.
    This is more robust than string-matching source lines â€” ruff
    understands Python scoping, __all__ re-exports, and type-only
    usage.
    """

    _CRITICAL_MODULES = [
        "enigma_engine/core/engine_chat.py",
        "enigma_engine/core/engine_generation.py",
        "enigma_engine/core/commands.py",
        "enigma_engine/training/training.py",
        "enigma_engine/core/inference.py",
        "enigma_engine/api/server.py",
    ]

    def test_no_unused_imports_in_critical_modules(self):
        """Critical modules should have zero unused imports (ruff F401)."""
        import subprocess
        root = Path(__file__).parent.parent
        result = subprocess.run(
            ["ruff", "check", "--select", "F401", "--no-fix", "--quiet"]
            + self._CRITICAL_MODULES,
            capture_output=True, text=True, cwd=str(root),
        )
        if result.returncode != 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            pytest.fail(
                "Unused imports found in critical modules:\n"
                + "\n".join(lines)
            )


class TestSourceEncodingHygiene:
    """Guard against replacement-character mojibake in source files."""

    def test_no_replacement_character_in_engine_sources(self):
        """Engine Python files should not contain U+FFFD replacement chars."""
        root = PROJECT_ROOT / "enigma_engine"
        bad_paths: list[str] = []
        for file_path in root.rglob("*.py"):
            text = file_path.read_text(encoding="utf-8")
            if "\ufffd" in text:
                bad_paths.append(str(file_path.relative_to(PROJECT_ROOT)))

        assert not bad_paths, (
            "Found replacement-character mojibake in source files: "
            + ", ".join(sorted(bad_paths))
        )


class TestImageGenIntegration:
    """Verify image generation command and chat integration."""

    def test_imagegen_generate_registered(self):
        """imagegen.generate command must have proper metadata."""
        from enigma_engine.core.commands import get_registry, Command
        registry = get_registry()
        assert "imagegen.generate" in registry._commands
        cmd = registry._commands["imagegen.generate"]
        assert isinstance(cmd, Command)
        assert cmd.name == "imagegen.generate"
        assert callable(cmd.handler)
        assert len(cmd.description) > 0

    def test_imagegen_status_registered(self):
        """imagegen.status command must have proper metadata."""
        from enigma_engine.core.commands import get_registry, Command
        registry = get_registry()
        assert "imagegen.status" in registry._commands
        cmd = registry._commands["imagegen.status"]
        assert isinstance(cmd, Command)
        assert cmd.name == "imagegen.status"
        assert callable(cmd.handler)
        assert len(cmd.description) > 0

    def test_imagegen_generate_requires_prompt(self):
        """imagegen.generate with no args should return error."""
        from enigma_engine.core.commands import get_registry
        registry = get_registry()
        result = registry.execute("imagegen.generate")
        assert not result.success
        assert "Usage" in result.message or "required" in result.message

    def test_imagegen_status_runs(self):
        """imagegen.status should return OK even with no backends."""
        from enigma_engine.core.commands import get_registry
        registry = get_registry()
        result = registry.execute("imagegen.status")
        assert result.success
        assert "backend" in result.message.lower()


class TestImageGenServiceDefaults:
    """2.1-imagegen slice (May 25 2026): standalone service default flipped
    placeholder -> local with loud-on-load-failure semantics."""

    @staticmethod
    def _load_imagegen_module():
        import importlib.util
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent / "mods" / "imagegen" / "imagegen.py"
        spec = importlib.util.spec_from_file_location("imagegen_service", path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_default_provider_is_local(self):
        """Constructor default must be 'local' (was 'placeholder' pre-slice)."""
        mod = self._load_imagegen_module()
        service = mod.ImageGen()
        assert service.default_provider == "local", (
            f"default_provider regressed to {service.default_provider!r}; "
            "2.1-imagegen flipped it to 'local' so SD is the out-of-box behaviour")

    def test_load_failure_does_not_silently_fallback(self):
        """When local provider fails to load (no diffusers / no weights),
        _cmd_generate must surface the failure, NOT silently switch to
        placeholder. Honors \u00a74 'loud-on-real-issue'."""
        mod = self._load_imagegen_module()
        service = mod.ImageGen()
        # Force the local provider into a known-fail state
        local = service.get_provider("local")
        assert local is not None
        local.load = lambda: False  # simulate ImportError / weights missing
        result = service._cmd_generate({"prompt": "test"})
        assert result.get("success") is False
        # Error message must name the failing provider so the operator can
        # diagnose (not the placeholder, not a generic message)
        assert "local" in result.get("error", "").lower(), (
            f"load-failure error must name the failing provider, got: {result!r}")


class TestVideoGenServiceDefaults:
    """2.1-videogen slice (May 25 2026): swapped AnimateDiff for CogVideoX-5B,
    flipped default builtin -> local, killed LocalVideo._fallback masquerade."""

    @staticmethod
    def _load_videogen_module():
        import importlib.util
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent / "mods" / "videogen" / "videogen.py"
        spec = importlib.util.spec_from_file_location("videogen_service", path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_default_provider_is_local(self):
        """Constructor default must be 'local' (was 'builtin' pre-slice)."""
        mod = self._load_videogen_module()
        service = mod.VideoGen()
        assert service.default_provider == "local", (
            f"default_provider regressed to {service.default_provider!r}; "
            "2.1-videogen flipped it to 'local' so CogVideoX is the out-of-box behaviour")

    def test_local_has_no_silent_fallback_attribute(self):
        """LocalVideo must NOT carry a BuiltinVideo fallback handle. The old
        AnimateDiff implementation hid load failures behind self._fallback,
        promoting placeholder GIFs as a successful 'local' run."""
        mod = self._load_videogen_module()
        local = mod.LocalVideo()
        assert not hasattr(local, "_fallback"), (
            "LocalVideo._fallback re-introduces the silent-fallback "
            "anti-pattern fixed by 2.1-videogen")

    def test_load_failure_does_not_silently_fallback(self):
        """When the local provider fails to load, _cmd_generate must surface
        the failure, NOT promote BuiltinVideo behind the caller's back."""
        mod = self._load_videogen_module()
        service = mod.VideoGen()
        local = service.get_provider("local")
        assert local is not None
        local.load = lambda: False  # simulate CogVideoX weights/deps missing
        result = service._cmd_generate({"prompt": "ocean waves"})
        assert result.get("success") is False
        assert "local" in result.get("error", "").lower(), (
            f"load-failure error must name the failing provider, got: {result!r}")

    def test_local_uses_cogvideox_not_animatediff(self):
        """LocalVideo.load() must wire CogVideoXPipeline, not the obsolete
        AnimateDiffPipeline. Behavioural gate via injected fake diffusers."""
        import sys
        import types
        mod = self._load_videogen_module()

        captured = {}

        class FakePipe:
            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                captured["model_id"] = model_id
                captured["dtype"] = kwargs.get("torch_dtype")
                inst = cls()
                inst.vae = types.SimpleNamespace(
                    enable_tiling=lambda: None, enable_slicing=lambda: None,
                )
                return inst

            def enable_sequential_cpu_offload(self):
                captured["offload"] = True

            def to(self, dev):
                captured["to_dev"] = dev
                return self

        fake_diffusers = types.ModuleType("diffusers")
        fake_diffusers.CogVideoXPipeline = FakePipe
        # Guard: if the code regressed to AnimateDiff, the import would fail
        # because we deliberately do NOT define AnimateDiffPipeline.
        original = sys.modules.get("diffusers")
        sys.modules["diffusers"] = fake_diffusers
        try:
            local = mod.LocalVideo()
            ok = local.load()
        finally:
            if original is not None:
                sys.modules["diffusers"] = original
            else:
                sys.modules.pop("diffusers", None)

        assert ok is True, "CogVideoX load path should succeed with fake pipeline"
        assert "CogVideoX" in captured.get("model_id", ""), (
            f"LocalVideo must load a CogVideoX model, got {captured.get('model_id')!r}")


class TestThreeDServiceDefaults:
    """2.1-threed slice (May 25 2026): standalone service default flipped
    builtin -> local; Local3DGen no longer silently falls back to Builtin3DGen."""

    @staticmethod
    def _load_threed_module():
        import importlib.util
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent / "mods" / "threed" / "threed.py"
        spec = importlib.util.spec_from_file_location("threed_service", path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_default_provider_is_local(self):
        """Constructor default must be 'local' (was 'builtin' pre-slice)."""
        mod = self._load_threed_module()
        service = mod.ThreeD()
        assert service.default_provider == "local", (
            f"default_provider regressed to {service.default_provider!r}; "
            "2.1-threed flipped it to 'local' so Shap-E is the out-of-box behaviour")

    def test_local_has_no_silent_fallback_attribute(self):
        """Local3DGen must NOT carry a Builtin3DGen fallback handle. The old
        implementation hid Shap-E load failures behind self._fallback, so
        callers couldn't tell when they were getting placeholder cubes."""
        mod = self._load_threed_module()
        local = mod.Local3DGen()
        assert not hasattr(local, "_fallback"), (
            "Local3DGen._fallback re-introduces the silent-fallback "
            "anti-pattern fixed by 2.1-threed")

    def test_load_failure_does_not_silently_fallback(self):
        """When local provider fails to load, _cmd_generate must surface the
        failure, NOT promote builtin behind the caller's back."""
        mod = self._load_threed_module()
        service = mod.ThreeD()
        local = service.get_provider("local")
        assert local is not None
        local.load = lambda: False  # simulate Shap-E weights/deps missing
        result = service._cmd_generate({"prompt": "a cube"})
        assert result.get("success") is False
        assert "local" in result.get("error", "").lower(), (
            f"load-failure error must name the failing provider, got: {result!r}")


class TestAiProfileCallableType:
    """AIProfileManager must accept callables for callbacks."""

    def test_profile_manager_has_callback_attributes(self):
        """AIProfileManager should have callable callback slots."""
        from enigma_engine.core.ai_profile import AIProfileManager
        manager = AIProfileManager()
        # Callbacks should exist as attributes and accept callables
        assert hasattr(manager, 'on_profile_loaded')
        assert hasattr(manager, 'on_profile_switched')
        # Should be settable to a callable
        manager.on_profile_loaded = lambda p: None
        assert callable(manager.on_profile_loaded)


class TestModelPresetsValidate:
    """validate() must work on frozen configs without raising."""

    def test_validate_does_not_modify_frozen_config(self):
        """validate() on a valid frozen config must return True without error."""
        from enigma_engine.core.model_presets import ForgeConfig
        config = ForgeConfig(
            vocab_size=100, dim=64, n_layers=2,
            n_heads=2, max_seq_len=128)
        # Frozen dataclass â€” any assignment would raise FrozenInstanceError
        result = config.validate()
        assert result is True

    def test_construction_rejects_invalid_config(self):
        """ForgeConfig must reject invalid values at construction time."""
        from enigma_engine.core.model_presets import ForgeConfig
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            ForgeConfig(
                vocab_size=0, dim=64, n_layers=2,
                n_heads=2, max_seq_len=128)


@pytest.mark.structural
class TestPolishAuditCore:
    """Verify polish fixes made to core engine files."""

    def test_hardware_profile_has_hardware_type(self):
        """HardwareProfile must expose hardware_type, not cpu_model."""
        from enigma_engine.core.hardware_detection import HardwareProfile
        profile = HardwareProfile()
        assert hasattr(profile, 'hardware_type'), (
            "HardwareProfile must have hardware_type attribute")
        assert not hasattr(profile, 'cpu_model'), (
            "HardwareProfile should not have cpu_model â€” use hardware_type")

    def test_hardware_profile_has_to_dict(self):
        """HardwareProfile should have a to_dict() method."""
        from enigma_engine.core.hardware_detection import HardwareProfile
        profile = HardwareProfile()
        result = profile.to_dict()
        assert isinstance(result, dict)
        assert "device" in result
        assert "ram_gb" in result


    def test_inference_engine_has_from_model(self):
        """EnigmaEngine must have a from_model factory method."""
        from enigma_engine.core.inference import EnigmaEngine
        assert hasattr(EnigmaEngine, 'from_model'), (
            "EnigmaEngine must have from_model factory")
        assert callable(EnigmaEngine.from_model)


class TestAIProfileFromDictMutation:
    """Test that AIProfile.from_dict does not mutate the input dict."""

    def test_from_dict_no_mutation(self):
        """from_dict should not modify the caller's dict."""
        from enigma_engine.core.ai_profile import AIProfile
        data = {
            "id": "test", "name": "Test",
            "system_prompt": "You are helpful.",
            "generation": {"temperature": 0.7},
        }
        original_gen = data["generation"]
        AIProfile.from_dict(data)
        # Original dict's 'generation' key should still be a plain dict
        assert data["generation"] is original_gen
        assert isinstance(data["generation"], dict)

    def test_to_dict_returns_plain_dict(self):
        """to_dict should return a plain dict via asdict."""
        from enigma_engine.core.ai_profile import AIProfile
        profile = AIProfile(id="t", name="T", system_prompt="Hello")
        d = profile.to_dict()
        assert isinstance(d, dict)
        assert d["id"] == "t"
        # Nested configs should be plain dicts, not dataclass instances
        assert isinstance(d.get("generation", {}), dict)


# =========================================================================
# Item 17 â€” Mixed Async/Sync Mods (standardised on threading)
# =========================================================================


class TestRouterStopLogging:
    """Verify router.stop() handles errors gracefully."""

    def test_stop_handles_errors_without_crashing(self):
        """stop() must not crash even if mods raise exceptions."""
        from enigma_engine.router import ModRouter
        router = ModRouter()
        # stop() on a clean router should complete without error
        router.stop()
        # Calling stop() twice should also be safe
        router.stop()


# =============================================================================
# VISION ENCODER TESTS
# =============================================================================

class TestWebUtilsCore:
    """Core tests for web_utils shared module."""

    def test_extract_preserves_paragraph_text(self):
        """extract_html_text preserves <p> content."""
        from enigma_engine.core.web_utils import extract_html_text
        html = "<div><p>First paragraph text here</p><p>Second paragraph</p></div>"
        result = extract_html_text(html)
        assert "First paragraph text here" in result
        assert "Second paragraph" in result

    def test_extract_skips_short_fragments(self):
        """extract_html_text ignores very short text fragments."""
        from enigma_engine.core.web_utils import extract_html_text
        html = "<div><span>Ok</span><p>This is a real sentence with content</p></div>"
        result = extract_html_text(html)
        # "Ok" is <= 2 chars, should be skipped
        assert "This is a real sentence" in result

    def test_headers_constant_exists(self):
        """Module defines _HEADERS for requests."""
        from enigma_engine.core.web_utils import _HEADERS
        assert "User-Agent" in _HEADERS


# ---------------------------------------------------------------------------
# Model class annotations
# ---------------------------------------------------------------------------

class TestAutoResearch:
    """Tests for enigma_engine.core.auto_research module."""

    def test_should_auto_research_short_query(self):
        """Short queries should not trigger research."""
        from enigma_engine.core.auto_research import should_auto_research
        assert not should_auto_research("")
        assert not should_auto_research("hi")
        assert not should_auto_research("ok")

    def test_should_auto_research_greetings(self):
        """Simple greetings should not trigger research."""
        from enigma_engine.core.auto_research import should_auto_research
        assert not should_auto_research("hello")
        assert not should_auto_research("thanks")
        assert not should_auto_research("bye")

    def test_should_auto_research_questions(self):
        """Questions should trigger research."""
        from enigma_engine.core.auto_research import should_auto_research
        assert should_auto_research("what is machine learning?")
        assert should_auto_research("how to train a neural network?")
        assert should_auto_research("who is Alan Turing?")

    def test_should_auto_research_question_mark(self):
        """Messages ending with ? should trigger research."""
        from enigma_engine.core.auto_research import should_auto_research
        assert should_auto_research(
            "can you tell me about quantum computing?")

    def test_should_auto_research_code_skip(self):
        """Code-only messages should not trigger research."""
        from enigma_engine.core.auto_research import should_auto_research
        assert not should_auto_research("```python\nprint('hello')\n```")
        assert not should_auto_research("def my_function():")
        assert not should_auto_research("class MyClass:")

    def test_should_auto_research_keywords(self):
        """Messages with research keywords should trigger."""
        from enigma_engine.core.auto_research import should_auto_research
        assert should_auto_research("explain the difference between CPU and GPU")
        assert should_auto_research("what are the latest trends in AI?")
        assert should_auto_research("compare python and javascript")

    def test_auto_research_empty_query(self):
        """auto_research returns empty string for empty query."""
        from enigma_engine.core.auto_research import auto_research
        assert auto_research("") == ""
        assert auto_research("ab") == ""

    # ----------------------------------------------------------------
    # AutoResearch-2 Stage A â€” post-generation uncertainty gate
    # (R-UNPREDICT-1, Pass 146 spec â†’ Pass 153 build)
    # Signal-driven, deterministic â€” no RNG.
    # ----------------------------------------------------------------

    def test_score_uncertainty_confident_response(self):
        """Confident factual response should score low."""
        from enigma_engine.core.auto_research import score_uncertainty
        r = score_uncertainty(
            "what is the capital of France",
            "The capital of France is Paris. It has been the "
            "capital since the 10th century.",
        )
        assert r.score < 0.3

    def test_score_uncertainty_hedge_phrases_drive_score_up(self):
        """Multiple hedge phrases push score above retry threshold."""
        from enigma_engine.core.auto_research import score_uncertainty
        r = score_uncertainty(
            "what is the population of Mars in 2026?",
            "I'm not sure, but I think it might be around several "
            "thousand. I don't know the exact number.",
        )
        assert r.score >= 0.55
        assert any("hedge" in s for s in r.reasons)

    def test_score_uncertainty_refusal_pattern(self):
        """Apology / refusal phrases trigger a high score."""
        from enigma_engine.core.auto_research import score_uncertainty
        r = score_uncertainty(
            "what are the latest quantum computing breakthroughs?",
            "I apologize, I don't have information on that topic.",
        )
        assert r.score >= 0.55
        assert any("refusal" in s for s in r.reasons)

    def test_score_uncertainty_empty_response(self):
        """Empty response is maximally uncertain."""
        from enigma_engine.core.auto_research import score_uncertainty
        r = score_uncertainty("a real question", "")
        assert r.score == 1.0
        assert "empty_response" in r.reasons

    def test_score_uncertainty_short_response_long_query(self):
        """Short reply to a substantive question contributes uncertainty."""
        from enigma_engine.core.auto_research import score_uncertainty
        r = score_uncertainty(
            "Can you explain in detail how transformers handle "
            "positional encoding for long contexts?",
            "Yes.",
        )
        assert "short_response" in r.reasons

    def test_score_uncertainty_deterministic(self):
        """Same input must produce same score â€” no RNG."""
        from enigma_engine.core.auto_research import score_uncertainty
        args = ("what is X?", "I'm not sure, I think it might be Y.")
        scores = [score_uncertainty(*args).score for _ in range(5)]
        assert len(set(scores)) == 1

    def test_should_retry_with_research_confident_skips(self):
        """Confident response â†’ no retry."""
        from enigma_engine.core.auto_research import should_retry_with_research
        assert not should_retry_with_research(
            "what is 2 plus 2?",
            "2 plus 2 equals 4. This is basic arithmetic.",
        )

    def test_should_retry_with_research_hedges_triggers(self):
        """Hedge-heavy response â†’ retry fires."""
        from enigma_engine.core.auto_research import should_retry_with_research
        assert should_retry_with_research(
            "what was the Q3 2025 GDP figure?",
            "I'm not sure, I don't know that. I think it might be "
            "something but I cannot say for certain.",
        )

    def test_should_retry_with_research_off_switch(self):
        """enabled=False suppresses retry even when score is high."""
        from enigma_engine.core.auto_research import should_retry_with_research
        assert not should_retry_with_research(
            "x",
            "I don't know. I'm not sure. I apologize, I'm unable.",
            enabled=False,
        )

    def test_should_retry_with_research_threshold_configurable(self):
        """Threshold tunable; same response straddles different cutoffs."""
        from enigma_engine.core.auto_research import should_retry_with_research
        response = (
            "I think this might be correct but I'm not sure."
        )
        assert should_retry_with_research(
            "explain it", response, threshold=0.4
        )
        assert not should_retry_with_research(
            "explain it", response, threshold=0.9
        )


# ================================================================
# GUI Command Registration (mod.start, mod.stop, mod.list)
# ================================================================

class TestTensorReshapeSafety:
    """Ensure model forward pass produces correct output shapes."""

    def test_forward_output_shape(self):
        """Model forward must return logits with shape (batch, seq, padded_vocab)."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.model import Enigma
        config = ForgeConfig(
            vocab_size=50, dim=32, n_layers=1, n_heads=2,
            max_seq_len=16)
        model = Enigma(config=config)
        model.eval()
        x = torch.randint(0, 50, (1, 4))
        with torch.no_grad():
            logits = model(x)
        # Vocab is padded to nearest 64 for GPU alignment
        padded_vocab = (config.vocab_size + 63) & ~63
        assert logits.shape == (1, 4, padded_vocab), (
            f"Expected (1, 4, {padded_vocab}), got {logits.shape}")
        assert logits.shape[2] >= config.vocab_size, (
            "Output dim must cover entire vocabulary")


# ================================================================
# Training data parser: multi-format support
# ================================================================

class TestOptimizerBetasConsistency:
    """All optimizer creation sites must use LM-friendly betas."""

    def test_reward_trainer_config_has_betas(self):
        from enigma_engine.core.rl_training import RewardTrainerConfig
        cfg = RewardTrainerConfig()
        assert cfg.adam_beta1 == 0.9
        assert cfg.adam_beta2 == 0.95
        assert cfg.adam_eps == 1e-8

    def test_rlhf_config_has_betas(self):
        from enigma_engine.core.rl_training import RLHFConfig
        cfg = RLHFConfig()
        assert cfg.adam_beta1 == 0.9
        assert cfg.adam_beta2 == 0.95
        assert cfg.adam_eps == 1e-8

    def test_selfplay_config_has_betas(self):
        from enigma_engine.core.rl_training import SelfPlayConfig
        cfg = SelfPlayConfig()
        assert cfg.adam_beta1 == 0.9
        assert cfg.adam_beta2 == 0.95
        assert cfg.adam_eps == 1e-8

    def test_lora_trainer_stores_betas(self):
        import torch.nn as nn
        from unittest.mock import MagicMock, patch
        with patch("enigma_engine.core.lora_utils.create_lora_model",
                   side_effect=lambda m, c: m):
            from enigma_engine.core.lora_utils import LoraTrainer
            model = nn.Linear(4, 4)
            tok = MagicMock()
            trainer = LoraTrainer(model, tok, adam_beta2=0.999)
            assert trainer.adam_beta2 == 0.999


# ================================================================
# Auto Research Cache, Rate Limiting, Parallel Fetch (#25)
# ================================================================

class TestAutoResearchCache:
    """Tests for auto_research LRU cache and rate limiting."""

    def test_normalize_query_strips_whitespace(self):
        """_normalize_query collapses whitespace and lowercases."""
        from enigma_engine.core.auto_research import _normalize_query
        assert _normalize_query("  Hello  World ") == "hello world"
        assert _normalize_query("FOO") == "foo"

    def test_cache_put_get_roundtrip(self):
        """_cache_put and _cache_get work together."""
        from enigma_engine.core import auto_research
        # Save and restore state
        old_cache = auto_research._search_cache.copy()
        try:
            auto_research._search_cache.clear()
            auto_research._cache_put("test_key", "test_value")
            assert auto_research._cache_get("test_key") == "test_value"
            assert auto_research._cache_get("missing") is None
        finally:
            auto_research._search_cache.clear()
            auto_research._search_cache.update(old_cache)

    def test_cache_evicts_oldest_at_max(self):
        """LRU eviction drops oldest entry when cache is full."""
        from enigma_engine.core import auto_research
        old_cache = auto_research._search_cache.copy()
        old_max = auto_research._CACHE_MAX
        try:
            auto_research._search_cache.clear()
            auto_research._CACHE_MAX = 3
            auto_research._cache_put("a", "1")
            auto_research._cache_put("b", "2")
            auto_research._cache_put("c", "3")
            # Cache is full â€” inserting "d" should evict "a"
            auto_research._cache_put("d", "4")
            assert auto_research._cache_get("a") is None
            assert auto_research._cache_get("d") == "4"
            assert len(auto_research._search_cache) == 3
        finally:
            auto_research._CACHE_MAX = old_max
            auto_research._search_cache.clear()
            auto_research._search_cache.update(old_cache)

    def test_rate_limit_blocks_rapid_calls(self):
        """_check_rate_limit blocks successive calls within interval."""
        from enigma_engine.core import auto_research
        old_time = auto_research._last_search_time
        try:
            auto_research._last_search_time = 0.0
            # First call should pass
            assert auto_research._check_rate_limit() is True
            # Immediate second call should be blocked
            assert auto_research._check_rate_limit() is False
        finally:
            auto_research._last_search_time = old_time

    def test_auto_research_uses_cache(self):
        """Repeated identical queries return cached result."""
        from enigma_engine.core import auto_research

        old_cache = auto_research._search_cache.copy()
        old_time = auto_research._last_search_time
        try:
            auto_research._search_cache.clear()
            auto_research._last_search_time = 0.0

            # Pre-fill cache
            key = auto_research._normalize_query("test cache query")
            auto_research._cache_put(key, "cached result")

            # Should return cached without doing any web search
            result = auto_research.auto_research("test cache query")
            assert result == "cached result"
        finally:
            auto_research._search_cache.clear()
            auto_research._search_cache.update(old_cache)
            auto_research._last_search_time = old_time

    def test_cache_max_constant(self):
        """Cache max is 100."""
        from enigma_engine.core.auto_research import _CACHE_MAX
        assert _CACHE_MAX == 100

    def test_min_search_interval_constant(self):
        """Min search interval is 5 seconds."""
        from enigma_engine.core.auto_research import _MIN_SEARCH_INTERVAL
        assert _MIN_SEARCH_INTERVAL == 5.0


# ================================================================
# KV Cache Clone Safety (#26)
# ================================================================

class TestRunPyLazyTorch:
    """Tests for run.py not importing torch at top level."""

    def test_no_top_level_torch_import(self):
        """run.py must not have a top-level 'import torch' statement."""
        import ast
        run_path = Path(__file__).parent.parent / "run.py"
        source = run_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "torch", (
                        f"Top-level 'import torch' at line {node.lineno}"
                    )
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "torch", (
                    f"Top-level 'from torch' at line {node.lineno}"
                )
            elif isinstance(node, ast.Try):
                # Check try body for top-level torch imports
                for child in node.body:
                    if isinstance(child, ast.Import):
                        for alias in child.names:
                            assert alias.name != "torch", (
                                f"Top-level 'import torch' in try block "
                                f"at line {child.lineno}"
                            )


def test_memory_search_finds_matches():
    '''Test that memory.search finds matching facts.'''
    from enigma_engine.core.builtin_commands import register_builtin_commands
    from enigma_engine.core.commands import CommandRegistry
    registry = CommandRegistry()
    register_builtin_commands(registry)
    
    # Add multiple facts
    facts = [
        "User's name is Alice",
        "User prefers Python",
        "User works at NASA",
        "User likes coffee"
    ]
    for fact in facts:
        registry.execute(f"memory.remember {fact}")
    
    # Search for "Python"
    result = registry.execute("memory.search Python")
    assert result.success
    assert "Python" in result.message
    assert "prefers Python" in result.message
    
    # Search for "name"
    result = registry.execute("memory.search name")
    assert result.success
    assert "Alice" in result.message


def test_memory_search_no_matches():
    '''Test memory.search when no facts match.'''
    from enigma_engine.core.builtin_commands import register_builtin_commands
    from enigma_engine.core.commands import CommandRegistry
    registry = CommandRegistry()
    register_builtin_commands(registry)
    
    registry.execute("memory.remember User's name is Bob")
    
    # Search for something that doesn't exist
    result = registry.execute("memory.search JavaScript")
    assert result.success
    assert "No memories found" in result.message


def test_memory_search_case_insensitive():
    '''Test memory.search is case-insensitive.'''
    from enigma_engine.core.builtin_commands import register_builtin_commands
    from enigma_engine.core.commands import CommandRegistry
    registry = CommandRegistry()
    register_builtin_commands(registry)
    
    registry.execute("memory.remember User prefers PYTHON language")
    
    # Search with different case
    result = registry.execute("memory.search python")
    assert result.success
    assert "PYTHON" in result.message


# ================================================================
# Audio Encoder â€” Whisper-style Conv1d + Transformer encoder
# ================================================================


# ================================================================
# Auto-Image in Web Research
# ================================================================


# ================================================================
# CLI Flags
# ================================================================

class TestCLIFlags:
    """Verify CLI flags and audio presets."""

    def test_presets_are_configs(self):
        """Each preset must be an AudioEncoderConfig."""
        from enigma_engine.core.audio_encoder import AudioEncoderConfig, AUDIO_PRESETS
        for name, config in AUDIO_PRESETS.items():
            assert isinstance(config, AudioEncoderConfig), f"{name} not AudioEncoderConfig"

    def test_presets_dims_increase(self):
        """Larger presets should have larger dimensions."""
        from enigma_engine.core.audio_encoder import AUDIO_PRESETS
        assert AUDIO_PRESETS["tiny"].dim < AUDIO_PRESETS["base"].dim
        assert AUDIO_PRESETS["base"].dim < AUDIO_PRESETS["small"].dim


class TestSentimentHeuristics:
    """Test the heuristic sentiment analysis module."""

    def test_analyze_positive(self):
        """Positive messages return positive valence."""
        from enigma_engine.core.sentiment import analyze_sentiment
        result = analyze_sentiment("I love this! Thank you so much, amazing work!")
        assert result["valence"] > 0.0

    def test_analyze_negative(self):
        """Negative messages return negative valence."""
        from enigma_engine.core.sentiment import analyze_sentiment
        result = analyze_sentiment("This is terrible, I hate it. Awful experience.")
        assert result["valence"] < 0.0

    def test_analyze_neutral(self):
        """Neutral messages return near-zero valence."""
        from enigma_engine.core.sentiment import analyze_sentiment
        result = analyze_sentiment("What time is it?")
        assert -0.3 <= result["valence"] <= 0.3

    def test_arousal_exclamation(self):
        """Messages with exclamation marks score higher arousal."""
        from enigma_engine.core.sentiment import analyze_sentiment
        calm = analyze_sentiment("That is nice.")
        excited = analyze_sentiment("That is nice!!!")
        assert excited["arousal"] > calm["arousal"]

    def test_arousal_caps(self):
        """ALL CAPS messages score higher arousal."""
        from enigma_engine.core.sentiment import analyze_sentiment
        normal = analyze_sentiment("this is great")
        caps = analyze_sentiment("THIS IS GREAT")
        assert caps["arousal"] > normal["arousal"]

    def test_engagement_question(self):
        """Questions indicate higher engagement."""
        from enigma_engine.core.sentiment import analyze_sentiment
        statement = analyze_sentiment("okay")
        question = analyze_sentiment("Can you tell me more about that?")
        assert question["engagement"] > statement["engagement"]

    def test_engagement_long_message(self):
        """Longer messages indicate higher engagement."""
        from enigma_engine.core.sentiment import analyze_sentiment
        short = analyze_sentiment("ok")
        long = analyze_sentiment(
            "I've been thinking about this for a while and I have several "
            "ideas I'd like to discuss with you in detail.")
        assert long["engagement"] > short["engagement"]

    def test_frustration_signals(self):
        """Frustration keywords boost frustration score."""
        from enigma_engine.core.sentiment import analyze_sentiment
        calm = analyze_sentiment("How do I do this?")
        frustrated = analyze_sentiment(
            "This doesn't work again! I already tried that!")
        assert frustrated["frustration"] > calm["frustration"]

    def test_return_keys(self):
        """analyze_sentiment returns all 5 state keys."""
        from enigma_engine.core.sentiment import analyze_sentiment
        result = analyze_sentiment("Hello there")
        expected_keys = {"valence", "arousal", "engagement", "trust", "frustration"}
        assert set(result.keys()) == expected_keys

    def test_values_in_range(self):
        """All returned values are within expected ranges."""
        from enigma_engine.core.sentiment import analyze_sentiment
        for text in ["I love you!", "I hate this", "ok", "AAAARGH!!!"]:
            result = analyze_sentiment(text)
            assert -1.0 <= result["valence"] <= 1.0
            assert 0.0 <= result["arousal"] <= 1.0
            assert 0.0 <= result["engagement"] <= 1.0
            assert 0.0 <= result["trust"] <= 1.0
            assert 0.0 <= result["frustration"] <= 1.0

    def test_empty_input(self):
        """Empty string returns neutral baseline."""
        from enigma_engine.core.sentiment import analyze_sentiment
        result = analyze_sentiment("")
        assert result["valence"] == 0.0
        assert result["arousal"] == 0.0

    def test_trust_polite_language(self):
        """Polite language indicates higher trust."""
        from enigma_engine.core.sentiment import analyze_sentiment
        rude = analyze_sentiment("just do it")
        polite = analyze_sentiment("Please help me with this, thank you!")
        assert polite["trust"] >= rude["trust"]

    def test_negation_flips_positive(self):
        """'not happy' should score negative, not positive (VADER-style)."""
        from enigma_engine.core.sentiment import analyze_sentiment
        result = analyze_sentiment("I am not happy about this")
        assert result["valence"] < 0.0

    def test_negation_flips_negative(self):
        """'not bad' should score positive, not negative."""
        from enigma_engine.core.sentiment import analyze_sentiment
        result = analyze_sentiment("That was not bad at all")
        assert result["valence"] > 0.0

    def test_double_negation_stays_positive(self):
        """'not not happy' (even negator count) stays positive."""
        from enigma_engine.core.sentiment import analyze_sentiment
        result = analyze_sentiment("I am not not happy")
        assert result["valence"] > 0.0

    def test_whitespace_only_returns_neutral(self):
        """Whitespace-only input returns neutral baseline like empty."""
        from enigma_engine.core.sentiment import analyze_sentiment
        result = analyze_sentiment("   \t\n  ")
        assert result["valence"] == 0.0
        assert result["arousal"] == 0.0


# ================================================================
# Emotional State in ModelContext
# ================================================================

class TestEmotionalState:
    """Test emotional state integration in ModelContext."""

    def test_emotional_state_defaults(self):
        """Emotional state starts at neutral baseline."""
        from enigma_engine.core.model_context import ModelContext
        ctx = ModelContext("test_model")
        state = ctx.emotional_state
        assert state["valence"] == 0.0
        assert state["arousal"] == 0.2
        assert state["engagement"] == 0.5
        assert state["trust"] == 0.5
        assert state["frustration"] == 0.0

    def test_update_emotional_state(self):
        """update_emotional_state changes values based on sentiment."""
        from enigma_engine.core.model_context import ModelContext
        ctx = ModelContext("test_model")
        initial_valence = ctx.emotional_state["valence"]
        ctx.update_emotional_state("I love this! You are amazing!")
        assert ctx.emotional_state["valence"] > initial_valence

    def test_emotional_state_clamped(self):
        """Values stay within their defined ranges after many updates."""
        from enigma_engine.core.model_context import ModelContext
        ctx = ModelContext("test_model")
        # Extreme positive
        for _ in range(50):
            ctx.update_emotional_state("AMAZING! WONDERFUL! BEST THING EVER!!!")
        assert ctx.emotional_state["valence"] <= 1.0
        assert ctx.emotional_state["arousal"] <= 1.0

    def test_decay_toward_baseline(self):
        """decay_emotional_state moves values toward baseline."""
        from enigma_engine.core.model_context import ModelContext
        ctx = ModelContext("test_model")
        # Push state to extreme
        ctx.emotional_state["valence"] = 0.9
        ctx.emotional_state["frustration"] = 0.8
        ctx.decay_emotional_state()
        assert ctx.emotional_state["valence"] < 0.9
        assert ctx.emotional_state["frustration"] < 0.8

    def test_emotional_state_persists(self, tmp_path):
        """Emotional state survives save/load cycle."""
        import json
        from enigma_engine.core.model_context import ModelContext
        # Save
        ctx = ModelContext("test_emo")
        # Override the context_dir
        ctx_dir = tmp_path / "test_emo"
        ctx_dir.mkdir()
        ctx.emotional_state["valence"] = 0.7
        ctx.emotional_state["frustration"] = 0.3
        # Write manually to test persistence
        data = {
            "model_key": "test_emo",
            "system_prompt": "test",
            "config": {},
            "last_used": 0.0,
            "emotional_state": ctx.emotional_state,
        }
        (ctx_dir / "context.json").write_text(
            json.dumps(data), encoding="utf-8")
        # Load into new context
        ctx2 = ModelContext("test_emo")
        # Patch path to read from tmp
        import enigma_engine.core.model_context as mc_mod
        orig_dir = mc_mod._CONTEXTS_DIR
        mc_mod._CONTEXTS_DIR = tmp_path
        try:
            ctx2.load()
        finally:
            mc_mod._CONTEXTS_DIR = orig_dir
        assert abs(ctx2.emotional_state["valence"] - 0.7) < 0.01
        assert abs(ctx2.emotional_state["frustration"] - 0.3) < 0.01

    def test_reset_emotional_state(self):
        """reset_emotional_state returns to baseline."""
        from enigma_engine.core.model_context import ModelContext
        ctx = ModelContext("test_model")
        ctx.emotional_state["valence"] = 0.9
        ctx.emotional_state["frustration"] = 0.8
        ctx.reset_emotional_state()
        assert ctx.emotional_state["valence"] == 0.0
        assert ctx.emotional_state["frustration"] == 0.0

    def test_export_includes_emotional_state(self):
        """export_identity includes emotional_state."""
        from enigma_engine.core.model_context import ModelContext
        ctx = ModelContext("test_model")
        ctx.emotional_state["valence"] = 0.5
        export = ctx.export_identity()
        assert "emotional_state" in export
        assert export["emotional_state"]["valence"] == 0.5

    def test_load_survives_corrupt_emotional_state(self, tmp_path):
        """Pass 156z9em: non-numeric emotional_state value must not crash load.

        Sibling-boundary fix: _load_context wraps load in
        try/except (JSONDecodeError, OSError), but float(saved_emo[key])
        could raise ValueError/TypeError on corrupt data, propagating
        out of load() instead of degrading to baseline like
        _load_history skips bad rows.
        """
        import json
        from enigma_engine.core.model_context import ModelContext
        import enigma_engine.core.model_context as mc_mod

        ctx_dir = tmp_path / "corrupt_emo"
        ctx_dir.mkdir()
        # Non-numeric emotional state value — would raise ValueError
        # inside the load try/except pre-fix.
        data = {
            "model_key": "corrupt_emo",
            "system_prompt": "preserved system prompt",
            "config": {},
            "last_used": 0.0,
            "emotional_state": {"valence": "not_a_number"},
        }
        (ctx_dir / "context.json").write_text(
            json.dumps(data), encoding="utf-8")

        ctx = ModelContext("corrupt_emo")
        orig_dir = mc_mod._CONTEXTS_DIR
        mc_mod._CONTEXTS_DIR = tmp_path
        try:
            ctx.load()  # Must NOT raise
        finally:
            mc_mod._CONTEXTS_DIR = orig_dir

        # Emotional state degraded to baseline, rest of context intact.
        assert ctx.emotional_state["valence"] == 0.0
        assert ctx.system_prompt == "preserved system prompt"


# =====================================================================
# Phase 3: State-Aware Generation
# =====================================================================

class TestBuildEmotionalPromptHint:
    """Tests for build_emotional_prompt_hint()."""

    def test_neutral_state_returns_empty(self):
        """Neutral/baseline state produces no hint (no injection)."""
        from enigma_engine.core.sentiment import build_emotional_prompt_hint
        from enigma_engine.core.model_context import _EMOTIONAL_BASELINE
        hint = build_emotional_prompt_hint(dict(_EMOTIONAL_BASELINE))
        assert hint == ""

    def test_high_frustration_produces_hint(self):
        """High frustration should mention directness/bluntness."""
        from enigma_engine.core.sentiment import build_emotional_prompt_hint
        from enigma_engine.core.model_context import _EMOTIONAL_BASELINE
        state = dict(_EMOTIONAL_BASELINE)
        state["frustration"] = 0.8
        hint = build_emotional_prompt_hint(state)
        assert hint  # Non-empty
        assert "direct" in hint.lower() or "blunt" in hint.lower()

    def test_low_valence_low_trust_guarded(self):
        """Low valence + low trust should suggest guarded/cautious tone."""
        from enigma_engine.core.sentiment import build_emotional_prompt_hint
        from enigma_engine.core.model_context import _EMOTIONAL_BASELINE
        state = dict(_EMOTIONAL_BASELINE)
        state["valence"] = -0.6
        state["trust"] = 0.1
        hint = build_emotional_prompt_hint(state)
        assert hint
        assert "cautious" in hint.lower() or "guarded" in hint.lower()

    def test_high_engagement_high_arousal_exploratory(self):
        """High engagement + arousal should suggest exploratory tone."""
        from enigma_engine.core.sentiment import build_emotional_prompt_hint
        from enigma_engine.core.model_context import _EMOTIONAL_BASELINE
        state = dict(_EMOTIONAL_BASELINE)
        state["engagement"] = 0.9
        state["arousal"] = 0.8
        hint = build_emotional_prompt_hint(state)
        assert hint
        assert "expan" in hint.lower() or "explor" in hint.lower() or "elaborate" in hint.lower()

    def test_positive_valence_high_trust_warm(self):
        """High positive valence + trust should suggest warmth."""
        from enigma_engine.core.sentiment import build_emotional_prompt_hint
        from enigma_engine.core.model_context import _EMOTIONAL_BASELINE
        state = dict(_EMOTIONAL_BASELINE)
        state["valence"] = 0.7
        state["trust"] = 0.9
        hint = build_emotional_prompt_hint(state)
        assert hint
        assert "warm" in hint.lower() or "open" in hint.lower() or "friendly" in hint.lower()


class TestModulateGenerationParams:
    """Tests for modulate_generation_params()."""

    def test_neutral_state_no_change(self):
        """Neutral/baseline state should not modify defaults."""
        from enigma_engine.core.sentiment import modulate_generation_params
        from enigma_engine.core.model_context import _EMOTIONAL_BASELINE
        params = modulate_generation_params(dict(_EMOTIONAL_BASELINE),
                                            temperature=0.8,
                                            repetition_penalty=1.1,
                                            top_p=0.9)
        assert abs(params["temperature"] - 0.8) < 0.05
        assert abs(params["repetition_penalty"] - 1.1) < 0.05
        assert abs(params["top_p"] - 0.9) < 0.05

    def test_high_arousal_raises_temperature(self):
        """High arousal should increase temperature."""
        from enigma_engine.core.sentiment import modulate_generation_params
        from enigma_engine.core.model_context import _EMOTIONAL_BASELINE
        state = dict(_EMOTIONAL_BASELINE)
        state["arousal"] = 0.9
        params = modulate_generation_params(state, temperature=0.8,
                                            repetition_penalty=1.1, top_p=0.9)
        assert params["temperature"] > 0.8

    def test_low_engagement_raises_repetition_penalty(self):
        """Low engagement should increase repetition penalty."""
        from enigma_engine.core.sentiment import modulate_generation_params
        from enigma_engine.core.model_context import _EMOTIONAL_BASELINE
        state = dict(_EMOTIONAL_BASELINE)
        state["engagement"] = 0.1
        params = modulate_generation_params(state, temperature=0.8,
                                            repetition_penalty=1.1, top_p=0.9)
        assert params["repetition_penalty"] > 1.1

    def test_high_frustration_lowers_top_p(self):
        """High frustration should tighten sampling (lower top_p)."""
        from enigma_engine.core.sentiment import modulate_generation_params
        from enigma_engine.core.model_context import _EMOTIONAL_BASELINE
        state = dict(_EMOTIONAL_BASELINE)
        state["frustration"] = 0.8
        params = modulate_generation_params(state, temperature=0.8,
                                            repetition_penalty=1.1, top_p=0.9)
        assert params["top_p"] < 0.9

    def test_params_stay_in_safe_range(self):
        """Even extreme states should produce safe parameter values."""
        from enigma_engine.core.sentiment import modulate_generation_params
        extreme = {"valence": -1.0, "arousal": 1.0, "engagement": 0.0,
                   "trust": 0.0, "frustration": 1.0}
        params = modulate_generation_params(extreme, temperature=0.8,
                                            repetition_penalty=1.1, top_p=0.9)
        assert 0.3 <= params["temperature"] <= 1.5
        assert 1.0 <= params["repetition_penalty"] <= 1.5
        assert 0.5 <= params["top_p"] <= 1.0

    def test_returns_all_three_keys(self):
        """Result always contains temperature, repetition_penalty, top_p."""
        from enigma_engine.core.sentiment import modulate_generation_params
        from enigma_engine.core.model_context import _EMOTIONAL_BASELINE
        params = modulate_generation_params(dict(_EMOTIONAL_BASELINE),
                                            temperature=0.8,
                                            repetition_penalty=1.1,
                                            top_p=0.9)
        assert "temperature" in params
        assert "repetition_penalty" in params
        assert "top_p" in params


# ====================================================================
# Dataset utility â€” process_text_corpus, clean_text, etc.
# ====================================================================

class TestProcessTextCorpus:
    """Tests for enigma_engine.core.dataset text processing."""

    def test_plain_text_passthrough(self, tmp_path):
        """Plain .txt files are returned cleaned."""
        from enigma_engine.core.dataset import process_text_corpus
        f = tmp_path / "data.txt"
        f.write_text("Hello world.\nThis is a test.\n", encoding="utf-8")
        result = process_text_corpus(f)
        assert "Hello world" in result
        assert "This is a test" in result

    def test_jsonl_text_key(self, tmp_path):
        """JSONL files with 'text' key are extracted."""
        import json
        from enigma_engine.core.dataset import process_text_corpus
        f = tmp_path / "data.jsonl"
        lines = [
            json.dumps({"text": "Story one."}),
            json.dumps({"text": "Story two."}),
        ]
        f.write_text("\n".join(lines), encoding="utf-8")
        result = process_text_corpus(f)
        assert "Story one" in result
        assert "Story two" in result

    def test_jsonl_custom_key(self, tmp_path):
        """JSONL extraction respects custom text_key."""
        import json
        from enigma_engine.core.dataset import process_text_corpus
        f = tmp_path / "stories.jsonl"
        lines = [json.dumps({"story": "Once upon a time."})]
        f.write_text("\n".join(lines), encoding="utf-8")
        result = process_text_corpus(f, text_key="story")
        assert "Once upon a time" in result

    def test_directory_of_txt(self, tmp_path):
        """Processes all .txt files in a directory."""
        from enigma_engine.core.dataset import process_text_corpus
        (tmp_path / "a.txt").write_text("File A content.", encoding="utf-8")
        (tmp_path / "b.txt").write_text("File B content.", encoding="utf-8")
        (tmp_path / "c.json").write_text("{}", encoding="utf-8")  # ignored
        result = process_text_corpus(tmp_path)
        assert "File A content" in result
        assert "File B content" in result

    def test_empty_file_returns_empty(self, tmp_path):
        """Empty file returns empty string."""
        from enigma_engine.core.dataset import process_text_corpus
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        result = process_text_corpus(f)
        assert result == ""

    def test_strips_null_bytes(self, tmp_path):
        """Null bytes are removed from text."""
        from enigma_engine.core.dataset import process_text_corpus
        f = tmp_path / "dirty.txt"
        f.write_bytes(b"Hello\x00World\x00Test")
        result = process_text_corpus(f)
        assert "\x00" not in result
        assert "Hello" in result

    def test_normalizes_whitespace(self, tmp_path):
        """Excessive whitespace is normalized."""
        from enigma_engine.core.dataset import process_text_corpus
        f = tmp_path / "spacey.txt"
        f.write_text("Too   many    spaces.\n\n\n\nToo many newlines.",
                      encoding="utf-8")
        result = process_text_corpus(f)
        # Should not have 4+ consecutive newlines
        assert "\n\n\n\n" not in result


class TestCleanText:
    """Tests for clean_text helper in dataset module."""

    def test_removes_null_bytes(self):
        from enigma_engine.core.dataset import clean_text
        assert "\x00" not in clean_text("hello\x00world")

    def test_normalizes_runs_of_newlines(self):
        from enigma_engine.core.dataset import clean_text
        result = clean_text("a\n\n\n\n\nb")
        assert result.count("\n") <= 3  # at most 2 blank lines

    def test_strips_trailing_whitespace(self):
        from enigma_engine.core.dataset import clean_text
        result = clean_text("line one   \nline two  \n")
        for line in result.split("\n"):
            assert line == line.rstrip()


class TestEstimateTokenCount:
    """Tests for estimate_token_count."""

    def test_roughly_correct(self):
        """4 chars per token is a reasonable estimate."""
        from enigma_engine.core.dataset import estimate_token_count
        text = "Hello world this is a test of the token counter."
        count = estimate_token_count(text)
        # ~48 chars / 4 = ~12 tokens
        assert 8 <= count <= 20

    def test_empty_returns_zero(self):
        from enigma_engine.core.dataset import estimate_token_count
        assert estimate_token_count("") == 0


class TestKnownDatasets:
    """KNOWN_DATASETS registry."""

    def test_tinystories_registered(self):
        from enigma_engine.core.dataset import KNOWN_DATASETS
        assert "tinystories" in KNOWN_DATASETS
        entry = KNOWN_DATASETS["tinystories"]
        assert isinstance(entry["name"], str) and len(entry["name"]) > 0
        assert isinstance(entry["description"], str) and len(entry["description"]) > 0

    def test_entries_have_required_fields(self):
        from enigma_engine.core.dataset import KNOWN_DATASETS
        assert len(KNOWN_DATASETS) > 0, "Registry should not be empty"
        for name, info in KNOWN_DATASETS.items():
            assert "name" in info, f"{name} missing 'name'"
            assert "description" in info, f"{name} missing 'description'"
            assert isinstance(info["name"], str), f"{name} 'name' must be str"
            assert len(info["name"]) > 0, f"{name} 'name' must not be empty"
            assert isinstance(info["description"], str), f"{name} 'description' must be str"
            assert len(info["description"]) > 0, f"{name} 'description' must not be empty"


# =====================================================================
# Suggestion batch: 5 fixes (March 2026)
# =====================================================================


class TestDatasetFileSizeGuard:
    """process_text_corpus must not OOM on oversized files."""

    def test_max_file_size_constant_exists(self):
        """dataset module defines a MAX_FILE_SIZE constant."""
        import enigma_engine.core.dataset as ds
        assert hasattr(ds, "MAX_FILE_SIZE"), (
            "dataset.py must define MAX_FILE_SIZE")
        assert isinstance(ds.MAX_FILE_SIZE, int)
        assert ds.MAX_FILE_SIZE > 0

    def test_oversized_file_returns_empty(self, tmp_path):
        """A file exceeding MAX_FILE_SIZE is skipped gracefully."""
        import enigma_engine.core.dataset as ds
        f = tmp_path / "huge.txt"
        f.write_text("x" * 100, encoding="utf-8")
        # Temporarily set a tiny limit
        orig = ds.MAX_FILE_SIZE
        try:
            ds.MAX_FILE_SIZE = 10  # 10 bytes
            result = ds.process_text_corpus(f)
            assert result == "", (
                "Oversized files should return empty string")
        finally:
            ds.MAX_FILE_SIZE = orig

    def test_normal_file_still_works(self, tmp_path):
        """Files under the limit are processed normally."""
        import enigma_engine.core.dataset as ds
        f = tmp_path / "small.txt"
        f.write_text("Hello world", encoding="utf-8")
        result = ds.process_text_corpus(f)
        assert "Hello world" in result


class TestStreamingDocstring:
    """Streaming module docstring accuracy."""

    def test_no_websocket_claim(self):
        """Module docstring must not claim WebSocket support if unimplemented."""
        import enigma_engine.core.streaming as sm
        doc = sm.__doc__ or ""
        # Either WebSocket is implemented OR not claimed
        has_websocket_methods = any(
            "websocket" in name.lower()
            for name in dir(sm.StreamingResponse)
        )
        if not has_websocket_methods:
            assert "WebSocket" not in doc, (
                "Module docstring claims WebSocket but no methods exist")


# ================================================================
# Phase 6: Emotional Learning
# ================================================================

class TestComputeEngagementScore:
    """compute_engagement_score maps emotional state to training weight."""

    def test_neutral_state_returns_one(self):
        """Neutral/baseline emotional state returns weight ~1.0."""
        from enigma_engine.core.sentiment import compute_engagement_score
        from enigma_engine.core.model_context import _EMOTIONAL_BASELINE
        score = compute_engagement_score(dict(_EMOTIONAL_BASELINE))
        assert 0.9 <= score <= 1.1

    def test_high_engagement_boosts_score(self):
        """High engagement + trust â†’ weight > 1.0."""
        from enigma_engine.core.sentiment import compute_engagement_score
        state = {
            "valence": 0.6, "arousal": 0.5,
            "engagement": 0.9, "trust": 0.8, "frustration": 0.0,
        }
        score = compute_engagement_score(state)
        assert score > 1.0

    def test_high_frustration_lowers_score(self):
        """High frustration â†’ weight < 1.0."""
        from enigma_engine.core.sentiment import compute_engagement_score
        state = {
            "valence": -0.3, "arousal": 0.6,
            "engagement": 0.3, "trust": 0.2, "frustration": 0.8,
        }
        score = compute_engagement_score(state)
        assert score < 1.0

    def test_clamped_range(self):
        """Output is always in [0.5, 2.0] regardless of input extremes."""
        from enigma_engine.core.sentiment import compute_engagement_score
        # Extreme positive
        high = compute_engagement_score({
            "valence": 1.0, "arousal": 1.0,
            "engagement": 1.0, "trust": 1.0, "frustration": 0.0,
        })
        assert 0.5 <= high <= 2.0
        # Extreme negative
        low = compute_engagement_score({
            "valence": -1.0, "arousal": 0.0,
            "engagement": 0.0, "trust": 0.0, "frustration": 1.0,
        })
        assert 0.5 <= low <= 2.0

    def test_empty_state_returns_one(self):
        """Empty dict returns neutral weight."""
        from enigma_engine.core.sentiment import compute_engagement_score
        score = compute_engagement_score({})
        assert 0.9 <= score <= 1.1


class TestEmotionalSelfPlayBonus:
    """Self-play rewards include emotional evaluation."""

    def test_positive_response_gets_bonus(self):
        """Helpful, engaging response gets positive bonus."""
        from enigma_engine.core.sentiment import evaluate_response_quality
        bonus = evaluate_response_quality(
            "Tell me about Python",
            "Python is a versatile programming language that's great "
            "for beginners and experts alike. Would you like to learn "
            "about specific features?"
        )
        assert bonus >= 0.0

    def test_bonus_clamped(self):
        """Emotional bonus is clamped to [-0.5, 0.5]."""
        from enigma_engine.core.sentiment import evaluate_response_quality
        bonus = evaluate_response_quality(
            "Hello!",
            "Hello! " * 100
        )
        assert -0.5 <= bonus <= 0.5

    def test_dismissive_response_gets_penalty(self):
        """Short dismissive response gets zero or negative bonus."""
        from enigma_engine.core.sentiment import evaluate_response_quality
        bonus = evaluate_response_quality(
            "Can you help me understand quantum physics?",
            "No."
        )
        assert bonus <= 0.0


# ================================================================
# VRAM-based preset recommendation
# ================================================================


class TestEstimateTrainingVram:
    """estimate_training_vram returns reasonable VRAM estimates."""

    def test_small_preset_low_vram(self):
        """Small preset should need ~1 GB with gradient checkpointing."""
        from enigma_engine.core.model_presets import (
            MODEL_PRESETS, estimate_training_vram)
        import copy
        cfg = copy.deepcopy(MODEL_PRESETS["small"])
        cfg.vocab_size = 32000
        vram = estimate_training_vram(cfg)
        assert 0.5 <= vram <= 2.0, f"small needs {vram} GB, expected 0.5-2"

    def test_large_preset_moderate_vram(self):
        """Large preset (~276M) should need 4-10 GB with gradient checkpointing."""
        from enigma_engine.core.model_presets import (
            MODEL_PRESETS, estimate_training_vram)
        import copy
        cfg = copy.deepcopy(MODEL_PRESETS["large"])
        cfg.vocab_size = 32000
        vram = estimate_training_vram(cfg)
        assert 4.0 <= vram <= 10.0, f"large needs {vram} GB, expected 4-10"

    def test_xl_preset_high_vram(self):
        """XL preset (~742M) should need 8-16 GB with gradient checkpointing."""
        from enigma_engine.core.model_presets import (
            MODEL_PRESETS, estimate_training_vram)
        import copy
        cfg = copy.deepcopy(MODEL_PRESETS["xl"])
        cfg.vocab_size = 32000
        vram = estimate_training_vram(cfg)
        assert 8.0 <= vram <= 16.0, f"xl needs {vram} GB, expected 8-16"

    def test_minimum_is_half_gb(self):
        """Even tiny models should return at least 0.5 GB."""
        from enigma_engine.core.model_presets import (
            MODEL_PRESETS, estimate_training_vram)
        import copy
        cfg = copy.deepcopy(MODEL_PRESETS["pi_zero"])
        cfg.vocab_size = 32000
        vram = estimate_training_vram(cfg)
        assert vram >= 0.5


class TestRecommendPresetForVram:
    """recommend_preset_for_vram picks the largest fitting preset."""

    def test_32gb_picks_large_or_above(self):
        """32 GB VRAM should pick xl (11.7 GB) â€” xxl needs ~34 GB."""
        from enigma_engine.core.model_presets import recommend_preset_for_vram
        result = recommend_preset_for_vram(32.0)
        assert result == "xl", f"32 GB got {result}, expected xl"

    def test_8gb_picks_large_or_smaller(self):
        """8 GB should pick large (6.3 GB) â€” xl needs ~11.7 GB."""
        from enigma_engine.core.model_presets import recommend_preset_for_vram
        result = recommend_preset_for_vram(8.0)
        ok_presets = {"small", "medium", "base", "large", "mini", "tiny",
                      "micro", "nano", "pi_5", "pi_4", "pi_zero"}
        assert result in ok_presets, f"8 GB got {result}"

    def test_2gb_picks_small_preset(self):
        """2 GB should pick small-ish preset."""
        from enigma_engine.core.model_presets import recommend_preset_for_vram
        result = recommend_preset_for_vram(2.0)
        small_presets = {"small", "medium", "base", "mini", "tiny",
                         "micro", "nano", "pi_5", "pi_4", "pi_zero"}
        assert result in small_presets, f"2 GB got {result}"

    def test_0_5gb_picks_tiny_preset(self):
        """0.5 GB should pick one of the smallest presets."""
        from enigma_engine.core.model_presets import recommend_preset_for_vram
        result = recommend_preset_for_vram(0.5)
        tiny_presets = {"pi_zero", "pi_4", "pi_5", "nano", "micro",
                        "tiny", "mini", "small"}
        assert result in tiny_presets, f"0.5 GB got {result}"

    def test_monotonic_bigger_vram_bigger_preset(self):
        """More VRAM should never pick a smaller preset."""
        from enigma_engine.core.model_presets import (
            MODEL_PRESETS, recommend_preset_for_vram,
            estimate_training_vram)
        import copy
        prev_vram_needed = 0
        for gb in [1, 4, 8, 16, 32, 64]:
            name = recommend_preset_for_vram(gb)
            cfg = copy.deepcopy(MODEL_PRESETS[name])
            cfg.vocab_size = 32000
            needed = estimate_training_vram(cfg)
            assert needed >= prev_vram_needed, (
                f"VRAM {gb}GB picked {name} (needs {needed}GB) "
                f"but previous needed {prev_vram_needed}GB")
            prev_vram_needed = needed


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# recommend_preset_for_tokens â€” Chinchilla-optimal preset selection
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestRecommendPresetForTokens:
    """recommend_preset_for_tokens picks the right preset for data size."""

    def test_small_data_picks_small_preset(self):
        """67M tokens should NOT recommend xl (742M params)."""
        from enigma_engine.core.model_presets import (
            recommend_preset_for_tokens)
        name, params = recommend_preset_for_tokens(67_000_000)
        # 67M tokens is far too little for xl (742M).
        # Should pick the smallest possible preset.
        assert name != "xl", (
            "67M tokens should not recommend xl")
        assert name != "large", (
            "67M tokens should not recommend large")
        # Should be one of the tiny presets
        small_presets = {"pi_zero", "pi_4", "pi_5", "nano", "micro",
                         "tiny", "mini"}
        assert name in small_presets, (
            f"67M tokens recommended '{name}' ({params:,} params)")

    def test_large_data_picks_larger_preset(self):
        """2B tokens should recommend a reasonable model."""
        from enigma_engine.core.model_presets import (
            recommend_preset_for_tokens)
        name, params = recommend_preset_for_tokens(2_000_000_000)
        # 2B tokens / 20 = 100M params â€” should pick medium or base
        assert params >= 10_000_000, (
            f"2B tokens recommended '{name}' with only {params:,} params")
        assert params <= 2_000_000_000 / 20 * 1.1

    def test_tiny_data_still_returns_result(self):
        """Even with 1K tokens, returns a valid preset."""
        from enigma_engine.core.model_presets import (
            recommend_preset_for_tokens)
        name, params = recommend_preset_for_tokens(1000)
        assert isinstance(name, str) and len(name) > 0
        assert params > 0

    def test_vram_constraint_limits_preset(self):
        """VRAM constraint should prevent picking too-large presets."""
        from enigma_engine.core.model_presets import (
            recommend_preset_for_tokens, estimate_training_vram, MODEL_PRESETS)
        import copy
        # 10B tokens, but only 2GB VRAM
        name, params = recommend_preset_for_tokens(
            10_000_000_000, vram_gb=2.0)
        cfg = copy.deepcopy(MODEL_PRESETS[name])
        cfg.vocab_size = 32000
        vram_needed = estimate_training_vram(cfg)
        assert vram_needed <= 2.0, (
            f"Recommended '{name}' needs {vram_needed}GB but only 2GB")

    def test_monotonic_more_tokens_bigger_or_equal(self):
        """More tokens should never recommend a smaller model."""
        from enigma_engine.core.model_presets import (
            recommend_preset_for_tokens)
        prev_params = 0
        for tokens in [100_000, 10_000_000, 500_000_000,
                       5_000_000_000]:
            _, params = recommend_preset_for_tokens(tokens)
            assert params >= prev_params, (
                f"{tokens:,} tokens got {params:,} params, "
                f"prev was {prev_params:,}")
            prev_params = params


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TC-2: estimate_parameters â€” verify param counts for known presets
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestEstimateParameters:
    """estimate_parameters returns plausible counts for known configs (TC-2)."""

    def test_returns_positive_int(self):
        """estimate_parameters always returns a positive integer."""
        from enigma_engine.core.model_presets import ForgeConfig, estimate_parameters
        cfg = ForgeConfig(dim=128, n_layers=4, n_heads=4, max_seq_len=256)
        params = estimate_parameters(cfg)
        assert isinstance(params, int)
        assert params > 0

    def test_small_preset_range(self):
        """Small preset should be roughly 30-100M params.

        MTP-2b: lower floor dropped from 50M to 30M when default
        n_predict_heads flipped 2 to 0. The MTP heads were padding
        the count without contributing to model capacity.
        """
        import copy
        from enigma_engine.core.model_presets import (
            MODEL_PRESETS, estimate_parameters)
        cfg = copy.deepcopy(MODEL_PRESETS["small"])
        cfg.vocab_size = 32000
        params = estimate_parameters(cfg)
        assert 30_000_000 <= params <= 100_000_000, f"small = {params:,}"

    def test_base_preset_range(self):
        """Base preset should be roughly 150-500M params."""
        import copy
        from enigma_engine.core.model_presets import (
            MODEL_PRESETS, estimate_parameters)
        cfg = copy.deepcopy(MODEL_PRESETS["base"])
        cfg.vocab_size = 32000
        params = estimate_parameters(cfg)
        assert 100_000_000 <= params <= 500_000_000, f"base = {params:,}"

    def test_more_layers_more_params(self):
        """Increasing n_layers should increase parameter count."""
        from enigma_engine.core.model_presets import ForgeConfig, estimate_parameters
        cfg4 = ForgeConfig(dim=256, n_layers=4, n_heads=4, max_seq_len=256)
        cfg8 = ForgeConfig(dim=256, n_layers=8, n_heads=4, max_seq_len=256)
        assert estimate_parameters(cfg8) > estimate_parameters(cfg4)

    def test_wider_dim_more_params(self):
        """Increasing dim should increase parameter count."""
        from enigma_engine.core.model_presets import ForgeConfig, estimate_parameters
        cfg_s = ForgeConfig(dim=256, n_layers=4, n_heads=4, max_seq_len=256)
        cfg_l = ForgeConfig(dim=512, n_layers=4, n_heads=8, max_seq_len=256)
        assert estimate_parameters(cfg_l) > estimate_parameters(cfg_s)

    def test_gqa_reduces_params(self):
        """GQA (fewer KV heads) should produce fewer params than MHA."""
        from enigma_engine.core.model_presets import ForgeConfig, estimate_parameters
        mha = ForgeConfig(dim=512, n_layers=8, n_heads=8, n_kv_heads=8,
                          max_seq_len=256)
        gqa = ForgeConfig(dim=512, n_layers=8, n_heads=8, n_kv_heads=2,
                          max_seq_len=256)
        assert estimate_parameters(gqa) < estimate_parameters(mha)

    def test_monotonic_across_presets(self):
        """Preset param counts should be monotonically increasing for the main tiers."""
        import copy
        from enigma_engine.core.model_presets import (
            MODEL_PRESETS, estimate_parameters)
        ordered = ["tiny", "small", "medium", "base", "large", "xl", "xxl"]
        prev = 0
        for name in ordered:
            cfg = copy.deepcopy(MODEL_PRESETS[name])
            cfg.vocab_size = 32000
            params = estimate_parameters(cfg)
            assert params > prev, f"{name} ({params:,}) <= previous ({prev:,})"
            prev = params

    def test_mtp_heads_add_params(self):
        """n_predict_heads > 0 should increase total params vs 0."""
        from enigma_engine.core.model_presets import ForgeConfig, estimate_parameters
        no_mtp = ForgeConfig(dim=256, n_layers=4, n_heads=4, max_seq_len=256,
                             n_predict_heads=0)
        with_mtp = ForgeConfig(dim=256, n_layers=4, n_heads=4, max_seq_len=256,
                               n_predict_heads=2)
        assert estimate_parameters(with_mtp) > estimate_parameters(no_mtp)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# S547  â€“ GGUF param estimation with quant-type detection
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestGGUFParamEstimation:
    """S547: GGUF param estimation should detect quant type from filename."""

    def test_detect_quant_q4_k_m(self):
        from enigma_engine.gui.gui_logic import _detect_gguf_quant_type
        assert _detect_gguf_quant_type("model-q4_k_m.gguf") == "q4_k_m"

    def test_detect_quant_q8_0(self):
        from enigma_engine.gui.gui_logic import _detect_gguf_quant_type
        assert _detect_gguf_quant_type("llama-q8_0.gguf") == "q8_0"

    def test_detect_quant_f16(self):
        from enigma_engine.gui.gui_logic import _detect_gguf_quant_type
        assert _detect_gguf_quant_type("model-f16.gguf") == "f16"

    def test_detect_quant_none(self):
        from enigma_engine.gui.gui_logic import _detect_gguf_quant_type
        assert _detect_gguf_quant_type("model.gguf") is None

    def test_detect_quant_iq3_xxs(self):
        from enigma_engine.gui.gui_logic import _detect_gguf_quant_type
        assert _detect_gguf_quant_type("model-iq3_xxs.gguf") == "iq3_xxs"

    def test_bytes_per_param_lookup(self):
        from enigma_engine.gui.gui_logic import _GGUF_BYTES_PER_PARAM
        assert _GGUF_BYTES_PER_PARAM["q4_k_m"] == 0.57
        assert _GGUF_BYTES_PER_PARAM["q8_0"] == 1.1
        assert _GGUF_BYTES_PER_PARAM["f16"] == 2.0

    def test_estimate_uses_quant_type(self):
        """Q8 file should estimate ~2x fewer params than Q4."""
        import os
        from enigma_engine.gui.gui_logic import _estimate_gguf_params

        class _FakeEngine:
            model = None

        engine = _FakeEngine()

        with tempfile.TemporaryDirectory() as td:
            # Create fake GGUF files of same size
            q4_path = os.path.join(td, "model-q4_k_m.gguf")
            q8_path = os.path.join(td, "model-q8_0.gguf")
            data = b"\x00" * 1_000_000
            with open(q4_path, "wb") as f:
                f.write(data)
            with open(q8_path, "wb") as f:
                f.write(data)

            q4_params = _estimate_gguf_params(engine, q4_path)
            q8_params = _estimate_gguf_params(engine, q8_path)
            # Q4 should estimate more params than Q8 for same file size
            assert q4_params > q8_params


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# S550  â€“ Shell metacharacter args rejected
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestSanitizeArgsRejection:
    """S550: All-metacharacter args should be dropped, not silently emptied."""

    def test_all_metachar_arg_dropped(self):
        from enigma_engine.core.commands import sanitize_args
        result = sanitize_args(["hello", ";|&", "world"])
        assert result == ["hello", "world"]

    def test_mixed_metachar_arg_kept(self):
        from enigma_engine.core.commands import sanitize_args
        result = sanitize_args(["he;llo"])
        assert result == ["hello"]

    def test_clean_args_unchanged(self):
        from enigma_engine.core.commands import sanitize_args
        result = sanitize_args(["hello", "world"])
        assert result == ["hello", "world"]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# S551  â€“ File permissions preserved in atomic writes
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# S546  â€“ GPU caching for reference model
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# AI Image Search â€” search.images command
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestImageSearchCommand:
    """AI Image Search: search.images command must be registered."""

    def test_search_images_registered(self):
        from enigma_engine.core.commands import get_registry
        registry = get_registry()
        cmds = registry.list_commands(prefix="search")
        names = [c.name for c in cmds]
        assert "search.images" in names, (
            "search.images must be registered in builtin commands")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Throughput Telemetry
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestThroughputTelemetry:
    """Throughput telemetry in TrainingMonitor."""

    def test_record_throughput(self):
        from enigma_engine.training.training_monitor import TrainingMonitor
        mon = TrainingMonitor()
        mon.start_run()
        mon.record_throughput(1024, 0.5)
        mon.record_throughput(1024, 0.6)
        assert mon.total_tokens == 2048
        assert mon.avg_tokens_per_sec > 0
        assert mon.avg_step_time > 0

    def test_chart_data_includes_throughput(self):
        from enigma_engine.training.training_monitor import TrainingMonitor
        mon = TrainingMonitor()
        mon.start_run()
        mon.record_loss(1.0)
        mon.record_throughput(512, 0.25)
        data = mon.get_chart_data()
        assert "tokens_per_sec" in data
        assert "step_times" in data
        assert "total_tokens" in data
        assert "avg_tokens_per_sec" in data
        assert data["total_tokens"] == 512

    def test_finish_run_includes_throughput(self):
        from pathlib import Path
        from enigma_engine.training.training_monitor import TrainingMonitor
        with tempfile.TemporaryDirectory() as td:
            mon = TrainingMonitor(history_path=Path(td) / "history.json")
            mon.start_run()
            mon.record_loss(2.0)
            mon.record_throughput(10000, 1.0)
            run = mon.finish_run(mode="sft", model_name="test")
            assert run.extra.get("total_tokens") == 10000
            assert run.extra.get("avg_tokens_per_sec") == 10000.0


# ================================================================
# S542: Dead code removal verification
# ================================================================

class TestS542DeadCodeRemoved:
    """Verify cache_model/get_cached_model/clear_cache removed from model_registry."""

    def test_all_list_no_cache_functions(self):
        from enigma_engine.core.model_registry import __all__
        assert 'cache_model' not in __all__
        assert 'get_cached_model' not in __all__
        assert 'clear_cache' not in __all__

    def test_still_has_useful_exports(self):
        from enigma_engine.core.model_registry import __all__
        assert 'ModelRegistry' in __all__
        assert 'safe_load_weights' in __all__
        assert 'get_state_dict' in __all__
        assert 'get_model_hash' in __all__


# ================================================================
# Tokenizer metrics module
# ================================================================

class TestTokenizerMetrics:
    """Tests for tokenizer_metrics.py analysis functions."""

    def _make_tokenizer(self):
        """Create a small BPE tokenizer for testing."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        tok.train(["hello world hello world foo bar baz " * 10],
                  vocab_size=300, min_frequency=1, verbose=False)
        return tok

    def test_analyze_vocabulary_returns_expected_keys(self):
        from enigma_engine.core.tokenizer_metrics import analyze_vocabulary
        tok = self._make_tokenizer()
        result = analyze_vocabulary(tok)
        # Verify keys exist AND have correct types
        assert isinstance(result['vocab_size'], int) and result['vocab_size'] > 0
        assert isinstance(result['num_merges'], int) and result['num_merges'] >= 0
        assert isinstance(result['num_special'], int) and result['num_special'] >= 0
        assert isinstance(result['token_lengths'], dict)
        assert isinstance(result['top_tokens'], list)
        assert isinstance(result['single_char_tokens'], int)
        assert isinstance(result['use_utf8_bytes'], bool)
        # Consistency checks
        assert result['vocab_size'] == len(tok.token_to_id)
        assert result['num_special'] == len(tok.special_tokens)

    def test_analyze_vocabulary_values(self):
        from enigma_engine.core.tokenizer_metrics import analyze_vocabulary
        tok = self._make_tokenizer()
        result = analyze_vocabulary(tok)
        assert result['vocab_size'] > 0
        assert result['num_special'] == len(tok.special_tokens)
        # Tok-2: fresh BPETokenizer defaults to byte-level mode so any
        # Unicode codepoint roundtrips without <unk>. Was False before.
        assert result['use_utf8_bytes'] is True

    def test_evaluate_coverage_keys(self):
        from enigma_engine.core.tokenizer_metrics import evaluate_coverage
        tok = self._make_tokenizer()
        result = evaluate_coverage(tok, ["hello world"])
        assert 'total_tokens' in result
        assert 'unique_tokens' in result
        assert 'unk_count' in result
        assert 'unk_rate' in result
        assert 'coverage' in result

    def test_evaluate_coverage_no_unk(self):
        from enigma_engine.core.tokenizer_metrics import evaluate_coverage
        tok = self._make_tokenizer()
        result = evaluate_coverage(tok, ["hello world"])
        assert result['unk_count'] == 0
        assert result['coverage'] == 1.0

    def test_compute_compression_ratio(self):
        from enigma_engine.core.tokenizer_metrics import compute_compression_ratio
        tok = self._make_tokenizer()
        result = compute_compression_ratio(tok, ["hello world foo bar"])
        assert result['total_chars'] > 0
        assert result['total_tokens'] > 0
        assert result['chars_per_token'] > 0

    def test_detect_issues_empty_on_healthy(self):
        from enigma_engine.core.tokenizer_metrics import detect_issues
        tok = self._make_tokenizer()
        issues = detect_issues(tok, ["hello world foo bar baz"])
        # healthy tokenizer should have no high-severity issues
        critical = [i for i in issues if "High UNK" in i]
        assert len(critical) == 0

    def test_format_report_returns_string(self):
        from enigma_engine.core.tokenizer_metrics import format_report
        tok = self._make_tokenizer()
        report = format_report(tok, ["hello world"])
        assert isinstance(report, str)
        assert "Tokenizer Analysis Report" in report
        assert "Vocab size" in report
        assert "Coverage" in report
        assert "Compression" in report

    def test_empty_texts_dont_crash(self):
        from enigma_engine.core.tokenizer_metrics import (
            evaluate_coverage, compute_compression_ratio, detect_issues)
        tok = self._make_tokenizer()
        cov = evaluate_coverage(tok, [])
        assert cov['total_tokens'] == 0
        comp = compute_compression_ratio(tok, [])
        assert comp['total_chars'] == 0
        issues = detect_issues(tok, [])
        assert isinstance(issues, list)


# ================================================================
# Byte-level BPE toggle wiring
# ================================================================


# ================================================================
# Tokenizer analysis wiring
# ================================================================


# =====================================================================
# PASS 29: Training pipeline fixes (S553â€“S557)
# =====================================================================


class TestS554TrainingConfigValidateExpanded:
    """S554: TrainingConfig.validate() must check more fields."""

    def test_validate_rejects_bad_adam_beta1(self):
        from enigma_engine.training.training import TrainingConfig
        cfg = TrainingConfig(adam_beta1=0.0)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validate_rejects_bad_adam_beta2(self):
        from enigma_engine.training.training import TrainingConfig
        cfg = TrainingConfig(adam_beta2=1.0)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validate_rejects_negative_ema_decay(self):
        from enigma_engine.training.training import TrainingConfig
        cfg = TrainingConfig(ema_decay=-0.1)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validate_rejects_label_smoothing_ge_1(self):
        from enigma_engine.training.training import TrainingConfig
        cfg = TrainingConfig(label_smoothing=1.0)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validate_rejects_negative_z_loss(self):
        from enigma_engine.training.training import TrainingConfig
        cfg = TrainingConfig(z_loss_weight=-1.0)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validate_rejects_zero_reasoning_weight(self):
        from enigma_engine.training.training import TrainingConfig
        cfg = TrainingConfig(reasoning_loss_weight=0.0)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validate_rejects_mix_ratio_out_of_range(self):
        from enigma_engine.training.training import TrainingConfig
        cfg = TrainingConfig(general_mix_ratio=1.5)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validate_rejects_negative_rolling_k(self):
        from enigma_engine.training.training import TrainingConfig
        cfg = TrainingConfig(rolling_best_k=-1)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validate_rejects_negative_gradient_noise_gamma(self):
        from enigma_engine.training.training import TrainingConfig
        cfg = TrainingConfig(gradient_noise_gamma=-0.5)
        with pytest.raises(ValueError):
            cfg.validate()

    def test_validate_rejects_min_lr_ratio_below_zero(self):
        """Pass 156z9au: min_lr_ratio in [0, 1]."""
        from enigma_engine.training.training import TrainingConfig
        cfg = TrainingConfig(min_lr_ratio=-0.01)
        with pytest.raises(ValueError, match="min_lr_ratio"):
            cfg.validate()

    def test_validate_rejects_min_lr_ratio_above_one(self):
        """Pass 156z9au: ratio > 1 makes the floor higher than peak."""
        from enigma_engine.training.training import TrainingConfig
        cfg = TrainingConfig(min_lr_ratio=1.01)
        with pytest.raises(ValueError, match="min_lr_ratio"):
            cfg.validate()

    def test_validate_accepts_min_lr_ratio_zero(self):
        """Pass 156z9au: 0.0 reproduces textbook cosine to-zero schedule."""
        from enigma_engine.training.training import TrainingConfig
        cfg = TrainingConfig(min_lr_ratio=0.0)
        cfg.validate()

    def test_min_lr_ratio_default_is_one_tenth(self):
        """Pass 156z9au: default LM-friendly floor."""
        from enigma_engine.training.training import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.min_lr_ratio == 0.1

    def test_min_lr_ratio_appears_in_to_dict(self):
        """Pass 156z9au: must round-trip through serialization."""
        from enigma_engine.training.training import TrainingConfig
        cfg = TrainingConfig(min_lr_ratio=0.05)
        assert cfg.to_dict()["min_lr_ratio"] == 0.05

    def test_validate_accepts_good_defaults(self):
        """Default config must pass validation."""
        from enigma_engine.training.training import TrainingConfig
        cfg = TrainingConfig()
        cfg.validate()  # Should not raise


# ================================================================
# MODEL UTILS â€” Registry, Sampling, Hardware Estimation
# ================================================================

class TestModelRegistryUtils:
    """Tests for the global model registry in model_utils.py."""

    def test_register_and_get(self):
        """register_model stores a model retrievable by get_model."""
        from enigma_engine.core.model_utils import (
            register_model, get_model, unregister_model)
        sentinel = object()
        register_model("__test_reg__", sentinel)
        try:
            assert get_model("__test_reg__") is sentinel
        finally:
            unregister_model("__test_reg__")

    def test_is_model_loaded(self):
        """is_model_loaded returns True after register, False after unregister."""
        from enigma_engine.core.model_utils import (
            register_model, unregister_model, is_model_loaded)
        register_model("__test_loaded__", object())
        assert is_model_loaded("__test_loaded__") is True
        unregister_model("__test_loaded__")
        assert is_model_loaded("__test_loaded__") is False

    def test_unregister_returns_model(self):
        """unregister_model returns the model that was registered."""
        from enigma_engine.core.model_utils import (
            register_model, unregister_model)
        sentinel = object()
        register_model("__test_unreg__", sentinel)
        result = unregister_model("__test_unreg__")
        assert result is sentinel

    def test_unregister_missing_returns_none(self):
        """unregister_model returns None for unknown names."""
        from enigma_engine.core.model_utils import unregister_model
        assert unregister_model("__nonexistent__") is None

    def test_get_running_models_is_copy(self):
        """get_running_models returns a copy, not the internal dict."""
        from enigma_engine.core.model_utils import (
            register_model, unregister_model, get_running_models)
        register_model("__test_copy__", object())
        try:
            models = get_running_models()
            models["__injected__"] = "bad"
            # Internal dict should not have the injected key
            models2 = get_running_models()
            assert "__injected__" not in models2
        finally:
            unregister_model("__test_copy__")

    def test_get_model_missing(self):
        """get_model returns None for unknown model name."""
        from enigma_engine.core.model_utils import get_model
        assert get_model("__does_not_exist__") is None


class TestRepetitionPenalty:
    """Tests for apply_repetition_penalty in model_utils.py."""

    def test_penalty_1_0_is_noop(self):
        """Penalty of 1.0 returns a clone with identical values."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_utils import apply_repetition_penalty
        logits = torch.randn(1, 100)
        tokens = torch.tensor([1, 2, 3])
        result = apply_repetition_penalty(logits, tokens, penalty=1.0)
        assert torch.equal(result, logits)
        # Must be a clone, not the same tensor
        assert result.data_ptr() != logits.data_ptr()

    def test_penalty_suppresses_repeated_tokens(self):
        """Penalty >1.0 reduces probability of token IDs already seen."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_utils import apply_repetition_penalty
        logits = torch.ones(1, 10)  # all logits = 1.0 (positive)
        tokens = torch.tensor([3, 5])
        result = apply_repetition_penalty(logits, tokens, penalty=2.0)
        # Token 3 and 5 should be penalized (divided by 2.0 since positive)
        assert result[0, 3].item() < 1.0
        assert result[0, 5].item() < 1.0
        # Token 0 should be unchanged
        assert result[0, 0].item() == 1.0

    def test_penalty_negative_logits_multiplied(self):
        """Negative logits are multiplied by penalty (pushed more negative)."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_utils import apply_repetition_penalty
        logits = torch.full((1, 10), -2.0)
        tokens = torch.tensor([4])
        result = apply_repetition_penalty(logits, tokens, penalty=2.0)
        # -2.0 * 2.0 = -4.0
        assert result[0, 4].item() == pytest.approx(-4.0, abs=0.01)
        # Non-repeated token stays at -2.0
        assert result[0, 0].item() == pytest.approx(-2.0, abs=0.01)

    def test_1d_logits(self):
        """Works with 1D logit tensors (no batch dimension)."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_utils import apply_repetition_penalty
        logits = torch.ones(10)
        tokens = torch.tensor([2])
        result = apply_repetition_penalty(logits, tokens, penalty=1.5)
        assert result[2].item() < 1.0
        assert result[0].item() == 1.0

    def test_window_limits_lookback(self):
        """Only the last `window` tokens are penalized."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_utils import apply_repetition_penalty
        logits = torch.ones(1, 100)
        # Token 5 appeared long ago (beyond window=3)
        tokens = torch.tensor([5, 10, 20, 30])
        result = apply_repetition_penalty(logits, tokens, penalty=2.0, window=3)
        # Token 5 is outside the window (last 3 = [10, 20, 30])
        assert result[0, 5].item() == 1.0
        # Token 30 is inside the window
        assert result[0, 30].item() < 1.0

    def test_out_of_range_tokens_ignored(self):
        """Token IDs >= vocab_size are safely ignored."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_utils import apply_repetition_penalty
        logits = torch.ones(1, 10)
        tokens = torch.tensor([999])  # way beyond vocab size 10
        result = apply_repetition_penalty(logits, tokens, penalty=2.0)
        # All logits should remain 1.0 (999 is out of range)
        assert torch.all(result == 1.0)


class TestSampleNextToken:
    """Tests for sample_next_token in model_utils.py."""

    def test_greedy_decode(self):
        """Temperature <= 0 should return argmax."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_utils import sample_next_token
        logits = torch.zeros(1, 100)
        logits[0, 42] = 10.0  # token 42 is the clear winner
        result = sample_next_token(
            logits, torch.zeros(1, 1, dtype=torch.long),
            temperature=0.0)
        assert result.item() == 42

    def test_returns_valid_token(self):
        """Sampled token ID is within vocab range."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_utils import sample_next_token
        logits = torch.randn(1, 50)
        result = sample_next_token(
            logits, torch.zeros(1, 1, dtype=torch.long),
            temperature=0.8)
        assert 0 <= result.item() < 50

    def test_output_shape(self):
        """Output shape is [batch, 1]."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_utils import sample_next_token
        logits = torch.randn(1, 50)
        result = sample_next_token(
            logits, torch.zeros(1, 1, dtype=torch.long))
        assert result.shape == (1, 1)

    def test_top_k_limits_candidates(self):
        """With top_k=1, sampling is deterministic (picks the max)."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_utils import sample_next_token
        logits = torch.zeros(1, 100)
        logits[0, 77] = 100.0
        result = sample_next_token(
            logits, torch.zeros(1, 1, dtype=torch.long),
            temperature=1.0, top_k=1)
        assert result.item() == 77


class TestMemoryEstimation:
    """Tests for estimate_memory_usage in model_utils.py."""

    def test_returns_expected_keys(self):
        """estimate_memory_usage returns model_size_mb, inference_ram_mb, training_ram_mb."""
        from enigma_engine.core.model_utils import estimate_memory_usage
        result = estimate_memory_usage("small")
        assert "model_size_mb" in result
        assert "inference_ram_mb" in result
        assert "training_ram_mb" in result

    def test_quantization_reduces_size(self):
        """Quantized models should be smaller than full precision."""
        from enigma_engine.core.model_utils import estimate_memory_usage
        full = estimate_memory_usage("small", "none")
        q8 = estimate_memory_usage("small", "int8")
        assert q8["model_size_mb"] < full["model_size_mb"]

    def test_larger_model_uses_more_memory(self):
        """Larger models should use more memory."""
        from enigma_engine.core.model_utils import estimate_memory_usage
        small = estimate_memory_usage("small")
        medium = estimate_memory_usage("medium")
        assert medium["model_size_mb"] > small["model_size_mb"]

    def test_unknown_size_has_fallback(self):
        """Unknown size names should still return a result."""
        from enigma_engine.core.model_utils import estimate_memory_usage
        result = estimate_memory_usage("unknown_size_xyz")
        assert result["model_size_mb"] > 0

    def test_hardware_estimate_uses_preset_dims(self):
        """estimate_memory_usage must read dim/n_layers from MODEL_PRESETS (S747)."""
        from enigma_engine.core.hardware_detection import estimate_memory_usage
        from enigma_engine.core.model_presets import MODEL_PRESETS
        for name in ("small", "medium", "large", "xl"):
            MODEL_PRESETS[name]
            result = estimate_memory_usage(name)
            # KV cache scales with n_layers * dim â€” verify it reflects
            # the real preset, not stale hardcoded fallbacks.
            result2 = estimate_memory_usage(name, seq_len=1024)
            # Doubling seq_len should roughly double KV cache
            ratio = result2["kv_cache"] / max(result["kv_cache"], 1e-12)
            assert 1.9 < ratio < 2.1, (
                f"{name}: KV cache ratio {ratio:.2f} when doubling "
                f"seq_len â€” expected ~2.0")
            assert result["model_memory"] > 0
            assert result["total"] > result["model_memory"]


# ================================================================
# WEIGHT MAPPING â€” Extended Edge Cases
# ================================================================

class TestWeightMappingExtended:
    """Extended tests for weight_mapping.py edge cases."""

    def test_detect_phi_model_type(self):
        """_detect_hf_model_type identifies Phi layout via mlp.fc1."""
        from enigma_engine.core.weight_mapping import WeightMapper
        mapper = WeightMapper()
        dummy = {"model.layers.0.self_attn.q_proj.weight": 1,
                 "model.layers.0.mlp.fc1.weight": 2}
        assert mapper._detect_hf_model_type(dummy) == "phi"

    def test_hf_llama_layer_attention(self):
        """Llama attention projections map correctly."""
        from enigma_engine.core.weight_mapping import WeightMapper
        mapper = WeightMapper()
        dummy = {
            "model.layers.3.self_attn.q_proj.weight": "q",
            "model.layers.3.self_attn.k_proj.weight": "k",
            "model.layers.3.self_attn.v_proj.weight": "v",
            "model.layers.3.self_attn.o_proj.weight": "o",
        }
        result = mapper.map_huggingface_to_forge(dummy, model_type="llama")
        assert "layers.3.attention.wq.weight" in result
        assert "layers.3.attention.wk.weight" in result
        assert "layers.3.attention.wv.weight" in result
        assert "layers.3.attention.wo.weight" in result

    def test_hf_llama_ffn(self):
        """Llama FFN projections (SwiGLU) map correctly."""
        from enigma_engine.core.weight_mapping import WeightMapper
        mapper = WeightMapper()
        dummy = {
            "model.layers.0.mlp.gate_proj.weight": "w1",
            "model.layers.0.mlp.down_proj.weight": "w2",
            "model.layers.0.mlp.up_proj.weight": "w3",
        }
        result = mapper.map_huggingface_to_forge(dummy, model_type="llama")
        assert "layers.0.feed_forward.w1.weight" in result
        assert "layers.0.feed_forward.w2.weight" in result
        assert "layers.0.feed_forward.w3.weight" in result

    def test_skipped_weights_counted(self):
        """Unmapped weight names raise ValueError when skip ratio > 10%."""
        from enigma_engine.core.weight_mapping import WeightMapper
        mapper = WeightMapper()
        dummy = {"some.random.weight.name": "tensor"}
        with pytest.raises(ValueError, match="unmapped"):
            mapper.map_huggingface_to_forge(dummy, model_type="llama")

    def test_gguf_layer_attention(self):
        """GGUF attention weights map to Forge format."""
        from enigma_engine.core.weight_mapping import WeightMapper
        mapper = WeightMapper()
        dummy = {
            "blk.2.attn_q.weight": "q",
            "blk.2.attn_k.weight": "k",
            "blk.2.attn_v.weight": "v",
            "blk.2.attn_output.weight": "o",
        }
        result = mapper.map_gguf_to_forge(dummy)
        assert "layers.2.attention.wq.weight" in result
        assert "layers.2.attention.wk.weight" in result
        assert "layers.2.attention.wv.weight" in result
        assert "layers.2.attention.wo.weight" in result

    def test_gguf_ffn(self):
        """GGUF FFN weights map to Forge format."""
        from enigma_engine.core.weight_mapping import WeightMapper
        mapper = WeightMapper()
        dummy = {
            "blk.1.ffn_gate.weight": "w1",
            "blk.1.ffn_down.weight": "w2",
            "blk.1.ffn_up.weight": "w3",
        }
        result = mapper.map_gguf_to_forge(dummy)
        assert "layers.1.feed_forward.w1.weight" in result
        assert "layers.1.feed_forward.w2.weight" in result
        assert "layers.1.feed_forward.w3.weight" in result

    def test_qwen_map_exists(self):
        """Qwen2/Qwen3 maps exist in HF_MODEL_MAPS."""
        from enigma_engine.core.weight_mapping import HF_MODEL_MAPS
        assert "qwen2" in HF_MODEL_MAPS
        assert "qwen3" in HF_MODEL_MAPS

    def test_mistral_shares_llama_map(self):
        """Mistral uses the same mapping as Llama."""
        from enigma_engine.core.weight_mapping import (
            HF_LLAMA_MAP, HF_MISTRAL_MAP)
        assert HF_MISTRAL_MAP is HF_LLAMA_MAP

    def test_shape_based_mapping_needs_config(self):
        """_shape_based_mapping returns empty dict without dim/vocab_size."""
        from enigma_engine.core.weight_mapping import WeightMapper
        mapper = WeightMapper()
        result = mapper._shape_based_mapping({"w": "t"}, config=None)
        assert result == {}


# ================================================================
# GGUF DEQUANT â€” Extended Validation Tests
# ================================================================

class TestGGUFDequantExtended:
    """Extended tests for gguf_dequant.py validation."""

    def test_extract_config_embed_length_alias(self):
        """extract_config_from_metadata handles embed_length alias."""
        from enigma_engine.core.gguf_dequant import extract_config_from_metadata
        metadata = {"llama.embed_length": 1024}
        config = extract_config_from_metadata(metadata)
        assert config["dim"] == 1024

    def test_extract_config_tokenizer_tokens(self):
        """extract_config_from_metadata gets vocab_size from tokenizer tokens."""
        from enigma_engine.core.gguf_dequant import extract_config_from_metadata
        metadata = {"tokenizer.ggml.tokens": ["a", "b", "c"]}
        config = extract_config_from_metadata(metadata)
        assert config["vocab_size"] == 3

    def test_dequantize_q4_0_values(self):
        """Q4_0 dequantization produces non-zero values with non-zero scale."""
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q4_0
        # Build a block with scale=1.0 and non-zero data
        scale = np.float16(1.0)
        data_bytes = bytes([0xFF] * 16)  # all nibbles = 0xF â†’ high=7, low=7
        block = scale.tobytes() + data_bytes
        result = dequantize_q4_0(block, (32,))
        assert result.shape == (32,)
        # With scale=1.0 and values of 7, should have non-zero entries
        assert result.abs().max().item() > 0

    def test_dequantize_q8_0_values(self):
        """Q8_0 dequantization produces expected values."""
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q8_0
        # Build a block with scale=2.0 and values = [1, 1, ..., 1]
        scale = np.float16(2.0)
        values = bytes([1] * 32)  # all values = 1 (unsigned, interpreted as int8)
        block = scale.tobytes() + values
        result = dequantize_q8_0(block, (32,))
        assert result.shape == (32,)
        # int8(1) * 2.0 = 2.0
        assert result[0].item() == pytest.approx(2.0, abs=0.1)

    def test_dequantize_q8_0_zero_blocks(self):
        """Q8_0 with empty input returns a zero tensor of the requested shape
        AND dtype=float32 (parity with the Q4_0/Q4_1/Q5_0/Q5_1 zero-blocks
        degeneracy gate â€” all five siblings must return float32 regardless
        of torch's default dtype)."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q8_0
        result = dequantize_q8_0(b"", (32,))
        assert result.shape == (32,)
        assert result.dtype == torch.float32
        assert torch.all(result == 0)

    def test_dequantize_q8_0_signed_values(self):
        """Q8_0 treats qs as int8: 0xFF (255 unsigned) decodes to -1 signed."""
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q8_0
        # scale=1.0; byte 0 = 0xFF (= -1 as int8), byte 1 = 0x7F (= +127)
        scale = np.float16(1.0)
        qs = bytearray(32)
        qs[0] = 0xFF
        qs[1] = 0x7F
        block = scale.tobytes() + bytes(qs)
        result = dequantize_q8_0(block, (32,))
        assert result[0].item() == pytest.approx(-1.0, abs=0.05)
        assert result[1].item() == pytest.approx(127.0, abs=0.5)

    def test_dequantize_q4_0_zero_scale(self):
        """Q4_0 with zero scale produces all zeros."""
        import numpy as np
        torch = pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q4_0
        scale = np.float16(0.0)
        data_bytes = bytes([0xFF] * 16)
        block = scale.tobytes() + data_bytes
        result = dequantize_q4_0(block, (32,))
        assert torch.all(result == 0)

    def test_dequantize_q4_0_layout(self):
        """Q4_0 layout matches ggml dequantize_row_q4_0:
        byte j low-nibble â†’ element j (low half), high-nibble â†’ element j+16.
        Both nibbles signed: q - 8.
        """
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q4_0
        d = np.float16(1.0).tobytes()
        # byte 0 = 0x91 â†’ low=1, high=9 â†’ element 0 = (1-8)*1 = -7, element 16 = (9-8)*1 = 1
        # byte 5 = 0xC3 â†’ low=3, high=12 â†’ element 5 = (3-8)*1 = -5, element 21 = (12-8)*1 = 4
        qs = bytearray(16)
        qs[0] = 0x91
        qs[5] = 0xC3
        block = d + bytes(qs)
        result = dequantize_q4_0(block, (32,))
        assert result[0].item() == pytest.approx(-7.0, abs=0.05)
        assert result[16].item() == pytest.approx(1.0, abs=0.05)
        assert result[5].item() == pytest.approx(-5.0, abs=0.05)
        assert result[21].item() == pytest.approx(4.0, abs=0.05)
        # Untouched bytes are 0x00 â†’ low=0, high=0 â†’ -8 in both halves
        assert result[1].item() == pytest.approx(-8.0, abs=0.05)
        assert result[17].item() == pytest.approx(-8.0, abs=0.05)

    def test_dequantize_q4_1_values(self):
        """Q4_1 dequant: y = q*d + m. q=15 (all-Fs), d=2.0, m=1.0 â†’ 31.0 every element."""
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q4_1
        d = np.float16(2.0).tobytes()
        m = np.float16(1.0).tobytes()
        qs = bytes([0xFF] * 16)  # all nibbles = 15
        block = d + m + qs
        result = dequantize_q4_1(block, (32,))
        assert result.shape == (32,)
        # 15 * 2.0 + 1.0 = 31.0
        assert result[0].item() == pytest.approx(31.0, abs=0.05)
        assert result[31].item() == pytest.approx(31.0, abs=0.05)

    def test_dequantize_q4_1_layout(self):
        """Q4_1 layout: byte j low-nibble â†’ element j (low half), high-nibble â†’ element j+16."""
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q4_1
        d = np.float16(1.0).tobytes()
        m = np.float16(0.0).tobytes()
        # byte 0 = 0x21 â†’ low nib 1 (elem 0), high nib 2 (elem 16)
        # byte 5 = 0x43 â†’ low nib 3 (elem 5), high nib 4 (elem 21)
        qs = bytearray([0] * 16)
        qs[0] = 0x21
        qs[5] = 0x43
        block = d + m + bytes(qs)
        result = dequantize_q4_1(block, (32,))
        assert result[0].item() == pytest.approx(1.0, abs=0.01)
        assert result[16].item() == pytest.approx(2.0, abs=0.01)
        assert result[5].item() == pytest.approx(3.0, abs=0.01)
        assert result[21].item() == pytest.approx(4.0, abs=0.01)

    def test_dequantize_q5_0_values(self):
        """Q5_0 dequant: y = (q - 16) * d. q=31 (all bits set), d=1.0 â†’ 15.0 every element."""
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q5_0
        d = np.float16(1.0).tobytes()
        qh = np.uint32(0xFFFFFFFF).tobytes()  # 5th bit set for every element
        qs = bytes([0xFF] * 16)               # low nibble 0xF for every position
        block = d + qh + qs
        result = dequantize_q5_0(block, (32,))
        assert result.shape == (32,)
        # q = 0xF | (1 << 4) = 31; (31 - 16) * 1.0 = 15.0
        assert result[0].item() == pytest.approx(15.0, abs=0.05)
        assert result[31].item() == pytest.approx(15.0, abs=0.05)

    def test_dequantize_q5_0_zero_q_gives_neg16(self):
        """Q5_0 with q=0 dequants to -16 * d (signed shift)."""
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q5_0
        d = np.float16(1.0).tobytes()
        qh = np.uint32(0).tobytes()
        qs = bytes([0] * 16)
        block = d + qh + qs
        result = dequantize_q5_0(block, (32,))
        assert result[0].item() == pytest.approx(-16.0, abs=0.05)

    def test_dequantize_q5_0_qh_bit_routing(self):
        """Q5_0 qh bit i must route to element i (NOT element 2i or 2i+1)."""
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q5_0
        d = np.float16(1.0).tobytes()
        # Only bit 16 of qh set â†’ only element 16 has the 5th bit
        qh = np.uint32(1 << 16).tobytes()
        qs = bytes([0] * 16)
        block = d + qh + qs
        result = dequantize_q5_0(block, (32,))
        # element 16: q = 0 | (1<<4) = 16; (16 - 16) * 1.0 = 0.0
        # every other element: q = 0; (0 - 16) * 1.0 = -16.0
        assert result[16].item() == pytest.approx(0.0, abs=0.05)
        assert result[0].item() == pytest.approx(-16.0, abs=0.05)
        assert result[15].item() == pytest.approx(-16.0, abs=0.05)
        assert result[17].item() == pytest.approx(-16.0, abs=0.05)

    def test_dequantize_q5_1_values(self):
        """Q5_1 dequant: y = q * d + m. q=31, d=2.0, m=1.0 â†’ 63.0 every element."""
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q5_1
        d = np.float16(2.0).tobytes()
        m = np.float16(1.0).tobytes()
        qh = np.uint32(0xFFFFFFFF).tobytes()
        qs = bytes([0xFF] * 16)
        block = d + m + qh + qs
        result = dequantize_q5_1(block, (32,))
        assert result.shape == (32,)
        # q = 31, 31 * 2.0 + 1.0 = 63.0
        assert result[0].item() == pytest.approx(63.0, abs=0.1)

    def test_dequantize_q5_1_layout(self):
        """Q5_1 layout matches Q5_0: low nibble â†’ low half, high nibble â†’ high half."""
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q5_1
        d = np.float16(1.0).tobytes()
        m = np.float16(0.0).tobytes()
        qh = np.uint32(0).tobytes()
        # byte 3 = 0x52 â†’ low nib 2 (elem 3), high nib 5 (elem 19)
        qs = bytearray([0] * 16)
        qs[3] = 0x52
        block = d + m + qh + bytes(qs)
        result = dequantize_q5_1(block, (32,))
        assert result[3].item() == pytest.approx(2.0, abs=0.05)
        assert result[19].item() == pytest.approx(5.0, abs=0.05)
        assert result[0].item() == pytest.approx(0.0, abs=0.05)

    def test_dequantize_q4_1_zero_blocks(self):
        """Q4_1 with empty input returns zero tensor of requested shape."""
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q4_1
        result = dequantize_q4_1(b"", (8,))
        assert result.shape == (8,)
        assert result.abs().sum().item() == 0.0

    def test_parse_gguf_tensors_validates_tensor_count(self):
        """parse_gguf_tensors rejects invalid tensor_count."""
        import io
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import parse_gguf_tensors
        f = io.BytesIO(b"")
        header = {"tensor_count": -1}
        result = parse_gguf_tensors(f, header)
        assert result == {}

    def test_parse_gguf_tensors_validates_large_count(self):
        """parse_gguf_tensors rejects unreasonably large tensor_count."""
        import io
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import parse_gguf_tensors
        f = io.BytesIO(b"")
        header = {"tensor_count": 200_000}
        result = parse_gguf_tensors(f, header)
        assert result == {}

    # â”€â”€ k-quants (Pass 156z9n: Q4_K + Q6_K) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def test_get_scale_min_k4_low_j(self):
        """j<4: d = scales[j] & 0x3F; m = scales[j+4] & 0x3F.

        Tight unit test on the bit-packing helper â€” without this, a
        regression that flips the j<4 / j>=4 branches passes every
        downstream Q4_K test only when the chosen scales happen to
        agree across both branches.
        """
        import numpy as np
        from enigma_engine.core.gguf_dequant import _get_scale_min_k4
        scales = np.array(
            [[0x05, 0x12, 0x21, 0x37, 0x08, 0x14, 0x22, 0x36,
              0x00, 0x00, 0x00, 0x00]],
            dtype=np.uint8,
        )
        d, m = _get_scale_min_k4(0, scales)
        assert int(d[0]) == 5
        assert int(m[0]) == 8
        d, m = _get_scale_min_k4(3, scales)
        # 0x37 & 0x3F = 0x37 = 55
        assert int(d[0]) == 0x37
        # 0x36 & 0x3F = 0x36 = 54
        assert int(m[0]) == 0x36

    def test_get_scale_min_k4_high_j(self):
        """j>=4: d = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4);
                 m = (scales[j+4] >> 4)  | ((scales[j  ] >> 6) << 4).

        Adversarial top-2-bits stitch: scales[0]=0xC5 (top bits 11),
        scales[4]=0x88 (top bits 10), scales[8]=0xCD. For j=4:
            d = (0xCD & 0xF) | ((0xC5 >> 6) << 4) = 0xD | (3<<4) = 0x3D
            m = (0xCD >> 4)  | ((0x88 >> 6) << 4) = 0xC | (2<<4) = 0x2C
        """
        import numpy as np
        from enigma_engine.core.gguf_dequant import _get_scale_min_k4
        scales = np.zeros((1, 12), dtype=np.uint8)
        scales[0, 0] = 0xC5
        scales[0, 4] = 0x88
        scales[0, 8] = 0xCD
        d, m = _get_scale_min_k4(4, scales)
        assert int(d[0]) == 0x3D
        assert int(m[0]) == 0x2C

    def test_dequantize_q4_K_zero_block(self):
        """Q4_K with empty input returns float32 zeros of requested shape."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q4_K
        result = dequantize_q4_K(b"", (256,))
        assert result.shape == (256,)
        assert result.dtype == torch.float32
        assert torch.all(result == 0)

    def test_dequantize_q4_K_handcrafted(self):
        """Q4_K dequant: y = d*sc - dmin*m for each 32-element sub-block.

        Block layout: 2 (d fp16) + 2 (dmin fp16) + 12 (scales) + 128 (qs).
        Set d=1.0, dmin=0.0; sub-block 0 (j=0): sc=2, m=0; all other
        sub-blocks: sc=0, m=0. qs byte 0 = 0x35 â†’ low nib 5 (sub-block 0
        elem 0), high nib 3 (sub-block 1 elem 0).

        Expected:
            out[0]  = 1.0 * 2 * 5 - 0.0 * 0 =  10.0   (sub-block 0)
            out[32] = 1.0 * 0 * 3 - 0.0 * 0 =   0.0   (sub-block 1, sc=0)
            out[1]  = 1.0 * 2 * 0 - 0.0 * 0 =   0.0   (sub-block 0, qs=0)
        """
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q4_K
        d = np.float16(1.0).tobytes()
        dmin = np.float16(0.0).tobytes()
        scales = bytearray(12)
        # j=0 â†’ d=scales[0]&0x3F â†’ set scales[0]=2 â†’ sc_0=2
        scales[0] = 0x02
        # j=0 â†’ m=scales[4]&0x3F â†’ set scales[4]=0 â†’ m_0=0  (already)
        qs = bytearray(128)
        qs[0] = 0x35
        block = d + dmin + bytes(scales) + bytes(qs)
        result = dequantize_q4_K(block, (256,))
        assert result.shape == (256,)
        assert result[0].item() == pytest.approx(10.0, abs=0.05)
        assert result[32].item() == pytest.approx(0.0, abs=0.05)
        assert result[1].item() == pytest.approx(0.0, abs=0.05)

    def test_dequantize_q4_K_dmin_subtracts(self):
        """Q4_K min term subtracts: with q=0 every element, dmin=1.0,
        m=3, sc=0 â†’ y = d*0*0 - 1.0*3 = -3.0 across the whole sub-block.
        Catches sign-flip regressions on the ``-m`` branch.
        """
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q4_K
        d = np.float16(1.0).tobytes()
        dmin = np.float16(1.0).tobytes()
        scales = bytearray(12)
        # j=0: sc = scales[0]&0x3F = 0 (default); m = scales[4]&0x3F â†’ 3
        scales[4] = 0x03
        qs = bytes(128)  # all q = 0
        block = d + dmin + bytes(scales) + qs
        result = dequantize_q4_K(block, (256,))
        # sub-block 0 (out[0:32]) â†’ y = 1.0 * 0 * 0 - 1.0 * 3 = -3.0
        assert result[0].item() == pytest.approx(-3.0, abs=0.05)
        assert result[31].item() == pytest.approx(-3.0, abs=0.05)
        # sub-block 1 (out[32:64]) â†’ m=scales[5]&0x3F = 0 â†’ y = 0
        assert result[32].item() == pytest.approx(0.0, abs=0.05)

    def test_dequantize_q4_K_high_j_stitch_d(self):
        """Adversarial gate on the j>=4 d-stitch (Pass 156z9s â€” sibling
        gap from Pass 156z9n's Q4_K coverage). For j=4:
            d = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4)
        Set scales[0]=0x80 â†’ scales[0]>>6 = 2 (high-stitch source);
        scales[0]&0x3F = 0 so j=0's sc stays 0 (clean). scales[8]=0x05
        â†’ low nibble = 5. Expected j=4 sc = 5 | (2<<4) = 37. With
        d_outer=1, dmin=0, qs[64]=0x01 (q=1 at sub-block 4 elem 0):
            out[128] = 1 * 37 * 1 - 0 = 37
        Drop the stitch â†’ sc=5 â†’ out[128] = 5. Adversarial.
        """
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q4_K
        d = np.float16(1.0).tobytes()
        dmin = np.float16(0.0).tobytes()
        scales = bytearray(12)
        scales[0] = 0x80
        scales[8] = 0x05
        qs = bytearray(128)
        qs[64] = 0x01            # sub-block 4 element 0 â†’ q=1
        block = d + dmin + bytes(scales) + bytes(qs)
        result = dequantize_q4_K(block, (256,))
        assert result[128].item() == pytest.approx(37.0, abs=0.05)

    def test_dequantize_q4_K_high_j_stitch_m(self):
        """Adversarial gate on the j>=4 m-stitch. For j=4:
            m = ((scales[8] >> 4) & 0x0F) | ((scales[4] >> 6) << 4)
        Note m's high-stitch source is scales[j]=scales[4], NOT
        scales[0] like d's. Set scales[4]=0x80 (high-stitch=2, m for
        j=0 stays 0 since &0x3F=0); scales[8]=0x20 (high nibble=2).
        Expected j=4 m = 2 | (2<<4) = 34. With qs=0, d_outer=1,
        dmin=1: out[128] = 1*sc*0 - 1*34 = -34. Drop stitch â†’ m=2 â†’
        out[128] = -2. Adversarial gate that the m-stitch source is
        scales[4], not scales[0] (a regression that confuses d's and
        m's stitch sources would produce wildly wrong output).
        """
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q4_K
        d = np.float16(1.0).tobytes()
        dmin = np.float16(1.0).tobytes()
        scales = bytearray(12)
        scales[4] = 0x80
        scales[8] = 0x20
        qs = bytes(128)
        block = d + dmin + bytes(scales) + qs
        result = dequantize_q4_K(block, (256,))
        assert result[128].item() == pytest.approx(-34.0, abs=0.05)

    def test_dequantize_q4_K_layout_routing(self):
        """Adversarial gate: each sub-block reads its OWN qs slot
        (low-nib for even sub-blocks, high-nib for odd) AND the j>=4
        path strides correctly to qs[64..].

        Pass 156z9w (post-audit hardening): the original Pass 156z9s
        version was weak â€” it set ONLY sub-block 0's sc nonzero and
        asserted out[32..256] == 0. A nibble-swap regression (sub-block
        1 reads low-nib instead of high-nib) was structurally invisible
        because sub-block 1's sc was zero, so out[32] stayed at 0
        either way. The test docstring claimed to catch stride bugs
        but couldn't.

        New design: give sub-blocks 0, 1, and 4 each a distinct nonzero
        sc and distinct nonzero q value, then assert each output
        position matches its OWN sub-block's product. A nibble-swap on
        the (0,1) pair flips out[0] and out[32]; a stride bug on the
        j=4 path changes out[128].
        """
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q4_K
        d = np.float16(1.0).tobytes()
        dmin = np.float16(0.0).tobytes()
        scales = bytearray(12)
        # j=0: sc = scales[0] & 0x3F = 2
        # j=1: sc = scales[1] & 0x3F = 3
        # j=4: sc = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4)
        #        = 5 | (0 << 4) = 5  (scales[0]=0x02 â†’ high-stitch=0)
        scales[0] = 0x02
        scales[1] = 0x03
        scales[8] = 0x05
        qs = bytearray(128)
        # qs[0..32] is the (0,1) pair: low nib â†’ sb0, high nib â†’ sb1.
        # Byte 0x91 â†’ sb0 elem 0 = 1, sb1 elem 0 = 9 (distinct values
        # that catch nibble-swap regressions).
        qs[0] = 0x91
        # qs[64..96] is the (4,5) pair. Byte 0x07 â†’ sb4 elem 0 = 7,
        # sb5 elem 0 = 0.  sb5 has sc=0 so its output stays 0.
        qs[64] = 0x07
        block = d + dmin + bytes(scales) + bytes(qs)
        result = dequantize_q4_K(block, (256,))
        # Sub-block 0 elem 0:  1.0 * 2 * 1 = 2.0
        assert result[0].item() == pytest.approx(2.0, abs=0.05)
        # Sub-block 1 elem 0:  1.0 * 3 * 9 = 27.0
        # A nibble-swap would give sb1 elem 0 = 1*3*1 = 3.0 here AND
        # sb0 elem 0 = 1*2*9 = 18.0 above â€” both assertions fail.
        assert result[32].item() == pytest.approx(27.0, abs=0.05)
        # Sub-block 4 elem 0 (j>=4 stitch path):  1.0 * 5 * 7 = 35.0
        # A stride bug that misroutes the j=4 qs source changes this.
        assert result[128].item() == pytest.approx(35.0, abs=0.05)
        # Sub-blocks 2, 3, 5, 6, 7 still have sc=0 â†’ their first
        # elements stay at 0 (independence check).
        for i in (64, 96, 160, 192, 224):
            assert result[i].item() == pytest.approx(0.0, abs=0.05)

    def test_dequantize_q6_K_zero_block(self):
        """Q6_K with empty input returns float32 zeros of requested shape."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q6_K
        result = dequantize_q6_K(b"", (256,))
        assert result.shape == (256,)
        assert result.dtype == torch.float32
        assert torch.all(result == 0)

    def test_dequantize_q6_K_signed_centering(self):
        """Q6_K is signed 6-bit: q = raw - 32, raw in [0, 63].

        With ql=0, qh=0 every byte â†’ raw=0 â†’ q=-32. d=1.0, scales[0]=1
        â†’ y[0..15] = 1 * 1 * (-32) = -32.0.
        """
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q6_K
        ql = bytes(128)
        qh = bytes(64)
        scales = bytearray(16)
        scales[0] = 1  # int8 +1 (covers l=0..15 of q1 path â†’ out[0..15])
        d = np.float16(1.0).tobytes()
        block = bytes(ql) + bytes(qh) + bytes(scales) + d
        result = dequantize_q6_K(block, (256,))
        assert result.shape == (256,)
        assert result[0].item() == pytest.approx(-32.0, abs=0.05)
        assert result[15].item() == pytest.approx(-32.0, abs=0.05)

    def test_dequantize_q6_K_zero_quant(self):
        """Q6_K q=0 (raw=32) produces y=0 regardless of d and scale.

        raw=32 = 0x20 = (low_nib=0) | (high_2bits=2 << 4). So set
        ql[0]=0x00 (low nib=0) and qh[0]=0x02 (bits 0-1 = 0b10 = 2).
        """
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q6_K
        ql = bytearray(128)
        qh = bytearray(64)
        qh[0] = 0x02
        scales = bytearray(16)
        scales[0] = 7   # arbitrary nonzero â€” output should still be 0
        d = np.float16(2.5).tobytes()
        block = bytes(ql) + bytes(qh) + bytes(scales) + d
        result = dequantize_q6_K(block, (256,))
        # element 0 = q1 path with l=0 â†’ uses scales[0]=7. q = (0|(2<<4))-32 = 0
        assert result[0].item() == pytest.approx(0.0, abs=0.05)

    def test_dequantize_q6_K_scale_split_within_sub_block(self):
        """Q6_K splits each 32-element output region into TWO 16-element
        slabs with DIFFERENT scales. l=0..15 uses scales[is+0],
        l=16..31 uses scales[is+1] (the +1 is critical â€” a regression
        that drops the +1 split silently rescales half of every output
        region). Catches the failure mode the author's-lens self-audit
        caught mid-implementation.
        """
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q6_K
        ql = bytearray(128)
        qh = bytearray(64)
        scales = bytearray(16)
        # Different scales for l=0..15 vs l=16..31 of the q1 path:
        scales[0] = 2   # covers out[0..15]
        scales[1] = 5   # covers out[16..31]
        # All ql=qh=0 â†’ q=-32 everywhere.
        d = np.float16(1.0).tobytes()
        block = bytes(ql) + bytes(qh) + bytes(scales) + d
        result = dequantize_q6_K(block, (256,))
        # out[0..15] = d * scales[0] * (-32) = 1 * 2 * -32 = -64
        # out[16..31] = d * scales[1] * (-32) = 1 * 5 * -32 = -160
        assert result[0].item() == pytest.approx(-64.0, abs=0.05)
        assert result[15].item() == pytest.approx(-64.0, abs=0.05)
        assert result[16].item() == pytest.approx(-160.0, abs=0.05)
        assert result[31].item() == pytest.approx(-160.0, abs=0.05)

    def test_dequantize_q6_K_layout_routing(self):
        """Q6_K element routing â€” q1 path takes ql_a low nibble + qh
        bits 0-1, q2 takes ql_b low nibble + qh bits 2-3, q3 takes
        ql_a high nibble + qh bits 4-5, q4 takes ql_b high nibble +
        qh bits 6-7. l=5 of q1 lands at output[5]; l=5 of q2 lands at
        output[37]; l=5 of q3 lands at output[69]; l=5 of q4 lands at
        output[101]. d=1.0, scales[0..7]=1 (all q1..q4 paths in half=0
        produce y = 1 * 1 * q).
        """
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q6_K
        ql = bytearray(128)
        qh = bytearray(64)
        # ql[5] low=0xA (q1 elem 5), high=0xB (q3 elem 5)
        ql[5] = 0xBA
        # ql[37] low=0xC (q2 elem 5), high=0xD (q4 elem 5)
        # (ql_b = ql[32:64], so ql_b[5] = ql[37])
        ql[37] = 0xDC
        # qh[5] bits: q1=3 (0b11), q2=2 (0b10), q3=1 (0b01), q4=0 (0b00)
        # â†’ 0b00_01_10_11 = 0x1B
        qh[5] = 0x1B
        scales = bytearray(16)
        for i in range(8):
            scales[i] = 1
        d = np.float16(1.0).tobytes()
        block = bytes(ql) + bytes(qh) + bytes(scales) + d
        result = dequantize_q6_K(block, (256,))
        # q1 elem 5: raw = 0xA | (3<<4) = 58 â†’ q = 26 â†’ y =  26
        # q2 elem 5: raw = 0xC | (2<<4) = 44 â†’ q = 12 â†’ y =  12
        # q3 elem 5: raw = 0xB | (1<<4) = 27 â†’ q = -5 â†’ y =  -5
        # q4 elem 5: raw = 0xD | (0<<4) = 13 â†’ q = -19 â†’ y = -19
        assert result[5].item() == pytest.approx(26.0, abs=0.05)
        assert result[37].item() == pytest.approx(12.0, abs=0.05)
        assert result[69].item() == pytest.approx(-5.0, abs=0.05)
        assert result[101].item() == pytest.approx(-19.0, abs=0.05)

    def test_dequantize_q6_K_second_half_independence(self):
        """Q6_K processes the 256-element super-block as two 128-element
        halves. The second half (out[128..256]) consumes ql[64..128],
        qh[32..64], and scales[8..16]. Setting only second-half bytes
        and verifying the FIRST half is untouched catches a regression
        that would index the wrong half of any of the three buffers.
        """
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q6_K
        ql = bytearray(128)
        qh = bytearray(64)
        scales = bytearray(16)
        # Touch only the second half: ql[64], qh[32], scales[8].
        scales[8] = 3   # second-half q1 path scale for l=0..15
        # ql[64]=0, qh[32]=0 â†’ q=-32 â†’ y[128] = 1 * 3 * -32 = -96
        d = np.float16(1.0).tobytes()
        block = bytes(ql) + bytes(qh) + bytes(scales) + d
        result = dequantize_q6_K(block, (256,))
        # First half: scales[0..7]=0 â†’ y[0..127] = 1 * 0 * q = 0
        assert result[0].item() == pytest.approx(0.0, abs=0.05)
        assert result[127].item() == pytest.approx(0.0, abs=0.05)
        # Second half: y[128] = 1 * 3 * -32 = -96
        assert result[128].item() == pytest.approx(-96.0, abs=0.05)
        assert result[143].item() == pytest.approx(-96.0, abs=0.05)

    # â”€â”€ Q5_K (Pass 156z9o) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def test_dequantize_q5_K_zero_block(self):
        """Q5_K with empty input returns float32 zeros of requested shape."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q5_K
        result = dequantize_q5_K(b"", (256,))
        assert result.shape == (256,)
        assert result.dtype == torch.float32
        assert torch.all(result == 0)

    def test_dequantize_q5_K_handcrafted(self):
        """Q5_K dequant: y = d*sc*q - dmin*m, q = ql_nib | (qh_bit << 4).

        Block layout: 2 (d fp16) + 2 (dmin fp16) + 12 (scales) + 32 (qh)
        + 128 (qs) = 176 bytes. Set d=1.0, dmin=0.0; sub-block 0 (j=0):
        sc=2, m=0; all other sub-blocks: sc=0, m=0. qs[0]=0x05 â†’ low
        nib 5 (sub-block 0, elem 0). qh all zeros â†’ 5th bit clear â†’ q=5.

        Expected:
            out[0]  = 1.0 * 2 * (5 + 0) - 0.0 * 0 = 10.0
            out[1]  = 1.0 * 2 * 0          = 0.0
            out[32] = 1.0 * 0 * (high nib of qs[0] = 0) = 0.0
        """
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q5_K
        d = np.float16(1.0).tobytes()
        dmin = np.float16(0.0).tobytes()
        scales = bytearray(12)
        scales[0] = 0x02   # sub_lo=0 â†’ sc=2
        qh = bytes(32)     # 5th bits all zero
        qs = bytearray(128)
        qs[0] = 0x05
        block = d + dmin + bytes(scales) + qh + bytes(qs)
        result = dequantize_q5_K(block, (256,))
        assert result.shape == (256,)
        assert result[0].item() == pytest.approx(10.0, abs=0.05)
        assert result[1].item() == pytest.approx(0.0, abs=0.05)
        assert result[32].item() == pytest.approx(0.0, abs=0.05)

    def test_dequantize_q5_K_qh_bit_lifts_value(self):
        """Q5_K 5th bit must lift q by 16. Set qs[0]=0x01 (low nib 1)
        and qh[0]=0x01 (bit 0 set, which is the low-nibble path bit for
        pair=0 â†’ sub-block 0). With sc=1, dmin=0 â†’ out[0] = 1*1*(1+16) = 17.

        Adversarial gate against a regression that drops the qh bit OR
        adds it to the wrong path (high-nibble bit_hi=1 instead of
        bit_lo=0).
        """
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q5_K
        d = np.float16(1.0).tobytes()
        dmin = np.float16(0.0).tobytes()
        scales = bytearray(12)
        scales[0] = 0x01   # sub_lo=0 â†’ sc=1
        qh = bytearray(32)
        qh[0] = 0x01       # bit 0 of qh[0] â†’ low-nibble path of pair=0
        qs = bytearray(128)
        qs[0] = 0x01
        block = d + dmin + bytes(scales) + bytes(qh) + bytes(qs)
        result = dequantize_q5_K(block, (256,))
        assert result[0].item() == pytest.approx(17.0, abs=0.05)
        # elem 1: qs[1]=0, qh[1]=0 â†’ q=0 â†’ out=0
        assert result[1].item() == pytest.approx(0.0, abs=0.05)

    def test_dequantize_q5_K_qh_bit_routes_to_correct_pair(self):
        """Q5_K bit `2*pair` of qh[l] feeds the LOW-nibble path of
        pair P (output indices 64*P + l), bit `2*pair+1` feeds the
        HIGH-nibble path of pair P (output indices 64*P + 32 + l).

        Set qh[5]=0x04 (bit 2) ONLY â†’ that's bit_lo for pair=1 â†’ lifts
        q at output index 64+5=69 by 16. With sc[pair=1 sub_lo=2]=1, qs
        all zero, dmin=0 â†’ out[69] = 1*1*(0+16) = 16. All other outputs
        = 0. Catches a regression that uses the wrong shift on qh.
        """
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q5_K
        d = np.float16(1.0).tobytes()
        dmin = np.float16(0.0).tobytes()
        scales = bytearray(12)
        # sub_lo=2 (pair=1 low) â†’ _get_scale_min_k4(2, scales): j<4 â†’
        # d = scales[2] & 0x3F. Set scales[2]=1.
        scales[2] = 0x01
        qh = bytearray(32)
        qh[5] = 0x04       # bit 2 â†’ bit_lo for pair=1
        qs = bytes(128)
        block = d + dmin + bytes(scales) + bytes(qh) + qs
        result = dequantize_q5_K(block, (256,))
        # out[64+5] = 1 * 1 * (0 + 16) - 0 = 16
        assert result[69].item() == pytest.approx(16.0, abs=0.05)
        # out[5] (pair=0 low-nib) â€” qh[5] bit 0 is clear â†’ q=0 â†’ out=0
        assert result[5].item() == pytest.approx(0.0, abs=0.05)
        # out[64+32+5]=out[101] (pair=1 high-nib) â€” qh[5] bit 3 clear â†’ q=0
        assert result[101].item() == pytest.approx(0.0, abs=0.05)

    def test_dequantize_q5_K_dmin_subtracts(self):
        """Q5_K min term subtracts. q=0 (qs=0, qh=0 â†’ ql_nib=0, 5th=0),
        dmin=1.0, m=4, sc=0 â†’ y = 1*0*0 - 1*4 = -4 across sub-block 0.
        """
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q5_K
        d = np.float16(1.0).tobytes()
        dmin = np.float16(1.0).tobytes()
        scales = bytearray(12)
        # j=0: m = scales[4] & 0x3F â†’ 4
        scales[4] = 0x04
        qh = bytes(32)
        qs = bytes(128)
        block = d + dmin + bytes(scales) + qh + qs
        result = dequantize_q5_K(block, (256,))
        assert result[0].item() == pytest.approx(-4.0, abs=0.05)
        assert result[31].item() == pytest.approx(-4.0, abs=0.05)
        # sub-block 1 (out[32:64]) â†’ m=scales[5]&0x3F=0 â†’ y=0
        assert result[32].item() == pytest.approx(0.0, abs=0.05)

    # â”€â”€ Q2_K (Pass 156z9q) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def test_dequantize_q2_K_zero_block(self):
        """Empty input â†’ float32 zeros, dtype check (sibling-parity gate)."""
        pytest.importorskip("torch")
        import torch
        from enigma_engine.core.gguf_dequant import dequantize_q2_K
        result = dequantize_q2_K(b"", (256,))
        assert result.shape == (256,)
        assert result.dtype == torch.float32
        assert torch.all(result == 0.0)

    def test_dequantize_q2_K_handcrafted(self):
        """d=1, dmin=0; scales[0]=0x02 (sc=2, mn=0), qs[0]=0x01 â†’ bits 0..1
        of byte 0 = q=1; shift=0, sub-block 0 maps element 0 to qs[0].
        Expected: out[0] = 1 * 2 * 1 - 0 = 2.0; out[1] uses qs[1] (=0).
        """
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q2_K
        scales = bytearray(16)
        scales[0] = 0x02                 # sc=2, mn=0
        qs = bytearray(64)
        qs[0] = 0x01                     # bits 0..1 = 1, all higher bits 0
        d = np.float16(1.0).tobytes()
        dmin = np.float16(0.0).tobytes()
        block = bytes(scales) + bytes(qs) + d + dmin
        result = dequantize_q2_K(block, (256,))
        assert result[0].item() == pytest.approx(2.0, abs=0.05)
        # element 1 reads qs[1]=0 â†’ q=0 â†’ y=0
        assert result[1].item() == pytest.approx(0.0, abs=0.05)
        # element 16 falls in sub-block 1 (scales[1]=0 â†’ sc=0) â†’ y=0
        assert result[16].item() == pytest.approx(0.0, abs=0.05)

    def test_dequantize_q2_K_dmin_subtracts(self):
        """q=0 everywhere; dmin=1.0; scales[0]=0x30 (sc=0, mn=3) â†’ y=-3
        across sub-block 0 (out[0..16]). Sub-block 1 (scales[1]=0) â†’ y=0.
        Catches sign-flip on the ``-ml`` term.
        """
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q2_K
        scales = bytearray(16)
        scales[0] = 0x30                 # sc=0, mn=3
        qs = bytes(64)
        d = np.float16(1.0).tobytes()
        dmin = np.float16(1.0).tobytes()
        block = bytes(scales) + qs + d + dmin
        result = dequantize_q2_K(block, (256,))
        assert result[0].item() == pytest.approx(-3.0, abs=0.05)
        assert result[15].item() == pytest.approx(-3.0, abs=0.05)
        # Sub-block 1 starts at out[16]; scales[1]=0 â†’ mn=0 â†’ y=0
        assert result[16].item() == pytest.approx(0.0, abs=0.05)

    def test_dequantize_q2_K_shift_routing(self):
        """Each sub-block uses a specific 2-bit shift of the same qs byte
        when both belong to the same nibble_half. Set qs[0]=0xE4 (binary
        ``11 10 01 00``) and put scale=1 only on the four sub-blocks that
        consume qs[0..16] in half=0:
          is=0 (j=0, shift=0): bits 0..1 = 0 â†’ out[0] = 0
          is=2 (j=1, shift=2): bits 2..3 = 1 â†’ out[32] = 1
          is=4 (j=2, shift=4): bits 4..5 = 2 â†’ out[64] = 2
          is=6 (j=3, shift=6): bits 6..7 = 3 â†’ out[96] = 3
        Catches a regression that swaps the shift mapping (e.g. uses
        ``2*nibble_half`` instead of ``2*j``) â€” that bug would land all
        four reads on the same shift and produce identical outputs.
        """
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q2_K
        scales = bytearray(16)
        for is_idx in (0, 2, 4, 6):
            scales[is_idx] = 0x01        # sc=1, mn=0
        qs = bytearray(64)
        qs[0] = 0xE4                     # 11 10 01 00 â€” different value per shift
        d = np.float16(1.0).tobytes()
        dmin = np.float16(0.0).tobytes()
        block = bytes(scales) + bytes(qs) + d + dmin
        result = dequantize_q2_K(block, (256,))
        assert result[0].item() == pytest.approx(0.0, abs=0.05)
        assert result[32].item() == pytest.approx(1.0, abs=0.05)
        assert result[64].item() == pytest.approx(2.0, abs=0.05)
        assert result[96].item() == pytest.approx(3.0, abs=0.05)

    def test_dequantize_q2_K_second_half_independence(self):
        """Touch only second-half bytes (qs[32], scales[8]); assert
        first half stays zero. Catches a regression that indexes the
        wrong half of qs or scales (e.g. drops the ``half = is // 8``
        offset and reads qs[0] for is=8).
        """
        import numpy as np
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q2_K
        scales = bytearray(16)
        scales[8] = 0x05                 # sub-block 8 (out[128..144]); sc=5
        qs = bytearray(64)
        qs[32] = 0x03                    # second half, byte 0, q=3 at shift=0
        d = np.float16(1.0).tobytes()
        dmin = np.float16(0.0).tobytes()
        block = bytes(scales) + bytes(qs) + d + dmin
        result = dequantize_q2_K(block, (256,))
        # First half stays zero
        assert result[0].item() == pytest.approx(0.0, abs=0.05)
        assert result[127].item() == pytest.approx(0.0, abs=0.05)
        # is=8: half=1, j=0, nibble_half=0 â†’ out[128] uses qs[32]
        assert result[128].item() == pytest.approx(15.0, abs=0.05)  # 1*5*3
        # is=8 covers out[128..144]; out[144] is is=9 (scales[9]=0 â†’ 0)
        assert result[144].item() == pytest.approx(0.0, abs=0.05)

    # â”€â”€ Q3_K (Pass 156z9r) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def test_dequantize_q3_K_zero_block(self):
        """Empty input â†’ float32 zeros, dtype check (sibling-parity gate)."""
        pytest.importorskip("torch")
        import torch
        from enigma_engine.core.gguf_dequant import dequantize_q3_K
        result = dequantize_q3_K(b"", (256,))
        assert result.shape == (256,)
        assert result.dtype == torch.float32
        assert torch.all(result == 0.0)

    def _q3_K_block(self, scales, hmask, qs, d_val):
        """Helper: assemble a 110-byte Q3_K block from per-buffer fragments."""
        import numpy as np
        assert len(scales) == 12 and len(hmask) == 32 and len(qs) == 64
        return bytes(hmask) + bytes(qs) + bytes(scales) + np.float16(d_val).tobytes()

    def test_dequantize_q3_K_signed_scale_zero(self):
        """signed_scale = scale_packed - 32; with scale_packed=32 â†’ dl=0,
        output is zero across sub-block 0 regardless of qs/hmask values.
        Set scales[0]=0x00 (low=0) and scales[8] bit 0..1 = 0x02 â†’ high=2 â†’
        scale_packed = 0 | (2<<4) = 32 â†’ signed=0.
        """
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q3_K
        scales = bytearray(12)
        scales[8] = 0x02
        hmask = bytearray(32)
        qs = bytearray([0xFF] * 64)        # arbitrary non-zero quants
        block = self._q3_K_block(scales, hmask, qs, 1.0)
        result = dequantize_q3_K(block, (256,))
        for i in range(16):
            assert result[i].item() == pytest.approx(0.0, abs=0.05)

    def test_dequantize_q3_K_high_bit_centering(self):
        """The hmask bit toggles the centering offset. Set d=1, scale_packed
        = 33 â†’ signed_scale = 1 â†’ dl = 1. q_low = 0 throughout.
          hmask bit clear â†’ q_full = 0 - 4 = -4 â†’ out = -4
          hmask bit set   â†’ q_full = 0 - 0 =  0 â†’ out =  0
        For is=0, hmask bit position = 0, hmask byte = hmask[0..16].
        Set scales[0]=0x01, scales[8]=0x02 â†’ scale_packed = 1 | (2<<4) = 33.
        Set hmask[0]=0x00 (bit 0 clear) and hmask[1]=0x01 (bit 0 set).
        """
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q3_K
        scales = bytearray(12)
        scales[0] = 0x01
        scales[8] = 0x02
        hmask = bytearray(32)
        hmask[0] = 0x00
        hmask[1] = 0x01
        qs = bytearray(64)
        block = self._q3_K_block(scales, hmask, qs, 1.0)
        result = dequantize_q3_K(block, (256,))
        assert result[0].item() == pytest.approx(-4.0, abs=0.05)
        assert result[1].item() == pytest.approx(0.0, abs=0.05)

    def test_dequantize_q3_K_scale_high_bit_stitch(self):
        """The 6-bit scale is the stitch of a 4-bit low nibble and 2-bit
        high pair. For is=4: low_byte_idx=4, low_shift=0, high_byte_idx=8,
        high_shift=2. Set scales[4]=0x01 (low=1), scales[8]=0x08 (bit 3,2 =
        0x02 after >>2&0x03) â†’ scale_packed = 1 | (2<<4) = 33 â†’ signed=1,
        dl=1. With qs all zero and hmask all zero â†’ q_full=-4 â†’ out[64]=-4.
        Adversarial gate: drop the 2-bit high stitch and scale_packed=1 â†’
        signed=-31 â†’ out[64] = +124, NOT -4. Catches a regression that
        forgets to combine the two scale-byte sources.
        """
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q3_K
        scales = bytearray(12)
        scales[4] = 0x01
        scales[8] = 0x08
        hmask = bytearray(32)
        qs = bytearray(64)
        block = self._q3_K_block(scales, hmask, qs, 1.0)
        result = dequantize_q3_K(block, (256,))
        # Sub-block 4 starts at out[64].
        assert result[64].item() == pytest.approx(-4.0, abs=0.05)

    def test_dequantize_q3_K_second_half_independence(self):
        """Sub-block is=8 starts the second half. Verifies:
          - low_byte_idx=0 with low_shift=4 â†’ reads scales[0] HIGH nibble.
          - high_byte_idx=8 with high_shift=4 â†’ reads scales[8] bits 4..5.
          - bit_pos=is//2=4 â†’ reads hmask[0] bit 4 (NOT bit 0).
        Set scales[0]=0x10 (high nibble=1), scales[8]=0x20 (bit 4..5 = 2
        after >>4&0x03) â†’ scale_packed = 1 | (2<<4) = 33 â†’ signed=1, dl=1.
        Toggle hmask[0] bit 4 between two blocks; out[128] flips from 0
        (bit set â†’ q_full=0) to -4 (bit clear â†’ q_full=-4). The DELTA at
        out[128] is the contract this test gates â€” adversarial against
        any regression that drops the >>4 on either scale source or that
        uses the wrong hmask bit position for the second half.
        """
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import dequantize_q3_K
        scales = bytearray(12)
        scales[0] = 0x10                  # is=8 low nibble (high half) = 1
        scales[8] = 0x20                  # is=8 high stitch bits = 2
        # Branch A: hmask[0] bit 4 SET â†’ q_full = 0 â†’ out[128] = 0
        hmask_a = bytearray(32)
        hmask_a[0] = 0x10
        qs = bytearray(64)
        block_a = self._q3_K_block(scales, hmask_a, qs, 1.0)
        result_a = dequantize_q3_K(block_a, (256,))
        assert result_a[128].item() == pytest.approx(0.0, abs=0.05)
        # Branch B: hmask[0] bit 4 CLEAR â†’ q_full = -4 â†’ out[128] = -4
        hmask_b = bytearray(32)              # bit 4 clear
        block_b = self._q3_K_block(scales, hmask_b, qs, 1.0)
        result_b = dequantize_q3_K(block_b, (256,))
        assert result_b[128].item() == pytest.approx(-4.0, abs=0.05)
        # Sub-block 9 (out[144..160]) reads scales[1] high nibble (=0) and
        # scales[9] bit 4..5 (=0) â†’ scale_packed=0 â†’ signed=-32; q_low=0,
        # hmask[16] bit 4=0 â†’ q_full=-4 â†’ out[144] = -32 * -4 = 128 in
        # both blocks. Verifies sub-block 9 is independent of sub-block 8.
        assert result_a[144].item() == pytest.approx(128.0, abs=0.05)
        assert result_b[144].item() == pytest.approx(128.0, abs=0.05)

    # â”€â”€ Dispatcher routing (Pass 156z9p, closes A2 from 156z9h-audit) â”€â”€

    @pytest.mark.parametrize(
        "tt_id,block_size,bytes_per_block,expected_fn",
        [
            (2,  32,  18,  "dequantize_q4_0"),
            (3,  32,  20,  "dequantize_q4_1"),
            (6,  32,  22,  "dequantize_q5_0"),
            (7,  32,  24,  "dequantize_q5_1"),
            (8,  32,  34,  "dequantize_q8_0"),
            (10, 256, 84,  "dequantize_q2_K"),
            (11, 256, 110, "dequantize_q3_K"),
            (12, 256, 144, "dequantize_q4_K"),
            (13, 256, 176, "dequantize_q5_K"),
            (14, 256, 210, "dequantize_q6_K"),
        ],
    )
    def test_parse_gguf_tensors_dispatch_routing(
        self, tt_id, block_size, bytes_per_block, expected_fn,
    ):
        """Each quantized type ID routes to the correct dequantizer with
        the correct bytes-per-block. Catches tuple swaps / off-by-one /
        missing branches in the dispatch chain in `parse_gguf_tensors`.

        Test shape: monkeypatch all 8 quantized dequant functions in the
        gguf_dequant module with sentinel recorders, feed a synthetic
        single-tensor GGUF body through `parse_gguf_tensors`, and assert
        the right recorder fired exactly once with the right byte count
        and shape.

        F32 / F16 routing is excluded â€” those branches use `np.fromfile`
        (incompatible with BytesIO) and have no conditional dispatch
        worth testing (one-line `np.fromfile + reshape`).
        """
        import io
        import struct
        torch = pytest.importorskip("torch")
        from enigma_engine.core import gguf_dequant as gd

        all_fn_names = [
            "dequantize_q4_0", "dequantize_q4_1", "dequantize_q5_0",
            "dequantize_q5_1", "dequantize_q8_0", "dequantize_q2_K",
            "dequantize_q3_K", "dequantize_q4_K", "dequantize_q5_K",
            "dequantize_q6_K",
        ]
        calls = {}

        def _make_recorder(name):
            def _rec(data, shape):
                calls[name] = (len(data), tuple(shape))
                return torch.zeros(shape, dtype=torch.float32)
            return _rec

        # Build the byte stream parse_gguf_tensors reads after the header:
        # name_len(Q) + b"x" + n_dims(I)=1 + dim(Q) + tensor_type(I) +
        # offset(Q)=0 + 32-byte alignment + payload.
        name = b"x"
        body = b""
        body += struct.pack("<Q", len(name))
        body += name
        body += struct.pack("<I", 1)
        body += struct.pack("<Q", block_size)
        body += struct.pack("<I", tt_id)
        body += struct.pack("<Q", 0)
        body += b"\x00" * ((32 - len(body) % 32) % 32)
        body += b"\x00" * bytes_per_block

        originals = {n: getattr(gd, n) for n in all_fn_names}
        try:
            for n in all_fn_names:
                setattr(gd, n, _make_recorder(n))
            tensors = gd.parse_gguf_tensors(
                io.BytesIO(body), {"tensor_count": 1},
            )
        finally:
            for n, fn in originals.items():
                setattr(gd, n, fn)

        assert len(calls) == 1, (
            f"expected exactly one dequant call, got {list(calls.keys())}"
        )
        assert expected_fn in calls
        data_len, shape = calls[expected_fn]
        assert data_len == bytes_per_block
        assert shape == (block_size,)
        assert "x" in tensors

    def test_parse_gguf_tensors_unknown_type_skipped(self):
        """Unknown tensor type IDs hit the fall-through branch and are
        skipped without raising. Uses type_id=99 (not in the ggml enum).
        """
        import io
        import struct
        pytest.importorskip("torch")
        from enigma_engine.core.gguf_dequant import parse_gguf_tensors

        body = b""
        body += struct.pack("<Q", 1)            # name_len
        body += b"x"
        body += struct.pack("<I", 1)            # n_dims
        body += struct.pack("<Q", 32)           # dim
        body += struct.pack("<I", 99)           # unknown tensor_type
        body += struct.pack("<Q", 0)            # offset
        pad = (32 - len(body) % 32) % 32
        body += b"\x00" * pad
        body += b"\x00" * 64                    # arbitrary trailing payload

        tensors = parse_gguf_tensors(io.BytesIO(body), {"tensor_count": 1})
        assert tensors == {}


# â”€â”€ Pass 32 tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestSlidingWindowMask:
    """_get_causal_mask applies sliding_window when config field is set."""

    def test_no_sliding_window_default(self):
        """Without sliding_window, mask is standard upper-triangular."""
        pytest.importorskip("torch")
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_presets import ForgeConfig

        cfg = ForgeConfig(dim=64, n_layers=1, n_heads=2, n_kv_heads=1,
                          vocab_size=100, max_seq_len=32)
        model = Enigma(config=cfg)
        mask = model._get_causal_mask(6)
        # Position (5, 0) should be 0.0 â€” token 5 can attend to token 0
        assert mask[5, 0] == 0.0

    def test_sliding_window_masks_distant_tokens(self):
        """With sliding_window=2, token 5 cannot attend to token 0."""
        pytest.importorskip("torch")
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_presets import ForgeConfig

        cfg = ForgeConfig(dim=64, n_layers=1, n_heads=2, n_kv_heads=1,
                          vocab_size=100, max_seq_len=32, sliding_window=2)
        model = Enigma(config=cfg)
        mask = model._get_causal_mask(6)
        # Token 5 can only attend to positions 3, 4, 5 (window of 2)
        assert mask[5, 0] == float('-inf'), "position 0 beyond window"
        assert mask[5, 1] == float('-inf'), "position 1 beyond window"
        assert mask[5, 2] == float('-inf'), "position 2 beyond window"
        assert mask[5, 3] == 0.0, "position 3 within window"
        assert mask[5, 4] == 0.0, "position 4 within window"
        assert mask[5, 5] == 0.0, "self-attend always ok"
        # Upper triangle still masked
        assert mask[0, 1] == float('-inf')

    def test_sliding_window_still_causal(self):
        """Sliding window mask doesn't break causality (future still -inf)."""
        pytest.importorskip("torch")
        from enigma_engine.core.model import Enigma
        from enigma_engine.core.model_presets import ForgeConfig

        cfg = ForgeConfig(dim=64, n_layers=1, n_heads=2, n_kv_heads=1,
                          vocab_size=100, max_seq_len=32, sliding_window=3)
        model = Enigma(config=cfg)
        mask = model._get_causal_mask(8)
        for i in range(8):
            for j in range(i + 1, 8):
                assert mask[i, j] == float('-inf'), (
                    f"mask[{i},{j}] should be -inf (future)")


@pytest.mark.structural
class TestBPEDropout:
    """BPE-Dropout (subword regularization) for training."""

    def test_dropout_produces_different_tokenizations(self):
        """With dropout > 0, the same word can produce different splits."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        # Add merges but do NOT add whole-word entries to token_to_id
        # so the merge loop actually runs
        tok.merge_ranks = {
            ('h', 'e'): 0, ('he', 'l'): 1, ('hel', 'l'): 2,
            ('hell', 'o'): 3, ('hello', '</w>'): 4,
        }
        tok.token_to_id = {
            'h': 10, 'e': 11, 'l': 12, 'o': 13, '</w>': 14,
            'he': 15, 'hel': 16, 'hell': 17, 'hello': 18,
            # 'hello</w>' deliberately omitted â€” forces merge path
        }
        tok.id_to_token = {v: k for k, v in tok.token_to_id.items()}
        # Canonical (no dropout) should always give the same result
        canonical = tok._tokenize_word('hello', dropout=0.0)
        assert canonical == tok._tokenize_word('hello', dropout=0.0)
        # High dropout should sometimes produce a different tokenization
        seen_different = False
        for _ in range(50):
            result = tok._tokenize_word('hello', dropout=0.5)
            if result != canonical:
                seen_different = True
                break
        assert seen_different, (
            "dropout=0.5 should produce at least one different "
            "tokenization in 50 tries")

    def test_dropout_zero_uses_cache(self):
        """dropout=0 uses the cache; dropout>0 bypasses it."""
        from enigma_engine.core.bpe_tokenizer import BPETokenizer
        tok = BPETokenizer()
        tok.merge_ranks = {('t', 'e'): 0, ('te', 's'): 1,
                           ('tes', 't'): 2, ('test', '</w>'): 3}
        tok.token_to_id = {
            't': 10, 'e': 11, 's': 12, '</w>': 13,
            'te': 14, 'tes': 15, 'test': 16,
            # 'test</w>' deliberately omitted
        }
        tok.id_to_token = {v: k for k, v in tok.token_to_id.items()}
        tok._tokenize_word('test', dropout=0.0)
        assert 'test' in tok.cache
        tok.cache.clear()
        tok._tokenize_word('test', dropout=0.1)
        assert 'test' not in tok.cache  # stochastic result not cached

    def test_training_config_has_bpe_dropout(self):
        """TrainingConfig has bpe_dropout field defaulting to 0.1."""
        from enigma_engine.training.training import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.bpe_dropout == 0.1

    def test_training_config_validates_bpe_dropout(self):
        """bpe_dropout must be in [0, 1)."""
        from enigma_engine.training.training import TrainingConfig
        bad = TrainingConfig(bpe_dropout=1.0)
        with pytest.raises(ValueError, match="bpe_dropout"):
            bad.validate()
        bad2 = TrainingConfig(bpe_dropout=-0.1)
        with pytest.raises(ValueError, match="bpe_dropout"):
            bad2.validate()


@pytest.mark.structural
class TestMinPSampling:
    """sample_next_token supports min_p filtering."""

    def test_min_p_filters_low_probability_tokens(self):
        """min_p > 0 removes tokens below min_p * max_probability."""
        torch = pytest.importorskip("torch")
        from enigma_engine.core.model_utils import sample_next_token

        # Create logits where one token dominates heavily
        logits = torch.full((1, 100), -10.0)
        logits[0, 42] = 5.0   # Very high â€” this is the max prob token
        logits[0, 7] = 4.9    # Close to max â€” should survive min_p
        logits[0, 99] = -5.0  # Much lower â€” should be filtered by min_p

        generated = torch.tensor([[42]])  # dummy history

        # With min_p=0.0, all tokens are candidates
        # With min_p=0.5, only tokens with prob >= 0.5 * max_prob survive
        # Run many samples with high min_p to verify filtering
        results = set()
        for _ in range(200):
            tok = sample_next_token(
                logits.clone(), generated, temperature=1.0,
                top_k=0, top_p=1.0, repetition_penalty=1.0,
                min_p=0.5)
            results.add(tok.item())

        # Token 42 and 7 should appear (close probs), but token 99 should not
        assert 42 in results, "Dominant token should always be sampled"
        assert 99 not in results, (
            "Low-probability token should be filtered by min_p=0.5")


@pytest.mark.structural
class TestRAGVocabCap:
    """RAG MAX_VOCAB should be 16000 for adequate vocabulary coverage."""

    def test_max_vocab_is_16000(self):
        """MAX_VOCAB constant should be 16000."""
        from enigma_engine.core.rag import MAX_VOCAB
        assert MAX_VOCAB == 16000, (
            f"MAX_VOCAB should be 16000, got {MAX_VOCAB}")


# ================================================================
# Pass 40 â€” S572, S573, S575
# ================================================================

class TestStreamFinishWithoutStart:
    """S572: finish() must not crash if start() was never called."""

    def test_finish_without_start_no_crash(self):
        """Calling finish() without start() should produce duration_ms=0."""
        from enigma_engine.core.streaming import StreamingResponse, StreamingConfig
        sp = StreamingResponse(config=StreamingConfig())
        sp.finish()
        assert sp._finished
        # Last chunk should be END with duration_ms == 0
        assert len(sp._chunks) > 0
        end_chunk = sp._chunks[-1]
        assert end_chunk.metadata["duration_ms"] == 0.0

    def test_finish_uses_start_time_when_available(self):
        """finish() computes real duration when start() was called."""
        import time
        from enigma_engine.core.streaming import StreamingResponse, StreamingConfig
        sp = StreamingResponse(config=StreamingConfig())
        sp.start()
        time.sleep(0.01)
        sp.finish()
        end_chunk = sp._chunks[-1]
        assert end_chunk.metadata["duration_ms"] > 0


# ================================================================
# AI Profile â€” Load / Save / List / Manager lifecycle
# ================================================================

class TestAIProfileLifecycle:
    """Tests for ai_profile.py file I/O and manager operations."""

    def test_to_dict_roundtrip(self):
        """AIProfile -> dict -> AIProfile preserves all fields."""
        from enigma_engine.core.ai_profile import (
            AIProfile, GenerationConfig, MemoryConfig,
        )
        original = AIProfile(
            name="Test AI", id="test_ai", version="2.0",
            description="A test profile",
            model_path="models/test.pth", model_type="pytorch",
            system_prompt="Be helpful.",
            generation=GenerationConfig(temperature=0.5, top_k=20),
            memory=MemoryConfig(conversation_dir="memory/test_ai"),
            commands=["file.read"], disabled_commands=["system.exec"],
            author="tester", tags=["test"],
        )
        d = original.to_dict()
        restored = AIProfile.from_dict(d)
        assert restored.name == original.name
        assert restored.id == original.id
        assert restored.generation.temperature == 0.5
        assert restored.generation.top_k == 20
        assert restored.memory.conversation_dir == "memory/test_ai"
        assert restored.commands == ["file.read"]
        assert restored.disabled_commands == ["system.exec"]
        assert restored.tags == ["test"]

    def test_from_dict_filters_unknown_keys(self):
        """Unknown keys are silently ignored, not raised."""
        from enigma_engine.core.ai_profile import AIProfile
        d = {"name": "X", "id": "x", "bogus_field": 999,
             "generation": {"temperature": 0.3, "fake_key": True}}
        profile = AIProfile.from_dict(d)
        assert profile.name == "X"
        assert profile.generation.temperature == 0.3
        assert not hasattr(profile, "bogus_field")

    def test_can_use_command_all_allowed(self):
        """Empty commands list means all allowed."""
        from enigma_engine.core.ai_profile import AIProfile
        p = AIProfile(commands=[], disabled_commands=[])
        assert p.can_use_command("anything") is True

    def test_can_use_command_allowlist(self):
        """Only listed commands are allowed when commands is non-empty."""
        from enigma_engine.core.ai_profile import AIProfile
        p = AIProfile(commands=["file.read", "note.add"])
        assert p.can_use_command("file.read") is True
        assert p.can_use_command("system.exec") is False

    def test_can_use_command_disabled_overrides(self):
        """Disabled list takes priority over everything."""
        from enigma_engine.core.ai_profile import AIProfile
        p = AIProfile(commands=[], disabled_commands=["system.exec"])
        assert p.can_use_command("system.exec") is False
        assert p.can_use_command("file.read") is True

    def test_save_and_load_roundtrip(self, tmp_path):
        """save_profile -> load_profile preserves data through disk."""
        from enigma_engine.core.ai_profile import (
            AIProfile, save_profile, load_profile, GenerationConfig,
        )
        original = AIProfile(
            name="Disk Test", id="disk_test",
            system_prompt="Round-trip test.",
            generation=GenerationConfig(temperature=0.42),
            tags=["roundtrip"],
        )
        path = tmp_path / "test_profile.json"
        save_profile(original, str(path))
        assert path.exists()

        loaded = load_profile(str(path))
        assert loaded.name == "Disk Test"
        assert loaded.id == "disk_test"
        assert loaded.generation.temperature == 0.42
        assert loaded.tags == ["roundtrip"]

    def test_load_profile_missing_file(self):
        """load_profile raises FileNotFoundError for missing file."""
        from enigma_engine.core.ai_profile import load_profile
        with pytest.raises(FileNotFoundError):
            load_profile("nonexistent/path/profile.json")

    def test_load_profile_invalid_json(self, tmp_path):
        """load_profile raises ValueError for bad JSON."""
        from enigma_engine.core.ai_profile import load_profile
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_profile(str(bad))

    def test_list_profiles_empty_dir(self, tmp_path):
        """list_profiles returns [] for empty directory."""
        from enigma_engine.core.ai_profile import list_profiles
        assert list_profiles(str(tmp_path)) == []

    def test_list_profiles_finds_profiles(self, tmp_path):
        """list_profiles discovers saved profile files."""
        import json
        from enigma_engine.core.ai_profile import list_profiles
        (tmp_path / "a.json").write_text(
            json.dumps({"name": "Alpha", "id": "alpha",
                         "description": "First"}),
            encoding="utf-8")
        (tmp_path / "b.json").write_text(
            json.dumps({"name": "Beta", "id": "beta",
                         "description": "Second"}),
            encoding="utf-8")
        result = list_profiles(str(tmp_path))
        assert len(result) == 2
        names = {p["name"] for p in result}
        assert names == {"Alpha", "Beta"}

    def test_list_profiles_nonexistent_dir(self):
        """list_profiles returns [] for directory that doesn't exist."""
        from enigma_engine.core.ai_profile import list_profiles
        assert list_profiles("no_such_directory_xyz") == []

    def test_manager_load_switch_unload(self, tmp_path):
        """AIProfileManager load -> switch -> unload lifecycle."""
        from enigma_engine.core.ai_profile import (
            AIProfileManager, AIProfile, save_profile,
        )
        manager = AIProfileManager(str(tmp_path))
        # Save two profiles
        p1 = AIProfile(name="One", id="one")
        p2 = AIProfile(name="Two", id="two")
        save_profile(p1, str(tmp_path / "one.json"))
        save_profile(p2, str(tmp_path / "two.json"))

        # Load and switch
        manager.load_profile(str(tmp_path / "one.json"))
        assert "one" in manager.list_loaded()
        assert manager.active_profile is None  # not switched yet

        manager.switch_profile("one")
        assert manager.active_profile.name == "One"

        manager.switch_profile("two")
        assert manager.active_profile.name == "Two"

        # Unload active
        manager.unload_profile("two")
        assert manager.active_profile is None
        assert "two" not in manager.list_loaded()

    def test_manager_create_profile(self, tmp_path):
        """AIProfileManager.create_profile saves to disk."""
        from enigma_engine.core.ai_profile import AIProfileManager
        manager = AIProfileManager(str(tmp_path))
        profile = manager.create_profile(
            "My Custom AI", model_path="models/custom.pth")
        assert profile.id == "my_custom_ai"
        assert (tmp_path / "my_custom_ai.json").exists()

    def test_manager_switch_triggers_callback(self, tmp_path):
        """on_profile_switched callback fires during switch."""
        from enigma_engine.core.ai_profile import (
            AIProfileManager, AIProfile, save_profile,
        )
        manager = AIProfileManager(str(tmp_path))
        save_profile(AIProfile(name="A", id="a"), str(tmp_path / "a.json"))

        calls = []
        manager.on_profile_switched = lambda old, new: calls.append(
            (old, new.id))
        manager.load_profile(str(tmp_path / "a.json"))
        manager.switch_profile("a")
        assert len(calls) == 1
        assert calls[0] == (None, "a")

    def test_manager_load_triggers_callback(self, tmp_path):
        """on_profile_loaded callback fires during load."""
        from enigma_engine.core.ai_profile import (
            AIProfileManager, AIProfile, save_profile,
        )
        manager = AIProfileManager(str(tmp_path))
        save_profile(AIProfile(name="B", id="b"), str(tmp_path / "b.json"))

        loaded_ids = []
        manager.on_profile_loaded = lambda p: loaded_ids.append(p.id)
        manager.load_profile(str(tmp_path / "b.json"))
        assert loaded_ids == ["b"]


# ================================================================
# Config defaults â€” get/update/save/load/env persistence
# ================================================================

class TestConfigPersistence:
    """Tests for config/defaults.py save/load/env functions."""

    def test_get_config_returns_known_key(self):
        """get_config retrieves known keys from CONFIG."""
        from enigma_engine.config.defaults import get_config
        val = get_config("temperature")
        assert isinstance(val, float)

    def test_get_config_default_for_missing_key(self):
        """get_config returns default for unknown keys."""
        from enigma_engine.config.defaults import get_config
        assert get_config("no_such_key_xyz", default=42) == 42

    def test_update_config_modifies_in_memory(self):
        """update_config changes CONFIG in memory."""
        from enigma_engine.config.defaults import CONFIG, update_config
        old = CONFIG.get("temperature")
        try:
            update_config({"temperature": 0.123})
            assert CONFIG["temperature"] == 0.123
        finally:
            CONFIG["temperature"] = old

    def test_update_config_rejects_non_dict(self):
        """update_config raises TypeError for non-dict."""
        from enigma_engine.config.defaults import update_config
        with pytest.raises(TypeError, match="must be a dict"):
            update_config("not a dict")

    def test_save_config_writes_json(self, tmp_path):
        """save_config writes CONFIG to a JSON file."""
        import json
        from enigma_engine.config.defaults import save_config
        out = tmp_path / "saved_config.json"
        save_config(str(out))
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert "temperature" in data

    def test_save_config_creates_parent_dirs(self, tmp_path):
        """save_config creates parent directories if needed."""
        from enigma_engine.config.defaults import save_config
        out = tmp_path / "sub" / "dir" / "config.json"
        save_config(str(out))
        assert out.exists()

    def test_validate_config_types_strips_wrong_types(self):
        """_validate_config_types removes values with wrong types."""
        from enigma_engine.config.defaults import _validate_config_types
        cleaned = _validate_config_types({
            "temperature": "not_a_float",  # should be float
            "api_host": 12345,             # should be str
            "epochs": 5,                   # correct int
        })
        assert "temperature" not in cleaned
        assert "api_host" not in cleaned
        assert cleaned["epochs"] == 5

    def test_validate_config_types_passes_unknown_keys(self):
        """Unknown keys are passed through unchanged."""
        from enigma_engine.config.defaults import _validate_config_types
        cleaned = _validate_config_types({"my_custom_key": [1, 2, 3]})
        assert cleaned["my_custom_key"] == [1, 2, 3]

    def test_load_env_config_applies_env_vars(self, monkeypatch):
        """Environment variables override CONFIG values."""
        from enigma_engine.config import defaults
        old_val = defaults.CONFIG.get("device")
        try:
            monkeypatch.setenv("FORGE_DEVICE", "test_device")
            defaults._load_env_config()
            assert defaults.CONFIG["device"] == "test_device"
        finally:
            defaults.CONFIG["device"] = old_val

    def test_load_env_config_validates_port(self, monkeypatch):
        """Invalid port numbers in env vars are rejected."""
        from enigma_engine.config import defaults
        old_port = defaults.CONFIG.get("api_port")
        try:
            monkeypatch.setenv("FORGE_API_PORT", "99999")
            defaults._load_env_config()
            # Port should remain unchanged (99999 > 65535)
            assert defaults.CONFIG.get("api_port") == old_port
        finally:
            defaults.CONFIG["api_port"] = old_port

    def test_load_env_config_valid_port(self, monkeypatch):
        """Valid port numbers in env vars are accepted."""
        from enigma_engine.config import defaults
        old_port = defaults.CONFIG.get("api_port")
        try:
            monkeypatch.setenv("FORGE_API_PORT", "8080")
            defaults._load_env_config()
            assert defaults.CONFIG["api_port"] == 8080
        finally:
            defaults.CONFIG["api_port"] = old_port

    def test_load_env_config_non_numeric_port(self, monkeypatch):
        """Non-numeric port in env var is rejected."""
        from enigma_engine.config import defaults
        old_port = defaults.CONFIG.get("api_port")
        try:
            monkeypatch.setenv("FORGE_API_PORT", "abc")
            defaults._load_env_config()
            assert defaults.CONFIG.get("api_port") == old_port
        finally:
            defaults.CONFIG["api_port"] = old_port

    def test_load_user_config_from_file(self, tmp_path, monkeypatch):
        """_load_user_config reads forge_config.json from CWD."""
        import json
        from enigma_engine.config import defaults

        config_file = tmp_path / "forge_config.json"
        config_file.write_text(
            json.dumps({"log_level": "DEBUG_TEST"}),
            encoding="utf-8")

        old_val = defaults.CONFIG.get("log_level")
        old_init = defaults._initialized
        try:
            monkeypatch.chdir(tmp_path)
            defaults._load_user_config()
            assert defaults.CONFIG["log_level"] == "DEBUG_TEST"
        finally:
            defaults.CONFIG["log_level"] = old_val
            defaults._initialized = old_init

    def test_load_user_config_skips_invalid_json(self, tmp_path, monkeypatch):
        """_load_user_config skips files with invalid JSON."""
        from enigma_engine.config import defaults

        config_file = tmp_path / "forge_config.json"
        config_file.write_text("{bad json", encoding="utf-8")

        old_val = defaults.CONFIG.get("log_level")
        try:
            monkeypatch.chdir(tmp_path)
            # Should not raise
            defaults._load_user_config()
            # CONFIG should remain unchanged
            assert defaults.CONFIG.get("log_level") == old_val
        finally:
            pass


# ================================================================
# Auto Research â€” auto_research() function tests
# ================================================================

class TestAutoResearchFunction:
    """Tests for the auto_research() function end-to-end with mocked web."""

    def test_empty_query_returns_empty(self):
        """Empty/short queries return empty string."""
        from enigma_engine.core.auto_research import auto_research
        assert auto_research("") == ""
        assert auto_research("ab") == ""

    def test_returns_formatted_context(self, monkeypatch):
        """auto_research formats search results into context block."""
        from enigma_engine.core import auto_research as ar_mod

        # Reset state for clean test
        old_cache = ar_mod._search_cache.copy()
        old_time = ar_mod._last_search_time
        try:
            ar_mod._search_cache.clear()
            ar_mod._last_search_time = 0.0

            # Mock ddg_search to return fake results
            fake_results = [
                {"title": "Python Guide", "snippet": "Learn Python",
                 "url": ""},
            ]
            monkeypatch.setattr(
                "enigma_engine.core.web_utils.ddg_search",
                lambda q, max_results=3: fake_results)
            monkeypatch.setattr(
                "enigma_engine.core.web_utils.ddg_image_search",
                lambda q, max_results=2: [])

            result = ar_mod.auto_research("what is python programming")
            assert "[WEB RESEARCH" in result
            assert "Python Guide" in result
            assert "Learn Python" in result
            assert "[END WEB RESEARCH]" in result
        finally:
            ar_mod._search_cache.clear()
            ar_mod._search_cache.update(old_cache)
            ar_mod._last_search_time = old_time

    def test_caches_result(self, monkeypatch):
        """Second call with same query returns cached result."""
        from enigma_engine.core import auto_research as ar_mod

        old_cache = ar_mod._search_cache.copy()
        old_time = ar_mod._last_search_time
        call_count = [0]

        def fake_search(q, max_results=3):
            call_count[0] += 1
            return [{"title": "Result", "snippet": "Info", "url": ""}]

        try:
            ar_mod._search_cache.clear()
            ar_mod._last_search_time = 0.0

            monkeypatch.setattr(
                "enigma_engine.core.web_utils.ddg_search", fake_search)
            monkeypatch.setattr(
                "enigma_engine.core.web_utils.ddg_image_search",
                lambda q, max_results=2: [])

            r1 = ar_mod.auto_research("test caching query here")
            assert call_count[0] == 1
            r2 = ar_mod.auto_research("test caching query here")
            assert call_count[0] == 1  # no second web call
            assert r1 == r2
        finally:
            ar_mod._search_cache.clear()
            ar_mod._search_cache.update(old_cache)
            ar_mod._last_search_time = old_time

    def test_rate_limited_returns_empty(self, monkeypatch):
        """Rate-limited calls return empty string."""
        from enigma_engine.core import auto_research as ar_mod
        import time

        old_cache = ar_mod._search_cache.copy()
        old_time = ar_mod._last_search_time
        try:
            ar_mod._search_cache.clear()
            # Set last search to now â€” next call should be rate-limited
            ar_mod._last_search_time = time.monotonic()

            result = ar_mod.auto_research("should be rate limited query")
            assert result == ""
        finally:
            ar_mod._search_cache.clear()
            ar_mod._search_cache.update(old_cache)
            ar_mod._last_search_time = old_time

    def test_no_results_caches_empty(self, monkeypatch):
        """Empty search results are cached as empty string."""
        from enigma_engine.core import auto_research as ar_mod

        old_cache = ar_mod._search_cache.copy()
        old_time = ar_mod._last_search_time
        try:
            ar_mod._search_cache.clear()
            ar_mod._last_search_time = 0.0

            monkeypatch.setattr(
                "enigma_engine.core.web_utils.ddg_search",
                lambda q, max_results=3: [])

            result = ar_mod.auto_research("no results query test")
            assert result == ""

            # Verify it was cached
            key = ar_mod._normalize_query("no results query test")
            assert ar_mod._cache_get(key) == ""
        finally:
            ar_mod._search_cache.clear()
            ar_mod._search_cache.update(old_cache)
            ar_mod._last_search_time = old_time

    def test_search_exception_returns_empty(self, monkeypatch):
        """If ddg_search raises, auto_research returns empty."""
        from enigma_engine.core import auto_research as ar_mod

        old_cache = ar_mod._search_cache.copy()
        old_time = ar_mod._last_search_time
        try:
            ar_mod._search_cache.clear()
            ar_mod._last_search_time = 0.0

            def raise_error(q, max_results=3):
                raise ConnectionError("Network down")

            monkeypatch.setattr(
                "enigma_engine.core.web_utils.ddg_search", raise_error)

            result = ar_mod.auto_research("error query test here")
            assert result == ""
        finally:
            ar_mod._search_cache.clear()
            ar_mod._search_cache.update(old_cache)
            ar_mod._last_search_time = old_time

    def test_image_results_included(self, monkeypatch):
        """Image search results are appended to context."""
        from enigma_engine.core import auto_research as ar_mod

        old_cache = ar_mod._search_cache.copy()
        old_time = ar_mod._last_search_time
        try:
            ar_mod._search_cache.clear()
            ar_mod._last_search_time = 0.0

            monkeypatch.setattr(
                "enigma_engine.core.web_utils.ddg_search",
                lambda q, max_results=3: [
                    {"title": "Main", "snippet": "Info", "url": ""}])
            monkeypatch.setattr(
                "enigma_engine.core.web_utils.ddg_image_search",
                lambda q, max_results=2: [
                    {"title": "photo", "url": "https://img.example.com/a.jpg"}
                ])

            result = ar_mod.auto_research("image test query here")
            assert "![photo]" in result
            assert "https://img.example.com/a.jpg" in result
        finally:
            ar_mod._search_cache.clear()
            ar_mod._search_cache.update(old_cache)
            ar_mod._last_search_time = old_time


# ================================================================
# Safe Save â€” atomic_safetensors_save
# ================================================================

class TestAtomicSafetensorsSave:
    """Tests for atomic_safetensors_save atomic write pattern."""

    def test_atomic_save_writes_file(self, tmp_path, monkeypatch):
        """atomic_safetensors_save creates the target file."""
        from enigma_engine.core import safe_save

        saved_calls = []

        def mock_save_file(tensors, path, metadata=None):
            saved_calls.append((tensors, path, metadata))
            Path(path).write_text("fake", encoding="utf-8")

        monkeypatch.setattr(
            "safetensors.torch.save_file", mock_save_file,
            raising=False)
        # Also mock the import inside the function
        import sys
        if "safetensors" not in sys.modules:
            import types
            st = types.ModuleType("safetensors")
            st_torch = types.ModuleType("safetensors.torch")
            st_torch.save_file = mock_save_file
            st.torch = st_torch
            monkeypatch.setitem(sys.modules, "safetensors", st)
            monkeypatch.setitem(sys.modules, "safetensors.torch", st_torch)
        else:
            monkeypatch.setattr(
                "safetensors.torch.save_file", mock_save_file)

        target = tmp_path / "weights.safetensors"
        safe_save.atomic_safetensors_save({"w": "fake_tensor"}, target)
        assert target.exists()
        assert len(saved_calls) == 1

    def test_atomic_save_cleans_tmp_on_failure(self, tmp_path, monkeypatch):
        """On failure, temp file is removed and target is untouched."""
        from enigma_engine.core import safe_save
        import sys
        import types

        def mock_save_file(tensors, path, metadata=None):
            Path(path).write_text("partial", encoding="utf-8")
            raise OSError("Disk full")

        if "safetensors" not in sys.modules:
            st = types.ModuleType("safetensors")
            st_torch = types.ModuleType("safetensors.torch")
            st_torch.save_file = mock_save_file
            st.torch = st_torch
            monkeypatch.setitem(sys.modules, "safetensors", st)
            monkeypatch.setitem(sys.modules, "safetensors.torch", st_torch)
        else:
            monkeypatch.setattr(
                "safetensors.torch.save_file", mock_save_file)

        target = tmp_path / "weights.safetensors"
        with pytest.raises(IOError, match="Disk full"):
            safe_save.atomic_safetensors_save({"w": "fake"}, target)

        assert not target.exists()
        tmp_file = target.with_suffix(".safetensors.tmp")
        assert not tmp_file.exists()

    def test_atomic_save_creates_parent_dirs(self, tmp_path, monkeypatch):
        """Parent directories are created if missing."""
        from enigma_engine.core import safe_save
        import sys
        import types

        def mock_save_file(tensors, path, metadata=None):
            Path(path).write_text("ok", encoding="utf-8")

        if "safetensors" not in sys.modules:
            st = types.ModuleType("safetensors")
            st_torch = types.ModuleType("safetensors.torch")
            st_torch.save_file = mock_save_file
            st.torch = st_torch
            monkeypatch.setitem(sys.modules, "safetensors", st)
            monkeypatch.setitem(sys.modules, "safetensors.torch", st_torch)
        else:
            monkeypatch.setattr(
                "safetensors.torch.save_file", mock_save_file)

        target = tmp_path / "sub" / "dir" / "weights.safetensors"
        safe_save.atomic_safetensors_save({"w": "fake"}, target)
        assert target.exists()

    def test_atomic_save_passes_metadata(self, tmp_path, monkeypatch):
        """Metadata dict is forwarded to save_file."""
        from enigma_engine.core import safe_save
        import sys
        import types

        received_meta = []

        def mock_save_file(tensors, path, metadata=None):
            received_meta.append(metadata)
            Path(path).write_text("ok", encoding="utf-8")

        if "safetensors" not in sys.modules:
            st = types.ModuleType("safetensors")
            st_torch = types.ModuleType("safetensors.torch")
            st_torch.save_file = mock_save_file
            st.torch = st_torch
            monkeypatch.setitem(sys.modules, "safetensors", st)
            monkeypatch.setitem(sys.modules, "safetensors.torch", st_torch)
        else:
            monkeypatch.setattr(
                "safetensors.torch.save_file", mock_save_file)

        target = tmp_path / "meta.safetensors"
        meta = {"format": "pt", "version": "1"}
        safe_save.atomic_safetensors_save(
            {"w": "fake"}, target, metadata=meta)
        assert received_meta == [meta]


class TestLoRAAcceleratorNoDeadDataLoader:
    """S575: accelerator.prepare() should not create unused data_loader."""

    def test_accelerator_prepare_no_data_loader(self):
        """accelerator.prepare() should only wrap model and optimizer."""
        from enigma_engine.core.lora_utils import LoraTrainer
        source = inspect.getsource(LoraTrainer.train)
        # The dead _create_dataloader call in prepare should be gone
        assert "_create_dataloader" not in source.split("accelerator.prepare")[0].split("accelerator.prepare")[-1] if "accelerator.prepare" in source else True
        # data_loader should not appear in accelerator.prepare unpacking
        if "accelerator.prepare" in source:
            # Find the line with accelerator.prepare result assignment
            for line in source.splitlines():
                if "= accelerator.prepare(" in line:
                    assert "data_loader" not in line, \
                        "data_loader should not be in accelerator.prepare() unpacking"


# ================================================================
# Pass 41 â€” S576, S577, S578
# ================================================================

class TestCuratedDatasetLoadSafety:
    """S576: load() must not lose in-memory data on parse failure."""

    def test_load_preserves_entries_on_corrupt_file(self, tmp_path):
        """If the JSONL file is corrupt, existing in-memory entries survive."""
        import json
        from enigma_engine.core.curated_dataset import CuratedDataset, DatasetEntry

        ds_path = tmp_path / "ds.jsonl"
        # Write one valid entry then a corrupt line
        valid = {"text": "hello world", "source": "test", "status": "approved"}
        ds_path.write_text(
            json.dumps(valid) + "\n" + "NOT VALID JSON\n",
            encoding="utf-8"
        )

        ds = CuratedDataset(ds_path)
        # Manually set some entries as if they were already loaded
        ds._entries = [DatasetEntry(text="existing entry", source="x")]
        # load() should fail on corrupt line but preserve old entries
        ds.load()
        assert len(ds._entries) == 1
        assert ds._entries[0].text == "existing entry"

    def test_load_replaces_entries_on_valid_file(self, tmp_path):
        """A fully valid file replaces old in-memory entries."""
        import json
        from enigma_engine.core.curated_dataset import CuratedDataset, DatasetEntry

        ds_path = tmp_path / "ds.jsonl"
        entries = [
            {"text": "first entry", "source": "t", "status": "approved"},
            {"text": "second entry", "source": "t", "status": "pending"},
        ]
        ds_path.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n",
            encoding="utf-8"
        )

        ds = CuratedDataset(ds_path)
        ds._entries = [DatasetEntry(text="old entry", source="old")]
        ds.load()
        assert len(ds._entries) == 2
        assert ds._entries[0].text == "first entry"


class TestYaRNDivByZero:
    """S577: YaRN RoPE must not divide by zero when dim == beta_fast (32)."""

    def test_yarn_dim_32_no_crash(self):
        """precompute_rope_frequencies with dim=32 + yarn should not crash."""
        from enigma_engine.core.model_components import precompute_rope_frequencies
        # dim=32 causes beta_fast/dim - beta_slow = 32/32 - 1 = 0
        result = precompute_rope_frequencies(
            dim=32, max_seq_len=64, theta=10000.0,
            scaling_type="yarn", scaling_factor=2.0
        )
        assert result.shape == (64, 16)  # max_seq_len x dim/2
        # No NaN or Inf
        import torch
        assert not torch.isnan(result.abs()).any()
        assert not torch.isinf(result.abs()).any()

    def test_yarn_dim_64_still_works(self):
        """Normal dim=64 with yarn should work as before."""
        from enigma_engine.core.model_components import precompute_rope_frequencies
        result = precompute_rope_frequencies(
            dim=64, max_seq_len=128, theta=10000.0,
            scaling_type="yarn", scaling_factor=2.0
        )
        assert result.shape == (128, 32)
        import torch
        assert not torch.isnan(result.abs()).any()


# ================================================================
# Pass 43 â€” S584, S590, S591, S592, S599, S607, S610, S611, S612, S616
# ================================================================

class TestGetStateDictNested:
    """S584: get_state_dict handles nested checkpoint wrappers."""

    def test_nested_single_key_checkpoint(self):
        """Single-key wrapper is unwrapped to find state dict."""
        import torch
        from enigma_engine.core.model_registry import get_state_dict
        inner = {"layer.0.weight": torch.ones(4)}
        checkpoint = {"checkpoint": {"model_state_dict": inner}}
        result = get_state_dict(checkpoint)
        assert "layer.0.weight" in result
        torch.testing.assert_close(result["layer.0.weight"], torch.ones(4))

    def test_standard_keys_unchanged(self):
        """Standard top-level keys still work and preserve tensor values."""
        import torch
        from enigma_engine.core.model_registry import get_state_dict
        sd = {"layer.0.weight": torch.tensor([1.0, 2.0, 3.0])}
        for key in ("model_state_dict", "state_dict", "model"):
            result = get_state_dict({key: sd})
            assert set(result.keys()) == {"layer.0.weight"}
            torch.testing.assert_close(
                result["layer.0.weight"], torch.tensor([1.0, 2.0, 3.0]))

    def test_raw_state_dict_still_works(self):
        """When no known keys foundâ€”not single-keyâ€”returns as-is."""
        import torch
        from enigma_engine.core.model_registry import get_state_dict
        sd = {"w1": torch.ones(4), "w2": torch.zeros(4)}
        result = get_state_dict(sd)
        assert set(result.keys()) == {"w1", "w2"}
        torch.testing.assert_close(result["w1"], torch.ones(4))
        torch.testing.assert_close(result["w2"], torch.zeros(4))


class TestAddBatchCrossDedup:
    """S592: add_batch deduplicates against existing entries."""

    def test_no_duplicate_across_batches(self, tmp_path):
        """Items already in the dataset are not added again."""
        from enigma_engine.core.curated_dataset import CuratedDataset
        ds = CuratedDataset(tmp_path / "test.jsonl")
        ds.add("existing", source="a")
        count = ds.add_batch(["existing", "new"], source="b")
        assert count == 1  # only "new" added
        assert ds.count == 2


class TestCharTokenizerUnicode:
    """S599: char_tokenizer word-boundary uses Unicode-aware \\w."""

    def test_encode_source_uses_unicode_boundary(self):
        """Regex lookbehinds use \\w instead of [A-Za-z]."""
        src = inspect.getsource(
            __import__('enigma_engine.core.char_tokenizer',
                       fromlist=['CharacterTokenizer']).CharacterTokenizer.encode
        )
        # Should use \w for Unicode awareness, not [A-Za-z]
        assert r"(?<!\w)" in src, (
            "encode() should use Unicode-aware \\w in lookbehinds")
        assert r"(?<![A-Za-z])" not in src, (
            "encode() should not use ASCII-only [A-Za-z] in lookbehinds")


class TestEvaluateModelEmptyPrompts:
    """S619: evaluate_model returns inf for empty prompt list."""

    def test_empty_prompts_returns_inf(self):
        """Empty prompt list should return infinite perplexity, not 0."""
        from enigma_engine.training.training_evaluation import evaluate_model
        result = evaluate_model(None, None, [])
        assert result["perplexity"] == float("inf")
        assert result["loss"] == float("inf")
        assert result["num_prompts"] == 0

# ================================================================
# ModelRegistry CRUD & Recovery Diagnostics
# ================================================================

class TestModelRegistryCRUD:
    """Diagnose which ModelRegistry operation failed.

    Each test isolates one failure mode so the name tells you
    exactly what broke: in-memory dict, disk persistence,
    removal, or corruption recovery.
    """

    def test_register_then_get_returns_info(self, tmp_path):
        """register_model â†’ get_model must return the stored info.

        DIAGNOSES: In-memory dict write/read broken.
        """
        from enigma_engine.core.model_registry import ModelRegistry
        reg = ModelRegistry(models_dir=tmp_path)
        reg.register_model("alpha", {"size": 125_000_000})
        result = reg.get_model("alpha")
        assert result is not None, "get_model returned None after register"
        assert result["size"] == 125_000_000, "Stored info corrupted"

    def test_model_exists_after_register(self, tmp_path):
        """model_exists must return True for registered models.

        DIAGNOSES: model_exists() not reading from same dict as register.
        """
        from enigma_engine.core.model_registry import ModelRegistry
        reg = ModelRegistry(models_dir=tmp_path)
        assert not reg.model_exists("beta"), "Ghost model before register"
        reg.register_model("beta", {"size": 50_000_000})
        assert reg.model_exists("beta"), "model_exists False after register"

    def test_remove_then_get_returns_none(self, tmp_path):
        """remove_model must make get_model return None.

        DIAGNOSES: remove_model not deleting from dict, or save not flushed.
        """
        from enigma_engine.core.model_registry import ModelRegistry
        reg = ModelRegistry(models_dir=tmp_path)
        reg.register_model("gamma", {"size": 10_000_000})
        reg.remove_model("gamma")
        assert reg.get_model("gamma") is None, "Model still visible after remove"
        assert not reg.model_exists("gamma"), "model_exists True after remove"

    def test_remove_nonexistent_does_not_crash(self, tmp_path):
        """remove_model on unknown name must not raise.

        DIAGNOSES: Missing guard on dict key lookup.
        """
        from enigma_engine.core.model_registry import ModelRegistry
        reg = ModelRegistry(models_dir=tmp_path)
        reg.remove_model("does_not_exist")  # Should not raise

    def test_persist_to_disk_and_reload(self, tmp_path):
        """Data must survive save â†’ new instance load from same dir.

        DIAGNOSES: _save_registry or _load_registry broken â€” narrows
        to disk I/O vs in-memory.
        """
        from enigma_engine.core.model_registry import ModelRegistry
        reg1 = ModelRegistry(models_dir=tmp_path)
        reg1.register_model("delta", {"size": 387_000_000, "type": "enigma"})

        # New instance reads from same directory
        reg2 = ModelRegistry(models_dir=tmp_path)
        result = reg2.get_model("delta")
        assert result is not None, "Model lost after reload from disk"
        assert result["size"] == 387_000_000, "Model info corrupted on disk"
        assert result["type"] == "enigma", "Model metadata lost on reload"

    def test_corrupted_json_recovers_gracefully(self, tmp_path):
        """Invalid JSON in registry.json must not crash init.

        DIAGNOSES: _load_registry exception handler broken â€”
        corrupt file kills the entire registry.
        """
        from enigma_engine.core.model_registry import ModelRegistry
        registry_file = tmp_path / "registry.json"
        registry_file.write_text("{invalid json!!!", encoding="utf-8")

        reg = ModelRegistry(models_dir=tmp_path)  # Must not raise
        # Should fall back to empty registry
        assert isinstance(reg.list_models(), dict), "list_models not a dict"

    def test_missing_models_key_recovers(self, tmp_path):
        """registry.json with no 'models' key must not crash.

        DIAGNOSES: Guard on line 48 of model_registry.py broken â€”
        missing key cascades to KeyError in every operation.
        """
        import json
        from enigma_engine.core.model_registry import ModelRegistry
        registry_file = tmp_path / "registry.json"
        registry_file.write_text(
            json.dumps({"version": 1}), encoding="utf-8")

        reg = ModelRegistry(models_dir=tmp_path)  # Must not raise
        models = reg.list_models()
        assert isinstance(models, dict), "list_models crashed on missing key"

    def test_get_model_returns_copy(self, tmp_path):
        """get_model must return a copy â€” mutations must not leak back.

        DIAGNOSES: Returning internal dict reference â†’ external code
        can silently corrupt registry state.
        """
        from enigma_engine.core.model_registry import ModelRegistry
        reg = ModelRegistry(models_dir=tmp_path)
        reg.register_model("epsilon", {"size": 100})
        info = reg.get_model("epsilon")
        info["size"] = 999  # Mutate the returned copy
        assert reg.get_model("epsilon")["size"] == 100, \
            "get_model returned internal reference, not a copy"


# ================================================================
# CF-11: Weight mapping should warn on high skip ratio
# ================================================================

class TestWeightMappingSkipWarning:
    """WeightMapper must warn/error when too many weights are unmapped."""

    def test_high_skip_ratio_warns(self):
        """Mapping where >10% of weights are skipped should log a warning."""
        from enigma_engine.core.weight_mapping import WeightMapper
        mapper = WeightMapper()
        # 2 valid + 8 unmapped = 80% skip rate
        dummy = {
            "model.embed_tokens.weight": "emb",
            "model.norm.weight": "norm",
        }
        for i in range(8):
            dummy[f"totally.unknown.layer.{i}.weight"] = f"t{i}"

        with pytest.raises(ValueError, match="unmapped"):
            mapper.map_huggingface_to_forge(dummy, model_type="llama")

    def test_low_skip_ratio_ok(self):
        """Mapping where <=10% of weights are skipped should not raise."""
        from enigma_engine.core.weight_mapping import WeightMapper
        mapper = WeightMapper()
        # 9 valid + 1 unmapped = 10% skip rate
        dummy = {
            "model.embed_tokens.weight": "emb",
            "model.norm.weight": "norm",
            "lm_head.weight": "out",
            "model.layers.0.self_attn.q_proj.weight": "q",
            "model.layers.0.self_attn.k_proj.weight": "k",
            "model.layers.0.self_attn.v_proj.weight": "v",
            "model.layers.0.self_attn.o_proj.weight": "o",
            "model.layers.0.input_layernorm.weight": "ln1",
            "model.layers.0.post_attention_layernorm.weight": "ln2",
            "totally.unknown.weight": "x",
        }
        result = mapper.map_huggingface_to_forge(dummy, model_type="llama")
        assert len(result) >= 9


# ---------------------------------------------------------------------------
# Model merging (N-17)
# ---------------------------------------------------------------------------

class TestModelMerging:
    """Tests for SLERP, TIES, and linear model merging."""

    @staticmethod
    def _make_checkpoint(tmp_path, name, seed, dim=32, n_layers=2):
        """Create a minimal fake checkpoint for merge tests."""
        import torch
        from enigma_engine.core.model_presets import ForgeConfig
        from enigma_engine.core.safe_save import atomic_torch_save

        cfg = ForgeConfig(
            vocab_size=64, dim=dim, n_layers=n_layers,
            n_heads=4, n_kv_heads=2, hidden_dim=dim * 4,
            max_seq_len=64,
        )

        torch.manual_seed(seed)
        sd = {
            "tok_embeddings.weight": torch.randn(64, dim),
            "norm.weight": torch.ones(dim),
            "output.weight": torch.randn(64, dim),
        }
        for i in range(n_layers):
            sd[f"layers.{i}.attention.wq.weight"] = torch.randn(dim, dim)
            sd[f"layers.{i}.attention.wk.weight"] = torch.randn(
                dim // 2, dim)
            sd[f"layers.{i}.attention.wv.weight"] = torch.randn(
                dim // 2, dim)
            sd[f"layers.{i}.attention.wo.weight"] = torch.randn(dim, dim)
            sd[f"layers.{i}.attention_norm.weight"] = torch.ones(dim)
            sd[f"layers.{i}.ffn_norm.weight"] = torch.ones(dim)
            sd[f"layers.{i}.feed_forward.w1.weight"] = torch.randn(
                dim * 4, dim)
            sd[f"layers.{i}.feed_forward.w2.weight"] = torch.randn(
                dim, dim * 4)
            sd[f"layers.{i}.feed_forward.w3.weight"] = torch.randn(
                dim * 4, dim)

        path = tmp_path / f"{name}.pth"
        atomic_torch_save({
            "model_state_dict": sd,
            "config": cfg.to_dict(),
        }, str(path))
        return path, sd

    def test_slerp_t0_returns_model_a(self, tmp_path):
        """SLERP with t=0 should return model A."""
        import torch
        from enigma_engine.core.model_merging import slerp_merge

        path_a, sd_a = self._make_checkpoint(tmp_path, "a", seed=42)
        path_b, _ = self._make_checkpoint(tmp_path, "b", seed=99)

        result = slerp_merge(path_a, path_b, t=0.0)
        merged_sd = result["model_state_dict"]

        for key in sd_a:
            assert torch.allclose(
                merged_sd[key].float(), sd_a[key].float(), atol=1e-5), key

    def test_slerp_t1_returns_model_b(self, tmp_path):
        """SLERP with t=1 should return model B."""
        import torch
        from enigma_engine.core.model_merging import slerp_merge

        path_a, _ = self._make_checkpoint(tmp_path, "a", seed=42)
        path_b, sd_b = self._make_checkpoint(tmp_path, "b", seed=99)

        result = slerp_merge(path_a, path_b, t=1.0)
        merged_sd = result["model_state_dict"]

        for key in sd_b:
            assert torch.allclose(
                merged_sd[key].float(), sd_b[key].float(), atol=1e-5), key

    def test_slerp_t05_differs_from_both(self, tmp_path):
        """SLERP with t=0.5 should produce something different from both."""
        import torch
        from enigma_engine.core.model_merging import slerp_merge

        path_a, sd_a = self._make_checkpoint(tmp_path, "a", seed=42)
        path_b, sd_b = self._make_checkpoint(tmp_path, "b", seed=99)

        result = slerp_merge(path_a, path_b, t=0.5)
        merged_sd = result["model_state_dict"]

        key = "layers.0.attention.wq.weight"
        assert not torch.allclose(
            merged_sd[key].float(), sd_a[key].float(), atol=1e-3)
        assert not torch.allclose(
            merged_sd[key].float(), sd_b[key].float(), atol=1e-3)

    def test_slerp_preserves_config(self, tmp_path):
        """Merged result should carry valid config from model A."""
        from enigma_engine.core.model_merging import slerp_merge

        path_a, _ = self._make_checkpoint(tmp_path, "a", seed=42)
        path_b, _ = self._make_checkpoint(tmp_path, "b", seed=99)

        result = slerp_merge(path_a, path_b, t=0.5)
        assert result["config"]["dim"] == 32
        assert result["config"]["n_layers"] == 2

    def test_slerp_saves_output(self, tmp_path):
        """SLERP should save checkpoint when output_path given."""
        from enigma_engine.core.model_merging import slerp_merge

        path_a, _ = self._make_checkpoint(tmp_path, "a", seed=42)
        path_b, _ = self._make_checkpoint(tmp_path, "b", seed=99)
        out_path = tmp_path / "merged.pth"

        slerp_merge(path_a, path_b, t=0.5, output_path=out_path)
        assert out_path.exists()

    def test_slerp_bad_t_raises(self, tmp_path):
        """t outside [0, 1] should raise ValueError."""
        import pytest
        from enigma_engine.core.model_merging import slerp_merge

        path_a, _ = self._make_checkpoint(tmp_path, "a", seed=42)
        path_b, _ = self._make_checkpoint(tmp_path, "b", seed=99)

        with pytest.raises(ValueError, match="t must be"):
            slerp_merge(path_a, path_b, t=1.5)

    def test_slerp_mismatched_arch_raises(self, tmp_path):
        """Models with different dimensions should raise ValueError."""
        import pytest
        from enigma_engine.core.model_merging import slerp_merge

        path_a, _ = self._make_checkpoint(
            tmp_path, "a", seed=42, dim=32)
        path_b, _ = self._make_checkpoint(
            tmp_path, "b", seed=99, dim=64)

        with pytest.raises(ValueError, match="Architecture mismatch"):
            slerp_merge(path_a, path_b, t=0.5)

    def test_slerp_progress_callback(self, tmp_path):
        """SLERP should call on_progress with increasing percentages."""
        from enigma_engine.core.model_merging import slerp_merge

        path_a, _ = self._make_checkpoint(tmp_path, "a", seed=42)
        path_b, _ = self._make_checkpoint(tmp_path, "b", seed=99)
        pcts = []
        slerp_merge(
            path_a, path_b, t=0.5,
            on_progress=lambda p, m: pcts.append(p))
        assert pcts[-1] == 100
        assert pcts == sorted(pcts)

    def test_linear_t05_is_average(self, tmp_path):
        """Linear merge at t=0.5 should give exact average."""
        import torch
        from enigma_engine.core.model_merging import linear_merge

        path_a, sd_a = self._make_checkpoint(tmp_path, "a", seed=42)
        path_b, sd_b = self._make_checkpoint(tmp_path, "b", seed=99)

        result = linear_merge(path_a, path_b, t=0.5)
        merged_sd = result["model_state_dict"]

        key = "layers.0.attention.wq.weight"
        expected = (sd_a[key].float() + sd_b[key].float()) / 2
        assert torch.allclose(
            merged_sd[key].float(), expected, atol=1e-5)

    def test_ties_produces_valid_output(self, tmp_path):
        """TIES merge should produce a valid state dict."""
        from enigma_engine.core.model_merging import ties_merge

        path_a, sd_a = self._make_checkpoint(tmp_path, "a", seed=42)
        path_b, _ = self._make_checkpoint(tmp_path, "b", seed=99)
        path_c, _ = self._make_checkpoint(tmp_path, "c", seed=7)

        result = ties_merge(
            [path_a, path_b, path_c],
            base_path=path_a,
            density=0.5,
        )
        merged_sd = result["model_state_dict"]

        # Should have same keys as base
        assert set(merged_sd.keys()) == set(sd_a.keys())
        # Should have same shapes
        for key in sd_a:
            assert merged_sd[key].shape == sd_a[key].shape

    def test_ties_too_few_models_raises(self, tmp_path):
        """TIES with fewer than 2 models should raise."""
        import pytest
        from enigma_engine.core.model_merging import ties_merge

        path_a, _ = self._make_checkpoint(tmp_path, "a", seed=42)

        with pytest.raises(ValueError, match="at least 2"):
            ties_merge([path_a])

    def test_ties_density_zero_raises(self, tmp_path):
        """TIES with density=0 should raise."""
        import pytest
        from enigma_engine.core.model_merging import ties_merge

        path_a, _ = self._make_checkpoint(tmp_path, "a", seed=42)
        path_b, _ = self._make_checkpoint(tmp_path, "b", seed=99)

        with pytest.raises(ValueError, match="density"):
            ties_merge([path_a, path_b], density=0.0)

